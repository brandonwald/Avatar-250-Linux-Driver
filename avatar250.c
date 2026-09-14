// SPDX-License-Identifier: GPL-2.0
/*
 * avatar250.c - standalone block driver for the Avatar Shark 250
 * parallel-port IDE drive (Shuttle EPAT adapter chip, mode 2 / 8-bit).
 *
 * This talks to the EPAT chip's registers directly and issues raw ATA
 * PIO commands (IDENTIFY, READ SECTORS, WRITE SECTORS). It does NOT use
 * libata, so it never sends SET FEATURES / Transfer Mode, which is the
 * command this specific drive aborts.
 *
 * The register-level connect/read/write sequences are transcribed from
 * the in-kernel drivers/ata/pata_parport/epat.c protocol driver
 * (Shuttle EPAT, mode 2 case), proven to connect to this exact drive
 * and read its IDENTIFY data correctly under pata_parport. This driver
 * reuses that register choreography but skips libata's post-IDENTIFY
 * mode-negotiation step entirely.
 *
 * Port discovery uses the parport_driver .match_port/.detach callback
 * model (verified against the current in-tree pata_parport.c and
 * drivers/auxdisplay/ks0108.c), not a one-shot parport_find_base() call
 * in module_init. That matters: .match_port gets invoked by the parport
 * core for every port present at registration time AND for any that
 * show up afterward, so this doesn't depend on load-order timing.
 *
 * KNOWN RISK AREAS (read before debugging blind):
 *   1. Timing: the original driver has delay/udelay tuning we're
 *      approximating with a single module parameter. If connect()
 *      fails or reads look corrupted, try increasing io_delay first.
 *   2. epat_read_block()/epat_write_block() are now a direct, verified
 *      transcription of the real drivers/ata/pata_parport/epat.c
 *      source (fetched and checked, not reconstructed from memory or
 *      fragments) - this replaced three earlier failed attempts at a
 *      per-word data access function that were all built on
 *      reasonable-sounding but wrong guesses. If data still looks
 *      wrong after this version, the likely remaining suspect is
 *      SECTOR_WORDS*2 byte-vs-word count math, not the transfer
 *      mechanism itself.
 *   3. blk_alloc_disk() takes a struct queue_limits* plus a node id on
 *      this kernel (6.12) - confirmed against actual compile errors on
 *      6.12.94+deb13-amd64, not guessed. If you're building on a
 *      meaningfully different kernel version and this fails again,
 *      check /usr/src/linux-headers-$(uname -r)/include/linux/blkdev.h
 *      for the exact prototype and adjust.
 *   4. struct pardev_cb flags: we pass flags=0 (no exclusivity flag)
 *      because the exact constant name (PARPORT_DEV_EXCL vs
 *      PARPORT_FLAG_EXCL) has differed across kernel eras and I can't
 *      confirm which your headers use without seeing them. Harmless to
 *      leave as 0 for a single-consumer driver like this; tighten it
 *      yourself if you want strict exclusivity.
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/blkdev.h>
#include <linux/err.h>
#include <linux/mutex.h>
#include <linux/parport.h>
#include <linux/delay.h>
#include <linux/errno.h>
#include <linux/bio.h>
#include <linux/highmem.h>
#include <asm/io.h>

#define DRV_NAME "avatar250"
#define AVATAR250_SECTOR_SIZE 512

/* ---- module parameters ------------------------------------------- */

static int io = 0x3048;         /* parport0 base on this system - override
                                    with insmod avatar250.ko io=0x.... if
                                    yours differs (check dmesg for
                                    "parportN: PC-style at 0x....") */
module_param(io, int, 0444);
MODULE_PARM_DESC(io, "Parallel port base I/O address to match against");

static int io_delay = 2;        /* microseconds between register strobes */
module_param(io_delay, int, 0644);
MODULE_PARM_DESC(io_delay, "Delay (us) between port register accesses");

static int epat_mode = 2;       /* 2 = 8-bit byte mode, proven working
                                    on this hardware; 0 = 4-bit nibble
                                    mode as a slower fallback */
module_param(epat_mode, int, 0444);
MODULE_PARM_DESC(epat_mode, "EPAT mode: 0=nibble, 2=byte (default, matches this hardware)");

/* ---- ATA constants -------------------------------------------------*/

#define ATA_REG_DATA    0
#define ATA_REG_NSECT   2
#define ATA_REG_LBAL    3
#define ATA_REG_LBAM    4
#define ATA_REG_LBAH    5
#define ATA_REG_DEVICE  6
#define ATA_REG_STATUS  7
#define ATA_REG_COMMAND 7

#define ATA_CMD_IDENTIFY      0xEC
#define ATA_CMD_READ_SECTORS  0x20
#define ATA_CMD_WRITE_SECTORS 0x30

#define ATA_STAT_ERR   0x01
#define ATA_STAT_DRQ   0x08
#define ATA_STAT_DRDY  0x40
#define ATA_STAT_BSY   0x80

#define SECTOR_WORDS   256   /* 512 bytes = 256 16-bit words */

/* EPAT register-file offsets, from the verified kernel source:
 *   cont=0 -> IDE task-file registers
 *   cont=1 -> IDE control registers (device control / alt status)
 *   cont=2 -> EPAT chip's own internal config registers
 */
static const int epat_cont_map[3] = { 0x18, 0x10, 0 };

struct avatar250_dev {
	struct parport *port;
	struct pardevice *pdev;
	struct gendisk *disk;
	sector_t capacity;   /* in 512-byte sectors */
	struct mutex io_lock; /* serializes all hardware access - submit_bio
				* has no other serialization, and the kernel's
				* async writeback path can and does submit
				* multiple bios concurrently. A mutex (not a
				* spinlock) because avatar250_wait_status() can
				* sleep via msleep()/usleep_range(). */
};

/* Single-instance driver: one drive, one global device. attach()/detach()
 * are serialized by the parport core's own registration_lock, so no
 * extra locking is needed around this pointer. */
static struct avatar250_dev *g_dev;
static int avatar250_major;

/* ---- raw port I/O (base+0=data, base+1=status, base+2=control) ---- */

static inline void w0(struct avatar250_dev *dev, u8 v) { outb(v, dev->port->base);       udelay(io_delay); }
static inline u8   r0(struct avatar250_dev *dev)        { u8 v = inb(dev->port->base);   udelay(io_delay); return v; }
static inline u8   r1(struct avatar250_dev *dev)        { u8 v = inb(dev->port->base + 1); udelay(io_delay); return v; }
static inline void w2(struct avatar250_dev *dev, u8 v) { outb(v, dev->port->base + 2);   udelay(io_delay); }

/* ---- EPAT register read/write ---------------------------------------
 * The write path is identical across modes 0-2 in the original driver
 * (only the EPP modes 3-5 differ) - no mode branch needed here, unlike
 * an earlier draft of this file which had a redundant if/else doing the
 * same thing in both arms. Only the read path actually differs by mode.
 */
static void epat_write_regr(struct avatar250_dev *dev, int cont, int regr, int val)
{
	int r = regr + epat_cont_map[cont];

	w0(dev, 0x60 + r); w2(dev, 1); w0(dev, val); w2(dev, 4);
}

static int epat_read_regr(struct avatar250_dev *dev, int cont, int regr)
{
	int r = regr + epat_cont_map[cont];
	u8 a, b;

	if (epat_mode == 0) {
		/* nibble mode: two 4-bit reads combined */
		w0(dev, r); w2(dev, 1); w2(dev, 3);
		a = r1(dev); w2(dev, 4); b = r1(dev);
		return (((a >> 4) & 0x0f) + (b & 0xf0));
	}
	/* mode 2, 8-bit direct - what this hardware actually negotiates */
	w0(dev, 0x20 + r); w2(dev, 1); w2(dev, 0x25);
	a = r0(dev); w2(dev, 4);
	return a;
}

/* CPP handshake, transcribed from epat.c verbatim */
static void epat_cpp(struct avatar250_dev *dev, u8 x)
{
	w2(dev, 4); w0(dev, 0x22); w0(dev, 0xaa); w0(dev, 0x55); w0(dev, 0);
	w0(dev, 0xff); w0(dev, 0x87); w0(dev, 0x78); w0(dev, x);
	w2(dev, 4); w2(dev, 5); w2(dev, 4); w0(dev, 0xff);
}

static void epat_connect(struct avatar250_dev *dev)
{
	epat_cpp(dev, 0);
	epat_cpp(dev, 0xe0);
	w0(dev, 0); w2(dev, 1); w2(dev, 4);

	/* internal chip config registers (cont=2), values from epat.c */
	epat_write_regr(dev, 2, 0x8, 0x10);
	epat_write_regr(dev, 2, 0xc, 0x14);
	epat_write_regr(dev, 2, 0xa, 0x38);
	epat_write_regr(dev, 2, 0x12, 0x10);
}

static void epat_disconnect(struct avatar250_dev *dev)
{
	w0(dev, 0x20); w2(dev, 1); w2(dev, 4);
}

/* Bulk 512-byte sector transfer - a direct, verified transcription of
 * the real epat_read_block()/epat_write_block() (mode 0/1/2 case) from
 * drivers/ata/pata_parport/epat.c, fetched and checked against source
 * rather than reconstructed from fragments. This replaces three earlier
 * per-word attempts that were all wrong for the same reason: none of
 * them matched how the chip's bulk-transfer path actually works. The
 * real driver selects "data transfer mode" ONCE before the loop
 * (w0(0x27) for reads, w0(0x67) for writes), then pulls/pushes each
 * byte with a much lighter per-byte step - alternating the control
 * register between two fixed values - instead of re-running the full
 * register-address selection on every byte, which is what
 * epat_read_regr()/epat_write_regr() do and why reusing them per-byte
 * never worked: repeating full register selects isn't the same
 * operation as the chip's actual streaming mode.
 *
 * count is in BYTES (512 for one sector), matching the real function's
 * signature - not words. */
static void epat_read_block(struct avatar250_dev *dev, u8 *buf, int count)
{
	int k, ph;

	w0(dev, 0x27); w2(dev, 1); w2(dev, 0x25); w0(dev, 0);
	ph = 0;
	for (k = 0; k < count - 1; k++) {
		w2(dev, 0x24 + ph);
		buf[k] = r0(dev);
		ph = 1 - ph;
	}
	w2(dev, 0x26); w2(dev, 0x27);
	buf[count - 1] = r0(dev);
	w2(dev, 0x25); w2(dev, 4);
}

static void epat_write_block(struct avatar250_dev *dev, const u8 *buf, int count)
{
	int k, ph;

	w0(dev, 0x67); w2(dev, 1); w2(dev, 5);
	ph = 0;
	for (k = 0; k < count; k++) {
		w0(dev, buf[k]);
		w2(dev, 4 + ph);
		ph = 1 - ph;
	}
	w2(dev, 7); w2(dev, 4);
}

/* ---- ATA command layer -------------------------------------------- */

static int avatar250_wait_status(struct avatar250_dev *dev, u8 mask_set,
				  u8 mask_clear, unsigned long timeout_ms)
{
	unsigned long deadline = jiffies + msecs_to_jiffies(timeout_ms);
	int spins = 0;
	u8 status;

	do {
		status = epat_read_regr(dev, 0, ATA_REG_STATUS);
		if (status & ATA_STAT_ERR)
			return -EIO;
		if ((status & mask_clear) == 0 && (status & mask_set) == mask_set)
			return status;

		/* Back off after the first chunk of fast spins instead of
		 * hammering a slow parallel port for the full timeout -
		 * writes especially can take a real while on old media. */
		if (++spins > 100)
			usleep_range(500, 1000);
		else
			cond_resched();
	} while (time_before(jiffies, deadline));

	return -ETIMEDOUT;
}

static int avatar250_softreset(struct avatar250_dev *dev)
{
	int status;

	/* ATA software reset: pulse SRST in the device control register
	 * (cont=1, regr=6 in EPAT's register map). Under libata/pata_parport
	 * this happens automatically as a standard part of device probing,
	 * separate from epat's own chip-level connect() - it's easy to miss
	 * when transcribing just the protocol driver. Without it the ATA
	 * bus may never actually leave its post-power-on state, and status
	 * reads can look coincidentally "ready" without a real drive behind
	 * them. */
	epat_write_regr(dev, 1, 6, 0x04);   /* SRST=1 */
	udelay(10);
	epat_write_regr(dev, 1, 6, 0x00);   /* SRST=0 */
	msleep(20);                          /* spec minimum settle before polling BSY */

	status = avatar250_wait_status(dev, 0, ATA_STAT_BSY, 5000);
	if (status < 0)
		return status;

	return 0;
}

static int avatar250_identify(struct avatar250_dev *dev, u16 *buf)
{
	int status;

	epat_write_regr(dev, 0, ATA_REG_DEVICE, 0xE0);
	/* Per the ATA spec, IDENTIFY DEVICE does not require DRDY first
	 * (unlike READ/WRITE SECTORS below) - only checking BSY here is
	 * intentional, not an oversight. */
	status = avatar250_wait_status(dev, 0, ATA_STAT_BSY, 2000);
	if (status < 0)
		return status;

	epat_write_regr(dev, 0, ATA_REG_COMMAND, ATA_CMD_IDENTIFY);
	status = avatar250_wait_status(dev, ATA_STAT_DRQ, ATA_STAT_BSY, 2000);
	if (status < 0)
		return status;

	epat_read_block(dev, (u8 *)buf, SECTOR_WORDS * 2);

	return 0;
}

static int avatar250_rw_sector(struct avatar250_dev *dev, u32 lba,
				u16 *buf, int write)
{
	int status;

	epat_write_regr(dev, 0, ATA_REG_NSECT, 1);
	epat_write_regr(dev, 0, ATA_REG_LBAL, lba & 0xff);
	epat_write_regr(dev, 0, ATA_REG_LBAM, (lba >> 8) & 0xff);
	epat_write_regr(dev, 0, ATA_REG_LBAH, (lba >> 16) & 0xff);
	epat_write_regr(dev, 0, ATA_REG_DEVICE, 0xE0 | ((lba >> 24) & 0x0f));

	status = avatar250_wait_status(dev, ATA_STAT_DRDY, ATA_STAT_BSY, 2000);
	if (status < 0) {
		if (write)
			pr_err(DRV_NAME ": write lba=%u: pre-command DRDY wait failed (%d), status=0x%02x\n",
			       lba, status, epat_read_regr(dev, 0, ATA_REG_STATUS));
		return status;
	}

	epat_write_regr(dev, 0, ATA_REG_COMMAND,
			 write ? ATA_CMD_WRITE_SECTORS : ATA_CMD_READ_SECTORS);

	status = avatar250_wait_status(dev, ATA_STAT_DRQ, ATA_STAT_BSY, 5000);
	if (status < 0) {
		if (write)
			pr_err(DRV_NAME ": write lba=%u: post-command DRQ wait failed (%d), status=0x%02x\n",
			       lba, status, epat_read_regr(dev, 0, ATA_REG_STATUS));
		return status;
	}

	if (write) {
		epat_write_block(dev, (const u8 *)buf, SECTOR_WORDS * 2);
		/* wait for the drive to finish committing the sector */
		status = avatar250_wait_status(dev, 0, ATA_STAT_BSY, 5000);
		if (status < 0) {
			pr_err(DRV_NAME ": write lba=%u: post-data BSY wait failed (%d), status=0x%02x\n",
			       lba, status, epat_read_regr(dev, 0, ATA_REG_STATUS));
			return status;
		}
	} else {
		epat_read_block(dev, (u8 *)buf, SECTOR_WORDS * 2);
	}

	return 0;
}

/* ---- block layer ---------------------------------------------------*/

static void avatar250_submit_bio(struct bio *bio)
{
	struct avatar250_dev *dev = bio->bi_bdev->bd_disk->private_data;
	struct bio_vec bvec;
	struct bvec_iter iter;
	sector_t sector = bio->bi_iter.bi_sector;
	int write = bio_data_dir(bio) == WRITE;
	int ret = 0;

	/* Only serialization in this driver - without it, concurrent bios
	 * (routine under the kernel's async writeback path, which is what
	 * "lost async page write" in dmesg was actually telling us) could
	 * enter this function from multiple contexts at once and interleave
	 * raw register accesses to the same physical parallel port,
	 * corrupting the ATA command sequence mid-transfer. That produces
	 * exactly the failure pattern seen during bring-up: deterministic-
	 * looking but position-independent write failures, immune to cable
	 * or media quality, because the actual cause was never hardware. */
	mutex_lock(&dev->io_lock);

	bio_for_each_segment(bvec, bio, iter) {
		void *p = bvec_kmap_local(&bvec);
		unsigned int len = bvec.bv_len;
		unsigned int off = 0;

		while (len >= AVATAR250_SECTOR_SIZE) {
			ret = avatar250_rw_sector(dev, (u32)sector,
						   (u16 *)(p + off), write);
			if (ret < 0)
				break;
			sector++;
			off += AVATAR250_SECTOR_SIZE;
			len -= AVATAR250_SECTOR_SIZE;
		}

		/* Should never happen with logical_block_size=512 set below,
		 * but fail loudly instead of silently dropping bytes if a
		 * segment ever isn't sector-aligned. */
		if (ret == 0 && len != 0) {
			pr_err(DRV_NAME ": segment not sector-aligned (%u bytes left over)\n", len);
			ret = -EIO;
		}

		kunmap_local(p);
		if (ret < 0)
			break;
	}

	mutex_unlock(&dev->io_lock);

	bio->bi_status = ret < 0 ? BLK_STS_IOERR : BLK_STS_OK;
	bio_endio(bio);
}

static const struct block_device_operations avatar250_fops = {
	.owner = THIS_MODULE,
	.submit_bio = avatar250_submit_bio,
};

/* ---- parport attach/detach (device discovery) -----------------------
 *
 * This is the modern parport_driver callback model, matching the real
 * in-tree pata_parport.c and drivers/auxdisplay/ks0108.c. The parport
 * core invokes attach() once per known port at driver-registration
 * time, AND again for any port that appears afterward - so unlike a
 * one-shot parport_find_base() in module_init, this isn't sensitive to
 * load-order timing relative to when the port itself was enumerated.
 */

static void avatar250_attach(struct parport *port)
{
	struct pardev_cb cb;
	u16 *idbuf;
	int err;

	if (port->base != io)
		return;         /* not the port we're looking for */

	if (g_dev) {
		pr_warn(DRV_NAME ": already attached, ignoring duplicate match on 0x%lx\n",
			port->base);
		return;
	}

	g_dev = kzalloc(sizeof(*g_dev), GFP_KERNEL);
	if (!g_dev)
		return;
	g_dev->port = port;
	mutex_init(&g_dev->io_lock);

	memset(&cb, 0, sizeof(cb));
	cb.private = g_dev;
	/* flags left at 0 - see risk note #4 at the top of this file */

	g_dev->pdev = parport_register_dev_model(port, DRV_NAME, &cb, 0);
	if (!g_dev->pdev) {
		pr_err(DRV_NAME ": parport didn't register new device\n");
		goto err_free_dev;
	}

	if (parport_claim(g_dev->pdev)) {
		pr_err(DRV_NAME ": could not claim parport at 0x%lx\n", port->base);
		goto err_unreg_device;
	}

	epat_connect(g_dev);

	err = avatar250_softreset(g_dev);
	if (err < 0) {
		pr_err(DRV_NAME ": ATA soft reset failed (%d) - no drive responding on this port.\n", err);
		goto err_release;
	}

	idbuf = kzalloc(SECTOR_WORDS * 2, GFP_KERNEL);
	if (!idbuf)
		goto err_release;

	err = avatar250_identify(g_dev, idbuf);
	if (err < 0) {
		pr_err(DRV_NAME ": IDENTIFY failed (%d) - drive not responding.\n", err);
		pr_err(DRV_NAME ": check io= parameter, cabling, and drive power.\n");
		kfree(idbuf);
		goto err_release;
	}

	g_dev->capacity = idbuf[60] | ((u32)idbuf[61] << 16);
	pr_info(DRV_NAME ": IDENTIFY ok, capacity=%llu sectors (%llu MB)\n",
		(u64)g_dev->capacity, (u64)g_dev->capacity / 2048);

	/* Sanity check for readers debugging byte order: this should read
	 * as "AVATAR  AR2250SP" or similar. If it's noise, see risk note
	 * #2 at the top of this file. */
	{
		char model[41];
		int i;
		for (i = 0; i < 20; i++) {
			u16 w = idbuf[27 + i];
			model[i * 2]     = (w >> 8) & 0xff;
			model[i * 2 + 1] = w & 0xff;
		}
		model[40] = '\0';
		pr_info(DRV_NAME ": model string: \"%s\"\n", model);
	}
	kfree(idbuf);

	avatar250_major = register_blkdev(0, DRV_NAME);
	if (avatar250_major < 0)
		goto err_release;

	{
		/* Current kernels set block-size/sector limits via a
		 * queue_limits struct passed into blk_alloc_disk() itself,
		 * not via separate blk_queue_*() setter calls afterward -
		 * those setters were removed as part of the 2024 queue_limits
		 * rework. Keep individual requests small (max_hw_sectors):
		 * this bus is PIO over a parallel port, there is no upside
		 * to letting the block layer hand us large multi-megabyte
		 * bios. */
		struct queue_limits lim = {
			.logical_block_size = AVATAR250_SECTOR_SIZE,
			.physical_block_size = AVATAR250_SECTOR_SIZE,
			.max_hw_sectors = 8,
		};

		g_dev->disk = blk_alloc_disk(&lim, NUMA_NO_NODE);
		if (IS_ERR(g_dev->disk)) {
			pr_err(DRV_NAME ": blk_alloc_disk failed (%ld)\n",
			       PTR_ERR(g_dev->disk));
			g_dev->disk = NULL;
			goto err_unreg_blkdev;
		}
	}

	g_dev->disk->major = avatar250_major;
	g_dev->disk->first_minor = 0;
	g_dev->disk->minors = 1;
	g_dev->disk->fops = &avatar250_fops;
	g_dev->disk->private_data = g_dev;
	snprintf(g_dev->disk->disk_name, DISK_NAME_LEN, DRV_NAME);
	set_capacity(g_dev->disk, g_dev->capacity);

	err = add_disk(g_dev->disk);
	if (err)
		goto err_put_disk;

	pr_info(DRV_NAME ": /dev/%s ready\n", g_dev->disk->disk_name);
	return;

err_put_disk:
	put_disk(g_dev->disk);
	g_dev->disk = NULL;
err_unreg_blkdev:
	unregister_blkdev(avatar250_major, DRV_NAME);
err_release:
	epat_disconnect(g_dev);
	parport_release(g_dev->pdev);
err_unreg_device:
	parport_unregister_device(g_dev->pdev);
err_free_dev:
	kfree(g_dev);
	g_dev = NULL;
}

static void avatar250_detach(struct parport *port)
{
	if (port->base != io)
		return;
	if (!g_dev) {
		pr_warn(DRV_NAME ": detach on 0x%lx but nothing attached\n", port->base);
		return;
	}

	if (g_dev->disk) {
		del_gendisk(g_dev->disk);
		put_disk(g_dev->disk);
		unregister_blkdev(avatar250_major, DRV_NAME);
	}
	epat_disconnect(g_dev);
	parport_release(g_dev->pdev);
	parport_unregister_device(g_dev->pdev);
	kfree(g_dev);
	g_dev = NULL;
}

static struct parport_driver avatar250_parport_driver = {
	.name = DRV_NAME,
	.match_port = avatar250_attach,
	.detach = avatar250_detach,
};
module_parport_driver(avatar250_parport_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("written for a Debian trixie system, targeting the Avatar Shark 250 / EPAT bridge");
MODULE_DESCRIPTION("Standalone PIO block driver for parallel-port EPAT IDE drives, bypassing libata");