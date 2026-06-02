import csv
import array
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"")
INPUT_LOG  = DATA_DIR / "tbird2"
SEGMENTS_DIR = DATA_DIR / "segments"

# ── Parameters ────────────────────────────────────────────────────────────────
N_SEGMENTS = 10

# ── State map ─────────────────────────────────────────────────────────────────
STATE_MAP = {
    "-":0,"N_AUTH":1,"N_CALL_TR":2,"N_CPU":3,"N_LUS_LBUG":4,
    "N_MAIL":5,"N_NFS":6,"N_OOM":7,"N_PBS_BAIL":8,"N_PBS_BFD1":9,
    "N_PBS_BFD2":10,"N_PBS_CON2":11,"N_PBS_EPI":12,"N_PBS_SIS":13,
    "R_CHK_DSK":14,"R_ECC":15,"R_EXT_FS":16,"R_EXT_FS_ABRT1":17,
    "R_EXT_FS_ABRT2":18,"R_EXT_FS_IO":19,"R_EXT_INODE1":20,
    "R_EXT_INODE2":21,"R_GPF":22,"R_MPT":23,"R_MTT":24,"R_NMI":25,
    "R_PAG":26,"R_PAN":27,"R_RIP":28,"R_SCSI0":29,"R_SCSI1":30,
    "R_SEG":31,"R_SERR":32,"R_VAPI":33,
}

def main():
    # ── Create output directory ───────────────────────────────────────────────
    SEGMENTS_DIR.mkdir(exist_ok=True)

    # ── Pass 1: count valid lines ─────────────────────────────────────────────
    print("Pass 1 — counting valid lines...")
    total_lines = 0
    with INPUT_LOG.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            parts = line.split(None, 1)
            if parts and parts[0] in STATE_MAP:
                total_lines += 1
            if (i + 1) % 10_000_000 == 0:
                print(f"  {i+1:,} raw lines scanned, {total_lines:,} valid...")

    seg_size = total_lines // N_SEGMENTS
    print(f"\n  Valid lines   : {total_lines:,}")
    print(f"  Segment size  : ~{seg_size:,} lines each")

    # ── Pass 2: assign segment per valid line ─────────────────────────────────
    print("\nPass 2 — assigning segments...")
    assignments = array.array("B", [0] * total_lines)
    valid_idx = 0
    with INPUT_LOG.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(None, 1)
            if not parts or parts[0] not in STATE_MAP:
                continue
            seg = min(valid_idx // seg_size, N_SEGMENTS - 1)
            assignments[valid_idx] = seg
            valid_idx += 1
            if valid_idx % 10_000_000 == 0:
                print(f"  {valid_idx:,} lines assigned...")

    # ── Pass 3: write segment files ───────────────────────────────────────────
    print("\nPass 3 — writing segment files...")

    seg_files   = [
        (SEGMENTS_DIR / f"segment_{i+1:02d}.log").open(
            "w", encoding="utf-8"
        ) for i in range(N_SEGMENTS)
    ]

    stats = [{"lines": 0, "anomalous": 0} for _ in range(N_SEGMENTS)]

    valid_idx = 0
    with INPUT_LOG.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(None, 1)
            if not parts or parts[0] not in STATE_MAP:
                continue
            seg = assignments[valid_idx]
            seg_files[seg].write(line)
            stats[seg]["lines"] += 1
            if parts[0] != "-":
                stats[seg]["anomalous"] += 1
            valid_idx += 1
            if valid_idx % 10_000_000 == 0:
                print(f"  {valid_idx:,} lines written...")

    for f in seg_files:
        f.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Segment summary ──────────────────────────────────────────────")
    print(f"  {'Segment':<12} {'Lines':>12} {'Anomalous':>12} {'Anomaly %':>10}")
    print(f"  {'-'*50}")
    for i, s in enumerate(stats):
        pct = s["anomalous"] / s["lines"] * 100 if s["lines"] > 0 else 0
        path = SEGMENTS_DIR / f"segment_{i+1:02d}.log"
        size_mb = path.stat().st_size / 1_048_576
        print(f"  segment_{i+1:02d}    {s['lines']:>12,} {s['anomalous']:>12,} "
              f"{pct:>9.4f}%  ({size_mb:,.0f} MB)")
    print(f"─────────────────────────────────────────────────────────────────")
    print(f"\nDone. Segments written to: {SEGMENTS_DIR}")

if __name__ == "__main__":
    main()
