import csv
import array
from pathlib import Path
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"")
INPUT_CSV   = DATA_DIR / "tbird2_vectorized_v2.csv"
TRAIN_CSV   = DATA_DIR / "tbird2_v2_train.csv"
VAL_CSV     = DATA_DIR / "tbird2_v2_val.csv"
TEST_CSV    = DATA_DIR / "tbird2_v2_test.csv"

# ── Parameters ────────────────────────────────────────────────────────────────
BATCH_SIZE    = 300
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10
# TEST_RATIO  = 0.20 (remainder)
N_SEGMENTS    = 10   # for stratified chronological split

# ── Stratum definitions ───────────────────────────────────────────────────────
# 0 = Normal (all state 0)
# 1 = N_ only (states 1-13, no 14-33)
# 2 = R_ only (states 14-33, no 1-13)
# 3 = Mixed (both N_ and R_ present)
STRATUM_NAMES = {0: "normal", 1: "N_only", 2: "R_only", 3: "mixed"}

def get_stratum(state_flags: list[int]) -> int:
    has_n = any(1 <= s <= 13 for s in state_flags)
    has_r = any(14 <= s <= 33 for s in state_flags)
    if has_n and has_r:
        return 3
    if has_n:
        return 1
    if has_r:
        return 2
    return 0

# ── Pass 1: read all rows and build batches ───────────────────────────────────
def build_batches(input_csv: Path, batch_size: int):
    """
    Stream input CSV and group into batches of batch_size lines.
    Returns:
        batches: list of lists of (state_flag, template_id) tuples
        strata:  list of stratum labels per batch
    """
    print("Pass 1 — reading and batching...")
    batches  = []
    strata   = []
    buf      = []
    total    = 0

    with input_csv.open("r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            buf.append((int(row[0]), int(row[1])))
            total += 1
            if len(buf) == batch_size:
                batches.append(buf)
                strata.append(get_stratum([r[0] for r in buf]))
                buf = []
            if total % 10_000_000 == 0:
                print(f"  {total:,} lines read, {len(batches):,} batches built...")

    # handle remainder batch if any
    if buf:
        batches.append(buf)
        strata.append(get_stratum([r[0] for r in buf]))

    print(f"  Done. {total:,} lines → {len(batches):,} batches.")
    return batches, strata

# ── Pass 2: stratified chronological assignment ───────────────────────────────
def assign_partitions(strata: list[int], n_segments: int,
                      train_ratio: float, val_ratio: float) -> array.array:
    """
    Divide batches into n_segments chronological segments.
    Within each segment, assign batches to train/val/test by stratum
    to maintain balance across partitions.
    Returns array of partition assignments: 0=train, 1=val, 2=test.
    """
    print("\nPass 2 — assigning partitions...")
    n_batches  = len(strata)
    seg_size   = n_batches // n_segments
    assignments = array.array("B", [0] * n_batches)

    TRAIN, VAL, TEST = 0, 1, 2

    for seg in range(n_segments):
        seg_start = seg * seg_size
        seg_end   = (seg + 1) * seg_size if seg < n_segments - 1 else n_batches

        # group batch indices by stratum within this segment
        stratum_batches: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        for i in range(seg_start, seg_end):
            stratum_batches[strata[i]].append(i)

        # assign each stratum group proportionally
        for s, indices in stratum_batches.items():
            n  = len(indices)
            t  = int(n * train_ratio)
            v  = int(n * (train_ratio + val_ratio))
            for j, idx in enumerate(indices):
                if j < t:
                    assignments[idx] = TRAIN
                elif j < v:
                    assignments[idx] = VAL
                else:
                    assignments[idx] = TEST

    print(f"  Done. {n_batches:,} batches assigned.")
    return assignments

# ── Pass 3: write output files ────────────────────────────────────────────────
def write_partitions(batches, strata, assignments,
                     train_csv, val_csv, test_csv):
    print("\nPass 3 — writing partition files...")
    HEADER = ["batch_id", "stratum", "state_flag", "template_id"]

    stats = {
        0: {"name": "train", "batches": 0, "lines": 0,
            "strata": Counter(), "anomalous_lines": 0},
        1: {"name": "val",   "batches": 0, "lines": 0,
            "strata": Counter(), "anomalous_lines": 0},
        2: {"name": "test",  "batches": 0, "lines": 0,
            "strata": Counter(), "anomalous_lines": 0},
    }

    files = {
        0: train_csv.open("w", newline=""),
        1: val_csv.open("w", newline=""),
        2: test_csv.open("w", newline=""),
    }
    writers = {p: csv.writer(f) for p, f in files.items()}
    for w in writers.values():
        w.writerow(HEADER)

    for batch_id, (batch, stratum, partition) in enumerate(
            zip(batches, strata, assignments)):
        s = stats[partition]
        s["batches"] += 1
        s["strata"][stratum] += 1
        for state_flag, template_id in batch:
            writers[partition].writerow(
                [batch_id, stratum, state_flag, template_id])
            s["lines"] += 1
            if state_flag != 0:
                s["anomalous_lines"] += 1

        if (batch_id + 1) % 100_000 == 0:
            print(f"  {batch_id+1:,} batches written...")

    for f in files.values():
        f.close()

    return stats

# ── Summary report ────────────────────────────────────────────────────────────
def print_summary(stats, total_batches):
    print(f"\n── Partition summary ────────────────────────────────────────────")
    print(f"  {'Partition':<8} {'Batches':>10} {'Lines':>12} "
          f"{'Anom lines':>12} {'Anom %':>8}")
    print(f"  {'-'*56}")
    for p in (0, 1, 2):
        s = stats[p]
        pct = s["anomalous_lines"] / s["lines"] * 100 if s["lines"] > 0 else 0
        print(f"  {s['name']:<8} {s['batches']:>10,} {s['lines']:>12,} "
              f"{s['anomalous_lines']:>12,} {pct:>7.4f}%")

    print(f"\n  Stratum distribution per partition:")
    print(f"  {'Partition':<8} {'normal':>10} {'N_only':>10} "
          f"{'R_only':>10} {'mixed':>10}")
    print(f"  {'-'*52}")
    for p in (0, 1, 2):
        s = stats[p]
        print(f"  {s['name']:<8} "
              f"{s['strata'][0]:>10,} "
              f"{s['strata'][1]:>10,} "
              f"{s['strata'][2]:>10,} "
              f"{s['strata'][3]:>10,}")
    print(f"─────────────────────────────────────────────────────────────────")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    batches, strata = build_batches(INPUT_CSV, BATCH_SIZE)

    # print stratum counts before partitioning
    stratum_counts = Counter(strata)
    total_batches  = len(batches)
    print(f"\n  Stratum breakdown across all {total_batches:,} batches:")
    for s, name in STRATUM_NAMES.items():
        pct = stratum_counts[s] / total_batches * 100
        print(f"    {name:<8} {stratum_counts[s]:>8,}  ({pct:.2f}%)")

    assignments = assign_partitions(
        strata, N_SEGMENTS, TRAIN_RATIO, VAL_RATIO)

    stats = write_partitions(
        batches, strata, assignments,
        TRAIN_CSV, VAL_CSV, TEST_CSV)

    print_summary(stats, total_batches)
    print(f"\nDone. Output written to: {DATA_DIR}")

if __name__ == "__main__":
    main()
