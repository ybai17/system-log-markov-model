import re
import csv
import json
import multiprocessing as mp
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(r"")
INPUT_LOG     = DATA_DIR / "tbird2"
TEMPLATE_CSV  = DATA_DIR / "Thunderbird_full.log_templates_v2.csv"
OUTPUT_CSV    = DATA_DIR / "tbird2_vectorized_v2.csv"
VOCAB_JSON    = DATA_DIR / "tbird2_vocab_v2.json"
STATES_JSON   = DATA_DIR / "tbird2_states.json"

# ── Parameters ────────────────────────────────────────────────────────────────
N_WORKERS   = 18
CHUNK_LINES = 500_000
MAX_LINES   = None        # set to None to process the full file

# ── State flag mapping (0 = normal, 1-33 = anomaly labels) ───────────────────
STATE_MAP = {
    "-":           0,
    "N_AUTH":      1,
    "N_CALL_TR":   2,
    "N_CPU":       3,
    "N_LUS_LBUG":  4,
    "N_MAIL":      5,
    "N_NFS":       6,
    "N_OOM":       7,
    "N_PBS_BAIL":  8,
    "N_PBS_BFD1":  9,
    "N_PBS_BFD2":  10,
    "N_PBS_CON2":  11,
    "N_PBS_EPI":   12,
    "N_PBS_SIS":   13,
    "R_CHK_DSK":   14,
    "R_ECC":       15,
    "R_EXT_FS":    16,
    "R_EXT_FS_ABRT1": 17,
    "R_EXT_FS_ABRT2": 18,
    "R_EXT_FS_IO": 19,
    "R_EXT_INODE1": 20,
    "R_EXT_INODE2": 21,
    "R_GPF":       22,
    "R_MPT":       23,
    "R_MTT":       24,
    "R_NMI":       25,
    "R_PAG":       26,
    "R_PAN":       27,
    "R_RIP":       28,
    "R_SCSI0":     29,
    "R_SCSI1":     30,
    "R_SEG":       31,
    "R_SERR":      32,
    "R_VAPI":      33,
}

UNMATCHED_ID = 0

# ── Extraction regexes ────────────────────────────────────────────────────────
PID_RE      = re.compile(r"^\S+?\[\d+\]:\s*")   # process[pid]:
BRACKET_RE  = re.compile(r"^\[.+?\]:\s*")        # [file:line]:
WORD_COL_RE = re.compile(r"^\w[\w/.-]*:\s*")     # WORD: or word/word:

# ── Template compiler ─────────────────────────────────────────────────────────
def compile_template(template: str) -> re.Pattern:
    escaped = re.escape(template).replace(r"<\*>", ".*?")
    return re.compile(f"^{escaped}$", re.DOTALL)

# ── Load and compile templates ────────────────────────────────────────────────
def load_templates(path: Path):
    raw = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            eid      = row["EventId"].strip()
            template = row["EventTemplate"].strip()
            count    = int(row["Occurrences"])
            tid      = int(eid[1:])
            raw.append((tid, eid, template, count))

    raw.sort(key=lambda x: x[3], reverse=True)

    patterns = []
    vocab = {"0": {"event_id": "UNMATCHED", "template": ""}}
    for tid, eid, template, count in raw:
        patterns.append((tid, compile_template(template)))
        vocab[str(tid)] = {"event_id": eid, "template": template}

    return patterns, vocab

# ── Message extraction strategies ─────────────────────────────────────────────
def extract_candidates(raw_msg: str) -> list[tuple[str, str]]:
    candidates = [("raw", raw_msg)]

    m = PID_RE.match(raw_msg)
    if m:
        candidates.append(("strip_pid", raw_msg[m.end():]))

    m = BRACKET_RE.match(raw_msg)
    if m:
        candidates.append(("strip_bracket", raw_msg[m.end():]))

    m = WORD_COL_RE.match(raw_msg)
    if m:
        remainder = raw_msg[m.end():]
        candidates.append(("strip_word", remainder))
        m2 = WORD_COL_RE.match(remainder)
        if m2:
            candidates.append(("strip_word2", remainder[m2.end():]))

    return candidates

# ── Match message against templates ──────────────────────────────────────────
def match_message(raw_msg: str, patterns: list) -> int:
    for strategy, candidate in extract_candidates(raw_msg):
        for tid, pattern in patterns:
            if pattern.match(candidate):
                return tid
    return UNMATCHED_ID

# ── Line parser ───────────────────────────────────────────────────────────────
def parse_line(line: str):
    parts = line.split(None, 9)
    if len(parts) < 10:
        return None
    state = STATE_MAP.get(parts[0], -1)
    if state == -1:
        return None
    return state, parts[9].strip()

# ── Worker function ───────────────────────────────────────────────────────────
def process_chunk(args):
    lines, patterns = args
    results = []
    for line in lines:
        parsed = parse_line(line.rstrip("\n"))
        if parsed is None:
            continue
        state, raw_msg = parsed
        tid = match_message(raw_msg, patterns)
        results.append((state, tid))
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading and compiling templates...")
    patterns, vocab = load_templates(TEMPLATE_CSV)
    print(f"  {len(patterns)} templates compiled, sorted by frequency.")

    # save vocab JSON
    with VOCAB_JSON.open("w") as f:
        json.dump(vocab, f, indent=2)
    print(f"  Vocab saved → {VOCAB_JSON}")

    # save states JSON
    states = {str(v): k for k, v in STATE_MAP.items()}
    with STATES_JSON.open("w") as f:
        json.dump(states, f, indent=2)
    print(f"  States saved → {STATES_JSON}")

    print(f"\nProcessing log file with {N_WORKERS} workers...")
    print(f"  Chunk size: {CHUNK_LINES:,} lines per worker\n")

    total_lines  = 0
    matched      = 0
    unmatched    = 0
    state_counts = {i: 0 for i in range(34)}

    def line_chunks(filepath, chunk_size):
        buf = []
        count = 0
        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                buf.append(line)
                count += 1
                if MAX_LINES and count >= MAX_LINES:
                    yield buf, patterns
                    return
                if len(buf) >= chunk_size:
                    yield buf, patterns
                    buf = []
        if buf:
            yield buf, patterns

    with OUTPUT_CSV.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["state_flag", "template_id"])

        with mp.Pool(processes=N_WORKERS) as pool:
            for batch_num, results in enumerate(
                pool.imap(process_chunk, line_chunks(INPUT_LOG, CHUNK_LINES)), 1
            ):
                writer.writerows(results)
                for state, tid in results:
                    total_lines += 1
                    state_counts[state] += 1
                    if tid == UNMATCHED_ID:
                        unmatched += 1
                    else:
                        matched += 1

                if batch_num % 10 == 0:
                    print(f"  {total_lines:,} lines processed...")

    match_rate = (matched / total_lines * 100) if total_lines > 0 else 0

    print(f"\n── Summary ───────────────────────────────────────────────────────")
    print(f"  Total lines processed : {total_lines:,}")
    print(f"  Matched               : {matched:,}  ({match_rate:.2f}%)")
    print(f"  Unmatched             : {unmatched:,}  ({100 - match_rate:.2f}%)")
    print(f"\n  State distribution:")
    for i in range(34):
        if state_counts[i] > 0:
            label = states[str(i)]
            pct   = state_counts[i] / total_lines * 100
            print(f"    {label:<20} (state {i:>2}): "
                  f"{state_counts[i]:>12,}  ({pct:.4f}%)")
    print(f"─────────────────────────────────────────────────────────────────")
    print(f"\nDone. Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
