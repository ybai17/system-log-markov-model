import csv
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(r"")
ORIGINAL_CSV    = DATA_DIR / "Thunderbird_full.log_templates.csv"
NEW_CSV         = DATA_DIR / "new_templates.csv"
MERGED_CSV      = DATA_DIR / "Thunderbird_full.log_templates_v2.csv"

def load_csv(path, delimiter=","):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append({
                "EventId":       row["EventId"].strip(),
                "EventTemplate": row["EventTemplate"].strip(),
                "Occurrences":   int(row["Occurrences"]),
            })
    return rows

def compile_template(template):
    escaped = re.escape(template).replace(r"<\*>", ".*?")
    return re.compile(f"^{escaped}$", re.DOTALL)

def main():
    print("Loading original templates...")
    original = load_csv(ORIGINAL_CSV, delimiter=",")
    print(f"  {len(original)} original templates loaded.")

    print("Loading new templates...")
    new = load_csv(NEW_CSV, delimiter="\t")
    print(f"  {len(new)} new templates loaded.")

    # ── Check for duplicate template strings ─────────────────────────────────
    print("\nChecking for duplicates...")
    orig_templates = {r["EventTemplate"] for r in original}
    orig_ids       = {r["EventId"] for r in original}
    new_ids        = {r["EventId"] for r in new}

    id_conflicts = orig_ids & new_ids
    if id_conflicts:
        print(f"  WARNING: EventId conflicts found: {sorted(id_conflicts)}")
        print("  These will be skipped.")
    else:
        print("  No EventId conflicts.")

    template_conflicts = []
    for r in new:
        if r["EventTemplate"] in orig_templates:
            template_conflicts.append(r["EventId"])

    if template_conflicts:
        print(f"  WARNING: Template string already exists for: "
              f"{template_conflicts}")
        print("  These will be skipped.")
    else:
        print("  No duplicate template strings.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    skip_ids = id_conflicts | set(template_conflicts)
    added = []
    skipped = []

    for r in new:
        if r["EventId"] in skip_ids or r["EventTemplate"] in orig_templates:
            skipped.append(r["EventId"])
        else:
            added.append(r)

    merged = original + added

    # ── Write merged CSV ──────────────────────────────────────────────────────
    with MERGED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["EventId","EventTemplate","Occurrences"])
        writer.writeheader()
        writer.writerows(merged)

    print(f"\n── Merge Summary ────────────────────────────────────────────────")
    print(f"  Original templates : {len(original)}")
    print(f"  New templates added: {len(added)}")
    print(f"  Skipped (conflicts): {len(skipped)}")
    print(f"  Total in merged    : {len(merged)}")
    print(f"\n  Added:")
    for r in added:
        print(f"    {r['EventId']:<10} {r['EventTemplate'][:70]}")
    if skipped:
        print(f"\n  Skipped: {skipped}")
    print(f"\nMerged CSV saved → {MERGED_CSV}")

    # ── Quick regex validation ────────────────────────────────────────────────
    print("\nValidating new templates compile correctly...")
    errors = []
    for r in added:
        try:
            compile_template(r["EventTemplate"])
        except re.error as e:
            errors.append((r["EventId"], str(e)))
    if errors:
        print(f"  WARNING: {len(errors)} templates failed to compile:")
        for eid, err in errors:
            print(f"    {eid}: {err}")
    else:
        print(f"  All {len(added)} new templates compile successfully.")

if __name__ == "__main__":
    main()
