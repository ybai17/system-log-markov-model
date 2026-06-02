import csv
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"")
MODEL_DIR = DATA_DIR / "model"
VAL_CSV   = DATA_DIR / "tbird2_v2_val.csv"

# ── Parameters ────────────────────────────────────────────────────────────────
DIM          = 1299
BATCH_SIZE   = 300
N_THRESHOLDS = 1000
EPSILON      = 0.001           # best from previous sweep

# Trim percentages to sweep for trimmed mean
TRIM_PCTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(model_dir):
    print("Loading model files...")
    pi = {}
    with (model_dir / "S0.csv").open("r") as f:
        for row in csv.DictReader(f):
            s0 = int(row["state_0"])
            pi[s0] = pi.get(s0, 0.0) + float(row["probability"])

    T1 = defaultdict(dict)
    with (model_dir / "S0S1.csv").open("r") as f:
        for row in csv.DictReader(f):
            s0, s1 = int(row["state_0"]), int(row["state_1"])
            T1[s0][s1] = T1[s0].get(s1, 0.0) + float(row["probability"])

    T2 = defaultdict(dict)
    with (model_dir / "S0S1S2.csv").open("r") as f:
        for row in csv.DictReader(f):
            s0 = int(row["state_0"])
            s1 = int(row["state_1"])
            s2 = int(row["state_2"])
            T2[(s0,s1)][s2] = T2[(s0,s1)].get(s2, 0.0) + float(row["probability"])

    print(f"  S0: {len(pi)} | S0S1: {sum(len(v) for v in T1.values())} | "
          f"S0S1S2: {sum(len(v) for v in T2.values())} entries")
    return pi, T1, T2

# ── Scoring ───────────────────────────────────────────────────────────────────
def transition_prob(s0, s1, s2, T2):
    inner = T2.get((s0, s1))
    if inner is None:
        return EPSILON
    return max(inner.get(s2, 0.0), EPSILON)

def window_log_probs(windows, T2):
    """Compute log probability for each window — shared across all methods."""
    return [math.log(transition_prob(s0, s1, s2, T2))
            for s0, s1, s2 in windows]

def score_mean(log_probs):
    return sum(log_probs) / len(log_probs) if log_probs else float("-inf")

def score_min(log_probs):
    return min(log_probs) if log_probs else float("-inf")

def score_trimmed_mean(log_probs, trim_pct):
    """
    Mean log probability after trimming the lowest trim_pct of windows.
    trim_pct=0.10 drops the 10% lowest-scoring windows before averaging.
    """
    if not log_probs:
        return float("-inf")
    n_trim = int(len(log_probs) * trim_pct)
    trimmed = sorted(log_probs)[n_trim:]  # drop lowest n_trim
    return sum(trimmed) / len(trimmed) if trimmed else float("-inf")

# ── Batch reader ──────────────────────────────────────────────────────────────
def iter_batches(val_csv):
    current_bid = None
    current_stratum = None
    current_templates = []

    with val_csv.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid      = int(row["batch_id"])
            stratum  = int(row["stratum"])
            template = int(row["template_id"])

            if bid != current_bid:
                if current_bid is not None and len(current_templates) >= 3:
                    windows = [(current_templates[i],
                                current_templates[i+1],
                                current_templates[i+2])
                               for i in range(len(current_templates) - 2)]
                    yield current_bid, current_stratum, \
                          0 if current_stratum == 0 else 1, windows
                current_bid      = bid
                current_stratum  = stratum
                current_templates = []
            current_templates.append(template)

    if current_bid is not None and len(current_templates) >= 3:
        windows = [(current_templates[i],
                    current_templates[i+1],
                    current_templates[i+2])
                   for i in range(len(current_templates) - 2)]
        yield current_bid, current_stratum, \
              0 if current_stratum == 0 else 1, windows

# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate_threshold(scores, labels, threshold):
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score < threshold else 0
        if   pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        else:                          tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1, tp, fp, fn, tn

def select_threshold_f1(scores, labels):
    candidates = np.linspace(min(scores), max(scores), N_THRESHOLDS)
    best = {"threshold": None, "f1": -1, "precision": 0, "recall": 0}
    for t in candidates:
        p, r, f1, *_ = evaluate_threshold(scores, labels, t)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1, "precision": p, "recall": r}
    return best

def select_threshold_precision(scores, labels, min_recall=0.80):
    """
    Find threshold that maximizes precision while maintaining recall >= min_recall.
    Useful for reducing false positives when recall can afford to drop slightly.
    """
    candidates = np.linspace(min(scores), max(scores), N_THRESHOLDS)
    best = {"threshold": None, "precision": -1, "f1": 0, "recall": 0}
    for t in candidates:
        p, r, f1, *_ = evaluate_threshold(scores, labels, t)
        if r >= min_recall and p > best["precision"]:
            best = {"threshold": float(t), "precision": p, "f1": f1, "recall": r}
    return best

def roc_auc(scores, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    candidates = np.linspace(max(scores), min(scores), N_THRESHOLDS)
    points = [(0.0, 0.0)]
    for t in candidates:
        tp = sum(1 for s, l in zip(scores, labels) if s < t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s < t and l == 0)
        points.append((fp / n_neg, tp / n_pos))
    points.append((1.0, 1.0))
    points.sort()
    return sum((points[i][0] - points[i-1][0]) *
               (points[i][1] + points[i-1][1]) / 2
               for i in range(1, len(points)))

def pr_auc(scores, labels):
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    candidates = np.linspace(max(scores), min(scores), N_THRESHOLDS)
    points = []
    for t in candidates:
        p, r, _, *_ = evaluate_threshold(scores, labels, t)
        points.append((r, p))
    points.sort()
    return sum((points[i][0] - points[i-1][0]) *
               (points[i][1] + points[i-1][1]) / 2
               for i in range(1, len(points)))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pi, T1, T2 = load_model(MODEL_DIR)

    print("\nReading validation batches and computing window log probs...")
    all_log_probs = []
    all_labels    = []
    all_strata    = []
    batch_count   = 0

    for bid, stratum, label, windows in iter_batches(VAL_CSV):
        all_log_probs.append(window_log_probs(windows, T2))
        all_labels.append(label)
        all_strata.append(stratum)
        batch_count += 1
        if batch_count % 10_000 == 0:
            print(f"  {batch_count:,} batches processed...")

    print(f"  Done. {batch_count:,} batches loaded.")

    # ── Baseline ──────────────────────────────────────────────────────────────
    n_anom   = sum(all_labels)
    n_normal = len(all_labels) - n_anom
    print(f"\n── Majority Class Baseline ──────────────────────────────────────")
    print(f"  Always predict normal: Accuracy={n_normal/len(all_labels)*100:.2f}%  "
          f"F1=0.0000")
    print(f"  ({n_anom:,} anomalous, {n_normal:,} normal batches)")

    # ── Trimmed mean sweep ────────────────────────────────────────────────────
    print(f"\n── Trimmed Mean Sweep (epsilon={EPSILON}) ────────────────────────")
    print(f"  {'Trim%':<8} {'F1(max)':>8} {'Prec':>7} {'Recall':>7} "
          f"{'F1(p80)':>8} {'Prec':>7} {'Recall':>7} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print(f"  {'-'*76}")

    best_overall = {"f1": -1}
    all_results  = {}

    for trim_pct in TRIM_PCTS:
        if trim_pct == 0.0:
            scores = [score_mean(lp) for lp in all_log_probs]
            label  = "mean"
        else:
            scores = [score_trimmed_mean(lp, trim_pct) for lp in all_log_probs]
            label  = f"trim{int(trim_pct*100):02d}"

        # F1-maximizing threshold
        best_f1   = select_threshold_f1(scores, all_labels)
        # Precision-maximizing threshold at recall >= 0.80
        best_prec = select_threshold_precision(scores, all_labels, min_recall=0.80)
        rauc      = roc_auc(scores, all_labels)
        prauc     = pr_auc(scores, all_labels)

        all_results[label] = {
            "trim_pct":        trim_pct,
            "f1_threshold":    best_f1["threshold"],
            "f1":              best_f1["f1"],
            "f1_precision":    best_f1["precision"],
            "f1_recall":       best_f1["recall"],
            "prec_threshold":  best_prec["threshold"],
            "prec_f1":         best_prec["f1"],
            "prec_precision":  best_prec["precision"],
            "prec_recall":     best_prec["recall"],
            "roc_auc":         rauc,
            "pr_auc":          prauc,
            "scores":          scores,
        }

        marker = " ◄" if best_f1["f1"] > best_overall["f1"] else ""
        print(f"  {trim_pct*100:>5.0f}%   "
              f"{best_f1['f1']:>8.4f} {best_f1['precision']:>7.4f} "
              f"{best_f1['recall']:>7.4f}   "
              f"{best_prec['f1']:>8.4f} {best_prec['precision']:>7.4f} "
              f"{best_prec['recall']:>7.4f}  "
              f"{rauc:>8.4f} {prauc:>8.4f}{marker}")

        if best_f1["f1"] > best_overall["f1"]:
            best_overall = {**all_results[label], "label": label}

    # ── Detailed report for best F1 config ────────────────────────────────────
    b      = best_overall
    scores = all_results[b["label"]]["scores"]

    print(f"\n── Best Configuration (max F1) ──────────────────────────────────")
    print(f"  Method    : {b['label']}")
    print(f"  Threshold : {b['f1_threshold']:.6f}")
    p, r, f1, tp, fp, fn, tn = evaluate_threshold(
        scores, all_labels, b["f1_threshold"])
    print(f"  Precision : {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    print(f"  ROC-AUC   : {b['roc_auc']:.4f}  PR-AUC: {b['pr_auc']:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TP={tp:>8,}  FP={fp:>8,}")
    print(f"    FN={fn:>8,}  TN={tn:>8,}")

    # ── Precision-focused report ───────────────────────────────────────────────
    print(f"\n── Precision-Focused Threshold (recall ≥ 80%) ───────────────────")
    for label, res in all_results.items():
        if res["prec_threshold"] is None:
            continue
        p2, r2, f2, tp2, fp2, fn2, tn2 = evaluate_threshold(
            res["scores"], all_labels, res["prec_threshold"])
        print(f"  {label:<10} threshold={res['prec_threshold']:>8.4f}  "
              f"P={p2:.4f}  R={r2:.4f}  F1={f2:.4f}  "
              f"TP={tp2:,}  FP={fp2:,}")

    # ── Per-stratum for best config ────────────────────────────────────────────
    print(f"\n  Per-stratum (best F1 config, F1-threshold):")
    for s, sname in {1: "N_only", 2: "R_only", 3: "mixed"}.items():
        s_scores = [sc for sc, st in zip(scores, all_strata) if st == s]
        s_labels = [lb for lb, st in zip(all_labels, all_strata) if st == s]
        if not s_scores:
            continue
        sp, sr, sf1, *_ = evaluate_threshold(
            s_scores, s_labels, b["f1_threshold"])
        print(f"    {sname:<8} P:{sp:.4f}  R:{sr:.4f}  F1:{sf1:.4f}  "
              f"(n={len(s_scores):,})")

    print(f"─────────────────────────────────────────────────────────────────")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "epsilon":         EPSILON,
        "best_label":      b["label"],
        "best_f1_threshold":   b["f1_threshold"],
        "best_prec_threshold": b["prec_threshold"],
        "best_f1":         b["f1"],
        "best_roc_auc":    b["roc_auc"],
        "best_pr_auc":     b["pr_auc"],
        "all_results": {
            k: {ky: v for ky, v in vv.items() if ky != "scores"}
            for k, vv in all_results.items()
        }
    }
    out_path = DATA_DIR / "validation_results.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nValidation results saved → {out_path}")

if __name__ == "__main__":
    main()
