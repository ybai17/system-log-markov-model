import csv
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"")
MODEL_DIR = DATA_DIR / "model"
TEST_CSV  = DATA_DIR / "tbird2_v2_test.csv"
VAL_JSON  = DATA_DIR / "validation_results.json"

# ── Locked hyperparameters (from validation) ──────────────────────────────────
EPSILON   = 0.001
THRESHOLD = -3.790046   # F1-maximizing threshold from plain mean validation
DIM       = 1299

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

def score_batch_mean(windows, T2):
    if not windows:
        return float("-inf")
    return sum(math.log(transition_prob(s0, s1, s2, T2))
               for s0, s1, s2 in windows) / len(windows)

def predict_next(s0, s1, T2):
    """
    Return (predicted_s2, full_distribution) for next-state prediction.
    predicted_s2 is the argmax of P(s2 | s0, s1).
    """
    inner = T2.get((s0, s1))
    if inner is None:
        return -1, {}
    predicted = max(inner, key=inner.get)
    return predicted, inner

# ── Batch reader ──────────────────────────────────────────────────────────────
def iter_batches(test_csv):
    current_bid       = None
    current_stratum   = None
    current_templates = []
    current_states    = []

    with test_csv.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid      = int(row["batch_id"])
            stratum  = int(row["stratum"])
            template = int(row["template_id"])
            state    = int(row["state_flag"])

            if bid != current_bid:
                if current_bid is not None and len(current_templates) >= 3:
                    windows = [(current_templates[i],
                                current_templates[i+1],
                                current_templates[i+2])
                               for i in range(len(current_templates) - 2)]
                    label = 0 if current_stratum == 0 else 1
                    yield (current_bid, current_stratum, label,
                           windows, current_states)
                current_bid       = bid
                current_stratum   = stratum
                current_templates = []
                current_states    = []
            current_templates.append(template)
            current_states.append(state)

    if current_bid is not None and len(current_templates) >= 3:
        windows = [(current_templates[i],
                    current_templates[i+1],
                    current_templates[i+2])
                   for i in range(len(current_templates) - 2)]
        label = 0 if current_stratum == 0 else 1
        yield current_bid, current_stratum, label, windows, current_states

# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(scores, labels, threshold):
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

def roc_auc(scores, labels, n=1000):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    candidates = np.linspace(max(scores), min(scores), n)
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

def pr_auc(scores, labels, n=1000):
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    candidates = np.linspace(max(scores), min(scores), n)
    points = []
    for t in candidates:
        p, r, _, *_ = evaluate(scores, labels, t)
        points.append((r, p))
    points.sort()
    return sum((points[i][0] - points[i-1][0]) *
               (points[i][1] + points[i-1][1]) / 2
               for i in range(1, len(points)))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pi, T1, T2 = load_model(MODEL_DIR)

    print(f"\nLocked hyperparameters:")
    print(f"  Epsilon   : {EPSILON}")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Method    : mean log probability")

    print(f"\nScoring test batches...")

    # anomaly detection accumulators
    scores  = []
    labels  = []
    strata  = []

    # next-state prediction accumulators
    pred_total          = 0
    pred_correct        = 0
    pred_correct_normal = 0
    pred_total_normal   = 0
    pred_by_stratum     = {s: {"correct": 0, "total": 0} for s in range(4)}

    # per-window log prob tracking for score distribution analysis
    normal_window_probs = []
    anom_window_probs   = []

    batch_count = 0
    for bid, stratum, label, windows, states in iter_batches(TEST_CSV):

        # ── Anomaly detection scoring ─────────────────────────────────────────
        log_probs = [math.log(transition_prob(s0, s1, s2, T2))
                     for s0, s1, s2 in windows]
        batch_score = sum(log_probs) / len(log_probs) if log_probs else float("-inf")
        scores.append(batch_score)
        labels.append(label)
        strata.append(stratum)

        # track window-level distributions for analysis
        if label == 0:
            normal_window_probs.extend(log_probs)
        else:
            anom_window_probs.extend(log_probs)

        # ── Next-state prediction ─────────────────────────────────────────────
        for s0, s1, s2 in windows:
            predicted, _ = predict_next(s0, s1, T2)
            if predicted == -1:
                continue  # unseen (s0,s1) pair — skip prediction
            pred_total += 1
            pred_by_stratum[stratum]["total"] += 1
            if predicted == s2:
                pred_correct += 1
                pred_by_stratum[stratum]["correct"] += 1
            if label == 0:
                pred_total_normal += 1
                if predicted == s2:
                    pred_correct_normal += 1

        batch_count += 1
        if batch_count % 10_000 == 0:
            print(f"  {batch_count:,} batches scored...")

    print(f"  Done. {batch_count:,} batches scored.")

    # ── Anomaly detection results ─────────────────────────────────────────────
    p, r, f1, tp, fp, fn, tn = evaluate(scores, labels, THRESHOLD)
    rauc = roc_auc(scores, labels)
    prauc = pr_auc(scores, labels)

    n_anom   = sum(labels)
    n_normal = len(labels) - n_anom

    print(f"\n── Majority Class Baseline ──────────────────────────────────────")
    print(f"  Always predict normal: Accuracy={n_normal/len(labels)*100:.2f}%  F1=0.0000")
    print(f"  ({n_anom:,} anomalous, {n_normal:,} normal batches)")

    print(f"\n── Anomaly Detection Results ────────────────────────────────────")
    print(f"  Method    : mean log probability")
    print(f"  Epsilon   : {EPSILON}")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {rauc:.4f}")
    print(f"  PR-AUC    : {prauc:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TP={tp:>8,}  FP={fp:>8,}")
    print(f"    FN={fn:>8,}  TN={tn:>8,}")

    print(f"\n  Per-stratum breakdown:")
    stratum_names = {1: "N_only", 2: "R_only", 3: "mixed"}
    for s, sname in stratum_names.items():
        s_scores = [sc for sc, st in zip(scores, strata) if st == s]
        s_labels = [lb for lb, st in zip(labels, strata) if st == s]
        if not s_scores:
            continue
        sp, sr, sf1, stp, sfp, sfn, stn = evaluate(
            s_scores, s_labels, THRESHOLD)
        print(f"    {sname:<8} P:{sp:.4f}  R:{sr:.4f}  F1:{sf1:.4f}  "
              f"TP={stp:,}  FP={sfp:,}  FN={sfn:,}  (n={len(s_scores):,})")

    # ── Score distribution summary ────────────────────────────────────────────
    print(f"\n── Score Distribution (window-level log probs) ──────────────────")
    if normal_window_probs:
        print(f"  Normal batches   — "
              f"mean={np.mean(normal_window_probs):.4f}  "
              f"median={np.median(normal_window_probs):.4f}  "
              f"std={np.std(normal_window_probs):.4f}")
    if anom_window_probs:
        print(f"  Anomalous batches — "
              f"mean={np.mean(anom_window_probs):.4f}  "
              f"median={np.median(anom_window_probs):.4f}  "
              f"std={np.std(anom_window_probs):.4f}")

    # ── Next-state prediction results ─────────────────────────────────────────
    print(f"\n── Next-State Prediction Results ────────────────────────────────")
    overall_acc = pred_correct / pred_total * 100 if pred_total > 0 else 0
    normal_acc  = pred_correct_normal / pred_total_normal * 100 \
                  if pred_total_normal > 0 else 0
    anom_total   = pred_total - pred_total_normal
    anom_correct = pred_correct - pred_correct_normal
    anom_acc    = anom_correct / anom_total * 100 if anom_total > 0 else 0

    print(f"  Overall accuracy  : {overall_acc:.2f}%  "
          f"({pred_correct:,} / {pred_total:,} windows)")
    print(f"  Normal batches    : {normal_acc:.2f}%  "
          f"({pred_correct_normal:,} / {pred_total_normal:,} windows)")
    print(f"  Anomalous batches : {anom_acc:.2f}%  "
          f"({anom_correct:,} / {anom_total:,} windows)")

    print(f"\n  Per-stratum prediction accuracy:")
    stratum_names_all = {0: "normal", 1: "N_only", 2: "R_only", 3: "mixed"}
    for s, sname in stratum_names_all.items():
        d = pred_by_stratum[s]
        if d["total"] == 0:
            continue
        acc = d["correct"] / d["total"] * 100
        print(f"    {sname:<8} {acc:.2f}%  "
              f"({d['correct']:,} / {d['total']:,} windows)")

    print(f"─────────────────────────────────────────────────────────────────")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "hyperparameters": {
            "epsilon":   EPSILON,
            "threshold": THRESHOLD,
            "method":    "mean_log_probability",
        },
        "anomaly_detection": {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "roc_auc":   round(rauc, 4),
            "pr_auc":    round(prauc, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "per_stratum": {
                stratum_names[s]: dict(zip(
                    ["precision","recall","f1","tp","fp","fn"],
                    evaluate([sc for sc,st in zip(scores,strata) if st==s],
                             [lb for lb,st in zip(labels,strata) if st==s],
                             THRESHOLD)[:6]
                ))
                for s in (1,2,3)
                if any(st == s for st in strata)
            }
        },
        "next_state_prediction": {
            "overall_accuracy":   round(overall_acc, 4),
            "normal_accuracy":    round(normal_acc, 4),
            "anomalous_accuracy": round(anom_acc, 4),
            "per_stratum": {
                stratum_names_all[s]: {
                    "accuracy": round(pred_by_stratum[s]["correct"] /
                                      pred_by_stratum[s]["total"] * 100, 4)
                                if pred_by_stratum[s]["total"] > 0 else 0,
                    "correct": pred_by_stratum[s]["correct"],
                    "total":   pred_by_stratum[s]["total"],
                }
                for s in range(4) if pred_by_stratum[s]["total"] > 0
            }
        },
        "score_distribution": {
            "normal_mean":   round(float(np.mean(normal_window_probs)), 4)
                             if normal_window_probs else None,
            "normal_std":    round(float(np.std(normal_window_probs)), 4)
                             if normal_window_probs else None,
            "anomalous_mean": round(float(np.mean(anom_window_probs)), 4)
                              if anom_window_probs else None,
            "anomalous_std":  round(float(np.std(anom_window_probs)), 4)
                              if anom_window_probs else None,
        }
    }

    out_path = DATA_DIR / "test_results.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nTest results saved → {out_path}")

if __name__ == "__main__":
    main()
