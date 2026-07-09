from __future__ import annotations

import numpy as np
import pandas as pd


def compute_separation_metrics(labels: pd.Series | np.ndarray, scores: np.ndarray) -> dict[str, object]:
    labels_arr = np.asarray(labels).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    fraud_scores = scores_arr[labels_arr == 1]
    legit_scores = scores_arr[labels_arr == 0]

    if len(fraud_scores) == 0 or len(legit_scores) == 0:
        return {
            "min_fraud_score": None,
            "max_legit_score": None,
            "separation_gap": None,
            "perfect_separation_possible": False,
            "min_false_positives_at_recall_1_0": None,
            "best_precision_at_recall_1_0": None,
            "fpr_at_recall_1_0": None,
        }

    min_fraud = float(np.min(fraud_scores))
    max_legit = float(np.max(legit_scores))
    separation_gap = min_fraud - max_legit
    threshold = min_fraud * (1.0 - 1e-9)
    min_fp = int((legit_scores >= threshold).sum())
    tp = int(len(fraud_scores))
    precision = tp / (tp + min_fp) if (tp + min_fp) > 0 else 0.0
    fpr = min_fp / len(legit_scores)

    return {
        "min_fraud_score": round(min_fraud, 6),
        "max_legit_score": round(max_legit, 6),
        "separation_gap": round(separation_gap, 6),
        "perfect_separation_possible": bool(separation_gap > 0),
        "min_false_positives_at_recall_1_0": min_fp,
        "best_precision_at_recall_1_0": round(precision, 6),
        "fpr_at_recall_1_0": round(fpr, 6),
    }
