from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score

from fraud_risk.domain.phishing_metrics import PhishingFilterMetrics
from fraud_risk.infrastructure.score_separation import compute_separation_metrics


class PhishingFilterEvaluator:
    def evaluate(self, labels: pd.Series, scores: pd.Series, min_recall: float = 0.90) -> PhishingFilterMetrics:
        threshold = self._select_threshold(labels, scores, min_recall)
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        separation = compute_separation_metrics(labels, scores.to_numpy())
        return PhishingFilterMetrics(
            threshold=threshold,
            precision=float(precision_score(labels, predictions, zero_division=0)),
            recall=float(recall_score(labels, predictions, zero_division=0)),
            f1=float(f1_score(labels, predictions, zero_division=0)),
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            true_positive=int(tp),
            false_positive_rate=fp / max(tn + fp, 1),
            separation_gap=float(separation["separation_gap"] or 0.0),
            perfect_separation_possible=bool(separation["perfect_separation_possible"]),
            min_false_positives_at_recall_1_0=int(separation["min_false_positives_at_recall_1_0"] or 0),
            best_precision_at_recall_1_0=float(separation["best_precision_at_recall_1_0"] or 0.0),
        )

    @staticmethod
    def _select_threshold(labels: pd.Series, scores: pd.Series, min_recall: float) -> float:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        candidates = np.where(recall[:-1] >= min_recall)[0]
        best = candidates[np.argmax(f1_values[candidates])] if len(candidates) else int(np.argmax(f1_values))
        return float(thresholds[best])
