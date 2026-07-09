from dataclasses import dataclass


@dataclass(frozen=True)
class PhishingFilterMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    false_positive_rate: float = 0.0
    separation_gap: float = 0.0
    perfect_separation_possible: bool = False
    min_false_positives_at_recall_1_0: int = 0
    best_precision_at_recall_1_0: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": round(self.threshold, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
            "separation_gap": round(self.separation_gap, 6),
            "perfect_separation_possible": self.perfect_separation_possible,
            "min_false_positives_at_recall_1_0": self.min_false_positives_at_recall_1_0,
            "best_precision_at_recall_1_0": round(self.best_precision_at_recall_1_0, 6),
            "confusion_matrix": {
                "tn": self.true_negative,
                "fp": self.false_positive,
                "fn": self.false_negative,
                "tp": self.true_positive,
            },
        }
