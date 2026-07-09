from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from fraud_risk.domain.metrics import ClassificationMetrics


class ReportPlotter:
    def save_all(
        self,
        output_dir: Path,
        y_true: pd.Series,
        scores: np.ndarray,
        threshold: float,
        metrics_by_model: dict[str, ClassificationMetrics],
        phishing_true: pd.Series,
        phishing_scores: pd.Series,
        phishing_threshold: float,
    ) -> None:
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        self._metric_bars(charts_dir / "model_metrics.png", metrics_by_model)
        self._confusion_matrix(charts_dir / "confusion_matrix.png", y_true, scores, threshold)
        self._precision_recall(charts_dir / "precision_recall_curve.png", y_true, scores, threshold)
        self._roc(charts_dir / "roc_curve.png", y_true, scores)
        self._score_distribution(charts_dir / "score_distribution.png", y_true, scores, threshold)
        self._phishing_distribution(charts_dir / "anti_phishing_filter.png", phishing_true, phishing_scores, phishing_threshold)

    @staticmethod
    def _metric_bars(path: Path, metrics_by_model: dict[str, ClassificationMetrics]) -> None:
        labels = list(metrics_by_model)
        x = np.arange(len(labels))
        width = 0.22
        fig, ax = plt.subplots(figsize=(9, 5))
        for offset, name, values in [
            (-width, "Precision", [m.precision for m in metrics_by_model.values()]),
            (0, "Recall", [m.recall for m in metrics_by_model.values()]),
            (width, "F1", [m.f1 for m in metrics_by_model.values()]),
        ]:
            ax.bar(x + offset, values, width, label=name)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x, labels, rotation=10)
        ax.set_title("Comparacao de modelos")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _confusion_matrix(path: Path, y_true: pd.Series, scores: np.ndarray, threshold: float) -> None:
        predictions = (scores >= threshold).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(matrix, display_labels=["Legitima", "Fraude"]).plot(ax=ax, colorbar=False)
        ax.set_title("Matriz de confusao")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _precision_recall(path: Path, y_true: pd.Series, scores: np.ndarray, threshold: float) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax)
        ax.axvline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:.3f}")
        ax.set_title("Curva Precision-Recall")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _roc(path: Path, y_true: pd.Series, scores: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, scores, ax=ax)
        ax.set_title("Curva ROC")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _score_distribution(path: Path, y_true: pd.Series, scores: np.ndarray, threshold: float) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores[np.asarray(y_true) == 0], bins=40, alpha=0.70, label="Legitima")
        ax.hist(scores[np.asarray(y_true) == 1], bins=40, alpha=0.70, label="Fraude")
        ax.axvline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:.3f}")
        ax.set_title("Distribuicao do score de risco")
        ax.set_xlabel("Score")
        ax.set_ylabel("Eventos")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _phishing_distribution(path: Path, y_true: pd.Series, scores: pd.Series, threshold: float) -> None:
        labels = np.asarray(y_true)
        values = np.asarray(scores)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values[labels == 0], bins=20, alpha=0.70, label="URL normal")
        ax.hist(values[labels == 1], bins=20, alpha=0.70, label="Phishing")
        ax.axvline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:.3f}")
        ax.set_title("Efetividade do filtro anti-phishing")
        ax.set_xlabel("Score anti-phishing")
        ax.set_ylabel("Eventos")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
