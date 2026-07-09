from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraud_risk.domain.features import PHISHING_TARGET, TARGET
from fraud_risk.infrastructure.datasets import temporal_split
from fraud_risk.infrastructure.model_repository import ModelRepository
from fraud_risk.infrastructure.phishing_filter_evaluator import PhishingFilterEvaluator
from fraud_risk.infrastructure.report_plots import ReportPlotter
from fraud_risk.infrastructure.sklearn_fraud_model import SklearnFraudModel


@dataclass(frozen=True)
class TrainingResult:
    selected_model: str
    selected_threshold: float
    dataset: dict[str, object]
    models: dict[str, object]
    anti_phishing_filter: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "models": self.models,
            "anti_phishing_filter": self.anti_phishing_filter,
            "selected_model": self.selected_model,
            "selected_threshold": round(self.selected_threshold, 6),
        }


class TrainFraudModel:
    def __init__(
        self,
        model_service: SklearnFraudModel,
        repository: ModelRepository,
        plotter: ReportPlotter,
        phishing_evaluator: PhishingFilterEvaluator,
    ) -> None:
        self._model_service = model_service
        self._repository = repository
        self._plotter = plotter
        self._phishing_evaluator = phishing_evaluator

    def execute(self, events: Any, output_dir: Path) -> TrainingResult:
        split = temporal_split(events)
        selected = self._model_service.train_select_and_evaluate(split)
        phishing_labels = events.attrs.get("anti_phishing_eval_labels", split.test[PHISHING_TARGET])
        phishing_scores = events.attrs.get("anti_phishing_eval_scores", split.test["url_phishing_score"])
        phishing_metrics = self._phishing_evaluator.evaluate(phishing_labels, phishing_scores)
        result = TrainingResult(
            selected_model=selected.name,
            selected_threshold=selected.threshold,
            dataset={
                "source": events.attrs.get("dataset_source", "unknown"),
                "rows": int(len(events)),
                "fraud_rate": round(float(events[TARGET].mean()), 6),
                "train_rows": int(len(split.train)),
                "valid_rows": int(len(split.valid)),
                "test_rows": int(len(split.test)),
            },
            models={name: metrics.to_dict() for name, metrics in selected.metrics_by_model.items()},
            anti_phishing_filter=phishing_metrics.to_dict(),
        )
        result.anti_phishing_filter["source"] = events.attrs.get("anti_phishing_source", "transaction_landing_url")
        self._repository.save(
            output_dir,
            selected.model,
            selected.threshold,
            result.to_dict(),
            split.train,
            url_threat_model=events.attrs.get("url_threat_model"),
            url_threat_evaluation={
                "labels": phishing_labels.tolist(),
                "scores": phishing_scores.tolist(),
            },
        )
        self._plotter.save_all(
            output_dir,
            selected.y_test,
            selected.test_scores,
            selected.threshold,
            selected.metrics_by_model,
            phishing_labels,
            phishing_scores,
            phishing_metrics.threshold,
        )
        return result
