from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from fraud_risk.domain.features import MODEL_FEATURES
from fraud_risk.infrastructure.monitoring import build_monitoring_reference


class ModelRepository:
    def save(
        self,
        output_dir: Path,
        model: Pipeline,
        threshold: float,
        metrics: dict[str, object],
        reference_events=None,
        url_threat_model: Pipeline | None = None,
        url_threat_evaluation: dict[str, object] | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "threshold": threshold, "features": MODEL_FEATURES}, output_dir / "fraud_model.joblib")
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if url_threat_model is not None:
            joblib.dump(
                {
                    "model": url_threat_model,
                    "evaluation": url_threat_evaluation or {},
                },
                output_dir / "url_threat_model.joblib",
            )
        if reference_events is not None:
            reference = build_monitoring_reference(reference_events)
            (output_dir / "monitoring_reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
