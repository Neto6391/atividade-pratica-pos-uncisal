from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from fraud_risk.domain.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET


DEFAULT_THRESHOLDS = {
    "max_numeric_psi": 0.20,
    "max_categorical_l1": 0.30,
    "min_precision": 0.05,
    "min_recall": 0.60,
    "min_pr_auc": 0.20,
}


def build_monitoring_reference(events: pd.DataFrame) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(events)),
        "fraud_rate": round(float(events[TARGET].mean()), 6),
        "numeric": {feature: _numeric_reference(events[feature]) for feature in NUMERIC_FEATURES},
        "categorical": {feature: _categorical_reference(events[feature]) for feature in CATEGORICAL_FEATURES},
    }


class ModelMonitor:
    def evaluate(self, model_dir: Path, events: pd.DataFrame, output_path: Path) -> dict[str, object]:
        bundle = joblib.load(model_dir / "fraud_model.joblib")
        reference = json.loads((model_dir / "monitoring_reference.json").read_text(encoding="utf-8"))
        model = bundle["model"]
        threshold = float(bundle["threshold"])
        scores = model.predict_proba(events[MODEL_FEATURES])[:, 1]
        predictions = (scores >= threshold).astype(int)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": events.attrs.get("dataset_source", "unknown"),
            "rows": int(len(events)),
            "threshold": round(threshold, 6),
            "performance": _performance(events[TARGET], scores, predictions),
            "drift": _drift(reference, events),
        }
        report["alerts"] = _alerts(report, DEFAULT_THRESHOLDS)
        report["retrain_recommended"] = bool(report["alerts"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def _numeric_reference(series: pd.Series) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").fillna(series.median())
    quantiles = np.unique(np.quantile(clean, np.linspace(0, 1, 11))).tolist()
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "quantiles": quantiles,
        "distribution": _histogram(clean, quantiles),
    }


def _categorical_reference(series: pd.Series) -> dict[str, object]:
    dist = series.astype(str).fillna("__missing__").value_counts(normalize=True).head(20)
    return {"distribution": {str(k): float(v) for k, v in dist.items()}}


def _histogram(series: pd.Series, bins: list[float]) -> list[float]:
    if len(bins) < 2:
        return [1.0]
    counts, _ = np.histogram(series, bins=bins)
    total = max(int(counts.sum()), 1)
    return (counts / total).clip(1e-6).tolist()


def _drift(reference: dict[str, object], events: pd.DataFrame) -> dict[str, object]:
    numeric = {}
    for feature, ref in reference["numeric"].items():
        current = pd.to_numeric(events[feature], errors="coerce").fillna(events[feature].median())
        current_dist = _histogram(current, ref["quantiles"])
        numeric[feature] = round(_psi(ref["distribution"], current_dist), 6)

    categorical = {}
    for feature, ref in reference["categorical"].items():
        current_dist = events[feature].astype(str).fillna("__missing__").value_counts(normalize=True).to_dict()
        categorical[feature] = round(_categorical_l1(ref["distribution"], current_dist), 6)

    return {
        "numeric_psi": numeric,
        "max_numeric_psi": round(max(numeric.values()) if numeric else 0.0, 6),
        "categorical_l1": categorical,
        "max_categorical_l1": round(max(categorical.values()) if categorical else 0.0, 6),
    }


def _performance(y_true: pd.Series, scores: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "fraud_rate": round(float(y_true.mean()), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 6),
        "average_precision_pr_auc": round(float(average_precision_score(y_true, scores)), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _alerts(report: dict[str, object], thresholds: dict[str, float]) -> list[str]:
    alerts = []
    performance = report["performance"]
    drift = report["drift"]
    if drift["max_numeric_psi"] > thresholds["max_numeric_psi"]:
        alerts.append(f"numeric drift PSI {drift['max_numeric_psi']} > {thresholds['max_numeric_psi']}")
    if drift["max_categorical_l1"] > thresholds["max_categorical_l1"]:
        alerts.append(f"categorical drift L1 {drift['max_categorical_l1']} > {thresholds['max_categorical_l1']}")
    if performance["precision"] < thresholds["min_precision"]:
        alerts.append(f"precision {performance['precision']} < {thresholds['min_precision']}")
    if performance["recall"] < thresholds["min_recall"]:
        alerts.append(f"recall {performance['recall']} < {thresholds['min_recall']}")
    if performance["average_precision_pr_auc"] < thresholds["min_pr_auc"]:
        alerts.append(f"PR-AUC {performance['average_precision_pr_auc']} < {thresholds['min_pr_auc']}")
    return alerts


def _psi(expected: list[float], actual: list[float]) -> float:
    expected_arr = np.asarray(expected).clip(1e-6)
    actual_arr = np.asarray(actual).clip(1e-6)
    return float(np.sum((actual_arr - expected_arr) * np.log(actual_arr / expected_arr)))


def _categorical_l1(reference: dict[str, float], current: dict[str, float]) -> float:
    keys = set(reference) | set(current)
    return float(sum(abs(reference.get(key, 0.0) - current.get(key, 0.0)) for key in keys) / 2)
