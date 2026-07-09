from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_risk.domain.features import TARGET
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.datasets import temporal_split
from fraud_risk.infrastructure.fdb_local_loader import load_fdb_local_events
from fraud_risk.infrastructure.score_separation import compute_separation_metrics

PROMO_FEATURES = [
    "source_ads",
    "new_account",
    "same_device",
    "same_ip",
    "fast_purchase",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate promotion/account-abuse rules on FDB fraudecom source.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-recall", type=float, default=0.80)
    args = parser.parse_args()

    events = load_fdb_local_events(args.data_dir, PhishingPolicy())
    split = temporal_split(events)
    train_features = _feature_frame(split.train)
    test_features = _feature_frame(split.test)
    labels_train = split.train[TARGET].astype(int)
    labels_test = split.test[TARGET].astype(int)

    rule_scores_train = _promo_rule_score(split.train)
    rule_scores_test = _promo_rule_score(split.test)
    rule_threshold = _best_threshold(labels_train, rule_scores_train, args.min_recall, mode="f1")
    rule_predictions = (rule_scores_test >= rule_threshold).astype(int)

    ml_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    ml_model.fit(train_features, labels_train)
    ml_scores_test = ml_model.predict_proba(test_features)[:, 1]
    ml_threshold = _best_threshold(labels_train, ml_model.predict_proba(train_features)[:, 1], args.min_recall, mode="f1")
    ml_predictions = (ml_scores_test >= ml_threshold).astype(int)

    rule_f1 = f1_score(labels_test, rule_predictions, zero_division=0)
    ml_f1 = f1_score(labels_test, ml_predictions, zero_division=0)
    use_ml = ml_f1 >= rule_f1

    if use_ml:
        scores = ml_scores_test
        threshold = ml_threshold
        predictions = ml_predictions
        policy = "logistic_balanced_ml"
        selected_model = ml_model
    else:
        scores = rule_scores_test
        threshold = rule_threshold
        predictions = rule_predictions
        policy = "ads_campaign_or_new_account_plus_velocity"
        selected_model = None

    tn, fp, fn, tp = confusion_matrix(labels_test, predictions, labels=[0, 1]).ravel()
    separation = compute_separation_metrics(labels_test, scores)
    report = {
        "source": "amazon_fdb_source:fraudecom",
        "rows": int(len(events)),
        "test_rows": int(len(split.test)),
        "policy": policy,
        "min_recall": round(float(args.min_recall), 6),
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(labels_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(labels_test, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(labels_test, predictions, zero_division=0)), 6),
        "false_positive_rate": round(fp / max(tn + fp, 1), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "candidate_comparison": {
            "rule_based": {
                "threshold": round(float(rule_threshold), 6),
                "f1_on_test": round(float(rule_f1), 6),
            },
            "logistic_balanced_ml": {
                "threshold": round(float(ml_threshold), 6),
                "f1_on_test": round(float(ml_f1), 6),
            },
        },
        **separation,
        "note": "Fraudecom has no coupon column; this evaluates campaign/source and fake-account promotion abuse risk using real ecommerce events.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "promo_abuse_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if selected_model is not None:
        joblib.dump(
            {"model": selected_model, "threshold": threshold, "features": PROMO_FEATURES, "policy": policy},
            args.output_dir / "promo_abuse_model.joblib",
        )
    _plot(args.output_dir / "charts" / "promo_abuse_confusion_matrix.png", labels_test, predictions)
    print(json.dumps(report, indent=2))


def _feature_frame(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_ads": events["channel"].astype(str).str.lower().eq("ads").astype(float),
            "new_account": (events["account_age_minutes"] <= 60).astype(float),
            "same_device": (events["device_users_24h"] > 1).astype(float),
            "same_ip": (events["ip_users_24h"] > 1).astype(float),
            "fast_purchase": (events["time_since_signup_minutes"] <= 5).astype(float),
        }
    )


def _promo_rule_score(events: pd.DataFrame) -> np.ndarray:
    features = _feature_frame(events)
    return np.asarray(
        0.20 * features["source_ads"]
        + 0.30 * features["new_account"]
        + 0.20 * features["same_device"]
        + 0.15 * features["same_ip"]
        + 0.15 * features["fast_purchase"]
    )


def _best_threshold(labels, scores: np.ndarray, min_recall: float, mode: str = "f1") -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    candidates = np.where((recall[:-1] >= min_recall) & (thresholds >= 0.05))[0]
    if len(candidates):
        return float(thresholds[candidates[np.argmax(f1_values[candidates])]])
    return float(thresholds[int(np.argmax(f1_values))])


def _plot(path: Path, labels, predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(labels, predictions, labels=[0, 1]), display_labels=["Legitima", "Fraude"]).plot(ax=ax, colorbar=False)
    ax.set_title("Abuso promocional/contas falsas")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
