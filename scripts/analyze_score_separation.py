from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score

from fraud_risk.domain.features import MODEL_FEATURES, TARGET
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.datasets import temporal_split
from fraud_risk.infrastructure.fdb_local_loader import load_fdb_local_events
from fraud_risk.infrastructure.score_separation import compute_separation_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze score separation and trust limits for all fraud components.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/system_trust_report.json"))
    args = parser.parse_args()

    components = [
        _analyze_ecommerce(args.data_dir, args.model_dir),
        _analyze_card(args.data_dir, args.model_dir),
        _analyze_phishing(args.model_dir),
        _analyze_promo(args.data_dir, args.model_dir),
    ]
    report = {
        "question": "Can the system achieve 100% recall with 0 false positives using a single threshold?",
        "answer": "Only if perfect_separation_possible is true for every component.",
        "any_perfect_separation": any(item["perfect_separation_possible"] for item in components),
        "all_perfect_separation": all(item["perfect_separation_possible"] for item in components),
        "components": components,
        "interpretation": (
            "When perfect_separation_possible is false, legitimate cases overlap fraud scores. "
            "The correct operational policy is block_or_step_up, not automatic hard decline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(args.model_dir / "charts" / "score_separation_all_components.png", components)
    print(json.dumps(report, indent=2))


def _analyze_ecommerce(data_dir: Path, model_dir: Path) -> dict[str, object]:
    bundle = joblib.load(model_dir / "fraud_model.joblib")
    events = load_fdb_local_events(data_dir, PhishingPolicy())
    split = temporal_split(events)
    scores = bundle["model"].predict_proba(split.test[MODEL_FEATURES])[:, 1]
    labels = split.test[TARGET]
    threshold = float(bundle["threshold"])
    predictions = (scores >= threshold).astype(int)
    separation = compute_separation_metrics(labels, scores)
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    return {
        "component": "ecommerce",
        "source": "amazon_fdb_source:fraudecom",
        "model_artifact": str(model_dir / "fraud_model.joblib"),
        "threshold": round(threshold, 6),
        "test_rows": int(len(split.test)),
        "recall_at_threshold": round(float((predictions[labels == 1]).mean()), 6),
        "precision_at_threshold": round(float((predictions[labels == 1]).sum() / max(predictions.sum(), 1)), 6),
        "false_positive_rate_at_threshold": round(fp / max(tn + fp, 1), 6),
        **separation,
    }


def _analyze_card(data_dir: Path, model_dir: Path) -> dict[str, object]:
    bundle = joblib.load(model_dir / "card_misuse_model.joblib")
    df = _load_card(data_dir / "creditcardfraud.zip")
    test = df.iloc[int(len(df) * 0.85) :].copy()
    features = bundle["features"]
    scores = bundle["model"].predict_proba(test[features])[:, 1]
    labels = test["Class"]
    threshold = float(bundle["threshold"])
    predictions = (scores >= threshold).astype(int)
    separation = compute_separation_metrics(labels, scores)
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    return {
        "component": "card",
        "source": "amazon_fdb_source:ccfraud",
        "model_artifact": str(model_dir / "card_misuse_model.joblib"),
        "model_name": bundle.get("model_name", "unknown"),
        "threshold": round(threshold, 6),
        "test_rows": int(len(test)),
        "recall_at_threshold": round(float(sk_recall_score(labels, predictions, zero_division=0)), 6),
        "precision_at_threshold": round(float(sk_precision_score(labels, predictions, zero_division=0)), 6),
        "false_positive_rate_at_threshold": round(fp / max(tn + fp, 1), 6),
        **separation,
    }


def _analyze_phishing(model_dir: Path) -> dict[str, object]:
    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    phishing = metrics["anti_phishing_filter"]
    url_bundle_path = model_dir / "url_threat_model.joblib"
    if url_bundle_path.exists():
        bundle = joblib.load(url_bundle_path)
        evaluation = bundle.get("evaluation", {})
        labels = np.asarray(evaluation.get("labels", []), dtype=int)
        scores = np.asarray(evaluation.get("scores", []), dtype=float)
        separation = compute_separation_metrics(labels, scores) if len(labels) else {}
    else:
        separation = {
            "min_fraud_score": None,
            "max_legit_score": None,
            "separation_gap": phishing.get("separation_gap"),
            "perfect_separation_possible": phishing.get("perfect_separation_possible", False),
            "min_false_positives_at_recall_1_0": phishing.get("min_false_positives_at_recall_1_0"),
            "best_precision_at_recall_1_0": phishing.get("best_precision_at_recall_1_0"),
            "fpr_at_recall_1_0": None,
        }
    cm = phishing["confusion_matrix"]
    return {
        "component": "phishing",
        "source": phishing.get("source", "amazon_fdb_source:malurl"),
        "model_artifact": str(url_bundle_path),
        "threshold": phishing["threshold"],
        "test_rows": int(cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]),
        "recall_at_threshold": phishing["recall"],
        "precision_at_threshold": phishing["precision"],
        "false_positive_rate_at_threshold": phishing.get("false_positive_rate", round(cm["fp"] / max(cm["tn"] + cm["fp"], 1), 6)),
        **separation,
    }


def _analyze_promo(data_dir: Path, model_dir: Path) -> dict[str, object]:
    promo_metrics = json.loads((model_dir / "promo_abuse_metrics.json").read_text(encoding="utf-8"))
    events = load_fdb_local_events(data_dir, PhishingPolicy())
    split = temporal_split(events)
    test = split.test
    labels = test[TARGET].astype(int)
    if promo_metrics.get("policy") == "logistic_balanced_ml":
        bundle = joblib.load(model_dir / "promo_abuse_model.joblib")
        features = bundle["features"]
        feature_frame = _promo_feature_frame(test)
        scores = bundle["model"].predict_proba(feature_frame[features])[:, 1]
        threshold = float(bundle["threshold"])
    else:
        scores = _promo_rule_score(test)
        threshold = float(promo_metrics["threshold"])
    predictions = (scores >= threshold).astype(int)
    separation = compute_separation_metrics(labels, scores)
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    return {
        "component": "promo_abuse",
        "source": "amazon_fdb_source:fraudecom",
        "model_artifact": str(model_dir / "promo_abuse_model.joblib") if promo_metrics.get("policy") == "logistic_balanced_ml" else "rule_based",
        "policy": promo_metrics.get("policy"),
        "threshold": round(threshold, 6),
        "test_rows": int(len(test)),
        "recall_at_threshold": round(float(sk_recall_score(labels, predictions, zero_division=0)), 6),
        "precision_at_threshold": round(float(sk_precision_score(labels, predictions, zero_division=0)), 6),
        "false_positive_rate_at_threshold": round(fp / max(tn + fp, 1), 6),
        **separation,
        "note": promo_metrics.get("note"),
    }


def _promo_feature_frame(events: pd.DataFrame) -> pd.DataFrame:
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
    features = _promo_feature_frame(events)
    return np.asarray(
        0.20 * features["source_ads"]
        + 0.30 * features["new_account"]
        + 0.20 * features["same_device"]
        + 0.15 * features["same_ip"]
        + 0.15 * features["fast_purchase"]
    )


def _load_card(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        with archive.open("creditcard.csv") as csv_file:
            return pd.read_csv(csv_file).sort_values("Time").reset_index(drop=True)


def _plot(path: Path, components: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(item["component"]) for item in components]
    min_fp = [item.get("min_false_positives_at_recall_1_0") or 0 for item in components]
    gaps = [item.get("separation_gap") or 0 for item in components]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, min_fp, color="tab:orange")
    axes[0].set_title("Min FP at recall 1.0")
    axes[0].set_ylabel("False positives")
    axes[0].tick_params(axis="x", rotation=20)

    colors = ["tab:green" if item["perfect_separation_possible"] else "tab:red" for item in components]
    axes[1].bar(names, gaps, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Separation gap (min fraud - max legit)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
