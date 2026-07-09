from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from fraud_risk.domain.features import MODEL_FEATURES, TARGET, TIME_COL
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.fdb_local_loader import load_fdb_local_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal backtest for evolving fraud patterns.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/temporal_backtest.json"))
    args = parser.parse_args()

    events = load_fdb_local_events(args.data_dir, PhishingPolicy())
    bundle = joblib.load(args.model_dir / "fraud_model.joblib")
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    events = events.sort_values(TIME_COL).copy()
    events["period"] = pd.to_datetime(events[TIME_COL]).dt.to_period("M").astype(str)
    rows = []
    for period, frame in events.groupby("period", sort=True):
        if frame[TARGET].nunique() < 2:
            continue
        scores = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
        predictions = (scores >= threshold).astype(int)
        rows.append(
            {
                "period": period,
                "rows": int(len(frame)),
                "fraud_rate": round(float(frame[TARGET].mean()), 6),
                "precision": round(float(precision_score(frame[TARGET], predictions, zero_division=0)), 6),
                "recall": round(float(recall_score(frame[TARGET], predictions, zero_division=0)), 6),
                "f1": round(float(f1_score(frame[TARGET], predictions, zero_division=0)), 6),
                "roc_auc": round(float(roc_auc_score(frame[TARGET], scores)), 6),
                "average_precision_pr_auc": round(float(average_precision_score(frame[TARGET], scores)), 6),
            }
        )
    report = {
        "source": "amazon_fdb_source:fraudecom",
        "purpose": "Evidence for evolving fraud monitoring by temporal windows.",
        "periods": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
