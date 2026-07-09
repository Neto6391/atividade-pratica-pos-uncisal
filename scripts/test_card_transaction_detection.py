from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Test detection of a real fraudulent credit-card transaction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/card_transaction_test.json"))
    args = parser.parse_args()

    bundle_path = args.model_dir / "card_misuse_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing {bundle_path}. Run: python scripts/evaluate_card_misuse.py --output-dir artifacts")

    df = _load(args.data_dir / "creditcardfraud.zip")
    _, _, test = _temporal_split(df)
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    features = bundle["features"]

    fraud_case = test[test["Class"].eq(1)].iloc[0]
    legit_case = test[test["Class"].eq(0)].iloc[0]
    fraud_score = float(model.predict_proba(pd.DataFrame([fraud_case[features]]))[0, 1])
    legit_score = float(model.predict_proba(pd.DataFrame([legit_case[features]]))[0, 1])

    report = {
        "source": "amazon_fdb_source:ccfraud",
        "dataset_rows": int(len(df)),
        "dataset_fraud_rate": round(float(df["Class"].mean()), 6),
        "model_artifact": str(bundle_path),
        "model": bundle.get("model_name", "loaded_from_artifact"),
        "threshold": threshold,
        "fraudulent_card_transaction": {
            "real_label": int(fraud_case["Class"]),
            "time": float(fraud_case["Time"]),
            "amount": float(fraud_case["Amount"]),
            "risk_score": round(fraud_score, 6),
            "decision": "block_or_step_up" if fraud_score >= threshold else "approve",
        },
        "legitimate_card_transaction": {
            "real_label": int(legit_case["Class"]),
            "time": float(legit_case["Time"]),
            "amount": float(legit_case["Amount"]),
            "risk_score": round(legit_score, 6),
            "decision": "block_or_step_up" if legit_score >= threshold else "approve",
        },
        "explanation": "The model sees anonymized card-transaction behavior (V1..V28), Amount and Time, then returns a fraud risk score. Scores above the calibrated threshold are treated as suspected card misuse.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/download_fdb_sources.py")
    with ZipFile(path) as archive:
        with archive.open("creditcard.csv") as csv_file:
            return pd.read_csv(csv_file).sort_values("Time").reset_index(drop=True)


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(df) * 0.70)
    valid_end = int(len(df) * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(), df.iloc[valid_end:].copy()


if __name__ == "__main__":
    main()
