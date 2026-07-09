from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit that card-fraud detections came from real ccfraud rows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/card_detection_audit.json"))
    args = parser.parse_args()

    zip_path = args.data_dir / "creditcardfraud.zip"
    csv_name = "creditcard.csv"
    df = _load(zip_path, csv_name)
    train_end = int(len(df) * 0.70)
    valid_end = int(len(df) * 0.85)
    test = df.iloc[valid_end:].copy()
    bundle = joblib.load(args.model_dir / "card_misuse_model.joblib")

    fraud_indices = test.index[test["Class"].eq(1)].tolist()
    legit_indices = test.index[test["Class"].eq(0)].tolist()
    cases = [
        _audit_case("first_fraud_in_test", df, fraud_indices[0], bundle),
        _audit_case("first_legitimate_in_test", df, legit_indices[0], bundle),
    ]
    test_scores = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    test_predictions = (test_scores >= float(bundle["threshold"])).astype(int)
    test_labels = test["Class"].astype(int).to_numpy()
    report = {
        "source": "amazon_fdb_source:ccfraud",
        "zip_path": str(zip_path),
        "zip_sha256": _sha256(zip_path),
        "csv_member": csv_name,
        "dataset_rows": int(len(df)),
        "test_slice": {"start_index": valid_end, "end_index": int(len(df) - 1), "rows": int(len(test))},
        "model_artifact": str(args.model_dir / "card_misuse_model.joblib"),
        "threshold": float(bundle["threshold"]),
        "aggregate_test_evidence": {
            "actual_frauds_in_test": int(test_labels.sum()),
            "predicted_frauds_in_test": int(test_predictions.sum()),
            "true_positives": int(((test_predictions == 1) & (test_labels == 1)).sum()),
            "false_positives": int(((test_predictions == 1) & (test_labels == 0)).sum()),
            "false_negatives": int(((test_predictions == 0) & (test_labels == 1)).sum()),
            "true_negatives": int(((test_predictions == 0) & (test_labels == 0)).sum()),
        },
        "audited_rows": cases,
        "interpretation": "The audited fraud row is a real row from creditcard.csv with Class=1. The model artifact was loaded from disk and scored the row with predict_proba.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _load(zip_path: Path, csv_name: str) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Run: python scripts/download_fdb_sources.py")
    with ZipFile(zip_path) as archive:
        with archive.open(csv_name) as csv_file:
            return pd.read_csv(csv_file).sort_values("Time").reset_index(drop=True)


def _audit_case(name: str, df: pd.DataFrame, row_index: int, bundle: dict[str, object]) -> dict[str, object]:
    row = df.loc[row_index]
    features = bundle["features"]
    score = float(bundle["model"].predict_proba(pd.DataFrame([row[features]]))[0, 1])
    action = "block_or_step_up" if score >= float(bundle["threshold"]) else "approve"
    row_payload = row.to_json()
    return {
        "case": name,
        "csv_row_index_after_time_sort": int(row_index),
        "real_label_Class": int(row["Class"]),
        "time": float(row["Time"]),
        "amount": float(row["Amount"]),
        "feature_row_sha256": hashlib.sha256(row_payload.encode("utf-8")).hexdigest(),
        "risk_score": round(score, 6),
        "action": action,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
