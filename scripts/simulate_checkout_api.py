from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from fraud_risk.infrastructure.card_cascade import score_card_cascade


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate checkout API calling the saved card fraud model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkout_api_decision_log.json"))
    args = parser.parse_args()

    bundle, mode = _load_card_bundle(args.model_dir)
    df = _load(args.data_dir / "creditcardfraud.zip")
    _, _, test = _temporal_split(df)
    fraud_case = test[test["Class"].eq(1)].iloc[0]
    legit_case = test[test["Class"].eq(0)].iloc[0]

    log = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "card_detection_mode": mode,
        "model_artifact": str(args.model_dir / ("card_cascade_model.joblib" if mode == "cascade" else "card_misuse_model.joblib")),
        "events": [
            _score_checkout("ord-real-fraud-001", fraud_case, bundle, mode),
            _score_checkout("ord-real-legit-001", legit_case, bundle, mode),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


def _load_card_bundle(model_dir: Path) -> tuple[dict[str, object], str]:
    cascade_path = model_dir / "card_cascade_model.joblib"
    if cascade_path.exists():
        return joblib.load(cascade_path), "cascade"
    bundle_path = model_dir / "card_misuse_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("Missing card model. Run: python scripts/evaluate_card_misuse.py --output-dir artifacts")
    return joblib.load(bundle_path), "single_stage"


def _score_checkout(order_id: str, row: pd.Series, bundle: dict[str, object], mode: str) -> dict[str, object]:
    if mode == "cascade":
        decision = score_card_cascade(bundle, row)
        return {
            "api": "POST /checkout/authorize",
            "order_id": order_id,
            "source_dataset": "amazon_fdb_source:ccfraud",
            "real_label": int(row["Class"]),
            "amount": float(row["Amount"]),
            "stage1_score": round(decision.stage1_score, 6),
            "stage1_threshold": round(decision.stage1_threshold, 6),
            "stage2_score": round(decision.stage2_score or 0.0, 6),
            "stage2_threshold": round(decision.stage2_threshold, 6),
            "action": decision.action,
            "system_effect": _effect(decision.action),
        }

    features = bundle["features"]
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    risk_score = float(model.predict_proba(pd.DataFrame([row[features]]))[0, 1])
    action = "block_or_step_up" if risk_score >= threshold else "approve"
    return {
        "api": "POST /checkout/authorize",
        "order_id": order_id,
        "source_dataset": "amazon_fdb_source:ccfraud",
        "real_label": int(row["Class"]),
        "amount": float(row["Amount"]),
        "risk_score": round(risk_score, 6),
        "action": action,
        "system_effect": _effect(action),
    }


def _effect(action: str) -> str:
    if action == "block_or_step_up":
        return "Payment is not captured; customer must pass stronger authentication or transaction goes to review."
    if action == "manual_review":
        return "Payment is held for specialist review after gatekeeper alert."
    return "Payment authorization can continue."


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
