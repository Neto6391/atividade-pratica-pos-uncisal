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

from fraud_risk.domain.features import MODEL_FEATURES
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.card_cascade import score_card_cascade
from fraud_risk.infrastructure.fdb_local_loader import load_fdb_local_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a full online purchase scored by fraud models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/online_purchase_decision_log.json"))
    args = parser.parse_args()

    thresholds = _load_thresholds(args.model_dir)
    ecommerce_model = joblib.load(args.model_dir / "fraud_model.joblib")
    card_bundle, card_mode = _load_card_bundle(args.model_dir)
    ecommerce_events = load_fdb_local_events(args.data_dir, PhishingPolicy())
    card_events = _load_card_events(args.data_dir / "creditcardfraud.zip")

    ecommerce_case = ecommerce_events[ecommerce_events["is_fraud"].eq(1)].iloc[0]
    card_case = _first_detected_card_case(card_events, card_bundle, card_mode)

    ecommerce_score = float(ecommerce_model["model"].predict_proba(pd.DataFrame([ecommerce_case[MODEL_FEATURES]]))[0, 1])
    promo_score = _promo_abuse_score(ecommerce_case, args.model_dir)
    phishing_score = float(ecommerce_case["url_phishing_score"])

    if card_mode == "cascade":
        card_decision = score_card_cascade(card_bundle, card_case)
        card_scores = {
            "card_stage1_score": round(card_decision.stage1_score, 6),
            "card_stage1_threshold": round(card_decision.stage1_threshold, 6),
            "card_stage2_score": round(card_decision.stage2_score or 0.0, 6),
            "card_stage2_threshold": round(card_decision.stage2_threshold, 6),
        }
        card_component_decision = card_decision.action if card_decision.action != "manual_review" else "review"
    else:
        card_score = float(card_bundle["model"].predict_proba(pd.DataFrame([card_case[card_bundle["features"]]]))[0, 1])
        card_scores = {
            "card_risk_score": round(card_score, 6),
            "card_threshold": round(float(card_bundle["threshold"]), 6),
        }
        card_component_decision = _decision(card_score, float(card_bundle["threshold"]))

    decisions = {
        "ecommerce_transaction": _decision(ecommerce_score, thresholds["ecommerce"]),
        "card_misuse": card_component_decision,
        "promo_account_abuse": "review" if promo_score >= thresholds["promo"] else "pass",
        "phishing": "review" if phishing_score >= thresholds["phishing"] else "pass",
    }
    if decisions["card_misuse"] == "review":
        decisions["card_misuse"] = "manual_review"
    final_action = "block_or_step_up" if "block_or_step_up" in decisions.values() else ("manual_review" if "review" in decisions.values() or "manual_review" in decisions.values() else "approve")

    log = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": "online_purchase_checkout",
        "api": "POST /checkout/purchase",
        "order_id": "ord-online-real-risk-001",
        "card_detection_mode": card_mode,
        "data_sources": {
            "ecommerce": "amazon_fdb_source:fraudecom",
            "card": "amazon_fdb_source:ccfraud",
            "url": "amazon_fdb_source:malurl via url_phishing_score",
        },
        "request_payload": {
            "purchase_amount_from_ecommerce_event": float(ecommerce_case["amount"]),
            "card_amount_from_card_event": float(card_case["Amount"]),
            "channel": str(ecommerce_case["channel"]),
            "browser": str(ecommerce_case["browser"]),
            "new_device": str(ecommerce_case["new_device"]),
            "landing_url": str(ecommerce_case["landing_url"]),
            "promo_code": "WELCOME10",
        },
        "ground_truth_from_datasets": {
            "ecommerce_is_fraud": int(ecommerce_case["is_fraud"]),
            "card_Class": int(card_case["Class"]),
        },
        "model_scores": {
            "ecommerce_risk_score": round(ecommerce_score, 6),
            "ecommerce_threshold": round(thresholds["ecommerce"], 6),
            **card_scores,
            "promo_abuse_score": round(float(promo_score), 6),
            "promo_threshold": round(thresholds["promo"], 6),
            "phishing_score": round(phishing_score, 6),
            "phishing_threshold": round(thresholds["phishing"], 6),
        },
        "component_decisions": decisions,
        "final_action": final_action,
        "system_effect": _system_effect(final_action),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


def _load_card_bundle(model_dir: Path) -> tuple[dict[str, object], str]:
    cascade_path = model_dir / "card_cascade_model.joblib"
    if cascade_path.exists():
        return joblib.load(cascade_path), "cascade"
    return joblib.load(model_dir / "card_misuse_model.joblib"), "single_stage"


def _load_thresholds(model_dir: Path) -> dict[str, float]:
    ecommerce = joblib.load(model_dir / "fraud_model.joblib")
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    promo = json.loads((model_dir / "promo_abuse_metrics.json").read_text(encoding="utf-8"))
    return {
        "ecommerce": float(ecommerce["threshold"]),
        "phishing": float(metrics["anti_phishing_filter"]["threshold"]),
        "promo": float(promo["threshold"]),
    }


def _load_card_events(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        with archive.open("creditcard.csv") as csv_file:
            return pd.read_csv(csv_file).sort_values("Time").reset_index(drop=True)


def _first_detected_card_case(card_events: pd.DataFrame, bundle: dict[str, object], mode: str) -> pd.Series:
    valid_end = int(len(card_events) * 0.85)
    test = card_events.iloc[valid_end:].copy()
    frauds = test[test["Class"].eq(1)]
    if mode == "cascade":
        for _, row in frauds.iterrows():
            decision = score_card_cascade(bundle, row)
            if decision.stage2_passed:
                return row
        raise RuntimeError("No cascade-detected fraudulent card transaction found in the test split.")
    scores = bundle["model"].predict_proba(frauds[bundle["features"]])[:, 1]
    detected = frauds[scores >= float(bundle["threshold"])]
    if detected.empty:
        raise RuntimeError("No detected fraudulent card transaction found in the test split.")
    return detected.iloc[0]


def _promo_abuse_score(event: pd.Series, model_dir: Path) -> float:
    bundle_path = model_dir / "promo_abuse_model.joblib"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        features = pd.DataFrame(
            [
                {
                    "source_ads": float(str(event["channel"]).lower() == "ads"),
                    "new_account": float(event["account_age_minutes"] <= 60),
                    "same_device": float(event["device_users_24h"] > 1),
                    "same_ip": float(event["ip_users_24h"] > 1),
                    "fast_purchase": float(event["time_since_signup_minutes"] <= 5),
                }
            ]
        )
        return float(bundle["model"].predict_proba(features[bundle["features"]])[0, 1])
    source_ads = float(str(event["channel"]).lower() == "ads")
    new_account = float(event["account_age_minutes"] <= 60)
    same_device = float(event["device_users_24h"] > 1)
    same_ip = float(event["ip_users_24h"] > 1)
    fast_purchase = float(event["time_since_signup_minutes"] <= 5)
    return 0.20 * source_ads + 0.30 * new_account + 0.20 * same_device + 0.15 * same_ip + 0.15 * fast_purchase


def _decision(score: float, threshold: float) -> str:
    return "block_or_step_up" if score >= threshold else "pass"


def _system_effect(action: str) -> str:
    if action == "block_or_step_up":
        return "Order is not captured. Payment is held, promo is not consumed, and customer must pass strong authentication or manual review."
    if action == "manual_review":
        return "Order is paused for security review before capture."
    return "Order is approved and payment capture can continue."


if __name__ == "__main__":
    main()
