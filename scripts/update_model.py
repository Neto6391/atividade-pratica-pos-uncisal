from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_risk.application.train_fraud_model import TrainFraudModel
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.datasets import load_events
from fraud_risk.infrastructure.model_repository import ModelRepository
from fraud_risk.infrastructure.monitoring import ModelMonitor
from fraud_risk.infrastructure.phishing_filter_evaluator import PhishingFilterEvaluator
from fraud_risk.infrastructure.report_plots import ReportPlotter
from fraud_risk.infrastructure.sklearn_fraud_model import SklearnFraudModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and update the fraud model when retraining is needed.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-source", choices=["fdb-local", "csv"], default="fdb-local")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    phishing = PhishingPolicy()
    events = load_events(args.input_csv, phishing, args.data_source, args.data_dir, args.max_rows)
    monitor_path = args.model_dir / "monitoring_report.json"
    report = ModelMonitor().evaluate(args.model_dir, events, monitor_path)
    should_retrain = args.force or report["retrain_recommended"]
    update_report = {"monitoring": report, "retrained": False, "model_dir": str(args.model_dir)}

    if should_retrain:
        backup_dir = args.model_dir.with_name(f"{args.model_dir.name}_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        if args.model_dir.exists():
            shutil.copytree(args.model_dir, backup_dir)
            update_report["backup_dir"] = str(backup_dir)
        trainer = TrainFraudModel(SklearnFraudModel(), ModelRepository(), ReportPlotter(), PhishingFilterEvaluator())
        result = trainer.execute(events, args.model_dir)
        update_report["retrained"] = True
        update_report["new_metrics"] = result.to_dict()

    update_path = args.model_dir / "update_report.json"
    update_path.write_text(json.dumps(update_report, indent=2), encoding="utf-8")
    print(json.dumps(update_report, indent=2))


if __name__ == "__main__":
    main()
