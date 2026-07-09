from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_risk.application.train_fraud_model import TrainFraudModel
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.datasets import load_events
from fraud_risk.infrastructure.model_repository import ModelRepository
from fraud_risk.infrastructure.phishing_filter_evaluator import PhishingFilterEvaluator
from fraud_risk.infrastructure.report_plots import ReportPlotter
from fraud_risk.infrastructure.sklearn_fraud_model import SklearnFraudModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud and phishing-aware risk detection pipeline.")
    parser.add_argument("--data-source", choices=["fdb-local", "csv"], default="fdb-local")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    phishing = PhishingPolicy()
    events = load_events(args.input_csv, phishing, args.data_source, args.data_dir, args.max_rows)
    use_case = TrainFraudModel(SklearnFraudModel(), ModelRepository(), ReportPlotter(), PhishingFilterEvaluator())
    print(json.dumps(use_case.execute(events, args.output_dir).to_dict(), indent=2))


if __name__ == "__main__":
    main()
