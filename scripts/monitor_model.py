from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.datasets import load_events
from fraud_risk.infrastructure.monitoring import ModelMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor model performance and data drift.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-source", choices=["fdb-local", "csv"], default="fdb-local")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring_report.json"))
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    events = load_events(args.input_csv, PhishingPolicy(), args.data_source, args.data_dir, args.max_rows)
    report = ModelMonitor().evaluate(args.model_dir, events, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
