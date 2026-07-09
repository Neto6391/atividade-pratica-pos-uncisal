from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all fraud components and build a trust summary.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)

    commands = [
        [str(python), str(root / "src" / "fraud_detection_pipeline.py"), "--output-dir", str(args.output_dir)],
        [str(python), str(root / "scripts" / "evaluate_card_misuse.py"), "--output-dir", str(args.output_dir), "--min-recall", "1.0"],
        [str(python), str(root / "scripts" / "evaluate_card_cascade.py"), "--output-dir", str(args.output_dir)],
        [str(python), str(root / "scripts" / "evaluate_promo_abuse.py"), "--output-dir", str(args.output_dir), "--min-recall", "0.80"],
        [str(python), str(root / "scripts" / "analyze_score_separation.py"), "--data-dir", str(args.data_dir), "--model-dir", str(args.output_dir), "--output", str(args.output_dir / "system_trust_report.json")],
    ]

    for command in commands:
        print(f"Running: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=root)

    summary = _build_summary(args.output_dir)
    summary_path = args.output_dir / "system_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _build_summary(output_dir: Path) -> dict[str, object]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    card = json.loads((output_dir / "card_misuse_metrics.json").read_text(encoding="utf-8"))
    cascade_path = output_dir / "card_cascade_metrics.json"
    cascade = json.loads(cascade_path.read_text(encoding="utf-8")) if cascade_path.exists() else None
    promo = json.loads((output_dir / "promo_abuse_metrics.json").read_text(encoding="utf-8"))
    trust = json.loads((output_dir / "system_trust_report.json").read_text(encoding="utf-8"))
    selected = metrics["models"][metrics["selected_model"]]
    phishing = metrics["anti_phishing_filter"]
    return {
        "selected_ecommerce_model": metrics["selected_model"],
        "components": [
            {
                "component": "ecommerce",
                "recall": selected["recall"],
                "precision": selected["precision"],
                "false_positive_rate": selected.get("false_positive_rate"),
                "perfect_separation_possible": selected.get("perfect_separation_possible"),
                "min_false_positives_at_recall_1_0": selected.get("min_false_positives_at_recall_1_0"),
            },
            {
                "component": "card_cascade",
                "recall": cascade["cascade_test"]["recall"] if cascade else None,
                "precision": cascade["cascade_test"]["precision"] if cascade else None,
                "false_positive_rate": cascade["cascade_test"]["false_positive_rate"] if cascade else None,
                "stage1_flagged_rate": cascade["cascade_test"].get("stage1_flagged_rate") if cascade else None,
            },
            {
                "component": "card_single_stage",
                "model": card["model"],
                "recall": card["recall"],
                "precision": card["precision"],
                "false_positive_rate": card.get("false_positive_rate"),
                "perfect_separation_possible": card.get("perfect_separation_possible"),
                "min_false_positives_at_recall_1_0": card.get("min_false_positives_at_recall_1_0"),
            },
            {
                "component": "phishing",
                "recall": phishing["recall"],
                "precision": phishing["precision"],
                "false_positive_rate": phishing.get("false_positive_rate"),
                "perfect_separation_possible": phishing.get("perfect_separation_possible"),
                "min_false_positives_at_recall_1_0": phishing.get("min_false_positives_at_recall_1_0"),
            },
            {
                "component": "promo_abuse",
                "policy": promo["policy"],
                "recall": promo["recall"],
                "precision": promo["precision"],
                "false_positive_rate": promo.get("false_positive_rate"),
                "perfect_separation_possible": promo.get("perfect_separation_possible"),
                "min_false_positives_at_recall_1_0": promo.get("min_false_positives_at_recall_1_0"),
            },
        ],
        "trust_report": {
            "all_perfect_separation": trust["all_perfect_separation"],
            "any_perfect_separation": trust["any_perfect_separation"],
            "answer": trust["answer"],
        },
    }


if __name__ == "__main__":
    main()
