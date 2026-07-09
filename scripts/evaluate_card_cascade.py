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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from fraud_risk.infrastructure.card_cascade import predict_card_cascade
from fraud_risk.infrastructure.score_separation import compute_separation_metrics

RANDOM_STATE = 42
COST_FN = 10.0
COST_FP = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a two-stage card fraud cascade.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--stage1-min-recall", type=float, default=0.98)
    parser.add_argument("--stage2-min-recall", type=float, default=0.90)
    args = parser.parse_args()

    df = _load(args.data_dir / "creditcardfraud.zip")
    train, valid, test = _temporal_split(df)
    features = [col for col in df.columns if col != "Class"]

    stage1 = _build_stage1()
    stage1.fit(train[features], train["Class"])
    stage1_scores_valid = stage1.predict_proba(valid[features])[:, 1]
    stage1_threshold = _threshold_for_recall(valid["Class"], stage1_scores_valid, args.stage1_min_recall)

    stage2 = _build_stage2()
    stage2.fit(train[features], train["Class"])
    stage2_threshold = _stage2_threshold(
        valid,
        features,
        stage1_scores_valid,
        stage1_threshold,
        stage2,
        args.stage2_min_recall,
    )

    bundle = {
        "architecture": "two_stage_cascade",
        "features": features,
        "stage1": {"name": "calibrated_logistic_gatekeeper", "model": stage1, "threshold": stage1_threshold},
        "stage2": {"name": "random_forest_specialist", "model": stage2, "threshold": stage2_threshold},
        "policy": {
            "stage1_role": "high_recall_gatekeeper",
            "stage2_role": "high_precision_specialist_on_flagged_only",
            "final_decision_rule": "block_or_step_up when stage1 and stage2 both fire; manual_review when only stage1 fires",
        },
        "cost_model": {"cost_false_negative": COST_FN, "cost_false_positive": COST_FP},
    }

    single_stage_metrics = _load_single_stage_metrics(args.output_dir)
    cascade_metrics = _evaluate_cascade(bundle, valid, test)
    report = {
        "source": "amazon_fdb_source:ccfraud",
        "architecture": "two_stage_cascade",
        "rows": int(len(df)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "stage1_min_recall_target": args.stage1_min_recall,
        "stage2_min_recall_target": args.stage2_min_recall,
        "stage1": {
            "model": bundle["stage1"]["name"],
            "threshold": round(stage1_threshold, 6),
            **cascade_metrics["stage1_only"],
        },
        "stage2": {
            "model": bundle["stage2"]["name"],
            "threshold": round(stage2_threshold, 6),
            "applied_only_to_stage1_flagged": True,
        },
        "cascade_test": cascade_metrics["cascade_test"],
        "comparison_single_stage": single_stage_metrics,
        "improvement_vs_single_stage": _improvement(single_stage_metrics, cascade_metrics["cascade_test"]),
        "literature_basis": [
            "ACM 2025: calibrated ensemble + cost-optimal threshold",
            "Two-stage cascade: high-recall gatekeeper + precision specialist",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_dir / "card_cascade_model.joblib")
    (args.output_dir / "card_cascade_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(args.output_dir / "charts", test, bundle, report)
    print(json.dumps(report, indent=2))


def _build_stage1() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
                    method="isotonic",
                    cv=3,
                ),
            ),
        ]
    )


def _build_stage2() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _threshold_for_recall(labels: pd.Series, scores: np.ndarray, min_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    candidates = np.where(recall[:-1] >= min_recall)[0]
    if len(candidates):
        f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best = candidates[np.argmax(f1_values[candidates])]
        return float(thresholds[best])
    return float(thresholds[int(np.argmax(2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)))])


def _stage2_threshold(
    valid: pd.DataFrame,
    features: list[str],
    stage1_scores: np.ndarray,
    stage1_threshold: float,
    stage2: Pipeline,
    min_recall: float,
) -> float:
    flagged = stage1_scores >= stage1_threshold
    if not flagged.any():
        return 0.5
    subset = valid.loc[flagged]
    stage2_scores = stage2.predict_proba(subset[features])[:, 1]
    cost_threshold = _cost_optimal_threshold(subset["Class"], stage2_scores)
    recall_threshold = _threshold_for_recall(subset["Class"], stage2_scores, min_recall)
    cost = _expected_cost(subset["Class"], stage2_scores, cost_threshold)
    recall_cost = _expected_cost(subset["Class"], stage2_scores, recall_threshold)
    return recall_threshold if recall_cost <= cost * 1.05 else cost_threshold


def _cost_optimal_threshold(labels: pd.Series, scores: np.ndarray) -> float:
    thresholds = np.unique(np.round(scores, 6))
    if len(thresholds) < 2:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_cost = 0.5, float("inf")
    labels_arr = labels.to_numpy()
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        fn = int(((predictions == 0) & (labels_arr == 1)).sum())
        fp = int(((predictions == 1) & (labels_arr == 0)).sum())
        cost = COST_FN * fn + COST_FP * fp
        if cost < best_cost:
            best_threshold, best_cost = float(threshold), cost
    return best_threshold


def _expected_cost(labels: pd.Series, scores: np.ndarray, threshold: float) -> float:
    predictions = (scores >= threshold).astype(int)
    labels_arr = labels.to_numpy()
    fn = int(((predictions == 0) & (labels_arr == 1)).sum())
    fp = int(((predictions == 1) & (labels_arr == 0)).sum())
    return COST_FN * fn + COST_FP * fp


def _evaluate_cascade(bundle: dict[str, object], valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    features = bundle["features"]
    stage1_scores = bundle["stage1"]["model"].predict_proba(valid[features])[:, 1]
    stage1_only = _metrics_from_scores(valid["Class"], stage1_scores, float(bundle["stage1"]["threshold"]))
    cascade_valid_preds = predict_card_cascade(bundle, valid[features])
    cascade_test_preds = predict_card_cascade(bundle, test[features])
    cascade_test_scores = _cascade_scores(bundle, test[features])
    stage1_test_scores = bundle["stage1"]["model"].predict_proba(test[features])[:, 1]
    stage1_test_pass = stage1_test_scores >= float(bundle["stage1"]["threshold"])
    labels = test["Class"].astype(int).to_numpy()
    return {
        "stage1_only": stage1_only,
        "cascade_valid": _metrics_from_predictions(valid["Class"], cascade_valid_preds),
        "cascade_test": {
            **_metrics_from_predictions(test["Class"], cascade_test_preds),
            **_cascade_operational_metrics(test, bundle, stage1_test_pass),
            "stage1_test_recall": round(float(recall_score(labels, stage1_test_pass.astype(int), zero_division=0)), 6),
            "recall_with_review_or_block": round(
                float(((labels == 1) & stage1_test_pass).sum() / max(labels.sum(), 1)),
                6,
            ),
            **compute_separation_metrics(test["Class"], cascade_test_scores),
        },
    }


def _cascade_scores(bundle: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    features = bundle["features"]
    stage1_scores = bundle["stage1"]["model"].predict_proba(frame[features])[:, 1]
    stage2_model = bundle["stage2"]["model"]
    stage2_scores = stage2_model.predict_proba(frame[features])[:, 1]
    stage1_pass = stage1_scores >= float(bundle["stage1"]["threshold"])
    combined = np.zeros(len(frame), dtype=float)
    combined[stage1_pass] = stage2_scores[stage1_pass]
    return combined


def _cascade_operational_metrics(test: pd.DataFrame, bundle: dict[str, object], stage1_pass: np.ndarray) -> dict[str, object]:
    features = bundle["features"]
    stage2_scores = bundle["stage2"]["model"].predict_proba(test[features])[:, 1]
    stage2_pass = stage2_scores >= float(bundle["stage2"]["threshold"])
    return {
        "stage1_flagged_count": int(stage1_pass.sum()),
        "stage1_flagged_rate": round(float(stage1_pass.mean()), 6),
        "final_blocked_count": int((stage1_pass & stage2_pass).sum()),
        "manual_review_count": int((stage1_pass & ~stage2_pass).sum()),
        "approved_without_review_count": int((~stage1_pass).sum()),
    }


def _metrics_from_scores(labels: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (scores >= threshold).astype(int)
    return _metrics_from_predictions(labels, predictions, scores=scores, threshold=threshold)


def _metrics_from_predictions(
    labels: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "threshold": round(float(threshold), 6) if threshold is not None else None,
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "false_positive_rate": round(fp / max(tn + fp, 1), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if scores is not None:
        result["roc_auc"] = round(float(roc_auc_score(labels, scores)), 6)
        result["average_precision_pr_auc"] = round(float(average_precision_score(labels, scores)), 6)
    return result


def _improvement(single: dict[str, object] | None, cascade: dict[str, object]) -> dict[str, object]:
    if not single:
        return {"note": "single-stage metrics unavailable"}
    return {
        "recall_delta": round(float(cascade["recall"]) - float(single.get("recall", 0)), 6),
        "precision_delta": round(float(cascade["precision"]) - float(single.get("precision", 0)), 6),
        "false_positive_delta": int(cascade["confusion_matrix"]["fp"] - single["confusion_matrix"]["fp"]),
        "false_negative_delta": int(cascade["confusion_matrix"]["fn"] - single["confusion_matrix"]["fn"]),
    }


def _load_single_stage_metrics(output_dir: Path) -> dict[str, object] | None:
    path = output_dir / "card_misuse_metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": payload.get("model"),
        "recall": payload.get("recall"),
        "precision": payload.get("precision"),
        "false_positive_rate": payload.get("false_positive_rate"),
        "confusion_matrix": payload.get("confusion_matrix"),
    }


def _plot(charts_dir: Path, test: pd.DataFrame, bundle: dict[str, object], report: dict[str, object]) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    features = bundle["features"]
    predictions = predict_card_cascade(bundle, test[features])
    stage1_scores = bundle["stage1"]["model"].predict_proba(test[features])[:, 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(test["Class"], predictions, labels=[0, 1]), display_labels=["Legitima", "Fraude"]).plot(
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Cartao cascata: matriz de confusao (bloqueio final)")
    fig.tight_layout()
    fig.savefig(charts_dir / "card_cascade_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(test["Class"], stage1_scores, ax=ax, name="Estagio 1")
    ax.axvline(float(bundle["stage1"]["threshold"]), color="tab:red", linestyle="--", label="threshold estagio 1")
    ax.set_title("Cartao cascata: curva Precision-Recall (estagio 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "card_cascade_precision_recall.png", dpi=160)
    plt.close(fig)

    cascade = report["cascade_test"]
    single = report.get("comparison_single_stage") or {}
    labels = ["Recall bloqueio", "Precision bloqueio", "Falsos positivos (escala log)"]
    cascade_values = [cascade["recall"], cascade["precision"], cascade["confusion_matrix"]["fp"]]
    single_values = [single.get("recall", 0), single.get("precision", 0), single.get("confusion_matrix", {}).get("fp", 1)]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, single_values, width, label="Single-stage")
    ax.bar(x + width / 2, cascade_values, width, label="Cascata")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Cartao: cascata vs single-stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "card_cascade_vs_single_stage.png", dpi=160)
    plt.close(fig)

    operational_labels = ["Aprovadas", "Revisao manual", "Bloqueio final"]
    operational_values = [
        cascade["approved_without_review_count"],
        cascade["manual_review_count"],
        cascade["final_blocked_count"],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(operational_labels, operational_values, color=["tab:green", "tab:orange", "tab:red"])
    ax.set_title("Cartao cascata: decisoes operacionais no teste")
    ax.set_ylabel("Transacoes")
    fig.tight_layout()
    fig.savefig(charts_dir / "card_cascade_operational_flow.png", dpi=160)
    plt.close(fig)


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
