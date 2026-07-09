from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
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
from sklearn.utils.class_weight import compute_sample_weight

from fraud_risk.infrastructure.score_separation import compute_separation_metrics

RANDOM_STATE = 42


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model: Pipeline
    threshold: float
    validation_recall: float
    validation_precision: float
    validation_roc_auc: float
    validation_separation_gap: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate card misuse detection on FDB ccfraud source.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-recall", type=float, default=1.0, help="Minimum recall target for threshold calibration on validation.")
    parser.add_argument(
        "--optimize-separation",
        action="store_true",
        help="Select the candidate with the highest validation separation gap (closest to perfect separation).",
    )
    args = parser.parse_args()

    df = _load(args.data_dir / "creditcardfraud.zip")
    train, valid, test = _temporal_split(df)
    features = [col for col in df.columns if col != "Class"]

    best = _select_best_candidate(train, valid, features, args.min_recall, args.optimize_separation)
    test_scores = best.model.predict_proba(test[features])[:, 1]
    metrics = _metrics(test["Class"], test_scores, best.threshold)
    missed_frauds = _missed_fraud_diagnostics(test, test_scores, best.threshold)
    separation = compute_separation_metrics(test["Class"], test_scores)

    report = {
        "source": "amazon_fdb_source:ccfraud",
        "rows": int(len(df)),
        "fraud_rate": round(float(df["Class"].mean()), 6),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "features": features,
        "model": best.name,
        "min_recall": round(float(args.min_recall), 6),
        "validation_recall": round(float(best.validation_recall), 6),
        "validation_precision": round(float(best.validation_precision), 6),
        "validation_roc_auc": round(float(best.validation_roc_auc), 6),
        "validation_separation_gap": round(float(best.validation_separation_gap), 6),
        "optimize_separation": bool(args.optimize_separation),
        **metrics,
        **separation,
        "missed_frauds_in_test": missed_frauds,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best.model, "threshold": best.threshold, "features": features, "model_name": best.name},
        args.output_dir / "card_misuse_model.joblib",
    )
    (args.output_dir / "card_misuse_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plots(args.output_dir / "charts", test["Class"], test_scores, best.threshold)
    print(json.dumps(report, indent=2))

    if metrics["confusion_matrix"]["fn"] > 0:
        print(
            f"WARNING: test recall is {metrics['recall']}; {metrics['confusion_matrix']['fn']} fraud(s) still missed.",
            file=sys.stderr,
        )


def _select_best_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    min_recall: float,
    optimize_separation: bool,
) -> CandidateResult:
    results: list[CandidateResult] = []
    for name, model in _candidates().items():
        _fit_model(model, train[features], train["Class"], name)
        valid_scores = model.predict_proba(valid[features])[:, 1]
        threshold = _select_threshold(valid["Class"], valid_scores, min_recall)
        valid_predictions = (valid_scores >= threshold).astype(int)
        valid_sep = compute_separation_metrics(valid["Class"], valid_scores)
        results.append(
            CandidateResult(
                name=name,
                model=model,
                threshold=threshold,
                validation_recall=float(recall_score(valid["Class"], valid_predictions, zero_division=0)),
                validation_precision=float(precision_score(valid["Class"], valid_predictions, zero_division=0)),
                validation_roc_auc=float(roc_auc_score(valid["Class"], valid_scores)),
                validation_separation_gap=float(valid_sep["separation_gap"] or -999.0),
            )
        )

    eligible = [item for item in results if item.validation_recall >= min_recall - 1e-9] or results
    if optimize_separation:
        eligible.sort(
            key=lambda item: (item.validation_separation_gap, item.validation_precision, item.validation_roc_auc),
            reverse=True,
        )
    elif min_recall >= 1.0:
        eligible.sort(
            key=lambda item: (item.validation_recall, item.validation_precision, item.validation_roc_auc),
            reverse=True,
        )
    else:
        eligible.sort(key=lambda item: (item.validation_recall, item.validation_roc_auc), reverse=True)
    return eligible[0]


def _base_prep() -> list[tuple[str, object]]:
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]


def _candidates() -> dict[str, Pipeline]:
    return {
        "logistic_balanced": Pipeline(
            _base_prep()
            + [("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]
        ),
        "calibrated_logistic": Pipeline(
            _base_prep()
            + [
                (
                    "model",
                    CalibratedClassifierCV(
                        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                        method="isotonic",
                        cv=3,
                    ),
                )
            ]
        ),
        "random_forest_balanced": Pipeline(
            _base_prep()
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            _base_prep()
            + [
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        learning_rate=0.05,
                        max_depth=None,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "stacking_ensemble": Pipeline(
            _base_prep()
            + [
                (
                    "model",
                    StackingClassifier(
                        estimators=[
                            (
                                "logistic",
                                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                            ),
                            (
                                "rf",
                                RandomForestClassifier(
                                    n_estimators=200,
                                    min_samples_leaf=10,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1,
                                    random_state=RANDOM_STATE,
                                ),
                            ),
                            (
                                "hgb",
                                HistGradientBoostingClassifier(
                                    max_iter=200,
                                    learning_rate=0.05,
                                    max_depth=8,
                                    random_state=RANDOM_STATE,
                                ),
                            ),
                        ],
                        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                        cv=3,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "extra_trees_balanced": Pipeline(
            _base_prep()
            + [
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def _fit_model(model: Pipeline, features: pd.DataFrame, labels: pd.Series, name: str) -> None:
    if name == "hist_gradient_boosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=labels)
        model.fit(features, labels, model__sample_weight=sample_weight)
        return
    model.fit(features, labels)


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


def _select_threshold(y_true: pd.Series, scores: np.ndarray, min_recall: float = 0.95) -> float:
    if min_recall >= 1.0:
        fraud_scores = scores[y_true.to_numpy() == 1]
        if len(fraud_scores) == 0:
            return 0.5
        return float(np.min(fraud_scores) * (1.0 - 1e-9))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    candidates = np.where(recall[:-1] >= min_recall)[0]
    best = candidates[np.argmax(f1_values[candidates])] if len(candidates) else int(np.argmax(f1_values))
    return float(thresholds[best])


def _metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 6),
        "average_precision_pr_auc": round(float(average_precision_score(y_true, scores)), 6),
        "false_positive_rate": round(fp / max(tn + fp, 1), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _missed_fraud_diagnostics(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> list[dict[str, object]]:
    fraud_mask = test["Class"].eq(1).to_numpy()
    missed = np.where(fraud_mask & (scores < threshold))[0]
    diagnostics: list[dict[str, object]] = []
    for index in missed:
        row = test.iloc[index]
        diagnostics.append(
            {
                "test_row_index": int(index),
                "time": float(row["Time"]),
                "amount": float(row["Amount"]),
                "risk_score": round(float(scores[index]), 6),
                "threshold": round(float(threshold), 6),
            }
        )
    return diagnostics


def _plots(charts_dir: Path, y_true: pd.Series, scores: np.ndarray, threshold: float) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    predictions = (scores >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, predictions, labels=[0, 1]), display_labels=["Legitima", "Fraude"]).plot(ax=ax, colorbar=False)
    ax.set_title("Cartao: matriz de confusao")
    fig.tight_layout()
    fig.savefig(charts_dir / "card_misuse_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax)
    ax.axvline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_title("Cartao: curva Precision-Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "card_misuse_precision_recall.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
