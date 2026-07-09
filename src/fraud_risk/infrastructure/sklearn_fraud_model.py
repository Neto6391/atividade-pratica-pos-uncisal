from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from fraud_risk.domain.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET
from fraud_risk.domain.metrics import ClassificationMetrics
from fraud_risk.infrastructure.datasets import RANDOM_SEED, SplitData
from fraud_risk.infrastructure.score_separation import compute_separation_metrics


@dataclass(frozen=True)
class SelectedModel:
    name: str
    model: Pipeline
    threshold: float
    metrics_by_model: dict[str, ClassificationMetrics]
    y_test: pd.Series
    test_scores: np.ndarray


class SklearnFraudModel:
    def train_select_and_evaluate(self, split: SplitData) -> SelectedModel:
        best_name, best_rank, best_threshold, best_model = "", (-1.0, -1.0), 0.5, None
        metrics_by_model: dict[str, ClassificationMetrics] = {}

        for name, model in self._candidates().items():
            self._fit_candidate(model, name, split.train[MODEL_FEATURES], split.train[TARGET])
            threshold = self._select_threshold(split.valid[TARGET], model.predict_proba(split.valid[MODEL_FEATURES])[:, 1])
            metrics = self._evaluate(split.test[TARGET], model.predict_proba(split.test[MODEL_FEATURES])[:, 1], threshold)
            metrics_by_model[name] = metrics
            rank = (metrics.f1, metrics.pr_auc)
            if rank > best_rank:
                best_name, best_rank, best_threshold, best_model = name, rank, threshold, model

        if best_model is None:
            raise RuntimeError("No model was trained.")
        test_scores = best_model.predict_proba(split.test[MODEL_FEATURES])[:, 1]
        return SelectedModel(best_name, best_model, best_threshold, metrics_by_model, split.test[TARGET], test_scores)

    @staticmethod
    def _candidates() -> dict[str, Pipeline]:
        return {
            "logistic_balanced": Pipeline([
                ("prep", _preprocessor()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
            ]),
            "random_forest_balanced": Pipeline([
                ("prep", _preprocessor()),
                ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=30, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED)),
            ]),
            "hist_gradient_boosting": Pipeline([
                ("prep", _preprocessor()),
                ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=8, random_state=RANDOM_SEED)),
            ]),
        }

    @staticmethod
    def _fit_candidate(model: Pipeline, name: str, features: pd.DataFrame, labels: pd.Series) -> None:
        if name == "hist_gradient_boosting":
            sample_weight = compute_sample_weight(class_weight="balanced", y=labels)
            model.fit(features, labels, model__sample_weight=sample_weight)
            return
        model.fit(features, labels)

    @staticmethod
    def _select_threshold(y_true: pd.Series, scores: np.ndarray, min_recall: float = 0.80) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        candidates = np.where(recall[:-1] >= min_recall)[0]
        best = candidates[np.argmax(f1_values[candidates])] if len(candidates) else int(np.argmax(f1_values))
        return float(thresholds[best])

    @staticmethod
    def _evaluate(y_true: pd.Series, scores: np.ndarray, threshold: float) -> ClassificationMetrics:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        separation = compute_separation_metrics(y_true, scores)
        return ClassificationMetrics(
            threshold=threshold,
            precision=float(precision_score(y_true, predictions, zero_division=0)),
            recall=float(recall_score(y_true, predictions, zero_division=0)),
            f1=float(f1_score(y_true, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, scores)),
            pr_auc=float(average_precision_score(y_true, scores)),
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            true_positive=int(tp),
            false_positive_rate=fp / max(tn + fp, 1),
            separation_gap=float(separation["separation_gap"] or 0.0),
            perfect_separation_possible=bool(separation["perfect_separation_possible"]),
            min_false_positives_at_recall_1_0=int(separation["min_false_positives_at_recall_1_0"] or 0),
            best_precision_at_recall_1_0=float(separation["best_precision_at_recall_1_0"] or 0.0),
        )


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), CATEGORICAL_FEATURES),
    ])
