from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.utils.class_weight import compute_sample_weight

from fraud_risk.infrastructure.score_separation import compute_separation_metrics

RANDOM_STATE = 42


def main() -> None:
    with ZipFile("data/raw/creditcardfraud.zip") as archive:
        df = pd.read_csv(archive.open("creditcard.csv")).sort_values("Time").reset_index(drop=True)
    train_end = int(len(df) * 0.70)
    valid_end = int(len(df) * 0.85)
    train, test = df.iloc[:train_end].copy(), df.iloc[valid_end:].copy()
    features = [col for col in df.columns if col != "Class"]
    prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())])
    x_train = prep.fit_transform(train[features])
    x_test = prep.transform(test[features])
    y_train, y_test = train["Class"], test["Class"]
    legit = x_train[y_train.to_numpy() == 0]

    candidates: list[tuple[str, object]] = [
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("logistic_tight", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.01, random_state=RANDOM_STATE)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=500, max_depth=None, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ("rf_deep", RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ("hgb", HistGradientBoostingClassifier(max_iter=500, max_depth=12, learning_rate=0.03, random_state=RANDOM_STATE)),
    ]

    best_name, best_gap, best_metrics = "", -999.0, {}
    for name, model in candidates:
        if name == "hgb":
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(x_train, y_train, sample_weight=weights)
        else:
            model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        metrics = compute_separation_metrics(y_test, scores)
        gap = float(metrics["separation_gap"] or -999)
        print(f"{name:20s} gap={gap:9.4f} perfect={metrics['perfect_separation_possible']} min_fp={metrics['min_false_positives_at_recall_1_0']}")
        if gap > best_gap:
            best_name, best_gap, best_metrics = name, gap, metrics

    for name, model in [
        ("iso_forest", IsolationForest(n_estimators=500, contamination=0.002, random_state=RANDOM_STATE, n_jobs=-1)),
        ("one_class_svm", OneClassSVM(kernel="rbf", gamma="scale", nu=0.002)),
    ]:
        model.fit(legit)
        raw = -model.score_samples(x_test) if hasattr(model, "score_samples") else -model.decision_function(x_test)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        metrics = compute_separation_metrics(y_test, scores)
        gap = float(metrics["separation_gap"] or -999)
        print(f"{name:20s} gap={gap:9.4f} perfect={metrics['perfect_separation_possible']} min_fp={metrics['min_false_positives_at_recall_1_0']}")
        if gap > best_gap:
            best_name, best_gap, best_metrics = name, gap, metrics

    print(f"BEST: {best_name} gap={best_gap} metrics={best_metrics}")


if __name__ == "__main__":
    main()
