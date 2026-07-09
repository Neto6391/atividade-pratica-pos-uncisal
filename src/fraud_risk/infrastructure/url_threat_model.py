from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class UrlThreatEvaluation:
    labels: pd.Series
    scores: pd.Series


@dataclass(frozen=True)
class UrlThreatModel:
    model: Pipeline
    evaluation: UrlThreatEvaluation

    def score(self, urls: pd.Series) -> pd.Series:
        return pd.Series(self.model.predict_proba(urls.astype(str))[:, 1], index=urls.index)


def train_url_threat_model(raw_urls: pd.DataFrame, sample_per_class: int = 30000) -> UrlThreatModel:
    phishing = raw_urls[raw_urls["type"].eq("phishing")].sample(n=sample_per_class, random_state=42)
    benign = raw_urls[raw_urls["type"].eq("benign")].sample(n=sample_per_class, random_state=42)
    dataset = pd.concat([phishing, benign]).sample(frac=1, random_state=42).reset_index(drop=True)
    labels = dataset["type"].eq("phishing").astype(int)
    train_urls, test_urls, y_train, y_test = train_test_split(
        dataset["url"].astype(str),
        labels,
        test_size=0.25,
        stratify=labels,
        random_state=42,
    )
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=60000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=500, class_weight="balanced"),
            ),
        ]
    )
    model.fit(train_urls, y_train)
    scores = pd.Series(model.predict_proba(test_urls)[:, 1], index=y_test.index)
    return UrlThreatModel(
        model=model,
        evaluation=UrlThreatEvaluation(labels=y_test.reset_index(drop=True), scores=scores.reset_index(drop=True)),
    )
