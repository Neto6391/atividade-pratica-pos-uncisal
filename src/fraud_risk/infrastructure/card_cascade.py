from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class CascadeDecision:
    stage1_score: float
    stage2_score: float | None
    stage1_threshold: float
    stage2_threshold: float
    stage1_passed: bool
    stage2_passed: bool
    action: str


def score_card_cascade(bundle: dict[str, object], row: pd.Series | pd.DataFrame) -> CascadeDecision:
    features = bundle["features"]
    frame = row.to_frame().T if isinstance(row, pd.Series) else row
    stage1_model: Pipeline = bundle["stage1"]["model"]
    stage1_threshold = float(bundle["stage1"]["threshold"])
    stage2_threshold = float(bundle["stage2"]["threshold"])

    stage1_score = float(stage1_model.predict_proba(frame[features])[0, 1])
    stage1_passed = stage1_score >= stage1_threshold

    stage2_score = None
    stage2_passed = False
    if stage1_passed:
        stage2_model: Pipeline = bundle["stage2"]["model"]
        stage2_score = float(stage2_model.predict_proba(frame[features])[0, 1])
        stage2_passed = stage2_score >= stage2_threshold

    action = "block_or_step_up" if stage1_passed and stage2_passed else ("manual_review" if stage1_passed else "approve")
    return CascadeDecision(
        stage1_score=stage1_score,
        stage2_score=stage2_score,
        stage1_threshold=stage1_threshold,
        stage2_threshold=stage2_threshold,
        stage1_passed=stage1_passed,
        stage2_passed=stage2_passed,
        action=action,
    )


def predict_card_cascade(bundle: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    features = bundle["features"]
    stage1_scores = bundle["stage1"]["model"].predict_proba(frame[features])[:, 1]
    stage1_pass = stage1_scores >= float(bundle["stage1"]["threshold"])
    predictions = np.zeros(len(frame), dtype=int)
    if not stage1_pass.any():
        return predictions

    flagged = frame.loc[stage1_pass, features]
    stage2_scores = bundle["stage2"]["model"].predict_proba(flagged)[:, 1]
    stage2_pass = stage2_scores >= float(bundle["stage2"]["threshold"])
    predictions[np.where(stage1_pass)[0][stage2_pass]] = 1
    return predictions
