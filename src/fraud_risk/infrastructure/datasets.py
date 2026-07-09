from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_risk.domain.features import PHISHING_TARGET, TARGET, TIME_COL
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.fdb_local_loader import load_fdb_local_events

RANDOM_SEED = 42


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_events(
    path: Path | None,
    phishing: PhishingPolicy,
    data_source: str = "fdb-local",
    data_dir: Path = Path("data/raw"),
    max_rows: int | None = None,
) -> pd.DataFrame:
    if data_source == "fdb-local":
        return load_fdb_local_events(data_dir, phishing, max_rows=max_rows)
    if data_source != "csv":
        raise ValueError(f"Unsupported data_source: {data_source}")

    if path is None:
        raise ValueError("--input-csv is required when --data-source csv is used.")
    events = pd.read_csv(path)
    missing = {TARGET, TIME_COL} - set(events.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {sorted(missing)}")
    if "landing_url" in events and "url_phishing_score" not in events:
        events["url_phishing_score"] = events["landing_url"].map(phishing.score)
    if PHISHING_TARGET not in events:
        events[PHISHING_TARGET] = (events["url_phishing_score"] >= 0.60).astype(int)
    events.attrs["dataset_source"] = str(path)
    return events


def temporal_split(events: pd.DataFrame) -> SplitData:
    ordered = events.sort_values(TIME_COL).reset_index(drop=True)
    train_end = math.floor(len(ordered) * 0.70)
    valid_end = math.floor(len(ordered) * 0.85)
    return SplitData(ordered.iloc[:train_end].copy(), ordered.iloc[train_end:valid_end].copy(), ordered.iloc[valid_end:].copy())
