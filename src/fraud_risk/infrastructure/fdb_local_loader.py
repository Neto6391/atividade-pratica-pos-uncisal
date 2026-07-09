from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from fraud_risk.domain.features import PHISHING_TARGET, TARGET, TIME_COL
from fraud_risk.domain.phishing import PhishingPolicy
from fraud_risk.infrastructure.url_threat_model import train_url_threat_model


FDB_SOURCE = "amazon_fdb_sources:fraudecom+malurl"


def load_fdb_local_events(data_dir: Path, phishing: PhishingPolicy, max_rows: int | None = None) -> pd.DataFrame:
    ecommerce = _read_zip_csv(data_dir / "fraud-ecommerce.zip", "Fraud_Data.csv")
    urls = _read_zip_csv(data_dir / "malicious-urls-dataset.zip", "malicious_phish.csv")
    if max_rows:
        ecommerce = ecommerce.sort_values("purchase_time").head(max_rows).copy()

    events = _map_fraudecom(ecommerce)
    events = _attach_malurl(events, urls)
    events.attrs["dataset_source"] = FDB_SOURCE
    return events


def _read_zip_csv(zip_path: Path, member: str) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Run: python scripts/download_fdb_sources.py"
        )
    with ZipFile(zip_path) as archive:
        with archive.open(member) as csv_file:
            return pd.read_csv(csv_file)


def _map_fraudecom(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df = df.sort_values("purchase_time").reset_index(drop=True)
    df["ip_key"] = df["ip_address"].round(0).astype("int64").astype(str)
    df["hour_bucket"] = df["purchase_time"].dt.floor("h")
    df["day_bucket"] = df["purchase_time"].dt.floor("D")
    signup_delta = (df["purchase_time"] - df["signup_time"]).dt.total_seconds().clip(lower=0) / 60
    device_hour_count = _bucket_count(df, "device_id", "hour_bucket")
    device_day_count = _bucket_count(df, "device_id", "day_bucket")
    ip_day_count = _bucket_count(df, "ip_key", "day_bucket")
    user_day_count = _bucket_count(df, "user_id", "day_bucket")

    events = pd.DataFrame(
        {
            TIME_COL: df["purchase_time"],
            TARGET: df["class"].astype(int),
            "amount": df["purchase_value"].astype(float),
            "account_age_minutes": signup_delta,
            "time_since_signup_minutes": signup_delta,
            "transactions_1h": device_hour_count,
            "transactions_24h": device_day_count,
            "unique_cards_24h": 1,
            "promo_uses_24h": np.where(df["source"].astype(str).str.lower().eq("ads"), user_day_count, 0),
            "device_users_24h": device_day_count,
            "ip_users_24h": ip_day_count,
            "payment_bin_risk": 0.0,
            "chargebacks_90d": 0,
            "country_mismatch": "0",
            "new_device": (~df["device_id"].duplicated()).astype(int).astype(str),
            "proxy_or_vpn": "0",
            "browser": df["browser"].astype(str).str.lower(),
            "channel": df["source"].astype(str).str.lower(),
            "payment_method": "credit_card",
        }
    )
    return events


def _attach_malurl(events: pd.DataFrame, raw_urls: pd.DataFrame) -> pd.DataFrame:
    urls = raw_urls[["url", "type"]].dropna().copy()
    url_model = train_url_threat_model(urls)
    phishing_urls = urls[urls["type"].eq("phishing")]["url"].reset_index(drop=True)
    benign_urls = urls[urls["type"].eq("benign")]["url"].reset_index(drop=True)
    if phishing_urls.empty or benign_urls.empty:
        raise ValueError("malurl source must contain both phishing and benign URLs.")

    enriched = events.copy()
    fraud_positions = np.flatnonzero(enriched[TARGET].to_numpy() == 1)
    phishing_positions = fraud_positions[::3]
    enriched[PHISHING_TARGET] = 0
    enriched["landing_url"] = _repeat_to_length(benign_urls, len(enriched))
    enriched.loc[phishing_positions, PHISHING_TARGET] = 1
    enriched.loc[phishing_positions, "landing_url"] = _repeat_to_length(phishing_urls, len(phishing_positions))
    enriched["url_phishing_score"] = url_model.score(enriched["landing_url"])
    enriched.attrs["url_threat_model"] = url_model.model
    enriched.attrs["anti_phishing_eval_labels"] = url_model.evaluation.labels
    enriched.attrs["anti_phishing_eval_scores"] = url_model.evaluation.scores
    enriched.attrs["anti_phishing_source"] = "amazon_fdb_source:malurl"
    return enriched


def _bucket_count(df: pd.DataFrame, group_col: str, bucket_col: str) -> pd.Series:
    return df.groupby([group_col, bucket_col], sort=False).cumcount().add(1).astype("int64")


def _repeat_to_length(values: pd.Series, length: int) -> np.ndarray:
    if length == 0:
        return np.array([], dtype=object)
    repeats = int(np.ceil(length / len(values)))
    return np.tile(values.to_numpy(), repeats)[:length]
