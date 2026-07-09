TARGET = "is_fraud"
PHISHING_TARGET = "is_phishing_attack"
TIME_COL = "event_ts"

NUMERIC_FEATURES = [
    "amount",
    "account_age_minutes",
    "time_since_signup_minutes",
    "transactions_1h",
    "transactions_24h",
    "unique_cards_24h",
    "promo_uses_24h",
    "device_users_24h",
    "ip_users_24h",
    "payment_bin_risk",
    "chargebacks_90d",
    "url_phishing_score",
]

CATEGORICAL_FEATURES = [
    "country_mismatch",
    "new_device",
    "proxy_or_vpn",
    "browser",
    "channel",
    "payment_method",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
