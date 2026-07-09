from __future__ import annotations

import re


SUSPICIOUS_TLDS = {"zip", "top", "xyz", "click", "work", "support", "info"}
BRAND_TOKENS = {"amazon", "mercado", "paypal", "bank", "visa", "mastercard", "secure"}
ACTION_TOKENS = {"login", "verify", "bonus", "gift", "promo"}


class PhishingPolicy:
    """Domain policy that scores URL risk without depending on ML libraries."""

    def score(self, url: str | None) -> float:
        if not url:
            return 0.0

        normalized = url.lower().strip()
        host = self._host(normalized)
        checks = [
            (normalized.startswith("http://"), 0.22),
            ("@" in normalized, 0.16),
            ("-" in host, 0.12),
            (len(normalized) > 90, 0.10),
            (bool(re.search(r"\d+\.\d+\.\d+\.\d+", host)), 0.12),
            (normalized.count(".") >= 4, 0.10),
            (self._tld(host) in SUSPICIOUS_TLDS, 0.12),
            (any(token in normalized for token in ACTION_TOKENS), 0.14),
        ]
        brand_score = min(0.16, 0.06 * sum(token in host for token in BRAND_TOKENS))
        return round(min(sum(weight for passed, weight in checks if passed) + brand_score, 1.0), 4)

    @staticmethod
    def _host(url: str) -> str:
        return url.split("/")[2] if "://" in url else url.split("/")[0]

    @staticmethod
    def _tld(host: str) -> str:
        return host.rsplit(".", 1)[-1] if "." in host else ""
