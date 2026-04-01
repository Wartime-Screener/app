"""
FRED (Federal Reserve Economic Data) API client.

Provides access to economic data series from the St. Louis Fed.
Currently used for egg prices (APU0000708111) but extensible to any FRED series.

Env var: FRED_API_KEY (free key from https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import os
import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "fred"


# FRED series definitions — add new series here as needed
FRED_SERIES = {
    "eggs": {
        "series_id": "APU0000708111",
        "label": "Eggs (Grade A Large)",
        "units": "$/dozen",
    },
    "treasury_10y": {
        "series_id": "DGS10",
        "label": "10-Year Treasury Yield",
        "units": "%",
    },
}

# Map period strings to approximate number of months to fetch
PERIOD_TO_MONTHS = {
    "1mo": 1,
    "3mo": 3,
    "6mo": 6,
    "1y": 12,
    "2y": 24,
    "5y": 60,
    "10y": 120,
    "max": 600,
}


class FREDClient:
    """Client for the FRED API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None, rate_delay: float = 0.3,
                 cache_ttl: int = 3600):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.rate_delay = rate_delay
        self.cache_ttl = cache_ttl
        self._last_request_time = 0.0

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("FRED_API_KEY not set. FRED API calls will fail.")

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_delay:
            time.sleep(self.rate_delay - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, series_id: str, period: str) -> str:
        raw = f"{series_id}|{period}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _read_cache(self, cache_key: str) -> list | None:
        cache_path = CACHE_DIR / f"{cache_key}.json"
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text())
            cached_at = datetime.fromisoformat(data["cached_at"])
            if datetime.now() - cached_at < timedelta(seconds=self.cache_ttl):
                return data["response"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def _write_cache(self, cache_key: str, response_data):
        cache_path = CACHE_DIR / f"{cache_key}.json"
        payload = {
            "cached_at": datetime.now().isoformat(),
            "response": response_data,
        }
        try:
            cache_path.write_text(json.dumps(payload, default=str))
        except Exception as e:
            logger.warning(f"Failed to write FRED cache: {e}")

    def get_series(self, series_name: str, period: str = "1y") -> list[dict]:
        """
        Fetch a FRED data series.

        Args:
            series_name: Key from FRED_SERIES (e.g., 'eggs').
            period: Time period string ('1mo', '3mo', '6mo', '1y', '2y',
                    '5y', '10y', 'max').

        Returns:
            List of dicts with 'period' (date string) and 'value' (float),
            ordered most recent first.
        """
        if not self.is_configured:
            logger.error("FRED API key is not configured.")
            return []

        if series_name not in FRED_SERIES:
            logger.error(f"Unknown FRED series: {series_name}")
            return []

        ck = self._cache_key(series_name, period)
        cached = self._read_cache(ck)
        if cached is not None:
            return cached

        series_def = FRED_SERIES[series_name]
        months = PERIOD_TO_MONTHS.get(period, 12)
        start_date = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")

        self._rate_limit()

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "series_id": series_def["series_id"],
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": start_date,
                    "sort_order": "desc",
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.error(f"FRED API error {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            observations = data.get("observations", [])

            result = []
            for obs in observations:
                val = obs.get("value")
                if val is not None and val != ".":
                    try:
                        result.append({
                            "period": obs["date"],
                            "value": float(val),
                        })
                    except (ValueError, TypeError):
                        pass

            self._write_cache(ck, result)
            return result

        except requests.RequestException as e:
            logger.error(f"FRED request failed: {e}")
            return []

    def get_latest(self, series_name: str) -> dict | None:
        """
        Get the most recent observation for a series.

        Returns:
            Dict with 'period', 'value', 'change', 'change_pct' or None.
        """
        data = self.get_series(series_name, period="3mo")
        if not data:
            return None

        latest = data[0]
        # Calculate change vs prior month
        if len(data) >= 2:
            prev = data[1]["value"]
            change = latest["value"] - prev
            change_pct = (change / prev * 100) if prev != 0 else 0.0
        else:
            change = 0.0
            change_pct = 0.0

        return {
            "price": latest["value"],
            "period": latest["period"],
            "change": change,
            "change_pct": change_pct,
        }

    def get_risk_free_rate(self) -> float | None:
        """
        Get the current 10-year Treasury yield as a decimal (e.g. 0.043 for 4.3%).

        Returns None if FRED is unavailable, allowing callers to fall back to a default.
        """
        data = self.get_series("treasury_10y", period="1mo")
        if data:
            return data[0]["value"] / 100.0  # FRED reports as percentage
        return None

    def clear_cache(self):
        """Delete all cached FRED responses."""
        count = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} FRED cache files.")
        return count
