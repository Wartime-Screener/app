"""
EIA (Energy Information Administration) API client.

Provides access to weekly petroleum inventories and natural gas storage data.
Uses the EIA API v2.

Env var: EIA_API_KEY
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
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "eia"


# EIA API v2 series definitions
# Petroleum Supply & Disposition (weekly)
PETROLEUM_SERIES = {
    "crude_stocks": {
        "product": "EPC0",
        "process": "SAE",
        "label": "Crude Oil Ending Stocks",
        "units": "Thousand Barrels",
    },
    "gasoline_stocks": {
        "product": "EPM0F",
        "process": "SAE",
        "label": "Motor Gasoline Ending Stocks",
        "units": "Thousand Barrels",
    },
    "distillate_stocks": {
        "product": "EPD0",
        "process": "SAE",
        "label": "Distillate Fuel Oil Ending Stocks",
        "units": "Thousand Barrels",
    },
    "crude_production": {
        "product": "EPC0",
        "process": "FPF",
        "label": "Crude Oil Production",
        "units": "Thousand Barrels per Day",
    },
    "crude_imports": {
        "product": "EPC0",
        "process": "IM0",
        "label": "Crude Oil Imports",
        "units": "Thousand Barrels per Day",
    },
    "refinery_inputs": {
        "product": "EPC0",
        "process": "SRI",
        "label": "Refinery & Blender Net Inputs",
        "units": "Thousand Barrels per Day",
    },
}


# Spot price series definitions
SPOT_PRICE_SERIES = {
    "wti_crude": {
        "endpoint": "petroleum/pri/spt",
        "frequency": "daily",
        "series": "RWTC",
        "label": "WTI Crude Oil",
        "units": "$/Barrel",
    },
    "brent_crude": {
        "endpoint": "petroleum/pri/spt",
        "frequency": "daily",
        "series": "RBRTE",
        "label": "Brent Crude Oil",
        "units": "$/Barrel",
    },
    "gasoline": {
        "endpoint": "petroleum/pri/spt",
        "frequency": "daily",
        "series": "EER_EPMRU_PF4_Y35NY_DPG",
        "label": "NY Harbor Gasoline (RBOB)",
        "units": "$/Gallon",
    },
    "diesel": {
        "endpoint": "petroleum/pri/spt",
        "frequency": "daily",
        "series": "EER_EPD2DXL0_PF4_RGC_DPG",
        "label": "Gulf Coast Ultra-Low Sulfur Diesel",
        "units": "$/Gallon",
    },
    "jet_fuel": {
        "endpoint": "petroleum/pri/spt",
        "frequency": "daily",
        "series": "EER_EPJK_PF4_RGC_DPG",
        "label": "Gulf Coast Jet Fuel",
        "units": "$/Gallon",
    },
    "propane": {
        "endpoint": "petroleum/pri/spt",
        "frequency": "daily",
        "series": "EER_EPLLPA_PF4_Y44MB_DPG",
        "label": "Mont Belvieu Propane",
        "units": "$/Gallon",
    },
    "henry_hub": {
        "endpoint": "natural-gas/pri/fut",
        "frequency": "daily",
        "series": "RNGWHHD",
        "label": "Henry Hub Natural Gas",
        "units": "$/MMBtu",
    },
}


class EIAClient:
    """Client for the EIA API v2."""

    BASE_URL = "https://api.eia.gov/v2"

    def __init__(self, api_key: str | None = None, rate_delay: float = 0.3,
                 cache_ttl: int = 3600):
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        self.rate_delay = rate_delay
        self.cache_ttl = cache_ttl
        self._last_request_time = 0.0

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("EIA_API_KEY not set. EIA API calls will fail.")

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_delay:
            time.sleep(self.rate_delay - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, series_name: str, weeks: int) -> str:
        raw = f"{series_name}|{weeks}"
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
            logger.warning(f"Failed to write EIA cache: {e}")

    def get_petroleum_series(self, series_name: str, weeks: int = 104) -> list[dict]:
        """
        Fetch a weekly petroleum series.

        Args:
            series_name: Key from PETROLEUM_SERIES (e.g., 'crude_stocks').
            weeks: Number of weekly data points to retrieve.

        Returns:
            List of dicts with 'period' (date string) and 'value' (float),
            ordered most recent first.
        """
        if not self.is_configured:
            logger.error("EIA API key is not configured.")
            return []

        if series_name not in PETROLEUM_SERIES:
            logger.error(f"Unknown petroleum series: {series_name}")
            return []

        ck = self._cache_key(series_name, weeks)
        cached = self._read_cache(ck)
        if cached is not None:
            return cached

        series_def = PETROLEUM_SERIES[series_name]
        self._rate_limit()

        try:
            # Request extra rows since API returns all regions;
            # we filter to NUS (U.S. total) after
            resp = requests.get(
                f"{self.BASE_URL}/petroleum/sum/sndw/data/",
                params={
                    "api_key": self.api_key,
                    "frequency": "weekly",
                    "data[0]": "value",
                    "facets[product][]": series_def["product"],
                    "facets[process][]": series_def["process"],
                    "facets[duoarea][]": "NUS",
                    "sort[0][column]": "period",
                    "sort[0][direction]": "desc",
                    "length": str(weeks),
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.error(f"EIA API error {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            rows = data.get("response", {}).get("data", [])

            result = []
            for row in rows:
                val = row.get("value")
                if val is not None:
                    try:
                        result.append({
                            "period": row["period"],
                            "value": float(val),
                        })
                    except (ValueError, TypeError):
                        pass

            self._write_cache(ck, result)
            return result

        except requests.RequestException as e:
            logger.error(f"EIA request failed: {e}")
            return []

    def get_natural_gas_storage(self, weeks: int = 104) -> list[dict]:
        """
        Fetch weekly natural gas storage (working gas in underground storage).

        Args:
            weeks: Number of weekly data points to retrieve.

        Returns:
            List of dicts with 'period' and 'value' (Bcf), most recent first.
        """
        if not self.is_configured:
            logger.error("EIA API key is not configured.")
            return []

        ck = self._cache_key("natgas_storage", weeks)
        cached = self._read_cache(ck)
        if cached is not None:
            return cached

        self._rate_limit()

        try:
            resp = requests.get(
                f"{self.BASE_URL}/natural-gas/stor/wkly/data/",
                params={
                    "api_key": self.api_key,
                    "frequency": "weekly",
                    "data[0]": "value",
                    "facets[process][]": "SWO",
                    "facets[duoarea][]": "R48",
                    "sort[0][column]": "period",
                    "sort[0][direction]": "desc",
                    "length": str(weeks),
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.error(f"EIA API error {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            rows = data.get("response", {}).get("data", [])

            result = []
            for row in rows:
                val = row.get("value")
                if val is not None:
                    try:
                        result.append({
                            "period": row["period"],
                            "value": float(val),
                        })
                    except (ValueError, TypeError):
                        pass

            self._write_cache(ck, result)
            return result

        except requests.RequestException as e:
            logger.error(f"EIA request failed: {e}")
            return []

    def get_spot_price(self, series_name: str, days: int = 365) -> list[dict]:
        """
        Fetch daily spot price for an energy commodity.

        Args:
            series_name: Key from SPOT_PRICE_SERIES (e.g., 'wti_crude').
            days: Number of data points to retrieve.

        Returns:
            List of dicts with 'period' and 'value', most recent first.
        """
        if not self.is_configured:
            logger.error("EIA API key is not configured.")
            return []

        if series_name not in SPOT_PRICE_SERIES:
            logger.error(f"Unknown spot price series: {series_name}")
            return []

        ck = self._cache_key(f"spot_{series_name}", days)
        cached = self._read_cache(ck)
        if cached is not None:
            return cached

        series_def = SPOT_PRICE_SERIES[series_name]
        self._rate_limit()

        try:
            resp = requests.get(
                f"{self.BASE_URL}/{series_def['endpoint']}/data/",
                params={
                    "api_key": self.api_key,
                    "frequency": series_def["frequency"],
                    "data[0]": "value",
                    "facets[series][]": series_def["series"],
                    "sort[0][column]": "period",
                    "sort[0][direction]": "desc",
                    "length": str(days),
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.error(f"EIA API error {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            rows = data.get("response", {}).get("data", [])

            result = []
            for row in rows:
                val = row.get("value")
                if val is not None:
                    try:
                        result.append({
                            "period": row["period"],
                            "value": float(val),
                        })
                    except (ValueError, TypeError):
                        pass

            self._write_cache(ck, result)
            return result

        except requests.RequestException as e:
            logger.error(f"EIA request failed: {e}")
            return []

    def clear_cache(self):
        """Delete all cached EIA responses."""
        count = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} EIA cache files.")
        return count
