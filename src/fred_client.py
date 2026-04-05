"""
FRED (Federal Reserve Economic Data) API client.

Provides access to economic data series from the St. Louis Fed.
Covers labor market, activity/leading indicators, inflation, GDP,
interest rates, and commodity prices.

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
# category field controls which tab/section displays the series
FRED_SERIES = {
    # --- Commodity (displayed in Commodity Prices tab) ---
    "eggs": {
        "series_id": "APU0000708111",
        "label": "Eggs (Grade A Large)",
        "units": "$/dozen",
        "category": "commodity",
    },

    # --- Labor Market ---
    "unemployment_rate": {
        "series_id": "UNRATE",
        "label": "Unemployment Rate",
        "units": "%",
        "category": "labor",
    },
    "initial_claims": {
        "series_id": "ICSA",
        "label": "Initial Jobless Claims",
        "units": "Thousands",
        "category": "labor",
    },
    "nonfarm_payrolls": {
        "series_id": "PAYEMS",
        "label": "Nonfarm Payrolls",
        "units": "Thousands",
        "category": "labor",
    },
    "job_openings": {
        "series_id": "JTSJOL",
        "label": "Job Openings (JOLTS)",
        "units": "Thousands",
        "category": "labor",
    },
    "quits_rate": {
        "series_id": "JTSQUR",
        "label": "Quits Rate",
        "units": "%",
        "category": "labor",
    },
    "agg_weekly_hours": {
        "series_id": "AWHI",
        "label": "Aggregate Weekly Hours (Goods)",
        "units": "Index",
        "category": "labor",
    },
    "kc_fed_labor_momentum": {
        "series_id": "FRBKCLMCIM",
        "label": "KC Fed Labor (Momentum)",
        "units": "Index",
        "category": "labor",
    },
    "kc_fed_labor_activity": {
        "series_id": "FRBKCLMCILA",
        "label": "KC Fed Labor (Activity)",
        "units": "Index",
        "category": "labor",
    },
    "continued_claims": {
        "series_id": "CCSA",
        "label": "Continued Jobless Claims",
        "units": "Thousands",
        "category": "labor",
    },

    # --- Activity & Leading ---
    "industrial_production": {
        "series_id": "INDPRO",
        "label": "Industrial Production",
        "units": "Index",
        "category": "activity",
    },
    "retail_sales": {
        "series_id": "RSAFS",
        "label": "Retail Sales",
        "units": "$M",
        "category": "activity",
    },
    "housing_starts": {
        "series_id": "HOUST",
        "label": "Housing Starts",
        "units": "Thousands",
        "category": "activity",
    },
    "building_permits": {
        "series_id": "PERMIT",
        "label": "Building Permits",
        "units": "Thousands",
        "category": "activity",
    },
    "consumer_sentiment": {
        "series_id": "UMCSENT",
        "label": "Consumer Sentiment",
        "units": "Index",
        "category": "activity",
    },
    "mfg_employment": {
        "series_id": "MANEMP",
        "label": "Manufacturing Employment",
        "units": "Thousands",
        "category": "activity",
    },
    "cfnai": {
        "series_id": "CFNAI",
        "label": "Chicago Fed National Activity",
        "units": "Index",
        "category": "activity",
    },
    "truck_tonnage": {
        "series_id": "TRUCKD11",
        "label": "ATA Truck Tonnage",
        "units": "Index",
        "category": "activity",
    },
    "freight_index": {
        "series_id": "TSIFRGHT",
        "label": "Freight Transportation Services",
        "units": "Index",
        "category": "activity",
    },
    "vix": {
        "series_id": "VIXCLS",
        "label": "CBOE VIX",
        "units": "Index",
        "category": "activity",
    },
    "nfci": {
        "series_id": "NFCI",
        "label": "Chicago Fed Financial Conditions",
        "units": "Index",
        "category": "activity",
    },

    # --- Inflation ---
    "cpi": {
        "series_id": "CPIAUCSL",
        "label": "CPI (All Urban)",
        "units": "Index",
        "category": "inflation",
    },
    "core_cpi": {
        "series_id": "CPILFESL",
        "label": "Core CPI",
        "units": "Index",
        "category": "inflation",
    },
    "pce": {
        "series_id": "PCEPI",
        "label": "PCE Price Index",
        "units": "Index",
        "category": "inflation",
    },
    "core_pce": {
        "series_id": "PCEPILFE",
        "label": "Core PCE",
        "units": "Index",
        "category": "inflation",
    },
    "ppi": {
        "series_id": "PPIACO",
        "label": "PPI (All Commodities)",
        "units": "Index",
        "category": "inflation",
    },

    # --- GDP ---
    "real_gdp": {
        "series_id": "GDPC1",
        "label": "Real GDP",
        "units": "$B (2017)",
        "category": "gdp",
    },
    "real_gdp_growth": {
        "series_id": "A191RL1Q225SBEA",
        "label": "Real GDP Growth (QoQ Ann.)",
        "units": "%",
        "category": "gdp",
    },

    # --- Rates & Money ---
    "fed_funds": {
        "series_id": "FEDFUNDS",
        "label": "Fed Funds Rate",
        "units": "%",
        "category": "rates",
    },
    "treasury_2y": {
        "series_id": "DGS2",
        "label": "2-Year Treasury",
        "units": "%",
        "category": "rates",
    },
    "treasury_10y": {
        "series_id": "DGS10",
        "label": "10-Year Treasury",
        "units": "%",
        "category": "rates",
    },
    "treasury_30y": {
        "series_id": "DGS30",
        "label": "30-Year Treasury",
        "units": "%",
        "category": "rates",
    },
    "yield_spread_10y2y": {
        "series_id": "T10Y2Y",
        "label": "10Y-2Y Spread",
        "units": "%",
        "category": "rates",
    },
    "m2_money_supply": {
        "series_id": "M2SL",
        "label": "M2 Money Supply",
        "units": "$B",
        "category": "rates",
    },

    # --- Credit & Lending ---
    "aaa_bond_yield": {
        "series_id": "AAA",
        "label": "Moody's Aaa Corporate Bond Yield",
        "units": "%",
        "category": "credit",
    },
    "hy_oas": {
        "series_id": "BAMLH0A0HYM2",
        "label": "High Yield OAS",
        "units": "%",
        "category": "credit",
    },
    "ig_oas": {
        "series_id": "BAMLC0A0CM",
        "label": "Investment Grade OAS",
        "units": "%",
        "category": "credit",
    },
    "mortgage_30y": {
        "series_id": "MORTGAGE30US",
        "label": "30-Year Mortgage Rate",
        "units": "%",
        "category": "credit",
    },
}

# Groupings for the Economic Indicators tab
ECON_CATEGORIES = {
    "Labor Market": [
        "unemployment_rate", "initial_claims", "continued_claims", "nonfarm_payrolls",
        "job_openings", "quits_rate", "agg_weekly_hours",
        "kc_fed_labor_momentum", "kc_fed_labor_activity",
    ],
    "Activity & Leading": [
        "industrial_production", "retail_sales", "housing_starts", "building_permits",
        "consumer_sentiment", "mfg_employment", "cfnai", "truck_tonnage", "freight_index",
        "vix", "nfci",
    ],
    "Inflation": ["cpi", "core_cpi", "pce", "core_pce", "ppi"],
    "GDP": ["real_gdp", "real_gdp_growth"],
    "Credit & Lending": [
        "aaa_bond_yield", "hy_oas", "ig_oas", "mortgage_30y",
    ],
    "Rates & Money": [
        "fed_funds", "treasury_2y", "treasury_10y", "treasury_30y",
        "yield_spread_10y2y", "m2_money_supply",
    ],
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


def format_econ_value(value: float, units: str) -> str:
    """Format a value based on its units for display."""
    if value is None:
        return "N/A"
    if units == "%":
        return f"{value:.2f}%"
    elif units in ("$M", "$B", "$B (2017)"):
        return f"{value:,.0f}"
    elif units == "$/dozen":
        return f"${value:.2f}"
    elif units == "Thousands":
        return f"{value:,.0f}"
    elif units == "Index":
        return f"{value:.1f}"
    else:
        return f"{value:,.2f}"


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

        max_retries = 3
        for attempt in range(max_retries):
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
                if resp.status_code == 500 and attempt < max_retries - 1:
                    logger.warning(f"FRED 500 error for {series_name}, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(1)
                    continue
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
                if attempt < max_retries - 1:
                    logger.warning(f"FRED request failed for {series_name}, retrying: {e}")
                    time.sleep(1)
                    continue
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
