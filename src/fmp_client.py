"""
Financial Modeling Prep (FMP) API client.

Provides access to financial statements, ratios, key metrics, company profiles,
and historical prices. Includes disk-based caching with configurable TTL and
rate limiting with exponential backoff.

Env var: FMP_API_KEY
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
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "fmp"

# Default cache TTLs in seconds
DEFAULT_TTLS = {
    "financial_statements": 86400,   # 24 hours
    "ratios": 86400,                 # 24 hours
    "key_metrics": 86400,            # 24 hours
    "profiles": 604800,              # 7 days
    "price_history": 14400,          # 4 hours
}


class FMPClient:
    """Client for the Financial Modeling Prep API."""

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str | None = None, rate_delay: float = 0.3,
                 cache_ttls: dict | None = None):
        """
        Initialize the FMP client.

        Args:
            api_key: FMP API key. Falls back to FMP_API_KEY env var.
            rate_delay: Minimum seconds between API requests.
            cache_ttls: Dict of cache category -> TTL in seconds.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self.rate_delay = rate_delay
        self.cache_ttls = {**DEFAULT_TTLS, **(cache_ttls or {})}
        self._last_request_time = 0.0

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning(
                "FMP_API_KEY not set. FMP API calls will fail. "
                "Set the FMP_API_KEY environment variable."
            )

    @property
    def is_configured(self) -> bool:
        """Return True if the API key is available."""
        return self.api_key is not None and len(self.api_key) > 0

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_delay:
            time.sleep(self.rate_delay - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, endpoint: str, params: dict) -> str:
        """Generate a unique cache key for a request."""
        # Exclude apikey from cache key
        filtered = {k: v for k, v in sorted(params.items()) if k != "apikey"}
        raw = f"{endpoint}|{json.dumps(filtered, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _read_cache(self, cache_key: str, category: str) -> dict | list | None:
        """Read cached response if it exists and is within TTL."""
        cache_path = CACHE_DIR / f"{cache_key}.json"
        if not cache_path.exists():
            return None

        ttl = self.cache_ttls.get(category, 86400)
        try:
            data = json.loads(cache_path.read_text())
            cached_at = datetime.fromisoformat(data["cached_at"])
            if datetime.now() - cached_at < timedelta(seconds=ttl):
                return data["response"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def _write_cache(self, cache_key: str, response_data):
        """Write response data to disk cache."""
        cache_path = CACHE_DIR / f"{cache_key}.json"
        payload = {
            "cached_at": datetime.now().isoformat(),
            "response": response_data,
        }
        try:
            cache_path.write_text(json.dumps(payload, default=str))
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def _make_request(self, endpoint: str, params: dict | None = None,
                      cache_category: str = "ratios",
                      max_retries: int = 3):
        """
        Make an authenticated GET request to the FMP API.

        Args:
            endpoint: API endpoint path (e.g., '/ratios/XOM').
            params: Additional query parameters.
            cache_category: Cache TTL category for this request.
            max_retries: Maximum retry attempts on rate limit/server errors.

        Returns:
            Parsed JSON response (list or dict) or None on failure.
        """
        if not self.is_configured:
            logger.error("FMP API key is not configured.")
            return None

        all_params = {"apikey": self.api_key}
        if params:
            all_params.update(params)

        # Check cache first
        ck = self._cache_key(endpoint, all_params)
        cached = self._read_cache(ck, cache_category)
        if cached is not None:
            logger.debug(f"Cache hit for {endpoint}")
            return cached

        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = requests.get(url, params=all_params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    # FMP returns 200 with error dict for invalid keys/endpoints
                    if isinstance(data, dict) and "Error Message" in data:
                        logger.error(f"FMP API error: {data['Error Message']}")
                        return None
                    self._write_cache(ck, data)
                    return data
                elif resp.status_code == 429:
                    wait = (2 ** attempt) * 1.0
                    logger.warning(f"FMP rate limited (429). Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    wait = (2 ** attempt) * 0.5
                    logger.warning(
                        f"FMP server error {resp.status_code}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"FMP API error {resp.status_code}: {resp.text[:200]}"
                    )
                    return None
            except requests.RequestException as e:
                logger.error(f"FMP request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error(f"Max retries exceeded for {endpoint}")
        return None

    # ------------------------------------------------------------------ #
    # Public API methods
    # ------------------------------------------------------------------ #

    def get_financial_ratios(self, ticker: str, period: str = "annual",
                             limit: int = 10) -> list[dict]:
        """
        Fetch financial ratios (P/E, P/S, P/B, EV/EBITDA, D/E, ROE, etc.).

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of ratio dicts ordered by date descending, or empty list.
        """
        data = self._make_request(
            "/ratios",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="ratios",
        )
        return data if isinstance(data, list) else []

    def get_key_metrics(self, ticker: str, period: str = "annual",
                        limit: int = 10) -> list[dict]:
        """
        Fetch key metrics (market cap, EV, revenue per share, earnings yield, etc.).

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of metric dicts ordered by date descending, or empty list.
        """
        data = self._make_request(
            "/key-metrics",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="key_metrics",
        )
        return data if isinstance(data, list) else []

    def get_income_statement(self, ticker: str, period: str = "annual",
                              limit: int = 10) -> list[dict]:
        """
        Fetch income statements.

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of income statement dicts, or empty list.
        """
        data = self._make_request(
            "/income-statement",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="financial_statements",
        )
        return data if isinstance(data, list) else []

    def get_balance_sheet(self, ticker: str, period: str = "annual",
                           limit: int = 10) -> list[dict]:
        """
        Fetch balance sheet statements.

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of balance sheet dicts, or empty list.
        """
        data = self._make_request(
            "/balance-sheet-statement",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="financial_statements",
        )
        return data if isinstance(data, list) else []

    def get_cash_flow(self, ticker: str, period: str = "annual",
                       limit: int = 10) -> list[dict]:
        """
        Fetch cash flow statements.

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of cash flow dicts, or empty list.
        """
        data = self._make_request(
            "/cash-flow-statement",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="financial_statements",
        )
        return data if isinstance(data, list) else []

    def get_company_profile(self, ticker: str) -> dict:
        """
        Fetch company profile (sector, description, market cap, employees, etc.).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Profile dict or empty dict.
        """
        data = self._make_request(
            "/profile",
            params={"symbol": ticker},
            cache_category="profiles",
        )
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}

    def get_analyst_estimates(self, ticker: str, period: str = "annual",
                              limit: int = 10) -> list[dict]:
        """
        Fetch analyst consensus estimates (forward revenue, EPS, EBITDA, etc.).

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of estimate dicts ordered by date descending, or empty list.
        """
        data = self._make_request(
            "/analyst-estimates",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="ratios",
        )
        return data if isinstance(data, list) else []

    def get_price_target_consensus(self, ticker: str) -> dict:
        """
        Fetch analyst price target consensus (high, low, median, consensus).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with targetHigh, targetLow, targetConsensus, targetMedian,
            or empty dict.
        """
        data = self._make_request(
            "/price-target-consensus",
            params={"symbol": ticker},
            cache_category="ratios",
        )
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}

    def get_earning_call_transcript(self, ticker: str, year: int,
                                     quarter: int) -> dict:
        """
        Fetch earnings call transcript for a specific quarter.

        Args:
            ticker: Stock ticker symbol.
            year: Calendar year (e.g., 2025).
            quarter: Quarter number (1-4).

        Returns:
            Dict with symbol, period, year, date, content, or empty dict.
        """
        data = self._make_request(
            "/earning-call-transcript",
            params={"symbol": ticker, "year": str(year), "quarter": str(quarter)},
            cache_category="financial_statements",
        )
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data if isinstance(data, dict) else {}

    def get_earning_call_transcript_dates(self, ticker: str,
                                           limit: int = 20) -> list[dict]:
        """
        Find available transcript dates by checking recent quarters.

        Args:
            ticker: Stock ticker symbol.
            limit: Max quarters to check.

        Returns:
            List of dicts with year, quarter info for available transcripts.
        """
        from datetime import datetime
        available = []
        now = datetime.now()
        year = now.year
        quarter = (now.month - 1) // 3 + 1

        checked = 0
        while checked < limit:
            transcript = self.get_earning_call_transcript(ticker, year, quarter)
            if transcript and transcript.get("content"):
                available.append({
                    "year": year,
                    "quarter": quarter,
                    "date": transcript.get("date", ""),
                    "period": f"Q{quarter} {year}",
                })
            checked += 1
            quarter -= 1
            if quarter < 1:
                quarter = 4
                year -= 1
            if len(available) >= 8:  # Stop after finding 8 transcripts
                break
        return available

    def get_historical_price(self, ticker: str, from_date: str | None = None,
                              to_date: str | None = None) -> list[dict]:
        """
        Fetch historical daily price data.

        Args:
            ticker: Stock ticker symbol.
            from_date: Start date string 'YYYY-MM-DD' (optional).
            to_date: End date string 'YYYY-MM-DD' (optional).

        Returns:
            List of price dicts with date, open, high, low, close, volume.
            Ordered most recent first.
        """
        params = {"symbol": ticker}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._make_request(
            "/historical-price-eod/full",
            params=params,
            cache_category="price_history",
        )
        # Stable endpoint returns a flat list of dicts
        if isinstance(data, list):
            return data
        # Legacy fallback: old endpoint returned dict with "historical" key
        if isinstance(data, dict):
            return data.get("historical", [])
        return []

    def get_revenue_product_segmentation(self, ticker: str,
                                           period: str = "annual",
                                           limit: int = 10) -> list[dict]:
        """
        Fetch revenue breakdown by product/business segment.

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of dicts with fiscalYear, date, and data (segment -> revenue),
            or empty list.
        """
        data = self._make_request(
            "/revenue-product-segmentation",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="financial_statements",
        )
        return data if isinstance(data, list) else []

    def get_revenue_geographic_segmentation(self, ticker: str,
                                             period: str = "annual",
                                             limit: int = 10) -> list[dict]:
        """
        Fetch revenue breakdown by geography (e.g., US vs Non-US).

        Args:
            ticker: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of dicts with fiscalYear, date, and data (region -> revenue),
            or empty list.
        """
        data = self._make_request(
            "/revenue-geographic-segmentation",
            params={"symbol": ticker, "period": period, "limit": str(limit)},
            cache_category="financial_statements",
        )
        return data if isinstance(data, list) else []

    def get_insider_trades(self, ticker: str, limit: int = 50) -> list[dict]:
        """
        Fetch recent insider trading transactions (Form 4 filings).

        Args:
            ticker: Stock ticker symbol.
            limit: Number of transactions to retrieve.

        Returns:
            List of dicts with filingDate, transactionDate, reportingName,
            transactionType, securitiesTransacted, price, typeOfOwner, etc.
        """
        data = self._make_request(
            "/insider-trading/search",
            params={"symbol": ticker, "limit": str(limit)},
            cache_category="financial_statements",
        )
        return data if isinstance(data, list) else []

    def get_insider_trade_statistics(self, ticker: str) -> list[dict]:
        """
        Fetch quarterly insider trading statistics (buy/sell summary).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of dicts with year, quarter, acquiredTransactions,
            disposedTransactions, totalAcquired, totalDisposed, etc.
        """
        data = self._make_request(
            "/insider-trading/statistics",
            params={"symbol": ticker},
            cache_category="financial_statements",
        )
        return data if isinstance(data, list) else []

    def clear_cache(self):
        """Delete all cached FMP responses."""
        count = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} FMP cache files.")
        return count
