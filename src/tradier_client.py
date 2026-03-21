"""
Tradier API client for real-time stock quotes.

Uses the TRADIER_TOKEN environment variable for authentication.
Designed for batch quote fetching with rate limiting and error handling.
"""

import os
import time
import logging
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Project root for cache paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "quotes"


class TradierClient:
    """Client for the Tradier API focused on stock quotes."""

    def __init__(self, token: str | None = None, base_url: str | None = None,
                 rate_delay: float = 0.15, cache_ttl: int = 300):
        """
        Initialize the Tradier client.

        Args:
            token: Tradier API token. Falls back to TRADIER_TOKEN env var.
            base_url: API base URL. Falls back to TRADIER_BASE_URL env var
                      or default https://api.tradier.com.
            rate_delay: Minimum seconds between API requests.
            cache_ttl: Cache time-to-live in seconds (default 5 minutes).
        """
        self.token = token or os.environ.get("TRADIER_TOKEN")
        self.base_url = (
            base_url
            or os.environ.get("TRADIER_BASE_URL")
            or "https://api.tradier.com"
        )
        self.rate_delay = rate_delay
        self.cache_ttl = cache_ttl
        self._last_request_time = 0.0

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.token:
            logger.warning(
                "TRADIER_TOKEN not set. Tradier API calls will fail. "
                "Set the TRADIER_TOKEN environment variable."
            )

    @property
    def is_configured(self) -> bool:
        """Return True if the API token is available."""
        return self.token is not None and len(self.token) > 0

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_delay:
            time.sleep(self.rate_delay - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, tickers: list[str]) -> Path:
        """Generate a cache file path for a set of tickers."""
        key = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()
        return CACHE_DIR / f"quotes_{key}.json"

    def _read_cache(self, tickers: list[str]) -> dict | None:
        """Read cached quote data if it exists and is fresh."""
        cache_path = self._get_cache_path(tickers)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text())
            cached_at = datetime.fromisoformat(data["cached_at"])
            if datetime.now() - cached_at < timedelta(seconds=self.cache_ttl):
                return data["quotes"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def _write_cache(self, tickers: list[str], quotes: list[dict]):
        """Write quote data to disk cache."""
        cache_path = self._get_cache_path(tickers)
        payload = {
            "cached_at": datetime.now().isoformat(),
            "quotes": quotes,
        }
        cache_path.write_text(json.dumps(payload, default=str))

    def _make_request(self, endpoint: str, params: dict | None = None,
                      max_retries: int = 3) -> dict | None:
        """
        Make an authenticated GET request to the Tradier API with retry logic.

        Args:
            endpoint: API endpoint path (e.g., '/v1/markets/quotes').
            params: Query parameters.
            max_retries: Maximum retry attempts on 429/5xx errors.

        Returns:
            Parsed JSON response or None on failure.
        """
        if not self.is_configured:
            logger.error("Tradier API token is not configured.")
            return None

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = (2 ** attempt) * 1.0
                    logger.warning(f"Rate limited (429). Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    wait = (2 ** attempt) * 0.5
                    logger.warning(
                        f"Server error {resp.status_code}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"Tradier API error {resp.status_code}: {resp.text[:200]}"
                    )
                    return None
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error(f"Max retries exceeded for {endpoint}")
        return None

    def get_quotes(self, tickers: list[str]) -> pd.DataFrame:
        """
        Fetch real-time quotes for a list of tickers.

        Tradier supports comma-separated symbols in a single request
        (up to ~100 at a time). This method batches as needed.

        Args:
            tickers: List of stock ticker symbols.

        Returns:
            DataFrame with columns: symbol, last, change, change_pct, volume,
            open, high, low, close, week_52_high, week_52_low, market_cap.
            Returns empty DataFrame on error.
        """
        if not tickers:
            return pd.DataFrame()

        # Check cache
        cached = self._read_cache(tickers)
        if cached is not None:
            logger.info(f"Using cached quotes for {len(tickers)} tickers")
            return pd.DataFrame(cached)

        all_quotes = []
        batch_size = 100

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            symbols_str = ",".join(batch)

            data = self._make_request(
                "/v1/markets/quotes",
                params={"symbols": symbols_str, "greeks": "false"},
            )

            if data is None:
                continue

            quotes_data = data.get("quotes", {})
            quote_list = quotes_data.get("quote", [])

            # Single quote comes back as dict, not list
            if isinstance(quote_list, dict):
                quote_list = [quote_list]

            for q in quote_list:
                all_quotes.append({
                    "symbol": q.get("symbol", ""),
                    "last": q.get("last"),
                    "change": q.get("change"),
                    "change_pct": q.get("change_percentage"),
                    "volume": q.get("volume"),
                    "open": q.get("open"),
                    "high": q.get("high"),
                    "low": q.get("low"),
                    "close": q.get("close"),
                    "week_52_high": q.get("week_52_high"),
                    "week_52_low": q.get("week_52_low"),
                    "market_cap": q.get("market_cap"),
                })

        if all_quotes:
            self._write_cache(tickers, all_quotes)

        df = pd.DataFrame(all_quotes)
        if df.empty:
            logger.warning("No quotes returned from Tradier.")
        return df
