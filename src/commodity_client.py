"""
Commodity price client using yfinance.

Covers metals and agricultural commodities not available through EIA.
Uses disk caching to avoid repeated API calls.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "commodities"

# Yahoo Finance futures symbols
COMMODITY_SYMBOLS = {
    # Metals
    "gold": {"symbol": "GC=F", "label": "Gold", "units": "$/oz"},
    "silver": {"symbol": "SI=F", "label": "Silver", "units": "$/oz"},
    "copper": {"symbol": "HG=F", "label": "Copper", "units": "$/lb"},
    "platinum": {"symbol": "PL=F", "label": "Platinum", "units": "$/oz"},
    "palladium": {"symbol": "PA=F", "label": "Palladium", "units": "$/oz"},
    # Agriculture
    "corn": {"symbol": "ZC=F", "label": "Corn", "units": "¢/bushel"},
    "wheat": {"symbol": "ZW=F", "label": "Wheat", "units": "¢/bushel"},
    "soybeans": {"symbol": "ZS=F", "label": "Soybeans", "units": "¢/bushel"},
    "sugar": {"symbol": "SB=F", "label": "Sugar", "units": "¢/lb"},
    "coffee": {"symbol": "KC=F", "label": "Coffee", "units": "¢/lb"},
    "cotton": {"symbol": "CT=F", "label": "Cotton", "units": "¢/lb"},
    "lumber": {"symbol": "LBS=F", "label": "Lumber", "units": "$/1000 bd ft"},
    # Uranium (ETF proxy — no futures contract)
    "uranium": {"symbol": "URA", "label": "Uranium (URA ETF)", "units": "$/share"},
}


def _cache_key(name: str, period: str) -> str:
    raw = f"{name}|{period}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(cache_key: str, ttl: int = 3600):
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        if datetime.now() - cached_at < timedelta(seconds=ttl):
            return data["response"]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _write_cache(cache_key: str, response_data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    payload = {
        "cached_at": datetime.now().isoformat(),
        "response": response_data,
    }
    try:
        cache_path.write_text(json.dumps(payload, default=str))
    except Exception as e:
        logger.warning(f"Failed to write commodity cache: {e}")


def get_commodity_history(name: str, period: str = "1y") -> list[dict]:
    """
    Fetch historical prices for a commodity.

    Args:
        name: Key from COMMODITY_SYMBOLS (e.g., 'gold', 'corn').
        period: yfinance period string ('1y', '2y', '5y', '10y', 'max').

    Returns:
        List of dicts with 'period' (date string) and 'value' (close price),
        ordered most recent first.
    """
    if name not in COMMODITY_SYMBOLS:
        logger.error(f"Unknown commodity: {name}")
        return []

    ck = _cache_key(name, period)
    cached = _read_cache(ck)
    if cached is not None:
        return cached

    symbol = COMMODITY_SYMBOLS[name]["symbol"]

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            logger.warning(f"No data returned for {symbol}")
            return []

        result = []
        for date, row in hist.iterrows():
            result.append({
                "period": date.strftime("%Y-%m-%d"),
                "value": round(float(row["Close"]), 4),
            })

        result.reverse()  # Most recent first
        _write_cache(ck, result)
        return result

    except Exception as e:
        logger.error(f"yfinance request failed for {symbol}: {e}")
        return []


def get_commodity_quote_by_symbol(symbol: str, label: str, units: str) -> dict | None:
    """
    Get current quote for an arbitrary yfinance symbol (e.g., CL=F for WTI futures).

    Returns:
        Dict with 'price', 'change', 'change_pct', 'label', 'units', or None.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty or len(hist) < 1:
            return None

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change = current - prev
        change_pct = (change / prev * 100) if prev != 0 else 0

        return {
            "price": round(current, 4),
            "change": round(change, 4),
            "change_pct": round(change_pct, 2),
            "label": label,
            "units": units,
        }
    except Exception as e:
        logger.error(f"yfinance quote failed for {symbol}: {e}")
        return None


def get_commodity_quote(name: str) -> dict | None:
    """
    Get current quote for a commodity.

    Returns:
        Dict with 'price', 'change', 'change_pct', 'label', 'units', or None.
    """
    if name not in COMMODITY_SYMBOLS:
        return None

    symbol = COMMODITY_SYMBOLS[name]["symbol"]
    info = COMMODITY_SYMBOLS[name]

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty or len(hist) < 1:
            return None

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change = current - prev
        change_pct = (change / prev * 100) if prev != 0 else 0

        return {
            "price": round(current, 4),
            "change": round(change, 4),
            "change_pct": round(change_pct, 2),
            "label": info["label"],
            "units": info["units"],
        }
    except Exception as e:
        logger.error(f"yfinance quote failed for {symbol}: {e}")
        return None


def clear_cache() -> int:
    """Delete all cached commodity responses."""
    count = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
    logger.info(f"Cleared {count} commodity cache files.")
    return count
