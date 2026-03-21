"""
Price cross-validation module.

Fetches stock prices from three sources (Tradier, FMP, yfinance) and
validates them against each other. If any source differs by more than 5%
from the others, it flags a discrepancy.

Uses majority consensus (2 out of 3 agreement) as the "validated" price.
"""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def _fetch_yfinance_price(ticker: str) -> float | None:
    """Fetch current price from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None)
        if price is None:
            # Fallback to previous close
            price = getattr(info, "previous_close", None)
        if price is not None and price > 0:
            return float(price)
    except Exception as e:
        logger.warning(f"yfinance price fetch failed for {ticker}: {e}")
    return None


def _fetch_tradier_price(ticker: str, tradier_client) -> float | None:
    """Fetch current price from Tradier."""
    try:
        df = tradier_client.get_quotes([ticker])
        if not df.empty:
            price = df.iloc[0].get("last")
            if price is not None and price > 0:
                return float(price)
    except Exception as e:
        logger.warning(f"Tradier price fetch failed for {ticker}: {e}")
    return None


def _fetch_fmp_price(ticker: str, fmp_client) -> float | None:
    """Fetch current price from FMP company profile."""
    try:
        profile = fmp_client.get_company_profile(ticker)
        if profile:
            price = profile.get("price")
            if price is not None and price > 0:
                return float(price)
    except Exception as e:
        logger.warning(f"FMP price fetch failed for {ticker}: {e}")
    return None


def _pct_diff(a: float, b: float) -> float:
    """Calculate percentage difference between two prices."""
    avg = (a + b) / 2
    if avg == 0:
        return 0.0
    return abs(a - b) / avg * 100


def cross_validate_price(ticker: str, tradier_client, fmp_client,
                          threshold_pct: float = 1.5) -> dict:
    """
    Fetch price from Tradier, FMP, and yfinance, then cross-validate.

    Args:
        ticker: Stock ticker symbol.
        tradier_client: Initialized TradierClient instance.
        fmp_client: Initialized FMPClient instance.
        threshold_pct: Maximum acceptable percentage difference (default 5%).

    Returns:
        Dict with keys:
            - sources: dict of {source_name: price_or_None}
            - validated_price: the consensus price (majority agreement)
            - has_discrepancy: True if any source differs > threshold
            - discrepancies: list of dicts describing each discrepancy
            - agreeing_sources: list of source names that agree
            - disagreeing_sources: list of source names that disagree
    """
    # Fetch from all three sources
    prices = {}
    prices["Tradier"] = _fetch_tradier_price(ticker, tradier_client)
    prices["FMP"] = _fetch_fmp_price(ticker, fmp_client)
    prices["yfinance"] = _fetch_yfinance_price(ticker)

    # Filter to sources that returned a valid price
    valid = {k: v for k, v in prices.items() if v is not None}

    result = {
        "sources": prices,
        "validated_price": None,
        "has_discrepancy": False,
        "discrepancies": [],
        "agreeing_sources": [],
        "disagreeing_sources": [],
    }

    if len(valid) == 0:
        return result

    if len(valid) == 1:
        # Only one source — use it as-is, no validation possible
        source_name = list(valid.keys())[0]
        result["validated_price"] = valid[source_name]
        result["agreeing_sources"] = [source_name]
        return result

    # Check pairwise discrepancies
    source_names = list(valid.keys())
    discrepancies = []
    for i in range(len(source_names)):
        for j in range(i + 1, len(source_names)):
            s1, s2 = source_names[i], source_names[j]
            diff = _pct_diff(valid[s1], valid[s2])
            if diff > threshold_pct:
                discrepancies.append({
                    "source1": s1,
                    "price1": valid[s1],
                    "source2": s2,
                    "price2": valid[s2],
                    "diff_pct": round(diff, 2),
                })

    result["discrepancies"] = discrepancies
    result["has_discrepancy"] = len(discrepancies) > 0

    if len(valid) == 2:
        # Two sources — use average as validated price
        vals = list(valid.values())
        result["validated_price"] = round((vals[0] + vals[1]) / 2, 4)
        if discrepancies:
            result["agreeing_sources"] = []
            result["disagreeing_sources"] = source_names
        else:
            result["agreeing_sources"] = source_names
            result["disagreeing_sources"] = []
        return result

    # Three sources — find majority consensus
    # Check which pairs agree (within threshold)
    agreement_matrix = {}
    for s in source_names:
        agreement_matrix[s] = []

    for i in range(len(source_names)):
        for j in range(i + 1, len(source_names)):
            s1, s2 = source_names[i], source_names[j]
            diff = _pct_diff(valid[s1], valid[s2])
            if diff <= threshold_pct:
                agreement_matrix[s1].append(s2)
                agreement_matrix[s2].append(s1)

    # Find the source(s) with the most agreements
    agreeing = set()
    disagreeing = set()

    for s in source_names:
        if len(agreement_matrix[s]) >= 1:
            agreeing.add(s)
            for partner in agreement_matrix[s]:
                agreeing.add(partner)

    disagreeing = set(source_names) - agreeing

    # If no clear agreement, all sources disagree — use median
    if not agreeing:
        vals = sorted(valid.values())
        result["validated_price"] = vals[1]  # median of 3
        result["agreeing_sources"] = []
        result["disagreeing_sources"] = source_names
    else:
        # Use average of agreeing sources as validated price
        agreeing_prices = [valid[s] for s in agreeing]
        result["validated_price"] = round(sum(agreeing_prices) / len(agreeing_prices), 4)
        result["agreeing_sources"] = sorted(agreeing)
        result["disagreeing_sources"] = sorted(disagreeing)

    return result
