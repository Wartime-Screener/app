"""Portfolio data management for the Stock Screener app.

Handles CRUD operations for stock positions, performance calculations,
and historical benchmarking via yfinance.
"""

import json
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yfinance as yf

PORTFOLIO_DIR = Path(__file__).resolve().parent.parent / "data"
PORTFOLIO_FILE = PORTFOLIO_DIR / "portfolio.json"

# ---------------------------------------------------------------------------
# Pluggable storage backend
# ---------------------------------------------------------------------------
_load_fn = None
_save_fn = None


def set_storage_backend(load_fn, save_fn):
    """Inject custom load/save functions (e.g. browser localStorage via session_state)."""
    global _load_fn, _save_fn
    _load_fn = load_fn
    _save_fn = save_fn


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_portfolio() -> dict:
    """Load portfolio. Uses injected backend if set, otherwise falls back to JSON file."""
    if _load_fn:
        return _load_fn()
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {"positions": []}


def save_portfolio(data: dict) -> None:
    """Save portfolio. Uses injected backend if set, otherwise falls back to JSON file."""
    if _save_fn:
        _save_fn(data)
        return
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def add_position(
    ticker: str,
    buy_date: str,
    amount_invested: float,
    cost_basis: float,
    thesis_tag: str,
    notes: str = "",
) -> dict:
    """Create a new position, persist it, and return the position dict."""
    shares = round(amount_invested / cost_basis, 4) if cost_basis else 0.0
    position = {
        "id": str(uuid4()),
        "ticker": ticker.upper().strip(),
        "buy_date": buy_date,
        "amount_invested": round(amount_invested, 2),
        "shares": shares,
        "cost_basis": round(cost_basis, 2),
        "thesis_tag": thesis_tag.strip(),
        "notes": notes,
    }
    data = load_portfolio()
    data["positions"].append(position)
    save_portfolio(data)
    return position


def remove_position(position_id: str) -> bool:
    """Remove a position by its UUID. Returns True if found and removed."""
    data = load_portfolio()
    original_len = len(data["positions"])
    data["positions"] = [p for p in data["positions"] if p["id"] != position_id]
    if len(data["positions"]) < original_len:
        save_portfolio(data)
        return True
    return False


def get_all_tags() -> list[str]:
    """Return a sorted list of unique thesis tags from existing positions."""
    data = load_portfolio()
    tags = {p["thesis_tag"] for p in data["positions"] if p.get("thesis_tag")}
    return sorted(tags)


# ---------------------------------------------------------------------------
# Performance calculations
# ---------------------------------------------------------------------------

def compute_position_performance(position: dict, current_price: float) -> dict:
    """Compute P&L metrics for a single position given its current price."""
    shares = position["shares"]
    amount_invested = position["amount_invested"]
    current_value = round(shares * current_price, 2)
    gain_loss_dollars = round(current_value - amount_invested, 2)
    gain_loss_pct = round((gain_loss_dollars / amount_invested) * 100, 2) if amount_invested else 0.0

    buy_date = datetime.strptime(position["buy_date"], "%Y-%m-%d").date()
    holding_days = (date.today() - buy_date).days

    return {
        "current_value": current_value,
        "gain_loss_dollars": gain_loss_dollars,
        "gain_loss_pct": gain_loss_pct,
        "holding_days": holding_days,
    }


def compute_portfolio_summary(positions: list[dict], current_prices: dict[str, float]) -> dict:
    """Aggregate performance across all positions.

    Parameters
    ----------
    positions : list of position dicts
    current_prices : mapping of ticker -> latest price
    """
    total_invested = 0.0
    current_value = 0.0

    for p in positions:
        total_invested += p["amount_invested"]
        price = current_prices.get(p["ticker"], p["cost_basis"])
        current_value += p["shares"] * price

    total_invested = round(total_invested, 2)
    current_value = round(current_value, 2)
    total_return_dollars = round(current_value - total_invested, 2)
    total_return_pct = round((total_return_dollars / total_invested) * 100, 2) if total_invested else 0.0

    return {
        "total_invested": total_invested,
        "current_value": current_value,
        "total_return_dollars": total_return_dollars,
        "total_return_pct": total_return_pct,
        "num_positions": len(positions),
    }


def compute_tag_performance(
    positions: list[dict], current_prices: dict[str, float]
) -> list[dict]:
    """Group positions by thesis_tag and compute aggregate stats per tag."""
    tag_map: dict[str, list[dict]] = {}
    for p in positions:
        tag = p.get("thesis_tag", "Untagged")
        tag_map.setdefault(tag, []).append(p)

    results = []
    for tag, group in sorted(tag_map.items()):
        total_invested = sum(p["amount_invested"] for p in group)
        current_value = sum(
            p["shares"] * current_prices.get(p["ticker"], p["cost_basis"])
            for p in group
        )
        total_invested = round(total_invested, 2)
        current_value = round(current_value, 2)
        return_pct = (
            round(((current_value - total_invested) / total_invested) * 100, 2)
            if total_invested
            else 0.0
        )
        results.append({
            "tag": tag,
            "count": len(group),
            "total_invested": total_invested,
            "current_value": current_value,
            "return_pct": return_pct,
        })

    return results


# ---------------------------------------------------------------------------
# Historical data (yfinance)
# ---------------------------------------------------------------------------

def get_position_history(
    ticker: str,
    buy_date_str: str,
    shares: float,
    cost_basis: float,
) -> pd.DataFrame:
    """Fetch daily price history from buy_date to today and compute returns.

    Returns a DataFrame with columns: date, close, position_value, return_pct.
    """
    start = datetime.strptime(buy_date_str, "%Y-%m-%d").date()
    try:
        df = yf.download(ticker, start=str(start), end=str(date.today()), progress=False)
    except Exception:
        return pd.DataFrame(columns=["date", "close", "position_value", "return_pct"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "position_value", "return_pct"])

    # yfinance may return multi-level columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    amount_invested = shares * cost_basis
    df["position_value"] = round(df["close"] * shares, 2)
    df["return_pct"] = round(((df["close"] - cost_basis) / cost_basis) * 100, 2)

    return df


def get_spy_benchmark(start_date_str: str, amount_invested: float) -> pd.DataFrame:
    """Fetch SPY daily history from start_date and compute a benchmark curve.

    Returns a DataFrame with columns: date, close, benchmark_value, return_pct.
    """
    start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    try:
        df = yf.download("SPY", start=str(start), end=str(date.today()), progress=False)
    except Exception:
        return pd.DataFrame(columns=["date", "close", "benchmark_value", "return_pct"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "benchmark_value", "return_pct"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    initial_price = df["close"].iloc[0]
    benchmark_shares = amount_invested / initial_price if initial_price else 0.0
    df["benchmark_value"] = round(df["close"] * benchmark_shares, 2)
    df["return_pct"] = round(((df["close"] - initial_price) / initial_price) * 100, 2)

    return df
