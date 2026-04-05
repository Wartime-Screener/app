"""Watchlist data management for the Stock Screener app.

Tracks tickers the user is watching but does not own.
Supports structured research notes (same format as portfolio).
"""

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"

# ---------------------------------------------------------------------------
# Pluggable storage backend (mirrors portfolio.py pattern)
# ---------------------------------------------------------------------------
_load_fn = None
_save_fn = None


def set_storage_backend(load_fn, save_fn):
    """Inject custom load/save functions (e.g. browser localStorage)."""
    global _load_fn, _save_fn
    _load_fn = load_fn
    _save_fn = save_fn


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_watchlist() -> dict:
    """Load watchlist. Uses injected backend if set, otherwise JSON file."""
    if _load_fn:
        return _load_fn()
    if WATCHLIST_FILE.exists():
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    return {"items": []}


def save_watchlist(data: dict) -> None:
    """Save watchlist. Uses injected backend if set, otherwise JSON file."""
    if _save_fn:
        _save_fn(data)
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def add_to_watchlist(ticker: str, reason: str = "", price_at_add: float | None = None) -> dict:
    """Add a ticker to the watchlist. Returns the new item dict."""
    item = {
        "id": str(uuid4()),
        "ticker": ticker.upper().strip(),
        "added_date": str(date.today()),
        "price_at_add": round(price_at_add, 2) if price_at_add else None,
        "reason": reason.strip(),
        "notes": [],
    }
    data = load_watchlist()
    data["items"].append(item)
    save_watchlist(data)
    return item


def remove_from_watchlist(item_id: str) -> bool:
    """Remove an item by UUID. Returns True if found and removed."""
    data = load_watchlist()
    original_len = len(data["items"])
    data["items"] = [i for i in data["items"] if i["id"] != item_id]
    if len(data["items"]) < original_len:
        save_watchlist(data)
        return True
    return False


def update_watchlist_notes(item_id: str, notes: list) -> bool:
    """Update structured notes for a watchlist item.

    *notes* is a list of dicts: [{"text": "...", "level": 0}, ...].
    Returns True if found.
    """
    data = load_watchlist()
    for item in data["items"]:
        if item["id"] == item_id:
            item["notes"] = notes
            save_watchlist(data)
            return True
    return False
