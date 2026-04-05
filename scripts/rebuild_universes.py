#!/usr/bin/env python3
"""
Rebuild universe CSVs from FMP's company screener endpoint.

Uses FMP's paid screener data (not Nasdaq) to ensure accuracy.
Filters to NYSE/NASDAQ/AMEX-listed stocks above a market cap minimum, excluding ETFs and funds.

Usage:
    python scripts/rebuild_universes.py [--min-market-cap 500000000] [--dry-run]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen, Request

PROJECT_ROOT = Path(__file__).parent.parent
UNIVERSES_DIR = PROJECT_ROOT / "config" / "universes"
FMP_API_KEY = "zLtOncZT58CzNDhPwEi7ZGDvfYgAvp7o"

DEFAULT_MIN_MARKET_CAP = 500_000_000  # $500M


def fetch_screener(min_market_cap: int) -> list[dict]:
    """Fetch all NYSE/NASDAQ/AMEX stocks above market cap minimum from FMP screener."""
    url = (
        f"https://financialmodelingprep.com/stable/company-screener"
        f"?marketCapMoreThan={min_market_cap}"
        f"&exchange=NYSE,NASDAQ,AMEX"
        f"&limit=10000"
        f"&apikey={FMP_API_KEY}"
    )
    req = Request(url, headers={"User-Agent": "StockScreener/1.0"})
    with urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    if isinstance(data, dict) and "Error Message" in data:
        print(f"API Error: {data['Error Message']}", file=sys.stderr)
        sys.exit(1)

    # Keywords that indicate a fund/ETF, not an operating company
    FUND_KEYWORDS = re.compile(
        r"\b(ETF|Fund|Trust|Portfolio|Index|CLO|BDC|REIT|Income|Growth|Dividend|"
        r"Balanced|Bond|Equity|Shares|Notes|Warrant|Unit|Certificate|"
        r"Interval|Closed.End|Open.End|Series|Class [A-Z]|NAV)\b",
        re.IGNORECASE
    )

    def is_operating_company(d: dict) -> bool:
        if d.get("isEtf") or d.get("isFund"):
            return False
        if not d.get("industry"):
            return False
        name = d.get("companyName", "")
        # Ticker ends in X (mutual fund share class convention)
        ticker = d.get("symbol", "")
        if len(ticker) == 5 and ticker.endswith("X"):
            return False
        if FUND_KEYWORDS.search(name):
            return False
        return True

    stocks = [d for d in data if is_operating_company(d)]
    return stocks


def industry_to_filename(industry: str) -> str:
    """Convert FMP industry name to a clean filename.

    e.g. 'Oil & Gas Exploration & Production' -> 'oil_and_gas_exploration_and_production'
    """
    name = industry.lower()
    name = name.replace("&", "and")
    name = name.replace("-", "_")
    name = name.replace(",", "")
    name = name.replace("'", "")
    name = re.sub(r"[^a-z0-9_\s]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name)
    return name


def build_universes(stocks: list[dict]) -> dict[str, list[dict]]:
    """Group stocks by industry into universe buckets."""
    universes: dict[str, list[dict]] = {}

    for stock in stocks:
        industry = stock.get("industry", "").strip()
        if not industry:
            continue

        filename = industry_to_filename(industry)

        if filename not in universes:
            universes[filename] = []

        universes[filename].append({
            "ticker": stock["symbol"],
            "company_name": stock.get("companyName", ""),
            "segment": industry_to_filename(industry),
            "sub_segment": "",
        })

    # Sort tickers within each universe
    for filename in universes:
        universes[filename].sort(key=lambda x: x["ticker"])

    return universes


def write_universes(universes: dict[str, list[dict]], dry_run: bool = False):
    """Write universe CSVs, replacing all existing files."""
    if not dry_run:
        # Remove all existing CSV files
        for existing in UNIVERSES_DIR.glob("*.csv"):
            existing.unlink()
        UNIVERSES_DIR.mkdir(parents=True, exist_ok=True)

    total_tickers = 0
    for filename, stocks in sorted(universes.items()):
        total_tickers += len(stocks)
        if dry_run:
            print(f"  {filename}.csv ({len(stocks)} tickers)")
            continue

        csv_path = UNIVERSES_DIR / f"{filename}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "company_name", "segment", "sub_segment"])
            writer.writeheader()
            writer.writerows(stocks)

    return total_tickers


def main():
    parser = argparse.ArgumentParser(description="Rebuild universe CSVs from FMP screener")
    parser.add_argument(
        "--min-market-cap", type=int, default=DEFAULT_MIN_MARKET_CAP,
        help=f"Minimum market cap in dollars (default: {DEFAULT_MIN_MARKET_CAP:,})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    print(f"Fetching NYSE/NASDAQ/AMEX stocks with market cap > ${args.min_market_cap:,} from FMP...")
    stocks = fetch_screener(args.min_market_cap)
    print(f"Found {len(stocks)} stocks (ETFs/funds excluded)")

    universes = build_universes(stocks)
    print(f"Grouped into {len(universes)} industry universes")

    if args.dry_run:
        print("\nDry run — would create:")
    else:
        print(f"\nWriting to {UNIVERSES_DIR}/")

    total = write_universes(universes, dry_run=args.dry_run)
    print(f"\nTotal: {total} tickers across {len(universes)} universes")


if __name__ == "__main__":
    main()
