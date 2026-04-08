"""
SEC EDGAR API client.

Provides access to SEC filing history (10-Q, 10-K) for coverage gap detection.
No API key required — only a User-Agent header per SEC policy.

Uses the free SEC EDGAR endpoints:
- https://www.sec.gov/files/company_tickers.json  (ticker -> CIK mapping)
- https://data.sec.gov/submissions/CIK{padded_cik}.json  (filing history)

Rate limit: max 10 requests/second per SEC fair-access policy.
"""

import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "edgar"

# Cache TTLs in seconds
DEFAULT_TTLS = {
    "ticker_map": 604800,    # 7 days — ticker-to-CIK mapping changes rarely
    "submissions": 86400,    # 24 hours — filing history
    "companyfacts": 86400,   # 24 hours — XBRL fact history
}

# us-gaap tag fallback chains for capital-action facts. Companies tag these
# inconsistently, so we walk the chain in order and take the first one with
# usable annual data.
BUYBACK_TAGS = [
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsForRepurchaseOfEquity",
    "StockRepurchasedAndRetiredDuringPeriodValue",
    "TreasuryStockValueAcquiredCostMethod",
]
DEBT_REPAYMENT_TAGS = [
    "RepaymentsOfLongTermDebt",
    "RepaymentsOfDebt",
    "RepaymentsOfNotesPayable",
    "RepaymentsOfLongTermDebtAndCapitalSecurities",
]
DEBT_ISSUANCE_TAGS = [
    "ProceedsFromIssuanceOfLongTermDebt",
    "ProceedsFromIssuanceOfDebt",
    "ProceedsFromNotesPayable",
    "ProceedsFromLongTermDebt",
]

USER_AGENT = "StockScreener/1.0 contact@example.com"


class EDGARClient:
    """Client for the SEC EDGAR API."""

    def __init__(self, rate_delay: float = 0.12, cache_ttls: dict | None = None):
        """
        Initialize the EDGAR client.

        Args:
            rate_delay: Minimum seconds between API requests (0.1s = 10 req/s max).
            cache_ttls: Dict of cache category -> TTL in seconds.
        """
        self.rate_delay = rate_delay
        self.cache_ttls = {**DEFAULT_TTLS, **(cache_ttls or {})}
        self._last_request_time = 0.0
        self._ticker_map: dict | None = None

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        """EDGAR requires no API key — always configured."""
        return True

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_delay:
            time.sleep(self.rate_delay - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, identifier: str) -> str:
        """Generate a unique cache key."""
        return hashlib.md5(identifier.encode()).hexdigest()

    def _read_cache(self, cache_key: str, category: str):
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
            logger.warning(f"Failed to write EDGAR cache: {e}")

    def _make_request(self, url: str, cache_key: str | None = None,
                      cache_category: str = "submissions",
                      max_retries: int = 3):
        """
        Make a GET request to SEC EDGAR with rate limiting and retries.

        Args:
            url: Full URL to request.
            cache_key: Optional cache key. If provided, checks/writes cache.
            cache_category: Cache TTL category.
            max_retries: Maximum retry attempts.

        Returns:
            Parsed JSON response or None on failure.
        """
        # Check cache first
        if cache_key:
            cached = self._read_cache(cache_key, cache_category)
            if cached is not None:
                logger.debug(f"EDGAR cache hit for {url}")
                return cached

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if cache_key:
                        self._write_cache(cache_key, data)
                    return data
                elif resp.status_code == 429:
                    wait = (2 ** attempt) * 1.0
                    logger.warning(f"EDGAR rate limited (429). Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    wait = (2 ** attempt) * 0.5
                    logger.warning(
                        f"EDGAR server error {resp.status_code}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"EDGAR API error {resp.status_code}: {resp.text[:200]}"
                    )
                    return None
            except requests.RequestException as e:
                logger.error(f"EDGAR request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error(f"Max retries exceeded for {url}")
        return None

    # ------------------------------------------------------------------ #
    # Ticker -> CIK mapping
    # ------------------------------------------------------------------ #

    def _load_ticker_map(self) -> dict:
        """
        Load the SEC ticker-to-CIK mapping.

        Returns:
            Dict mapping uppercase ticker -> CIK integer.
        """
        if self._ticker_map is not None:
            return self._ticker_map

        ck = self._cache_key("company_tickers_json")
        data = self._make_request(
            "https://www.sec.gov/files/company_tickers.json",
            cache_key=ck,
            cache_category="ticker_map",
        )

        ticker_map = {}
        if data:
            for entry in data.values():
                ticker_map[entry["ticker"].upper()] = entry["cik_str"]

        self._ticker_map = ticker_map
        return ticker_map

    def get_cik(self, ticker: str) -> int | None:
        """
        Get the CIK number for a ticker symbol.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            CIK number or None if not found.
        """
        ticker_map = self._load_ticker_map()
        cik = ticker_map.get(ticker.upper())
        if cik is not None:
            return int(cik)
        return None

    # ------------------------------------------------------------------ #
    # Filing history
    # ------------------------------------------------------------------ #

    def get_filing_history(self, ticker: str) -> list[dict]:
        """
        Get 10-Q and 10-K filing history for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of dicts with keys: form, filingDate, reportDate, accessionNumber,
            primaryDocument. Ordered most recent first.
        """
        cik = self.get_cik(ticker)
        if cik is None:
            logger.warning(f"No CIK found for ticker {ticker}")
            return []

        padded_cik = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        ck = self._cache_key(f"submissions_{padded_cik}")

        data = self._make_request(url, cache_key=ck, cache_category="submissions")
        if not data:
            return []

        # Extract recent filings from the main object
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])

        filings = []
        for i in range(len(forms)):
            form_type = forms[i] if i < len(forms) else ""
            if form_type in ("10-Q", "10-K", "20-F"):
                filings.append({
                    "form": form_type,
                    "filingDate": filing_dates[i] if i < len(filing_dates) else "",
                    "reportDate": report_dates[i] if i < len(report_dates) else "",
                    "accessionNumber": accession_numbers[i] if i < len(accession_numbers) else "",
                    "primaryDocument": primary_documents[i] if i < len(primary_documents) else "",
                })

        return filings

    def get_filing_quarters(self, ticker: str, limit: int = 12) -> list[dict]:
        """
        Get SEC filing quarters with fiscal quarter/year mapping.

        Maps each 10-Q/10-K filing to an approximate fiscal quarter based on
        the report date. Returns the most recent filings up to limit.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of filings to return.

        Returns:
            List of dicts with: form, filingDate, reportDate, fiscalYear,
            fiscalQuarter, period (e.g. "Q3 FY2026").
        """
        filings = self.get_filing_history(ticker)
        if not filings:
            return []

        results = []
        for filing in filings[:limit]:
            report_date = filing.get("reportDate", "")
            filing_date = filing.get("filingDate", "")
            form_type = filing.get("form", "")

            if not report_date:
                continue

            try:
                rd = datetime.strptime(report_date, "%Y-%m-%d")
            except ValueError:
                continue

            # Map report date month to fiscal quarter
            # 10-K covers a full year; we label it as Q4
            if form_type == "10-K":
                fiscal_quarter = 4
                fiscal_year = rd.year
            else:
                # Map month to quarter: Jan-Mar=Q1, Apr-Jun=Q2, Jul-Sep=Q3, Oct-Dec=Q4
                # But SEC report dates are period-ending, so:
                # Mar -> Q1, Jun -> Q2, Sep -> Q3, Dec -> Q4
                month = rd.month
                if month <= 3:
                    fiscal_quarter = 1
                elif month <= 6:
                    fiscal_quarter = 2
                elif month <= 9:
                    fiscal_quarter = 3
                else:
                    fiscal_quarter = 4
                fiscal_year = rd.year

            period = f"Q{fiscal_quarter} FY{fiscal_year}"
            if form_type == "10-K":
                period = f"FY{fiscal_year} (10-K)"

            results.append({
                "form": form_type,
                "filingDate": filing_date,
                "reportDate": report_date,
                "fiscalYear": fiscal_year,
                "fiscalQuarter": fiscal_quarter,
                "period": period,
            })

        return results

    def get_recent_filing_links(self, ticker: str, limit: int = 8) -> list[dict]:
        """
        Get direct EDGAR links for recent 10-K and 10-Q filings.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of filings to return.

        Returns:
            List of dicts with: form, filingDate, reportDate, url, period.
        """
        cik = self.get_cik(ticker)
        if cik is None:
            return []

        filings = self.get_filing_history(ticker)
        if not filings:
            return []

        results = []
        for filing in filings[:limit]:
            accession = filing.get("accessionNumber", "")
            primary_doc = filing.get("primaryDocument", "")
            form_type = filing.get("form", "")
            filing_date = filing.get("filingDate", "")
            report_date = filing.get("reportDate", "")

            if not accession or not primary_doc:
                continue

            # Build EDGAR URL: accession number without dashes for the path
            accession_no_dashes = accession.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{accession_no_dashes}/{primary_doc}"
            )

            # Build period label
            try:
                rd = datetime.strptime(report_date, "%Y-%m-%d")
                if form_type == "10-K":
                    period = f"FY {rd.year}"
                elif form_type == "20-F":
                    period = f"FY {rd.year}"
                else:
                    month = rd.month
                    q = (month - 1) // 3 + 1
                    period = f"Q{q} {rd.year}"
            except (ValueError, AttributeError):
                period = form_type

            results.append({
                "form": form_type,
                "filingDate": filing_date,
                "reportDate": report_date,
                "url": url,
                "period": period,
            })

        return results

    # ------------------------------------------------------------------ #
    # Company facts (XBRL) — capital actions verification
    # ------------------------------------------------------------------ #

    def get_company_facts(self, ticker: str) -> dict | None:
        """
        Fetch the full XBRL companyfacts JSON for a ticker.

        Returns the raw SEC response or None if unavailable. Foreign filers
        (20-F with IFRS tags) often return either no data or a near-empty
        us-gaap section — callers should handle that gracefully.
        """
        cik = self.get_cik(ticker)
        if cik is None:
            return None

        padded_cik = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json"
        ck = self._cache_key(f"companyfacts_{padded_cik}")
        return self._make_request(url, cache_key=ck, cache_category="companyfacts")

    def _extract_annual_fact_series(self, facts_root: dict, tag_chain: list[str],
                                     years: int) -> tuple[list[dict], str | None]:
        """
        Walk a fallback chain of us-gaap tags and return the first one with
        usable annual data.

        Returns:
            (series, source_tag) where series is a list of
            {fiscal_year, value, end_date, filed, accession} dicts ordered
            most-recent-first, and source_tag is the tag actually used (or
            None if no chain entry produced data).
        """
        us_gaap = (facts_root or {}).get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            return [], None

        for tag in tag_chain:
            tag_data = us_gaap.get(tag)
            if not tag_data:
                continue
            usd_units = tag_data.get("units", {}).get("USD", [])
            if not usd_units:
                continue

            # Filter to annual periods only (fp == "FY") and dedup by period
            # end date, keeping the latest-filed entry to capture restatements.
            by_period: dict[str, dict] = {}
            for entry in usd_units:
                if entry.get("fp") != "FY":
                    continue
                form = entry.get("form", "")
                # Only trust 10-K / 10-K/A for annual figures (10-Q "FY" is rare
                # but possible for stub periods — skip those)
                if not form.startswith("10-K"):
                    continue
                end = entry.get("end")
                if not end:
                    continue
                existing = by_period.get(end)
                if existing is None or entry.get("filed", "") > existing.get("filed", ""):
                    by_period[end] = entry

            if not by_period:
                continue

            # Sort by period end date desc, take the most recent N years
            sorted_entries = sorted(by_period.values(),
                                    key=lambda e: e.get("end", ""),
                                    reverse=True)[:years]

            series = []
            for e in sorted_entries:
                end = e.get("end", "")
                try:
                    fiscal_year = int(end[:4])
                except (ValueError, IndexError):
                    continue
                series.append({
                    "fiscal_year": fiscal_year,
                    "value": float(e.get("val", 0)),
                    "end_date": end,
                    "filed": e.get("filed"),
                    "accession": e.get("accn"),
                })

            if series:
                return series, tag

        return [], None

    def get_capital_actions(self, ticker: str, years: int = 5) -> dict:
        """
        Pull buyback, debt repayment, and debt issuance series from XBRL
        cash flow statement tags. Net debt paydown = repayments - issuance.

        This is the verification overlay for the DCF auto-paydown estimate
        (which is currently derived from balance sheet differencing). EDGAR
        cash flow tags catch refinancing activity that balance sheet
        differencing misses.

        Args:
            ticker: Stock ticker symbol.
            years: Number of most-recent annual periods to return.

        Returns:
            Dict with:
              available (bool): True if any cash-flow data was found
              annual_data (list): Most-recent-first list of per-year dicts
                with fiscal_year, buybacks, debt_repayments, debt_issuance,
                net_debt_paydown
              avg_annual_buybacks (float | None): mean across available years
              avg_annual_net_paydown (float | None): mean across available years
              source_tags (dict): which us-gaap tag was actually used for each
              note (str | None): human-readable note for foreign filers / gaps
        """
        result = {
            "available": False,
            "annual_data": [],
            "avg_annual_buybacks": None,
            "avg_annual_net_paydown": None,
            "source_tags": {"buybacks": None, "debt_repayments": None, "debt_issuance": None},
            "note": None,
        }

        facts = self.get_company_facts(ticker)
        if not facts:
            result["note"] = (
                "No EDGAR companyfacts data — likely a foreign private issuer "
                "(20-F filer with IFRS tags) or a ticker not in the SEC ticker map."
            )
            return result

        buyback_series, buyback_tag = self._extract_annual_fact_series(
            facts, BUYBACK_TAGS, years)
        repay_series, repay_tag = self._extract_annual_fact_series(
            facts, DEBT_REPAYMENT_TAGS, years)
        issue_series, issue_tag = self._extract_annual_fact_series(
            facts, DEBT_ISSUANCE_TAGS, years)

        result["source_tags"] = {
            "buybacks": buyback_tag,
            "debt_repayments": repay_tag,
            "debt_issuance": issue_tag,
        }

        if not (buyback_series or repay_series or issue_series):
            result["note"] = (
                "EDGAR returned data but none of the standard us-gaap cash "
                "flow tags were populated. Likely a non-standard filer."
            )
            return result

        # Index each series by fiscal year so we can join them
        def _by_year(series: list[dict]) -> dict[int, float]:
            return {row["fiscal_year"]: row["value"] for row in series}

        buybacks_by_year = _by_year(buyback_series)
        repayments_by_year = _by_year(repay_series)
        issuance_by_year = _by_year(issue_series)

        all_years = sorted(
            set(buybacks_by_year) | set(repayments_by_year) | set(issuance_by_year),
            reverse=True,
        )[:years]

        annual_data = []
        for fy in all_years:
            bb = buybacks_by_year.get(fy)
            rp = repayments_by_year.get(fy)
            iss = issuance_by_year.get(fy)
            net_paydown = None
            if rp is not None:
                # Net = repayments - issuance. If issuance is missing, treat
                # it as zero rather than dropping the year (better than nothing).
                net_paydown = rp - (iss or 0)
            annual_data.append({
                "fiscal_year": fy,
                "buybacks": bb,
                "debt_repayments": rp,
                "debt_issuance": iss,
                "net_debt_paydown": net_paydown,
            })

        # Averages over years where the metric was actually present
        bb_values = [r["buybacks"] for r in annual_data if r["buybacks"] is not None]
        np_values = [r["net_debt_paydown"] for r in annual_data if r["net_debt_paydown"] is not None]

        result["available"] = True
        result["annual_data"] = annual_data
        result["avg_annual_buybacks"] = sum(bb_values) / len(bb_values) if bb_values else None
        result["avg_annual_net_paydown"] = sum(np_values) / len(np_values) if np_values else None
        return result

    # ------------------------------------------------------------------ #

    def find_transcript_gaps(self, ticker: str,
                              fmp_transcript_dates: list[dict],
                              limit: int = 12) -> list[dict]:
        """
        Find SEC filings that don't have corresponding FMP transcripts.

        Args:
            ticker: Stock ticker symbol.
            fmp_transcript_dates: List of dicts with 'year' and 'quarter' keys
                from FMP's get_earning_call_transcript_dates().
            limit: Number of recent filings to check.

        Returns:
            List of dicts for filings missing transcripts, each with:
            form, filingDate, reportDate, fiscalYear, fiscalQuarter, period.
        """
        sec_filings = self.get_filing_quarters(ticker, limit=limit)
        if not sec_filings:
            return []

        # Build set of (year, quarter) pairs covered by FMP transcripts
        fmp_covered = set()
        for t in fmp_transcript_dates:
            fmp_covered.add((t.get("year"), t.get("quarter")))

        gaps = []
        for filing in sec_filings:
            fy = filing["fiscalYear"]
            fq = filing["fiscalQuarter"]
            form = filing["form"]

            # For 10-K, check if Q4 transcript exists
            if form == "10-K":
                if (fy, 4) not in fmp_covered:
                    gaps.append(filing)
            else:
                # For 10-Q, check if that quarter's transcript exists
                if (fy, fq) not in fmp_covered:
                    gaps.append(filing)

        return gaps

    def clear_cache(self):
        """Delete all cached EDGAR responses."""
        count = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} EDGAR cache files.")
        return count
