"""
Core analytical engine for historical ratio analysis and percentile scoring.

Pulls financial ratios from FMP, computes percentile ranks against each
ticker's own history, and generates a composite "undervaluation" score.
"""

import logging
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SCORING_DEFAULTS_PATH = PROJECT_ROOT / "config" / "scoring" / "defaults.json"

# ------------------------------------------------------------------ #
# Mapping from our internal metric names to FMP response field names
# ------------------------------------------------------------------ #
RATIO_FIELD_MAP = {
    "pe_ratio": "priceToEarningsRatio",
    "ps_ratio": "priceToSalesRatio",
    "pb_ratio": "priceToBookRatio",
    "ev_ebitda": "enterpriseValueMultiple",
    # ev_revenue: handled correctly via KEY_METRICS_FIELD_MAP -> evToSales
    "gross_margin": "grossProfitMargin",
    "operating_margin": "operatingProfitMargin",
    "net_margin": "netProfitMargin",
    "debt_to_equity": "debtToEquityRatio",
    "current_ratio": "currentRatio",
    "quick_ratio": "quickRatio",
    "interest_coverage": "interestCoverageRatio",
    "price_to_fcf": "priceToFreeCashFlowRatio",
    "peg_ratio": "priceToEarningsGrowthRatio",
    "debt_to_assets": "debtToAssetsRatio",
}

KEY_METRICS_FIELD_MAP = {
    "ev_ebitda": "evToEBITDA",
    "ev_revenue": "evToSales",
    "fcf_yield": "freeCashFlowYield",
    "earnings_yield": "earningsYield",
    "revenue_per_share": "revenuePerShare",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "roic": "returnOnInvestedCapital",
    "cash_conversion_cycle": "cashConversionCycle",
}


# Valuation ratios that are meaningless when negative (negative earnings, etc.)
# Negative values are excluded from historical series and current is set to None
POSITIVE_ONLY_METRICS = {
    "pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda", "ev_revenue", "price_to_fcf",
    "peg_ratio",
}


def _load_scoring_config() -> dict:
    """Load the scoring configuration from defaults.json."""
    try:
        return json.loads(SCORING_DEFAULTS_PATH.read_text())
    except Exception as e:
        logger.warning(f"Could not load scoring config: {e}. Using built-in defaults.")
        return {
            "weights": {
                "pe_ratio": 0.20, "ps_ratio": 0.15, "pb_ratio": 0.10,
                "ev_ebitda": 0.15, "fcf_yield": 0.15, "roe": 0.10,
                "debt_to_equity": 0.15,
            },
            "higher_is_better": [
                "fcf_yield", "roe", "roa", "gross_margin", "operating_margin",
                "net_margin", "current_ratio", "quick_ratio", "interest_coverage",
                "earnings_yield",
            ],
            "opportunity_thresholds": {"low_percentile": 20, "high_percentile": 80},
            "history_years": 5,
        }


def compute_percentile_rank(current_value: float,
                             historical_values: list[float]) -> float | None:
    """
    Compute the percentile rank of current_value within historical_values.

    Args:
        current_value: The current metric value.
        historical_values: List of historical values for the same metric.

    Returns:
        Percentile rank (0-100) or None if insufficient data.
    """
    if current_value is None or not historical_values:
        return None

    clean = [v for v in historical_values if v is not None and np.isfinite(v)]
    if len(clean) < 2:
        return None

    below = sum(1 for v in clean if v < current_value)
    equal = sum(1 for v in clean if v == current_value)
    # Midpoint method: values equal to current count as half
    percentile = ((below + equal / 2) / len(clean)) * 100.0
    return round(percentile, 1)


def _extract_metric_series(records: list[dict], field_name: str) -> list[float]:
    """Extract a list of values for a given field from a list of period records."""
    values = []
    for rec in records:
        val = rec.get(field_name)
        if val is not None:
            try:
                val = float(val)
                if np.isfinite(val):
                    values.append(val)
            except (TypeError, ValueError):
                pass
    return values


def _trimmed_mean(values: list[float], trim_pct: float = 0.1) -> float | None:
    """Compute mean after trimming top and bottom `trim_pct` of values (IQR-aware)."""
    if not values:
        return None
    arr = sorted(values)
    n = len(arr)
    if n < 4:
        return float(np.median(arr))  # too few values to trim, use median
    trim_count = max(1, int(n * trim_pct))
    trimmed = arr[trim_count: n - trim_count]
    return float(np.mean(trimmed)) if trimmed else float(np.median(arr))


def _build_metric_entry(current: float, clean: list[float], percentile: float) -> dict:
    """Build a metric dict with mean, median, trimmed mean, and range stats."""
    return {
        "current": round(current, 4),
        "hist_avg": round(np.mean(clean), 4) if clean else None,
        "hist_median": round(float(np.median(clean)), 4) if clean else None,
        "hist_trimmed_avg": round(_trimmed_mean(clean), 4) if clean else None,
        "hist_low": round(min(clean), 4) if clean else None,
        "hist_high": round(max(clean), 4) if clean else None,
        "percentile": percentile,
        "years_of_data": len(clean),
    }


def _safe_float(val) -> float | None:
    """Convert to float safely, returning None for invalid values."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _compute_growth(values: list[float]) -> float | None:
    """Compute YoY growth from the two most recent values (index 0 is most recent)."""
    if len(values) < 2:
        return None
    current, prior = values[0], values[1]
    if prior == 0 or prior is None:
        return None
    return round((current - prior) / abs(prior) * 100, 2)


def _compute_cagr(values: list[float], years: int) -> float | None:
    """Compute CAGR from oldest to newest value. values[0] is most recent."""
    if len(values) < 2 or years <= 0:
        return None
    end, start = values[0], values[-1]
    if start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1.0 / years) - 1) * 100, 2)


def _compute_trend(values: list[float]) -> str:
    """Determine trend from a series of values (most recent first).
    Returns 'growing', 'declining', 'volatile', or 'insufficient_data'."""
    if len(values) < 2:
        return "insufficient_data"
    changes = []
    for i in range(len(values) - 1):
        if values[i + 1] != 0:
            changes.append(values[i] - values[i + 1])
    if not changes:
        return "insufficient_data"
    up = sum(1 for c in changes if c > 0)
    down = sum(1 for c in changes if c < 0)
    if up >= len(changes) * 0.7:
        return "growing"
    elif down >= len(changes) * 0.7:
        return "declining"
    return "volatile"


def reconcile_capital_actions(edgar_data: dict,
                               fmp_auto_debt_paydown: float | None,
                               fmp_hist_share_change_pct: float | None,
                               shares_outstanding: float | None) -> dict:
    """
    Cross-check the FMP balance-sheet auto-estimates against EDGAR XBRL cash
    flow statement values for debt paydown and buybacks.

    The FMP auto-estimate is derived from year-over-year balance sheet
    differencing, which has a structural blind spot for refinancing activity:
    a company that repays $5B and issues $5B of new debt shows zero balance
    sheet change but had $5B of cash flow paydown. EDGAR cash flow tags
    (RepaymentsOfLongTermDebt - ProceedsFromIssuanceOfLongTermDebt) catch this.

    Args:
        edgar_data: Output of EDGARClient.get_capital_actions(ticker).
        fmp_auto_debt_paydown: Annual debt paydown ($) from balance sheet
            differencing — same value the DCF currently uses as auto-default.
        fmp_hist_share_change_pct: Historical annual share count change as
            decimal (e.g. -0.03 for 3% buyback shrinkage).
        shares_outstanding: Current diluted shares (for converting EDGAR
            buyback dollars into an implied % share change).

    Returns:
        Dict with comparison rows and a verdict for each metric:
          debt_paydown: {fmp, edgar, delta_pct, agreement, note}
          buybacks: {fmp_implied_pct, edgar_dollars, edgar_implied_pct, agreement, note}
          available: bool — False if EDGAR returned no usable data
    """
    result = {
        "available": bool(edgar_data and edgar_data.get("available")),
        "debt_paydown": None,
        "buybacks": None,
    }
    if not result["available"]:
        result["note"] = (edgar_data or {}).get("note") if edgar_data else "No EDGAR data"
        return result

    # ---- Debt paydown reconciliation ----
    edgar_paydown = edgar_data.get("avg_annual_net_paydown")
    if edgar_paydown is not None and fmp_auto_debt_paydown is not None:
        # Sign-aware comparison: a sign mismatch (one says paydown, other says
        # net issuance) is the most important thing to flag.
        sign_match = (edgar_paydown >= 0) == (fmp_auto_debt_paydown >= 0)
        # Use max of abs values as denominator to avoid div-by-zero
        scale = max(abs(edgar_paydown), abs(fmp_auto_debt_paydown), 1.0)
        delta_pct = (edgar_paydown - fmp_auto_debt_paydown) / scale * 100

        if not sign_match:
            agreement = "mismatch"
            note = (
                "Sign disagreement: one method says paying down debt, the other "
                "says taking on debt. EDGAR (cash flow statement) is the more "
                "reliable signal — refinancing activity often hides this on "
                "the balance sheet."
            )
        elif abs(delta_pct) <= 30:
            agreement = "agree"
            note = "Methods agree within 30% — auto-estimate is reliable."
        elif abs(delta_pct) <= 100:
            agreement = "caution"
            note = (
                "Methods disagree by 30-100%. Likely refinancing activity that "
                "balance sheet differencing partially misses. Consider EDGAR value."
            )
        else:
            agreement = "major"
            note = (
                "Methods disagree by >100%. Heavy refinancing or non-standard "
                "debt structure. EDGAR cash flow value is more reliable."
            )

        result["debt_paydown"] = {
            "fmp": fmp_auto_debt_paydown,
            "edgar": edgar_paydown,
            "delta_pct": round(delta_pct, 1),
            "agreement": agreement,
            "note": note,
        }
    elif edgar_paydown is not None:
        result["debt_paydown"] = {
            "fmp": None,
            "edgar": edgar_paydown,
            "delta_pct": None,
            "agreement": "edgar_only",
            "note": "Balance sheet auto-estimate unavailable; EDGAR is the only source.",
        }

    # ---- Buyback reconciliation ----
    edgar_buybacks = edgar_data.get("avg_annual_buybacks")
    if edgar_buybacks is not None:
        # Convert FMP's % share change into an approximate dollar buyback
        # so the comparison is apples-to-apples. We can't use price directly
        # (that's what we're valuing), so we just report both: FMP says "X%
        # share count change", EDGAR says "$Y buybacks". Caller can show both.
        edgar_implied_pct = None
        if shares_outstanding and shares_outstanding > 0 and edgar_buybacks > 0:
            # Rough proxy: avg buyback $ / (shares × ~$avg_price). We don't
            # have avg historical price here, so we just leave this as a
            # display hint and let the UI compute it from current price.
            pass

        agreement = None
        note = None
        if fmp_hist_share_change_pct is not None:
            # FMP says shares are shrinking → company is buying back.
            # EDGAR says how much they spent on buybacks.
            if fmp_hist_share_change_pct < -0.01 and edgar_buybacks > 0:
                agreement = "agree"
                note = (
                    "Both methods confirm active buybacks: FMP shows share count "
                    "shrinking, EDGAR shows real cash spent on repurchases."
                )
            elif fmp_hist_share_change_pct > 0.01 and edgar_buybacks > 0:
                agreement = "caution"
                note = (
                    "Disagreement: share count is GROWING (likely from stock comp "
                    "or secondary issuance) despite material buyback spending. "
                    "Net dilution = SBC outpacing repurchases. The FMP share-count "
                    "trend is the more useful signal for per-share DCF math."
                )
            elif edgar_buybacks == 0:
                agreement = "agree"
                note = "No buyback activity per either source."
            else:
                agreement = "agree"
                note = "Both methods broadly consistent."

        result["buybacks"] = {
            "fmp_share_change_pct": fmp_hist_share_change_pct,
            "edgar_dollars": edgar_buybacks,
            "edgar_implied_pct": edgar_implied_pct,
            "agreement": agreement,
            "note": note,
        }

    return result


def compute_analyst_accuracy(analyst_estimates: list[dict],
                             income_statements: list[dict]) -> dict | None:
    """Compare analyst consensus estimates to actual results for completed fiscal years.

    Returns a dict with per-year matches, average surprise %, beat rates,
    and an overall reliability assessment. Returns None if insufficient data.
    """
    if not analyst_estimates or not income_statements:
        return None

    from datetime import datetime as _dt

    # Build lookup of actuals by fiscal year
    actuals_by_year = {}
    for stmt in income_statements:
        d = stmt.get("date", "")
        if not d:
            continue
        try:
            year = int(d[:4])
        except (ValueError, TypeError):
            continue
        actuals_by_year[year] = {
            "eps": _safe_float(stmt.get("epsDiluted") or stmt.get("eps")),
            "revenue": _safe_float(stmt.get("revenue")),
        }

    today = _dt.now().date()
    matches = []

    for est in analyst_estimates:
        d = est.get("date", "")
        if not d:
            continue
        try:
            est_date = _dt.strptime(d[:10], "%Y-%m-%d").date()
            year = est_date.year
        except (ValueError, TypeError):
            continue

        # Only compare completed periods (estimate date has passed)
        if est_date > today:
            continue

        if year not in actuals_by_year:
            continue

        actual = actuals_by_year[year]
        est_eps = _safe_float(est.get("epsAvg"))
        est_rev = _safe_float(est.get("revenueAvg"))
        act_eps = actual["eps"]
        act_rev = actual["revenue"]

        row = {"year": year}

        # EPS surprise
        if est_eps is not None and act_eps is not None and abs(est_eps) > 0.01:
            eps_surprise = ((act_eps - est_eps) / abs(est_eps)) * 100
            row["est_eps"] = round(est_eps, 2)
            row["act_eps"] = round(act_eps, 2)
            row["eps_surprise"] = round(eps_surprise, 1)
        else:
            row["est_eps"] = est_eps
            row["act_eps"] = act_eps
            row["eps_surprise"] = None

        # Revenue surprise
        if est_rev is not None and act_rev is not None and est_rev > 0:
            rev_surprise = ((act_rev - est_rev) / est_rev) * 100
            row["est_rev"] = round(est_rev, 0)
            row["act_rev"] = round(act_rev, 0)
            row["rev_surprise"] = round(rev_surprise, 1)
        else:
            row["est_rev"] = est_rev
            row["act_rev"] = act_rev
            row["rev_surprise"] = None

        matches.append(row)

    if not matches:
        return None

    # Sort by year descending (most recent first)
    matches.sort(key=lambda x: x["year"], reverse=True)

    # Compute summary stats
    eps_surprises = [m["eps_surprise"] for m in matches if m["eps_surprise"] is not None]
    rev_surprises = [m["rev_surprise"] for m in matches if m["rev_surprise"] is not None]

    result = {"matches": matches}

    if eps_surprises:
        result["avg_eps_surprise"] = round(sum(eps_surprises) / len(eps_surprises), 1)
        result["avg_abs_eps_miss"] = round(sum(abs(s) for s in eps_surprises) / len(eps_surprises), 1)
        result["eps_beat_count"] = sum(1 for s in eps_surprises if s > 0)
        result["eps_total"] = len(eps_surprises)
    if rev_surprises:
        result["avg_rev_surprise"] = round(sum(rev_surprises) / len(rev_surprises), 1)
        result["avg_abs_rev_miss"] = round(sum(abs(s) for s in rev_surprises) / len(rev_surprises), 1)
        result["rev_beat_count"] = sum(1 for s in rev_surprises if s > 0)
        result["rev_total"] = len(rev_surprises)

    # Overall reliability based on average absolute EPS miss
    avg_abs = result.get("avg_abs_eps_miss", 0)
    if avg_abs <= 10:
        result["reliability"] = "High"
    elif avg_abs <= 20:
        result["reliability"] = "Moderate"
    else:
        result["reliability"] = "Low"

    return result


def build_fundamentals_context(income_statements: list[dict],
                               cash_flow_statements: list[dict] | None = None,
                               balance_sheets: list[dict] | None = None,
                               history_years: int = 5) -> dict:
    """
    Analyze earnings, cash flow, and balance sheet trends to distinguish
    genuinely cheap stocks from value traps.

    Args:
        income_statements: List of income statement dicts (most recent first).
        cash_flow_statements: List of cash flow statement dicts (most recent first).
        balance_sheets: List of balance sheet dicts (most recent first).
        history_years: Number of years of history being analyzed.

    Returns:
        Dict with fundamentals context fields for earnings, cash flow, and leverage.
    """
    result = {
        # Earnings
        "eps_history": [],
        "eps_trend": "insufficient_data",
        "eps_yoy_change": None,
        "eps_3yr_cagr": None,
        "revenue_trend": "insufficient_data",
        "revenue_yoy_change": None,
        "net_income_positive_years": 0,
        "total_years": 0,
        # Cash flow
        "fcf_trend": "insufficient_data",
        "fcf_yoy_change": None,
        "fcf_vs_net_income": None,  # "healthy", "diverging", or "negative_fcf"
        "operating_cf_yoy_change": None,
        # Balance sheet / leverage
        "net_debt_to_ebitda": None,
        "net_cash_position": None,  # True if net cash, False if net debt
        # Flags
        "context_flags": [],
    }

    cash_flow_statements = cash_flow_statements or []
    balance_sheets = balance_sheets or []

    if not income_statements or len(income_statements) < 2:
        return result

    # ---- Earnings analysis (existing logic) ----
    eps_data = []
    revenue_data = []
    net_income_data = []
    for stmt in income_statements:
        year = stmt.get("calendarYear") or (stmt.get("date", "")[:4])
        eps = _safe_float(stmt.get("epsDiluted") or stmt.get("eps"))
        rev = _safe_float(stmt.get("revenue"))
        ni = _safe_float(stmt.get("netIncome"))
        eps_data.append({"year": year, "eps": eps})
        if rev is not None:
            revenue_data.append(rev)
        if ni is not None:
            net_income_data.append(ni)

    result["eps_history"] = eps_data
    result["total_years"] = len(eps_data)
    result["net_income_positive_years"] = sum(1 for ni in net_income_data if ni > 0)

    eps_values = [d["eps"] for d in eps_data if d["eps"] is not None]
    if len(eps_values) >= 2:
        result["eps_yoy_change"] = _compute_growth(eps_values)
        if len(eps_values) >= 4:
            result["eps_3yr_cagr"] = _compute_cagr(eps_values[:4], 3)
        result["eps_trend"] = _compute_trend(eps_values)

    if len(revenue_data) >= 2:
        result["revenue_yoy_change"] = _compute_growth(revenue_data)
        result["revenue_trend"] = _compute_trend(revenue_data)

    # ---- Cash flow analysis ----
    fcf_data = _extract_metric_series(cash_flow_statements, "freeCashFlow")
    ocf_data = _extract_metric_series(cash_flow_statements, "operatingCashFlow")

    if len(fcf_data) >= 2:
        result["fcf_yoy_change"] = _compute_growth(fcf_data)
        result["fcf_trend"] = _compute_trend(fcf_data)

    if len(ocf_data) >= 2:
        result["operating_cf_yoy_change"] = _compute_growth(ocf_data)

    # FCF vs net income comparison (most recent year)
    if fcf_data and net_income_data:
        latest_fcf = fcf_data[0]
        latest_ni = net_income_data[0]
        if latest_fcf <= 0:
            result["fcf_vs_net_income"] = "negative_fcf"
        elif latest_ni > 0 and latest_fcf >= latest_ni * 0.5:
            result["fcf_vs_net_income"] = "healthy"
        else:
            result["fcf_vs_net_income"] = "diverging"

    # ---- Balance sheet / leverage ----
    if balance_sheets:
        latest_bs = balance_sheets[0]
        total_debt = _safe_float(latest_bs.get("totalDebt") or latest_bs.get("longTermDebt"))
        cash = _safe_float(latest_bs.get("cashAndCashEquivalents") or
                           latest_bs.get("cashAndShortTermInvestments"))

        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
            result["net_cash_position"] = net_debt <= 0

            # Net debt / EBITDA from income statement
            if income_statements:
                ebitda = _safe_float(income_statements[0].get("ebitda"))
                if ebitda and ebitda > 0:
                    result["net_debt_to_ebitda"] = round(net_debt / ebitda, 2)

    # ---- Generate context flags ----
    flags = []
    eps_yoy = result["eps_yoy_change"]
    eps_trend = result["eps_trend"]
    rev_trend = result["revenue_trend"]
    fcf_trend = result["fcf_trend"]
    positive_yrs = result["net_income_positive_years"]
    total_yrs = result["total_years"]

    # Earnings flags
    if eps_trend == "declining":
        flags.append("Earnings declining -- low P/E may reflect deteriorating fundamentals")
    elif eps_trend == "growing" and eps_yoy is not None and eps_yoy > 0:
        flags.append(f"Earnings growing (EPS +{eps_yoy:.1f}% YoY) -- low valuation may be genuine opportunity")

    if eps_yoy is not None and eps_yoy < -20:
        flags.append(f"EPS dropped {eps_yoy:.1f}% YoY -- check if temporary or structural")
    elif eps_yoy is not None and eps_yoy > 50:
        flags.append(f"EPS surged +{eps_yoy:.1f}% YoY -- check if sustainable or one-time")

    if total_yrs >= 3 and positive_yrs < total_yrs * 0.5:
        flags.append(f"Profitable only {positive_yrs} of {total_yrs} years -- inconsistent earner")

    if rev_trend == "declining" and eps_trend != "declining":
        flags.append("Revenue declining while earnings hold -- check margin sustainability")
    elif rev_trend == "declining" and eps_trend == "declining":
        flags.append("Both revenue and earnings declining -- potential structural issue")

    cagr = result["eps_3yr_cagr"]
    if cagr is not None:
        if cagr > 10:
            flags.append(f"3-year EPS CAGR of +{cagr:.1f}% supports valuation")
        elif cagr < -10:
            flags.append(f"3-year EPS CAGR of {cagr:.1f}% -- long-term earnings erosion")

    # Cash flow flags
    if fcf_trend == "declining" and eps_trend == "growing":
        flags.append("FCF declining while earnings grow -- potential earnings quality issue")
    elif fcf_trend == "declining" and eps_trend == "declining":
        flags.append("Both FCF and earnings declining -- cash generation weakening")
    elif fcf_trend == "growing":
        fcf_yoy = result["fcf_yoy_change"]
        if fcf_yoy is not None and fcf_yoy > 0:
            flags.append(f"FCF growing (+{fcf_yoy:.1f}% YoY) -- strong cash generation")

    if result["fcf_vs_net_income"] == "negative_fcf":
        flags.append("Negative free cash flow -- company burning cash despite reported earnings")
    elif result["fcf_vs_net_income"] == "diverging":
        flags.append("FCF significantly lagging net income -- check earnings quality")

    # Leverage flags
    nd_ebitda = result["net_debt_to_ebitda"]
    if result["net_cash_position"] is True:
        flags.append("Net cash position -- no net debt")
    elif nd_ebitda is not None:
        if nd_ebitda > 4:
            flags.append(f"Net Debt/EBITDA at {nd_ebitda:.1f}x -- highly leveraged")
        elif nd_ebitda > 2.5:
            flags.append(f"Net Debt/EBITDA at {nd_ebitda:.1f}x -- moderate leverage")
        elif nd_ebitda >= 0:
            flags.append(f"Net Debt/EBITDA at {nd_ebitda:.1f}x -- conservative leverage")

    result["context_flags"] = flags
    return result


def compute_implied_prices(metrics: dict, current_price: float | None,
                           income_statements: list[dict],
                           cash_flow_statements: list[dict],
                           key_metrics_data: list[dict],
                           fundamentals_context: dict) -> dict:
    """
    Derive implied stock prices from mean reversion of historical valuation ratios.

    For each valuation ratio with historical data, computes:
        implied_price = hist_avg_ratio × current_fundamental_per_share

    Cross-references with fundamentals context to flag unreliable estimates.

    Returns:
        Dict with 'valuations' list, 'median_implied_price', 'current_price',
        'upside_pct', and 'warnings'.
    """
    valuations = []
    warnings = []

    if not current_price or current_price <= 0:
        return {"valuations": [], "median_implied_price": None,
                "current_price": current_price, "upside_pct": None, "warnings": ["No current price available"]}

    # Get current fundamentals from most recent statements
    current_eps = None
    current_revenue_per_share = None
    current_bvps = None
    current_fcf_per_share = None
    current_ebitda_per_share = None

    if income_statements:
        current_eps = _safe_float(income_statements[0].get("epsDiluted") or income_statements[0].get("eps"))
        shares = _safe_float(income_statements[0].get("weightedAverageShsOutDil")
                             or income_statements[0].get("weightedAverageShsOut"))
        ebitda = _safe_float(income_statements[0].get("ebitda"))
        if shares and shares > 0 and ebitda:
            current_ebitda_per_share = ebitda / shares

    if key_metrics_data:
        current_revenue_per_share = _safe_float(key_metrics_data[0].get("revenuePerShare"))
        current_bvps = _safe_float(key_metrics_data[0].get("bookValuePerShare"))

    if cash_flow_statements and income_statements:
        fcf = _safe_float(cash_flow_statements[0].get("freeCashFlow"))
        shares = _safe_float(income_statements[0].get("weightedAverageShsOutDil")
                             or income_statements[0].get("weightedAverageShsOut"))
        if fcf is not None and shares and shares > 0:
            current_fcf_per_share = fcf / shares

    # Define which ratios map to which fundamentals
    ratio_configs = [
        ("pe_ratio", "P/E", current_eps, "eps_trend"),
        ("ps_ratio", "P/S", current_revenue_per_share, "revenue_trend"),
        ("pb_ratio", "P/B", current_bvps, None),
        ("price_to_fcf", "P/FCF", current_fcf_per_share, "fcf_trend"),
    ]

    # EV/EBITDA needs special handling (enterprise value, not equity price)
    # implied EV = hist_avg EV/EBITDA × current EBITDA, then back into price
    # Skip for now — use equity-based ratios only for clean implied prices

    eps_trend = fundamentals_context.get("eps_trend", "insufficient_data")
    fcf_trend = fundamentals_context.get("fcf_trend", "insufficient_data")
    fcf_quality = fundamentals_context.get("fcf_vs_net_income")

    for metric_key, label, fundamental, trend_key in ratio_configs:
        if metric_key not in metrics:
            continue
        # Use trimmed mean (outlier-resistant) for the reversion target
        hist_val = metrics[metric_key].get("hist_trimmed_avg") or metrics[metric_key].get("hist_median") or metrics[metric_key].get("hist_avg")
        if hist_val is None or fundamental is None or fundamental <= 0:
            continue

        implied = round(hist_val * fundamental, 2)
        upside = round((implied / current_price - 1) * 100, 1)

        entry = {
            "method": f"{label} Mean Reversion",
            "formula": f"Trimmed Avg {label} ({hist_val:.2f}) × Current {label.split('/')[1] if '/' in label else 'fundamental'}/share (${fundamental:.2f})",
            "implied_price": implied,
            "upside_pct": upside,
            "confidence": "normal",
            "warning": None,
        }

        # Annotate confidence based on fundamentals context
        if metric_key == "pe_ratio":
            if eps_trend == "declining":
                entry["confidence"] = "low"
                entry["warning"] = "Earnings declining — implied price may be overstated"
            elif eps_trend == "growing":
                entry["confidence"] = "high"
        elif metric_key == "ps_ratio":
            rev_trend = fundamentals_context.get("revenue_trend", "insufficient_data")
            if rev_trend == "declining":
                entry["confidence"] = "low"
                entry["warning"] = "Revenue declining — P/S reversion target may be overstated"
        elif metric_key == "price_to_fcf":
            if fcf_trend == "declining":
                entry["confidence"] = "low"
                entry["warning"] = "FCF declining — implied price may be overstated"
            elif fcf_quality == "negative_fcf":
                entry["confidence"] = "low"
                entry["warning"] = "Company is burning cash — FCF-based target unreliable"
            elif fcf_quality == "diverging":
                entry["confidence"] = "low"
                entry["warning"] = "FCF lagging net income — earnings quality concern"

        # Sanity check LAST — overrides fundamentals confidence when implied is extreme
        ratio_to_current = implied / current_price
        if ratio_to_current > 5:
            entry["confidence"] = "low"
            entry["warning"] = f"Implied price is {ratio_to_current:.0f}× current — historical ratios contain extreme outliers"
        elif ratio_to_current > 3:
            entry["confidence"] = "low"
            entry["warning"] = "Implied price far above current — treat with caution"

        valuations.append(entry)

    # Compute median implied price
    implied_prices = [v["implied_price"] for v in valuations if v["implied_price"] > 0]
    median_price = round(float(np.median(implied_prices)), 2) if implied_prices else None
    median_upside = round((median_price / current_price - 1) * 100, 1) if median_price else None

    # Global warnings from fundamentals context
    if eps_trend == "declining" and fcf_trend == "declining":
        warnings.append("Both earnings and FCF declining — all implied prices should be viewed skeptically")
    if fundamentals_context.get("net_debt_to_ebitda") is not None and fundamentals_context["net_debt_to_ebitda"] > 4:
        warnings.append("Highly leveraged — mean reversion assumes financial stability")

    low_confidence = [v for v in valuations if v["confidence"] == "low"]
    if len(low_confidence) == len(valuations) and valuations:
        warnings.append("All valuation methods flagged low confidence — fundamentals don't support mean reversion thesis")

    return {
        "valuations": valuations,
        "median_implied_price": median_price,
        "current_price": current_price,
        "upside_pct": median_upside,
        "warnings": warnings,
    }


def _compute_nopat_and_roic(income_statements: list[dict],
                             balance_sheets: list[dict]) -> dict | None:
    """
    Compute NOPAT, invested capital, and ROIC for the reinvestment-aware DCF.

    NOPAT = EBIT * (1 - effective tax rate)
    Invested Capital = Total Equity + Total Debt - Cash
    ROIC = NOPAT / Invested Capital (using prior-year invested capital where available)

    Uses median of up to 3 most recent years for NOPAT to smooth one-off items.
    Returns None if data is insufficient.
    """
    if not income_statements or not balance_sheets:
        return None

    # --- NOPAT: median of up to 3 most recent years ---
    nopat_values = []
    tax_rates = []
    for stmt in income_statements[:3]:
        ebit = _safe_float(stmt.get("operatingIncome"))
        if ebit is None:
            continue
        pre_tax = _safe_float(stmt.get("incomeBeforeTax"))
        income_tax = _safe_float(stmt.get("incomeTaxExpense"))
        if pre_tax and pre_tax > 0 and income_tax is not None:
            tr = max(0.0, min(0.40, income_tax / pre_tax))
        else:
            tr = 0.25
        tax_rates.append(tr)
        nopat_values.append(ebit * (1 - tr))

    if not nopat_values:
        return None

    nopat_values_sorted = sorted(nopat_values)
    base_nopat = nopat_values_sorted[len(nopat_values_sorted) // 2]
    effective_tax_rate = sum(tax_rates) / len(tax_rates) if tax_rates else 0.25

    if base_nopat <= 0:
        return None

    # --- Invested capital: use prior-year if available (standard ROIC convention) ---
    def _invested_capital(bs: dict) -> float | None:
        equity = _safe_float(bs.get("totalStockholdersEquity") or bs.get("totalEquity"))
        debt = _safe_float(bs.get("totalDebt") or bs.get("longTermDebt"))
        cash = _safe_float(bs.get("cashAndCashEquivalents")
                           or bs.get("cashAndShortTermInvestments")) or 0.0
        if equity is None or debt is None:
            return None
        ic = equity + debt - cash
        return ic if ic > 0 else None

    ic_latest = _invested_capital(balance_sheets[0])
    ic_prior = _invested_capital(balance_sheets[1]) if len(balance_sheets) > 1 else None

    # Standard convention: use average of beginning + ending invested capital
    if ic_latest and ic_prior:
        invested_capital = (ic_latest + ic_prior) / 2
    elif ic_latest:
        invested_capital = ic_latest
    else:
        return None

    roic = base_nopat / invested_capital
    # Sanity clamp: ROIC above 60% is almost certainly a measurement artifact
    # (e.g., asset-light companies with negligible invested capital)
    roic_capped = min(roic, 0.60)

    return {
        "base_nopat": base_nopat,
        "invested_capital": invested_capital,
        "roic": roic,
        "roic_capped": roic_capped,
        "effective_tax_rate": effective_tax_rate,
    }


def compute_dcf_valuation(cash_flow_statements: list[dict],
                           income_statements: list[dict],
                           profile: dict,
                           balance_sheets: list[dict] | None = None,
                           growth_rate_override: float | None = None,
                           discount_rate_override: float | None = None,
                           terminal_growth_override: float | None = None,
                           projection_years: int = 10,
                           analyst_estimates: list[dict] | None = None,
                           growth_stages: list[dict] | None = None,
                           risk_free_rate: float | None = None,
                           annual_debt_paydown: float | None = None,
                           annual_share_change: float | None = None,
                           use_reinvestment_model: bool = False,
                           use_fade: bool = False,
                           fade_start_year: int = 5,
                           use_mid_year_discounting: bool = True) -> dict:
    """
    Discounted Cash Flow valuation — "what has to be true" model.

    Computes intrinsic value per share from projected free cash flows, plus a
    sensitivity table showing implied prices at various growth × discount rate
    combinations.

    Args:
        cash_flow_statements: Annual cash flow statements (newest first).
        income_statements: Annual income statements (for shares outstanding).
        profile: Company profile dict (for beta, price).
        balance_sheets: Annual balance sheets (newest first) for net debt adjustment.
        growth_rate_override: User-specified FCF growth rate (as decimal, e.g. 0.08).
        discount_rate_override: User-specified discount/WACC rate (as decimal).
        terminal_growth_override: User-specified terminal growth rate (as decimal).
        projection_years: Number of years to project (default 10).
        analyst_estimates: Forward analyst estimates from FMP (newest first).

    Returns:
        Dict with dcf_price, assumptions, projected_fcfs, sensitivity_table, warnings.
    """
    warnings = []

    # --- Extract starting FCF (use average of last 2-3 years to smooth) ---
    fcf_values = []
    for stmt in cash_flow_statements[:5]:  # up to 5 years
        fcf = _safe_float(stmt.get("freeCashFlow"))
        if fcf is not None:
            fcf_values.append(fcf)

    # --- Reinvestment-aware mode (Damodaran): derive FCF from NOPAT × (1 - g/ROIC) ---
    # Computed early so we can fall back gracefully if it fails, and so projections
    # use NOPAT compounding instead of FCF compounding when enabled.
    reinvestment_info = None
    if use_reinvestment_model and balance_sheets:
        reinvestment_info = _compute_nopat_and_roic(income_statements, balance_sheets)
        if reinvestment_info is None:
            warnings.append(
                "Reinvestment-aware mode requested but NOPAT/ROIC could not be computed "
                "-- falling back to reported FCF base"
            )

    if len(fcf_values) < 2 and not reinvestment_info:
        return {"dcf_price": None, "warnings": ["Insufficient FCF history for DCF"],
                "assumptions": {}, "has_data": False}

    if fcf_values:
        # Use weighted median of up to 5 years as base (recent years weighted more heavily)
        # Weights: most recent = 5, next = 4, ..., oldest = 1
        n = len(fcf_values)
        weights = list(range(n, 0, -1))  # e.g. [5, 4, 3, 2, 1] for 5 years
        # Weighted median: sort by value, pick the value where cumulative weight crosses 50%
        paired = sorted(zip(fcf_values, weights), key=lambda x: x[0])
        total_weight = sum(weights)
        cumulative = 0
        base_fcf = paired[len(paired) // 2][0]  # fallback to simple median
        for val, w in paired:
            cumulative += w
            if cumulative >= total_weight / 2:
                base_fcf = val
                break

        if base_fcf <= 0:
            # Fallback: try simple mean of all available years
            base_fcf = np.mean(fcf_values)
            if base_fcf <= 0 and not reinvestment_info:
                return {"dcf_price": None, "warnings": ["Negative weighted median FCF — DCF not applicable"],
                        "assumptions": {}, "has_data": False}
            if base_fcf <= 0:
                # Reinvestment mode will replace base_fcf using NOPAT below
                base_fcf = reinvestment_info["base_nopat"]
            else:
                warnings.append("Weighted median FCF negative — using mean of all years as base")
    else:
        # No FCF history at all -- only reachable in reinvestment mode
        base_fcf = reinvestment_info["base_nopat"]

    # --- Shares outstanding ---
    shares = None
    if income_statements:
        shares = _safe_float(
            income_statements[0].get("weightedAverageShsOutDil")
            or income_statements[0].get("weightedAverageShsOut")
        )
    if not shares or shares <= 0:
        return {"dcf_price": None, "warnings": ["No shares outstanding data"],
                "assumptions": {}, "has_data": False}

    # --- Historical share count change (buybacks = negative, dilution = positive) ---
    hist_share_change = None
    share_count_values = []
    for stmt in income_statements[:5]:
        s = _safe_float(stmt.get("weightedAverageShsOutDil") or stmt.get("weightedAverageShsOut"))
        if s and s > 0:
            share_count_values.append(s)
    if len(share_count_values) >= 2:
        hist_share_change = (share_count_values[0] / share_count_values[-1]) ** (1 / (len(share_count_values) - 1)) - 1

    resolved_share_change = annual_share_change if annual_share_change is not None else (hist_share_change or 0.0)
    resolved_share_change = max(-0.20, min(0.20, resolved_share_change))
    terminal_shares = max(shares * (1 + resolved_share_change) ** projection_years, shares * 0.01)

    # --- Net debt (total debt - cash) for EV → equity bridge ---
    net_debt = 0.0  # default: assume no net debt if data unavailable
    balance_sheets = balance_sheets or []
    auto_debt_paydown = None  # auto-estimated from historical balance sheets
    if balance_sheets:
        latest_bs = balance_sheets[0]
        total_debt = _safe_float(latest_bs.get("totalDebt") or latest_bs.get("longTermDebt"))
        cash = _safe_float(latest_bs.get("cashAndCashEquivalents") or
                           latest_bs.get("cashAndShortTermInvestments"))
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash  # positive = net debt, negative = net cash
        else:
            warnings.append("Missing debt/cash data — skipping net debt adjustment")

        # Auto-estimate annual debt paydown from historical balance sheets (up to 3 years)
        debt_values = []
        for bs in balance_sheets[:4]:
            d = _safe_float(bs.get("totalDebt") or bs.get("longTermDebt"))
            if d is not None:
                debt_values.append(d)
        if len(debt_values) >= 2:
            # Average YoY reduction (positive = paying down, negative = taking on more)
            reductions = [debt_values[i] - debt_values[i + 1] for i in range(len(debt_values) - 1)]
            avg_reduction = sum(reductions) / len(reductions)
            if avg_reduction > 0:
                auto_debt_paydown = avg_reduction
    else:
        warnings.append("No balance sheet data — skipping net debt adjustment")

    # Resolve annual debt paydown — use user override if provided, else auto-estimate
    resolved_debt_paydown = annual_debt_paydown if annual_debt_paydown is not None else (auto_debt_paydown or 0.0)
    # Project net debt to end of projection period (floor at zero — can't have negative debt)
    net_debt_at_terminal = max(net_debt - resolved_debt_paydown * projection_years, 0.0)

    # --- Historical FCF growth rate (CAGR) ---
    hist_fcf_growth = None
    positive_fcfs = [f for f in fcf_values if f > 0]
    if len(positive_fcfs) >= 3:
        oldest = positive_fcfs[-1]
        newest = positive_fcfs[0]
        years_span = len(positive_fcfs) - 1
        if oldest > 0 and newest > 0:
            hist_fcf_growth = (newest / oldest) ** (1 / years_span) - 1

    # --- Analyst-estimated revenue growth rate (CAGR from forward estimates) ---
    analyst_revenue_growth = None
    analyst_num_analysts = None
    if analyst_estimates:
        # Estimates come newest-first; sort by date ascending for CAGR calc
        sorted_est = sorted(analyst_estimates, key=lambda x: x.get("date", ""))
        rev_estimates = [(e["date"], e["revenueAvg"]) for e in sorted_est
                         if e.get("revenueAvg") and e["revenueAvg"] > 0]
        if len(rev_estimates) >= 2:
            first_rev = rev_estimates[0][1]
            last_rev = rev_estimates[-1][1]
            years_span_est = len(rev_estimates) - 1
            if first_rev > 0 and last_rev > 0:
                analyst_revenue_growth = (last_rev / first_rev) ** (1 / years_span_est) - 1
                # Use the estimate with the most analyst coverage for display
                analyst_num_analysts = max(
                    (e.get("numAnalystsRevenue", 0) for e in sorted_est), default=None
                )

    # --- Default assumptions ---
    # Growth rate: use historical if available, cap at reasonable bounds
    if growth_rate_override is not None:
        growth_rate = growth_rate_override
    elif hist_fcf_growth is not None:
        # Cap historical growth between -5% and 20%
        growth_rate = max(-0.05, min(0.20, hist_fcf_growth))
        if hist_fcf_growth > 0.20:
            warnings.append(f"Historical FCF CAGR ({hist_fcf_growth*100:.1f}%) capped at 20% — unsustainable long-term")
        elif hist_fcf_growth < -0.05:
            warnings.append(f"Historical FCF CAGR is negative ({hist_fcf_growth*100:.1f}%) — using -5% floor")
    else:
        growth_rate = 0.05  # conservative default
        warnings.append("Could not compute historical FCF growth — using 5% default")

    # --- WACC calculation (proper weighted average) ---
    if risk_free_rate is None:
        risk_free_rate = 0.043  # fallback ~10yr Treasury
    equity_risk_premium = 0.055
    beta = _safe_float(profile.get("beta")) or 1.0
    if beta < 0.3:
        beta = 1.0
        warnings.append("Beta unusually low — defaulting to 1.0")
    elif beta > 3.0:
        warnings.append(f"Beta very high ({beta:.2f}) — WACC will be elevated")

    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # Compute cost of debt and capital structure weights
    market_cap = _safe_float(profile.get("mktCap") or profile.get("marketCap")) or 0
    total_debt_for_wacc = 0.0
    cost_of_debt = 0.0
    effective_tax_rate = 0.25  # fallback
    wacc_breakdown = {}

    if balance_sheets and income_statements and market_cap > 0:
        latest_bs = balance_sheets[0]
        latest_is = income_statements[0]

        total_debt_for_wacc = _safe_float(
            latest_bs.get("totalDebt") or latest_bs.get("longTermDebt")
        ) or 0

        interest_expense = abs(_safe_float(latest_is.get("interestExpense")) or 0)
        pre_tax_income = _safe_float(latest_is.get("incomeBeforeTax")) or 0
        income_tax = _safe_float(latest_is.get("incomeTaxExpense")) or 0

        # Cost of debt = interest expense / total debt
        if total_debt_for_wacc > 0 and interest_expense > 0:
            cost_of_debt = interest_expense / total_debt_for_wacc
            # Cap cost of debt at reasonable bounds (2% - 20%)
            cost_of_debt = max(0.02, min(0.20, cost_of_debt))
        else:
            cost_of_debt = 0.05  # fallback if no debt or no interest

        # Effective tax rate
        if pre_tax_income > 0 and income_tax > 0:
            effective_tax_rate = min(0.40, max(0.0, income_tax / pre_tax_income))

    # Capital structure weights
    total_capital = market_cap + total_debt_for_wacc
    if total_capital > 0 and total_debt_for_wacc > 0:
        weight_equity = market_cap / total_capital
        weight_debt = total_debt_for_wacc / total_capital
        default_wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - effective_tax_rate))
        wacc_breakdown = {
            "cost_of_equity": round(cost_of_equity * 100, 2),
            "cost_of_debt": round(cost_of_debt * 100, 2),
            "effective_tax_rate": round(effective_tax_rate * 100, 1),
            "weight_equity": round(weight_equity * 100, 1),
            "weight_debt": round(weight_debt * 100, 1),
            "market_cap": market_cap,
            "total_debt": total_debt_for_wacc,
        }
    else:
        # No debt or no market cap data — fall back to cost of equity only
        default_wacc = cost_of_equity
        wacc_breakdown = {
            "cost_of_equity": round(cost_of_equity * 100, 2),
            "cost_of_debt": 0,
            "effective_tax_rate": round(effective_tax_rate * 100, 1),
            "weight_equity": 100.0,
            "weight_debt": 0.0,
        }

    discount_rate = discount_rate_override if discount_rate_override is not None else default_wacc

    # Terminal growth rate
    terminal_growth = terminal_growth_override if terminal_growth_override is not None else 0.025

    # Safety: discount rate must exceed terminal growth
    if discount_rate <= terminal_growth:
        warnings.append("Discount rate must exceed terminal growth — adjusting terminal growth down")
        terminal_growth = discount_rate - 0.01

    # --- Build per-year growth + ROIC schedules ---
    # Single source of truth used by projection loop AND sensitivity table.
    # Modes (in order of precedence):
    #   1. growth_stages (manual multi-stage) — expanded year by year
    #   2. use_fade — linear fade from growth_rate to terminal_growth between
    #      fade_start_year and projection_years (and ROIC fades to WACC over the
    #      same window when in reinvestment mode — mature companies can't earn
    #      excess returns forever).
    #   3. flat growth_rate
    base_roic_for_fade = reinvestment_info["roic_capped"] if reinvestment_info else None

    def _build_growth_schedule(initial_g: float) -> list[float]:
        """Returns a list of length projection_years with the growth rate per year."""
        schedule = []
        if growth_stages:
            for stage in growth_stages:
                for _ in range(stage["years"]):
                    if len(schedule) < projection_years:
                        schedule.append(stage["rate"])
            # Pad if stages don't fill the horizon
            last_rate = growth_stages[-1]["rate"] if growth_stages else initial_g
            while len(schedule) < projection_years:
                schedule.append(last_rate)
        elif use_fade:
            fs = max(1, min(int(fade_start_year), projection_years))
            for y in range(1, projection_years + 1):
                if y < fs:
                    schedule.append(initial_g)
                elif projection_years == fs:
                    schedule.append(terminal_growth)
                else:
                    t = (y - fs) / (projection_years - fs)
                    schedule.append(initial_g + (terminal_growth - initial_g) * t)
        else:
            schedule = [initial_g] * projection_years
        return schedule

    def _build_roic_schedule(initial_roic: float) -> list[float]:
        """ROIC fade schedule. ROIC fades to WACC (discount_rate) when fade is on."""
        if not use_fade:
            return [initial_roic] * projection_years
        fs = max(1, min(int(fade_start_year), projection_years))
        schedule = []
        for y in range(1, projection_years + 1):
            if y < fs:
                schedule.append(initial_roic)
            elif projection_years == fs:
                schedule.append(discount_rate)
            else:
                t = (y - fs) / (projection_years - fs)
                schedule.append(initial_roic + (discount_rate - initial_roic) * t)
        return schedule

    year_growth_rates = _build_growth_schedule(growth_rate)
    year_roics = _build_roic_schedule(base_roic_for_fade) if reinvestment_info else None

    # Mid-year discounting convention: cash flows arrive throughout the year on
    # average, not at year-end. Discount factor uses (year - 0.5) instead of year.
    # Standard sell-side practice; adds ~3-5% accuracy vs year-end convention.
    disc_offset = 0.5 if use_mid_year_discounting else 0.0

    # --- Project FCFs ---
    # In reinvestment-aware mode, FCF = NOPAT × (1 - g/ROIC), where NOPAT compounds
    # at the per-year growth rate. This makes growth and reinvestment internally
    # consistent: you can't grow faster than your ROIC supports without burning cash.
    projected_fcfs = []
    if reinvestment_info:
        nopat = reinvestment_info["base_nopat"]
        flagged_growth_above_roic = False
        for year in range(1, projection_years + 1):
            g = year_growth_rates[year - 1]
            roic_y = year_roics[year - 1]
            if g > roic_y and not flagged_growth_above_roic:
                warnings.append(
                    f"Growth rate ({g*100:.1f}%) exceeds ROIC ({roic_y*100:.1f}%) in year {year} "
                    "-- reinvestment capped at 100% (FCF=0 for affected years)"
                )
                flagged_growth_above_roic = True
            nopat = nopat * (1 + g)
            rr = max(0.0, min(1.0, g / roic_y)) if roic_y > 0 else 1.0
            fcf_t = nopat * (1 - rr)
            pv = fcf_t / (1 + discount_rate) ** (year - disc_offset)
            projected_fcfs.append({
                "year": year,
                "fcf": round(fcf_t, 0),
                "present_value": round(pv, 0),
                "stage_rate": round(g * 100, 1),
                "nopat": round(nopat, 0),
                "reinvestment_rate": round(rr * 100, 1),
                "roic": round(roic_y * 100, 1),
            })
        final_nopat = nopat
    else:
        fcf = base_fcf
        for year in range(1, projection_years + 1):
            g = year_growth_rates[year - 1]
            fcf = fcf * (1 + g)
            pv = fcf / (1 + discount_rate) ** (year - disc_offset)
            projected_fcfs.append({
                "year": year,
                "fcf": round(fcf, 0),
                "present_value": round(pv, 0),
                "stage_rate": round(g * 100, 1),
            })

    # --- Terminal value ---
    if reinvestment_info:
        # When fade is on, ROIC has converged toward WACC by year 10 — terminal
        # uses that converged ROIC, so reinvestment rate at terminal = tg/WACC.
        terminal_roic = year_roics[-1]
        terminal_nopat = final_nopat * (1 + terminal_growth)
        terminal_rr = max(0.0, min(1.0, terminal_growth / terminal_roic)) if terminal_roic > 0 else 1.0
        terminal_fcf = terminal_nopat * (1 - terminal_rr)
    else:
        terminal_fcf = projected_fcfs[-1]["fcf"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    # Discount terminal back using same mid-year convention as explicit period
    terminal_pv = terminal_value / (1 + discount_rate) ** (projection_years - disc_offset)

    # --- Intrinsic value ---
    sum_pv_fcfs = sum(p["present_value"] for p in projected_fcfs)
    enterprise_value = sum_pv_fcfs + terminal_pv
    # Use projected net debt at terminal (accounts for debt paydown over projection period)
    equity_value = enterprise_value - net_debt_at_terminal
    if equity_value <= 0:
        warnings.append("Equity value negative after subtracting net debt — company may be over-leveraged")
        current_price = _safe_float(profile.get("price"))
        return {
            "dcf_price": None,
            "warnings": warnings,
            "has_data": True,
            "current_price": current_price,
            "assumptions": {
                "base_fcf": base_fcf,
                "growth_rate": round(growth_rate * 100, 2),
                "discount_rate": round(discount_rate * 100, 2),
                "terminal_growth": round(terminal_growth * 100, 2),
                "shares_outstanding": shares,
                "beta": beta,
                "net_debt": round(net_debt, 0),
                "net_debt_at_terminal": round(net_debt_at_terminal, 0),
                "annual_debt_paydown": round(resolved_debt_paydown, 0),
                "auto_debt_paydown": round(auto_debt_paydown, 0) if auto_debt_paydown is not None else None,
                "hist_fcf_growth": round(hist_fcf_growth * 100, 1) if hist_fcf_growth is not None else None,
                "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
                "analyst_num_analysts": analyst_num_analysts,
                "wacc_breakdown": wacc_breakdown,
                "annual_share_change": round(resolved_share_change * 100, 2),
                "hist_share_change": round(hist_share_change * 100, 1) if hist_share_change is not None else None,
                "reinvestment_model": bool(reinvestment_info),
                "base_nopat": round(reinvestment_info["base_nopat"], 0) if reinvestment_info else None,
                "roic": round(reinvestment_info["roic"] * 100, 2) if reinvestment_info else None,
                "roic_capped": round(reinvestment_info["roic_capped"] * 100, 2) if reinvestment_info else None,
                "invested_capital": round(reinvestment_info["invested_capital"], 0) if reinvestment_info else None,
                "use_fade": bool(use_fade),
                "fade_start_year": int(fade_start_year) if use_fade else None,
                "mid_year_discounting": bool(use_mid_year_discounting),
            },
            "projected_fcfs": projected_fcfs,
            "sensitivity": [],
        }
    intrinsic_per_share = equity_value / terminal_shares

    current_price = _safe_float(profile.get("price"))
    upside = None
    if current_price and current_price > 0:
        upside = round((intrinsic_per_share / current_price - 1) * 100, 1)

    # --- Terminal value as % of total (sanity check) ---
    terminal_pct = (terminal_pv / enterprise_value * 100) if enterprise_value > 0 else 0
    if terminal_pct > 80:
        warnings.append(f"Terminal value is {terminal_pct:.0f}% of total — valuation heavily depends on long-term assumptions")

    # --- Sensitivity table ---
    # Center on weighted average of multi-stage rates, or flat growth rate
    if growth_stages:
        _total_years = sum(s["years"] for s in growth_stages) or 1
        _sens_center_rate = sum(s["rate"] * s["years"] for s in growth_stages) / _total_years
    else:
        _sens_center_rate = growth_rate
    # Growth rates: centered on selected, +/- 4 steps of 2%
    growth_steps = [_sens_center_rate + (i - 4) * 0.02 for i in range(9)]
    growth_steps = [g for g in growth_steps if -0.10 <= g <= 0.50]  # clamp
    # Discount rates: centered on selected, +/- 3 steps of 1%
    discount_steps = [discount_rate + (i - 3) * 0.01 for i in range(7)]
    discount_steps = [d for d in discount_steps if 0.04 <= d <= 0.20]  # clamp

    sensitivity = []
    for g in growth_steps:
        row = {"growth_rate": round(g * 100, 1)}
        # Per-year growth schedule for this cell — honors fade if enabled
        cell_growth_schedule = _build_growth_schedule(g)
        for d in discount_steps:
            if d <= terminal_growth + 0.005:  # discount must exceed terminal growth
                row[f"{d*100:.1f}%"] = "N/A"
                continue
            # Quick DCF calc for this combo
            pv_sum = 0
            if reinvestment_info:
                base_roic_s = reinvestment_info["roic_capped"]
                # ROIC fades to *this cell's* discount rate, not the base WACC
                if use_fade:
                    fs = max(1, min(int(fade_start_year), projection_years))
                    cell_roic_schedule = []
                    for yr in range(1, projection_years + 1):
                        if yr < fs:
                            cell_roic_schedule.append(base_roic_s)
                        elif projection_years == fs:
                            cell_roic_schedule.append(d)
                        else:
                            t = (yr - fs) / (projection_years - fs)
                            cell_roic_schedule.append(base_roic_s + (d - base_roic_s) * t)
                else:
                    cell_roic_schedule = [base_roic_s] * projection_years
                n_t = reinvestment_info["base_nopat"]
                for yr in range(1, projection_years + 1):
                    g_yr = cell_growth_schedule[yr - 1]
                    roic_yr = cell_roic_schedule[yr - 1]
                    rr_yr = max(0.0, min(1.0, g_yr / roic_yr)) if roic_yr > 0 else 1.0
                    n_t = n_t * (1 + g_yr)
                    f_t = n_t * (1 - rr_yr)
                    pv_sum += f_t / (1 + d) ** (yr - disc_offset)
                terminal_roic_s = cell_roic_schedule[-1]
                t_rr = max(0.0, min(1.0, terminal_growth / terminal_roic_s)) if terminal_roic_s > 0 else 1.0
                t_fcf = n_t * (1 + terminal_growth) * (1 - t_rr)
            else:
                f = base_fcf
                for yr in range(1, projection_years + 1):
                    f = f * (1 + cell_growth_schedule[yr - 1])
                    pv_sum += f / (1 + d) ** (yr - disc_offset)
                t_fcf = f * (1 + terminal_growth)
            t_val = t_fcf / (d - terminal_growth) if d > terminal_growth else 0
            t_pv = t_val / (1 + d) ** (projection_years - disc_offset)
            ev = pv_sum + t_pv
            eq = ev - net_debt_at_terminal
            price = max(eq, 0) / terminal_shares
            row[f"{d*100:.1f}%"] = round(price, 2)
        sensitivity.append(row)

    return {
        "dcf_price": round(intrinsic_per_share, 2),
        "current_price": current_price,
        "upside_pct": upside,
        "assumptions": {
            "base_fcf": round(base_fcf, 0),
            "growth_rate": round(growth_rate * 100, 2),
            "discount_rate": round(discount_rate * 100, 2),
            "terminal_growth": round(terminal_growth * 100, 2),
            "beta": round(beta, 2),
            "risk_free_rate": round(risk_free_rate * 100, 2),
            "projection_years": projection_years,
            "shares_outstanding": shares,
            "net_debt": round(net_debt, 0),
            "net_debt_at_terminal": round(net_debt_at_terminal, 0),
            "annual_debt_paydown": round(resolved_debt_paydown, 0),
            "auto_debt_paydown": round(auto_debt_paydown, 0) if auto_debt_paydown is not None else None,
            "hist_fcf_growth": round(hist_fcf_growth * 100, 1) if hist_fcf_growth is not None else None,
            "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
            "analyst_num_analysts": analyst_num_analysts,
            "wacc_breakdown": wacc_breakdown,
            "annual_share_change": round(resolved_share_change * 100, 2),
            "hist_share_change": round(hist_share_change * 100, 1) if hist_share_change is not None else None,
            "terminal_shares": round(terminal_shares, 0),
            "reinvestment_model": bool(reinvestment_info),
            "base_nopat": round(reinvestment_info["base_nopat"], 0) if reinvestment_info else None,
            "roic": round(reinvestment_info["roic"] * 100, 2) if reinvestment_info else None,
            "roic_capped": round(reinvestment_info["roic_capped"] * 100, 2) if reinvestment_info else None,
            "invested_capital": round(reinvestment_info["invested_capital"], 0) if reinvestment_info else None,
            "use_fade": bool(use_fade),
            "fade_start_year": int(fade_start_year) if use_fade else None,
            "year_growth_rates": [round(g * 100, 2) for g in year_growth_rates],
            "year_roics": [round(r * 100, 2) for r in year_roics] if year_roics else None,
            "mid_year_discounting": bool(use_mid_year_discounting),
        },
        "projected_fcfs": projected_fcfs,
        "terminal_value": round(terminal_value, 0),
        "terminal_pv": round(terminal_pv, 0),
        "terminal_pct": round(terminal_pct, 1),
        "sensitivity": sensitivity,
        "sensitivity_discount_rates": [f"{d*100:.1f}%" for d in discount_steps],
        "warnings": warnings,
    }


def run_dcf_monte_carlo(
    base_fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
    current_price: float | None,
    projection_years: int = 10,
    n_simulations: int = 10_000,
    growth_std: float | None = None,
    discount_std: float | None = None,
    terminal_std: float | None = None,
    use_mid_year_discounting: bool = True,
) -> dict:
    """
    Monte Carlo simulation over DCF assumptions.

    Samples growth rate, discount rate, and terminal growth from normal
    distributions centered on the base assumptions, runs N independent DCF
    valuations, and returns the distribution of implied prices.

    Args:
        base_fcf: Starting free cash flow (dollars).
        growth_rate: Central FCF growth rate (decimal).
        discount_rate: Central WACC (decimal).
        terminal_growth: Central terminal growth rate (decimal).
        net_debt: Net debt for EV→equity bridge (dollars).
        shares: Diluted shares outstanding.
        current_price: Current stock price (for upside probability calc).
        projection_years: DCF projection horizon.
        n_simulations: Number of Monte Carlo iterations.
        growth_std: Std dev of growth rate draws. Defaults to max(4%, half of abs(growth_rate)).
        discount_std: Std dev of discount rate draws. Defaults to 1.5%.
        terminal_std: Std dev of terminal growth draws. Defaults to 0.5%.

    Returns:
        Dict with prices (list), stats (dict), upside_probability, and
        percentiles for histogram rendering.
    """
    rng = np.random.default_rng(seed=42)

    # Default standard deviations — calibrated to give a meaningful spread
    if growth_std is None:
        growth_std = max(0.04, abs(growth_rate) * 0.5)
    if discount_std is None:
        discount_std = 0.015
    if terminal_std is None:
        terminal_std = 0.005

    # Draw samples
    growth_samples = rng.normal(growth_rate, growth_std, n_simulations)
    discount_samples = rng.normal(discount_rate, discount_std, n_simulations)
    terminal_samples = rng.normal(terminal_growth, terminal_std, n_simulations)

    # Clamp to realistic bounds
    growth_samples = np.clip(growth_samples, -0.20, 0.60)
    discount_samples = np.clip(discount_samples, 0.04, 0.25)
    terminal_samples = np.clip(terminal_samples, 0.005, 0.05)

    # Ensure discount > terminal for every simulation
    mask = discount_samples <= terminal_samples + 0.005
    discount_samples[mask] = terminal_samples[mask] + 0.01

    # Vectorised DCF — project FCFs and discount (with mid-year convention)
    # Shape: (n_simulations, projection_years)
    disc_offset = 0.5 if use_mid_year_discounting else 0.0
    years = np.arange(1, projection_years + 1)
    discount_exponents = years - disc_offset                         # (T,)
    growth_factors = (1 + growth_samples[:, None]) ** years          # (N, T)
    discount_factors = (1 + discount_samples[:, None]) ** discount_exponents  # (N, T)

    fcf_projections = base_fcf * growth_factors                      # (N, T)
    pv_fcfs = fcf_projections / discount_factors                     # (N, T)
    sum_pv = pv_fcfs.sum(axis=1)                                     # (N,)

    # Terminal value (also uses mid-year offset for terminal discount)
    terminal_fcf = fcf_projections[:, -1] * (1 + terminal_samples)
    terminal_value = terminal_fcf / (discount_samples - terminal_samples)
    terminal_pv = terminal_value / discount_factors[:, -1]

    # Equity value per share
    ev = sum_pv + terminal_pv
    equity_value = ev - net_debt
    prices = np.where(equity_value > 0, equity_value / shares, 0.0)

    # Cap extreme outliers at 99th percentile for display
    p99 = float(np.percentile(prices, 99))
    prices_clipped = np.clip(prices, 0, p99)

    # Stats
    p10 = float(np.percentile(prices_clipped, 10))
    p25 = float(np.percentile(prices_clipped, 25))
    p50 = float(np.percentile(prices_clipped, 50))
    p75 = float(np.percentile(prices_clipped, 75))
    p90 = float(np.percentile(prices_clipped, 90))
    mean = float(np.mean(prices_clipped))

    upside_probability = None
    if current_price and current_price > 0:
        upside_probability = float(np.mean(prices_clipped > current_price) * 100)

    return {
        "prices": prices_clipped.tolist(),
        "n_simulations": n_simulations,
        "stats": {
            "mean": round(mean, 2),
            "p10": round(p10, 2),
            "p25": round(p25, 2),
            "p50": round(p50, 2),
            "p75": round(p75, 2),
            "p90": round(p90, 2),
        },
        "upside_probability": round(upside_probability, 1) if upside_probability is not None else None,
        "assumptions": {
            "growth_rate": round(growth_rate * 100, 2),
            "growth_std": round(growth_std * 100, 2),
            "discount_rate": round(discount_rate * 100, 2),
            "discount_std": round(discount_std * 100, 2),
            "terminal_growth": round(terminal_growth * 100, 2),
            "terminal_std": round(terminal_std * 100, 2),
        },
    }


def compute_revenue_dcf_valuation(income_statements: list[dict],
                                   cash_flow_statements: list[dict],
                                   profile: dict,
                                   balance_sheets: list[dict] | None = None,
                                   revenue_growth_override: float | None = None,
                                   target_fcf_margin_override: float | None = None,
                                   discount_rate_override: float | None = None,
                                   terminal_growth_override: float | None = None,
                                   projection_years: int = 10,
                                   analyst_estimates: list[dict] | None = None,
                                   growth_stages: list[dict] | None = None,
                                   risk_free_rate: float | None = None,
                                   annual_debt_paydown: float | None = None,
                                   use_margin_reversion: bool = True,
                                   annual_share_change: float | None = None,
                                   use_mid_year_discounting: bool = True) -> dict:
    """
    Revenue-based DCF — derives FCF from projected revenue × target FCF margin.

    Useful for companies with negative/distorted FCF (SaaS transitions, high-growth,
    turnarounds) where revenue is the more reliable starting point.
    """
    warnings = []

    # --- Extract base revenue ---
    revenues = []
    for stmt in income_statements[:5]:
        rev = _safe_float(stmt.get("revenue"))
        if rev is not None and rev > 0:
            revenues.append(rev)

    if len(revenues) < 2:
        return {"dcf_price": None, "warnings": ["Insufficient revenue history for DCF"],
                "assumptions": {}, "has_data": False}

    base_revenue = revenues[0]  # most recent annual revenue

    # --- Historical FCF margin (FCF / Revenue) ---
    hist_fcf_margins = []
    for i, stmt in enumerate(income_statements[:5]):
        rev = _safe_float(stmt.get("revenue"))
        if rev and rev > 0 and i < len(cash_flow_statements):
            fcf = _safe_float(cash_flow_statements[i].get("freeCashFlow"))
            if fcf is not None:
                hist_fcf_margins.append(fcf / rev)

    hist_median_fcf_margin = float(np.median(hist_fcf_margins)) if hist_fcf_margins else None

    # --- Historical revenue growth (CAGR) ---
    hist_rev_growth = None
    if len(revenues) >= 3:
        oldest = revenues[-1]
        newest = revenues[0]
        years_span = len(revenues) - 1
        if oldest > 0 and newest > 0:
            hist_rev_growth = (newest / oldest) ** (1 / years_span) - 1

    # --- Analyst revenue growth ---
    analyst_revenue_growth = None
    analyst_num_analysts = None
    if analyst_estimates:
        sorted_est = sorted(analyst_estimates, key=lambda x: x.get("date", ""))
        rev_estimates = [(e["date"], e["revenueAvg"]) for e in sorted_est
                         if e.get("revenueAvg") and e["revenueAvg"] > 0]
        if len(rev_estimates) >= 2:
            first_rev = rev_estimates[0][1]
            last_rev = rev_estimates[-1][1]
            years_span_est = len(rev_estimates) - 1
            if first_rev > 0 and last_rev > 0:
                analyst_revenue_growth = (last_rev / first_rev) ** (1 / years_span_est) - 1
                analyst_num_analysts = max(
                    (e.get("numAnalystsRevenue", 0) for e in sorted_est), default=None
                )

    # --- Shares outstanding ---
    shares = None
    if income_statements:
        shares = _safe_float(
            income_statements[0].get("weightedAverageShsOutDil")
            or income_statements[0].get("weightedAverageShsOut")
        )
    if not shares or shares <= 0:
        return {"dcf_price": None, "warnings": ["No shares outstanding data"],
                "assumptions": {}, "has_data": False}

    # --- Historical share count change (buybacks = negative, dilution = positive) ---
    hist_share_change_r = None
    share_count_values_r = []
    for stmt in income_statements[:5]:
        s = _safe_float(stmt.get("weightedAverageShsOutDil") or stmt.get("weightedAverageShsOut"))
        if s and s > 0:
            share_count_values_r.append(s)
    if len(share_count_values_r) >= 2:
        hist_share_change_r = (share_count_values_r[0] / share_count_values_r[-1]) ** (1 / (len(share_count_values_r) - 1)) - 1

    resolved_share_change_r = annual_share_change if annual_share_change is not None else (hist_share_change_r or 0.0)
    resolved_share_change_r = max(-0.20, min(0.20, resolved_share_change_r))
    terminal_shares_r = max(shares * (1 + resolved_share_change_r) ** projection_years, shares * 0.01)

    # --- Net debt ---
    net_debt = 0.0
    auto_debt_paydown = None
    balance_sheets = balance_sheets or []
    if balance_sheets:
        latest_bs = balance_sheets[0]
        total_debt = _safe_float(latest_bs.get("totalDebt") or latest_bs.get("longTermDebt"))
        cash = _safe_float(latest_bs.get("cashAndCashEquivalents") or
                           latest_bs.get("cashAndShortTermInvestments"))
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
        else:
            warnings.append("Missing debt/cash data — skipping net debt adjustment")

        # Auto-estimate annual debt paydown from historical balance sheets
        debt_values = []
        for bs in balance_sheets[:4]:
            d = _safe_float(bs.get("totalDebt") or bs.get("longTermDebt"))
            if d is not None:
                debt_values.append(d)
        if len(debt_values) >= 2:
            reductions = [debt_values[i] - debt_values[i + 1] for i in range(len(debt_values) - 1)]
            avg_reduction = sum(reductions) / len(reductions)
            if avg_reduction > 0:
                auto_debt_paydown = avg_reduction

    resolved_debt_paydown = annual_debt_paydown if annual_debt_paydown is not None else (auto_debt_paydown or 0.0)
    net_debt_at_terminal = max(net_debt - resolved_debt_paydown * projection_years, 0.0)

    # --- Growth rate ---
    if revenue_growth_override is not None:
        revenue_growth = revenue_growth_override
    elif hist_rev_growth is not None:
        revenue_growth = max(-0.05, min(0.30, hist_rev_growth))
    else:
        revenue_growth = 0.08
        warnings.append("Could not compute historical revenue growth — using 8% default")

    # --- Target FCF margin (terminal margin the company converges to) ---
    if target_fcf_margin_override is not None:
        target_fcf_margin = target_fcf_margin_override
    elif hist_median_fcf_margin is not None and hist_median_fcf_margin > 0:
        target_fcf_margin = hist_median_fcf_margin
    else:
        target_fcf_margin = 0.10  # conservative 10% default
        warnings.append("No positive historical FCF margin — using 10% default target")

    # Starting margin for reversion — most recent year's FCF/Revenue
    starting_fcf_margin = hist_fcf_margins[0] if hist_fcf_margins else target_fcf_margin
    # If starting margin is negative, begin at 0 and grow to target
    if starting_fcf_margin < 0:
        starting_fcf_margin = 0.0
        warnings.append("Most recent FCF margin is negative — margin reversion starts from 0%")

    # --- WACC calculation (proper weighted average) ---
    if risk_free_rate is None:
        risk_free_rate = 0.043  # fallback ~10yr Treasury
    equity_risk_premium = 0.055
    beta = _safe_float(profile.get("beta")) or 1.0
    if beta < 0.3:
        beta = 1.0

    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    market_cap = _safe_float(profile.get("mktCap") or profile.get("marketCap")) or 0
    total_debt_for_wacc = 0.0
    cost_of_debt = 0.0
    effective_tax_rate = 0.25
    wacc_breakdown = {}

    if balance_sheets and income_statements and market_cap > 0:
        latest_bs = balance_sheets[0]
        latest_is = income_statements[0]

        total_debt_for_wacc = _safe_float(
            latest_bs.get("totalDebt") or latest_bs.get("longTermDebt")
        ) or 0

        interest_expense = abs(_safe_float(latest_is.get("interestExpense")) or 0)
        pre_tax_income = _safe_float(latest_is.get("incomeBeforeTax")) or 0
        income_tax = _safe_float(latest_is.get("incomeTaxExpense")) or 0

        if total_debt_for_wacc > 0 and interest_expense > 0:
            cost_of_debt = interest_expense / total_debt_for_wacc
            cost_of_debt = max(0.02, min(0.20, cost_of_debt))
        else:
            cost_of_debt = 0.05

        if pre_tax_income > 0 and income_tax > 0:
            effective_tax_rate = min(0.40, max(0.0, income_tax / pre_tax_income))

    total_capital = market_cap + total_debt_for_wacc
    if total_capital > 0 and total_debt_for_wacc > 0:
        weight_equity = market_cap / total_capital
        weight_debt = total_debt_for_wacc / total_capital
        default_wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - effective_tax_rate))
        wacc_breakdown = {
            "cost_of_equity": round(cost_of_equity * 100, 2),
            "cost_of_debt": round(cost_of_debt * 100, 2),
            "effective_tax_rate": round(effective_tax_rate * 100, 1),
            "weight_equity": round(weight_equity * 100, 1),
            "weight_debt": round(weight_debt * 100, 1),
            "market_cap": market_cap,
            "total_debt": total_debt_for_wacc,
        }
    else:
        default_wacc = cost_of_equity
        wacc_breakdown = {
            "cost_of_equity": round(cost_of_equity * 100, 2),
            "cost_of_debt": 0,
            "effective_tax_rate": round(effective_tax_rate * 100, 1),
            "weight_equity": 100.0,
            "weight_debt": 0.0,
        }

    discount_rate = discount_rate_override if discount_rate_override is not None else default_wacc

    # --- Terminal growth ---
    terminal_growth = terminal_growth_override if terminal_growth_override is not None else 0.025
    if discount_rate <= terminal_growth:
        warnings.append("Discount rate must exceed terminal growth — adjusting terminal growth down")
        terminal_growth = discount_rate - 0.01

    # --- Project revenue → FCF ---
    projected_fcfs = []
    rev = base_revenue
    # Mid-year discounting: cash flows arrive on average mid-year, not year-end.
    disc_offset = 0.5 if use_mid_year_discounting else 0.0

    def _margin_for_year(year: int) -> float:
        """Linearly interpolate from starting_fcf_margin → target_fcf_margin over projection_years."""
        if not use_margin_reversion or projection_years <= 1:
            return target_fcf_margin
        t = year / projection_years  # 0→1 over the horizon
        return starting_fcf_margin + t * (target_fcf_margin - starting_fcf_margin)

    if growth_stages:
        year_counter = 1
        for stage in growth_stages:
            stage_rate = stage["rate"]
            stage_years = stage["years"]
            for _ in range(stage_years):
                if year_counter > projection_years:
                    break
                rev = rev * (1 + stage_rate)
                margin = _margin_for_year(year_counter)
                fcf = rev * margin
                pv = fcf / (1 + discount_rate) ** (year_counter - disc_offset)
                projected_fcfs.append({
                    "year": year_counter,
                    "revenue": round(rev, 0),
                    "fcf": round(fcf, 0),
                    "present_value": round(pv, 0),
                    "fcf_margin": round(margin * 100, 2),
                    "stage_rate": round(stage_rate * 100, 1),
                })
                year_counter += 1
    else:
        for year in range(1, projection_years + 1):
            rev = rev * (1 + revenue_growth)
            margin = _margin_for_year(year)
            fcf = rev * margin
            pv = fcf / (1 + discount_rate) ** (year - disc_offset)
            projected_fcfs.append({
                "year": year,
                "revenue": round(rev, 0),
                "fcf": round(fcf, 0),
                "present_value": round(pv, 0),
                "fcf_margin": round(margin * 100, 2),
            })

    # --- Terminal value ---
    terminal_fcf = projected_fcfs[-1]["fcf"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** (projection_years - disc_offset)

    # --- Intrinsic value ---
    sum_pv_fcfs = sum(p["present_value"] for p in projected_fcfs)
    enterprise_value = sum_pv_fcfs + terminal_pv
    equity_value = enterprise_value - net_debt_at_terminal
    if equity_value <= 0:
        warnings.append("Equity value negative after subtracting net debt")
        current_price = _safe_float(profile.get("price"))
        return {
            "dcf_price": None, "warnings": warnings, "has_data": True,
            "current_price": current_price,
            "assumptions": {
                "base_revenue": round(base_revenue, 0),
                "revenue_growth": round(revenue_growth * 100, 2),
                "target_fcf_margin": round(target_fcf_margin * 100, 2),
                "starting_fcf_margin": round(starting_fcf_margin * 100, 2),
                "use_margin_reversion": use_margin_reversion,
                "discount_rate": round(discount_rate * 100, 2),
                "terminal_growth": round(terminal_growth * 100, 2),
                "shares_outstanding": shares,
                "beta": beta,
                "net_debt": round(net_debt, 0),
                "net_debt_at_terminal": round(net_debt_at_terminal, 0),
                "annual_debt_paydown": round(resolved_debt_paydown, 0),
                "auto_debt_paydown": round(auto_debt_paydown, 0) if auto_debt_paydown is not None else None,
                "hist_rev_growth": round(hist_rev_growth * 100, 1) if hist_rev_growth is not None else None,
                "hist_median_fcf_margin": round(hist_median_fcf_margin * 100, 1) if hist_median_fcf_margin is not None else None,
                "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
                "analyst_num_analysts": analyst_num_analysts,
                "wacc_breakdown": wacc_breakdown,
                "annual_share_change": round(resolved_share_change_r * 100, 2),
                "hist_share_change": round(hist_share_change_r * 100, 1) if hist_share_change_r is not None else None,
                "mid_year_discounting": bool(use_mid_year_discounting),
            },
            "projected_fcfs": projected_fcfs,
            "sensitivity": [],
        }

    intrinsic_per_share = equity_value / terminal_shares_r
    current_price = _safe_float(profile.get("price"))
    upside = None
    if current_price and current_price > 0:
        upside = round((intrinsic_per_share / current_price - 1) * 100, 1)

    terminal_pct = (terminal_pv / enterprise_value * 100) if enterprise_value > 0 else 0
    if terminal_pct > 80:
        warnings.append(f"Terminal value is {terminal_pct:.0f}% of total — valuation heavily depends on long-term assumptions")

    # --- Sensitivity table (revenue growth × discount rate) ---
    # Center on weighted average of multi-stage rates, or flat revenue growth rate
    if growth_stages:
        _total_years = sum(s["years"] for s in growth_stages) or 1
        _sens_center_rate = sum(s["rate"] * s["years"] for s in growth_stages) / _total_years
    else:
        _sens_center_rate = revenue_growth
    growth_steps = [_sens_center_rate + (i - 4) * 0.02 for i in range(9)]
    growth_steps = [g for g in growth_steps if -0.10 <= g <= 0.50]
    discount_steps = [discount_rate + (i - 3) * 0.01 for i in range(7)]
    discount_steps = [d for d in discount_steps if 0.04 <= d <= 0.20]

    sensitivity = []
    for g in growth_steps:
        row = {"growth_rate": round(g * 100, 1)}
        for d in discount_steps:
            if d <= terminal_growth + 0.005:
                row[f"{d*100:.1f}%"] = "N/A"
                continue
            r = base_revenue
            pv_sum = 0
            for yr in range(1, projection_years + 1):
                r = r * (1 + g)
                m = _margin_for_year(yr)
                f = r * m
                pv_sum += f / (1 + d) ** (yr - disc_offset)
            t_fcf = (r * target_fcf_margin) * (1 + terminal_growth)
            t_val = t_fcf / (d - terminal_growth) if d > terminal_growth else 0
            t_pv = t_val / (1 + d) ** (projection_years - disc_offset)
            ev = pv_sum + t_pv
            eq = ev - net_debt_at_terminal
            price = max(eq, 0) / terminal_shares_r
            row[f"{d*100:.1f}%"] = round(price, 2)
        sensitivity.append(row)

    return {
        "dcf_price": round(intrinsic_per_share, 2),
        "current_price": current_price,
        "upside_pct": upside,
        "dcf_mode": "revenue",
        "assumptions": {
            "base_revenue": round(base_revenue, 0),
            "revenue_growth": round(revenue_growth * 100, 2),
            "target_fcf_margin": round(target_fcf_margin * 100, 2),
            "starting_fcf_margin": round(starting_fcf_margin * 100, 2),
            "use_margin_reversion": use_margin_reversion,
            "discount_rate": round(discount_rate * 100, 2),
            "terminal_growth": round(terminal_growth * 100, 2),
            "beta": round(beta, 2),
            "risk_free_rate": round(risk_free_rate * 100, 2),
            "projection_years": projection_years,
            "shares_outstanding": shares,
            "net_debt": round(net_debt, 0),
            "net_debt_at_terminal": round(net_debt_at_terminal, 0),
            "annual_debt_paydown": round(resolved_debt_paydown, 0),
            "auto_debt_paydown": round(auto_debt_paydown, 0) if auto_debt_paydown is not None else None,
            "hist_rev_growth": round(hist_rev_growth * 100, 1) if hist_rev_growth is not None else None,
            "hist_median_fcf_margin": round(hist_median_fcf_margin * 100, 1) if hist_median_fcf_margin is not None else None,
            "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
            "analyst_num_analysts": analyst_num_analysts,
            "wacc_breakdown": wacc_breakdown,
            "annual_share_change": round(resolved_share_change_r * 100, 2),
            "hist_share_change": round(hist_share_change_r * 100, 1) if hist_share_change_r is not None else None,
            "terminal_shares": round(terminal_shares_r, 0),
        },
        "projected_fcfs": projected_fcfs,
        "terminal_value": round(terminal_value, 0),
        "terminal_pv": round(terminal_pv, 0),
        "terminal_pct": round(terminal_pct, 1),
        "sensitivity": sensitivity,
        "sensitivity_discount_rates": [f"{d*100:.1f}%" for d in discount_steps],
        "warnings": warnings,
    }


def compute_reverse_dcf(
    base_fcf: float,
    current_price: float,
    shares: float,
    net_debt: float,
    discount_rate: float,
    terminal_growth: float,
    projection_years: int = 10,
    annual_debt_paydown: float = 0.0,
    annual_share_change: float = 0.0,
    use_mid_year_discounting: bool = True,
) -> dict:
    """
    Reverse DCF: solve for the implied flat FCF growth rate that makes the DCF
    intrinsic value equal the current market price.

    Uses bisection (60 iterations → accuracy < 0.0001%).

    Returns:
        dict with:
          implied_growth (float | None) — solved growth rate in %
          direction ("solved" | "below_floor" | "above_ceiling")
          message (str | None) — human-readable edge-case note
    """
    if not (base_fcf and base_fcf > 0 and current_price and current_price > 0
            and shares and shares > 0):
        return {"implied_growth": None, "direction": "error",
                "message": "Insufficient data for Reverse DCF"}

    net_debt_at_terminal = max(net_debt - annual_debt_paydown * projection_years, 0.0)
    # Project terminal shares (buybacks reduce denominator, boosting per-share value)
    _rev_share_chg = max(-0.20, min(0.20, annual_share_change))
    terminal_shares = max(shares * (1 + _rev_share_chg) ** projection_years, shares * 0.01)
    # Equity value implied by current price (using terminal shares to match forward DCF)
    target_equity_value = current_price * terminal_shares
    # Enterprise value implied by current price
    target_ev = target_equity_value + net_debt_at_terminal

    disc_offset = 0.5 if use_mid_year_discounting else 0.0

    def _ev_at_growth(g: float) -> float:
        fcf = base_fcf
        total_pv = 0.0
        for year in range(1, projection_years + 1):
            fcf = fcf * (1 + g)
            total_pv += fcf / (1 + discount_rate) ** (year - disc_offset)
        terminal_fcf = fcf * (1 + terminal_growth)
        tv = terminal_fcf / (discount_rate - terminal_growth)
        tv_pv = tv / (1 + discount_rate) ** (projection_years - disc_offset)
        return total_pv + tv_pv

    lo, hi = -0.30, 1.00  # search from -30% to +100% annual FCF growth
    ev_lo = _ev_at_growth(lo)
    ev_hi = _ev_at_growth(hi)

    if ev_lo > target_ev:
        # Even at -30% annual FCF decline the company is worth more than market price →
        # market is pricing in worse than -30% — stock looks deeply undervalued vs. FCF
        return {
            "implied_growth": None,
            "direction": "below_floor",
            "message": (
                "Market price implies worse than −30% annual FCF decline — "
                "stock appears deeply undervalued relative to current cash flows."
            ),
        }
    if ev_hi < target_ev:
        # Even at +100% annual FCF growth the DCF can't justify current price →
        # market expects extraordinary growth beyond the search range
        return {
            "implied_growth": None,
            "direction": "above_ceiling",
            "message": (
                "Market price implies more than +100% annual FCF growth — "
                "stock is priced for extraordinary expectations."
            ),
        }

    # Bisection — _ev_at_growth is monotonically increasing in g
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _ev_at_growth(mid) < target_ev:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break

    implied_g = (lo + hi) / 2.0
    return {
        "implied_growth": round(implied_g * 100, 2),
        "direction": "solved",
        "message": None,
    }


def compute_reverse_revenue_dcf(
    base_revenue: float,
    fcf_margin: float,
    current_price: float,
    shares: float,
    net_debt: float,
    discount_rate: float,
    terminal_growth: float,
    projection_years: int = 10,
    annual_debt_paydown: float = 0.0,
    annual_share_change: float = 0.0,
    use_mid_year_discounting: bool = True,
) -> dict:
    """
    Reverse Revenue DCF: solve for the implied annual revenue growth rate that
    makes the revenue-based DCF intrinsic value equal the current market price.

    FCF each year = projected_revenue × fcf_margin (constant margin assumption).
    Uses bisection (60 iterations).

    Args:
        base_revenue:      Most recent annual revenue (dollars).
        fcf_margin:        Target FCF margin as a decimal (e.g. 0.12 for 12%).
        current_price:     Current stock price per share.
        shares:            Diluted shares outstanding.
        net_debt:          Current net debt (total debt - cash).
        discount_rate:     WACC as decimal.
        terminal_growth:   Terminal growth rate as decimal.
        projection_years:  Number of projection years (default 10).
        annual_debt_paydown: Annual debt reduction in dollars.

    Returns:
        dict with implied_growth (%), direction, message.
    """
    if not (base_revenue and base_revenue > 0 and fcf_margin and fcf_margin > 0
            and current_price and current_price > 0 and shares and shares > 0):
        return {"implied_growth": None, "direction": "error",
                "message": "Insufficient data for Reverse Revenue DCF"}

    net_debt_at_terminal = max(net_debt - annual_debt_paydown * projection_years, 0.0)
    _rev_share_chg_r = max(-0.20, min(0.20, annual_share_change))
    terminal_shares_rev = max(shares * (1 + _rev_share_chg_r) ** projection_years, shares * 0.01)
    target_equity_value = current_price * terminal_shares_rev
    target_ev = target_equity_value + net_debt_at_terminal

    disc_offset = 0.5 if use_mid_year_discounting else 0.0

    def _ev_at_rev_growth(g: float) -> float:
        revenue = base_revenue
        total_pv = 0.0
        for year in range(1, projection_years + 1):
            revenue = revenue * (1 + g)
            fcf = revenue * fcf_margin
            total_pv += fcf / (1 + discount_rate) ** (year - disc_offset)
        terminal_fcf = revenue * fcf_margin * (1 + terminal_growth)
        tv = terminal_fcf / (discount_rate - terminal_growth)
        tv_pv = tv / (1 + discount_rate) ** (projection_years - disc_offset)
        return total_pv + tv_pv

    lo, hi = -0.30, 1.00
    ev_lo = _ev_at_rev_growth(lo)
    ev_hi = _ev_at_rev_growth(hi)

    if ev_lo > target_ev:
        return {
            "implied_growth": None,
            "direction": "below_floor",
            "message": (
                "Market price implies worse than −30% annual revenue decline — "
                "stock appears deeply undervalued relative to current revenue."
            ),
        }
    if ev_hi < target_ev:
        return {
            "implied_growth": None,
            "direction": "above_ceiling",
            "message": (
                "Market price implies more than +100% annual revenue growth — "
                "stock is priced for extraordinary expectations."
            ),
        }

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _ev_at_rev_growth(mid) < target_ev:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break

    implied_g = (lo + hi) / 2.0
    return {
        "implied_growth": round(implied_g * 100, 2),
        "direction": "solved",
        "message": None,
    }


# ---------------------------------------------------------------------------
# Industry → preferred DCF model detection
# ---------------------------------------------------------------------------

_UTILITY_KEYWORDS = {
    "utilities", "utility", "electric", "gas distribution", "water utility",
    "natural gas", "power generation", "regulated",
}

def infer_dcf_model(profile: dict, segment: str = "") -> str:
    """
    Return the preferred DCF model type for a ticker based on its sector/segment.

    Returns one of: "Earnings", "FCF", "Revenue"
    """
    sector   = (profile.get("sector")   or "").lower()
    industry = (profile.get("industry") or "").lower()
    seg      = (segment or "").lower()

    combined = f"{sector} {industry} {seg}"
    if any(k in combined for k in _UTILITY_KEYWORDS):
        return "Earnings"
    return "FCF"


# ---------------------------------------------------------------------------
# Earnings Power DCF  (utilities and regulated businesses)
# ---------------------------------------------------------------------------

def compute_earnings_dcf_valuation(
    income_statements: list[dict],
    profile: dict,
    balance_sheets: list[dict] | None = None,
    analyst_estimates: list[dict] | None = None,
    earnings_growth_override: float | None = None,
    discount_rate_override: float | None = None,
    terminal_pe_override: float | None = None,
    projection_years: int = 10,
    risk_free_rate: float | None = None,
    growth_stages: list[dict] | None = None,
    use_mid_year_discounting: bool = True,
) -> dict:
    """
    Earnings Power DCF — projects diluted EPS, discounts annual dividends
    (EPS × payout ratio) back to present, then adds a terminal value based
    on a P/E multiple applied to year-N EPS.

    Intrinsic Value = Σ [EPS_t × payout / (1+r)^t]  +  [EPS_N × terminal_PE / (1+r)^N]

    Appropriate for regulated utilities and similar businesses where:
    - FCF is distorted by growth capex
    - Earnings are stable and dividend-oriented
    - The market prices on P/E rather than EV/FCF multiples
    """
    warnings_out: list[str] = []

    # ---- EPS history --------------------------------------------------------
    eps_values: list[float] = []
    for stmt in income_statements[:5]:
        eps = _safe_float(stmt.get("epsdiluted") or stmt.get("eps"))
        if eps is not None and eps > 0:
            eps_values.append(eps)

    if len(eps_values) < 2:
        return {"dcf_price": None,
                "warnings": ["Insufficient positive EPS history for Earnings DCF"],
                "assumptions": {}, "has_data": False}

    # Weighted median EPS (most-recent year weighted highest)
    n = len(eps_values)
    weights = list(range(n, 0, -1))
    paired = sorted(zip(eps_values, weights), key=lambda x: x[0])
    total_w = sum(weights)
    cum = 0
    base_eps = paired[len(paired) // 2][0]
    for val, w in paired:
        cum += w
        if cum >= total_w / 2:
            base_eps = val
            break

    if base_eps <= 0:
        return {"dcf_price": None,
                "warnings": ["Weighted median EPS is zero or negative"],
                "assumptions": {}, "has_data": False}

    # ---- Shares outstanding -------------------------------------------------
    shares = None
    if income_statements:
        shares = _safe_float(
            income_statements[0].get("weightedAverageShsOutDil")
            or income_statements[0].get("weightedAverageShsOut")
        )
    if not shares or shares <= 0:
        return {"dcf_price": None, "warnings": ["No shares outstanding data"],
                "assumptions": {}, "has_data": False}

    # ---- Payout ratio -------------------------------------------------------
    payout_ratio = 0.50   # sensible default for regulated utilities
    last_div = _safe_float(profile.get("lastDiv"))
    if last_div and last_div > 0 and base_eps > 0:
        payout_ratio = min(0.95, max(0.0, last_div / base_eps))
    else:
        # Try income statement dividends-per-share
        dps = _safe_float(
            income_statements[0].get("dividendPerShare")
            if income_statements else None
        )
        if dps and dps > 0 and base_eps > 0:
            payout_ratio = min(0.95, max(0.0, dps / base_eps))

    # ---- Historical EPS CAGR ------------------------------------------------
    hist_eps_growth = None
    pos_eps = [e for e in eps_values if e > 0]
    if len(pos_eps) >= 3:
        oldest, newest = pos_eps[-1], pos_eps[0]
        span = len(pos_eps) - 1
        if oldest > 0 and newest > 0:
            hist_eps_growth = (newest / oldest) ** (1 / span) - 1

    # ---- Analyst EPS growth (from forward estimates) ------------------------
    analyst_eps_growth = None
    analyst_num = None
    if analyst_estimates:
        sorted_est = sorted(analyst_estimates, key=lambda x: x.get("date", ""))
        eps_ests = [
            (e["date"], e.get("epsAvg") or e.get("epsDilutedAvg"))
            for e in sorted_est
            if (e.get("epsAvg") or e.get("epsDilutedAvg"))
        ]
        eps_ests = [(d, v) for d, v in eps_ests if v and v > 0]
        if len(eps_ests) >= 2:
            first_e, last_e = eps_ests[0][1], eps_ests[-1][1]
            span_e = len(eps_ests) - 1
            if first_e > 0 and last_e > 0:
                analyst_eps_growth = (last_e / first_e) ** (1 / span_e) - 1
                analyst_num = max(
                    (e.get("numAnalystsEps", 0) for e in sorted_est), default=None
                )

    # ---- WACC (same logic as FCF DCF) ---------------------------------------
    if risk_free_rate is None:
        risk_free_rate = 0.043
    equity_risk_premium = 0.055
    beta = _safe_float(profile.get("beta")) or 1.0
    if beta < 0.2:
        beta = 0.4          # utility-appropriate minimum
        warnings_out.append("Beta unusually low — floored at 0.4 for utility")
    elif beta > 3.0:
        warnings_out.append(f"Beta very high ({beta:.2f}) — cost of equity will be elevated")
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    market_cap = _safe_float(profile.get("mktCap") or profile.get("marketCap")) or 0
    balance_sheets = balance_sheets or []
    total_debt_wacc, cost_of_debt, effective_tax_rate = 0.0, 0.05, 0.25
    wacc_breakdown: dict = {}

    if balance_sheets and income_statements and market_cap > 0:
        bs, inc = balance_sheets[0], income_statements[0]
        total_debt_wacc = _safe_float(bs.get("totalDebt") or bs.get("longTermDebt")) or 0
        interest_exp = abs(_safe_float(inc.get("interestExpense")) or 0)
        pretax = _safe_float(inc.get("incomeBeforeTax")) or 0
        tax = _safe_float(inc.get("incomeTaxExpense")) or 0
        if total_debt_wacc > 0 and interest_exp > 0:
            cost_of_debt = max(0.02, min(0.20, interest_exp / total_debt_wacc))
        if pretax > 0 and tax > 0:
            effective_tax_rate = min(0.40, max(0.0, tax / pretax))

    total_capital = market_cap + total_debt_wacc
    if total_capital > 0 and total_debt_wacc > 0:
        w_eq = market_cap / total_capital
        w_de = total_debt_wacc / total_capital
        default_wacc = w_eq * cost_of_equity + w_de * cost_of_debt * (1 - effective_tax_rate)
        wacc_breakdown = {
            "cost_of_equity": round(cost_of_equity * 100, 2),
            "cost_of_debt": round(cost_of_debt * 100, 2),
            "effective_tax_rate": round(effective_tax_rate * 100, 1),
            "weight_equity": round(w_eq * 100, 1),
            "weight_debt": round(w_de * 100, 1),
        }
    else:
        default_wacc = cost_of_equity
        wacc_breakdown = {
            "cost_of_equity": round(cost_of_equity * 100, 2),
            "cost_of_debt": 0, "effective_tax_rate": round(effective_tax_rate * 100, 1),
            "weight_equity": 100.0, "weight_debt": 0.0,
        }

    discount_rate = discount_rate_override if discount_rate_override is not None else default_wacc

    # ---- Growth rate --------------------------------------------------------
    if earnings_growth_override is not None:
        growth_rate = earnings_growth_override
    elif hist_eps_growth is not None:
        growth_rate = max(-0.05, min(0.15, hist_eps_growth))
        if hist_eps_growth > 0.15:
            warnings_out.append(f"Historical EPS CAGR ({hist_eps_growth*100:.1f}%) capped at 15%")
    else:
        growth_rate = 0.05
        warnings_out.append("Could not compute historical EPS growth — using 5% default")

    # ---- Terminal P/E -------------------------------------------------------
    default_terminal_pe = 16.0   # typical regulated utility
    terminal_pe = terminal_pe_override if terminal_pe_override is not None else default_terminal_pe

    # ---- Project EPS (multi-stage or flat) ----------------------------------
    # Mid-year discounting for the dividend stream.
    disc_offset = 0.5 if use_mid_year_discounting else 0.0
    projected: list[dict] = []
    eps = base_eps
    if growth_stages:
        yr = 1
        for stage in growth_stages:
            r, yrs = stage["rate"], stage["years"]
            for _ in range(yrs):
                if yr > projection_years:
                    break
                eps = eps * (1 + r)
                div = eps * payout_ratio
                projected.append({
                    "year": yr, "eps": round(eps, 4),
                    "dividend": round(div, 4),
                    "present_value": round(div / (1 + discount_rate) ** (yr - disc_offset), 4),
                    "stage_rate": round(r * 100, 1),
                })
                yr += 1
    else:
        for yr in range(1, projection_years + 1):
            eps = eps * (1 + growth_rate)
            div = eps * payout_ratio
            projected.append({
                "year": yr, "eps": round(eps, 4),
                "dividend": round(div, 4),
                "present_value": round(div / (1 + discount_rate) ** (yr - disc_offset), 4),
            })

    # ---- Terminal value (P/E on terminal EPS) --------------------------------
    terminal_eps = projected[-1]["eps"]
    terminal_value_ps = terminal_eps * terminal_pe          # per share
    terminal_pv = terminal_value_ps / (1 + discount_rate) ** (projection_years - disc_offset)

    # ---- Intrinsic value -----------------------------------------------------
    sum_pv_divs = sum(p["present_value"] for p in projected)
    intrinsic_value = sum_pv_divs + terminal_pv
    terminal_pct = (terminal_pv / intrinsic_value * 100) if intrinsic_value > 0 else 0

    current_price = _safe_float(profile.get("price")) or 0
    upside_pct = round((intrinsic_value / current_price - 1) * 100, 1) if current_price > 0 else None

    # ---- Sensitivity table (EPS growth % vs terminal P/E) -------------------
    growth_steps = [-0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    pe_steps     = [12, 14, 16, 18, 20, 22]
    sensitivity: list[dict] = []
    for g in growth_steps:
        row: dict = {"growth_rate": round(g * 100, 1)}
        for pe in pe_steps:
            _eps2 = base_eps
            _pv2  = 0.0
            for yr2 in range(1, projection_years + 1):
                _eps2 = _eps2 * (1 + g)
                _pv2 += (_eps2 * payout_ratio) / (1 + discount_rate) ** (yr2 - disc_offset)
            _tv2   = (_eps2 * pe) / (1 + discount_rate) ** (projection_years - disc_offset)
            _price2 = round(_pv2 + _tv2, 2)
            row[f"{pe}x"] = _price2 if _price2 > 0 else "N/A"
        sensitivity.append(row)

    return {
        "dcf_price": round(intrinsic_value, 2),
        "upside_pct": upside_pct,
        "current_price": current_price,
        "terminal_pct": round(terminal_pct, 1),
        "projected_earnings": projected,
        "terminal_eps": round(terminal_eps, 4),
        "terminal_pe": terminal_pe,
        "terminal_value_per_share": round(terminal_value_ps, 2),
        "terminal_pv": round(terminal_pv, 2),
        "sensitivity": sensitivity,
        "sensitivity_pe_steps": [f"{pe}x" for pe in pe_steps],
        "warnings": warnings_out,
        "has_data": True,
        "assumptions": {
            "base_eps": round(base_eps, 4),
            "growth_rate": round(growth_rate * 100, 2),
            "discount_rate": round(discount_rate * 100, 2),
            "terminal_pe": terminal_pe,
            "payout_ratio": round(payout_ratio * 100, 1),
            "shares_outstanding": shares,
            "beta": beta,
            "wacc_breakdown": wacc_breakdown,
            "risk_free_rate": round(risk_free_rate * 100, 3) if risk_free_rate else None,
            "hist_eps_growth": round(hist_eps_growth * 100, 2) if hist_eps_growth is not None else None,
            "analyst_eps_growth": round(analyst_eps_growth * 100, 2) if analyst_eps_growth is not None else None,
            "analyst_num_analysts": analyst_num,
            "mid_year_discounting": bool(use_mid_year_discounting),
        },
    }


def compute_reverse_earnings_dcf(
    base_eps: float,
    payout_ratio: float,
    current_price: float,
    discount_rate: float,
    terminal_pe: float,
    projection_years: int = 10,
    use_mid_year_discounting: bool = True,
) -> dict:
    """
    Reverse Earnings DCF: solve for the implied flat EPS growth rate that
    makes the Earnings Power model value equal the current stock price.
    Uses bisection (60 iterations).
    """
    if not (base_eps > 0 and current_price > 0 and 0 <= payout_ratio <= 1):
        return {"implied_growth": None, "direction": "error",
                "message": "Insufficient data for Reverse Earnings DCF"}

    disc_offset = 0.5 if use_mid_year_discounting else 0.0

    def _price_at_growth(g: float) -> float:
        eps = base_eps
        pv_divs = 0.0
        for yr in range(1, projection_years + 1):
            eps = eps * (1 + g)
            pv_divs += (eps * payout_ratio) / (1 + discount_rate) ** (yr - disc_offset)
        terminal_pv = (eps * terminal_pe) / (1 + discount_rate) ** (projection_years - disc_offset)
        return pv_divs + terminal_pv

    lo, hi = -0.20, 0.50
    p_lo, p_hi = _price_at_growth(lo), _price_at_growth(hi)

    if p_lo > current_price:
        return {
            "implied_growth": None, "direction": "below_floor",
            "message": (
                "Market price implies worse than −20% annual EPS decline — "
                "stock appears deeply undervalued vs. current earnings."
            ),
        }
    if p_hi < current_price:
        return {
            "implied_growth": None, "direction": "above_ceiling",
            "message": (
                "Market price implies more than +50% annual EPS growth — "
                "priced for extraordinary earnings expansion."
            ),
        }

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _price_at_growth(mid) < current_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break

    return {
        "implied_growth": round((lo + hi) / 2.0 * 100, 2),
        "direction": "solved",
        "message": None,
    }


def analyze_ticker(ticker: str, fmp_client, universe_info: dict | None = None,
                    history_years: int | None = None,
                    risk_free_rate: float | None = None) -> dict:
    """
    Pull all ratios for a ticker, compute percentile ranks against its own history.

    Args:
        ticker: Stock ticker symbol.
        fmp_client: An initialized FMPClient instance.
        universe_info: Optional dict with company_name, segment, sub_segment.
        history_years: Override for number of years of history (default from config).

    Returns:
        Analysis dict with ticker, company_name, segment, metrics (each with
        current, avg, low, high, percentile, years_of_data),
        composite_score, and opportunity_flags.
    """
    config = _load_scoring_config()
    if history_years is None:
        history_years = config.get("history_years", 10)
    higher_is_better = set(config.get("higher_is_better", []))

    # Fetch data from FMP
    ratios = fmp_client.get_financial_ratios(ticker, period="annual", limit=history_years + 1)
    key_metrics = fmp_client.get_key_metrics(ticker, period="annual", limit=history_years + 1)
    profile = fmp_client.get_company_profile(ticker)

    company_name = (
        (universe_info or {}).get("company_name")
        or profile.get("companyName", ticker)
    )
    segment = (universe_info or {}).get("segment", "")

    # Build metric analyses
    metrics = {}

    # Process ratio-based metrics
    for metric_name, fmp_field in RATIO_FIELD_MAP.items():
        series = _extract_metric_series(ratios, fmp_field)
        current = _safe_float(ratios[0].get(fmp_field)) if ratios else None

        # Filter out negative values for valuation ratios (meaningless when negative)
        if metric_name in POSITIVE_ONLY_METRICS:
            series = [v for v in series if v > 0]
            if current is not None and current <= 0:
                current = None

        if not series or current is None:
            continue

        percentile = compute_percentile_rank(current, series)
        clean = [v for v in series if v is not None]

        metrics[metric_name] = _build_metric_entry(current, clean, percentile)

    # Override / supplement with key metrics data
    for metric_name, fmp_field in KEY_METRICS_FIELD_MAP.items():
        series = _extract_metric_series(key_metrics, fmp_field)
        current = _safe_float(key_metrics[0].get(fmp_field)) if key_metrics else None

        # Filter out negative values for valuation ratios
        if metric_name in POSITIVE_ONLY_METRICS:
            series = [v for v in series if v > 0]
            if current is not None and current <= 0:
                current = None

        if not series or current is None:
            continue

        percentile = compute_percentile_rank(current, series)
        clean = [v for v in series if v is not None]

        metrics[metric_name] = _build_metric_entry(current, clean, percentile)

    # Operating ratio (operating expenses / revenue) — key for railroads, industrials
    income = fmp_client.get_income_statement(ticker, period="annual", limit=history_years + 1)
    if income:
        or_series = []
        for stmt in income:
            rev = _safe_float(stmt.get("revenue"))
            op_inc = _safe_float(stmt.get("operatingIncome"))
            if rev and rev > 0 and op_inc is not None:
                or_series.append(1.0 - (op_inc / rev))
        if or_series:
            current_or = or_series[0]
            percentile_or = compute_percentile_rank(current_or, or_series)
            metrics["operating_ratio"] = _build_metric_entry(current_or, or_series, percentile_or)

    # Compute growth metrics from income statements
    if income and len(income) >= 2:
        revenues = _extract_metric_series(income, "revenue")
        rev_growth = _compute_growth(revenues)
        if rev_growth is not None:
            metrics["revenue_growth_yoy"] = {
                "current": rev_growth,
                "hist_avg": None,
                "hist_low": None,
                "hist_high": None,
                "percentile": None,
                "years_of_data": len(revenues),
            }

        earnings = _extract_metric_series(income, "netIncome")
        earn_growth = _compute_growth(earnings)
        if earn_growth is not None:
            metrics["earnings_growth_yoy"] = {
                "current": earn_growth,
                "hist_avg": None,
                "hist_low": None,
                "hist_high": None,
                "percentile": None,
                "years_of_data": len(earnings),
            }

    # Fetch cash flow and balance sheet for fundamentals context
    cash_flow = fmp_client.get_cash_flow(ticker, period="annual", limit=history_years + 1)
    balance_sheet = fmp_client.get_balance_sheet(ticker, period="annual", limit=history_years + 1)

    # FCF growth metric
    if cash_flow and len(cash_flow) >= 2:
        fcf_values = _extract_metric_series(cash_flow, "freeCashFlow")
        fcf_growth = _compute_growth(fcf_values)
        if fcf_growth is not None:
            metrics["fcf_growth_yoy"] = {
                "current": fcf_growth,
                "hist_avg": None,
                "hist_low": None,
                "hist_high": None,
                "percentile": None,
                "years_of_data": len(fcf_values),
            }

    # Fundamentals context — explains the "why" behind valuation ratios
    earnings_context = build_fundamentals_context(income, cash_flow, balance_sheet, history_years)

    # Composite score
    composite = compute_composite_score(metrics, config.get("weights", {}), higher_is_better)

    # Opportunity flags
    flags = flag_opportunities(metrics, higher_is_better, config.get("opportunity_thresholds", {}))

    # Mean reversion implied prices
    implied_prices = compute_implied_prices(
        metrics=metrics,
        current_price=profile.get("price"),
        income_statements=income,
        cash_flow_statements=cash_flow,
        key_metrics_data=key_metrics,
        fundamentals_context=earnings_context,
    )

    # Fetch analyst estimates for DCF growth rate context
    analyst_estimates = fmp_client.get_analyst_estimates(ticker, period="annual", limit=15)

    # DCF valuation (default assumptions — UI lets user override via sliders)
    dcf_valuation = compute_dcf_valuation(
        cash_flow_statements=cash_flow,
        income_statements=income,
        profile=profile,
        balance_sheets=balance_sheet,
        analyst_estimates=analyst_estimates,
        risk_free_rate=risk_free_rate,
    )

    # Determine the most recent filing period for display context
    _latest_filing_date = None
    _latest_filing_period = None
    if income:
        _stmt = income[0]
        _latest_filing_date = _stmt.get("fillingDate") or _stmt.get("acceptedDate") or _stmt.get("date")
        _cal_year = _stmt.get("calendarYear", "")
        _period = _stmt.get("period", "")  # e.g. "FY", "Q3"
        if _cal_year and _period:
            _latest_filing_period = f"{_period} {_cal_year}"
        elif _cal_year:
            _latest_filing_period = f"FY {_cal_year}"

    return {
        "ticker": ticker,
        "company_name": company_name,
        "segment": segment,
        "current_price": profile.get("price"),
        "market_cap": profile.get("mktCap"),
        "metrics": metrics,
        "composite_score": composite,
        "opportunity_flags": flags,
        "earnings_context": earnings_context,
        "implied_prices": implied_prices,
        "dcf_valuation": dcf_valuation,
        "profile": profile,
        "cash_flow_statements": cash_flow,
        "income_statements": income,
        "balance_sheets": balance_sheet,
        "analyst_estimates": analyst_estimates,
        "latest_filing_date": _latest_filing_date,
        "latest_filing_period": _latest_filing_period,
    }


def compute_composite_score(metrics: dict, weights: dict,
                              higher_is_better: set | None = None) -> float | None:
    """
    Compute a weighted average of percentile ranks.

    For "higher is better" metrics, the percentile is inverted (100 - percentile)
    so that a LOW composite score means the stock is potentially undervalued/strong.

    Args:
        metrics: Dict of metric_name -> {current, percentile, ...}.
        weights: Dict of metric_name -> weight (0-1). Must sum to ~1.
        higher_is_better: Set of metric names where higher values are favorable.

    Returns:
        Composite score (0-100) or None if insufficient data.
    """
    if higher_is_better is None:
        config = _load_scoring_config()
        higher_is_better = set(config.get("higher_is_better", []))

    total_weight = 0.0
    weighted_sum = 0.0

    for metric_name, weight in weights.items():
        if metric_name not in metrics:
            continue
        pct = metrics[metric_name].get("percentile")
        if pct is None:
            continue

        # Invert for "higher is better" so low composite = good
        if metric_name in higher_is_better:
            pct = 100.0 - pct

        weighted_sum += pct * weight
        total_weight += weight

    if total_weight == 0:
        return None

    return round(weighted_sum / total_weight, 1)


def flag_opportunities(metrics: dict, higher_is_better: set | None = None,
                        thresholds: dict | None = None) -> list[str]:
    """
    Generate human-readable flags for notable metric positions.

    Args:
        metrics: Dict of metric_name -> {current, percentile, hist_low, hist_high, ...}.
        higher_is_better: Set of metric names where higher values are favorable.
        thresholds: Dict with 'low_percentile' and 'high_percentile' keys.

    Returns:
        List of flag strings like "P/E at 5-year low (12th percentile)".
    """
    if higher_is_better is None:
        config = _load_scoring_config()
        higher_is_better = set(config.get("higher_is_better", []))

    if thresholds is None:
        thresholds = {"low_percentile": 20, "high_percentile": 80}

    low_thresh = thresholds.get("low_percentile", 20)
    high_thresh = thresholds.get("high_percentile", 80)

    # Pretty names for display
    pretty_names = {
        "pe_ratio": "P/E",
        "ps_ratio": "P/S",
        "pb_ratio": "P/B",
        "ev_ebitda": "EV/EBITDA",
        "ev_revenue": "EV/Revenue",
        "fcf_yield": "FCF Yield",
        "roe": "ROE",
        "roa": "ROA",
        "gross_margin": "Gross Margin",
        "operating_margin": "Operating Margin",
        "net_margin": "Net Margin",
        "debt_to_equity": "Debt/Equity",
        "current_ratio": "Current Ratio",
        "quick_ratio": "Quick Ratio",
        "interest_coverage": "Interest Coverage",
        "earnings_yield": "Earnings Yield",
        "price_to_fcf": "P/FCF",
        "peg_ratio": "PEG Ratio",
        "roic": "ROIC",
        "cash_conversion_cycle": "Cash Conversion Cycle",
        "debt_to_assets": "Debt/Assets",
        "fcf_growth_yoy": "FCF Growth YoY",
    }

    flags = []

    for metric_name, data in metrics.items():
        pct = data.get("percentile")
        if pct is None:
            continue

        display = pretty_names.get(metric_name, metric_name.replace("_", " ").title())
        ordinal = _ordinal(int(pct))

        if metric_name in higher_is_better:
            # Higher is better: high percentile = good, low = caution
            if pct >= high_thresh:
                flags.append(f"{display} historically strong ({ordinal} percentile)")
            elif pct <= low_thresh:
                flags.append(f"{display} near historical low ({ordinal} percentile) -- caution")
        else:
            # Lower is better (valuation metrics): low percentile = cheap
            if pct <= low_thresh:
                flags.append(f"{display} near 5-year low ({ordinal} percentile)")
            elif pct >= high_thresh:
                flags.append(f"{display} near 5-year high ({ordinal} percentile) -- caution")

    return flags


def _ordinal(n: int) -> str:
    """Convert integer to ordinal string (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
