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
    "dividend_yield": "dividendYieldPercentage",
    "dividend_payout_ratio": "dividendPayoutRatio",
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


def compute_dcf_valuation(cash_flow_statements: list[dict],
                           income_statements: list[dict],
                           profile: dict,
                           balance_sheets: list[dict] | None = None,
                           growth_rate_override: float | None = None,
                           discount_rate_override: float | None = None,
                           terminal_growth_override: float | None = None,
                           projection_years: int = 10,
                           analyst_estimates: list[dict] | None = None,
                           growth_stages: list[dict] | None = None) -> dict:
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

    if len(fcf_values) < 2:
        return {"dcf_price": None, "warnings": ["Insufficient FCF history for DCF"],
                "assumptions": {}, "has_data": False}

    # Use average of last 2 years as base (smooths one-off spikes)
    base_fcf = np.mean(fcf_values[:2])
    if base_fcf <= 0:
        # Try 3-year average as fallback
        base_fcf = np.mean(fcf_values[:3]) if len(fcf_values) >= 3 else base_fcf
        if base_fcf <= 0:
            return {"dcf_price": None, "warnings": ["Negative average FCF — DCF not applicable"],
                    "assumptions": {}, "has_data": False}
        warnings.append("Recent FCF mixed — using 3-year average as base")

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

    # --- Net debt (total debt - cash) for EV → equity bridge ---
    net_debt = 0.0  # default: assume no net debt if data unavailable
    balance_sheets = balance_sheets or []
    if balance_sheets:
        latest_bs = balance_sheets[0]
        total_debt = _safe_float(latest_bs.get("totalDebt") or latest_bs.get("longTermDebt"))
        cash = _safe_float(latest_bs.get("cashAndCashEquivalents") or
                           latest_bs.get("cashAndShortTermInvestments"))
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash  # positive = net debt, negative = net cash
        else:
            warnings.append("Missing debt/cash data — skipping net debt adjustment")
    else:
        warnings.append("No balance sheet data — skipping net debt adjustment")

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

    # Discount rate (WACC approximation): risk-free + beta × equity risk premium
    risk_free_rate = 0.043  # ~10yr Treasury as of 2026
    equity_risk_premium = 0.055
    beta = _safe_float(profile.get("beta")) or 1.0
    if beta < 0.3:
        beta = 1.0
        warnings.append("Beta unusually low — defaulting to 1.0")
    elif beta > 3.0:
        warnings.append(f"Beta very high ({beta:.2f}) — WACC will be elevated")

    default_wacc = risk_free_rate + beta * equity_risk_premium
    discount_rate = discount_rate_override if discount_rate_override is not None else default_wacc

    # Terminal growth rate
    terminal_growth = terminal_growth_override if terminal_growth_override is not None else 0.025

    # Safety: discount rate must exceed terminal growth
    if discount_rate <= terminal_growth:
        warnings.append("Discount rate must exceed terminal growth — adjusting terminal growth down")
        terminal_growth = discount_rate - 0.01

    # --- Project FCFs (multi-stage or single-stage) ---
    projected_fcfs = []
    fcf = base_fcf
    if growth_stages:
        # Multi-stage: each stage has {"rate": float, "years": int}
        year_counter = 1
        for stage in growth_stages:
            stage_rate = stage["rate"]
            stage_years = stage["years"]
            for _ in range(stage_years):
                if year_counter > projection_years:
                    break
                fcf = fcf * (1 + stage_rate)
                pv = fcf / (1 + discount_rate) ** year_counter
                projected_fcfs.append({
                    "year": year_counter,
                    "fcf": round(fcf, 0),
                    "present_value": round(pv, 0),
                    "stage_rate": round(stage_rate * 100, 1),
                })
                year_counter += 1
    else:
        # Single-stage (backward compatible)
        for year in range(1, projection_years + 1):
            fcf = fcf * (1 + growth_rate)
            pv = fcf / (1 + discount_rate) ** year
            projected_fcfs.append({
                "year": year,
                "fcf": round(fcf, 0),
                "present_value": round(pv, 0),
            })

    # --- Terminal value ---
    terminal_fcf = projected_fcfs[-1]["fcf"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** projection_years

    # --- Intrinsic value ---
    sum_pv_fcfs = sum(p["present_value"] for p in projected_fcfs)
    enterprise_value = sum_pv_fcfs + terminal_pv
    equity_value = enterprise_value - net_debt  # EV - net debt = equity value
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
                "hist_fcf_growth": round(hist_fcf_growth * 100, 1) if hist_fcf_growth is not None else None,
                "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
                "analyst_num_analysts": analyst_num_analysts,
            },
            "projected_fcfs": projected_fcfs,
            "sensitivity": [],
        }
    intrinsic_per_share = equity_value / shares

    current_price = _safe_float(profile.get("price"))
    upside = None
    if current_price and current_price > 0:
        upside = round((intrinsic_per_share / current_price - 1) * 100, 1)

    # --- Terminal value as % of total (sanity check) ---
    terminal_pct = (terminal_pv / enterprise_value * 100) if enterprise_value > 0 else 0
    if terminal_pct > 80:
        warnings.append(f"Terminal value is {terminal_pct:.0f}% of total — valuation heavily depends on long-term assumptions")

    # --- Sensitivity table ---
    # Growth rates: centered on selected, +/- 4 steps of 2%
    growth_steps = [growth_rate + (i - 4) * 0.02 for i in range(9)]
    growth_steps = [g for g in growth_steps if -0.10 <= g <= 0.30]  # clamp
    # Discount rates: centered on selected, +/- 3 steps of 1%
    discount_steps = [discount_rate + (i - 3) * 0.01 for i in range(7)]
    discount_steps = [d for d in discount_steps if 0.04 <= d <= 0.20]  # clamp

    sensitivity = []
    for g in growth_steps:
        row = {"growth_rate": round(g * 100, 1)}
        for d in discount_steps:
            if d <= terminal_growth + 0.005:  # discount must exceed terminal growth
                row[f"{d*100:.1f}%"] = "N/A"
                continue
            # Quick DCF calc for this combo
            f = base_fcf
            pv_sum = 0
            for yr in range(1, projection_years + 1):
                f = f * (1 + g)
                pv_sum += f / (1 + d) ** yr
            t_fcf = f * (1 + terminal_growth)
            t_val = t_fcf / (d - terminal_growth) if d > terminal_growth else 0
            t_pv = t_val / (1 + d) ** projection_years
            ev = pv_sum + t_pv
            eq = ev - net_debt
            price = max(eq, 0) / shares
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
            "hist_fcf_growth": round(hist_fcf_growth * 100, 1) if hist_fcf_growth is not None else None,
            "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
            "analyst_num_analysts": analyst_num_analysts,
        },
        "projected_fcfs": projected_fcfs,
        "terminal_value": round(terminal_value, 0),
        "terminal_pv": round(terminal_pv, 0),
        "terminal_pct": round(terminal_pct, 1),
        "sensitivity": sensitivity,
        "sensitivity_discount_rates": [f"{d*100:.1f}%" for d in discount_steps],
        "warnings": warnings,
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
                                   growth_stages: list[dict] | None = None) -> dict:
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

    # --- Net debt ---
    net_debt = 0.0
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

    # --- Growth rate ---
    if revenue_growth_override is not None:
        revenue_growth = revenue_growth_override
    elif hist_rev_growth is not None:
        revenue_growth = max(-0.05, min(0.30, hist_rev_growth))
    else:
        revenue_growth = 0.08
        warnings.append("Could not compute historical revenue growth — using 8% default")

    # --- Target FCF margin ---
    if target_fcf_margin_override is not None:
        target_fcf_margin = target_fcf_margin_override
    elif hist_median_fcf_margin is not None and hist_median_fcf_margin > 0:
        target_fcf_margin = hist_median_fcf_margin
    else:
        target_fcf_margin = 0.10  # conservative 10% default
        warnings.append("No positive historical FCF margin — using 10% default target")

    # --- Discount rate ---
    risk_free_rate = 0.043
    equity_risk_premium = 0.055
    beta = _safe_float(profile.get("beta")) or 1.0
    if beta < 0.3:
        beta = 1.0
    default_wacc = risk_free_rate + beta * equity_risk_premium
    discount_rate = discount_rate_override if discount_rate_override is not None else default_wacc

    # --- Terminal growth ---
    terminal_growth = terminal_growth_override if terminal_growth_override is not None else 0.025
    if discount_rate <= terminal_growth:
        warnings.append("Discount rate must exceed terminal growth — adjusting terminal growth down")
        terminal_growth = discount_rate - 0.01

    # --- Project revenue → FCF ---
    projected_fcfs = []
    rev = base_revenue

    if growth_stages:
        year_counter = 1
        for stage in growth_stages:
            stage_rate = stage["rate"]
            stage_years = stage["years"]
            for _ in range(stage_years):
                if year_counter > projection_years:
                    break
                rev = rev * (1 + stage_rate)
                fcf = rev * target_fcf_margin
                pv = fcf / (1 + discount_rate) ** year_counter
                projected_fcfs.append({
                    "year": year_counter,
                    "revenue": round(rev, 0),
                    "fcf": round(fcf, 0),
                    "present_value": round(pv, 0),
                    "stage_rate": round(stage_rate * 100, 1),
                })
                year_counter += 1
    else:
        for year in range(1, projection_years + 1):
            rev = rev * (1 + revenue_growth)
            fcf = rev * target_fcf_margin
            pv = fcf / (1 + discount_rate) ** year
            projected_fcfs.append({
                "year": year,
                "revenue": round(rev, 0),
                "fcf": round(fcf, 0),
                "present_value": round(pv, 0),
            })

    # --- Terminal value ---
    terminal_fcf = projected_fcfs[-1]["fcf"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** projection_years

    # --- Intrinsic value ---
    sum_pv_fcfs = sum(p["present_value"] for p in projected_fcfs)
    enterprise_value = sum_pv_fcfs + terminal_pv
    equity_value = enterprise_value - net_debt
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
                "discount_rate": round(discount_rate * 100, 2),
                "terminal_growth": round(terminal_growth * 100, 2),
                "shares_outstanding": shares,
                "beta": beta,
                "hist_rev_growth": round(hist_rev_growth * 100, 1) if hist_rev_growth is not None else None,
                "hist_median_fcf_margin": round(hist_median_fcf_margin * 100, 1) if hist_median_fcf_margin is not None else None,
                "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
                "analyst_num_analysts": analyst_num_analysts,
            },
            "projected_fcfs": projected_fcfs,
            "sensitivity": [],
        }

    intrinsic_per_share = equity_value / shares
    current_price = _safe_float(profile.get("price"))
    upside = None
    if current_price and current_price > 0:
        upside = round((intrinsic_per_share / current_price - 1) * 100, 1)

    terminal_pct = (terminal_pv / enterprise_value * 100) if enterprise_value > 0 else 0
    if terminal_pct > 80:
        warnings.append(f"Terminal value is {terminal_pct:.0f}% of total — valuation heavily depends on long-term assumptions")

    # --- Sensitivity table (revenue growth × discount rate) ---
    growth_steps = [revenue_growth + (i - 4) * 0.02 for i in range(9)]
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
                f = r * target_fcf_margin
                pv_sum += f / (1 + d) ** yr
            t_fcf = (r * target_fcf_margin) * (1 + terminal_growth)
            t_val = t_fcf / (d - terminal_growth) if d > terminal_growth else 0
            t_pv = t_val / (1 + d) ** projection_years
            ev = pv_sum + t_pv
            eq = ev - net_debt
            price = max(eq, 0) / shares
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
            "discount_rate": round(discount_rate * 100, 2),
            "terminal_growth": round(terminal_growth * 100, 2),
            "beta": round(beta, 2),
            "projection_years": projection_years,
            "shares_outstanding": shares,
            "net_debt": round(net_debt, 0),
            "hist_rev_growth": round(hist_rev_growth * 100, 1) if hist_rev_growth is not None else None,
            "hist_median_fcf_margin": round(hist_median_fcf_margin * 100, 1) if hist_median_fcf_margin is not None else None,
            "analyst_revenue_growth": round(analyst_revenue_growth * 100, 1) if analyst_revenue_growth is not None else None,
            "analyst_num_analysts": analyst_num_analysts,
        },
        "projected_fcfs": projected_fcfs,
        "terminal_value": round(terminal_value, 0),
        "terminal_pv": round(terminal_pv, 0),
        "terminal_pct": round(terminal_pct, 1),
        "sensitivity": sensitivity,
        "sensitivity_discount_rates": [f"{d*100:.1f}%" for d in discount_steps],
        "warnings": warnings,
    }


def analyze_ticker(ticker: str, fmp_client, universe_info: dict | None = None,
                    history_years: int | None = None) -> dict:
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

    # Compute growth metrics from income statements
    income = fmp_client.get_income_statement(ticker, period="annual", limit=history_years + 1)
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
    analyst_estimates = fmp_client.get_analyst_estimates(ticker, period="annual", limit=5)

    # DCF valuation (default assumptions — UI lets user override via sliders)
    dcf_valuation = compute_dcf_valuation(
        cash_flow_statements=cash_flow,
        income_statements=income,
        profile=profile,
        balance_sheets=balance_sheet,
        analyst_estimates=analyst_estimates,
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
        "dividend_yield": "Dividend Yield",
        "dividend_payout_ratio": "Dividend Payout Ratio",
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
