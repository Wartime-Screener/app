"""
Forensic-accounting quality scores: Piotroski F-Score, Altman Z-Score, Beneish M-Score.

These three scores are the standard "quality screen" toolkit for distinguishing
genuine value from value traps:

  - Piotroski F-Score (1996): 9 binary fundamental tests. Companies scoring 8-9
    historically outperformed the value universe by ~7.5%/yr in Piotroski's
    original 1976-1996 backtest. Best as a filter applied AFTER cheapness.

  - Altman Z-Score (1968): 5-factor weighted bankruptcy predictor. Original
    accuracy was ~72% one year before bankruptcy on the 1968 sample. The
    classic version is most accurate for public manufacturers; financials,
    REITs, and asset-light services need other tools.

  - Beneish M-Score (1999): 8-ratio earnings manipulation detector. Beneish's
    original sample correctly flagged ~76% of known manipulators (e.g., Enron
    at the time). Cheap stocks with high M-Scores are statistically more
    likely to be value traps masking deteriorating fundamentals.

All three functions take pre-fetched FMP financial statements (annual, newest
first) and return a structured dict with the score, sub-component breakdown,
sector applicability flag, and a human-readable interpretation.

None of these scores apply cleanly to financials, REITs, or insurance
companies — those have fundamentally different balance sheet structures and
the formulas were not designed for them. The applicability flag warns when
a score should be interpreted with extra skepticism or ignored entirely.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    """Convert to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return None


def _safe_div(numerator, denominator, default=None):
    """Divide, returning default on zero/None denominator."""
    n = _safe_float(numerator)
    d = _safe_float(denominator)
    if n is None or d is None or d == 0:
        return default
    return n / d


def _is_financial_or_reit(sector: str, industry: str) -> bool:
    """
    Returns True if the company is a bank, insurer, or REIT.

    These three score types were not designed for these business models —
    Altman especially gives misleading results because the working-capital
    and asset-turnover terms don't apply to financial intermediation.
    """
    s = (sector or "").lower()
    i = (industry or "").lower()
    if "financial" in s or "bank" in i or "insurance" in i or "insurer" in i:
        return True
    if "real estate" in s or "reit" in i or "real estate" in i:
        return True
    return False


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------

def compute_piotroski_f_score(income_statements: list[dict],
                                balance_sheets: list[dict],
                                cash_flow_statements: list[dict],
                                sector: str = "",
                                industry: str = "") -> dict:
    """
    Compute the 9-point Piotroski F-Score.

    The score has three sections:

    Profitability (4 points):
      F1. Net Income > 0 (current year)
      F2. Operating Cash Flow > 0 (current year)
      F3. ROA improving year-over-year
      F4. CFO > Net Income (cash earnings exceed accruals — quality signal)

    Leverage / Liquidity (3 points):
      F5. Long-term debt ratio decreasing (LTD/TA lower vs prior year)
      F6. Current ratio improving year-over-year
      F7. No new shares issued (share count not growing)

    Operating Efficiency (2 points):
      F8. Gross margin improving year-over-year
      F9. Asset turnover improving year-over-year

    Returns:
        Dict with:
          score (int 0-9): total points earned
          components (list[dict]): per-test breakdown with name/value/passed
          interpretation (str): "Strong" (8-9) / "Average" (5-7) / "Weak" (0-4)
          applicability (str): "good" / "questionable" — financials/REITs flagged
          has_data (bool): False if not enough data to compute
          note (str | None): explanation when has_data is False
    """
    # Need at least 2 years of each statement type
    if (len(income_statements) < 2 or len(balance_sheets) < 2
            or len(cash_flow_statements) < 1):
        return {
            "score": None,
            "max_score": 9,
            "components": [],
            "interpretation": None,
            "applicability": None,
            "has_data": False,
            "note": "Insufficient history (need 2 years of statements)",
        }

    inc_t  = income_statements[0]   # most recent
    inc_p  = income_statements[1]   # prior year
    bs_t   = balance_sheets[0]
    bs_p   = balance_sheets[1]
    cf_t   = cash_flow_statements[0]

    # Extract raw values (None if missing)
    ni_t       = _safe_float(inc_t.get("netIncome"))
    ni_p       = _safe_float(inc_p.get("netIncome"))
    rev_t      = _safe_float(inc_t.get("revenue"))
    rev_p      = _safe_float(inc_p.get("revenue"))
    gp_t       = _safe_float(inc_t.get("grossProfit"))
    gp_p       = _safe_float(inc_p.get("grossProfit"))
    ta_t       = _safe_float(bs_t.get("totalAssets"))
    ta_p       = _safe_float(bs_p.get("totalAssets"))
    ltd_t      = _safe_float(bs_t.get("longTermDebt"))
    ltd_p      = _safe_float(bs_p.get("longTermDebt"))
    ca_t       = _safe_float(bs_t.get("totalCurrentAssets"))
    ca_p       = _safe_float(bs_p.get("totalCurrentAssets"))
    cl_t       = _safe_float(bs_t.get("totalCurrentLiabilities"))
    cl_p       = _safe_float(bs_p.get("totalCurrentLiabilities"))
    cfo_t      = _safe_float(
        cf_t.get("operatingCashFlow") or cf_t.get("netCashProvidedByOperatingActivities")
    )
    shares_t   = _safe_float(
        inc_t.get("weightedAverageShsOutDil") or inc_t.get("weightedAverageShsOut")
    )
    shares_p   = _safe_float(
        inc_p.get("weightedAverageShsOutDil") or inc_p.get("weightedAverageShsOut")
    )

    components = []

    def _add(name: str, passed: bool | None, detail: str = ""):
        components.append({
            "name": name,
            "passed": bool(passed) if passed is not None else None,
            "detail": detail,
        })

    # ---- Profitability ----
    # F1: Net Income > 0
    f1 = ni_t is not None and ni_t > 0
    _add("F1: Net Income > 0", f1, f"NI: ${(ni_t or 0)/1e9:.2f}B")

    # F2: Operating Cash Flow > 0
    f2 = cfo_t is not None and cfo_t > 0
    _add("F2: Operating Cash Flow > 0", f2, f"CFO: ${(cfo_t or 0)/1e9:.2f}B")

    # F3: ROA improving (NI_t/TA_t > NI_p/TA_p)
    roa_t = _safe_div(ni_t, ta_t)
    roa_p = _safe_div(ni_p, ta_p)
    f3 = roa_t is not None and roa_p is not None and roa_t > roa_p
    _add(
        "F3: ROA improving YoY",
        f3,
        f"ROA: {(roa_t or 0)*100:.1f}% vs {(roa_p or 0)*100:.1f}% prior",
    )

    # F4: CFO > Net Income (accrual quality)
    f4 = cfo_t is not None and ni_t is not None and cfo_t > ni_t
    _add(
        "F4: CFO > Net Income (accrual quality)",
        f4,
        f"CFO ${(cfo_t or 0)/1e9:.2f}B vs NI ${(ni_t or 0)/1e9:.2f}B",
    )

    # ---- Leverage / Liquidity ----
    # F5: Long-term debt ratio decreasing
    ltd_ratio_t = _safe_div(ltd_t, ta_t)
    ltd_ratio_p = _safe_div(ltd_p, ta_p)
    f5 = (ltd_ratio_t is not None and ltd_ratio_p is not None
          and ltd_ratio_t < ltd_ratio_p)
    _add(
        "F5: LTD/Assets decreasing YoY",
        f5,
        f"LTD/TA: {(ltd_ratio_t or 0)*100:.1f}% vs {(ltd_ratio_p or 0)*100:.1f}% prior",
    )

    # F6: Current ratio improving
    cr_t = _safe_div(ca_t, cl_t)
    cr_p = _safe_div(ca_p, cl_p)
    f6 = cr_t is not None and cr_p is not None and cr_t > cr_p
    _add(
        "F6: Current ratio improving YoY",
        f6,
        f"CR: {(cr_t or 0):.2f}x vs {(cr_p or 0):.2f}x prior",
    )

    # F7: No new shares issued (shares not growing)
    # Allow 0.5% noise for floating point / rounding
    f7 = (shares_t is not None and shares_p is not None
          and shares_t <= shares_p * 1.005)
    if shares_t is not None and shares_p is not None and shares_p > 0:
        share_chg_pct = (shares_t / shares_p - 1) * 100
    else:
        share_chg_pct = 0
    _add(
        "F7: No share dilution YoY",
        f7,
        f"Shares: {share_chg_pct:+.1f}% YoY",
    )

    # ---- Operating Efficiency ----
    # F8: Gross margin improving
    gm_t = _safe_div(gp_t, rev_t)
    gm_p = _safe_div(gp_p, rev_p)
    f8 = gm_t is not None and gm_p is not None and gm_t > gm_p
    _add(
        "F8: Gross margin improving YoY",
        f8,
        f"GM: {(gm_t or 0)*100:.1f}% vs {(gm_p or 0)*100:.1f}% prior",
    )

    # F9: Asset turnover improving (Sales/Assets)
    at_t = _safe_div(rev_t, ta_t)
    at_p = _safe_div(rev_p, ta_p)
    f9 = at_t is not None and at_p is not None and at_t > at_p
    _add(
        "F9: Asset turnover improving YoY",
        f9,
        f"AT: {(at_t or 0):.2f}x vs {(at_p or 0):.2f}x prior",
    )

    score = sum(1 for c in components if c["passed"])

    if score >= 8:
        interpretation = "Strong"
    elif score >= 5:
        interpretation = "Average"
    else:
        interpretation = "Weak"

    applicability = "questionable" if _is_financial_or_reit(sector, industry) else "good"

    return {
        "score": score,
        "max_score": 9,
        "components": components,
        "interpretation": interpretation,
        "applicability": applicability,
        "has_data": True,
        "note": (
            "Financials, banks, and REITs have non-standard balance sheet structures — "
            "treat the F-Score as suggestive only."
            if applicability == "questionable" else None
        ),
    }


# ---------------------------------------------------------------------------
# Altman Z-Score
# ---------------------------------------------------------------------------

def compute_altman_z_score(income_statements: list[dict],
                            balance_sheets: list[dict],
                            profile: dict,
                            sector: str = "",
                            industry: str = "") -> dict:
    """
    Compute the original Altman Z-Score (5-factor public manufacturer model).

    Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

    Where:
      A = Working Capital / Total Assets
      B = Retained Earnings / Total Assets
      C = EBIT / Total Assets
      D = Market Value of Equity / Total Liabilities
      E = Sales / Total Assets

    Bands:
      Z > 2.99   → Safe Zone (low bankruptcy risk)
      1.81-2.99  → Grey Zone (caution)
      Z < 1.81   → Distress Zone (elevated bankruptcy risk)

    NOT applicable to financials, REITs, or insurance — their balance sheet
    structure (high leverage as a feature, not a bug) breaks the model.

    Returns:
        Dict with:
          score (float): the computed Z value
          classification (str): "Safe" / "Grey" / "Distress"
          components (list[dict]): A through E with raw values and contributions
          applicability (str): "good" / "questionable" / "not_applicable"
          has_data (bool)
          note (str | None)
    """
    if not income_statements or not balance_sheets or not profile:
        return {
            "score": None, "classification": None, "components": [],
            "applicability": None, "has_data": False,
            "note": "Insufficient data for Altman Z-Score",
        }

    inc = income_statements[0]
    bs  = balance_sheets[0]

    total_assets       = _safe_float(bs.get("totalAssets"))
    total_liabilities  = _safe_float(bs.get("totalLiabilities"))
    current_assets     = _safe_float(bs.get("totalCurrentAssets"))
    current_liabs      = _safe_float(bs.get("totalCurrentLiabilities"))
    retained_earnings  = _safe_float(bs.get("retainedEarnings"))
    revenue            = _safe_float(inc.get("revenue"))
    ebit               = _safe_float(inc.get("ebit") or inc.get("operatingIncome"))
    market_cap         = _safe_float(profile.get("mktCap") or profile.get("marketCap"))

    if total_assets is None or total_assets == 0:
        return {
            "score": None, "classification": None, "components": [],
            "applicability": None, "has_data": False,
            "note": "Total assets unavailable",
        }

    # Applicability check — financials/REITs are not applicable, period
    if _is_financial_or_reit(sector, industry):
        return {
            "score": None,
            "classification": None,
            "components": [],
            "applicability": "not_applicable",
            "has_data": False,
            "note": (
                "Altman Z-Score is not applicable to banks, insurers, or REITs — "
                "their leverage ratios and asset structure break the formula."
            ),
        }

    # Compute the five components
    working_capital = None
    if current_assets is not None and current_liabs is not None:
        working_capital = current_assets - current_liabs

    a = _safe_div(working_capital, total_assets, default=0.0)
    b = _safe_div(retained_earnings, total_assets, default=0.0)
    c = _safe_div(ebit, total_assets, default=0.0)
    d = _safe_div(market_cap, total_liabilities, default=0.0)
    e = _safe_div(revenue, total_assets, default=0.0)

    z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e

    if z >= 2.99:
        classification = "Safe"
    elif z >= 1.81:
        classification = "Grey"
    else:
        classification = "Distress"

    components = [
        {"label": "A: Working Capital / Total Assets", "value": a, "weight": 1.2,
         "contribution": 1.2 * a},
        {"label": "B: Retained Earnings / Total Assets", "value": b, "weight": 1.4,
         "contribution": 1.4 * b},
        {"label": "C: EBIT / Total Assets", "value": c, "weight": 3.3,
         "contribution": 3.3 * c},
        {"label": "D: Market Cap / Total Liabilities", "value": d, "weight": 0.6,
         "contribution": 0.6 * d},
        {"label": "E: Sales / Total Assets", "value": e, "weight": 1.0,
         "contribution": 1.0 * e},
    ]

    # Note about manufacturer-specificity
    s = (sector or "").lower()
    i = (industry or "").lower()
    is_manufacturer = "industrial" in s or "manufactur" in i or "auto" in i
    note = None
    if not is_manufacturer:
        note = (
            "The classic Z-Score was calibrated on public manufacturers. "
            "Asset-light services and tech companies often score lower than "
            "their actual default risk warrants."
        )

    return {
        "score": round(z, 2),
        "classification": classification,
        "components": components,
        "applicability": "good" if is_manufacturer else "questionable",
        "has_data": True,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Beneish M-Score
# ---------------------------------------------------------------------------

def compute_beneish_m_score(income_statements: list[dict],
                              balance_sheets: list[dict],
                              cash_flow_statements: list[dict],
                              sector: str = "",
                              industry: str = "") -> dict:
    """
    Compute the Beneish M-Score (8-ratio earnings manipulation detector).

    M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Indices (each one a t-vs-prior ratio that flags warning signals):
      DSRI — Days Sales in Receivables Index (revenue recognition aggressiveness)
      GMI  — Gross Margin Index (deteriorating margins motive)
      AQI  — Asset Quality Index (capitalization aggressiveness)
      SGI  — Sales Growth Index (growth pressure motive)
      DEPI — Depreciation Index (extending asset lives)
      SGAI — SG&A Index (cost discipline)
      TATA — Total Accruals to Total Assets (accrual quality)
      LVGI — Leverage Index (leverage pressure)

    Threshold:
      M > -1.78  → Likely manipulator (high probability)
      M < -1.78  → Likely non-manipulator

    Like the other scores, financials and REITs are excluded — their accrual
    structures and asset turnover patterns break the formula's calibration.

    Returns:
        Dict with score, classification, components, applicability, has_data, note.
    """
    if (len(income_statements) < 2 or len(balance_sheets) < 2
            or len(cash_flow_statements) < 1):
        return {
            "score": None, "classification": None, "components": [],
            "applicability": None, "has_data": False,
            "note": "Insufficient history (need 2 years of statements)",
        }

    if _is_financial_or_reit(sector, industry):
        return {
            "score": None, "classification": None, "components": [],
            "applicability": "not_applicable", "has_data": False,
            "note": (
                "Beneish M-Score is not applicable to banks, insurers, or REITs — "
                "their accrual and asset turnover patterns break the formula."
            ),
        }

    inc_t, inc_p = income_statements[0], income_statements[1]
    bs_t,  bs_p  = balance_sheets[0],    balance_sheets[1]
    cf_t         = cash_flow_statements[0]

    # Pull all the raw inputs
    sales_t   = _safe_float(inc_t.get("revenue"))
    sales_p   = _safe_float(inc_p.get("revenue"))
    cogs_t    = _safe_float(inc_t.get("costOfRevenue"))
    cogs_p    = _safe_float(inc_p.get("costOfRevenue"))
    sga_t     = _safe_float(inc_t.get("sellingGeneralAndAdministrativeExpenses"))
    sga_p     = _safe_float(inc_p.get("sellingGeneralAndAdministrativeExpenses"))
    dep_t     = _safe_float(inc_t.get("depreciationAndAmortization"))
    dep_p     = _safe_float(inc_p.get("depreciationAndAmortization"))
    ni_t      = _safe_float(inc_t.get("netIncome"))

    ar_t      = _safe_float(bs_t.get("netReceivables") or bs_t.get("accountsReceivables"))
    ar_p      = _safe_float(bs_p.get("netReceivables") or bs_p.get("accountsReceivables"))
    ta_t      = _safe_float(bs_t.get("totalAssets"))
    ta_p      = _safe_float(bs_p.get("totalAssets"))
    ca_t      = _safe_float(bs_t.get("totalCurrentAssets"))
    ca_p      = _safe_float(bs_p.get("totalCurrentAssets"))
    ppe_t     = _safe_float(bs_t.get("propertyPlantEquipmentNet"))
    ppe_p     = _safe_float(bs_p.get("propertyPlantEquipmentNet"))
    cl_t      = _safe_float(bs_t.get("totalCurrentLiabilities"))
    cl_p      = _safe_float(bs_p.get("totalCurrentLiabilities"))
    ltd_t     = _safe_float(bs_t.get("longTermDebt"))
    ltd_p     = _safe_float(bs_p.get("longTermDebt"))

    cfo_t     = _safe_float(
        cf_t.get("operatingCashFlow") or cf_t.get("netCashProvidedByOperatingActivities")
    )

    # Validate the critical inputs we can't proceed without
    required = [sales_t, sales_p, ar_t, ar_p, ta_t, ta_p, ni_t, cfo_t]
    if any(v is None or v == 0 for v in [sales_t, sales_p, ta_t, ta_p]):
        return {
            "score": None, "classification": None, "components": [],
            "applicability": None, "has_data": False,
            "note": "Missing required fields (sales / total assets)",
        }

    # 1. DSRI — Days Sales in Receivables Index
    dsri = _safe_div(_safe_div(ar_t, sales_t), _safe_div(ar_p, sales_p), default=1.0)

    # 2. GMI — Gross Margin Index (note: prior over current — falling margins flag)
    gm_t = _safe_div((sales_t - (cogs_t or 0)), sales_t)
    gm_p = _safe_div((sales_p - (cogs_p or 0)), sales_p)
    gmi = _safe_div(gm_p, gm_t, default=1.0)

    # 3. AQI — Asset Quality Index
    # Quality = (CA + PP&E) / TA. Non-quality assets = 1 - quality.
    nq_t = 1 - _safe_div((ca_t or 0) + (ppe_t or 0), ta_t, default=0.0)
    nq_p = 1 - _safe_div((ca_p or 0) + (ppe_p or 0), ta_p, default=0.0)
    aqi = _safe_div(nq_t, nq_p, default=1.0)

    # 4. SGI — Sales Growth Index
    sgi = _safe_div(sales_t, sales_p, default=1.0)

    # 5. DEPI — Depreciation Index
    if dep_t and ppe_t and dep_p and ppe_p:
        rate_t = _safe_div(dep_t, dep_t + ppe_t)
        rate_p = _safe_div(dep_p, dep_p + ppe_p)
        depi = _safe_div(rate_p, rate_t, default=1.0)
    else:
        depi = 1.0

    # 6. SGAI — SG&A Index
    if sga_t and sga_p:
        sga_ratio_t = _safe_div(sga_t, sales_t)
        sga_ratio_p = _safe_div(sga_p, sales_p)
        sgai = _safe_div(sga_ratio_t, sga_ratio_p, default=1.0)
    else:
        sgai = 1.0

    # 7. TATA — Total Accruals to Total Assets
    tata = _safe_div((ni_t - (cfo_t or 0)), ta_t, default=0.0)

    # 8. LVGI — Leverage Index
    if ltd_t is not None and cl_t is not None and ltd_p is not None and cl_p is not None:
        lev_t = _safe_div((ltd_t + cl_t), ta_t)
        lev_p = _safe_div((ltd_p + cl_p), ta_p)
        lvgi = _safe_div(lev_t, lev_p, default=1.0)
    else:
        lvgi = 1.0

    m = (-4.84
         + 0.92  * dsri
         + 0.528 * gmi
         + 0.404 * aqi
         + 0.892 * sgi
         + 0.115 * depi
         - 0.172 * sgai
         + 4.679 * tata
         - 0.327 * lvgi)

    if m > -1.78:
        classification = "Likely Manipulator"
    else:
        classification = "Non-Manipulator"

    components = [
        {"label": "DSRI: Days Sales Receivables Index", "value": round(dsri, 2),
         "interpretation": "↑ = aggressive revenue recognition" if dsri > 1.1 else "normal"},
        {"label": "GMI: Gross Margin Index",            "value": round(gmi, 2),
         "interpretation": "↑ = deteriorating margins" if gmi > 1.1 else "stable"},
        {"label": "AQI: Asset Quality Index",           "value": round(aqi, 2),
         "interpretation": "↑ = capitalizing more costs" if aqi > 1.1 else "normal"},
        {"label": "SGI: Sales Growth Index",            "value": round(sgi, 2),
         "interpretation": "↑↑ = high growth pressure" if sgi > 1.3 else "moderate"},
        {"label": "DEPI: Depreciation Index",           "value": round(depi, 2),
         "interpretation": "↑ = slowing depreciation" if depi > 1.1 else "normal"},
        {"label": "SGAI: SG&A Index",                   "value": round(sgai, 2),
         "interpretation": "↑ = SG&A growing faster than sales" if sgai > 1.1 else "normal"},
        {"label": "TATA: Total Accruals / Total Assets","value": round(tata, 3),
         "interpretation": "↑ = high accruals (low cash backing)" if tata > 0.05 else "normal"},
        {"label": "LVGI: Leverage Index",               "value": round(lvgi, 2),
         "interpretation": "↑ = leverage rising" if lvgi > 1.1 else "stable"},
    ]

    return {
        "score": round(m, 2),
        "classification": classification,
        "threshold": -1.78,
        "components": components,
        "applicability": "good",
        "has_data": True,
        "note": (
            "Beneish was calibrated on US public companies in the 1980s-90s. "
            "False positives are common for legitimate high-growth firms — use "
            "as a flag to investigate, not a verdict."
        ),
    }
