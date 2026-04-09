"""
Main screening orchestrator.

Coordinates universe loading, quote fetching (Tradier), ratio analysis (FMP),
and result filtering/ranking.
"""

import logging

import pandas as pd

from src.universe_loader import load_universe, list_universes
from src.ratio_analyzer import analyze_ticker

logger = logging.getLogger(__name__)


def scan_universe(universe_name: str, fmp_client, tradier_client,
                  progress_callback=None, history_years: int | None = None) -> pd.DataFrame:
    """
    Scan a single universe: load tickers, fetch quotes, run ratio analysis.

    Args:
        universe_name: Name of the universe CSV (without extension).
        fmp_client: Initialized FMPClient instance.
        tradier_client: Initialized TradierClient instance.
        progress_callback: Optional callable(current, total, ticker) for progress updates.

    Returns:
        DataFrame of analysis results sorted by composite_score ascending.
    """
    universe_df = load_universe(universe_name)
    if universe_df.empty:
        logger.warning(f"Universe '{universe_name}' is empty or not found.")
        return pd.DataFrame()

    tickers = universe_df["ticker"].tolist()

    # Fetch quotes from Tradier
    quotes_df = pd.DataFrame()
    if tradier_client.is_configured:
        quotes_df = tradier_client.get_quotes(tickers)

    # Run ratio analysis for each ticker
    results = []
    total = len(tickers)

    for i, row in universe_df.iterrows():
        ticker = row["ticker"]
        if progress_callback:
            progress_callback(i + 1, total, ticker)

        universe_info = {
            "company_name": row.get("company_name", ""),
            "segment": row.get("segment", ""),
            "sub_segment": row.get("sub_segment", ""),
        }

        if fmp_client.is_configured:
            analysis = analyze_ticker(ticker, fmp_client, universe_info,
                                      history_years=history_years)
        else:
            # Degraded mode: no FMP data
            analysis = {
                "ticker": ticker,
                "company_name": row.get("company_name", ticker),
                "segment": row.get("segment", ""),
                "current_price": None,
                "market_cap": None,
                "metrics": {},
                "composite_score": None,
                "opportunity_flags": [],
            }

        # Merge quote data if available
        if not quotes_df.empty:
            quote_row = quotes_df[quotes_df["symbol"] == ticker]
            if not quote_row.empty:
                q = quote_row.iloc[0]
                if analysis["current_price"] is None:
                    analysis["current_price"] = q.get("last")
                analysis["volume"] = q.get("volume")
                analysis["change"] = q.get("change")
                analysis["change_pct"] = q.get("change_pct")
                analysis["week_52_high"] = q.get("week_52_high")
                analysis["week_52_low"] = q.get("week_52_low")

        results.append(_flatten_analysis(analysis))

    df = pd.DataFrame(results)
    if not df.empty and "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=True, na_position="last")
        df = df.reset_index(drop=True)

    return df


def scan_all_universes(fmp_client, tradier_client,
                        universe_names: list[str] | None = None,
                        progress_callback=None,
                        cancel_check=None,
                        history_years: int | None = None) -> pd.DataFrame:
    """
    Run screening across multiple (or all) universes and combine results.

    Args:
        fmp_client: Initialized FMPClient instance.
        tradier_client: Initialized TradierClient instance.
        universe_names: List of universe names to scan. If None, scans all.
        progress_callback: Optional callable(current, total, ticker) for progress.
        cancel_check: Optional callable() that returns True if scan should stop.

    Returns:
        Combined DataFrame sorted by composite_score ascending.
    """
    if universe_names is None:
        universe_names = list_universes()

    frames = []
    global_idx = 0

    # Count total tickers for progress
    total_tickers = 0
    for name in universe_names:
        uni = load_universe(name)
        total_tickers += len(uni)

    _cancelled = False
    for name in universe_names:
        if _cancelled:
            break

        uni_df = load_universe(name)
        tickers = uni_df["ticker"].tolist()

        # Fetch quotes for this universe
        quotes_df = pd.DataFrame()
        if tradier_client.is_configured:
            quotes_df = tradier_client.get_quotes(tickers)

        for _, row in uni_df.iterrows():
            if cancel_check and cancel_check():
                _cancelled = True
                break

            ticker = row["ticker"]
            global_idx += 1

            if progress_callback:
                progress_callback(global_idx, total_tickers, ticker)

            universe_info = {
                "company_name": row.get("company_name", ""),
                "segment": row.get("segment", ""),
                "sub_segment": row.get("sub_segment", ""),
            }

            if fmp_client.is_configured:
                analysis = analyze_ticker(ticker, fmp_client, universe_info,
                                          history_years=history_years)
            else:
                analysis = {
                    "ticker": ticker,
                    "company_name": row.get("company_name", ticker),
                    "segment": row.get("segment", ""),
                    "current_price": None,
                    "market_cap": None,
                    "metrics": {},
                    "composite_score": None,
                    "opportunity_flags": [],
                }

            # Merge quotes
            if not quotes_df.empty:
                quote_row = quotes_df[quotes_df["symbol"] == ticker]
                if not quote_row.empty:
                    q = quote_row.iloc[0]
                    if analysis["current_price"] is None:
                        analysis["current_price"] = q.get("last")
                    analysis["volume"] = q.get("volume")
                    analysis["change"] = q.get("change")
                    analysis["change_pct"] = q.get("change_pct")

            flat = _flatten_analysis(analysis)
            flat["universe"] = name
            frames.append(flat)

    df = pd.DataFrame(frames)
    if not df.empty:
        df = df.drop_duplicates(subset="ticker", keep="first")
        if "composite_score" in df.columns:
            df = df.sort_values("composite_score", ascending=True, na_position="last")
        df = df.reset_index(drop=True)

    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply threshold-based filters to screening results.

    Args:
        df: Screening results DataFrame.
        filters: Dict of filter specifications. Supported keys:
            - max_composite_score: float — exclude scores above this
            - min_composite_score: float
            - max_pe: float
            - min_pe: float
            - max_debt_to_equity: float
            - min_roe: float
            - segments: list[str] — include only these segments

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    result = df.copy()

    if "max_composite_score" in filters and "composite_score" in result.columns:
        result = result[
            result["composite_score"].isna()
            | (result["composite_score"] <= filters["max_composite_score"])
        ]

    if "min_composite_score" in filters and "composite_score" in result.columns:
        result = result[
            result["composite_score"].isna()
            | (result["composite_score"] >= filters["min_composite_score"])
        ]

    if "max_pe" in filters and "pe_ratio" in result.columns:
        result = result[
            result["pe_ratio"].isna()
            | (result["pe_ratio"] <= filters["max_pe"])
        ]

    if "min_pe" in filters and "pe_ratio" in result.columns:
        result = result[
            result["pe_ratio"].isna()
            | (result["pe_ratio"] >= filters["min_pe"])
        ]

    if "max_debt_to_equity" in filters and "debt_to_equity" in result.columns:
        result = result[
            result["debt_to_equity"].isna()
            | (result["debt_to_equity"] <= filters["max_debt_to_equity"])
        ]

    if "min_roe" in filters and "roe" in result.columns:
        result = result[
            result["roe"].isna()
            | (result["roe"] >= filters["min_roe"])
        ]

    if "segments" in filters and "segment" in result.columns:
        result = result[result["segment"].isin(filters["segments"])]

    # Quality score filters
    if "min_piotroski" in filters and "piotroski_f_score" in result.columns:
        result = result[
            result["piotroski_f_score"].isna()
            | (result["piotroski_f_score"] >= filters["min_piotroski"])
        ]

    if "min_altman" in filters and "altman_z_score" in result.columns:
        result = result[
            result["altman_z_score"].isna()
            | (result["altman_z_score"] >= filters["min_altman"])
        ]

    if "max_beneish" in filters and "beneish_m_score" in result.columns:
        # Lower M-Score = less manipulation risk, so this is a max filter
        result = result[
            result["beneish_m_score"].isna()
            | (result["beneish_m_score"] <= filters["max_beneish"])
        ]

    return result.reset_index(drop=True)


def _flatten_analysis(analysis: dict) -> dict:
    """
    Flatten nested analysis dict into a single-level dict for DataFrame creation.

    Extracts current values and percentiles from the metrics sub-dict.
    """
    flat = {
        "ticker": analysis.get("ticker"),
        "company_name": analysis.get("company_name"),
        "segment": analysis.get("segment"),
        "current_price": analysis.get("current_price"),
        "market_cap": analysis.get("market_cap"),
        "composite_score": analysis.get("composite_score"),
        "flags": "; ".join(analysis.get("opportunity_flags", [])),
        "eps_trend": (analysis.get("earnings_context") or {}).get("eps_trend"),
        "eps_yoy_change": (analysis.get("earnings_context") or {}).get("eps_yoy_change"),
        "eps_3yr_cagr": (analysis.get("earnings_context") or {}).get("eps_3yr_cagr"),
        "fcf_trend": (analysis.get("earnings_context") or {}).get("fcf_trend"),
        "fcf_yoy_change": (analysis.get("earnings_context") or {}).get("fcf_yoy_change"),
        "fcf_vs_net_income": (analysis.get("earnings_context") or {}).get("fcf_vs_net_income"),
        "net_debt_to_ebitda": (analysis.get("earnings_context") or {}).get("net_debt_to_ebitda"),
        "fundamentals_flags": "; ".join((analysis.get("earnings_context") or {}).get("context_flags", [])),
        "volume": analysis.get("volume"),
        "change": analysis.get("change"),
        "change_pct": analysis.get("change_pct"),
        "week_52_high": analysis.get("week_52_high"),
        "week_52_low": analysis.get("week_52_low"),
    }

    # Flatten individual metrics
    metrics = analysis.get("metrics", {})
    for metric_name, data in metrics.items():
        flat[metric_name] = data.get("current")
        flat[f"{metric_name}_pct"] = data.get("percentile")
        flat[f"{metric_name}_hist_avg"] = data.get("hist_avg")
        flat[f"{metric_name}_hist_low"] = data.get("hist_low")
        flat[f"{metric_name}_hist_high"] = data.get("hist_high")

    # Flatten quality scores (Piotroski / Altman / Beneish)
    _piot = analysis.get("piotroski_f_score") or {}
    _alt  = analysis.get("altman_z_score") or {}
    _ben  = analysis.get("beneish_m_score") or {}
    flat["piotroski_f_score"] = _piot.get("score") if _piot.get("has_data") else None
    flat["altman_z_score"]    = _alt.get("score")  if _alt.get("has_data") else None
    flat["beneish_m_score"]   = _ben.get("score")  if _ben.get("has_data") else None

    return flat
