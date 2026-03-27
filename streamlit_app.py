"""
Prospector — Streamlit App

A fundamental analysis and stock screening tool. Uses Tradier for
real-time quotes and Financial Modeling Prep for financial ratios, metrics,
and statements. Scores stocks by percentile rank against their own history.

Run: streamlit run streamlit_app.py
Port configured to 8502 via .streamlit/config.toml
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Load API keys from .env file
except ImportError:
    pass  # python-dotenv not required on Streamlit Cloud

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.tradier_client import TradierClient
from src.fmp_client import FMPClient
from src.eia_client import EIAClient, PETROLEUM_SERIES, SPOT_PRICE_SERIES
from src.fred_client import FREDClient, FRED_SERIES
from src.commodity_client import (
    COMMODITY_SYMBOLS, get_commodity_quote, get_commodity_quote_by_symbol,
    get_commodity_history, clear_cache as clear_commodity_cache
)
from src.ratio_analyzer import analyze_ticker, compute_dcf_valuation, compute_revenue_dcf_valuation, _load_scoring_config
from src.screener import scan_universe, scan_all_universes, apply_filters
from src.edgar_client import EDGARClient
from src.price_validator import cross_validate_price
from src.universe_loader import list_universes, load_universe
import src.portfolio as _portfolio_mod
from src.portfolio import (
    load_portfolio, save_portfolio, add_position, remove_position,
    update_position_notes,
    get_all_tags, compute_position_performance, compute_portfolio_summary,
    compute_tag_performance, get_position_history, get_spy_benchmark,
)
from streamlit_js_eval import streamlit_js_eval

# ------------------------------------------------------------------ #
# Setup
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
SCORING_PATH = PROJECT_ROOT / "config" / "scoring" / "defaults.json"

# Ensure data directories exist
for d in ["data/cache/fmp", "data/cache/quotes", "data/cache/eia", "data/cache/fred", "data/cache/edgar", "data/cache/transcripts", "data/outputs"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    """Load application settings from config/settings.json."""
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}


def load_scoring_config() -> dict:
    """Load scoring weights and thresholds."""
    try:
        return json.loads(SCORING_PATH.read_text())
    except Exception:
        return _load_scoring_config()


settings = load_settings()

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title=settings.get("app", {}).get("title", "Prospector"),
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Global CSS — larger expander/section titles ---
st.markdown("""
<style>
/* Make expander titles larger and bolder */
details summary span p {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
/* Make st.subheader slightly larger */
[data-testid="stSubheader"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# Load Streamlit Cloud secrets into environment variables if available
try:
    _secrets_path = Path.home() / ".streamlit" / "secrets.toml"
    _local_secrets = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if _secrets_path.exists() or _local_secrets.exists() or os.environ.get("STREAMLIT_SHARING_MODE"):
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
except (AttributeError, FileNotFoundError, Exception):
    pass


@st.cache_resource
def get_tradier_client() -> TradierClient:
    """Create a singleton TradierClient."""
    rate_delay = settings.get("rate_limits", {}).get("tradier_delay", 0.15)
    cache_ttl = settings.get("cache_ttl", {}).get("quotes", 300)
    return TradierClient(rate_delay=rate_delay, cache_ttl=cache_ttl)


@st.cache_resource
def get_fmp_client() -> FMPClient:
    """Create a singleton FMPClient."""
    rate_delay = settings.get("rate_limits", {}).get("fmp_delay", 0.3)
    cache_ttls = {
        "financial_statements": settings.get("cache_ttl", {}).get("financial_statements", 86400),
        "ratios": settings.get("cache_ttl", {}).get("ratios", 86400),
        "key_metrics": settings.get("cache_ttl", {}).get("ratios", 86400),
        "profiles": settings.get("cache_ttl", {}).get("profiles", 604800),
        "price_history": settings.get("cache_ttl", {}).get("price_history", 14400),
    }
    return FMPClient(rate_delay=rate_delay, cache_ttls=cache_ttls)


@st.cache_resource
def get_eia_client() -> EIAClient:
    """Create a singleton EIAClient."""
    cache_ttl = settings.get("cache_ttl", {}).get("eia", 3600)
    return EIAClient(cache_ttl=cache_ttl)


@st.cache_resource
def get_fred_client() -> FREDClient:
    """Create a singleton FREDClient."""
    cache_ttl = settings.get("cache_ttl", {}).get("fred", 3600)
    return FREDClient(cache_ttl=cache_ttl)


@st.cache_resource
def get_edgar_client() -> EDGARClient:
    """Create a singleton EDGARClient (no API key needed)."""
    return EDGARClient()


tradier = get_tradier_client()
fmp = get_fmp_client()
eia = get_eia_client()
fred = get_fred_client()
edgar = get_edgar_client()

# ---------------------------------------------------------------------------
# Portfolio — browser localStorage backend
# ---------------------------------------------------------------------------
_LS_KEY = "prospector_portfolio"


def _init_portfolio_from_localstorage():
    """Load portfolio from browser localStorage into session_state on first run."""
    if "portfolio_loaded" in st.session_state:
        return  # already loaded this session

    raw = streamlit_js_eval(
        js_expressions=f"localStorage.getItem('{_LS_KEY}')",
        key="portfolio_load",
    )

    if raw is None:
        # JS component hasn't mounted yet — retry a few times
        attempts = st.session_state.get("_portfolio_load_attempts", 0) + 1
        st.session_state["_portfolio_load_attempts"] = attempts
        if attempts < 5:
            import time
            time.sleep(0.3)
            st.rerun()
        else:
            # Give up waiting, start with empty portfolio
            st.session_state["portfolio_data"] = {"positions": []}
            st.session_state["portfolio_loaded"] = True
        return

    if raw in (0, "null", ""):
        st.session_state["portfolio_data"] = {"positions": []}
    else:
        try:
            st.session_state["portfolio_data"] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            st.session_state["portfolio_data"] = {"positions": []}

    st.session_state["portfolio_loaded"] = True
    st.session_state["portfolio_dirty"] = False


def _push_portfolio_to_localstorage():
    """If portfolio was modified this run, write it to browser localStorage."""
    if st.session_state.get("portfolio_dirty"):
        data_json = json.dumps(st.session_state["portfolio_data"])
        escaped = data_json.replace("\\", "\\\\").replace("'", "\\'")
        # Use a unique key each time so Streamlit re-executes the JS
        save_counter = st.session_state.get("_portfolio_save_counter", 0) + 1
        st.session_state["_portfolio_save_counter"] = save_counter
        streamlit_js_eval(
            js_expressions=f"localStorage.setItem('{_LS_KEY}', '{escaped}')",
            key=f"portfolio_save_{save_counter}",
        )
        st.session_state["portfolio_dirty"] = False


def _session_load() -> dict:
    return st.session_state.get("portfolio_data", {"positions": []})


def _session_save(data: dict):
    st.session_state["portfolio_data"] = data
    st.session_state["portfolio_dirty"] = True


# Detect deployment: Streamlit Cloud sets STREAMLIT_SHARING_MODE or runs from /mount/src
_IS_CLOUD = bool(os.environ.get("STREAMLIT_SHARING_MODE") or
                 os.environ.get("HOSTNAME", "").startswith("streamlit") or
                 str(Path(__file__)).startswith("/mount/"))

if _IS_CLOUD:
    # Deployed — each visitor gets their own portfolio via browser localStorage
    _portfolio_mod.set_storage_backend(_session_load, _session_save)
    _init_portfolio_from_localstorage()
# else: local — portfolio.py falls back to data/portfolio.json (file-based)


def format_universe_name(name: str) -> str:
    """Format universe file name for display (e.g. 'inverse_airlines' -> '⚠️ INVERSE — Airlines')."""
    if name.startswith("inverse_"):
        clean = name.replace("inverse_", "").replace("_", " ").title()
        return f"⚠️ INVERSE — {clean}"
    return name.replace("_", " ").title()


def universe_display_map() -> dict[str, str]:
    """Return {display_name: file_name} mapping for all universes."""
    names = list_universes()
    return {format_universe_name(n): n for n in names}


# ------------------------------------------------------------------ #
# Sidebar — API status
# ------------------------------------------------------------------ #
st.title("⛏️ Prospector")

# API status as a collapsible expander in the header area
from src.transcript_summarizer import is_configured as _anthropic_ok

# Build a quick status summary for the expander label
_api_count = sum([tradier.is_configured, fmp.is_configured, eia.is_configured,
                  fred.is_configured, _anthropic_ok(), True])  # SEC EDGAR always connected
_api_label = f"API Status — {_api_count}/6 connected"

with st.expander(_api_label, expanded=False):
    _status_cols = st.columns(6)
    with _status_cols[0]:
        if tradier.is_configured:
            st.success("Tradier ✅")
        else:
            st.warning("Tradier ⚠️")
    with _status_cols[1]:
        if fmp.is_configured:
            st.success("FMP ✅")
        else:
            st.warning("FMP ⚠️")
    with _status_cols[2]:
        if eia.is_configured:
            st.success("EIA ✅")
        else:
            st.warning("EIA ⚠️")
    with _status_cols[3]:
        if fred.is_configured:
            st.success("FRED ✅")
        else:
            st.warning("FRED ⚠️")
    with _status_cols[4]:
        if _anthropic_ok():
            st.success("Claude AI ✅")
        else:
            st.warning("Claude AI ⚠️")
    with _status_cols[5]:
        st.success("SEC EDGAR ✅")

    if not fmp.is_configured:
        st.info(
            "Running in degraded mode. Only Tradier quotes available. "
            "Set FMP_API_KEY for full fundamental analysis."
        )

# ------------------------------------------------------------------ #
# Tabs
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Screener Dashboard",
    "Ticker Deep Dive",
    "Ratio Comparison",
    "Balance Sheet Health",
    "EIA Inventories",
    "Commodity Prices",
    "Portfolio Tracker",
    "Settings",
])


# ================================================================== #
# TAB 1: Screener Dashboard
# ================================================================== #
with tab1:
    st.header("Screener Dashboard")

    st.markdown(
        """**Dig deeper. Find value others miss.**

Prospector is a fundamental analysis tool built for investors who want to do their own work.
Scan industries, compare peers, read earnings transcripts, and stress-test your assumptions
with a multi-stage DCF model — all in one place.

*Start by selecting an industry below, or jump straight to the Ticker Deep Dive tab to
analyze a specific company.*"""
    )

    st.divider()

    # --- Screener filters (inline) ---
    uni_map = universe_display_map()
    selected_display = st.multiselect(
        "Select Industries",
        options=sorted(uni_map.keys(), key=lambda x: ("⚠️" in x, x)),
        default=[],
        help="Choose which industries to scan. ⚠️ = inverse plays (sectors likely hurt by conflict).",
    )
    selected_universes = [uni_map[d] for d in selected_display]

    filter_cols = st.columns(3)
    with filter_cols[0]:
        history_years = st.selectbox(
            "History Window",
            options=[5, 10, 20],
            index=1,
            format_func=lambda y: f"{y} years",
            help="How many years of historical data to use for percentile rankings.",
            key="history_years",
        )
    with filter_cols[1]:
        max_pe = st.number_input(
            "Max P/E Ratio",
            min_value=0.0,
            max_value=500.0,
            value=500.0,
            step=5.0,
            help="Exclude tickers with P/E above this value.",
        )

    # Scan button — centered
    _left, _center, _right = st.columns([2, 1, 2])
    col_export = _right  # export button goes on the right after results load
    with _center:
        scan_clicked = st.button("Scan Selected Industries", type="primary", use_container_width=True)

    if scan_clicked:
        if not selected_universes:
            st.warning("Please select at least one industry to scan.")
        else:
            progress_bar = st.progress(0, text="Starting scan...")
            status_text = st.empty()

            def update_progress(current, total, ticker):
                pct = current / total
                progress_bar.progress(pct, text=f"Analyzing {ticker} ({current}/{total})")

            with st.spinner("Scanning industries..."):
                results = scan_all_universes(
                    fmp_client=fmp,
                    tradier_client=tradier,
                    universe_names=selected_universes,
                    progress_callback=update_progress,
                    history_years=history_years,
                )

            progress_bar.empty()
            st.session_state["scan_results"] = results
            st.success(f"Scan complete. {len(results)} tickers analyzed.")

    # Display results
    if "scan_results" in st.session_state and not st.session_state["scan_results"].empty:
        results_df = st.session_state["scan_results"].copy()

        # Apply filters
        filters = {}
        if max_pe < 500:
            filters["max_pe"] = max_pe

        if filters:
            results_df = apply_filters(results_df, filters)

        # Select display columns
        display_cols = [
            "ticker", "company_name", "segment", "current_price",
        ]

        # Add ratio columns if they exist
        ratio_display = {
            "pe_ratio": "P/E",
            "pe_ratio_pct": "P/E %ile",
            "ps_ratio": "P/S",
            "ps_ratio_pct": "P/S %ile",
            "ev_ebitda": "EV/EBITDA",
            "ev_ebitda_pct": "EV/EBITDA %ile",
            "debt_to_equity": "D/E",
            "roe": "ROE",
            "fcf_yield": "FCF Yield",
        }
        for col in ratio_display:
            if col in results_df.columns:
                display_cols.append(col)

        # Only keep columns that exist
        display_cols = [c for c in display_cols if c in results_df.columns]
        display_df = results_df[display_cols].copy()

        # Rename for display
        rename_map = {
            "ticker": "Ticker",
            "company_name": "Company",
            "segment": "Segment",
            "current_price": "Price",
            "flags": "Flags",
        }
        rename_map.update({k: v for k, v in ratio_display.items() if k in display_df.columns})
        display_df = display_df.rename(columns=rename_map)

        st.subheader(f"Results ({len(display_df)} tickers)")
        st.caption("💡 Click a row to jump to Ticker Deep Dive")

        # Color-code percentile columns (direction-aware)
        # Map display %ile column names back to raw metric keys
        _pct_raw_keys = {v: k.replace("_pct", "") for k, v in ratio_display.items() if k.endswith("_pct")}

        # Metrics where high percentile = good
        _hib_screener = {
            "gross_margin", "operating_margin", "net_margin",
            "roe", "roa", "roic",
            "current_ratio", "quick_ratio", "interest_coverage",
            "fcf_yield", "earnings_yield", "dividend_yield",
        }

        def color_percentile(val, col_name):
            """Color percentiles: green = good, red = bad, direction-aware."""
            if pd.isna(val):
                return ""
            raw_key = _pct_raw_keys.get(col_name, "")
            high_is_good = raw_key in _hib_screener
            if high_is_good:
                if val > 80:
                    return "background-color: rgba(0, 180, 0, 0.3)"
                elif val < 20:
                    return "background-color: rgba(220, 50, 50, 0.3)"
            else:
                if val < 20:
                    return "background-color: rgba(0, 180, 0, 0.3)"
                elif val > 80:
                    return "background-color: rgba(220, 50, 50, 0.3)"
            return "background-color: rgba(200, 200, 0, 0.2)"

        percentile_cols = [c for c in display_df.columns if "%ile" in c]

        styled = display_df.style
        if percentile_cols:
            for pc in percentile_cols:
                styled = styled.map(lambda val, _c=pc: color_percentile(val, _c), subset=[pc])

        event = st.dataframe(
            styled,
            use_container_width=True,
            height=600,
            on_select="rerun",
            selection_mode="single-row",
            key="screener_table",
        )

        # Handle row click → jump to Ticker Deep Dive
        if event and event.selection and event.selection.rows:
            clicked_row = event.selection.rows[0]
            clicked_ticker = display_df.iloc[clicked_row]["Ticker"]
            st.session_state["jump_to_ticker"] = clicked_ticker
            # Use JS to click the "Ticker Deep Dive" tab (2nd tab, index 1)
            streamlit_js_eval(
                js_expressions="""
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs.length > 1) { tabs[1].click(); }
                """,
                key="click_deep_dive_tab",
            )

        # Export button
        with col_export:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "Export Full Results to CSV",
                data=csv_data,
                file_name="screener_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.markdown(
            "<p style='text-align:center; color:gray;'>Select industries above and click <b>Scan</b> to begin.</p>",
            unsafe_allow_html=True,
        )


# ================================================================== #
# TAB 2: Ticker Deep Dive
# ================================================================== #
with tab2:
    st.header("Ticker Deep Dive")

    # --- Handle jump from Screener Dashboard click ---
    _jump_ticker = st.session_state.pop("jump_to_ticker", None)

    # Universe selector — pick which universes to draw tickers from
    dd_uni_map = universe_display_map()

    # If jumping from screener, force-set the universe and ticker widgets
    if _jump_ticker:
        for display_name, file_name in dd_uni_map.items():
            try:
                udf = load_universe(file_name)
                if _jump_ticker in udf["ticker"].values:
                    # Force the multiselect widget value via session_state
                    st.session_state["dd_universes"] = [display_name]
                    break
            except Exception:
                continue

    # Build master ticker list from ALL universes for the searchable dropdown
    @st.cache_data(ttl=3600)
    def _load_all_tickers():
        """Load every ticker from every universe CSV into one list."""
        all_frames = []
        for _fn in dd_uni_map.values():
            try:
                all_frames.append(load_universe(_fn))
            except Exception:
                continue
        if not all_frames:
            return pd.DataFrame(columns=["ticker", "company_name", "segment"])
        df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset="ticker", keep="first")
        return df.sort_values("ticker").reset_index(drop=True)

    _master_df = _load_all_tickers()
    _master_options = [""] + _master_df["ticker"].tolist()
    _master_labels = {"": "🔍 Type to search any ticker..."}
    _master_labels.update({
        row["ticker"]: f"{row['ticker']} — {row.get('company_name', '')}"
        for _, row in _master_df.iterrows() if row.get("company_name")
    })

    # If jumping from screener, pre-set the search dropdown
    if _jump_ticker and _jump_ticker in _master_options:
        st.session_state["dd_search_ticker"] = _jump_ticker

    _search_ticker = st.selectbox(
        "🔍 Search any ticker",
        options=_master_options,
        format_func=lambda t: _master_labels.get(t, t),
        help="Start typing to search across all industries. Select a ticker to auto-analyze.",
        key="dd_search_ticker",
    )

    st.caption("— or browse by industry —")

    dd_display = st.multiselect(
        "Select Industries",
        options=sorted(dd_uni_map.keys(), key=lambda x: ("⚠️" in x, x)),
        default=[],
        help="Choose one or more industries to browse tickers from. ⚠️ = inverse plays.",
        key="dd_universes",
    )

    # If industries are selected, show a ticker picker from those industries
    _industry_ticker = None
    if dd_display:
        dd_universes = [dd_uni_map[d] for d in dd_display]
        frames = [load_universe(u) for u in dd_universes]
        industry_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ticker", keep="first")
        industry_options = sorted(industry_df["ticker"].unique().tolist())
        industry_labels = {
            row["ticker"]: f"{row['ticker']} — {row['company_name']}"
            for _, row in industry_df.iterrows() if row.get("company_name")
        }

        _industry_ticker = st.selectbox(
            "Select Ticker",
            options=industry_options,
            format_func=lambda t: industry_labels.get(t, t),
            help="Choose a ticker from the selected industries.",
            key="dd_industry_ticker",
        )

    # Determine selected ticker — search box takes priority, then industry picker
    selected_ticker = _search_ticker if _search_ticker else _industry_ticker

    # Build all_tickers_df for universe info lookup
    all_tickers_df = _master_df

    # Clear stale analysis if the user changed the ticker
    if "deep_dive" in st.session_state:
        _prev_ticker = st.session_state["deep_dive"].get("ticker")
        if _prev_ticker and selected_ticker and _prev_ticker != selected_ticker:
            del st.session_state["deep_dive"]

    # Auto-analyze: when a ticker is selected and no analysis exists yet, or jumped from screener
    _trigger_analyze = bool(_jump_ticker and selected_ticker)
    if (selected_ticker and st.button("Analyze", key="deep_dive_btn")) or _trigger_analyze:
        if not fmp.is_configured:
            st.error("FMP API key is required for deep dive analysis. Set FMP_API_KEY.")
        else:
            with st.spinner(f"Analyzing {selected_ticker}..."):
                # Get universe info (may be empty if using direct ticker search)
                universe_info = {}
                if not all_tickers_df.empty and "ticker" in all_tickers_df.columns:
                    info_row = all_tickers_df[all_tickers_df["ticker"] == selected_ticker]
                else:
                    info_row = pd.DataFrame()
                if not info_row.empty:
                    r = info_row.iloc[0]
                    universe_info = {
                        "company_name": r.get("company_name", ""),
                        "segment": r.get("segment", ""),
                        "sub_segment": r.get("sub_segment", ""),
                    }

                analysis = analyze_ticker(selected_ticker, fmp, universe_info,
                                          history_years=history_years)

            st.session_state["deep_dive"] = analysis

    if "deep_dive" in st.session_state:
        analysis = st.session_state["deep_dive"]

        # Price cross-validation FIRST (before rendering header)
        _pv = cross_validate_price(analysis["ticker"], tradier, fmp)
        _price_corrected = False
        if _pv["has_discrepancy"] and _pv.get("validated_price"):
            analysis["current_price"] = _pv["validated_price"]
            _price_corrected = True

        # Company header (now uses corrected price if applicable)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Company", analysis["company_name"])
        with col2:
            price = analysis.get("current_price")
            st.metric("Price", f"${price:,.2f}" if price else "N/A")
        with col3:
            # Analyst price target consensus
            _pt = fmp.get_price_target_consensus(analysis["ticker"])
            if _pt:
                _pt_consensus = _pt.get("targetConsensus")
                _pt_current = analysis.get("current_price", 0) or 0
                if _pt_consensus and _pt_current > 0:
                    _pt_upside = ((_pt_consensus - _pt_current) / _pt_current) * 100
                    _pt_label = "upside" if _pt_upside >= 0 else "downside"
                    st.metric(
                        "Analyst Target",
                        f"${_pt_consensus:,.2f}",
                        delta=f"{_pt_upside:+.1f}% {_pt_label}",
                        delta_color="normal",
                        help=f"Low: ${_pt.get('targetLow', 'N/A')}  ·  Median: ${_pt.get('targetMedian', 'N/A')}  ·  High: ${_pt.get('targetHigh', 'N/A')}",
                    )
                else:
                    st.metric("Analyst Target", "N/A")
            else:
                st.metric("Analyst Target", "N/A")

        # Show price validation status
        if _price_corrected:
            price = _pv["validated_price"]
            _pv_msg_parts = []
            for _d in _pv["discrepancies"]:
                _pv_msg_parts.append(
                    f"{_d['source1']} (${_d['price1']:,.2f}) vs "
                    f"{_d['source2']} (${_d['price2']:,.2f}) — "
                    f"{_d['diff_pct']:.1f}% apart"
                )
            _agree_str = ", ".join(_pv["agreeing_sources"]) if _pv["agreeing_sources"] else "None"
            _disagree_str = ", ".join(_pv["disagreeing_sources"]) if _pv["disagreeing_sources"] else "None"
            st.warning(
                f"**⚠️ Price Corrected — Discrepancy Detected**  \n"
                f"{'  |  '.join(_pv_msg_parts)}  \n"
                f"**Agreeing:** {_agree_str}  ·  **Disagreeing:** {_disagree_str}  \n"
                f"All calculations below use the validated price: **${_pv['validated_price']:,.2f}**"
            )
        else:
            _pv_sources = {k: v for k, v in _pv["sources"].items() if v is not None}
            if len(_pv_sources) >= 2:
                _src_strs = [f"{k}: ${v:,.2f}" for k, v in _pv_sources.items()]
                st.caption(f"✓ Price validated across {len(_pv_sources)} sources ({' · '.join(_src_strs)})")

        # Show which industries this ticker belongs to
        _ticker_industries = []
        _dd_uni_map = universe_display_map()
        for _disp_name, _file_name in _dd_uni_map.items():
            try:
                _udf = load_universe(_file_name)
                if analysis["ticker"] in _udf["ticker"].values:
                    _ticker_industries.append(_disp_name)
            except Exception:
                continue
        if _ticker_industries:
            _ind_tags = " · ".join(f"**{ind}**" for ind in _ticker_industries)
            st.caption(f"🏷️ Industries: {_ind_tags}")

        # Company profile
        profile = fmp.get_company_profile(analysis["ticker"])
        if profile:
            with st.expander("Company Profile", expanded=False):
                prof_cols = st.columns(4)
                with prof_cols[0]:
                    st.write(f"**Sector:** {profile.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {profile.get('industry', 'N/A')}")
                with prof_cols[1]:
                    mktcap = profile.get("mktCap", 0)
                    st.write(f"**Market Cap:** ${mktcap / 1e9:,.1f}B" if mktcap else "**Market Cap:** N/A")
                    emp = profile.get('fullTimeEmployees')
                    st.write(f"**Employees:** {int(emp):,}" if emp and str(emp).isdigit() else "**Employees:** N/A")
                with prof_cols[2]:
                    st.write(f"**Exchange:** {profile.get('exchangeShortName', 'N/A')}")
                    st.write(f"**Country:** {profile.get('country', 'N/A')}")
                with prof_cols[3]:
                    st.write(f"**CEO:** {profile.get('ceo', 'N/A')}")
                    st.write(f"**IPO Date:** {profile.get('ipoDate', 'N/A')}")
                desc = profile.get("description", "")
                if desc:
                    st.write(f"**Description:** {desc}")

        # Metrics table
        st.subheader("Fundamental Metrics")
        _filing_period = analysis.get("latest_filing_period")
        _filing_date = analysis.get("latest_filing_date")
        if _filing_period or _filing_date:
            _parts = []
            if _filing_period:
                _parts.append(f"**{_filing_period}**")
            if _filing_date:
                _parts.append(f"filed {_filing_date}")
            st.caption(f"📅 Based on most recent annual filing: {' — '.join(_parts)}")
        metrics = analysis.get("metrics", {})

        if metrics:
            pretty_names = {
                "pe_ratio": "P/E Ratio", "ps_ratio": "P/S Ratio",
                "pb_ratio": "P/B Ratio", "ev_ebitda": "EV/EBITDA",
                "ev_revenue": "EV/Revenue", "fcf_yield": "FCF Yield",
                "roe": "ROE", "roa": "ROA",
                "gross_margin": "Gross Margin", "operating_margin": "Operating Margin",
                "net_margin": "Net Margin", "debt_to_equity": "Debt/Equity",
                "current_ratio": "Current Ratio", "quick_ratio": "Quick Ratio",
                "interest_coverage": "Interest Coverage",
                "earnings_yield": "Earnings Yield",
                "dividend_yield": "Dividend Yield",
                "dividend_payout_ratio": "Dividend Payout Ratio",
                "price_to_fcf": "P/FCF",
                "peg_ratio": "PEG Ratio",
                "roic": "ROIC",
                "cash_conversion_cycle": "Cash Conversion Cycle",
                "debt_to_assets": "Debt/Assets",
                "revenue_growth_yoy": "Revenue Growth YoY",
                "earnings_growth_yoy": "Earnings Growth YoY",
                "fcf_growth_yoy": "FCF Growth YoY",
            }

            # Metrics where a HIGH percentile is good (green) not bad (red)
            higher_is_better = {
                "gross_margin", "operating_margin", "net_margin",
                "roe", "roa", "roic",
                "current_ratio", "quick_ratio", "interest_coverage",
                "fcf_yield", "earnings_yield",
                "dividend_yield",
                "revenue_growth_yoy", "earnings_growth_yoy", "fcf_growth_yoy",
            }

            # Metrics that should display as percentages (value × 100 + %)
            _pct_display_metrics = {
                "gross_margin", "operating_margin", "net_margin",
                "roe", "roa", "roic",
                "fcf_yield", "earnings_yield",
                "dividend_yield", "dividend_payout_ratio",
                "debt_to_assets",
                "revenue_growth_yoy", "earnings_growth_yoy", "fcf_growth_yoy",
            }

            # YoY growth metrics where extreme values are base-effect distortions
            _yoy_metrics = {"revenue_growth_yoy", "earnings_growth_yoy", "fcf_growth_yoy"}

            def _fmt_metric_val(val, metric_name):
                """Format a metric value, converting decimals to % where appropriate."""
                if val is None:
                    return None
                if metric_name in _pct_display_metrics:
                    pct = val * 100
                    # Cap extreme YoY growth values (base-effect distortions)
                    if metric_name in _yoy_metrics:
                        if pct > 200:
                            return ">200%"
                        if pct < -100:
                            return "<-100%"
                    return f"{pct:.2f}%"
                return val

            # Metrics to hide entirely when data is clearly garbage
            _garbage_thresholds = {
                "dividend_yield": 0.20,       # > 20% dividend yield is bad data
                "dividend_payout_ratio": 5.0,  # > 500% payout ratio is bad data
            }

            rows = []
            percentiles = []
            labels = []
            metric_keys = []
            for metric_name, data in metrics.items():
                # Skip metrics with garbage values
                current_val = data.get("current")
                if metric_name in _garbage_thresholds and current_val is not None:
                    if abs(current_val) > _garbage_thresholds[metric_name]:
                        continue  # hide entirely

                display_name = pretty_names.get(metric_name, metric_name)
                rows.append({
                    "Metric": display_name,
                    "Current": _fmt_metric_val(data.get("current"), metric_name),
                    f"{history_years}yr Median": _fmt_metric_val(data.get("hist_median"), metric_name),
                    f"{history_years}yr Low": _fmt_metric_val(data.get("hist_low"), metric_name),
                    f"{history_years}yr High": _fmt_metric_val(data.get("hist_high"), metric_name),
                    "Percentile": data.get("percentile"),
                    "Years of Data": data.get("years_of_data"),
                })
                if data.get("percentile") is not None:
                    percentiles.append(data["percentile"])
                    labels.append(display_name)
                    metric_keys.append(metric_name)

            metrics_df = pd.DataFrame(rows)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Percentile bar charts — split by metric direction
            if percentiles:
                st.subheader("Percentile Ranks")

                # Separate into two groups
                val_labels, val_pcts = [], []  # lower is better (valuation/debt)
                qual_labels, qual_pcts = [], []  # higher is better (quality/returns)
                for lbl, pct, key in zip(labels, percentiles, metric_keys):
                    if key in higher_is_better:
                        qual_labels.append(lbl)
                        qual_pcts.append(pct)
                    else:
                        val_labels.append(lbl)
                        val_pcts.append(pct)

                def _gradient_color(p, invert=False):
                    """Smooth green-yellow-red gradient. If invert, high = green."""
                    if invert:
                        p = 100 - p
                    # p=0 -> green, p=50 -> yellow, p=100 -> red
                    if p <= 50:
                        r = int(220 * (p / 50))
                        g = 180
                    else:
                        r = 220
                        g = int(180 * (1 - (p - 50) / 50))
                    return f"rgba({r}, {g}, 0, 0.75)"

                # --- Valuation & Debt (lower is better) ---
                if val_labels:
                    st.caption("**Valuation & Debt** — lower percentile = cheaper / less leveraged")
                    fig_val = go.Figure()
                    fig_val.add_trace(go.Bar(
                        x=val_labels, y=val_pcts,
                        marker_color=[_gradient_color(p) for p in val_pcts],
                        text=[f"{p:.0f}" for p in val_pcts],
                        textposition="outside",
                    ))
                    fig_val.update_layout(
                        yaxis_title="Percentile (0-100)",
                        yaxis_range=[0, 110],
                        height=350,
                        template="plotly_dark",
                    )
                    fig_val.add_hline(y=20, line_dash="dash", line_color="green",
                                      annotation_text="Cheap (20th)")
                    fig_val.add_hline(y=80, line_dash="dash", line_color="red",
                                      annotation_text="Expensive (80th)")
                    st.plotly_chart(fig_val, use_container_width=True)

                # --- Quality & Returns (higher is better) ---
                if qual_labels:
                    st.caption("**Quality & Returns** — higher percentile = stronger fundamentals")
                    fig_qual = go.Figure()
                    fig_qual.add_trace(go.Bar(
                        x=qual_labels, y=qual_pcts,
                        marker_color=[_gradient_color(p, invert=True) for p in qual_pcts],
                        text=[f"{p:.0f}" for p in qual_pcts],
                        textposition="outside",
                    ))
                    fig_qual.update_layout(
                        yaxis_title="Percentile (0-100)",
                        yaxis_range=[0, 110],
                        height=350,
                        template="plotly_dark",
                    )
                    fig_qual.add_hline(y=80, line_dash="dash", line_color="green",
                                      annotation_text="Strong (80th)")
                    fig_qual.add_hline(y=20, line_dash="dash", line_color="red",
                                      annotation_text="Weak (20th)")
                    st.plotly_chart(fig_qual, use_container_width=True)

        # Fundamentals context — explains why valuation ratios are where they are
        fund_ctx = analysis.get("earnings_context", {})
        eps_history = fund_ctx.get("eps_history", [])
        context_flags = fund_ctx.get("context_flags", [])

        if eps_history or context_flags:
            st.subheader("Fundamentals Context")
            st.caption("Earnings, cash flow, and balance sheet health behind the valuation.")

            # Earnings row
            earn_cols = st.columns(4)
            trend_map = {"growing": "Growing", "declining": "Declining",
                         "volatile": "Volatile", "insufficient_data": "N/A"}
            with earn_cols[0]:
                st.metric("EPS Trend", trend_map.get(fund_ctx.get("eps_trend", "N/A"), "N/A"))
            with earn_cols[1]:
                eps_yoy = fund_ctx.get("eps_yoy_change")
                st.metric("EPS YoY", f"{eps_yoy:+.1f}%" if eps_yoy is not None else "N/A")
            with earn_cols[2]:
                cagr = fund_ctx.get("eps_3yr_cagr")
                st.metric("3-Year EPS CAGR", f"{cagr:+.1f}%" if cagr is not None else "N/A")
            with earn_cols[3]:
                pos = fund_ctx.get("net_income_positive_years", 0)
                total = fund_ctx.get("total_years", 0)
                st.metric("Profitable Years", f"{pos}/{total}" if total > 0 else "N/A")

            # Cash flow & leverage row
            cf_cols = st.columns(4)
            with cf_cols[0]:
                st.metric("FCF Trend", trend_map.get(fund_ctx.get("fcf_trend", "N/A"), "N/A"))
            with cf_cols[1]:
                fcf_yoy = fund_ctx.get("fcf_yoy_change")
                st.metric("FCF YoY", f"{fcf_yoy:+.1f}%" if fcf_yoy is not None else "N/A")
            with cf_cols[2]:
                fcf_quality = fund_ctx.get("fcf_vs_net_income")
                quality_display = {"healthy": "Healthy", "diverging": "Diverging",
                                   "negative_fcf": "Negative"}.get(fcf_quality, "N/A")
                st.metric("FCF vs Net Income", quality_display)
            with cf_cols[3]:
                nd_ebitda = fund_ctx.get("net_debt_to_ebitda")
                if fund_ctx.get("net_cash_position"):
                    st.metric("Net Debt/EBITDA", "Net Cash")
                elif nd_ebitda is not None:
                    st.metric("Net Debt/EBITDA", f"{nd_ebitda:.1f}x")
                else:
                    st.metric("Net Debt/EBITDA", "N/A")

            # Context flags
            if context_flags:
                for cflag in context_flags:
                    if any(w in cflag.lower() for w in ["declining", "dropped", "erosion",
                            "inconsistent", "structural", "negative", "lagging",
                            "burning", "highly leveraged"]):
                        st.warning(cflag)
                    elif any(w in cflag.lower() for w in ["growing", "supports",
                            "opportunity", "strong", "net cash", "conservative"]):
                        st.success(cflag)
                    else:
                        st.info(cflag)

            # EPS history table
            if eps_history:
                with st.expander("EPS History", expanded=True):
                    import pandas as pd
                    eps_df = pd.DataFrame([
                        {"Year": d["year"], "EPS": round(d["eps"], 2) if d["eps"] is not None else None}
                        for d in eps_history if d.get("year")
                    ])
                    if not eps_df.empty:
                        st.dataframe(eps_df, use_container_width=True, hide_index=True)

        # Revenue Segmentation — product and geographic breakdown
        with st.expander("Revenue Segmentation", expanded=False):
            prod_seg = fmp.get_revenue_product_segmentation(analysis["ticker"])
            geo_seg = fmp.get_revenue_geographic_segmentation(analysis["ticker"])

            has_prod = bool(prod_seg)
            has_geo = bool(geo_seg)

            if not has_prod and not has_geo:
                st.info("No revenue segmentation data available for this ticker.")
            else:
                def _build_seg_chart_and_table(seg_data, title):
                    """Build a stacked bar chart and summary table from segmentation data."""
                    # Parse into rows: [{year, segment, revenue}, ...]
                    rows = []
                    for record in seg_data:
                        year = record.get("fiscalYear")
                        segments = record.get("data", {})
                        if not year or not segments:
                            continue
                        for seg_name, rev in segments.items():
                            if rev is not None:
                                rows.append({"Year": int(year), "Segment": seg_name,
                                             "Revenue": float(rev)})
                    if not rows:
                        st.info(f"No {title.lower()} data available.")
                        return

                    seg_df = pd.DataFrame(rows)
                    seg_df = seg_df.sort_values("Year")

                    # Stacked bar chart
                    st.caption(f"**{title}** — Revenue by segment over time")
                    fig = go.Figure()
                    for seg_name in seg_df["Segment"].unique():
                        subset = seg_df[seg_df["Segment"] == seg_name]
                        fig.add_trace(go.Bar(
                            x=subset["Year"],
                            y=subset["Revenue"] / 1e9,
                            name=seg_name,
                        ))
                    fig.update_layout(
                        barmode="stack",
                        yaxis_title="Revenue ($B)",
                        xaxis_title="Fiscal Year",
                        height=400,
                        template="plotly_dark",
                        legend=dict(orientation="h", yanchor="bottom",
                                    y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Most recent year breakdown table
                    latest_year = seg_df["Year"].max()
                    latest = seg_df[seg_df["Year"] == latest_year].copy()
                    total_rev = latest["Revenue"].sum()
                    if total_rev > 0:
                        latest["% of Total"] = (latest["Revenue"] / total_rev * 100).round(1)
                    else:
                        latest["% of Total"] = 0.0
                    latest["Revenue ($B)"] = (latest["Revenue"] / 1e9).round(2)
                    display_df = latest[["Segment", "Revenue ($B)", "% of Total"]].sort_values(
                        "% of Total", ascending=False).reset_index(drop=True)
                    st.caption(f"FY{latest_year} Breakdown")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Concentration flag
                    if total_rev > 0:
                        top_pct = latest["Revenue"].max() / total_rev * 100
                        if top_pct > 70:
                            top_seg = latest.loc[latest["Revenue"].idxmax(), "Segment"]
                            st.warning(
                                f"Revenue concentration: {top_seg} accounts for "
                                f"{top_pct:.0f}% of total revenue."
                            )

                    # Coverage check — compare segment total to actual revenue
                    income_stmts = analysis.get("income_statements", [])
                    if total_rev > 0 and income_stmts:
                        actual_rev = None
                        for stmt in income_stmts:
                            if stmt.get("calendarYear") and int(stmt["calendarYear"]) == latest_year:
                                actual_rev = stmt.get("revenue")
                                break
                        if actual_rev and actual_rev > 0:
                            coverage = total_rev / actual_rev * 100
                            if coverage < 80:
                                st.warning(
                                    f"Incomplete data: segments sum to ${total_rev/1e9:.1f}B "
                                    f"but total revenue is ${actual_rev/1e9:.1f}B "
                                    f"({coverage:.0f}% coverage). Some segments may be missing from the API."
                                )

                if has_prod:
                    _build_seg_chart_and_table(prod_seg, "Product Segmentation")

                if has_geo:
                    if has_prod:
                        st.divider()
                    _build_seg_chart_and_table(geo_seg, "Geographic Segmentation")

        # Insider Trading — recent Form 4 filings
        with st.expander("Insider Trading", expanded=False):
            insider_trades = fmp.get_insider_trades(analysis["ticker"], limit=50)
            insider_stats = fmp.get_insider_trade_statistics(analysis["ticker"])

            if not insider_trades and not insider_stats:
                st.info("No insider trading data available for this ticker.")
            else:
                # Quarterly statistics summary chart
                if insider_stats:
                    st.caption("**Quarterly Insider Activity** — Shares acquired vs disposed")
                    # Take last 8 quarters
                    stats_display = insider_stats[:8]
                    stats_display = list(reversed(stats_display))

                    quarters = [f"Q{s['quarter']} {s['year']}" for s in stats_display]
                    acquired = [s.get("totalAcquired", 0) for s in stats_display]
                    disposed = [s.get("totalDisposed", 0) for s in stats_display]

                    fig_insider = go.Figure()
                    fig_insider.add_trace(go.Bar(
                        x=quarters, y=acquired,
                        name="Acquired",
                        marker_color="#4CAF50",
                    ))
                    fig_insider.add_trace(go.Bar(
                        x=quarters, y=[-d for d in disposed],
                        name="Disposed",
                        marker_color="#EF5350",
                    ))
                    fig_insider.update_layout(
                        barmode="relative",
                        yaxis_title="Shares",
                        height=350,
                        template="plotly_dark",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom",
                                    y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_insider, use_container_width=True)

                    # Buy/sell ratio for latest quarter
                    latest_stat = insider_stats[0]
                    acq = latest_stat.get("acquiredTransactions", 0)
                    disp = latest_stat.get("disposedTransactions", 0)
                    ratio = latest_stat.get("acquiredDisposedRatio", 0)
                    q_label = f"Q{latest_stat['quarter']} {latest_stat['year']}"

                    ratio_cols = st.columns(4)
                    with ratio_cols[0]:
                        st.metric("Quarter", q_label)
                    with ratio_cols[1]:
                        st.metric("Buy Transactions", acq)
                    with ratio_cols[2]:
                        st.metric("Sell Transactions", disp)
                    with ratio_cols[3]:
                        if ratio > 1:
                            st.metric("Buy/Sell Ratio", f"{ratio:.2f}", delta="Net Buying",
                                      delta_color="normal")
                        elif ratio > 0:
                            st.metric("Buy/Sell Ratio", f"{ratio:.2f}", delta="Net Selling",
                                      delta_color="inverse")
                        else:
                            st.metric("Buy/Sell Ratio", "N/A")

                # Recent transactions table
                if insider_trades:
                    st.divider()
                    st.caption("**Recent Insider Transactions** (Form 4)")

                    trade_rows = []
                    for t in insider_trades:
                        tx_type = t.get("transactionType", "")
                        price = t.get("price", 0)
                        shares = t.get("securitiesTransacted", 0)
                        trade_rows.append({
                            "Date": t.get("transactionDate", ""),
                            "Insider": t.get("reportingName", ""),
                            "Title": t.get("typeOfOwner", ""),
                            "Type": tx_type,
                            "Shares": f"{shares:,.0f}",
                            "Price": f"${price:,.2f}" if price > 0 else "—",
                            "Value": f"${price * shares:,.0f}" if price > 0 else "—",
                        })

                    if trade_rows:
                        trades_df = pd.DataFrame(trade_rows)
                        st.dataframe(trades_df, use_container_width=True, hide_index=True,
                                     height=min(len(trade_rows) * 35 + 38, 500))

        # SEC EDGAR Filing Links
        with st.expander("📄 SEC Filings", expanded=False):
            _filing_links = edgar.get_recent_filing_links(analysis["ticker"], limit=8)
            if _filing_links:
                for _fl in _filing_links:
                    _form = _fl["form"]
                    _period = _fl["period"]
                    _filed = _fl["filingDate"]
                    _url = _fl["url"]
                    st.markdown(
                        f"[**{_form}** — {_period}]({_url}) &nbsp; · &nbsp; filed {_filed}"
                    )
            else:
                st.info("No SEC filings found for this ticker (may be a foreign issuer without 10-K/10-Q filings).")

        # Earnings Call Transcript — Management Commentary
        # --- Local transcript save/load helpers ---
        _TRANSCRIPT_DIR = Path(__file__).parent / "data" / "transcripts"

        def _save_local_transcript(ticker: str, year: int, quarter: int, content: str, summary: dict):
            """Save a pasted transcript and its summary to disk."""
            tdir = _TRANSCRIPT_DIR / ticker.upper()
            tdir.mkdir(parents=True, exist_ok=True)
            payload = {
                "ticker": ticker.upper(),
                "year": year,
                "quarter": quarter,
                "content": content,
                "summary": summary,
                "saved_at": str(pd.Timestamp.now()),
            }
            fpath = tdir / (f"FY_{year}.json" if quarter == 0 else f"Q{quarter}_{year}.json")
            with open(fpath, "w") as f:
                json.dump(payload, f, indent=2)

        def _load_local_transcripts(ticker: str) -> list[dict]:
            """Load all locally saved transcripts for a ticker."""
            tdir = _TRANSCRIPT_DIR / ticker.upper()
            if not tdir.exists():
                return []
            results = []
            for fpath in sorted(list(tdir.glob("Q*_*.json")) + list(tdir.glob("FY_*.json")), reverse=True):
                try:
                    with open(fpath) as f:
                        results.append(json.load(f))
                except Exception:
                    continue
            return results

        def _load_local_transcript(ticker: str, year: int, quarter: int) -> dict | None:
            """Load a specific locally saved transcript."""
            _fname = f"FY_{year}.json" if quarter == 0 else f"Q{quarter}_{year}.json"
            fpath = _TRANSCRIPT_DIR / ticker.upper() / _fname
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        return json.load(f)
                except Exception:
                    return None
            return None

        @st.fragment
        def _transcript_fragment():
            with st.expander("Earnings Calls & Annual Reports — Management Commentary", expanded=True):
                from datetime import datetime as _dt
                from src.transcript_summarizer import summarize_transcript, is_configured as _anthropic_configured

                # Fetch available transcript dates from FMP
                _available_dates = fmp.get_earning_call_transcript_dates(analysis["ticker"], limit=20)

                # Merge locally saved transcripts into available dates
                _local_transcripts = _load_local_transcripts(analysis["ticker"])
                _existing_periods = {d["period"] for d in _available_dates} if _available_dates else set()
                if not _available_dates:
                    _available_dates = []
                for _lt in _local_transcripts:
                    _lt_q = _lt.get("quarter", 1)
                    _lt_period = f"FY {_lt['year']}" if _lt_q == 0 else f"Q{_lt_q} {_lt['year']}"
                    if _lt_period not in _existing_periods:
                        _available_dates.insert(0, {
                            "period": _lt_period,
                            "year": _lt["year"],
                            "quarter": _lt_q,
                            "source": "local",
                        })
                        _existing_periods.add(_lt_period)

                if _available_dates:
                    _quarter_options = [d["period"] for d in _available_dates]

                    _tc_cols = st.columns([3, 1])
                    with _tc_cols[0]:
                        _selected_quarter = st.selectbox(
                            "Select Period",
                            options=_quarter_options,
                            index=0,
                            key="transcript_quarter",
                        )
                    with _tc_cols[1]:
                        _show_full = st.checkbox("Show full transcript", key="show_full_transcript")

                    if not _anthropic_configured():
                        st.warning("⚠️ ANTHROPIC_API_KEY not set in .env — transcript summarization unavailable.")

                    # Find the matching date entry
                    _selected_entry = next(
                        (d for d in _available_dates if d["period"] == _selected_quarter), None
                    )
                    _tq = _selected_entry["quarter"] if _selected_entry else 1
                    _ty = _selected_entry["year"] if _selected_entry else 2025

                    # Use a form to prevent page scroll on submit
                    with st.form(key=f"transcript_form_{analysis['ticker']}"):
                        _load_clicked = st.form_submit_button("Load Transcript")

                    if _load_clicked:
                        # Check local saved transcripts first
                        _local = _load_local_transcript(analysis["ticker"], _ty, _tq)
                        if _local:
                            st.session_state["transcript_result"] = {
                                "parsed": _local.get("summary", {}),
                                "ticker": analysis["ticker"],
                                "quarter": _selected_quarter,
                                "date": _local.get("saved_at", ""),
                                "content": _local.get("content", ""),
                                "source": "local",
                            }
                        else:
                            with st.spinner(f"Fetching {analysis['ticker']} {_selected_quarter} transcript..."):
                                _transcript = fmp.get_earning_call_transcript(analysis["ticker"], _ty, _tq)
                            if _transcript and _transcript.get("content"):
                                with st.spinner("Analyzing transcript with Claude..."):
                                    _parsed = summarize_transcript(
                                        _transcript["content"],
                                        ticker=analysis["ticker"],
                                        year=_ty,
                                        quarter=_tq,
                                    )
                                st.session_state["transcript_result"] = {
                                    "parsed": _parsed,
                                    "ticker": analysis["ticker"],
                                    "quarter": _selected_quarter,
                                    "date": _transcript.get("date", ""),
                                    "content": _transcript["content"],
                                }
                            else:
                                st.session_state["transcript_result"] = None
                                st.warning(f"No transcript available for {analysis['ticker']} {_selected_quarter}. This may be due to limited coverage for smaller companies or the quarter hasn't reported yet.")

                    # Display cached transcript result
                    _tr = st.session_state.get("transcript_result")
                    if _tr and _tr.get("parsed") and _tr["ticker"] == analysis["ticker"] and _tr["quarter"] == _selected_quarter:
                        _parsed = _tr["parsed"]
                        st.caption(f"**{_tr['ticker']}** — {_tr['quarter']} Earnings Call  ·  Date: {_tr['date']}")

                        # Executive summary + sentiment
                        if _parsed.get("summary"):
                            _sentiment_colors = {
                                "bullish": "🟢",
                                "neutral": "🟡",
                                "bearish": "🔴",
                            }
                            _sent_icon = _sentiment_colors.get(_parsed.get("sentiment", "neutral"), "🟡")
                            st.markdown(f"**{_sent_icon} Overall Sentiment: {_parsed.get('sentiment', 'neutral').title()}**")
                            st.info(_parsed["summary"])

                        # Key numbers called out
                        if _parsed.get("key_numbers"):
                            st.markdown("**📊 Key Numbers Mentioned**")
                            _num_cols = st.columns(min(len(_parsed["key_numbers"]), 4))
                            for _i, _num in enumerate(_parsed["key_numbers"][:8]):
                                with _num_cols[_i % len(_num_cols)]:
                                    st.markdown(f"• {_num}")

                        # Categorized commentary
                        _cat_icons = {
                            "Pricing & Revenue": "💰",
                            "Margins & Profitability": "📈",
                            "Costs & Efficiency": "⚙️",
                            "Demand & Volume": "📦",
                            "Capital & Investment": "🏦",
                            "Outlook & Guidance": "🔮",
                            "Risks & Headwinds": "⚠️",
                        }

                        if _parsed.get("categories"):
                            st.markdown("---")
                            for _cat_name, _quotes in _parsed["categories"].items():
                                _icon = _cat_icons.get(_cat_name, "💬")
                                st.markdown(f"**{_icon} {_cat_name}**")
                                for _quote in _quotes:
                                    st.markdown(f"> {_quote}")
                                st.markdown("")
                        elif _parsed.get("source") == "error":
                            st.warning(f"Could not summarize transcript: {_parsed.get('summary', 'Unknown error')}")
                        else:
                            st.warning("Could not extract management commentary from this transcript.")

                        if _parsed.get("source") == "claude":
                            _src_label = "📁 Saved locally" if _tr.get("source") == "local" else "Cached for 90 days"
                            st.caption(f"✨ Summarized by Claude Haiku  ·  {_src_label}")

                        # Optionally show full transcript
                        if _show_full:
                            st.markdown("---")
                            st.markdown("#### Full Transcript")
                            st.markdown(_tr["content"])
                else:
                    st.info(f"No earnings call transcripts found for {analysis['ticker']} on FMP. "
                            "This may be due to limited coverage for smaller companies.")

                # Transcript Coverage Gap Detector (SEC EDGAR vs FMP) — only recent gaps
                _edgar_gaps_raw = edgar.find_transcript_gaps(
                    analysis["ticker"],
                    _available_dates if _available_dates else [],
                    limit=6,
                )
                # Filter to only gaps from the last 12 months
                from datetime import datetime as _dt, timedelta as _td
                _cutoff = (_dt.now() - _td(days=365)).strftime("%Y-%m-%d")
                _edgar_gaps = [g for g in _edgar_gaps_raw if g.get("filingDate", "2000-01-01") >= _cutoff]
                if _edgar_gaps:
                    st.markdown("---")
                    st.markdown("**📂 SEC Filing Coverage Gaps**")
                    for _gap in _edgar_gaps:
                        _gap_date = _gap.get("filingDate", "unknown date")
                        _gap_period = _gap.get("period", "unknown period")
                        _gap_form = _gap.get("form", "")
                        st.warning(
                            f"SEC {_gap_form} filing found for {_gap_period} "
                            f"(filed {_gap_date}) but no FMP transcript available. "
                            f"Try pasting the transcript manually."
                        )

                # Manual transcript paste section
                st.markdown("---")
                st.markdown("**📋 Paste a Transcript or Report Manually**")
                st.caption("Paste earnings call text or annual report commentary from any source. Claude will summarize it and save it for future use.")

                # Period type outside the form so it controls layout dynamically
                _paste_period_type = st.selectbox("Period", options=["Quarterly", "Annual"], index=0, key="paste_period_type")

                with st.form(key=f"manual_transcript_form_{analysis['ticker']}"):
                    _paste_cols = st.columns([1, 3])
                    with _paste_cols[0]:
                        if _paste_period_type == "Quarterly":
                            _paste_quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=0, key="paste_quarter")
                        else:
                            _paste_quarter = 0
                        _paste_year = st.number_input("Year", min_value=2015, max_value=2030, value=2025, key="paste_year")
                    _manual_text = st.text_area(
                        "Paste transcript text here",
                        height=200,
                        key="manual_transcript_text",
                        placeholder="Paste the full or partial earnings call transcript here...",
                    )
                    _manual_submit = st.form_submit_button("📝 Summarize & Save")

                if _manual_submit and _manual_text and len(_manual_text.strip()) > 100:
                    if not _anthropic_configured():
                        st.warning("⚠️ ANTHROPIC_API_KEY not set in .env — transcript summarization unavailable.")
                    else:
                        _effective_quarter = 0 if _paste_period_type == "Annual" else int(_paste_quarter)
                        _period_label = f"FY {_paste_year}" if _effective_quarter == 0 else f"Q{_effective_quarter} {_paste_year}"
                        with st.spinner(f"Analyzing {_period_label} with Claude..."):
                            _manual_parsed = summarize_transcript(
                                _manual_text.strip(),
                                ticker=analysis["ticker"],
                                year=int(_paste_year),
                                quarter=_effective_quarter,
                                skip_cache=True,
                            )
                        # Save to disk
                        _save_local_transcript(
                            analysis["ticker"],
                            int(_paste_year),
                            _effective_quarter,
                            _manual_text.strip(),
                            _manual_parsed,
                        )
                        st.session_state["manual_transcript_result"] = {
                            "parsed": _manual_parsed,
                            "ticker": analysis["ticker"],
                        }
                        st.success(f"Saved {_period_label} transcript for {analysis['ticker']}. "
                                   "It will now appear in the quarter selector.")
                elif _manual_submit and _manual_text and len(_manual_text.strip()) <= 100:
                    st.warning("Transcript text is too short. Paste at least a few paragraphs for meaningful analysis.")

                _mtr = st.session_state.get("manual_transcript_result")
                if _mtr and _mtr.get("parsed") and _mtr["ticker"] == analysis["ticker"]:
                    _mp = _mtr["parsed"]

                    if _mp.get("summary"):
                        _sentiment_colors = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}
                        _sent_icon = _sentiment_colors.get(_mp.get("sentiment", "neutral"), "🟡")
                        st.markdown(f"**{_sent_icon} Overall Sentiment: {_mp.get('sentiment', 'neutral').title()}**")
                        st.info(_mp["summary"])

                    if _mp.get("key_numbers"):
                        st.markdown("**📊 Key Numbers Mentioned**")
                        _mn_cols = st.columns(min(len(_mp["key_numbers"]), 4))
                        for _i, _num in enumerate(_mp["key_numbers"][:8]):
                            with _mn_cols[_i % len(_mn_cols)]:
                                st.markdown(f"• {_num}")

                    if _mp.get("categories"):
                        st.markdown("---")
                        _cat_icons = {
                            "Pricing & Revenue": "💰", "Margins & Profitability": "📈",
                            "Costs & Efficiency": "⚙️", "Demand & Volume": "📦",
                            "Capital & Investment": "🏦", "Outlook & Guidance": "🔮",
                            "Risks & Headwinds": "⚠️",
                        }
                        for _cat_name, _quotes in _mp["categories"].items():
                            _icon = _cat_icons.get(_cat_name, "💬")
                            st.markdown(f"**{_icon} {_cat_name}**")
                            for _quote in _quotes:
                                st.markdown(f"> {_quote}")
                            st.markdown("")
                    elif _mp.get("source") == "error":
                        st.warning(f"Could not summarize: {_mp.get('summary', 'Unknown error')}")

                    if _mp.get("source") == "claude":
                        st.caption("✨ Summarized by Claude Haiku")

        _transcript_fragment()

        # DCF Valuation — "What Has to Be True"
        dcf = analysis.get("dcf_valuation", {})
        # Always show DCF section — Revenue-based DCF works even when FCF data is missing
        _has_income = bool(analysis.get("income_statements"))
        dcf_has_sliders = dcf.get("dcf_price") is not None or dcf.get("has_data")
        if dcf_has_sliders or _has_income:
            # Store analysis data in session state so the fragment can access it
            st.session_state["_dcf_analysis"] = analysis
            st.session_state["_dcf_initial"] = dcf

            @st.fragment
            def _dcf_fragment():
                _analysis = st.session_state.get("_dcf_analysis", {})
                _dcf = st.session_state.get("_dcf_initial", {})

                with st.expander("DCF Valuation — What Has to Be True", expanded=True):
                    st.caption(
                        "Discounted Cash Flow model projecting future free cash flows back to today's dollars. "
                        "Adjust the inputs to test different assumptions and see how they change the implied price."
                    )

                    # Warnings
                    for w in _dcf.get("warnings", []):
                        st.warning(w)

                    assumptions = _dcf.get("assumptions", {})

                    # --- DCF Mode Toggle ---
                    dcf_mode = st.radio(
                        "Model from:",
                        options=["FCF", "Revenue"],
                        horizontal=True,
                        key="dcf_mode_toggle",
                        help="**FCF**: project from historical free cash flow (default for most stocks). "
                             "**Revenue**: derive FCF from projected revenue × target margin (better for "
                             "high-growth, negative-FCF, or SaaS-transition companies).",
                    )

                    st.markdown("#### Assumptions")

                    # Show analyst growth context based on selected mode
                    if dcf_mode == "Revenue":
                        analyst_growth = assumptions.get("analyst_revenue_growth")
                        analyst_num = assumptions.get("analyst_num_analysts")
                        if analyst_growth is not None:
                            analyst_label = f"Analyst Consensus Revenue Growth (CAGR): **{analyst_growth:+.1f}%**"
                            if analyst_num:
                                analyst_label += f"  ({analyst_num} analysts)"
                            st.info(analyst_label)
                    else:
                        # FCF mode — show FCF growth context
                        hist_fcf_growth = assumptions.get("hist_fcf_growth")
                        analyst_growth = assumptions.get("analyst_revenue_growth")
                        analyst_num = assumptions.get("analyst_num_analysts")
                        if hist_fcf_growth is not None:
                            st.info(f"Historical FCF CAGR: **{hist_fcf_growth:+.1f}%**")
                        if analyst_growth is not None:
                            _rev_label = f"Analyst Consensus Revenue Growth: {analyst_growth:+.1f}%"
                            if analyst_num:
                                _rev_label += f"  ({analyst_num} analysts)"
                            st.caption(_rev_label)

                    # Initialize dcf_display so Valuation section always has something to read
                    dcf_display = _dcf or {}

                    if dcf_mode == "FCF" and not dcf_has_sliders:
                        # FCF mode selected but no usable FCF data
                        st.warning(
                            "⚠️ Insufficient or negative historical FCF — cannot run FCF-based DCF. "
                            "Switch to **Revenue** mode above to model from revenue × target margin instead."
                        )

                    elif dcf_mode == "FCF":
                        # ===== FCF-BASED DCF =====
                        hist_growth = assumptions.get("hist_fcf_growth")
                        default_growth = assumptions.get("growth_rate", 5.0)
                        default_discount = assumptions.get("discount_rate", 10.0)
                        default_terminal = assumptions.get("terminal_growth", 2.5)

                        with st.form(key="dcf_form"):
                            ref_parts = []
                            if hist_growth is not None:
                                ref_parts.append(f"Hist FCF CAGR: {hist_growth:+.1f}%")
                            if analyst_growth is not None:
                                ref_parts.append(f"Analyst Rev Growth: {analyst_growth:+.1f}%")
                            if ref_parts:
                                st.caption(f"Reference: {' | '.join(ref_parts)}")

                            use_multistage = st.checkbox("Use multi-stage growth (recommended)", value=True,
                                                         help="Model growth deceleration over time.")

                            if use_multistage:
                                st.markdown("**Growth Phases** *(total must equal 10 years)*")
                                phase_cols = st.columns(6)
                                with phase_cols[0]:
                                    p1_rate = st.number_input("Phase 1 Rate (%)", min_value=-20.0, max_value=75.0,
                                                              value=float(default_growth), step=0.25, format="%.2f",
                                                              key="dcf_p1_rate")
                                with phase_cols[1]:
                                    p1_years = st.number_input("Years", min_value=1, max_value=9,
                                                               value=3, step=1, key="dcf_p1_years")
                                with phase_cols[2]:
                                    p2_rate = st.number_input("Phase 2 Rate (%)", min_value=-20.0, max_value=75.0,
                                                              value=max(float(default_growth) * 0.6, 2.0),
                                                              step=0.25, format="%.2f", key="dcf_p2_rate")
                                with phase_cols[3]:
                                    p2_years = st.number_input("Years", min_value=1, max_value=9,
                                                               value=4, step=1, key="dcf_p2_years")
                                with phase_cols[4]:
                                    p3_rate = st.number_input("Phase 3 Rate (%)", min_value=-20.0, max_value=75.0,
                                                              value=max(float(default_growth) * 0.3, 2.0),
                                                              step=0.25, format="%.2f", key="dcf_p3_rate")
                                with phase_cols[5]:
                                    p3_years = st.number_input("Years", min_value=0, max_value=9,
                                                               value=3, step=1, key="dcf_p3_years")

                                total_years = p1_years + p2_years + p3_years
                                if total_years != 10:
                                    st.warning(f"⚠️ Phases total {total_years} years — should equal 10")
                                user_growth = p1_rate
                            else:
                                user_growth = st.number_input(
                                    "FCF Growth Rate (%) — flat for all 10 years",
                                    min_value=-20.0, max_value=75.0,
                                    value=float(default_growth),
                                    step=0.25, format="%.2f", key="dcf_growth",
                                )

                            wacc_term_cols = st.columns(2)
                            with wacc_term_cols[0]:
                                user_discount = st.number_input(
                                    f"Discount Rate / WACC (%)  ·  Beta: {assumptions.get('beta', 1.0):.2f}",
                                    min_value=3.0, max_value=20.0,
                                    value=float(default_discount),
                                    step=0.25, format="%.2f", key="dcf_discount",
                                )
                            with wacc_term_cols[1]:
                                user_terminal = st.number_input(
                                    "Terminal Growth Rate (%)",
                                    min_value=1.0, max_value=4.0,
                                    value=float(default_terminal),
                                    step=0.25, format="%.2f", key="dcf_terminal",
                                )

                            # WACC breakdown
                            _wb = assumptions.get("wacc_breakdown", {})
                            if _wb and _wb.get("weight_debt", 0) > 0:
                                st.caption(
                                    f"WACC = ({_wb.get('weight_equity', 0):.0f}% equity × "
                                    f"{_wb.get('cost_of_equity', 0):.1f}% CoE) + "
                                    f"({_wb.get('weight_debt', 0):.0f}% debt × "
                                    f"{_wb.get('cost_of_debt', 0):.1f}% CoD × "
                                    f"(1 - {_wb.get('effective_tax_rate', 0):.0f}% tax))"
                                )

                            recalc = st.form_submit_button("🔄 Recalculate DCF", use_container_width=True)

                        dcf_display = _dcf
                        if recalc:
                            stages = None
                            if use_multistage:
                                stages = []
                                if p1_years > 0:
                                    stages.append({"rate": p1_rate / 100.0, "years": p1_years})
                                if p2_years > 0:
                                    stages.append({"rate": p2_rate / 100.0, "years": p2_years})
                                if p3_years > 0:
                                    stages.append({"rate": p3_rate / 100.0, "years": p3_years})

                            dcf_display = compute_dcf_valuation(
                                cash_flow_statements=_analysis.get("cash_flow_statements", []),
                                income_statements=_analysis.get("income_statements", []),
                                profile=_analysis.get("profile", {}),
                                balance_sheets=_analysis.get("balance_sheets", []),
                                growth_rate_override=user_growth / 100.0 if not use_multistage else None,
                                discount_rate_override=user_discount / 100.0,
                                terminal_growth_override=user_terminal / 100.0,
                                analyst_estimates=_analysis.get("analyst_estimates", []),
                                growth_stages=stages,
                            )
                            for w in dcf_display.get("warnings", []):
                                st.warning(w)

                    if dcf_mode == "Revenue":
                        # ===== REVENUE-BASED DCF =====
                        # Compute initial revenue-based DCF for defaults
                        _rev_dcf_init = compute_revenue_dcf_valuation(
                            income_statements=_analysis.get("income_statements", []),
                            cash_flow_statements=_analysis.get("cash_flow_statements", []),
                            profile=_analysis.get("profile", {}),
                            balance_sheets=_analysis.get("balance_sheets", []),
                            analyst_estimates=_analysis.get("analyst_estimates", []),
                        )
                        _rev_assumptions = _rev_dcf_init.get("assumptions", {})
                        _default_rev_growth = max(min(_rev_assumptions.get("revenue_growth", 8.0), 75.0), -20.0)
                        _default_fcf_margin = max(_rev_assumptions.get("target_fcf_margin", 10.0), 1.0)
                        _default_discount_r = _rev_assumptions.get("discount_rate", 10.0)
                        _default_terminal_r = _rev_assumptions.get("terminal_growth", 2.5)
                        _hist_rev_growth = _rev_assumptions.get("hist_rev_growth")
                        _hist_fcf_margin = _rev_assumptions.get("hist_median_fcf_margin")

                        with st.form(key="dcf_rev_form"):
                            ref_parts = []
                            if _hist_rev_growth is not None:
                                ref_parts.append(f"Hist Rev CAGR: {_hist_rev_growth:+.1f}%")
                            if _hist_fcf_margin is not None:
                                ref_parts.append(f"Hist FCF Margin: {_hist_fcf_margin:.1f}%")
                            if analyst_growth is not None:
                                ref_parts.append(f"Analyst Rev Growth: {analyst_growth:+.1f}%")
                            if ref_parts:
                                st.caption(f"Reference: {' | '.join(ref_parts)}")

                            use_multistage_rev = st.checkbox("Use multi-stage growth (recommended)", value=True,
                                                             help="Model revenue growth deceleration over time.",
                                                             key="dcf_rev_multistage")

                            if use_multistage_rev:
                                st.markdown("**Revenue Growth Phases** *(total must equal 10 years)*")
                                phase_cols = st.columns(6)
                                with phase_cols[0]:
                                    rp1_rate = st.number_input("Phase 1 Rate (%)", min_value=-20.0, max_value=75.0,
                                                               value=float(_default_rev_growth), step=0.25, format="%.2f",
                                                               key="dcf_rp1_rate")
                                with phase_cols[1]:
                                    rp1_years = st.number_input("Years", min_value=1, max_value=9,
                                                                value=3, step=1, key="dcf_rp1_years")
                                with phase_cols[2]:
                                    rp2_rate = st.number_input("Phase 2 Rate (%)", min_value=-20.0, max_value=75.0,
                                                               value=max(float(_default_rev_growth) * 0.6, 2.0),
                                                               step=0.25, format="%.2f", key="dcf_rp2_rate")
                                with phase_cols[3]:
                                    rp2_years = st.number_input("Years", min_value=1, max_value=9,
                                                                value=4, step=1, key="dcf_rp2_years")
                                with phase_cols[4]:
                                    rp3_rate = st.number_input("Phase 3 Rate (%)", min_value=-20.0, max_value=75.0,
                                                               value=max(float(_default_rev_growth) * 0.3, 2.0),
                                                               step=0.25, format="%.2f", key="dcf_rp3_rate")
                                with phase_cols[5]:
                                    rp3_years = st.number_input("Years", min_value=0, max_value=9,
                                                                value=3, step=1, key="dcf_rp3_years")
                                total_years_rev = rp1_years + rp2_years + rp3_years
                                if total_years_rev != 10:
                                    st.warning(f"⚠️ Phases total {total_years_rev} years — should equal 10")
                                user_rev_growth = rp1_rate
                            else:
                                user_rev_growth = st.number_input(
                                    "Revenue Growth Rate (%) — flat for all 10 years",
                                    min_value=-20.0, max_value=75.0,
                                    value=float(_default_rev_growth),
                                    step=0.25, format="%.2f", key="dcf_rev_growth",
                                )

                            # Target FCF margin
                            user_fcf_margin = st.number_input(
                                "Target FCF Margin (%)",
                                min_value=1.0, max_value=50.0,
                                value=float(_default_fcf_margin),
                                step=0.5, format="%.1f", key="dcf_fcf_margin",
                                help="What % of revenue converts to free cash flow at maturity. "
                                     "Use historical median as a starting point.",
                            )

                            wacc_term_cols_r = st.columns(2)
                            with wacc_term_cols_r[0]:
                                user_discount_r = st.number_input(
                                    f"Discount Rate / WACC (%)  ·  Beta: {_rev_assumptions.get('beta', 1.0):.2f}",
                                    min_value=3.0, max_value=20.0,
                                    value=float(_default_discount_r),
                                    step=0.25, format="%.2f", key="dcf_rev_discount",
                                )
                            with wacc_term_cols_r[1]:
                                user_terminal_r = st.number_input(
                                    "Terminal Growth Rate (%)",
                                    min_value=1.0, max_value=4.0,
                                    value=float(_default_terminal_r),
                                    step=0.25, format="%.2f", key="dcf_rev_terminal",
                                )

                            # WACC breakdown
                            _wb_r = _rev_assumptions.get("wacc_breakdown", {})
                            if _wb_r and _wb_r.get("weight_debt", 0) > 0:
                                st.caption(
                                    f"WACC = ({_wb_r.get('weight_equity', 0):.0f}% equity × "
                                    f"{_wb_r.get('cost_of_equity', 0):.1f}% CoE) + "
                                    f"({_wb_r.get('weight_debt', 0):.0f}% debt × "
                                    f"{_wb_r.get('cost_of_debt', 0):.1f}% CoD × "
                                    f"(1 - {_wb_r.get('effective_tax_rate', 0):.0f}% tax))"
                                )

                            recalc_rev = st.form_submit_button("🔄 Recalculate DCF", use_container_width=True)

                        dcf_display = _rev_dcf_init
                        if recalc_rev:
                            stages = None
                            if use_multistage_rev:
                                stages = []
                                if rp1_years > 0:
                                    stages.append({"rate": rp1_rate / 100.0, "years": rp1_years})
                                if rp2_years > 0:
                                    stages.append({"rate": rp2_rate / 100.0, "years": rp2_years})
                                if rp3_years > 0:
                                    stages.append({"rate": rp3_rate / 100.0, "years": rp3_years})

                            dcf_display = compute_revenue_dcf_valuation(
                                income_statements=_analysis.get("income_statements", []),
                                cash_flow_statements=_analysis.get("cash_flow_statements", []),
                                profile=_analysis.get("profile", {}),
                                balance_sheets=_analysis.get("balance_sheets", []),
                                revenue_growth_override=user_rev_growth / 100.0 if not use_multistage_rev else None,
                                target_fcf_margin_override=user_fcf_margin / 100.0,
                                discount_rate_override=user_discount_r / 100.0,
                                terminal_growth_override=user_terminal_r / 100.0,
                                analyst_estimates=_analysis.get("analyst_estimates", []),
                                growth_stages=stages,
                            )
                            for w in dcf_display.get("warnings", []):
                                st.warning(w)

                        # Update variables for the shared display section below
                        user_growth = user_rev_growth
                        user_discount = user_discount_r
                        user_terminal = user_terminal_r
                        use_multistage = use_multistage_rev
                        recalc = recalc_rev
                        if use_multistage_rev:
                            p1_rate, p1_years = rp1_rate, rp1_years
                            p2_rate, p2_years = rp2_rate, rp2_years
                            p3_rate, p3_years = rp3_rate, rp3_years

                    # --- Summary metrics ---
                    st.markdown("#### Valuation")
                    # Use validated price (corrected by price validator) instead of stale FMP price
                    _dcf_current_price = analysis.get("current_price") or dcf_display.get("current_price", 0)
                    dcf_cols = st.columns(4)
                    with dcf_cols[0]:
                        st.metric("Current Price",
                                  f"${_dcf_current_price:.2f}" if _dcf_current_price else "N/A")
                    with dcf_cols[1]:
                        dcf_price = dcf_display.get("dcf_price")
                        st.metric("DCF Intrinsic Value",
                                  f"${dcf_price:,.2f}" if dcf_price else "N/A")
                    with dcf_cols[2]:
                        # Recalculate upside using validated price
                        if dcf_price and _dcf_current_price and _dcf_current_price > 0:
                            dcf_upside = round((dcf_price / _dcf_current_price - 1) * 100, 1)
                        else:
                            dcf_upside = dcf_display.get("upside_pct")
                        if dcf_upside is not None:
                            st.metric("Upside/Downside", f"{dcf_upside:+.1f}%",
                                      delta=f"{dcf_upside:+.1f}%",
                                      delta_color="normal" if dcf_upside >= 0 else "inverse")
                        else:
                            st.metric("Upside/Downside", "N/A")
                    with dcf_cols[3]:
                        terminal_pct = dcf_display.get("terminal_pct", 0)
                        st.metric("Terminal Value %", f"{terminal_pct:.0f}%",
                                  help="Percentage of total value from terminal value. Over 75% means the valuation leans heavily on long-term assumptions.")

                    # --- Key assumptions summary ---
                    assumptions_updated = dcf_display.get("assumptions", assumptions)
                    if use_multistage and recalc:
                        growth_desc = f"Growth: {p1_rate:.1f}%×{p1_years}yr → {p2_rate:.1f}%×{p2_years}yr → {p3_rate:.1f}%×{p3_years}yr"
                    else:
                        growth_desc = f"Growth: {user_growth:.2f}%"

                    if dcf_mode == "Revenue":
                        base_rev = assumptions_updated.get("base_revenue", 0)
                        fcf_margin = assumptions_updated.get("target_fcf_margin", 0)
                        st.caption(
                            f"📊 Revenue-based  ·  Base Rev: ${base_rev/1e9:.2f}B  ·  "
                            f"FCF Margin: {fcf_margin:.1f}%  ·  "
                            f"{growth_desc}  ·  "
                            f"WACC: {user_discount:.2f}%  ·  "
                            f"Terminal: {user_terminal:.2f}%  ·  "
                            f"Shares: {assumptions_updated.get('shares_outstanding', 0)/1e9:.2f}B"
                        )
                    else:
                        base_fcf = assumptions_updated.get("base_fcf", 0)
                        st.caption(
                            f"Base FCF: ${base_fcf/1e9:.2f}B  ·  "
                            f"{growth_desc}  ·  "
                            f"WACC: {user_discount:.2f}%  ·  "
                            f"Terminal: {user_terminal:.2f}%  ·  "
                            f"Shares: {assumptions_updated.get('shares_outstanding', 0)/1e9:.2f}B"
                        )

                    # --- Sensitivity table (the real value) ---
                    st.markdown("#### Sensitivity Table — Implied Price per Share")
                    st.caption("Rows = FCF growth rate, Columns = discount rate (WACC). Find what has to be true for the stock to be worth its current price.")

                    sensitivity = dcf_display.get("sensitivity", [])
                    if sensitivity:
                        sens_df = pd.DataFrame(sensitivity)
                        sens_df = sens_df.rename(columns={"growth_rate": "Growth %"})
                        sens_df = sens_df.set_index("Growth %")

                        # Style: highlight cells near current price (use validated price)
                        current_p = _dcf_current_price or 0

                        def color_cell(val):
                            if val == "N/A" or not isinstance(val, (int, float)):
                                return "background-color: #1a1a2e; color: #555"
                            if current_p > 0:
                                ratio = val / current_p
                                if ratio > 1.3:
                                    return "background-color: #1a3a1a; color: #4ade80"  # green — undervalued
                                elif ratio > 0.9:
                                    return "background-color: #3a3a1a; color: #fbbf24"  # yellow — fair
                                else:
                                    return "background-color: #3a1a1a; color: #f87171"  # red — overvalued
                            return ""

                        styled = sens_df.style.map(color_cell)
                        styled = styled.format(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
                        st.dataframe(styled, use_container_width=True, height=360)

                        st.caption(
                            "🟢 Green = implied price > 130% of current (undervalued at these assumptions)  ·  "
                            "🟡 Yellow = within ±30% of current (fairly valued)  ·  "
                            "🔴 Red = implied price < 90% of current (overvalued)"
                        )

                    # --- Projected FCF chart ---
                    projected = dcf_display.get("projected_fcfs", [])
                    if projected:
                        st.markdown("#### Projected Free Cash Flows")
                        fig_fcf = go.Figure()
                        years = [p["year"] for p in projected]
                        fcfs = [p["fcf"] / 1e9 for p in projected]  # in billions
                        pvs = [p["present_value"] / 1e9 for p in projected]

                        fig_fcf.add_trace(go.Bar(
                            x=years, y=fcfs,
                            name="Projected FCF",
                            marker_color="rgba(100, 149, 237, 0.7)",
                            text=[f"${f:.2f}B" for f in fcfs],
                            textposition="outside",
                        ))
                        fig_fcf.add_trace(go.Bar(
                            x=years, y=pvs,
                            name="Present Value",
                            marker_color="rgba(100, 149, 237, 0.3)",
                            text=[f"${p:.2f}B" for p in pvs],
                            textposition="outside",
                        ))
                        fig_fcf.update_layout(
                            xaxis_title="Year",
                            yaxis_title="$ Billions",
                            barmode="group",
                            height=500,
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig_fcf, use_container_width=True)

            _dcf_fragment()

        # (Removed — DCF section now always renders with Revenue toggle available)


# ================================================================== #
# TAB 3: Ratio Comparison
# ================================================================== #
with tab3:
    st.header("Ratio Comparison")

    # Universe selector — pick which universes to draw tickers from
    rc_uni_map = universe_display_map()
    rc_display = st.multiselect(
        "Select Industries",
        options=sorted(rc_uni_map.keys(), key=lambda x: ("⚠️" in x, x)),
        default=[],
        help="Choose one or more industries to load tickers from. ⚠️ = inverse plays.",
        key="rc_universes",
    )

    rc_universes = [rc_uni_map[d] for d in rc_display]
    all_tickers_df2 = pd.DataFrame()
    ticker_labels2 = {}
    if rc_universes:
        frames2 = [load_universe(u) for u in rc_universes]
        all_tickers_df2 = pd.concat(frames2, ignore_index=True).drop_duplicates(subset="ticker", keep="first")
        ticker_labels2 = {
            row["ticker"]: f"{row['ticker']} — {row['company_name']}"
            for _, row in all_tickers_df2.iterrows() if row.get("company_name")
        }
    ticker_opts = sorted(all_tickers_df2["ticker"].unique().tolist()) if not all_tickers_df2.empty else []

    # Quick-fill by segment
    segments = sorted(all_tickers_df2["segment"].dropna().unique().tolist()) if not all_tickers_df2.empty else []
    selected_segment = st.selectbox(
        "Quick-fill by Segment",
        options=["(manual selection)"] + segments,
        index=0,
        help="Pick a segment to auto-populate all its tickers, or choose manual selection.",
        key="compare_segment",
    )

    if selected_segment != "(manual selection)":
        segment_tickers = sorted(
            all_tickers_df2[all_tickers_df2["segment"] == selected_segment]["ticker"].unique().tolist()
        )
        default_tickers = segment_tickers[:10]
    else:
        segment_tickers = ticker_opts
        default_tickers = []

    compare_tickers = st.multiselect(
        "Select Tickers to Compare",
        options=segment_tickers if selected_segment != "(manual selection)" else ticker_opts,
        default=default_tickers,
        max_selections=10,
        format_func=lambda t: ticker_labels2.get(t, t),
        help="Select up to 10 tickers for side-by-side comparison.",
    )

    ratio_choices = [
        "pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda", "ev_revenue",
        "fcf_yield", "roe", "roa", "gross_margin", "operating_margin",
        "net_margin", "debt_to_equity", "current_ratio", "quick_ratio",
        "interest_coverage",
    ]
    ratio_pretty = {
        "pe_ratio": "P/E Ratio", "ps_ratio": "P/S Ratio",
        "pb_ratio": "P/B Ratio", "ev_ebitda": "EV/EBITDA",
        "ev_revenue": "EV/Revenue", "fcf_yield": "FCF Yield",
        "roe": "ROE", "roa": "ROA",
        "gross_margin": "Gross Margin", "operating_margin": "Operating Margin",
        "net_margin": "Net Margin", "debt_to_equity": "Debt/Equity",
        "current_ratio": "Current Ratio", "quick_ratio": "Quick Ratio",
        "interest_coverage": "Interest Coverage",
    }

    selected_ratio = st.selectbox(
        "Select Ratio to Compare",
        options=ratio_choices,
        format_func=lambda x: ratio_pretty.get(x, x),
    )

    if compare_tickers and st.button("Compare", key="compare_btn"):
        if not fmp.is_configured:
            st.error("FMP API key is required for ratio comparison. Set FMP_API_KEY.")
        else:
            comparison_data = []
            progress = st.progress(0)
            for i, ticker in enumerate(compare_tickers):
                progress.progress((i + 1) / len(compare_tickers), text=f"Fetching {ticker}...")
                info_row = all_tickers_df2[all_tickers_df2["ticker"] == ticker]
                universe_info = {}
                if not info_row.empty:
                    r = info_row.iloc[0]
                    universe_info = {"company_name": r.get("company_name", ""), "segment": r.get("segment", "")}

                analysis = analyze_ticker(ticker, fmp, universe_info,
                                          history_years=history_years)
                metric_data = analysis.get("metrics", {}).get(selected_ratio, {})
                comparison_data.append({
                    "Ticker": ticker,
                    "Company": analysis.get("company_name", ""),
                    "Current": metric_data.get("current"),
                    f"{history_years}yr Median": metric_data.get("hist_median"),
                    f"{history_years}yr Low": metric_data.get("hist_low"),
                    f"{history_years}yr High": metric_data.get("hist_high"),
                    "Percentile": metric_data.get("percentile"),
                })
            progress.empty()

            comp_df = pd.DataFrame(comparison_data)
            st.session_state["comparison_df"] = comp_df
            st.session_state["comparison_ratio"] = selected_ratio

    if "comparison_df" in st.session_state:
        comp_df = st.session_state["comparison_df"]
        ratio_name = ratio_pretty.get(
            st.session_state.get("comparison_ratio", ""),
            st.session_state.get("comparison_ratio", "")
        )

        st.subheader(f"{ratio_name} Comparison")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Bar chart
        chart_df = comp_df.dropna(subset=["Current"])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="Ticker",
                y="Current",
                text="Current",
                title=f"{ratio_name} — Current Values",
                template="plotly_dark",
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# ================================================================== #
# TAB 4: Balance Sheet Health
# ================================================================== #
with tab4:
    st.header("Balance Sheet Health")

    st.write(
        "Focus on balance sheet strength: debt-to-equity, current ratio, "
        "quick ratio, and interest coverage. Ranked by overall balance sheet score."
    )

    # Use scan results if available, otherwise prompt
    if "scan_results" in st.session_state and not st.session_state["scan_results"].empty:
        bs_df = st.session_state["scan_results"].copy()
    else:
        bs_uni_map = universe_display_map()
        bs_display = st.multiselect(
            "Select industries for balance sheet analysis",
            options=sorted(bs_uni_map.keys(), key=lambda x: ("⚠️" in x, x)),
            default=[],
            help="⚠️ = inverse plays (sectors likely hurt by conflict).",
            key="bs_universes",
        )
        bs_universes = [bs_uni_map[d] for d in bs_display]
        if st.button("Analyze Balance Sheets", key="bs_scan"):
            if not fmp.is_configured:
                st.error("FMP API key is required. Set FMP_API_KEY.")
                st.stop()
            progress = st.progress(0, text="Starting...")

            def bs_progress(cur, total, ticker):
                progress.progress(cur / total, text=f"Analyzing {ticker} ({cur}/{total})")

            bs_df = scan_all_universes(
                fmp_client=fmp,
                tradier_client=tradier,
                universe_names=bs_universes,
                progress_callback=bs_progress,
                history_years=history_years,
            )
            progress.empty()
            st.session_state["scan_results"] = bs_df
        else:
            bs_df = pd.DataFrame()

    if not bs_df.empty:
        bs_cols = {
            "debt_to_equity": "Debt/Equity",
            "debt_to_equity_pct": "D/E %ile",
            "current_ratio": "Current Ratio",
            "current_ratio_pct": "CR %ile",
            "quick_ratio": "Quick Ratio",
            "quick_ratio_pct": "QR %ile",
            "interest_coverage": "Interest Coverage",
            "interest_coverage_pct": "IC %ile",
        }

        avail_bs_cols = ["ticker", "company_name", "segment"]
        for col in bs_cols:
            if col in bs_df.columns:
                avail_bs_cols.append(col)

        bs_display = bs_df[avail_bs_cols].copy()

        # Compute balance sheet composite
        scoring_cfg = load_scoring_config()
        higher_is_better = set(scoring_cfg.get("higher_is_better", []))
        pct_cols = [c for c in bs_display.columns if c.endswith("_pct")]

        if pct_cols:
            def bs_score(row):
                vals = []
                for pc in pct_cols:
                    v = row.get(pc)
                    if pd.notna(v):
                        # Determine base metric name
                        base = pc.replace("_pct", "")
                        if base in higher_is_better:
                            vals.append(100.0 - v)
                        else:
                            vals.append(v)
                return sum(vals) / len(vals) if vals else None

            bs_display["BS Score"] = bs_display.apply(bs_score, axis=1)
            bs_display = bs_display.sort_values("BS Score", ascending=True, na_position="last")

        rename_map = {"ticker": "Ticker", "company_name": "Company", "segment": "Segment"}
        rename_map.update({k: v for k, v in bs_cols.items() if k in bs_display.columns})
        bs_display = bs_display.rename(columns=rename_map)

        st.subheader(f"Balance Sheet Rankings ({len(bs_display)} tickers)")

        # Map display %ile column names back to raw metric keys
        _bs_pct_raw = {v: k.replace("_pct", "") for k, v in bs_cols.items() if k.endswith("_pct")}
        _bs_hib = {"current_ratio", "quick_ratio", "interest_coverage"}

        def color_pct_bs(val, col_name):
            if pd.isna(val):
                return ""
            raw_key = _bs_pct_raw.get(col_name, "")
            high_is_good = raw_key in _bs_hib
            if high_is_good:
                if val > 80:
                    return "background-color: rgba(0, 180, 0, 0.3)"
                elif val < 20:
                    return "background-color: rgba(220, 50, 50, 0.3)"
            else:
                if val < 20:
                    return "background-color: rgba(0, 180, 0, 0.3)"
                elif val > 80:
                    return "background-color: rgba(220, 50, 50, 0.3)"
            return "background-color: rgba(200, 200, 0, 0.2)"

        pct_display_cols = [c for c in bs_display.columns if "%ile" in c]
        styled_bs = bs_display.style
        if pct_display_cols:
            for pc in pct_display_cols:
                styled_bs = styled_bs.map(lambda val, _c=pc: color_pct_bs(val, _c), subset=[pc])

        st.dataframe(styled_bs, use_container_width=True, height=600, hide_index=True)

        # Flag deteriorating balance sheets
        st.subheader("Balance Sheet Warnings")
        if "D/E %ile" in bs_display.columns:
            warnings = bs_display[bs_display["D/E %ile"] > 80] if "D/E %ile" in bs_display.columns else pd.DataFrame()
            if not warnings.empty:
                st.warning(f"{len(warnings)} tickers have Debt/Equity near 5-year highs:")
                for _, row in warnings.iterrows():
                    st.write(f"  - **{row['Ticker']}** ({row['Company']}): D/E at {row.get('D/E %ile', 'N/A'):.0f}th percentile")
            else:
                st.success("No tickers with critically high Debt/Equity levels.")
    else:
        st.info("Run a scan from the Screener Dashboard or click 'Analyze Balance Sheets' above.")


# ================================================================== #
# TAB 5: EIA Inventories
# ================================================================== #
with tab5:
    st.header("EIA Inventories")

    if not eia.is_configured:
        st.error("EIA API key is required. Set EIA_API_KEY environment variable.")
    else:
        weeks_back = st.selectbox(
            "History",
            options=[52, 104, 156, 260],
            index=1,
            format_func=lambda w: f"{w} weeks ({w // 52} yr{'s' if w > 52 else ''})",
            key="eia_weeks",
        )

        if st.button("Fetch EIA Data", type="primary", key="eia_fetch"):
            with st.spinner("Fetching EIA data..."):
                eia_data = {}
                for series_name in ["crude_stocks", "gasoline_stocks", "distillate_stocks"]:
                    eia_data[series_name] = eia.get_petroleum_series(series_name, weeks=weeks_back)
                eia_data["natgas_storage"] = eia.get_natural_gas_storage(weeks=weeks_back)

                # Also fetch production/imports/refinery if available
                for series_name in ["crude_production", "crude_imports", "refinery_inputs"]:
                    eia_data[series_name] = eia.get_petroleum_series(series_name, weeks=weeks_back)

            st.session_state["eia_data"] = eia_data
            st.success("EIA data fetched.")

        if "eia_data" in st.session_state:
            eia_data = st.session_state["eia_data"]

            # --- Petroleum Inventories ---
            st.subheader("Petroleum Inventories")

            inv_tabs = st.tabs(["Crude Oil", "Gasoline", "Distillates", "Natural Gas Storage",
                                "Production & Imports"])

            def _build_inv_df(data: list[dict]) -> pd.DataFrame:
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                df["period"] = pd.to_datetime(df["period"])
                df = df.sort_values("period")
                return df

            def _show_inventory(data: list[dict], label: str, units: str, tab):
                with tab:
                    df = _build_inv_df(data)
                    if df.empty:
                        st.warning(f"No data available for {label}.")
                        return

                    # Latest values summary
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else None

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            f"Latest ({latest['period'].strftime('%Y-%m-%d')})",
                            f"{latest['value']:,.0f}",
                            delta=f"{latest['value'] - prev['value']:+,.0f}" if prev is not None else None,
                        )
                    with col2:
                        if len(df) >= 52:
                            yr_ago = df.iloc[-52]
                            yoy_change = latest["value"] - yr_ago["value"]
                            st.metric("vs Year Ago", f"{yr_ago['value']:,.0f}", delta=f"{yoy_change:+,.0f}")
                        else:
                            st.metric("vs Year Ago", "N/A")
                    with col3:
                        avg_val = df["value"].mean()
                        vs_avg = latest["value"] - avg_val
                        st.metric(f"vs {len(df)}wk Avg", f"{avg_val:,.0f}", delta=f"{vs_avg:+,.0f}")

                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["period"], y=df["value"],
                        mode="lines", name=label,
                        line=dict(color="#1f77b4", width=2),
                    ))

                    # Add average line
                    fig.add_hline(
                        y=avg_val, line_dash="dash", line_color="gray",
                        annotation_text=f"Avg: {avg_val:,.0f}",
                    )

                    fig.update_layout(
                        title=f"{label} ({units})",
                        xaxis_title="Date",
                        yaxis_title=units,
                        template="plotly_dark",
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Weekly changes table
                    with st.expander("Weekly Changes"):
                        changes_df = df.copy()
                        changes_df["Weekly Change"] = changes_df["value"].diff()
                        changes_df = changes_df.sort_values("period", ascending=False).head(20)
                        changes_df["period"] = changes_df["period"].dt.strftime("%Y-%m-%d")
                        changes_df = changes_df.rename(columns={
                            "period": "Week", "value": units,
                        })
                        st.dataframe(changes_df[["Week", units, "Weekly Change"]],
                                     use_container_width=True, hide_index=True)

            _show_inventory(
                eia_data.get("crude_stocks", []),
                "U.S. Crude Oil Ending Stocks", "Thousand Barrels", inv_tabs[0])
            _show_inventory(
                eia_data.get("gasoline_stocks", []),
                "U.S. Motor Gasoline Ending Stocks", "Thousand Barrels", inv_tabs[1])
            _show_inventory(
                eia_data.get("distillate_stocks", []),
                "U.S. Distillate Fuel Oil Ending Stocks", "Thousand Barrels", inv_tabs[2])
            _show_inventory(
                eia_data.get("natgas_storage", []),
                "Lower 48 Natural Gas Working Storage", "Billion Cubic Feet", inv_tabs[3])

            # Production & Imports tab
            with inv_tabs[4]:
                prod_series = {
                    "crude_production": ("U.S. Crude Oil Production", "#2ca02c"),
                    "crude_imports": ("U.S. Crude Oil Imports", "#ff7f0e"),
                    "refinery_inputs": ("Refinery & Blender Net Inputs", "#d62728"),
                }

                fig = go.Figure()
                for series_name, (label, color) in prod_series.items():
                    data = eia_data.get(series_name, [])
                    df = _build_inv_df(data)
                    if df.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=df["period"], y=df["value"],
                        mode="lines", name=label,
                        line=dict(color=color, width=2),
                    ))

                fig.update_layout(
                    title="U.S. Crude Oil Supply & Demand (Thousand Barrels per Day)",
                    xaxis_title="Date",
                    yaxis_title="Thousand Barrels per Day",
                    template="plotly_dark",
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Latest values table
                latest_rows = []
                for series_name, (label, _) in prod_series.items():
                    data = eia_data.get(series_name, [])
                    if data:
                        latest_rows.append({
                            "Series": label,
                            "Latest": f"{data[0]['value']:,.0f}",
                            "Date": data[0]["period"],
                        })
                if latest_rows:
                    st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)

        else:
            st.info("Click 'Fetch EIA Data' to load inventory data.")


# ================================================================== #
# TAB 6: Settings
# ================================================================== #
# ------------------------------------------------------------------ #
# Tab 6 — Commodity Prices
# ------------------------------------------------------------------ #
with tab6:
    st.header("Commodity Prices")
    st.write("Live and historical commodity prices from EIA (energy) and Yahoo Finance (metals & agriculture).")

    commodity_tabs = st.tabs(["Energy", "Metals", "Agriculture"])

    # --- Stock overlay helpers ---
    _PERIOD_TO_DAYS = {
        "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
        "2y": 730, "5y": 1825, "10y": 3650,
    }

    def _from_date_str(days: int) -> str:
        """Convert a day count to a 'YYYY-MM-DD' start date string."""
        return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    def _fetch_stock_overlay(tickers: list[str], from_date: str) -> dict:
        """Fetch historical close prices for overlay tickers via FMP."""
        result = {}
        for t in tickers:
            data = fmp.get_historical_price(t, from_date=from_date)
            if data:
                result[t] = [{"period": d["date"], "value": d["close"]} for d in data]
        return result

    def _normalize_series(values: list[float]) -> list[float]:
        """Convert a price series to % change from start."""
        if not values or values[0] == 0:
            return values
        base = values[0]
        return [(v / base - 1) * 100 for v in values]

    def _render_stock_overlay_controls(tab_key: str):
        """Render universe selector + ticker multiselect for a commodity sub-tab."""
        st.markdown("##### 📈 Stock Price Overlay")
        overlay_cols = st.columns([1, 2])
        with overlay_cols[0]:
            overlay_uni_map = universe_display_map()
            overlay_options = sorted(overlay_uni_map.keys(), key=lambda x: ("⚠️" in x, x))
            selected_uni_display = st.selectbox(
                "Industry",
                options=[""] + overlay_options,
                index=0,
                key=f"{tab_key}_overlay_universe",
                help="Select an industry to populate the ticker list.",
            )
            selected_uni = overlay_uni_map.get(selected_uni_display, "")
        with overlay_cols[1]:
            ticker_options = []
            ticker_labels = {}
            if selected_uni:
                uni_df = load_universe(selected_uni)
                for _, row in uni_df.iterrows():
                    t = row["ticker"]
                    name = row.get("company_name", "")
                    ticker_options.append(t)
                    ticker_labels[t] = f"{t} — {name}" if name else t
            selected_tickers = st.multiselect(
                "Overlay Tickers",
                options=ticker_options,
                format_func=lambda x: ticker_labels.get(x, x),
                key=f"{tab_key}_overlay_tickers",
                help="Select stocks to overlay on the commodity chart.",
            )
        normalize = st.checkbox(
            "Normalize to % change (rebase all series to 0%)",
            key=f"{tab_key}_normalize",
        )
        return selected_tickers, normalize

    def _add_stock_traces(fig, tickers, from_date, normalize=False):
        """Add stock price traces to a Plotly figure on a secondary y-axis."""
        if not tickers:
            return
        stock_data = _fetch_stock_overlay(tickers, from_date)
        for ticker, data in stock_data.items():
            dates = [d["period"] for d in reversed(data)]
            values = [d["value"] for d in reversed(data)]
            if normalize:
                values = _normalize_series(values)
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode="lines",
                name=f"{ticker}",
                line=dict(dash="dash"),
                yaxis="y2" if not normalize else "y",
            ))

    # --- Energy sub-tab (EIA) ---
    with commodity_tabs[0]:
        st.subheader("Energy Commodities (EIA)")

        energy_items = list(SPOT_PRICE_SERIES.items())

        # Live futures quotes for WTI & Brent (yfinance, near-real-time)
        _LIVE_FUTURES = {
            "wti_crude": "CL=F",
            "brent_crude": "BZ=F",
        }

        # Current prices — two rows of 4
        for row_start in range(0, len(energy_items), 4):
            row_items = energy_items[row_start:row_start + 4]
            row_cols = st.columns(4)
            for idx, (key, info) in enumerate(row_items):
                with row_cols[idx]:
                    # Use live futures for WTI/Brent, EIA spot for everything else
                    if key in _LIVE_FUTURES:
                        quote = get_commodity_quote_by_symbol(
                            _LIVE_FUTURES[key], info["label"], info["units"]
                        )
                        if quote:
                            st.metric(
                                label=info["label"],
                                value=f"{quote['price']:.2f}",
                                delta=f"{quote['change']:+.2f} ({info['units']})",
                            )
                        else:
                            # Fallback to EIA if yfinance fails
                            data = eia.get_spot_price(key, days=5)
                            if data:
                                current = data[0]["value"]
                                prev = data[1]["value"] if len(data) > 1 else current
                                change = current - prev
                                st.metric(
                                    label=f"{info['label']} (as of {data[0]['period']})",
                                    value=f"{current:.2f}",
                                    delta=f"{change:+.2f} ({info['units']})",
                                )
                            else:
                                st.metric(label=info["label"], value="N/A")
                    else:
                        data = eia.get_spot_price(key, days=5)
                        if data:
                            current = data[0]["value"]
                            prev = data[1]["value"] if len(data) > 1 else current
                            change = current - prev
                            st.metric(
                                label=f"{info['label']} (as of {data[0]['period']})",
                                value=f"{current:.2f}",
                                delta=f"{change:+.2f} ({info['units']})",
                            )
                        else:
                            st.metric(label=info["label"], value="N/A")

        st.divider()

        # Historical chart
        energy_chart_options = {k: v["label"] for k, v in SPOT_PRICE_SERIES.items()}
        energy_period = st.selectbox(
            "History Period",
            options=[("260", "1 Year"), ("520", "2 Years"), ("1300", "5 Years"), ("2600", "10 Years")],
            index=0,
            format_func=lambda x: x[1],
            key="energy_period",
        )

        energy_selected = st.multiselect(
            "Select Commodities",
            options=list(energy_chart_options.keys()),
            default=["wti_crude", "brent_crude", "henry_hub"],
            format_func=lambda x: energy_chart_options[x],
            key="energy_commodities",
        )

        # Stock overlay controls
        energy_overlay_tickers, energy_normalize = _render_stock_overlay_controls("energy")

        if energy_selected or energy_overlay_tickers:
            energy_days = int(energy_period[0])
            energy_from_date = _from_date_str(energy_days)

            fig = go.Figure()
            for key in energy_selected:
                info = SPOT_PRICE_SERIES[key]
                data = eia.get_spot_price(key, days=energy_days)
                if data:
                    dates = [d["period"] for d in reversed(data)]
                    values = [d["value"] for d in reversed(data)]
                    if energy_normalize:
                        values = _normalize_series(values)
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        mode="lines",
                        name=f"{info['label']} ({info['units']})",
                    ))

            _add_stock_traces(fig, energy_overlay_tickers, energy_from_date, energy_normalize)

            layout_kwargs = dict(
                title="Energy Commodity Prices" + (" (% Change)" if energy_normalize else ""),
                xaxis_title="Date",
                yaxis_title="% Change" if energy_normalize else "Price",
                height=500,
                hovermode="x unified",
            )
            if energy_overlay_tickers and not energy_normalize:
                layout_kwargs["yaxis2"] = dict(
                    title="Stock Price ($)", overlaying="y", side="right",
                )
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)

    # --- Metals sub-tab (yfinance) ---
    with commodity_tabs[1]:
        st.subheader("Metals (Yahoo Finance)")

        metals = {k: v for k, v in COMMODITY_SYMBOLS.items()
                  if k in ("gold", "silver", "copper", "platinum", "palladium", "uranium")}

        # Current prices — two rows of 4
        metals_items = list(metals.items())
        for row_start in range(0, len(metals_items), 4):
            row_items = metals_items[row_start:row_start + 4]
            row_cols = st.columns(4)
            for idx, (key, info) in enumerate(row_items):
                with row_cols[idx]:
                    quote = get_commodity_quote(key)
                    if quote:
                        st.metric(
                            label=info["label"],
                            value=f"{quote['price']:,.2f}",
                            delta=f"{quote['change']:+.2f} ({quote['change_pct']:+.1f}%)",
                        )
                    else:
                        st.metric(label=info["label"], value="N/A")

        st.divider()

        metals_period = st.selectbox(
            "History Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            key="metals_period",
        )

        metals_selected = st.multiselect(
            "Select Metals",
            options=list(metals.keys()),
            default=["gold", "copper"],
            format_func=lambda x: metals[x]["label"],
            key="metals_selected",
        )

        # Stock overlay controls
        metals_overlay_tickers, metals_normalize = _render_stock_overlay_controls("metals")

        if metals_selected or metals_overlay_tickers:
            metals_from_date = _from_date_str(_PERIOD_TO_DAYS.get(metals_period, 365))

            fig = go.Figure()
            for key in metals_selected:
                data = get_commodity_history(key, metals_period)
                if data:
                    dates = [d["period"] for d in reversed(data)]
                    values = [d["value"] for d in reversed(data)]
                    if metals_normalize:
                        values = _normalize_series(values)
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        mode="lines",
                        name=f"{metals[key]['label']} ({metals[key]['units']})",
                    ))

            _add_stock_traces(fig, metals_overlay_tickers, metals_from_date, metals_normalize)

            layout_kwargs = dict(
                title="Metals Prices" + (" (% Change)" if metals_normalize else ""),
                xaxis_title="Date",
                yaxis_title="% Change" if metals_normalize else "Price",
                height=500,
                hovermode="x unified",
            )
            if metals_overlay_tickers and not metals_normalize:
                layout_kwargs["yaxis2"] = dict(
                    title="Stock Price ($)", overlaying="y", side="right",
                )
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)

    # --- Agriculture sub-tab (yfinance) ---
    with commodity_tabs[2]:
        st.subheader("Agriculture")

        ag_commodities = {k: v for k, v in COMMODITY_SYMBOLS.items()
                         if k in ("corn", "wheat", "soybeans", "sugar", "coffee", "cotton", "lumber")}

        # Build a combined lookup for display: yfinance commodities + FRED series
        ag_all_options = {}
        for k, v in ag_commodities.items():
            ag_all_options[k] = {"label": v["label"], "units": v["units"], "source": "yfinance"}
        if fred.is_configured:
            for k, v in FRED_SERIES.items():
                ag_all_options[k] = {"label": v["label"], "units": v["units"], "source": "fred"}

        # Current prices — yfinance commodities + FRED series (two rows of 4)
        all_ag_items = list(ag_commodities.items())
        fred_items = list(FRED_SERIES.items()) if fred.is_configured else []

        # First row
        row1_cols = st.columns(4)
        for idx, (key, info) in enumerate(all_ag_items[:4]):
            with row1_cols[idx]:
                quote = get_commodity_quote(key)
                if quote:
                    st.metric(
                        label=info["label"],
                        value=f"{quote['price']:,.2f}",
                        delta=f"{quote['change']:+.2f} ({quote['change_pct']:+.1f}%)",
                    )
                else:
                    st.metric(label=info["label"], value="N/A")

        # Second row — remaining yfinance + FRED series
        remaining = all_ag_items[4:]
        row2_items = len(remaining) + len(fred_items)
        if row2_items > 0:
            row2_cols = st.columns(4)
            col_idx = 0
            for key, info in remaining:
                with row2_cols[col_idx]:
                    quote = get_commodity_quote(key)
                    if quote:
                        st.metric(
                            label=info["label"],
                            value=f"{quote['price']:,.2f}",
                            delta=f"{quote['change']:+.2f} ({quote['change_pct']:+.1f}%)",
                        )
                    else:
                        st.metric(label=info["label"], value="N/A")
                col_idx += 1

            for key, info in fred_items:
                with row2_cols[col_idx]:
                    fred_quote = fred.get_latest(key)
                    if fred_quote:
                        st.metric(
                            label=info["label"],
                            value=f"${fred_quote['price']:,.2f}",
                            delta=f"{fred_quote['change']:+.3f} ({fred_quote['change_pct']:+.1f}%)",
                            help=f"Monthly data from FRED ({info['series_id']}). Last: {fred_quote['period']}",
                        )
                    else:
                        st.metric(label=info["label"], value="N/A")
                col_idx += 1
                col_idx += 1

        st.divider()

        ag_period = st.selectbox(
            "History Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            key="ag_period",
        )

        ag_selected = st.multiselect(
            "Select Commodities",
            options=list(ag_all_options.keys()),
            default=["corn", "wheat", "soybeans"],
            format_func=lambda x: ag_all_options[x]["label"],
            key="ag_selected",
        )

        # Stock overlay controls
        ag_overlay_tickers, ag_normalize = _render_stock_overlay_controls("ag")

        if ag_selected or ag_overlay_tickers:
            ag_from_date = _from_date_str(_PERIOD_TO_DAYS.get(ag_period, 365))

            fig = go.Figure()
            for key in ag_selected:
                opt = ag_all_options[key]
                if opt["source"] == "yfinance":
                    data = get_commodity_history(key, ag_period)
                elif opt["source"] == "fred":
                    data = fred.get_series(key, ag_period)
                else:
                    data = []

                if data:
                    dates = [d["period"] for d in reversed(data)]
                    values = [d["value"] for d in reversed(data)]
                    if ag_normalize:
                        values = _normalize_series(values)
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        mode="lines",
                        name=f"{opt['label']} ({opt['units']})",
                    ))

            _add_stock_traces(fig, ag_overlay_tickers, ag_from_date, ag_normalize)

            layout_kwargs = dict(
                title="Agricultural Commodity Prices" + (" (% Change)" if ag_normalize else ""),
                xaxis_title="Date",
                yaxis_title="% Change" if ag_normalize else "Price",
                height=500,
                hovermode="x unified",
            )
            if ag_overlay_tickers and not ag_normalize:
                layout_kwargs["yaxis2"] = dict(
                    title="Stock Price ($)", overlaying="y", side="right",
                )
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# Tab 7 — Portfolio Tracker
# ------------------------------------------------------------------ #
with tab7:
    st.header("Portfolio Tracker")
    st.caption("Track positions, measure performance, and test investment hypotheses.")

    portfolio_data = load_portfolio()
    positions = portfolio_data.get("positions", [])

    # ---- Add Position Form ---- #
    with st.expander("➕ Add New Position", expanded=len(positions) == 0):
        with st.form("add_position_form", clear_on_submit=True):
            ap_cols = st.columns([1, 1, 1])
            with ap_cols[0]:
                ap_ticker = st.text_input("Ticker", placeholder="e.g. BAH").upper().strip()
            with ap_cols[1]:
                ap_date = st.date_input("Buy Date", value=datetime.now().date())
            with ap_cols[2]:
                ap_amount = st.number_input("Amount Invested ($)", min_value=1.0, value=1000.0, step=100.0)

            ap_cols2 = st.columns([1, 1])
            with ap_cols2[0]:
                existing_tags = get_all_tags()
                preset_tags = ["High FCF Growth", "Low P/E", "Catalyst", "Dividend Compounder",
                               "Turnaround Play", "Momentum", "Energy Exposure", "Defense/Intel", "Custom"]
                all_tag_options = sorted(set(preset_tags + existing_tags))
                ap_tag = st.selectbox("Thesis Tag", all_tag_options, index=0)
            with ap_cols2[1]:
                ap_custom_tag = st.text_input("Custom Tag (if Custom selected)", placeholder="My custom tag")

            ap_notes = st.text_input("Notes (optional)", placeholder="Why I'm buying this...")
            ap_submit = st.form_submit_button("Add Position", use_container_width=True)

        if ap_submit and ap_ticker:
            tag = ap_custom_tag.strip() if ap_tag == "Custom" and ap_custom_tag.strip() else ap_tag
            buy_date_str = ap_date.strftime("%Y-%m-%d")

            # Get cost basis: use historical price for past dates, live for today
            try:
                if ap_date == datetime.now().date():
                    quotes_df = tradier.get_quotes([ap_ticker])
                    if quotes_df is not None and not quotes_df.empty:
                        cost = float(quotes_df.iloc[0]["last"])
                    else:
                        st.error(f"Could not get quote for {ap_ticker}")
                        cost = None
                else:
                    import yfinance as yf
                    hist = yf.download(ap_ticker, start=buy_date_str,
                                       end=(ap_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                                       progress=False)
                    if hist.empty:
                        st.error(f"No historical price data for {ap_ticker} on {buy_date_str}")
                        cost = None
                    else:
                        if isinstance(hist.columns, pd.MultiIndex):
                            hist.columns = hist.columns.get_level_values(0)
                        cost = float(hist["Close"].iloc[0])

                if cost and cost > 0:
                    new_pos = add_position(ap_ticker, buy_date_str, ap_amount, cost, tag, ap_notes)
                    st.success(f"Added {new_pos['shares']:.4f} shares of {ap_ticker} at ${cost:.2f}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error adding position: {e}")

    # ---- Portfolio Summary ---- #
    if positions:
        # Fetch current prices
        unique_tickers = list(set(p["ticker"] for p in positions))
        try:
            quotes_df = tradier.get_quotes(unique_tickers)
            if quotes_df is not None and not quotes_df.empty:
                current_prices = dict(zip(quotes_df["symbol"], quotes_df["last"]))
            else:
                current_prices = {}
        except Exception:
            current_prices = {}

        summary = compute_portfolio_summary(positions, current_prices)

        st.subheader("Portfolio Summary")
        sum_cols = st.columns(5)
        with sum_cols[0]:
            st.metric("Positions", summary["num_positions"])
        with sum_cols[1]:
            st.metric("Total Invested", f"${summary['total_invested']:,.2f}")
        with sum_cols[2]:
            st.metric("Current Value", f"${summary['current_value']:,.2f}")
        with sum_cols[3]:
            color = "green" if summary["total_return_dollars"] >= 0 else "red"
            st.metric("Total Return ($)", f"${summary['total_return_dollars']:,.2f}",
                      delta=f"{summary['total_return_pct']:+.2f}%")
        with sum_cols[4]:
            st.metric("Total Return (%)", f"{summary['total_return_pct']:+.2f}%")

        st.divider()

        # ---- Positions Table ---- #
        st.subheader("Current Positions")
        rows = []
        for p in positions:
            price = current_prices.get(p["ticker"], p["cost_basis"])
            perf = compute_position_performance(p, price)
            rows.append({
                "Ticker": p["ticker"],
                "Tag": p["thesis_tag"],
                "Buy Date": p["buy_date"],
                "Shares": round(p["shares"], 4),
                "Cost Basis": f"${p['cost_basis']:.2f}",
                "Current": f"${price:.2f}",
                "Gain/Loss $": f"${perf['gain_loss_dollars']:+,.2f}",
                "Return %": f"{perf['gain_loss_pct']:+.2f}%",
                "Days Held": perf["holding_days"],
                "_id": p["id"],
            })

        display_df = pd.DataFrame(rows)
        st.dataframe(
            display_df.drop(columns=["_id"]),
            use_container_width=True,
            hide_index=True,
        )

        # Remove position
        remove_cols = st.columns([3, 1])
        with remove_cols[0]:
            remove_options = [f"{p['ticker']} — {p['buy_date']} — {p['thesis_tag']}" for p in positions]
            remove_selection = st.selectbox("Select position to remove", remove_options, key="remove_select")
        with remove_cols[1]:
            st.write("")  # spacer
            st.write("")
            if st.button("🗑️ Remove", key="remove_btn"):
                idx = remove_options.index(remove_selection)
                remove_position(positions[idx]["id"])
                st.success(f"Removed {positions[idx]['ticker']}")
                st.rerun()

        st.divider()

        # ---- Research Notes ---- #
        @st.fragment
        def _research_notes_fragment():
            st.subheader("📝 Research Notes")

            # Select which position to add notes to
            note_options = [f"{p['ticker']} — {p['buy_date']} ({p['thesis_tag']})" for p in positions]
            note_selection = st.selectbox("Select position", note_options, key="note_pos_select")
            note_idx = note_options.index(note_selection)
            note_pos = positions[note_idx]

            # Get existing notes (migrate from old string format if needed)
            existing_notes = note_pos.get("notes", "")
            if isinstance(existing_notes, str):
                if existing_notes.strip():
                    structured_notes = [{"text": existing_notes, "level": 0}]
                else:
                    structured_notes = []
            else:
                structured_notes = existing_notes if existing_notes else []

            # Display existing notes with per-line delete buttons
            if structured_notes:
                _del_counters = {0: 0, 1: 0, 2: 0}
                _display_lines = []
                for idx, item in enumerate(structured_notes):
                    lvl = item.get("level", 0)
                    txt = item.get("text", "")
                    if not txt.strip():
                        continue
                    for l in range(lvl + 1, 3):
                        _del_counters[l] = 0
                    indent = "&nbsp;" * (8 * lvl)
                    if lvl == 0:
                        marker = "•"
                    elif lvl == 1:
                        c = _del_counters[1] % 26
                        marker = f"{chr(97 + c)})"
                        _del_counters[1] += 1
                    else:
                        c = _del_counters[2] % 26
                        marker = f"{chr(97 + c)}{chr(97 + c)})"
                        _del_counters[2] += 1
                    _display_lines.append((idx, f"{indent}{marker} {txt}"))

                st.markdown(
                    """<div style="background-color: #1e1e2e; border: 1px solid #444;
                    border-radius: 8px; padding: 16px; margin-bottom: 8px;
                    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 14px;
                    line-height: 1.8; color: #e0e0e0;">""",
                    unsafe_allow_html=True,
                )
                # Use session_state flag to handle deletion without shifting keys
                _del_key = st.session_state.get("_note_to_delete", None)
                if _del_key is not None:
                    st.session_state.pop("_note_to_delete", None)
                    if 0 <= _del_key < len(structured_notes):
                        del structured_notes[_del_key]
                        update_position_notes(note_pos["id"], structured_notes)
                        st.rerun()

                for note_idx_display, (orig_idx, line_html) in enumerate(_display_lines):
                    dcol1, dcol2 = st.columns([20, 1])
                    with dcol1:
                        st.markdown(line_html, unsafe_allow_html=True)
                    with dcol2:
                        if st.button("✕", key=f"delnote_{note_pos['id']}_{note_idx_display}",
                                     help="Delete this note"):
                            st.session_state["_note_to_delete"] = orig_idx
                            st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.caption("Start a line with **~** to indent (a, b, c) or **~~** to double-indent (aa, bb, cc). "
                       "Each line becomes a separate note.")

            with st.form(key="add_note_form", clear_on_submit=True):
                new_note_text = st.text_area(
                    "Note",
                    height=150,
                    placeholder="Type your thoughts here...\n~supporting detail\n~~sub-detail\n\nEach line saves as a separate note.",
                    key="note_text_input",
                    label_visibility="collapsed",
                )
                save_note = st.form_submit_button("💾 Save Notes", use_container_width=True)

            if save_note and new_note_text.strip():
                new_entries = []
                for line in new_note_text.split("\n"):
                    if not line.strip():
                        continue
                    text = line
                    if text.startswith("~~"):
                        level = 2
                        text = text[2:].lstrip()
                    elif text.startswith("~"):
                        level = 1
                        text = text[1:].lstrip()
                    else:
                        level = 0
                    if text.strip():
                        new_entries.append({"text": text.strip(), "level": level})
                if new_entries:
                    updated_notes = structured_notes + new_entries
                    update_position_notes(note_pos["id"], updated_notes)
                    st.rerun()

            if structured_notes:
                if st.button("🗑️ Clear all notes", key="clear_notes_btn"):
                    update_position_notes(note_pos["id"], [])
                    st.rerun()

        _research_notes_fragment()

        st.divider()

        # ---- Charts ---- #
        chart_tab1, chart_tab2, chart_tab3 = st.tabs([
            "📈 Individual Performance", "💼 Portfolio vs SPY", "🏷️ Tag Analysis"
        ])

        with chart_tab1:
            st.subheader("Individual Position Performance vs SPY")
            pos_options = [f"{p['ticker']} — {p['buy_date']} ({p['thesis_tag']})" for p in positions]
            selected_pos = st.selectbox("Select position", pos_options, key="chart_pos_select")
            pos_idx = pos_options.index(selected_pos)
            p = positions[pos_idx]

            with st.spinner(f"Loading history for {p['ticker']}..."):
                pos_hist = get_position_history(p["ticker"], p["buy_date"], p["shares"], p["cost_basis"])
                spy_hist = get_spy_benchmark(p["buy_date"], p["amount_invested"])

            if not pos_hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pos_hist["date"], y=pos_hist["return_pct"],
                    name=p["ticker"], line=dict(color="#00cc96", width=2),
                ))
                if not spy_hist.empty:
                    fig.add_trace(go.Scatter(
                        x=spy_hist["date"], y=spy_hist["return_pct"],
                        name="SPY", line=dict(color="#636efa", width=2, dash="dash"),
                    ))
                fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                fig.update_layout(
                    template="plotly_dark",
                    title=f"{p['ticker']} vs SPY — Return Since {p['buy_date']}",
                    yaxis_title="Return (%)",
                    xaxis_title="Date",
                    height=450,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Dollar value chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=pos_hist["date"], y=pos_hist["position_value"],
                    name=p["ticker"], line=dict(color="#00cc96", width=2),
                    fill="tozeroy", fillcolor="rgba(0,204,150,0.1)",
                ))
                if not spy_hist.empty:
                    fig2.add_trace(go.Scatter(
                        x=spy_hist["date"], y=spy_hist["benchmark_value"],
                        name="SPY Equivalent", line=dict(color="#636efa", width=2, dash="dash"),
                    ))
                fig2.add_hline(y=p["amount_invested"], line_dash="dot", line_color="yellow",
                               opacity=0.5, annotation_text="Invested")
                fig2.update_layout(
                    template="plotly_dark",
                    title=f"${p['amount_invested']:,.0f} Invested in {p['ticker']} vs SPY",
                    yaxis_title="Value ($)",
                    xaxis_title="Date",
                    height=450,
                    hovermode="x unified",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No historical data available yet (position may be too new).")

        with chart_tab2:
            st.subheader("Portfolio Performance vs SPY")
            if len(positions) > 0:
                # Find earliest buy date
                earliest = min(p["buy_date"] for p in positions)
                total_invested = sum(p["amount_invested"] for p in positions)

                with st.spinner("Loading portfolio history..."):
                    spy_bench = get_spy_benchmark(earliest, total_invested)

                    # Build portfolio value over time by summing position histories
                    all_histories = {}
                    for p in positions:
                        hist = get_position_history(p["ticker"], p["buy_date"], p["shares"], p["cost_basis"])
                        if not hist.empty:
                            all_histories[p["id"]] = hist

                if all_histories:
                    # Merge all position histories on date
                    combined = None
                    for pid, hist in all_histories.items():
                        h = hist[["date", "position_value"]].rename(columns={"position_value": pid})
                        if combined is None:
                            combined = h
                        else:
                            combined = pd.merge(combined, h, on="date", how="outer")

                    combined = combined.sort_values("date")
                    combined = combined.ffill().fillna(0)
                    value_cols = [c for c in combined.columns if c != "date"]
                    combined["portfolio_value"] = combined[value_cols].sum(axis=1)
                    combined["return_pct"] = ((combined["portfolio_value"] - total_invested) / total_invested * 100).round(2)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=combined["date"], y=combined["portfolio_value"],
                        name="Portfolio", line=dict(color="#00cc96", width=2),
                        fill="tozeroy", fillcolor="rgba(0,204,150,0.1)",
                    ))
                    if not spy_bench.empty:
                        fig.add_trace(go.Scatter(
                            x=spy_bench["date"], y=spy_bench["benchmark_value"],
                            name="SPY Equivalent", line=dict(color="#636efa", width=2, dash="dash"),
                        ))
                    fig.add_hline(y=total_invested, line_dash="dot", line_color="yellow",
                                  opacity=0.5, annotation_text=f"Invested (${total_invested:,.0f})")
                    fig.update_layout(
                        template="plotly_dark",
                        title="Portfolio Value vs SPY Benchmark",
                        yaxis_title="Value ($)",
                        xaxis_title="Date",
                        height=500,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough historical data to chart portfolio performance.")

        with chart_tab3:
            st.subheader("Performance by Thesis Tag")
            tag_perf = compute_tag_performance(positions, current_prices)

            if tag_perf:
                tag_df = pd.DataFrame(tag_perf)

                # Bar chart
                colors = ["#00cc96" if r >= 0 else "#ef553b" for r in tag_df["return_pct"]]
                fig = go.Figure(go.Bar(
                    x=tag_df["tag"],
                    y=tag_df["return_pct"],
                    marker_color=colors,
                    text=[f"{r:+.2f}%" for r in tag_df["return_pct"]],
                    textposition="auto",
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                fig.update_layout(
                    template="plotly_dark",
                    title="Return (%) by Thesis Tag",
                    yaxis_title="Return (%)",
                    xaxis_title="Thesis Tag",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                st.dataframe(
                    tag_df.rename(columns={
                        "tag": "Thesis Tag", "count": "Positions",
                        "total_invested": "Invested ($)", "current_value": "Current ($)",
                        "return_pct": "Return (%)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Add positions to see tag analysis.")
    else:
        st.info("No positions yet. Use the form above to add your first position.")

    # ---- Export / Import ---- #
    st.divider()
    st.subheader("💾 Export / Import Portfolio")
    st.caption("Your portfolio is saved in this browser. Use export to back it up or move it to another device.")

    exp_cols = st.columns([1, 1])
    with exp_cols[0]:
        _export_data = json.dumps(
            st.session_state.get("portfolio_data", {"positions": []}), indent=2
        )
        st.download_button(
            "⬇️ Export Portfolio (JSON)",
            data=_export_data,
            file_name="prospector_portfolio.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp_cols[1]:
        _uploaded = st.file_uploader(
            "⬆️ Import Portfolio (JSON)", type=["json"], key="portfolio_import"
        )
        if _uploaded is not None:
            try:
                _imported = json.loads(_uploaded.read().decode("utf-8"))
                if "positions" in _imported and isinstance(_imported["positions"], list):
                    st.session_state["portfolio_data"] = _imported
                    st.session_state["portfolio_dirty"] = True
                    st.success(f"Imported {len(_imported['positions'])} positions!")
                    st.rerun()
                else:
                    st.error("Invalid portfolio file — expected a JSON object with a 'positions' array.")
            except (json.JSONDecodeError, Exception) as e:
                st.error(f"Could not read file: {e}")

    # Push any changes to browser localStorage (cloud only)
    if _IS_CLOUD:
        _push_portfolio_to_localstorage()

# ------------------------------------------------------------------ #
# Tab 8 — Settings
# ------------------------------------------------------------------ #
with tab8:
    st.header("Settings")

    # Cache TTLs
    st.subheader("Cache TTL Settings")
    ttl_cfg = settings.get("cache_ttl", {})
    st.write("Cache time-to-live values in seconds.")

    ttl_cols = st.columns(3)
    new_ttls = {}
    ttl_items = list(ttl_cfg.items())
    for i, (key, val) in enumerate(ttl_items):
        col_idx = i % 3
        with ttl_cols[col_idx]:
            new_ttls[key] = st.number_input(
                key.replace("_", " ").title(),
                min_value=60, max_value=604800, value=int(val),
                step=300, key=f"ttl_{key}",
            )

    if st.button("Save Cache TTLs"):
        settings["cache_ttl"] = new_ttls
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        st.success("Cache TTL settings saved.")

    st.divider()

    # API Key Status
    st.subheader("API Key Status")
    api_col1, api_col2, api_col3 = st.columns(3)
    with api_col1:
        tradier_set = bool(os.environ.get("TRADIER_TOKEN"))
        st.write(f"**TRADIER_TOKEN:** {'Set' if tradier_set else 'Not set'}")
    with api_col2:
        fmp_set = bool(os.environ.get("FMP_API_KEY"))
        st.write(f"**FMP_API_KEY:** {'Set' if fmp_set else 'Not set'}")
    with api_col3:
        eia_set = bool(os.environ.get("EIA_API_KEY"))
        st.write(f"**EIA_API_KEY:** {'Set' if eia_set else 'Not set'}")

    st.divider()

    # Cache management
    st.subheader("Cache Management")
    cache_cols = st.columns(5)

    with cache_cols[0]:
        if st.button("Clear FMP Cache"):
            count = fmp.clear_cache()
            st.success(f"Cleared {count} FMP cache files.")

    with cache_cols[1]:
        quotes_cache = PROJECT_ROOT / "data" / "cache" / "quotes"
        if st.button("Clear Quotes Cache"):
            count = 0
            for f in quotes_cache.glob("*.json"):
                f.unlink()
                count += 1
            st.success(f"Cleared {count} quote cache files.")

    with cache_cols[2]:
        if st.button("Clear EIA Cache"):
            count = eia.clear_cache()
            st.success(f"Cleared {count} EIA cache files.")

    with cache_cols[3]:
        if st.button("Clear Commodity Cache"):
            count = clear_commodity_cache()
            st.success(f"Cleared {count} commodity cache files.")

    with cache_cols[4]:
        if st.button("Clear All Caches"):
            fmp_count = fmp.clear_cache()
            eia_count = eia.clear_cache()
            comm_count = clear_commodity_cache()
            q_count = 0
            for f in (PROJECT_ROOT / "data" / "cache" / "quotes").glob("*.json"):
                f.unlink()
                q_count += 1
            st.success(f"Cleared {fmp_count + eia_count + comm_count + q_count} total cache files.")

    st.divider()

    # Industry info
    st.subheader("Loaded Industries")
    for name in list_universes():
        uni = load_universe(name)
        st.write(f"**{name}**: {len(uni)} tickers")
