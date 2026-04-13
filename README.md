# GoldRatio

A Streamlit-based stock screener and fundamental analysis tool built for value investing research. It pulls financial data from multiple sources, scores stocks on forensic quality metrics, and runs multi-flavor DCF valuations -- all in one interface.

This is a personal research tool, not a product. It exists to speed up the boring parts of bottom-up equity analysis so more time can go toward reading 10-Ks and thinking.

## What It Does

### Screener

- Scans configurable stock universes (143 industries, refreshed from FMP with delisted-ticker filtering)
- Piotroski F-Score, Altman Z-Score, and Beneish M-Score as filterable and sortable columns
- Standard valuation filters (P/E, EV/EBITDA, ROE, D/E)
- Opportunity flags surfaced per ticker

### DCF Engine

- **Free Cash Flow DCF** with reinvestment-aware / Damodaran NOPAT model
- **Revenue DCF** and **Earnings DCF** as alternative valuation approaches
- Fade / convergence model (growth and ROIC fade to terminal rate / WACC over time)
- Mid-year discounting convention across all DCF flavors
- Reverse DCF (what growth rate is the market pricing in?)
- Monte Carlo simulation for probability-weighted valuation ranges
- Sensitivity tables for WACC vs. growth assumptions
- Owner Earnings cross-check (Buffett's formula) with maintenance vs. growth capex split
- Dividend metrics panel (yield, payout ratios, CAGR, cut risk detection)

### Ticker Deep Dive

- Full financial statements viewer (income statement, balance sheet, cash flow -- quarterly and annual)
- SEC EDGAR capital actions overlay -- verifies buybacks, debt paydown, and debt issuance against actual XBRL filings
- SBC absorption metric (detects the share-count treadmill: how much of the buyback budget is wasted offsetting stock comp)
- Cash conversion ratio and ROIC-WACC economic profit spread
- Analyst estimate accuracy tracking
- Insider trading activity
- Revenue segmentation (product and geographic)

### Commodity and Macro Data

- EIA petroleum data (inventories, production, spot prices)
- FRED economic indicators (rates, inflation, employment, credit spreads)
- Commodity quotes and historical charts

### Portfolio and Watchlist

- Portfolio tracker with position-level and aggregate performance
- Watchlist for tracking tickers under research
- Research notes per position

## Project Structure

```
streamlit_app.py            # Main Streamlit application
src/
  fmp_client.py             # Financial Modeling Prep API client
  tradier_client.py         # Tradier market data client
  eia_client.py             # EIA energy data client
  fred_client.py            # FRED economic data client
  commodity_client.py       # Commodity quotes and history
  edgar_client.py           # SEC EDGAR XBRL client (capital actions, filings)
  ratio_analyzer.py         # DCF models, scoring, and fundamental analysis
  screener.py               # Universe scanning and filtering
  quality_scores.py         # Piotroski, Altman, Beneish implementations
  price_validator.py        # Cross-source price validation
  universe_loader.py        # Stock universe CSV management
  portfolio.py              # Portfolio tracker persistence
  watchlist.py              # Watchlist persistence
  transcript_parser.py      # Earnings call transcript parsing
  transcript_summarizer.py  # AI-powered transcript summarization
scripts/
  rebuild_universes.py      # Refresh all universe CSVs from FMP screener
config/
  universes/                # Industry-specific ticker lists (143 CSVs)
  scoring/                  # Scoring thresholds
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Wartime-Screener/GoldRatio.git
cd GoldRatio
```

### 2. Install dependencies

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root:

```
FMP_API_KEY=your_key_here
FRED_API_KEY=your_key_here
EIA_API_KEY=your_key_here
TRADIER_TOKEN=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

**Required:**

| Key | What it's for | Where to get it |
|-----|--------------|-----------------|
| `FMP_API_KEY` | Financial data, ratios, statements, and metrics | [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/) |
| `FRED_API_KEY` | Federal Reserve economic data (rates, inflation, etc.) | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `EIA_API_KEY` | Energy Information Administration data (oil, gas, inventories) | [eia.gov/opendata](https://www.eia.gov/opendata/register.php) |

**Optional:**

| Key | What it's for | Where to get it |
|-----|--------------|-----------------|
| `TRADIER_TOKEN` | Real-time market quotes and options data | [tradier.com](https://developer.tradier.com/) |
| `ANTHROPIC_API_KEY` | AI-powered earnings transcript summarization | [console.anthropic.com](https://console.anthropic.com/) |

The app works without the optional keys -- those features will just be unavailable.

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

## Disclaimer

This is a personal research and learning tool. It is not financial advice. The valuations, scores, and analysis produced by this tool are mechanical outputs based on reported financial data and user-supplied assumptions. They can be wrong. Do your own work before making any investment decisions.
