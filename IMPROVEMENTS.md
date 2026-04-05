# GoldRatio — Improvement Roadmap

After a full audit of the codebase, here are 20 concrete improvements organized by area. Each is independent — we can tackle them in any order.

---

## Screener Dashboard

**1. Expose More Valuation Filters**
The screener only lets you filter by Max P/E. The backend (`apply_filters()`) already supports more, but the UI doesn't expose them. Add filters for: Max P/B, Max EV/EBITDA, Min ROE, Min FCF Yield, and Max Debt/Equity.

**2. Show Opportunity Flags Column**
The scoring engine generates opportunity flags (e.g., "P/E at 10yr low") but they're never displayed in the results table. Surface them so you can see *why* a stock scored well at a glance.

**3. Add Industry Summary Statistics**
After a scan, show a summary row per segment: average P/E, median ROE, average D/E. Helps you spot which industries within the scan are cheap/expensive relative to each other.

**4. Persist Scan Results**
Scan results vanish on page refresh. Auto-save each scan to a timestamped CSV so you can reference past scans and track how opportunities evolve over time.

---

## Ticker Deep Dive

**5. Add Dividend Metrics**
No dividend data is currently shown. Add: current yield, 5-year dividend CAGR, payout ratio (both earnings-based and FCF-based), and flag dividend cut risk when payout ratio exceeds 80%.

**6. Share Dilution / Buyback Tracker**
Share count changes are computed for the DCF but never displayed as a standalone metric. Add a visual showing historical shares outstanding trend, buyback yield (repurchases / market cap), and the impact on per-share metrics.

**7. Quality Metrics Panel (Piotroski / DuPont)**
Add asset turnover, receivables turnover, inventory turnover, working capital trends, and a Piotroski F-Score. These help distinguish cheap-and-improving from cheap-and-deteriorating.

**8. Capital Allocation Dashboard**
Show how management spends cash: capex vs dividends vs buybacks vs debt paydown as a stacked area chart over time. Reveals whether management is shareholder-aligned or empire-building.

**9. Analyst Accuracy Tracker**
Compare past analyst estimates to actual results. If analysts have consistently missed by 20%+, flag that forward guidance is unreliable. The data is already being fetched — just not benchmarked.

---

## Ratio Comparison

**10. Peer Group Percentile Ranking**
Currently shows each ticker's percentile vs its own history. Add a "Rank vs Peers" column showing where each stock ranks within the comparison group (e.g., "2nd cheapest of 6 on P/E").

**11. Scatter Plot Matrix**
Let users plot any two ratios against each other (P/E vs ROE, D/E vs ROIC). This reveals which stocks are mispriced relative to their quality — the core of value investing.

**12. Statistical Summary Row**
Add a "Peer Group" row showing mean, median, min, max for each ratio. Makes it immediately clear which company is the outlier.

---

## Balance Sheet Health

**13. Expand Balance Sheet Metrics**
Currently only shows D/E, current ratio, quick ratio, and interest coverage. Add: net debt / EBITDA (refinancing risk), cash / total debt (liquidity buffer), tangible book value per share, and working capital as % of revenue.

**14. Balance Sheet Trend Arrows**
Show if each metric is improving or deteriorating YoY. A D/E of 0.8 means something very different if it was 0.5 last year (worsening) vs 1.2 last year (improving).

**15. Solvency Risk Flag**
Auto-flag companies where current ratio is falling AND debt is rising simultaneously. Combine D/E + current ratio + interest coverage into a simple traffic-light solvency score.

---

## DCF Valuation

**16. WACC / Terminal Growth Sensitivity Grid**
Add a simple grid showing intrinsic value at WACC ±1% and terminal growth ±0.5%. This is the standard way to understand margin of safety — currently missing.

**17. Terminal Value Dominance Warning**
When terminal value exceeds 70% of total DCF value, show a warning. High terminal value % means most of the valuation depends on assumptions beyond 10 years — risky.

---

## Cross-Cutting / UX

**18. Screener → Deep Dive Flow Improvements**
When clicking a screener result to jump to Deep Dive, auto-run the analysis instead of requiring a second click. Reduce friction in the research workflow.

**19. Insider Trading Signals**
Insider trade data is already being fetched but not scored. Add a simple net buying/selling indicator: "Insiders net bought $2.3M in last 90 days" as a signal alongside the fundamentals.

**20. Watchlist → Portfolio Promotion**
Add a "Move to Portfolio" button on watchlist items that pre-fills the Portfolio Tracker add form with the ticker. Streamlines the workflow when you decide to buy something you've been watching.
