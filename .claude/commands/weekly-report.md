---
description: Generate weekly Plotly HTML dashboard + markdown summary
---

1. Determine ISO week: `date -u +%G-W%V` (e.g. `2026-W20`).
2. Run `python report.py --week $(date -u +%G-W%V) --out reports/$(date -u +%G-W%V).html` (when P5 lands).
3. The HTML should contain tabs: equity curve, trade overlay (candles + RSI + volume + funding subpanels with entry triggers highlighted), PnL distribution, dashboard.
4. Output a markdown summary:
   - Week tag and trade count
   - Net P&L (USDT and %)
   - Winrate, profit factor, max DD this week
   - Best trade and worst trade (entry/exit times, R-multiple, reason notes)
   - Top 3 observations (e.g. "Tuesday's FOMC reaction caused 2 stops in 30 min — consider news blackout filter")
   - Suggested tweaks for next week (small, one at a time)
   - Path to the HTML file
5. End with `memory_write(type="weekly_review", tags="snapback-btc,review,<week>")` saving the markdown summary so trends are queryable.
