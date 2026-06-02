---
description: Generate weekly Plotly HTML dashboard + markdown summary
---

**Time display rule:** week boundary and all trade timestamps shown to the user are GMT+7 ICT. The bot stores UTC; convert on display. The ISO week label still uses UTC (`date -u +%G-W%V`) so it matches log filenames and CI artifacts — call this out in the report header as `Week 2026-W20 (UTC reference, times shown ICT)`.

1. Determine ISO week (UTC): `date -u +%G-W%V` (e.g. `2026-W20`).
2. Run `python report.py --week $(date -u +%G-W%V) --tz Asia/Bangkok --out reports/$(date -u +%G-W%V).html` (when P5 lands; if `--tz` not yet supported, render in UTC and add a "displayed times are UTC; ICT = UTC+7" note).
3. The HTML should contain tabs: equity curve, trade overlay (candles + RSI + volume + funding subpanels with entry triggers highlighted), PnL distribution, dashboard. All x-axes labeled GMT+7 ICT (or UTC if `--tz` unsupported).
4. Output a markdown summary:
   - Week tag and trade count (header: `Week YYYY-W## (UTC reference, times shown ICT)`)
   - Net P&L (USDT and %)
   - Winrate, profit factor, max DD this week
   - Best trade and worst trade (entry/exit times as GMT+7 ICT, R-multiple, reason notes)
   - Top 3 observations (e.g. "Tuesday's FOMC reaction caused 2 stops in 30 min — consider news blackout filter")
   - Suggested tweaks for next week (small, one at a time)
   - Path to the HTML file
   - Footer: `_Times shown in GMT+7 (ICT). Bot stores UTC._`
5. End with `memory_write(type="weekly_review", tags="snapback-btc,review,<week>")` saving the markdown summary so trends are queryable.
