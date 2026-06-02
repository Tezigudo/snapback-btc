---
description: Re-run backtest with current params, compare to live performance
---

**Time display rule:** the `last-90d` window math is in UTC (matches bot storage). Any timestamps shown to the user in the comparison output (e.g. "live window: 2026-03-04 → 2026-06-02") are converted to GMT+7 ICT and suffixed. Footer the table with `_Times shown in GMT+7 (ICT). Bot stores UTC._`

1. `python backtest.py --params config/params.yaml --period last-90d` (when P1 lands).
2. Capture metrics: trades, winrate, profit factor, Sharpe, max DD, avg R, total return.
3. Pull last-90d **live** metrics from `data/state.db`.
4. Output a side-by-side table:
   ```
                  | Backtest | Live   | Δ
   trades         |   x      |  x     |  x
   winrate        |   x%     |  x%    |  x%
   profit_factor  |   x.x    |  x.x   |  ±x.x
   ...
   ```
5. **Divergence flags** (in this priority):
   - Live winrate > 10pp below backtest → likely overfit; suggest re-run walk-forward
   - Live profit_factor < 1.0 while backtest > 1.5 → execution slippage worse than modeled
   - Live trade count far below backtest → signal filter too tight live, or data feed issue
6. Suggest one specific param change OR "no change — within tolerance".

**Never auto-apply param changes.** Output the suggested diff for `config/params.yaml`, let the user decide.
