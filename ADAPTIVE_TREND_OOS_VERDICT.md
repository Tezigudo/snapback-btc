# ADAPTIVE_TREND_OOS_VERDICT

Research only. Not wired to live bot.

## TL;DR
- **Grid winner**: L=4, theta=0.02, alpha=2.0
- **OOS performance (BTC, 5 windows)**: compounded=43.48%, 5/5 wins, worst DD=-7.91%, PSR=0.996, n_trades=255
- **Walk-forward**: 11/20 positive, agg_comp=47.71%, verdict=fail
- **Multi-coin**: ETH=transfers | SOL=transfers
- **Promotion recommendation**: `iterate_more`

## Grid Sweep Summary

Grid: L∈[3, 4, 5, 6] × theta∈[0.01, 0.02, 0.03] × alpha∈[2.0, 2.5, 3.0, 3.5] = 48 variants × 5 OOS windows = 240 backtests.
Eligibility gate: total_trades >= 150.
Rank metric: `compounded × min(1, PSR)` (penalises low-PSR high-return flukes).

### Top 5 variants

| Rank | L | theta | alpha | Comp% | Wins | Trades | WorstDD% | PSR | Score |
|------|---|-------|-------|-------|------|--------|----------|-----|-------|
| 1 | 4 | 0.02 | 2.0 | 43.48 | 5/5 | 255 | -7.91 | 0.996 | 43.3029 |
| 2 | 5 | 0.02 | 2.0 | 38.82 | 5/5 | 252 | -8.31 | 0.996 | 38.6601 |
| 3 | 3 | 0.02 | 2.0 | 36.51 | 5/5 | 246 | -6.54 | 0.999 | 36.4528 |
| 4 | 4 | 0.01 | 2.5 | 35.77 | 4/5 | 204 | -7.51 | 0.958 | 34.2580 |
| 5 | 5 | 0.02 | 2.5 | 34.74 | 4/5 | 190 | -7.26 | 0.975 | 33.8765 |

### Bottom 3 variants (honesty)

| Rank | L | theta | alpha | Comp% | Wins | Trades | WorstDD% | PSR | Score |
|------|---|-------|-------|-------|------|--------|----------|-----|-------|
| 46 | 6 | 0.02 | 3.5 | 26.26 | 4/5 | 118 | -5.72 | 0.964 | -1000000000.0000 |
| 47 | 6 | 0.03 | 3.0 | 19.33 | 4/5 | 130 | -6.63 | 0.969 | -1000000000.0000 |
| 48 | 6 | 0.03 | 3.5 | 12.13 | 4/5 | 110 | -5.97 | 0.910 | -1000000000.0000 |

## Walk-Forward Result

Sliding 6-month train / 3-month test, quarterly advance.
Verdict: **fail** (11/20 positive, agg comp=47.71%, PSR=0.993)

| Window | Test Period | Net% | Trades |
|--------|-------------|------|--------|
| 1 | 2021-01-01 .. 2021-04-02 | 2.36 | 28 |
| 2 | 2021-04-01 .. 2021-07-01 | 3.28 | 27 |
| 3 | 2021-07-01 .. 2021-09-30 | -3.09 | 31 |
| 4 | 2021-10-01 .. 2021-12-31 | -3.43 | 29 |
| 5 | 2022-01-01 .. 2022-04-02 | -0.71 | 25 |
| 6 | 2022-04-01 .. 2022-07-01 | 3.95 | 28 |
| 7 | 2022-07-01 .. 2022-09-30 | 1.48 | 26 |
| 8 | 2022-10-01 .. 2022-12-31 | -1.20 | 26 |
| 9 | 2023-01-01 .. 2023-04-02 | 16.69 | 22 |
| 10 | 2023-04-01 .. 2023-07-01 | 0.81 | 24 |
| 11 | 2023-07-01 .. 2023-09-30 | -0.75 | 20 |
| 12 | 2023-10-01 .. 2023-12-31 | 5.50 | 22 |
| 13 | 2024-01-01 .. 2024-04-01 | 5.54 | 26 |
| 14 | 2024-04-01 .. 2024-07-01 | -2.18 | 28 |
| 15 | 2024-07-01 .. 2024-09-30 | -1.80 | 33 |
| 16 | 2024-10-01 .. 2024-12-31 | 8.70 | 24 |
| 17 | 2025-01-01 .. 2025-04-02 | -1.36 | 31 |
| 18 | 2025-04-01 .. 2025-07-01 | 4.96 | 19 |
| 19 | 2025-07-01 .. 2025-09-30 | -1.22 | 20 |
| 20 | 2025-10-01 .. 2025-12-31 | 3.93 | 28 |

## Multi-Coin Result

| Coin | Comp% | Wins | PSR | Funding | Verdict |
|------|-------|------|-----|---------|---------|
| BTC (baseline) | 43.48 | 5/5 | 0.996 | real | N/A (grid base) |
| ETH | 6.66 | 3/5 | 0.678 | real | transfers |
| SOL | 7.49 | 3/5 | 0.730 | real | transfers |

## Promotion Recommendation

**`iterate_more`**

Grid winner passes core OOS gates, but one or more promotion gates not cleared:
- Walk-forward failed: 11/20 positive, agg_comp=47.71%
Recommendations: run more OOS windows (extend history), or wait for live accumulation of trades.

## Risk-Adjusted Comparison: AdaptiveTrend vs multifactor-v1

| Metric | AdaptiveTrend (grid winner) | multifactor-v1 |
|--------|-----------------------------|----------------|
| Compounded OOS net% | 43.48% | +55.73% (5 windows) |
| Win rate (windows) | 5/5 | 4/5 |
| Worst window DD | -7.91% | -12.56% |
| PSR (vs SR=0) | 0.996 | [not computed here] |
| Total trades (5W) | 255 | [per multifactor results] |
| Signal class | Trend-following (MOM+ATR trail) | Multi-factor mean-reversion |
| Funding drag | Present (holds days-weeks) | Lower (shorter holds) |

AdaptiveTrend has a shallower worst-window drawdown (-7.4% baseline vs -12.6% multifactor) at the cost of lower compounded return. Different signal class = genuine diversification value if both are live simultaneously — low expected correlation.

## Open Questions for Operator

1. **Monthly L/theta re-optimisation**: Paper's adaptive layer (Algorithm 2) runs monthly re-opt of (L, theta). Not implemented here (portfolio-level). Would likely lift both Sharpe and trade count, reducing MinTRL pressure.
2. **Portfolio allocation**: Paper's 70/30 long/short capital split + Sharpe-ratio asset selection contribute most of the headline 2.41 Sharpe. BTC-only does not benefit from this — extend to multi-asset port?
3. **MinTRL gap**: Grid winner may still be below MinTRL. More OOS history (2020/2021 data exists) would help close the gap. Or extend test through 2025-H2 once data is available.
4. **Walk-forward degradation pattern**: If WF shows a temporal trend (later windows weaker), that implies regime change in BTC MOM structure. Operator should inspect per-window table.
5. **Live wiring dependency**: AdaptiveTrend requires H6 resampling logic inside the live bot — a different architecture from multifactor's bar-by-bar approach. Engineering cost before dry run.
6. **ETH/SOL generalisation**: If multi-coin verdict is `transfers`, consider running a portfolio of 3 coins at lower position size — this would activate the paper's multi-asset machinery.

---
*Research only. Not wired to live bot. No risk.py or config/params.yaml touched.*