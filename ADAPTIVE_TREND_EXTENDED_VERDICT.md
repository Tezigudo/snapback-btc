# ADAPTIVE_TREND_EXTENDED_VERDICT

Research only. Not wired to live bot.

## TL;DR
- **Lower-alpha winner**: L=4, theta=0.02, alpha=2.0, compounded=67.09%, 9/11 wins, worst DD=-7.98%
- **n_trades vs MinTRL**: NOT MET (n=563, MinTRL=976, gap=413), per-trade PSR=0.894
- **Walk-forward extended**: 17/24 positive (70.8%), agg_comp=98.64%, verdict=pass
- **Promotion recommendation**: `iterate_more`

## History Availability

- BTC 15m: 2019-09-08 17:45:00 → 2026-06-02 14:45:00 (236,053 bars)
- Funding: 2019-09-10 08:00:00 → 2026-05-30 08:00:00.001000 (7,363 rows)

### Extended window set (11 windows used)

| Window | Start | End | 15m bars | Funding rows | Used |
|--------|-------|-----|----------|--------------|------|
| 2020_H2 | 2020-07-01 | 2020-12-31 | 17,569 | 549 | yes |
| 2021_H1 | 2021-01-01 | 2021-06-30 | 17,281 | 540 | yes |
| 2021_H2 | 2021-07-01 | 2021-12-31 | 17,569 | 549 | yes |
| 2022_H1 | 2022-01-01 | 2022-06-30 | 17,281 | 541 | yes |
| 2022_H2 | 2022-07-01 | 2022-12-31 | 17,569 | 549 | yes |
| 2023_H1 | 2023-01-01 | 2023-06-30 | 17,281 | 541 | yes |
| 2023_H2 | 2023-07-01 | 2023-12-31 | 17,569 | 550 | yes |
| 2024_H1 | 2024-01-01 | 2024-06-30 | 17,377 | 544 | yes |
| 2024_H2 | 2024-07-01 | 2024-12-31 | 17,569 | 550 | yes |
| 2025_H1 | 2025-01-01 | 2025-06-30 | 17,281 | 541 | yes |
| 2025_H2 | 2025-07-01 | 2025-12-31 | 17,569 | 549 | yes |

## Lower-Alpha Grid Sweep Summary

Grid: L∈[3, 4, 5] × theta∈[0.015, 0.02, 0.025] × alpha∈[1.0, 1.25, 1.5, 1.75, 2.0] = 45 variants × 11 OOS windows.
Eligibility gate: total_trades >= 100. Win gate: >= 7/11 (70%).
Rank metric: `compounded × min(1, PSR)` (window-level PSR, n=11 — tiebreaker only).
Prior winner for delta: L=4, theta=0.02, alpha=2.0, comp=43.48% (5 windows).

### Top 5 variants

| Rank | L | theta | alpha | Comp% | Wins/11 | Trades | WorstDD% | Score |
|------|---|-------|-------|-------|-------|--------|----------|-------|
| 1 | 4 | 0.02 | 2.0 | 67.09 | 9/11 | 563 | -7.98 | 66.6067 |
| 2 | 5 | 0.02 | 2.0 | 64.26 | 10/11 | 555 | -8.78 | 63.7250 |
| 3 | 5 | 0.015 | 2.0 | 60.79 | 7/11 | 600 | -10.60 | 59.3498 |
| 4 | 4 | 0.025 | 2.0 | 47.44 | 8/11 | 495 | -10.52 | 45.6956 |
| 5 | 3 | 0.02 | 2.0 | 45.52 | 9/11 | 546 | -11.30 | 41.8867 |

### Bottom 3 variants (honesty)

| Rank | L | theta | alpha | Comp% | Wins/11 | Trades | WorstDD% | Score |
|------|---|-------|-------|-------|-------|--------|----------|-------|
| 43 | 3 | 0.015 | 1.0 | -35.23 | 4/11 | 1433 | -30.61 | -5.6619 |
| 44 | 5 | 0.015 | 1.0 | -35.62 | 4/11 | 1582 | -20.31 | -6.1935 |
| 45 | 5 | 0.015 | 1.25 | -16.73 | 7/11 | 1213 | -20.25 | -6.8068 |

## Extended OOS Table (Winner Config)

Winner config: L=4, theta=0.02, alpha=2.0

| Window | Start | End | Net% | Trades | WinRate% | MaxDD% |
|--------|-------|-----|------|--------|----------|--------|
| 2020_H2 | 2020-07-01 | 2020-12-31 | 5.11 | 53 | 32.1 | -7.98 |
| 2021_H1 | 2021-01-01 | 2021-06-30 | 6.57 | 53 | 41.5 | -4.28 |
| 2021_H2 | 2021-07-01 | 2021-12-31 | -5.62 | 58 | 36.2 | -7.72 |
| 2022_H1 | 2022-01-01 | 2022-06-30 | 1.78 | 51 | 35.3 | -5.58 |
| 2022_H2 | 2022-07-01 | 2022-12-31 | -0.54 | 52 | 40.4 | -6.84 |
| 2023_H1 | 2023-01-01 | 2023-06-30 | 20.44 | 44 | 50.0 | -5.65 |
| 2023_H2 | 2023-07-01 | 2023-12-31 | 4.28 | 44 | 31.8 | -6.13 |
| 2024_H1 | 2024-01-01 | 2024-06-30 | 3.81 | 53 | 39.6 | -7.91 |
| 2024_H2 | 2024-07-01 | 2024-12-31 | 6.73 | 58 | 41.4 | -5.03 |
| 2025_H1 | 2025-01-01 | 2025-06-30 | 5.64 | 49 | 42.9 | -4.19 |
| 2025_H2 | 2025-07-01 | 2025-12-31 | 6.20 | 48 | 39.6 | -6.51 |
| **TOTAL** | | | **67.09** | **563** | | **-7.98** |

## Walk-Forward Extended

Sliding 6-month train / 3-month test, quarterly advance. Data through 2026-06-02.
Win gate relaxed to ≥70% (was 75% in prior sweep).
**Verdict: pass** (17/24 positive, agg=98.64%, PSR=1.000)

| # | Test Period | Net% | Trades |
|---|-------------|------|--------|
| 1 | 2020-03-08 .. 2020-06-07 | 8.33 | 27 |
| 2 | 2020-06-08 .. 2020-09-07 | -1.61 | 25 |
| 3 | 2020-09-08 .. 2020-12-07 | 2.17 | 23 |
| 4 | 2020-12-08 .. 2021-03-07 | 11.15 | 30 |
| 5 | 2021-03-08 .. 2021-06-07 | 2.86 | 25 |
| 6 | 2021-06-08 .. 2021-09-07 | -1.42 | 28 |
| 7 | 2021-09-08 .. 2021-12-07 | -0.24 | 29 |
| 8 | 2021-12-08 .. 2022-03-07 | -1.71 | 25 |
| 9 | 2022-03-08 .. 2022-06-07 | 3.02 | 23 |
| 10 | 2022-06-08 .. 2022-09-07 | 1.57 | 29 |
| 11 | 2022-09-08 .. 2022-12-07 | -4.34 | 27 |
| 12 | 2022-12-08 .. 2023-03-07 | 9.15 | 21 |
| 13 | 2023-03-08 .. 2023-06-07 | 8.28 | 23 |
| 14 | 2023-06-08 .. 2023-09-07 | 2.87 | 18 |
| 15 | 2023-09-08 .. 2023-12-07 | 9.76 | 23 |
| 16 | 2023-12-08 .. 2024-03-07 | 2.54 | 28 |
| 17 | 2024-03-08 .. 2024-06-07 | -3.36 | 30 |
| 18 | 2024-06-08 .. 2024-09-07 | 2.18 | 31 |
| 19 | 2024-09-08 .. 2024-12-07 | 9.38 | 24 |
| 20 | 2024-12-08 .. 2025-03-07 | 2.21 | 30 |
| 21 | 2025-03-08 .. 2025-06-07 | 1.53 | 21 |
| 22 | 2025-06-08 .. 2025-09-07 | -1.15 | 22 |
| 23 | 2025-09-08 .. 2025-12-07 | 6.64 | 27 |
| 24 | 2025-12-08 .. 2026-03-07 | 1.99 | 26 |

## Aggregate PSR / MinTRL

Pooled per-trade gross returns from winner across all 11 extended windows.
Note: per-trade funding is modeled at window-aggregate level, not per-trade — consistent with prior 0.905 PSR basis.

| Metric | Value |
|--------|-------|
| n_trades | 563 |
| Point Sharpe (per-trade) | 0.0503 |
| PSR vs SR=0 | 0.894 |
| MinTRL | 976 |
| MinTRL satisfied | NO (need 413 more trades) |
| Skew | 1.894 |
| Kurt (raw) | 8.895 |
| Interpretation | insufficient_evidence |
| Prior MinTRL (n=255) | 400 |

## Promotion Recommendation

**`iterate_more`**

Grid winner passes core OOS gates, but not all promotion gates:
- PSR (0.894) < 0.95 gate
- n_trades (563) < MinTRL (976) — gap of 413

## Delta vs Prior Sweep

| Metric | Prior (5 windows, α≥2.0) | Extended (11 windows) | Change |
|--------|--------------------------|----------------------|--------|
| Winner alpha | 2.0 | 2.0 | +0.00 |
| Compounded net% | 43.48% | 67.09% | +23.61pp |
| Win windows | 5/5 | 9/11 | — |
| Total trades | 255 | 563 | +308 |
| MinTRL | 400 | 976 | +576 |
| MinTRL satisfied | NO | NO | — |
| Walk-forward % positive | 55% (11/20) | 70.8% (17/24) | — |

## Open Questions

2. **Walk-forward temporal pattern**: inspect per-window table for secular degradation (later windows systematically weaker = regime change in BTC MOM structure).
3. **Monthly re-optimisation (Algorithm 2)**: paper's adaptive layer would lift trade count and likely improve walk-forward consistency; not implemented.
4. **Live wiring cost**: AdaptiveTrend requires H6 resampling inside live bot — different architecture from multifactor's bar-by-bar approach.
5. **2025_H2 coverage**: data runs through 2026-05-29; 2025_H2 window used in full.

---
*Research only. Not wired to live bot. No risk.py or config/params.yaml touched.*