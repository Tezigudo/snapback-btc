---
tags: [concept, caveat]
---

# Hold-time mismatch with original ask

The user asked for "3-7 days per position" when scoping the 3rd leg
search. [[HYBRID-short-strategy]] actually closes most trades **inside a
day**.

## Distribution (live sim at $100/2.75%)

| Statistic | Days | Bars |
|---|--:|--:|
| q10 | 0.17 | 1 |
| q25 | 0.33 | 2 |
| **median** | **0.83** | 5 |
| q75 | 1.67 | 10 |
| q90 | 3.70 | 22 |
| max | 7.33 | 44 |
| mean | 1.45 | 8.7 |

Only **48% of trades held 1-7 days**. The rest close in <1 day.

## Why

- Tight SL: 1.5×ATR ≈ 1.76% — gets hit by routine 4h noise on losers
  (median 0.83 days)
- Tight TP: distance to EMA(100), usually 1.5-3% from entry — wins also
  finish fast (median 0.92 days)
- Time stop never fires (96 bars = 16 days; max observed = 7.33 days)

The strategy is "snap into breakdown, take quick profit or cut quick" —
not the swing-trade profile the user envisioned.

## What this means

- Frequency is genuinely low (~12 trades/yr) even though hold time is short
- Capital turnover per year is high (each $1 deployed cycles through
  ~12 trades) — risk-adjusted returns benefit
- If user wanted "lock in for a week" psychological profile, HYBRID is the
  wrong shape. The mechanics still work; just the experience differs.

## Possible adjustments (not recommended without re-validation)

- Wider TP (e.g., to EMA(200) instead of EMA(100)) would extend holds —
  but Phase 1 showed tp_emas=("ema200",) underperforms (Sharpe drops)
- Trailing stop would extend winners — adds complexity for marginal gain

The current config is optimal per the strict-gates validation. Surface
this mismatch to the user; if they want longer holds, treat it as a
separate design exercise.

## See also

- [[HYBRID-short-strategy]]
- [[artifacts/html-report]] §5b for the hold-time table
