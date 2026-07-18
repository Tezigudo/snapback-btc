---
tags: [phase, completed, pass]
gate_result: PASS
---

# Phase 1 — Walk-forward audit

The honest go/no-go for [[HYBRID-short-strategy]]. Hold out the last 4 of
12 windows, fit on the first 8, score on the held-out slice.

## Tool

`../../tools/hybrid_walkforward.py`

## Setup

- IS: windows 1-8 (2020-H2 → 2024-H1)
- OOS: windows 9-12 (2024-H2 → 2026-H1)
- For each dedup ∈ {5, 10, 15}, sweep knobs (sl_atr × tp_emas) on IS, pick
  best by IS Sharpe, score on OOS

## Result

All three dedup variants picked the SAME knob set on IS (sl=1.5×ATR,
tp=EMA(100)) — a stability signal.

| dedup | IS Sharpe | IS cum | OOS n | OOS cum | OOS Sharpe | worst OOS window |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | +4.08 | +60% | 28 | +4.8% | +1.40 | -4.8% |
| 10 | +4.10 | +44% | 21 | +14.8% | +5.60 | -2.8% |
| **15** | +4.56 | +47% | 18 | **+18.8%** | **+8.44** | **-0.1%** |

Gate: PASS for all three. Dedup=15 wins on OOS Sharpe and worst-window
metric.

## Note on the OOS Sharpe IMPROVING vs IS

Unusual; typical pattern is IS Sharpe > OOS Sharpe. With N=18 OOS trades
this is statistically noisy; [[concepts/oos-window-concentration|2024-H2
dominates OOS PnL]] which inflates the result. Phase 4 portfolio sim is
the firmer test.

## Next phases

→ [[phases/phase-2-friction-sizing]]

## See also

- Memory: `snapback_hybrid_short_phase1_pass.md`
- Data: `../../data/hybrid_walkforward_results.json`
