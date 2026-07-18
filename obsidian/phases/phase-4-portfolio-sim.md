---
tags: [phase, completed, pass]
gate_result: PASS
---

# Phase 4 — Portfolio simulation

The load-bearing portfolio test: does adding [[HYBRID-short-strategy]] as a
3rd leg lift the combined Sharpe?

## Tool

`../../tools/hybrid_phase4_portfolio.py`

## Setup

- Load v1 + Donchian-v3 cons trade CSVs from `reports/full_history_*`
- Replay HYBRID via the live evaluator (post-3b stateful dedup)
- Build daily P&L per leg
- Compute Sharpe(2-leg) vs Sharpe(3-leg), correlation matrix

## Result

| | Sharpe | Cum | Lift vs 2-leg |
|---|--:|--:|--:|
| v1 alone | 0.97 | +124% | — |
| Donchian-v3 alone | 2.53 | +943% | — |
| HYBRID live (post-3b) | 1.13 | +86% | — |
| HYBRID ideal (backtest) | 1.27 | +98% | — |
| 2-leg baseline (50/50 v1+Don) | 2.43 | +398% | 0 |
| **3-leg w/ live HYBRID** | **2.69** | +264% | **+0.258** |
| 3-leg w/ ideal HYBRID | 2.74 | +271% | +0.311 |

Gates:
- Sharpe lift ≥ 0.1 → PASS (+0.258)
- |corr(hybrid, v1)| < 0.3 → PASS (0.010)
- |corr(hybrid, donchian)| < 0.3 → PASS (0.010)

## Why live captures only 83% of ideal

Live evaluator's per-bar entry-point search for ICnH cross-downs uses
`is_ema_breakdown(df, j, "ema24")` which has minimally different timing
than backtest's `simulate_trades` inline entry loop. Acceptable residual.

## Correlation matrix (active days only)

```
                v1   donchian  hybrid
v1           1.000     0.067   0.010
donchian     0.067     1.000  -0.010
hybrid       0.010    -0.010   1.000
```

Three near-zero off-diagonals = [[concepts/regime-complementarity|genuine
regime diversification]].

## Next phases

→ [[phases/phase-5-dedup-choice]]

## See also

- Memory: `snapback_hybrid_short_phase4_ideal_pass.md`
- Data: `../../data/hybrid_phase4_portfolio_results.json`
