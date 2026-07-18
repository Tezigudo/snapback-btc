---
tags: [artifact, data]
---

# Phase 4 portfolio results

Daily-P&L portfolio sim showing the 3-leg combined Sharpe vs the 2-leg
baseline. The strategy's load-bearing PASS gate.

Tool: `../../tools/hybrid_phase4_portfolio.py`
Data: `../../data/hybrid_phase4_portfolio_results.json`

## Per-leg metrics

| | Sharpe | Cum | Trades |
|---|--:|--:|--:|
| [[strategies/multifactor-v1]] | 0.97 | +124% | 671 |
| [[strategies/donchian-v3]] | 2.53 | +943% | 172 |
| [[HYBRID-short-strategy]] live | 1.13 | +86% | 80 |
| [[HYBRID-short-strategy]] ideal | 1.27 | +98% | 81 |

## Portfolio (equal-weighted daily P&L)

| | Sharpe | Cum |
|---|--:|--:|
| 2-leg baseline (50/50 v1+Don) | 2.43 | +398% |
| **3-leg w/ live HYBRID** | **2.69** | +264% |
| 3-leg w/ ideal HYBRID | 2.74 | +271% |

Sharpe lift LIVE: **+0.258** (gate ≥ 0.10 → PASS)
Sharpe lift IDEAL: +0.311

## Correlation matrix

```
              v1   donchian  hybrid
v1         1.000     0.067   0.010
donchian   0.067     1.000  -0.010
hybrid     0.010    -0.010   1.000
```

All pairs well below 0.30 → [[concepts/regime-complementarity|true
diversification]].

## Note on cum return

3-leg cum (+264%) is LOWER than 2-leg (+398%) because the equal-weight
math dilutes Donchian's outsized contribution (+943%). The portfolio
Sharpe rises because total volatility falls more than mean does.

In real deploy with separate sub-accounts (each leg at its own equity),
Donchian's full PnL is retained AND the HYBRID's $100 → $197 is ADDITIVE
— so adding HYBRID adds ~$97 absolute dollars on $50 base of allocated
capital over the 6.4-year window.

## See also

- [[phases/phase-4-portfolio-sim]]
- [[concepts/regime-complementarity]]
- Memory: `snapback_hybrid_short_phase4_ideal_pass.md`
