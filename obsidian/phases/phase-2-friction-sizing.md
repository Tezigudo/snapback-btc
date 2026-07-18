---
tags: [phase, completed, mitigated]
gate_result: friction PASS / sizing MITIGATION REQUIRED
---

# Phase 2 — Friction stress + sizing reality

Two parallel checks: (a) can the strategy survive realistic fees, and (b)
can it be sized at the planned $50/leg?

## Tool

`../../tools/hybrid_friction_sizing.py`

## Friction stress — PASS with 3× margin

| Round-trip fees | OOS cum | Mean/trade |
|--:|--:|--:|
| 8 bps (Binance live taker) | +19.2% | +99.8 bps |
| 13 bps (live + 5 bps slip) | +18.2% | +94.8 bps |
| 15 bps (stress) | +17.7% | +92.8 bps |

Gate was ≥30 bps after-fee edge per trade. Strategy survives easily.

## Sizing reality — discovered the [[concepts/min-qty-floor]]

| Equity | Risk% | Skip rate |
|---:|---:|---:|
| $50 | 1.5% | **100%** — strategy unsizeable |
| $50 | 2.5% | 72% |
| $80 | 1.5% | 72% |
| $100 | 1.5% | 50% |

ATR(14, 4h) median at entry is 1.17% of close. At BTC ~$80-100k, the
0.001 BTC min-qty floor = $80-100 minimum notional, which a $50/1.5%
position can't reach.

## Verdict

Strategy SAFE on friction. CAPITAL needs to be ≥$100/leg, not $50 as the
pitch had assumed. See [[decisions/deploy-capital-floor]].

## Next phases

→ [[phases/phase-3-live-evaluator]]

## See also

- Memory: `snapback_hybrid_short_phase2_sizing_floor.md`
- [[concepts/min-qty-floor]]
- Data: `../../data/hybrid_friction_sizing_results.json`
