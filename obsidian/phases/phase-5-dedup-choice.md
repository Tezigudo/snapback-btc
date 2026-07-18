---
tags: [phase, completed, locked]
gate_result: User pick = dedup=15
---

# Phase 5 — Dedup head-to-head

Final tuning step: pick dedup ∈ {5, 10, 15} after seeing portfolio-level
metrics for each variant.

## Tool

`../../tools/hybrid_phase5_dedup_choice.py`

## Setup

For each dedup, run the live HYBRID through the same portfolio sim as
[[phases/phase-4-portfolio-sim]]. Compare:

- Trades per year (frequency)
- Win rate
- Hybrid solo cum
- Portfolio Sharpe lift
- Correlations

## Result

| dedup | trades/yr | WR | Hybrid cum | Sharpe lift | 3-leg cum |
|------:|---------:|---:|----------:|-----------:|----------:|
| 5 | 15.3 (1.3/mo) | 58.2% | +65% | +0.123 | +250% |
| 10 | 13.5 (1.1/mo) | 62.8% | +93% | +0.236 | **+269%** |
| **15** | 12.5 (1.0/mo) | **65.0%** | +86% | **+0.258** | +264% |

All three pass the Sharpe-lift gate. dedup=5 clearly weakest (low WR drags
hybrid Sharpe). dedup=10 wins by 4pp on absolute cum; dedup=15 wins on
Sharpe lift and WR.

## Decision

[[decisions/dedup-choice-15|dedup=15 locked]] — matches the Phase 1 pick,
cleanest risk-adjusted contribution.

## See also

- Memory: `snapback_hybrid_short_phase5_dedup_compare.md`
- Data: `../../data/hybrid_phase5_dedup_choice_results.json`
- [[decisions/dedup-choice-15]]
