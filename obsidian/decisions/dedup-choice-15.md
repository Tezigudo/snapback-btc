---
tags: [decision, locked]
locked: 2026-05-26
---

# Dedup choice — 15 bars

The `dedup_bars` parameter for [[detectors/pattern-dedup]] is locked at 15.

## Numbers (from [[phases/phase-5-dedup-choice]])

| dedup | trades/yr | WR | Hybrid cum | Sharpe lift |
|------:|---------:|---:|----------:|-----------:|
| 5 | 15.3 | 58.2% | +65% | +0.123 |
| 10 | 13.5 | 62.8% | +93% | +0.236 |
| **15** | 12.5 | **65.0%** | +86% | **+0.258** |

## Why 15

- Best portfolio Sharpe lift (+0.258)
- Best win rate (65%)
- Matches [[phases/phase-1-walk-forward]]'s pick (which used IS Sharpe as
  the picker)
- Cleanest risk-adjusted contribution; the 2-leg → 3-leg lift is largely
  about RISK reduction, not absolute return — dedup=15's higher Sharpe is
  the relevant metric

## What I rejected

- **dedup=10**: marginally better absolute 3-leg cum (+268.6% vs +264.0%)
  but lower Sharpe. Trade-off doesn't favor it.
- **dedup=5**: strictly dominated. Lower WR, lower Sharpe, lower hybrid
  cum. The extra trades are noise.

## Implementation

The locked config sets `dedup_bars: 15` in `strategy:` section of
`../../config/params_cnh_hybrid_short.yaml`. The default in
`HybridConfig` (`../../strategy/cnh_detectors.py`) is also 15.

## See also

- [[detectors/pattern-dedup]]
- [[phases/phase-5-dedup-choice]]
- Memory: `snapback_hybrid_short_phase5_dedup_compare.md`
