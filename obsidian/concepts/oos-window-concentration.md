---
tags: [concept, caveat]
---

# 2024-H2 dominates OOS PnL

Of the 18 OOS trades from [[phases/phase-1-walk-forward]] at dedup=15,
**7 fall in 2024-H2** and they produced **+13.5%** cum out of the total
+18.8% OOS cum. ~72% of OOS profit came from a single 6-month window.

## Why this matters

- Phase 1's headline Sharpe (8.44) and worst-window (-0.1%) numbers are
  partially driven by that single regime
- 2024-H2 was a strong-bull-then-pullback regime — perfect for
  [[detectors/distribution-top|DT distribution-top]] entries
- If 2027 looks more like 2025 (HYBRID's weakest year, +0.4%), expect a
  flat-to-negative slice for the year

## Mitigations applied in validation

- [[phases/phase-4-portfolio-sim]] uses the full 6.4-year window, not
  just OOS — broader regime exposure
- Per-year breakdown ([[decisions/keep-both-detectors]]) confirms HYBRID
  is positive in 6 of 7 calendar years observed
- The portfolio role is to ADD diversification, not be a standalone return
  source — so even a flat year is acceptable if v1+Donchian carry

## What NOT to do

- Don't re-pick the OOS window to avoid 2024-H2. That's reverse-cherry-
  picking and the picked window's metric isn't fair.
- Don't downweight HYBRID below 33% in the portfolio because of this.
  Equal-weight gave +0.258 Sharpe lift; the 2024-H2 cluster is the
  pre-conditioned outcome of using detectors that target this regime.

## See also

- [[phases/phase-1-walk-forward]]
- [[phases/phase-4-portfolio-sim]]
- [[decisions/keep-both-detectors]]
