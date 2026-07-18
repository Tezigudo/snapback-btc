---
tags: [concept, diversification]
---

# Regime complementarity

Two strategies (or detectors) are *regime-complementary* when their PnL
peaks in DIFFERENT market regimes. The combined portfolio gets paid in
more regimes than either alone — that's the source of the Sharpe lift,
not just lower variance from diversification.

## The HYBRID example

[[detectors/distribution-top]] wins in **strong-trend years**:
- 2021 bull market top: DT-only +34% vs ICnH-only +0.7%

[[detectors/inverse-cup-handle]] wins in **chop/transition years**:
- 2023 recovery: ICnH-only +12.4% vs DT-only +1.4%
- 2024 mixed: ICnH-only +14.8% vs DT-only +0.7%

Combined HYBRID > either alone because the detectors cover non-overlapping
regimes. See [[decisions/keep-both-detectors]].

## The 3-leg portfolio example

[[strategies/multifactor-v1]] — pullback in trend (RSI extreme)
[[strategies/donchian-v3]] — trend-following breakout
[[HYBRID-short-strategy]] — pattern-based top short

All three near-zero correlated (≈0.01 pairwise). Each contributes when
the other two are silent.

## Difference from "diversification" (vanilla)

Vanilla diversification: assets with returns that don't move together
(correlation < 0.5) reduce portfolio vol. Sharpe goes up via lower σ.

Regime complementarity: assets that each have peak ER in DIFFERENT
regimes. Both vol AND expected return go up because the portfolio earns
in more periods. Stronger lift than pure decorrelation.

## How to measure

- Per-year cum return matrix (see [[phases/phase-4-portfolio-sim]])
- Daily-P&L pairwise correlation
- Active-day overlap (% of trading days where ≥2 legs trade)

For HYBRID: the 2024-H2 cluster is the ICnH peak; DT is silent there.
2021-Q4 is the DT peak; ICnH is silent there. That's the pattern.

## See also

- [[detectors/distribution-top]] vs [[detectors/inverse-cup-handle]]
- [[phases/phase-4-portfolio-sim]]
- [[decisions/keep-both-detectors]]
