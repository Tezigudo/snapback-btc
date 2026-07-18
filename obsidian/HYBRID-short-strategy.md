---
tags: [strategy]
status: validated, awaiting capital
deploy_target: 2026-06-25
---

# cnh-hybrid-short-v1

Short-only 4h pattern strategy combining two pattern detectors:

- [[detectors/distribution-top]] (DT) — uptrend → chop → breakdown
- [[detectors/inverse-cup-handle]] (ICnH) — concave-up parabola with handle

The two are unioned via [[detectors/pattern-dedup]] (15-bar window). Entry on
the breakdown bar (DT) or EMA24 cross-down following the ICnH pattern. Stop
at 1.5 × ATR(14, 4h), TP at distance to EMA(100), time-stop 96 bars (16 days).

## Validation status

| Phase | Outcome |
|---|---|
| [[phases/phase-1-walk-forward]] | PASS — dedup=15 best (OOS +18.8%, Sharpe 8.44) |
| [[phases/phase-2-friction-sizing]] | Friction PASS / Sizing mitigation: ≥$100/leg |
| [[phases/phase-3-live-evaluator]] | PASS — pure-function module + tests |
| [[phases/phase-3b-stateful-dedup]] | PASS — 100% backtest reproduction |
| [[phases/phase-4-portfolio-sim]] | PASS — Sharpe lift +0.258 |
| [[phases/phase-5-dedup-choice]] | Locked at dedup=15 |
| [[phases/phase-6-deploy-plumbing]] | Code done; exchange-side pending |

## Portfolio role

Third leg in the deploy alongside [[strategies/multifactor-v1]] and
[[strategies/donchian-v3]]. Near-zero correlation (≈0.01) with both,
proving genuine [[concepts/regime-complementarity|regime diversification]].

| | Sharpe | Cum | Max DD |
|---|--:|--:|--:|
| 2-leg baseline (v1 + Donchian) | 2.43 | +398% | small |
| **3-leg (v1 + Donchian + HYBRID)** | **2.69** | +264% | smaller |
| Sharpe lift | +0.26 | — | — |

## Deploy config

- File: `../config/params_cnh_hybrid_short.yaml`
- Risk per trade: 2.75%
- Leverage: 20× ceiling
- Capital floor: $100/leg (see [[decisions/deploy-capital-floor]])
- Killswitch: -35.5% (matches v1/Donchian; never trips in backtest)

## Honest caveats

- [[concepts/hold-time-mismatch]] — median 0.83 days, not the 3-7d user originally asked for
- [[concepts/oos-window-concentration]] — 2024-H2 dominates OOS PnL
- Live evaluator captures ~83% of "ideal" backtest edge (residual gap from
  ICnH entry-timing edge cases)

## See also

- [[artifacts/live-evaluator]] — the runtime module
- [[artifacts/html-report]] — `HYBRID_VS_ALL.html` for headline charts
- Failed alternatives that led here: [[strategies/chop-reverter-FAILED]],
  [[strategies/retest-resume-FAILED]]
