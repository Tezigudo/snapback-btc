---
tags: [decision, locked]
locked: 2026-05-26
---

# Deploy capital floor — ≥$100/leg for HYBRID

Empirical result from [[phases/phase-2-friction-sizing]] and the
[[artifacts/realistic-deploy-matrix|realistic deploy sim]].

## The numbers

| Equity | Risk% | Final | Skip rate |
|---:|---:|---:|---:|
| $50 | 1.5% | $50.40 | 95% — strategy effectively dead |
| $50 | 2.75% | $69.56 | 46% |
| **$100** | **2.75%** | **$197.31** | **4%** — recommended |
| $150 | 2.75% | $296.05 | 0% |

## Why

[[concepts/min-qty-floor|Binance 0.001 BTC min-qty]] at BTC ~$80-100k =
$80-100 minimum notional. A $50/1.5% position has $0.75 risk budget
which, at typical ATR (1.17% of close) and 1.5×ATR stop, produces only
$42 notional. Below the min-qty floor for any meaningful subset of bars.

## Rule

- DO NOT deploy HYBRID with `start_equity < $80`.
- The locked config (`config/params_cnh_hybrid_short.yaml`) sets
  `min_capital_warn_usdt: 100.0` for this reason.
- Total Futures capital for the full 3-leg portfolio = ≥$200 (v1+Donchian
  at $50.50 each + HYBRID at $100+).
- This SUPERSEDES the earlier "≥$150 total" relaxation that was based on
  an incorrect sizing model.

## What I don't do

- Lower risk_per_trade_pct to "fit" $50/leg. Empirically worse — see
  the $50/1.5% row.
- Raise leverage to compensate. The 20× ceiling is a hard wall and not
  the binding constraint here (min-qty is).

## See also

- [[phases/phase-2-friction-sizing]]
- [[concepts/min-qty-floor]]
- [[decisions/strategy-not-found-at-50]]
- Memory: `snapback_hybrid_short_realistic_deploy_matrix.md`
