---
tags: [strategy, deployed]
status: deployed mainnet
---

# multifactor-v1

The first deployed leg in the snapback-btc portfolio. Long+short on 15m bars
using a multifactor filter stack: RSI(14) + volume spike + EMA(200) trend +
funding-rate gate.

## Key params

- TF: 15m entry, 1h trend
- RSI long < 40, short > 70
- Volume > 2× SMA(20)
- Close vs EMA(200) for trend
- Funding extreme threshold ±0.05%/8h
- SL 1.5%, TP 3.0% (2:1 R:R)
- Time-stop 14 days

## Backtest

5 OOS windows 2022 H1 → 2025 H1: +55.73% compounded, 4 of 5 windows positive.
Worst window: 2024 H1 chop (-12.56%). See `../../PATH2_RESULTS.html`.

## Role in portfolio

Trend-pullback both directions. Captures momentum-with-reversion regimes
where RSI extremes flag mean-reversion ops within a directional trend.

Complementary to [[strategies/donchian-v3]] (breakouts) and
[[HYBRID-short-strategy]] (pattern-based shorts) — see
[[concepts/regime-complementarity]].

## Files

- Config: `../../config/params.yaml`
- Live evaluator: `../../strategy/live_multifactor_v1.py`
- Service: `../../deploy/snapback-btc.service`

## See also

- [[phases/phase-4-portfolio-sim]] for combined-portfolio role
