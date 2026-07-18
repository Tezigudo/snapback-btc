---
tags: [strategy, deployed]
status: deployed mainnet
---

# donchian-v3 cons

The second deployed leg in the snapback-btc portfolio. Classic turtle-style
Donchian breakout on 4h bars with EMA-slope regime gate.

## Key params

- TF: 4h
- Entry: 80-bar Donchian channel breakout
- Exit: 20-bar opposite Donchian
- Regime gate: EMA(120) slope > +3% (long) / < -3% (short)
- SL: 1.5 × ATR(20, 4h)
- Time stop: 48 bars (8 days)

## Backtest

Realistic sim 2019-09 → 2026-05 at $50.50 base → $675 final, Sharpe 0.86,
99.4% fire rate of 177 signals. See `../../V1_DONCHIAN_RESULTS.html`.

## Role in portfolio

Trend-following breakout — opposite hypothesis vs [[strategies/multifactor-v1]]
(which assumes mean-reversion in trend). Donchian fires when v1's RSI gate
won't (no extreme on RSI 14).

Donchian is the "trend ride" leg. [[strategies/multifactor-v1]] is the
"pullback in trend" leg. [[HYBRID-short-strategy]] is the "pattern-based
top short" leg. Together they span trend, pullback, and topping regimes.

## Files

- Config: `../../config/params_donchian.yaml`
- Live evaluator: `../../strategy/live_donchian_v3.py`
- Service: `../../deploy/snapback-btc-donchian.service`

## See also

- [[phases/phase-4-portfolio-sim]] for combined-portfolio role
- [[concepts/regime-complementarity]] — why all three legs together work
