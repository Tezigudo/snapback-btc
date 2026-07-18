---
tags: [concept, constraint]
---

# Binance min-qty floor (0.001 BTC)

The hard constraint that shapes the entire deploy economics. Binance
Futures BTCUSDT-perp requires:

- `min_qty: 0.001 BTC` (per exchange/constraints.py)
- `min_notional: $50 USDT`

At BTC ~$80-100k, the **min-qty rule is binding** — 0.001 × $90k = $90
minimum notional, which exceeds the $50 min-notional rule.

## Sizing math

For a risk-based-sized position to fit:

```
notional = equity × risk_pct / stop_pct ≥ max($50, 0.001 × BTC_price)
```

At $90k BTC: notional ≥ $90.
- $100 equity × 2.75% / 1.76% stop = $156 — fits ✓
- $50 equity × 2.75% / 1.76% stop = $78 — borderline ✗ (skips ~30% of bars)
- $50 equity × 1.5% / 1.76% stop = $43 — fails ✗ (skips 100% of bars)

## Why we measure ATR(14, 4h)

`stop_pct = sl_atr_mult × ATR / close`. The ATR distribution drives
how often the position fits.

For [[HYBRID-short-strategy]]:
- Median ATR%(14, 4h) at entry: 1.17%
- 1.5×ATR → ~1.76% stop_pct
- This sets the binding sizing constraint

## Implications

- [[decisions/deploy-capital-floor|HYBRID needs ≥$100/leg]]
- [[decisions/strategy-not-found-at-50|No strategy at $50/leg fits]] given
  normal stop widths AND fees survival

## What if BTC drops?

At BTC $40k, min-qty = $40 notional. Falls below the $50 min-notional
floor; the latter becomes binding. The deploy economics IMPROVE — $50/leg
strategies become more viable at lower BTC prices.

This is one structural reason to NOT short-deploy at high BTC prices: the
size constraint itself moves with the asset price.

## See also

- `../../exchange/constraints.py`
- [[phases/phase-2-friction-sizing]]
