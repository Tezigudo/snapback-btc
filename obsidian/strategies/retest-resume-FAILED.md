---
tags: [strategy, failed]
status: abandoned 2026-05-26
---

# Retest-Resume v1 (FAILED Phase 1)

The fourth pitched 3rd-leg candidate. Donchian-20 breakout on 4h + retest
within 5 bars + confirmation close + 0.8×ATR stop + 2.5R TP. Multi-day hold.

The pitch: tight structural stops at retest entry should fit $50/leg sizing
even though wider-stop strategies like [[HYBRID-short-strategy]] don't.

## Why it failed

Audit: `../../tools/retest_feasibility.py`. Frequency was fine (57/yr,
above the 40/yr gate) but the outcome was disastrous:

- Win rate: **25.2%** (target 45%)
- Mean R: -0.12  Median R: -1.00
- Cum: **-61.7%** net over 5.9 years
- Exits: SL 252 / TP 85 / time 0 (3:1 SL:TP — stops too tight)
- Per year: only 2022 bear positive (+8.3%); all others negative

## Root cause

The 0.8×ATR stop (median 1.22% on 4h BTC) is too tight — routine intraday
noise eats it before the 2.5R move can complete. The retest pattern itself
also doesn't transfer well from equities to crypto perp; BTC's liquidity-grab
behavior fades retests constantly.

## What we learned

This was the 4th rejection in 3 days at $50/leg. After this, the pattern
was clear: [[decisions/strategy-not-found-at-50|no strategy fits $50/leg
under current BTC prices]] given Binance's 0.001 BTC min-qty floor.

Led to the explicit pivot: deploy HYBRID at ≥$100/leg when capital arrives.

## See also

- [[strategies/chop-reverter-FAILED]]
- [[concepts/min-qty-floor]]
