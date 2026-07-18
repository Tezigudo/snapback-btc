---
tags: [concept, edge-case]
---

# Ghost pattern (the dedup edge case that almost killed deploy)

A "ghost pattern" is a pattern that gets ADMITTED by the
[[detectors/pattern-dedup|dedup rule]] but never produces an actual trade
— typically because:

- ICnH admitted, but no EMA(24) cross-down within `entry_max_bars_after_handle`
- DT or ICnH admitted, but EMA(100) sits above current close (no SHORT TP slot)

The backtest's `find_hybrid_patterns` admits patterns at DETECTION time,
before `simulate_trades` filters for trade-ability. So a ghost pattern
still HOLDS the 15-bar dedup window — blocking subsequent patterns that
might have produced real trades.

## Why this matters

A stateless live evaluator can't see ghosts. It looks at the current bar
in isolation: "is there a pattern here? If yes, does it have a valid
trade? Fire."

But the backtest, at that same bar, may have already DENIED admission
because a ghost pattern in the prior 15 bars holds the slot.

Live without state → MORE signals than backtest. The extras are mostly
*lower-quality* signals that ghost patterns had wisely suppressed.

## Empirically (from [[phases/phase-3-live-evaluator]])

Initial stateless live evaluator: **138% reproduction** vs backtest.

| | Stateless live | Backtest ideal |
|---|--:|--:|
| OOS trades | 25 | 18 |
| Win rate | 53.8% | 70.5% |
| Cum | +17% | +75% |
| 3-leg Sharpe lift | **-0.10** | **+0.31** |

The over-firing dragged portfolio Sharpe NEGATIVE.

## Fix (Phase 3b)

[[phases/phase-3b-stateful-dedup]]: reconstruct backtest admission state
by scanning visible bars each call. Stateless API for the bot, but the
math is now correct.

After fix: 100% reproduction (18/18 OOS trades match), Sharpe lift +0.258.

## Lesson

When porting research-tree dedup to live: the live evaluator MUST replicate
the admission state, not just check the current bar. Otherwise the live
fire set is a superset that includes low-quality "extras" the research
dedup had filtered out.

## See also

- [[detectors/pattern-dedup]]
- [[phases/phase-3b-stateful-dedup]]
