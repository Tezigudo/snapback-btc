---
tags: [detector, mechanism]
---

# Pattern-level dedup (15-bar)

Mechanism that prevents [[detectors/distribution-top|DT]] and
[[detectors/inverse-cup-handle|ICnH]] hits from firing back-to-back. Locked
at `dedup_bars = 15` per [[decisions/dedup-choice-15]].

## How it works

Walk bars forward chronologically. For each bar:

1. Check DT detector. If a pattern terminates at this bar AND ≥ 15 bars
   since the last admitted pattern, ADMIT it. Mark this bar as the new
   "last admitted index."
2. Else, check ICnH detector with the same dedup rule.
3. If both detected at the same bar, DT wins (checked first).

This is implemented in `find_hybrid_patterns` (`../../tools/icnh_final_tune.py`)
for the backtest, and re-implemented in `_admitted_patterns`
(`../../strategy/live_cnh_hybrid_short.py`) for the live evaluator.

## Why pattern-level, not signal-level

A pattern that gets ADMITTED but never produces a trade (e.g., ICnH with no
EMA24 cross-down, or DT/ICnH with no valid TP slot) still "holds" the
15-bar dedup window. This prevents over-firing — see
[[concepts/ghost-pattern]] for the edge case this resolves.

[[phases/phase-3b-stateful-dedup]] added stateful tracking to the live
evaluator so it reconstructs this admission state each call. Before that
fix, live fired 138% of backtest signals — the over-firing dragged
portfolio Sharpe from +0.31 (ideal) to **-0.10** (broken).

## Variants tested (Phase 5)

| dedup | trades/yr | WR | hybrid cum | Sharpe lift |
|------:|---------:|---:|----------:|-----------:|
| 5 | 15.3 | 58.2% | +65% | +0.123 |
| 10 | 13.5 | 62.8% | +93% | +0.236 |
| **15** | 12.5 | **65.0%** | +86% | **+0.258** |

## See also

- [[phases/phase-3-live-evaluator]]
- [[phases/phase-3b-stateful-dedup]]
- [[phases/phase-5-dedup-choice]]
- [[decisions/dedup-choice-15]]
