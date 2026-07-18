---
tags: [strategy, failed]
status: abandoned 2026-05-25
---

# chop-reverter-v1 (FAILED Phase 1)

The first pitch for a 3rd portfolio leg. Mean-reversion on 15m gated by a
1-day chop classifier — designed to fire when v1 and Donchian are both
silent (chop regime).

## Why it failed

Phase 1 feasibility audit (`../../tools/phase1_chop_feasibility.py`):

| Chop threshold (EMA-slope-strength on 4h) | chop-trade-days/yr | Verdict |
|---|---:|---|
| slope < 0.05 (project-canonical) | **53.5** | FAIL (gate 60/yr) |
| slope < 0.07 | 65 | passes only by moving the line |
| slope < 0.10 | 83 | reframes to a different strategy |
| ER < 0.25 | 110 | classifier broken on BTC perp |

Loosening the threshold would have been [[concepts/tuning-into-compliance]] —
the exact anti-pattern the validation gates were designed to prevent.

Per-year at the canonical threshold: 2022=44, 2023=68, 2024=45, 2025 H1=30.
Two of four full years sit at ~12% of weeks — structurally thin contributor
regardless of edge.

## Plan file (kept for history)

`../../CHOPREVERTER_PLAN.md` — marked FAILED at the top.

## What we learned

The pattern across this and 3 other rejections led to the realisation that
[[decisions/strategy-not-found-at-50|the constraint is capital, not strategy design]]:
at $50/leg with BTC at $80-100k, the 0.001 BTC min-qty floor blocks almost
any normal-width stop.

## See also

- [[strategies/retest-resume-FAILED]] — sibling failed attempt
- [[concepts/min-qty-floor]]
