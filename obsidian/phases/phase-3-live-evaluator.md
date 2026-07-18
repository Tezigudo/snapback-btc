---
tags: [phase, completed, pass]
gate_result: PASS (but with documented over-firing — see phase-3b)
---

# Phase 3 — Live signal evaluator

Build a production-safe Python module that the bot calls each closed 4h bar.
Pure function, matches the existing `live_donchian_v3.py` shape.

## Tool / artifact

- `../../strategy/cnh_detectors.py` — extracted pattern detectors
- `../../strategy/live_cnh_hybrid_short.py` — pure-function evaluator
- `../../tools/hybrid_phase3_validate.py` — reproduction check
- `../../tests/test_cnh_hybrid_short.py` — 7 unit tests including the
  invariant `_admitted_patterns ≡ find_hybrid_patterns`

## Production-safety note

`tools/` is NOT in the project wheel (per `pyproject.toml`'s
`[tool.hatch.build.targets.wheel] packages = ["strategy", "exchange"]`).
The detectors were copied into `strategy/cnh_detectors.py` so the live bot
doesn't depend on the research tree.

A regression test guards against drift between the two copies.

## Initial result

Stateless live evaluator: reproduction = **138%** of backtest signals
(25 OOS fires vs 18 expected). Strict gate of ≥95% passes, but the
over-firing is meaningful — see [[concepts/ghost-pattern]] for why.

[[phases/phase-3b-stateful-dedup]] fixed this.

## Next phases

→ [[phases/phase-3b-stateful-dedup]] → [[phases/phase-4-portfolio-sim]]

## See also

- Memory: `snapback_hybrid_short_phase3_pass_overfire.md`
