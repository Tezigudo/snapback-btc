---
tags: [artifact, code]
---

# Live evaluator — `strategy/live_cnh_hybrid_short.py`

The pure-function module the bot calls each closed 4h bar to decide whether
to enter a HYBRID short.

## Signature

```python
def evaluate_signal_cnh_hybrid_short(
    bars_4h: pd.DataFrame,
    funding_rate: float,         # unused; signature compat
    params: dict,
) -> tuple[str | None, float, float, dict]:
    # Returns (side, sl_distance, tp_distance, debug)
```

Matches `evaluate_signal_donchian_v3`'s shape. The `bars_4h` slot takes
Capitalised OHLCV; internally normalised to lowercase and indicators
attached.

## What it does

1. Compute EMAs and ATR via [[detectors/distribution-top|cnh_detectors.attach_indicators]]
2. Reconstruct [[detectors/pattern-dedup|admitted-pattern state]] via
   `_admitted_patterns` over the visible bars window
3. If the LAST admitted pattern is a DT at the current bar → fire short
4. Else if there's an admitted ICnH in the last `entry_max_bars_after_handle`
   bars AND the current bar is an EMA(24) cross-down → fire short
5. Else → return None
6. Skip fire if EMA(100) ≥ entry price (no valid SHORT TP)

## Production-safety

- Lives in `strategy/` (packaged in the wheel; `tools/` is research-only)
- Imports only from `strategy.indicators` and `strategy.cnh_detectors`
- Stateless: no per-call state to manage by the caller
- Internally O(N×50) per call (~5ms at N=300 visible bars)

## Test coverage

`../../tests/test_cnh_hybrid_short.py` — 7 passing tests including the
load-bearing invariant that `_admitted_patterns` reproduces
`find_hybrid_patterns` exactly on real OOS data.

## Bot integration

Dispatch via `evaluate_for_strategy` in `../../bot_internals.py`:

```python
if strategy_name == "cnh-hybrid-short-v1":
    side, sl_dist, tp_dist, dbg = evaluate_signal_cnh_hybrid_short(...)
```

## See also

- [[phases/phase-3-live-evaluator]] — design rationale
- [[phases/phase-3b-stateful-dedup]] — the dedup fix
- [[detectors/pattern-dedup]] — the admission mechanism
