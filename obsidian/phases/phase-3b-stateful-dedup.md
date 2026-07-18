---
tags: [phase, completed, pass, critical]
gate_result: PASS
---

# Phase 3b — Stateful pattern dedup

Critical fix to the live evaluator. Reconstructs `find_hybrid_patterns`'
pattern-level admission state by scanning the visible bars window each
call.

## Why it was needed

Without stateful tracking, the live evaluator fires on patterns that the
backtest's dedup would have blocked via [[concepts/ghost-pattern|ghost
patterns]] — admitted-but-never-traded prior detections that hold the
dedup window. These extras turned out to be LOW-QUALITY signals.

[[phases/phase-4-portfolio-sim]] showed that without 3b, live HYBRID drags
the portfolio Sharpe by **-0.10** instead of lifting it by +0.31.

## Implementation

```python
# strategy/live_cnh_hybrid_short.py
def _admitted_patterns(df, cfg, up_to_idx, dedup_bars):
    """Replay find_hybrid_patterns' admission logic over visible bars."""
    admitted = []
    last_idx = None
    for j in range(start, up_to_idx + 1):
        dt_hit = detect_distribution_top(df, j, cfg)
        icnh_hit = detect_inverse_cnh(df, j, cfg)
        if dt_hit and (last_idx is None or j - last_idx >= dedup_bars):
            admitted.append((j, "DT")); last_idx = j
        elif icnh_hit and (last_idx is None or j - last_idx >= dedup_bars):
            admitted.append((j, "ICNH")); last_idx = j
    return admitted
```

Stateless API (no caller-side state), but internally O(N×50) per call.
Production-fine (~5ms per call, bot fires once per 4h bar). Audit tools
needed refactoring to O(N) precomputed scan.

## Result

- Phase 3 reproduction: 138% → **100%** match on all 4 OOS windows
- Phase 4 Sharpe lift: -0.10 → **+0.258** (live now within 83% of ideal)

## Test invariant

`tests/test_cnh_hybrid_short.py::test_admitted_patterns_match_find_hybrid_patterns`
— locks the live admission output to backtest's. If a future refactor
breaks this, the test catches it before deploy.

## See also

- [[detectors/pattern-dedup]]
- [[concepts/ghost-pattern]]
- Memory: `snapback_hybrid_short_phase3b_pass.md`
