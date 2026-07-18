---
tags: [detector]
---

# Inverse Cup-and-Handle (ICnH) detector

The second pattern detector in [[HYBRID-short-strategy]]. Detects an
inverted (concave-down) parabolic cup followed by a small handle pullback.
A bearish pattern — the cup top is where buyers got exhausted, the handle
is a relief rally before the breakdown.

## Defining params (locked)

| | value |
|---|--:|
| cup_len | 20 |
| handle_len | 4 |
| min_r2 | 0.50 (parabola fit quality) |
| min_cup_depth_atr | 1.0 |
| handle_max_depth_frac | 0.70 |
| peak_tolerance | 6 (bars off-centre) |
| entry_max_bars_after_handle | 8 |

## Entry trigger

Unlike DT (where the pattern bar IS the entry), ICnH waits for an EMA(24)
cross-down within `entry_max_bars_after_handle` bars after the pattern.

## Implementation

`detect_inverse_cnh` in `../../strategy/cnh_detectors.py`. Quadratic fit
via `np.polyfit`, R² + peak position + lip-depth checks.

## Regime fit

Per the [[decisions/keep-both-detectors|DT vs ICnH ablation experiment]], ICnH dominates
**chop/transition years**:

- 2023 recovery: ICnH-only +12.4% vs DT-only +1.4%
- 2024 mixed: ICnH-only +14.8% vs DT-only +0.7%
- ICnH mean per-trade edge: +64.6 bps (smaller but more frequent)
- ICnH win rate: 64.8%

## Why "inverse"?

Standard C&H is bullish (concave-up cup → continuation higher). Inverse
flips the geometry — concave-down cup, small handle, breakdown short.

## See also

- [[detectors/distribution-top]] — complementary detector
- [[detectors/pattern-dedup]] — how the two are merged
- [[concepts/regime-complementarity]]
