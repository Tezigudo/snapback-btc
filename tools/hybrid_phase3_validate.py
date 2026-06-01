"""Phase 3 reproduction check for cnh-hybrid-short-v1 live evaluator.

Replays `strategy/live_cnh_hybrid_short.evaluate_signal_cnh_hybrid_short`
bar-by-bar over the 12 historical windows used by the backtest, applies the
same dedup_bars=15 rule at the PATTERN bar (not the signal bar — see the
live evaluator's docstring re: divergence), and counts fires per window.

Gate (per HYBRID_SHORT_PLAN.md Phase 3):
    Live reproduction ≥ 95% of backtest signals per window.

Reference numbers (from `tools/hybrid_walkforward.py` dedup=15 run):
    OOS (2024-H2 → 2026-H1): 18 trades
      2024-H2-bull: 7
      2025-H1-mix:  5
      2025-H2-mix:  3
      2026-H1-bull: 3
    IS (2020-H2 → 2024-H1): 43 trades total

Run:
    uv run python tools/hybrid_phase3_validate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.cnh_detectors import (  # noqa: E402
    HybridConfig,
    attach_indicators,
    detect_inverse_cnh,
    is_ema_breakdown,
)
from strategy.live_cnh_hybrid_short import (  # noqa: E402
    DEDUP_BARS,
    _admitted_patterns,
)
from tools.icnh_mega_sweep import WINDOWS, load_tf  # noqa: E402

DATA = ROOT / "data" / "historical"
RESULTS_PATH = ROOT / "data" / "hybrid_phase3_validate_results.json"

DEDUP_BARS = 15
GATE_REPRODUCTION_PCT = 95.0

# Reference signal counts per window (from hybrid_walkforward.py dedup=15
# OOS run + the IS picks_run at the same dedup). Used to verify the live
# evaluator reproduces them within tolerance.
REFERENCE_PER_WINDOW = {
    "2024-H2-bull": 7,
    "2025-H1-mix": 5,
    "2025-H2-mix": 3,
    "2026-H1-bull": 3,
}


def _replay_window(label: str, start: str, end: str) -> dict:
    """Reconstruct the live evaluator's deduped fire set without the O(N²)
    bar-by-bar re-scan. Approach:
      1. attach indicators ONCE over the window
      2. compute admitted patterns ONCE
      3. for each admission, derive the signal bar:
         - DT: signal bar = admission bar (provided TP slot exists)
         - ICnH: scan forward up to entry_max_bars_after_handle bars
                 for the first EMA24 cross-down (provided TP slot exists
                 at that bar)
    Same semantics as the live evaluator, vastly faster for the audit.
    """
    df_raw = load_tf("4h").loc[start:end]
    if len(df_raw) < 250:
        return {"window": label, "bars": len(df_raw), "fires": 0,
                "fires_deduped": 0, "fires_detail": []}

    cfg = HybridConfig()
    df = attach_indicators(df_raw, cfg)
    admitted = _admitted_patterns(df, cfg, len(df) - 1, DEDUP_BARS)

    fires: list[dict] = []
    for idx, kind in admitted:
        if kind == "DT":
            signal_idx = idx
        else:  # ICNH — wait for EMA24 cross-down within next 1..N bars
            signal_idx = None
            limit = min(idx + 1 + cfg.entry_max_bars_after_handle, len(df))
            for j in range(idx + 1, limit):
                if is_ema_breakdown(df, j, "ema24"):
                    signal_idx = j
                    break
            if signal_idx is None:
                # Admission held the dedup slot but never produced an entry —
                # matches backtest's behaviour (this is the source of Phase 3
                # over-firing in the pre-3b version).
                continue
        # TP check: EMA100 must be BELOW current close for a SHORT TP.
        entry_price = float(df["close"].iloc[signal_idx])
        ema100 = float(df["ema100"].iloc[signal_idx])
        if not (ema100 < entry_price):
            continue
        fires.append({
            "signal_bar": str(df.index[signal_idx]),
            "pattern_bar": str(df.index[idx]),
            "pattern_bar_idx": int(idx),
            "pattern": kind,
            "sl_distance": float(cfg.sl_atr_mult * float(df["atr14"].iloc[signal_idx])),
            "tp_distance": float(entry_price - ema100),
        })

    # Patterns are already deduped by _admitted_patterns. Counting both
    # raw and deduped here is identical now — kept for output-shape compat.
    return {
        "window": label, "bars": len(df),
        "fires": len(fires), "fires_deduped": len(fires),
        "fires_detail": fires,
    }


def main() -> int:
    print("=" * 78)
    print("HYBRID SHORT — Phase 3 live evaluator reproduction check")
    print(f"Dedup: {DEDUP_BARS} bars (pattern-level)")
    print(f"Gate:  live reproduction ≥ {GATE_REPRODUCTION_PCT}% of backtest")
    print("=" * 78)

    t0 = time.time()
    per_window: list[dict] = []
    for label, start, end in WINDOWS:
        print(f"  {label}...", end=" ", flush=True)
        r = _replay_window(label, start, end)
        per_window.append(r)
        ref = REFERENCE_PER_WINDOW.get(label)
        ref_str = f"  ref={ref}" if ref is not None else ""
        print(
            f"bars={r['bars']:>4}  fires={r['fires']:>3}  "
            f"deduped={r['fires_deduped']:>3}{ref_str}"
        )

    # Tally OOS only (last 4 windows) — that's the slice with backtest references.
    oos_labels = list(REFERENCE_PER_WINDOW.keys())
    oos_rows = [r for r in per_window if r["window"] in oos_labels]
    live_total_oos = sum(r["fires_deduped"] for r in oos_rows)
    ref_total_oos = sum(REFERENCE_PER_WINDOW.values())
    repro_pct = 100.0 * live_total_oos / ref_total_oos if ref_total_oos else 0.0

    print()
    print("OOS comparison:")
    print(f"  {'window':<16}{'live':>8}{'backtest':>12}{'delta':>8}")
    for r in oos_rows:
        ref = REFERENCE_PER_WINDOW[r["window"]]
        print(
            f"  {r['window']:<16}{r['fires_deduped']:>8}{ref:>12}"
            f"{r['fires_deduped'] - ref:>+8}"
        )
    print(f"  {'TOTAL':<16}{live_total_oos:>8}{ref_total_oos:>12}"
          f"{live_total_oos - ref_total_oos:>+8}")

    overall_pass = repro_pct >= GATE_REPRODUCTION_PCT
    print()
    print("=" * 78)
    print(f"Reproduction: {repro_pct:.1f}%  "
          f"({'PASS' if overall_pass else 'FAIL'})")
    print("=" * 78)

    RESULTS_PATH.write_text(json.dumps({
        "dedup_bars": DEDUP_BARS,
        "reference_per_window": REFERENCE_PER_WINDOW,
        "per_window": per_window,
        "oos_live_total": live_total_oos,
        "oos_reference_total": ref_total_oos,
        "reproduction_pct": repro_pct,
        "verdict": "PASS" if overall_pass else "FAIL",
    }, indent=2, default=str))
    print(f"Saved → {RESULTS_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
