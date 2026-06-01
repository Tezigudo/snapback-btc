"""Phase 1 walk-forward audit for cnh-hybrid-short-v1.

Splits the 12-window backtest universe into:
  - IS  = windows 1..8  (2020-H2 → 2024-H1-bull)
  - OOS = windows 9..12 (2024-H2-bull → 2026-H1-bull)

For each dedup ∈ {5, 10, 15} we sweep a small knob grid on IS only, pick the
best by IS Sharpe, and report OOS metrics for that pick. Gate per
HYBRID_SHORT_PLAN.md Phase 1:

    OOS 4-window cum > 0  AND  worst OOS window > -15%  AND  OOS trades >= 4

for at least one dedup. If all three dedup variants fail, the strategy is
abandoned and we pivot to LONG_OPTIMAL (memory: snapback-cnh-long-candidate).

This is a single-split fit/test, not a rolling walk-forward. One split is
sufficient for a Phase 1 go/no-go; rolling refits are Phase 2 territory.

Run:
    uv run python tools/hybrid_walkforward.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the in-house detectors / sim — exact same math as the existing
# `data/final_tune_results.json` runs, so PASS here means OOS lift on top of
# the already-validated in-sample picture, not a different metric.
from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
from tools.icnh_mega_sweep import (  # noqa: E402
    Config,
    FRICTION_BPS,
    WINDOWS,
    load_tf,
    simulate_trades,
)

RESULTS_PATH = ROOT / "data" / "hybrid_walkforward_results.json"

IS_LABELS = {w[0] for w in WINDOWS[:8]}   # windows 1..8
OOS_LABELS = {w[0] for w in WINDOWS[8:]}  # windows 9..12

# Knob grid swept on IS — small and explicit. No bayesian/random search:
# anything fancier here is a free way to overfit on 8 windows.
KNOB_GRID = [
    {"sl_atr": 1.5, "tp_emas": ("ema100",)},
    {"sl_atr": 1.5, "tp_emas": ("ema50",)},
    {"sl_atr": 1.5, "tp_emas": ("ema200",)},
    {"sl_atr": 1.5, "tp_emas": ("ema100", "ema200")},
    {"sl_atr": 2.0, "tp_emas": ("ema100",)},
    {"sl_atr": 2.0, "tp_emas": ("ema50",)},
]
DEDUPS = [5, 10, 15]
ENTRY_EMAS = ("ema24",)   # only entry_emas value that has produced positive cum
TF = "4h"

# Gate thresholds (per HYBRID_SHORT_PLAN.md)
GATE_MIN_OOS_CUM = 0.0
GATE_MIN_OOS_WORST_WINDOW = -0.15
GATE_MIN_OOS_TRADES = 4


def _run_hybrid_on_subset(
    df: pd.DataFrame,
    dedup: int,
    sl_atr: float,
    tp_emas: tuple,
    entry_emas: tuple,
    window_filter: set[str],
) -> dict:
    """Run the HYBRID detector but only score the windows whose label is in
    `window_filter`. Detector params match icnh_final_tune.run_hybrid exactly.
    """
    dt_cfg = Config(
        name="hybrid_dt", pattern_type="distribution_top", direction="short", tf=TF,
        uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
        require_chop_at_top=True, breakdown_mode="chop_low_or_ema24",
        sl_atr_mult=sl_atr, regime_sl_mode="off", tp_emas=tp_emas,
        entry_emas=entry_emas, dedup_bars=dedup,
    )
    icnh_cfg = Config(
        name="hybrid_icnh", pattern_type="inverse_cnh", direction="short", tf=TF,
        cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
        handle_max_depth_frac=0.70, peak_tolerance=6,
        entry_emas=entry_emas, sl_atr_mult=sl_atr, regime_sl_mode="off",
        tp_emas=tp_emas, dedup_bars=dedup,
    )

    per_window: list[dict] = []
    all_trades: list[dict] = []

    for label, start, end in WINDOWS:
        if label not in window_filter:
            continue
        sub = df.loc[start:end]
        if len(sub) < 100:
            continue
        hits = find_hybrid_patterns(sub, dt_cfg, icnh_cfg)
        dt_idxs = [h for h, src in hits if src == "DT"]
        icnh_idxs = [h for h, src in hits if src == "ICNH"]
        dt_trades = simulate_trades(sub, dt_idxs, dt_cfg, label)
        for t in dt_trades:
            t["pattern"] = "DT"
        icnh_trades = simulate_trades(sub, icnh_idxs, icnh_cfg, label)
        for t in icnh_trades:
            t["pattern"] = "ICNH"
        trades = dt_trades + icnh_trades

        if trades:
            nets = np.array([t["net_pct"] for t in trades])
            per_window.append({
                "window": label,
                "trades": len(trades),
                "win_rate": float((nets > 0).mean()),
                "cum": float(np.prod(1.0 + nets) - 1.0),
                "sharpe": (
                    float(nets.mean() / nets.std() * np.sqrt(250))
                    if nets.std() > 0 else 0.0
                ),
            })
        else:
            # Track zero-trade windows so the OOS gate sees them.
            per_window.append({
                "window": label, "trades": 0, "win_rate": 0.0,
                "cum": 0.0, "sharpe": 0.0,
            })
        all_trades.extend(trades)

    if not all_trades:
        return {
            "trades": 0, "win_rate": 0.0, "cum": 0.0, "sharpe": 0.0,
            "per_window": per_window,
        }
    nets = np.array([t["net_pct"] for t in all_trades])
    return {
        "trades": len(all_trades),
        "win_rate": float((nets > 0).mean()),
        "cum": float(np.prod(1.0 + nets) - 1.0),
        "sharpe": (
            float(nets.mean() / nets.std() * np.sqrt(250))
            if nets.std() > 0 else 0.0
        ),
        "per_window": per_window,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+6.1f}%"


def _summarise(per_window: list[dict]) -> tuple[float, int, int]:
    """Returns (worst_window_cum, total_trades, num_windows_negative)."""
    if not per_window:
        return 0.0, 0, 0
    cums = [w["cum"] for w in per_window]
    worst = min(cums)
    total = sum(w["trades"] for w in per_window)
    n_neg = sum(1 for c in cums if c < 0)
    return worst, total, n_neg


def main() -> int:
    print("=" * 78)
    print("HYBRID SHORT — Phase 1 walk-forward")
    print(f"IS windows:  {sorted(IS_LABELS)}")
    print(f"OOS windows: {sorted(OOS_LABELS)}")
    print(f"Friction:    {FRICTION_BPS} bps (existing in mega_sweep)")
    print("=" * 78)

    df = load_tf(TF)
    t0 = time.time()

    summary = []

    for dedup in DEDUPS:
        print(f"\n=== dedup={dedup} ===")
        is_runs = []
        for knobs in KNOB_GRID:
            r = _run_hybrid_on_subset(
                df, dedup=dedup, **knobs, entry_emas=ENTRY_EMAS,
                window_filter=IS_LABELS,
            )
            r["knobs"] = knobs
            is_runs.append(r)
            tp_str = ",".join(knobs["tp_emas"])
            print(
                f"  IS: sl={knobs['sl_atr']} tp={tp_str:<15} "
                f"trades={r['trades']:>3}  WR={r['win_rate']*100:>5.1f}%  "
                f"cum={_fmt_pct(r['cum'])}  Sh={r['sharpe']:>+5.2f}"
            )

        # Pick the knob set with the best IS Sharpe (tie-break by cum).
        is_runs.sort(key=lambda r: (r["sharpe"], r["cum"]), reverse=True)
        best = is_runs[0]
        bk = best["knobs"]
        print(
            f"  → IS pick: sl={bk['sl_atr']} tp={bk['tp_emas']} "
            f"(IS Sharpe {best['sharpe']:+.2f}, cum {_fmt_pct(best['cum'])})"
        )

        # Score the IS pick on OOS.
        oos = _run_hybrid_on_subset(
            df, dedup=dedup, **bk, entry_emas=ENTRY_EMAS,
            window_filter=OOS_LABELS,
        )
        worst_w, n_trades_oos, n_neg = _summarise(oos["per_window"])
        print(
            f"  OOS: trades={oos['trades']:>3}  WR={oos['win_rate']*100:>5.1f}% "
            f"cum={_fmt_pct(oos['cum'])}  Sh={oos['sharpe']:+.2f}  "
            f"worst-window={_fmt_pct(worst_w)}  neg-windows={n_neg}/{len(oos['per_window'])}"
        )
        for w in oos["per_window"]:
            print(
                f"    {w['window']:<14} trades={w['trades']:>3}  "
                f"WR={w['win_rate']*100:>5.1f}%  cum={_fmt_pct(w['cum'])}"
            )

        gate_pass = (
            oos["cum"] > GATE_MIN_OOS_CUM
            and worst_w > GATE_MIN_OOS_WORST_WINDOW
            and n_trades_oos >= GATE_MIN_OOS_TRADES
        )
        summary.append({
            "dedup": dedup,
            "is_pick": bk,
            "is_sharpe": best["sharpe"],
            "is_cum": best["cum"],
            "oos_trades": oos["trades"],
            "oos_cum": oos["cum"],
            "oos_sharpe": oos["sharpe"],
            "oos_worst_window": worst_w,
            "oos_negative_windows": n_neg,
            "oos_per_window": oos["per_window"],
            "gate_pass": gate_pass,
        })
        print(
            f"  → Gate (OOS cum > 0 AND worst > -15% AND trades >= 4): "
            f"{'PASS' if gate_pass else 'FAIL'}"
        )

    print("\n" + "=" * 78)
    print("PHASE 1 VERDICT")
    print("=" * 78)
    print(
        f"{'dedup':<8}{'IS Sh':>8}{'IS cum':>10}{'OOS n':>8}"
        f"{'OOS cum':>10}{'worst-w':>10}{'gate':>8}"
    )
    print("-" * 78)
    for s in summary:
        print(
            f"{s['dedup']:<8}{s['is_sharpe']:>+8.2f}{_fmt_pct(s['is_cum']):>10}"
            f"{s['oos_trades']:>8}{_fmt_pct(s['oos_cum']):>10}"
            f"{_fmt_pct(s['oos_worst_window']):>10}"
            f"{('PASS' if s['gate_pass'] else 'FAIL'):>8}"
        )

    any_pass = any(s["gate_pass"] for s in summary)
    print()
    if any_pass:
        winners = [s for s in summary if s["gate_pass"]]
        winners.sort(key=lambda s: (s["oos_cum"], s["oos_sharpe"]), reverse=True)
        w = winners[0]
        print(
            f"PASS — at least one dedup survives OOS. Best: dedup={w['dedup']}, "
            f"OOS cum={_fmt_pct(w['oos_cum'])}, Sharpe {w['oos_sharpe']:+.2f}."
        )
        print("Proceed to Phase 2 (friction stress + sizing reality).")
    else:
        print(
            "FAIL — no dedup variant passes the OOS gate. Abandon "
            "cnh-hybrid-short-v1; pivot to LONG_OPTIMAL (grid_results.json)."
        )

    RESULTS_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → {RESULTS_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
