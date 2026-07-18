"""
Run the 15-backtest divergence final verdict at $1M cash.

v1: DivergenceV1 (5 windows)
v2: DivergenceV2 (5 windows)
v2_loose: DivergenceV2Loose (5 windows)

Then compute PSR for each, produce verdict JSON.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_strategy_experiment import run
from tools.aggregate import (
    build_canonical_block,
    equity_impact_returns,
    legacy_stitched_psr,
)
from tools.psr_eval import compute_psr
from strategy.signals_divergence import DivergenceV1
from strategy.signals_divergence_v2 import DivergenceV2, DivergenceV2Loose

import numpy as np

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]

CASH = 1_000_000.0

STRATEGIES = {
    "v1":       (DivergenceV1, {}),
    "v2":       (DivergenceV2, {}),
    "v2_loose": (DivergenceV2Loose, {}),
}

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def run_strategy_windows(strat_key: str, strat_class, config: dict) -> dict:
    windows_results = {}
    all_trades_pnl = []          # legacy stitched per-trade ReturnPct% union
    per_window_canon = []        # canonical per-window dicts for build_canonical_block

    for window_name, start, end in WINDOWS:
        print(f"\n[verdict] Running {strat_key} on {window_name} ...", flush=True)
        window_pnl: list[float] = []
        eq_impact: list[float] = []
        try:
            result = run(
                config=config,
                start=start,
                end=end,
                strategy_class=strat_class,
                cash=CASH,
            )
            trades_df = result.pop("_trades_df", None)
            if trades_df is not None and len(trades_df) > 0:
                for col in ["ReturnPct", "PnL", "return_pct", "pnl_pct"]:
                    if col in trades_df.columns:
                        pnl = trades_df[col].dropna().values.astype(float)
                        if col == "ReturnPct":
                            pnl = pnl * 100.0
                        window_pnl = pnl.tolist()
                        all_trades_pnl.extend(window_pnl)
                        break
                # Sizing-aware equity-impact returns (PnL / equity-at-entry %)
                # within this single contiguous window. Feeds psr_per_window.
                if "PnL" in trades_df.columns:
                    eq_impact = equity_impact_returns(
                        {"_trades": trades_df}, cash=CASH
                    ).tolist()
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            result = {
                "trades": 0, "total_return_pct": 0.0, "win_rate_pct": 0.0,
                "max_dd_pct": 0.0, "sharpe": 0.0, "equity_final": CASH,
                "config_applied": config,
            }
            result.pop("_trades_df", None)

        windows_results[window_name] = result
        per_window_canon.append({
            "label":             window_name,
            # v2 headline = engine equity-curve Return% (sizing-aware).
            "return_pct":        result["total_return_pct"],
            "trades":            result["trades"],
            "pnl_pct":           window_pnl,
            "eq_impact_pnl_pct": eq_impact,
        })
        n = result["trades"]
        ret = result["total_return_pct"]
        wr = result["win_rate_pct"]
        print(f"  {window_name}: {n} trades, {ret:+.2f}% return, {wr:.1f}% WR", flush=True)

    return windows_results, all_trades_pnl, per_window_canon


def compute_summary(windows_results: dict) -> dict:
    rets = [v["total_return_pct"] for v in windows_results.values()]
    trades_per_window = [v["trades"] for v in windows_results.values()]
    win_rates = [v["win_rate_pct"] for v in windows_results.values() if v["trades"] > 0]

    total_trades = sum(trades_per_window)
    windows_positive = sum(1 for r in rets if r > 0)

    # Compounded return: product of (1 + r/100) across all 5 windows
    compounded = 1.0
    for r in rets:
        compounded *= (1.0 + r / 100.0)
    compounded_pct = (compounded - 1.0) * 100.0

    worst_window_pct = min(rets)
    avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0

    return {
        "compounded_return_pct": round(compounded_pct, 4),
        "worst_window_pct": round(worst_window_pct, 4),
        "windows_positive": windows_positive,
        "windows_total": 5,
        "total_trades": total_trades,
        "avg_win_rate_pct": round(avg_win_rate, 4),
    }


def verdict(summary: dict, psr_result: dict) -> str:
    c = summary["compounded_return_pct"]
    wins = summary["windows_positive"]
    total_trades = summary["total_trades"]
    psr = psr_result.get("psr_vs_hurdle", 0.0)
    min_trl = psr_result.get("min_trl", int(1e9))

    if c <= 0 or wins < 2 or (psr < 0.5 and total_trades >= min_trl):
        return "shelf"
    if c > 0 and wins >= 3 and psr > 0.95 and total_trades >= min_trl:
        return "revive"
    # marginal: c>0 and >=2 wins and PSR in [0.5, 0.95]
    if c > 0 and wins >= 2 and 0.5 <= psr <= 0.95:
        return "marginal"
    # Anything else that doesn't fit cleanly
    return "shelf"


def main():
    comparison = {}
    verdict_per_strat = {}

    for strat_key, (strat_cls, config) in STRATEGIES.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Strategy: {strat_key}", flush=True)
        print(f"{'='*60}", flush=True)

        windows_results, all_pnl, per_window_canon = run_strategy_windows(
            strat_key, strat_cls, config
        )
        summary = compute_summary(windows_results)

        # CANONICAL (v2) dual-emit block. Headline PSR = window-level
        # psr_walkforward (n == n_windows), which defeats the N-inflation of the
        # old stitched per-trade PSR that produced the PSR=0.36 verdict.
        canon = build_canonical_block(per_window_canon)
        psr_result = canon["psr_walkforward"]  # CANONICAL headline
        # LEGACY stitched-per-trade PSR — observability sidecar only.
        legacy_psr = legacy_stitched_psr(per_window_canon)

        comparison[strat_key] = {
            "windows": windows_results,
            "summary": summary,
            "psr": psr_result,                    # canonical headline
            "legacy_psr_stitched": legacy_psr,    # observability only
            "canonical": canon,
            "aggregation_method": canon["aggregation_method"],
        }
        verdict_per_strat[strat_key] = verdict(summary, psr_result)

        print(f"\n  Summary: {summary}", flush=True)
        print(f"  PSR (canonical psr_walkforward): {psr_result}", flush=True)
        print(f"  legacy_stitched PSR (observability, N-inflated): "
              f"psr_vs_hurdle={legacy_psr.get('psr_vs_hurdle')} "
              f"n_trades={legacy_psr.get('n_trades')}", flush=True)
        print(f"  Verdict: {verdict_per_strat[strat_key]}", flush=True)

    # Determine best
    revive_strats = [k for k, v in verdict_per_strat.items() if v == "revive"]
    marginal_strats = [k for k, v in verdict_per_strat.items() if v == "marginal"]

    if revive_strats:
        # Pick the one with highest compounded return
        best = max(revive_strats, key=lambda k: comparison[k]["summary"]["compounded_return_pct"])
    elif marginal_strats:
        best = max(marginal_strats, key=lambda k: comparison[k]["summary"]["compounded_return_pct"])
    else:
        best = "none"

    all_shelf = all(v == "shelf" for v in verdict_per_strat.values())
    if all_shelf:
        notes = (
            "All three divergence variants (v1, v2, v2-loose) shelved at $1M cash "
            "on BTC 15m OOS windows 2022-2025. Empirical anti-edge. "
            "Recommend pivoting to Volume Profile POC (FUTURE_DIRECTIONS #2) or "
            "AdaptiveTrend (FUTURE_DIRECTIONS #3)."
        )
    else:
        best_summary = comparison[best]["summary"] if best != "none" else {}
        notes = (
            f"Best strategy: {best}. "
            f"Compounded: {best_summary.get('compounded_return_pct', 0):+.2f}%, "
            f"wins: {best_summary.get('windows_positive', 0)}/5, "
            f"trades: {best_summary.get('total_trades', 0)}. "
            f"Recommended next step: extended walk-forward on {best}, then multi-coin."
        )

    output = {
        "cash": CASH,
        "aggregation_method": "v2_equity_curve",
        "comparison": comparison,
        "verdict_per_strategy": verdict_per_strat,
        "best_strategy_if_any": best,
        "notes": notes,
    }

    out_path = REPORTS_DIR / "divergence_final_verdict_1M.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to {out_path}", flush=True)
    print(f"\nVerdict per strategy: {verdict_per_strat}", flush=True)
    print(f"Best: {best}", flush=True)
    print(f"Notes: {notes}", flush=True)


if __name__ == "__main__":
    main()
