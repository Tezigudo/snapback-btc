"""Targeted reproduction of SOL x cnh-hybrid-short Phase-1 gate result.

Reuses the EXACT functions from cross_coin_backtest.py (no parallel harness)
so the numbers are directly comparable to the 2026-05-27 cross-coin run.

Run: uv run python tools/repro_sol_hybrid_short.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.cross_coin_backtest import (  # noqa: E402
    IS_LABELS,
    OOS_LABELS,
    _aggregate,
    _run_cnh_hybrid_per_window,
)


def _verdict(oos: dict) -> str:
    n = int(oos.get("n_windows", 0))
    if n == 0:
        return "n/a (no OOS data)"
    ok = (
        float(oos.get("cum_ret_pct", 0)) > 0
        and int(oos.get("positive_windows", 0)) >= n / 2
        and float(oos.get("worst_window_pct", 0)) > -15.0
    )
    return "PASS" if ok else "FAIL"


def main() -> int:
    for coin in ("SOL", "BTC"):  # BTC as a known-good control
        per_window = _run_cnh_hybrid_per_window(coin)
        is_a = _aggregate(per_window, IS_LABELS)
        oos_a = _aggregate(per_window, OOS_LABELS)
        print(f"\n===== {coin} x cnh-hybrid-short =====")
        print(
            f"  IS : cum={is_a['cum_ret_pct']:+.2f}%  "
            f"{is_a['positive_windows']}/{is_a['n_windows']} pos  "
            f"trades={is_a['trades']}  worst={is_a['worst_window_pct']:+.2f}%"
        )
        print(
            f"  OOS: cum={oos_a['cum_ret_pct']:+.2f}%  "
            f"{oos_a['positive_windows']}/{oos_a['n_windows']} pos  "
            f"trades={oos_a['trades']}  worst={oos_a['worst_window_pct']:+.2f}%  "
            f"medSharpe={oos_a['median_sharpe']:+.2f}"
        )
        print(f"  VERDICT: {_verdict(oos_a)}")
        print("  per-OOS-window:")
        for w in per_window:
            if w.get("window") in OOS_LABELS:
                tag = " (no_data)" if w.get("no_data") else ""
                print(
                    f"    {w['window']}: ret={w.get('ret_pct', 0.0):+.2f}%  "
                    f"trades={w.get('trades', 0)}  "
                    f"wr={w.get('win_rate_pct', 0.0):.0f}%{tag}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
