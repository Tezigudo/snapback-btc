"""
One-shot re-baseline: adx-dual-regime-v1 with raw_breakout vs retest_entry.
Saves to reports/adx_dual_regime_retest_100k.json.

Run from repo root:
    uv run python tools/_adx_retest_rebaseline.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_strategy_experiment import run  # noqa: E402
from strategy.signals_adx_dual_regime import ADXDualRegimeV1  # noqa: E402

CASH = 100_000.0

WINDOWS = {
    "2022_H1": ("2022-01-01", "2022-06-30"),
    "2023_H1": ("2023-01-01", "2023-06-30"),
    "2024_H1": ("2024-01-01", "2024-06-30"),
    "2024_H2": ("2024-07-01", "2024-12-31"),
    "2025_H1": ("2025-01-01", "2025-06-30"),
}


def _summary(window_results: dict) -> dict:
    compound = 1.0
    worst = float("inf")
    positive = 0
    total_trades = 0
    win_rate_sum = 0.0
    win_rate_count = 0
    for label, r in window_results.items():
        if "error" in r:
            continue
        ret = r.get("total_return_pct", 0.0)
        compound *= (1 + ret / 100.0)
        if ret < worst:
            worst = ret
        if ret > 0:
            positive += 1
        total_trades += r.get("trades", 0)
        wr = r.get("win_rate_pct", 0.0)
        if r.get("trades", 0) > 0:
            win_rate_sum += wr
            win_rate_count += 1

    compounded_pct = round((compound - 1) * 100, 4)
    worst_pct = round(worst, 4) if worst != float("inf") else None
    avg_wr = round(win_rate_sum / win_rate_count, 4) if win_rate_count > 0 else 0.0

    return {
        "compounded_return_pct": compounded_pct,
        "worst_window_pct": worst_pct,
        "windows_positive": positive,
        "windows_total": 5,
        "total_trades": total_trades,
        "avg_win_rate_pct": avg_wr,
    }


def run_windows(config: dict, label: str) -> dict:
    results = {}
    for window_label, (start, end) in WINDOWS.items():
        print(f"  [{label}] {window_label} ({start} → {end}) ...", file=sys.stderr)
        try:
            r = run(
                config=config,
                start=start,
                end=end,
                strategy_class=ADXDualRegimeV1,
                cash=CASH,
            )
            results[window_label] = r
            print(
                f"    trades={r['trades']} ret={r['total_return_pct']:+.2f}% "
                f"wr={r['win_rate_pct']:.1f}% dd={r['max_dd_pct']:.2f}%",
                file=sys.stderr,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"    ERROR: {exc}", file=sys.stderr)
            results[window_label] = {"error": str(exc), "traceback": tb}
    return results


def main() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    variants_config = {
        "raw_breakout_reference": {"use_donchian_retest": False},
        "retest_entry": {},  # default = retest ON
    }

    variants_out = {}
    for name, config in variants_config.items():
        print(f"\n=== adx-dual-regime-v1 / {name} @ $100k ===", file=sys.stderr)
        print(f"  config={config}", file=sys.stderr)
        w = run_windows(config, name)
        variants_out[name] = {
            "windows": w,
            "summary": _summary(w),
        }
        s = variants_out[name]["summary"]
        print(
            f"  compounded={s['compounded_return_pct']:+.2f}% "
            f"worst={s['worst_window_pct']:+.2f}% "
            f"pos={s['windows_positive']}/5 "
            f"trades={s['total_trades']} "
            f"avg_wr={s['avg_win_rate_pct']:.1f}%",
            file=sys.stderr,
        )

    output = {
        "strategy": "adx-dual-regime-v1-retest",
        "cash": CASH,
        "variants": variants_out,
    }
    out_path = reports_dir / "adx_dual_regime_retest_100k.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[done] Written: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
