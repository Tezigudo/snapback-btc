"""
One-shot rebaseline script: runs divergence-v1 and adx-dual-regime-v1 at $100k
across 5 OOS windows and saves results to reports/.

Run from repo root:
    python tools/_rebaseline_100k.py

Outputs:
    reports/divergence_v1_rebaseline_100k.json
    reports/adx_dual_regime_baseline_100k.json
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_strategy_experiment import run  # noqa: E402
from strategy.signals_divergence import DivergenceV1  # noqa: E402
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
    """Aggregate window results into a summary block.

    compounded_return_pct = (∏(1 + r_i/100) − 1) × 100
    """
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


def run_windows(strategy_cls, config: dict, label: str) -> dict:
    results = {}
    for window_label, (start, end) in WINDOWS.items():
        print(f"  [{label}] {window_label} ({start} → {end}) ...", file=sys.stderr)
        try:
            r = run(
                config=config,
                start=start,
                end=end,
                strategy_class=strategy_cls,
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

    # ------------------------------------------------------------------ divergence-v1
    print("\n=== divergence-v1 @ $100k ===", file=sys.stderr)
    div_windows = run_windows(DivergenceV1, {}, "divergence-v1")
    div_output = {
        "strategy": "divergence-v1",
        "cash": CASH,
        "windows": div_windows,
        "summary": _summary(div_windows),
    }
    out_path = reports_dir / "divergence_v1_rebaseline_100k.json"
    out_path.write_text(json.dumps(div_output, indent=2))
    print(f"\n[done] Written: {out_path}", file=sys.stderr)
    print(f"  compounded={div_output['summary']['compounded_return_pct']:+.2f}% "
          f"worst={div_output['summary']['worst_window_pct']:+.2f}% "
          f"pos={div_output['summary']['windows_positive']}/5 "
          f"trades={div_output['summary']['total_trades']}",
          file=sys.stderr)

    # ------------------------------------------------------------------ adx-dual-regime
    adx_variants = {
        "default": {},
        "stricter_rsi": {
            "range_rsi_long_threshold": 5.0,
            "range_rsi_short_threshold": 95.0,
        },
        "longer_donchian": {"donchian_period": 40},
        "lower_leverage": {"leverage": 2},
    }

    variants_out = {}
    for variant_name, config in adx_variants.items():
        print(f"\n=== adx-dual-regime-v1 / {variant_name} @ $100k ===", file=sys.stderr)
        print(f"  config={config}", file=sys.stderr)
        w = run_windows(ADXDualRegimeV1, config, f"adx/{variant_name}")
        variants_out[variant_name] = {
            "windows": w,
            "summary": _summary(w),
        }
        s = variants_out[variant_name]["summary"]
        print(f"  compounded={s['compounded_return_pct']:+.2f}% "
              f"worst={s['worst_window_pct']:+.2f}% "
              f"pos={s['windows_positive']}/5 "
              f"trades={s['total_trades']} "
              f"avg_wr={s['avg_win_rate_pct']:.1f}%",
              file=sys.stderr)

    adx_output = {
        "strategy": "adx-dual-regime-v1",
        "cash": CASH,
        "variants": variants_out,
    }
    out_path2 = reports_dir / "adx_dual_regime_baseline_100k.json"
    out_path2.write_text(json.dumps(adx_output, indent=2))
    print(f"\n[done] Written: {out_path2}", file=sys.stderr)


if __name__ == "__main__":
    main()
