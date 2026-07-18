"""
Re-baseline divergence-v1 across 5 OOS windows, 3 variants.
Saves results to reports/divergence_v1_postfix_100k.json
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_strategy_experiment import run
from strategy.signals_divergence import DivergenceV1

WINDOWS = {
    "2022_H1": ("2022-01-01", "2022-06-30"),
    "2023_H1": ("2023-01-01", "2023-06-30"),
    "2024_H1": ("2024-01-01", "2024-06-30"),
    "2024_H2": ("2024-07-01", "2024-12-31"),
    "2025_H1": ("2025-01-01", "2025-06-30"),
}

CASH = 100_000.0

# Variant configs — overrides on top of the class defaults.
# all_fixes: use all class defaults (already have Fix 1+2+3 baked in)
# ablation_a: Fix 3 only (revert Fix 1+2 by disabling OBV slope and using old confirmation)
# ablation_b: Fix 1+2 only (revert Fix 3 to old unsafe defaults)

VARIANTS = {
    "all_fixes": {},   # all class defaults — fixes 1+2+3 active

    # ablation_a: Fix 3 only (live-safe defaults). Fix 1 + Fix 2 reverted.
    # Fix 1 revert: use_obv_divergence=False (no slope gate, OBV check disabled).
    #   Note: disabling OBV is a proxy for reverting to cumulative-OBV; in
    #   practice cumulative-OBV was near-tautological so disabling ≈ no filter.
    # Fix 2 revert: strengthened_confirmation=False → original close > high[b2].
    # Fix 3 stays: leverage=5, trend_filter_enabled=True, rsi zones 30/70 (class defaults).
    "ablation_a": {
        "use_obv_divergence": False,
        "strengthened_confirmation": False,
    },

    # ablation_b: Fix 1 + Fix 2 only. Fix 3 reverted to old unsafe defaults.
    # Fix 1 stays: use_obv_divergence=True (OBV slope check, class default).
    # Fix 2 stays: strengthened_confirmation=True (class default).
    # Fix 3 reverted: old leverage, old RSI zones, trend filter off, veto disabled.
    "ablation_b": {
        "trend_filter_enabled": False,
        "leverage": 20,
        "rsi_oversold_zone": 35.0,
        "rsi_overbought_zone": 65.0,
        "atr_close_ratio_veto": 9999.0,
    },
}


def run_window(config: dict, start: str, end: str) -> dict:
    try:
        return run(config=config, start=start, end=end, strategy_class=DivergenceV1, cash=CASH)
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return {"trades": 0, "total_return_pct": 0.0, "win_rate_pct": 0.0,
                "max_dd_pct": 0.0, "sharpe": 0.0, "equity_final": CASH,
                "config_applied": config, "error": str(exc)}


def summarize(windows_results: dict) -> dict:
    rets = [r["total_return_pct"] for r in windows_results.values()]
    compounded = 1.0
    for r in rets:
        compounded *= (1.0 + r / 100.0)
    compounded_pct = round((compounded - 1.0) * 100.0, 2)
    worst = round(min(rets), 2)
    wins = sum(1 for r in rets if r > 0)
    trades = sum(r["trades"] for r in windows_results.values())
    wrs = [r["win_rate_pct"] for r in windows_results.values() if r["trades"] > 0]
    avg_wr = round(sum(wrs) / len(wrs), 1) if wrs else 0.0
    return {
        "compounded_return_pct": compounded_pct,
        "worst_window_pct": worst,
        "windows_positive": wins,
        "windows_total": 5,
        "total_trades": trades,
        "avg_win_rate_pct": avg_wr,
    }


results = {"strategy": "divergence-v1-postfix", "cash": CASH, "variants": {}}

for variant_name, config in VARIANTS.items():
    print(f"\n=== Variant: {variant_name} ===", file=sys.stderr)
    windows_out = {}
    for win_name, (start, end) in WINDOWS.items():
        print(f"  window {win_name} ({start} → {end})...", file=sys.stderr)
        r = run_window(config, start, end)
        windows_out[win_name] = r
        print(f"    trades={r['trades']} ret={r['total_return_pct']:+.2f}% wr={r['win_rate_pct']:.1f}%", file=sys.stderr)
    results["variants"][variant_name] = {
        "windows": windows_out,
        "summary": summarize(windows_out),
    }

# Add pre-fix reference from the spec
results["variants"]["pre_fix_reference"] = {
    "summary": {
        "compounded_return_pct": -16.58,
        "worst_window_pct": -6.53,
        "windows_positive": 0,
        "windows_total": 5,
        "total_trades": 44,
        "avg_win_rate_pct": 15.1,
    }
}

out_path = ROOT / "reports" / "divergence_v1_postfix_100k.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {out_path}", file=sys.stderr)
print(json.dumps(results, indent=2))
