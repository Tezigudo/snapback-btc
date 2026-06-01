"""Regime-aware hybrid: trade the cup-and-handle pattern in WHICHEVER direction
the market regime favors. If price > EMA(200)+5% → only take LONG (classic C&H).
If price < EMA(200)-5% → only take SHORT (inverse C&H). Otherwise skip.

This combines the best of both worlds: avoids fighting the trend.

Run: uv run python tools/icnh_regime_aware.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.icnh_grid_sweep import Config, load_tf, find_all_patterns, simulate_trades, WINDOWS  # noqa: E402


def run_regime_aware(uptrend_pct: float = 5.0) -> dict:
    """Run regime-aware version on 4h."""
    df = load_tf("4h")

    long_cfg = Config(
        name="REGIME_LONG", pattern_type="classic_cnh", direction="long", tf="4h",
        cup_len=20, handle_len=5, min_r2=0.70, min_cup_depth_atr=2.5,
        entry_emas=("ema24",), sl_atr_mult=2.0, regime_sl_mode="off",
        tp_emas=("ema200",), require_uptrend_for_short=False,
    )
    short_cfg = Config(
        name="REGIME_SHORT", pattern_type="inverse_cnh", direction="short", tf="4h",
        cup_len=20, handle_len=5, min_r2=0.70, min_cup_depth_atr=2.5,
        entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
        tp_emas=("ema100", "ema200"), require_uptrend_for_short=False,
    )

    all_trades: list[dict] = []
    per_window: list[dict] = []

    for label, start, end in WINDOWS:
        sub = df.loc[start:end]
        if len(sub) < 100:
            continue

        # Long patterns
        pats_long = find_all_patterns(sub, long_cfg)
        trades_long_raw = simulate_trades(sub, pats_long, long_cfg, label)
        # Short patterns
        pats_short = find_all_patterns(sub, short_cfg)
        trades_short_raw = simulate_trades(sub, pats_short, short_cfg, label)

        kept: list[dict] = []
        # For each trade, gate by regime at entry
        for t in trades_long_raw:
            entry_ts = t["entry_ts"]
            row = sub.loc[entry_ts] if entry_ts in sub.index else None
            if row is None:
                continue
            close, ema200 = float(row["close"]), float(row["ema200"])
            if not np.isfinite(ema200):
                continue
            # Only take LONG if price is well ABOVE EMA200 (clear uptrend)
            if close >= ema200 * (1.0 + uptrend_pct / 100):
                kept.append(dict(t, regime="long_in_uptrend"))

        for t in trades_short_raw:
            entry_ts = t["entry_ts"]
            row = sub.loc[entry_ts] if entry_ts in sub.index else None
            if row is None:
                continue
            close, ema200 = float(row["close"]), float(row["ema200"])
            if not np.isfinite(ema200):
                continue
            # Only take SHORT if price is well BELOW EMA200 (clear downtrend)
            if close <= ema200 * (1.0 - uptrend_pct / 100):
                kept.append(dict(t, regime="short_in_downtrend"))

        kept.sort(key=lambda t: t["entry_ts"])
        if kept:
            nets = np.array([t["net_pct"] for t in kept])
            per_window.append({
                "window": label,
                "trades": len(kept),
                "long_trades": sum(1 for t in kept if t["regime"] == "long_in_uptrend"),
                "short_trades": sum(1 for t in kept if t["regime"] == "short_in_downtrend"),
                "win_rate": float((nets > 0).mean()),
                "cum": float(np.prod(1.0 + nets) - 1.0),
            })
        all_trades.extend(kept)

    if not all_trades:
        return {"trades": 0, "long_trades": 0, "short_trades": 0,
                "win_rate": 0, "cum": 0, "sharpe": 0, "per_window": []}
    nets = np.array([t["net_pct"] for t in all_trades])
    n_long = sum(1 for t in all_trades if t["regime"] == "long_in_uptrend")
    n_short = sum(1 for t in all_trades if t["regime"] == "short_in_downtrend")
    return {
        "trades": len(all_trades),
        "long_trades": n_long,
        "short_trades": n_short,
        "win_rate": float((nets > 0).mean()),
        "cum": float(np.prod(1.0 + nets) - 1.0),
        "sharpe": float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0,
        "per_window": per_window,
        "trades_detail": all_trades,
    }


if __name__ == "__main__":
    for pct in [3.0, 5.0, 8.0]:
        print(f"\n=== Regime-aware C&H (threshold ±{pct}% from EMA200) ===")
        r = run_regime_aware(pct)
        print(f"  trades:  {r['trades']}  (long: {r['long_trades']}, short: {r['short_trades']})")
        print(f"  win rate: {r['win_rate']*100:.1f}%")
        print(f"  cum:     {r['cum']*100:+.2f}%")
        print(f"  sharpe:  {r['sharpe']:+.2f}")
        print(f"\n  per-window:")
        for w in r['per_window']:
            print(f"    {w['window']:<20} trades={w['trades']:>3} "
                  f"(L:{w['long_trades']:>2}/S:{w['short_trades']:>2}) "
                  f"WR={w['win_rate']*100:>5.1f}% cum={w['cum']*100:>+7.2f}%")

    # Save the 5% result as JSON for the report builder
    result = run_regime_aware(5.0)
    out = ROOT / "data" / "regime_aware_result.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved 5% result → {out}")
