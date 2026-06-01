"""Final tuning pass. Combines best findings + tests a HYBRID detector
(union of distribution-top and loose-ICnH). Pushes for 1.5-3 trades/month
while preserving positive Sharpe.

Run: uv run python tools/icnh_final_tune.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.icnh_mega_sweep import (  # noqa: E402
    Config, load_tf, find_all_patterns, simulate_trades,
    WINDOWS, FRICTION_BPS, _detect_distribution_top, _detect_cnh
)

RESULTS = ROOT / "data" / "final_tune_results.json"


def find_hybrid_patterns(df: pd.DataFrame, dt_cfg: Config, icnh_cfg: Config) -> list[tuple[int, str]]:
    """Returns list of (bar_idx, source) — patterns from either detector,
    dedup'd by closer (smaller bar index wins in ties)."""
    hits: list[tuple[int, str]] = []
    start = max(dt_cfg.uptrend_bars + dt_cfg.chop_bars + 1,
                icnh_cfg.cup_len + icnh_cfg.handle_len,
                200)
    for i in range(start, len(df)):
        dt_hit = _detect_distribution_top(df, i, dt_cfg)
        icnh_hit = _detect_cnh(df, i, icnh_cfg)
        if dt_hit is not None:
            if hits and (i - hits[-1][0]) < dt_cfg.dedup_bars:
                continue
            hits.append((i, "DT"))
        elif icnh_hit is not None:
            if hits and (i - hits[-1][0]) < icnh_cfg.dedup_bars:
                continue
            hits.append((i, "ICNH"))
    return hits


def run_hybrid(tf: str, dedup: int, sl_atr: float, tp_emas: tuple,
               entry_emas: tuple) -> dict:
    """Run the hybrid (DT + ICnH) detector with given knobs."""
    df = load_tf(tf)

    dt_cfg = Config(
        name="hybrid_dt", pattern_type="distribution_top", direction="short", tf=tf,
        uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
        require_chop_at_top=True, breakdown_mode="chop_low_or_ema24",
        sl_atr_mult=sl_atr, regime_sl_mode="off", tp_emas=tp_emas,
        entry_emas=entry_emas, dedup_bars=dedup,
    )
    icnh_cfg = Config(
        name="hybrid_icnh", pattern_type="inverse_cnh", direction="short", tf=tf,
        cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
        handle_max_depth_frac=0.70, peak_tolerance=6,
        entry_emas=entry_emas, sl_atr_mult=sl_atr, regime_sl_mode="off",
        tp_emas=tp_emas, dedup_bars=dedup,
    )

    all_trades: list[dict] = []
    per_window: list[dict] = []

    for label, start, end in WINDOWS:
        sub = df.loc[start:end]
        if len(sub) < 100:
            continue
        # Detect with hybrid logic (union, dedup'd)
        hits = find_hybrid_patterns(sub, dt_cfg, icnh_cfg)

        # For DT patterns, the bar IS the trigger; for ICnH, need EMA breakdown
        # Simulate each separately using the appropriate config
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
                "dt_trades": sum(1 for t in trades if t["pattern"] == "DT"),
                "icnh_trades": sum(1 for t in trades if t["pattern"] == "ICNH"),
                "win_rate": float((nets > 0).mean()),
                "cum": float(np.prod(1.0 + nets) - 1.0),
                "sharpe": float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0,
            })
        all_trades.extend(trades)

    if not all_trades:
        return {"trades": 0, "trades_per_year": 0, "trades_per_month": 0,
                "win_rate": 0, "cum": 0, "sharpe": 0, "per_window": []}
    nets = np.array([t["net_pct"] for t in all_trades])
    n_years = 5.9
    return {
        "config": {"name": f"HYBRID_{tf}_dedup{dedup}_atr{sl_atr}_tp{','.join(tp_emas)}_ent{','.join(entry_emas)}",
                   "tf": tf, "dedup": dedup, "sl_atr": sl_atr, "tp": list(tp_emas),
                   "entry": list(entry_emas)},
        "trades": len(all_trades),
        "dt_count": sum(1 for t in all_trades if t["pattern"] == "DT"),
        "icnh_count": sum(1 for t in all_trades if t["pattern"] == "ICNH"),
        "trades_per_year": len(all_trades) / n_years,
        "trades_per_month": len(all_trades) / (n_years * 12),
        "win_rate": float((nets > 0).mean()),
        "cum": float(np.prod(1.0 + nets) - 1.0),
        "sharpe": float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0,
        "per_window": per_window,
    }


# ============================================================
# CONFIG-BASED single-detector runs (extended push for frequency)
# ============================================================

PUSH_CONFIGS = [
    # Higher-frequency loose ICnH variants
    Config(name="FT_loose_dedup20", pattern_type="inverse_cnh", direction="short", tf="4h",
           cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
           handle_max_depth_frac=0.70, peak_tolerance=6,
           entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
           tp_emas=("ema100",), dedup_bars=20,
           note="Loose ICnH dedup=20"),
    Config(name="FT_loose_dedup8", pattern_type="inverse_cnh", direction="short", tf="4h",
           cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
           handle_max_depth_frac=0.70, peak_tolerance=6,
           entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
           tp_emas=("ema100",), dedup_bars=8,
           note="Loose ICnH dedup=8"),

    # Different DT chop sizes (DT_up16b_4h was the winner)
    Config(name="FT_DT_up16_chop4", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=16, chop_bars=4, min_rise_pct=2.5, max_chop_ratio=0.55,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT up16 chop4"),
    Config(name="FT_DT_up16_chop6", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=16, chop_bars=6, min_rise_pct=2.5, max_chop_ratio=0.55,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT up16 chop6"),
    Config(name="FT_DT_up16_chop10", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=16, chop_bars=10, min_rise_pct=2.5, max_chop_ratio=0.55,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT up16 chop10"),

    # DT_up20+
    Config(name="FT_DT_up20", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=20, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT up20 chop8"),
    Config(name="FT_DT_up24", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=24, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT up24 chop8"),

    # DT with different rise %
    Config(name="FT_DT_up16_rise1.5", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=16, chop_bars=8, min_rise_pct=1.5, max_chop_ratio=0.55,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT up16 rise1.5%"),
    Config(name="FT_DT_up16_rise4", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=16, chop_bars=8, min_rise_pct=4.0, max_chop_ratio=0.55,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT up16 rise4%"),

    # DT with chop_at_top OFF (matches user's image — chop CAN dip a bit)
    Config(name="FT_DT_chop_not_at_top", pattern_type="distribution_top", direction="short", tf="4h",
           uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.65,
           require_chop_at_top=False,
           sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
           entry_emas=("ema24",), note="DT relaxed chop placement"),

    # LONG DT (accumulation bottom) variants
    Config(name="FT_AB_up16", pattern_type="accumulation_bottom", direction="long", tf="4h",
           uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
           sl_atr_mult=2.0, regime_sl_mode="off", tp_emas=("ema200",),
           entry_emas=("ema24",), note="AB long up16"),
    Config(name="FT_AB_up20", pattern_type="accumulation_bottom", direction="long", tf="4h",
           uptrend_bars=20, chop_bars=10, min_rise_pct=3.0, max_chop_ratio=0.55,
           sl_atr_mult=2.0, regime_sl_mode="off", tp_emas=("ema200",),
           entry_emas=("ema24",), note="AB long up20"),

    # Loose-ICnH LONG version with dedup tuning
    Config(name="FT_LONG_loose_dedup10", pattern_type="classic_cnh", direction="long", tf="4h",
           cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
           handle_max_depth_frac=0.70, peak_tolerance=6,
           entry_emas=("ema24",), sl_atr_mult=2.0, regime_sl_mode="off",
           tp_emas=("ema200",), dedup_bars=10,
           note="Loose LONG C&H dedup=10"),
    Config(name="FT_LONG_loose_dedup15", pattern_type="classic_cnh", direction="long", tf="4h",
           cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
           handle_max_depth_frac=0.70, peak_tolerance=6,
           entry_emas=("ema24",), sl_atr_mult=2.0, regime_sl_mode="off",
           tp_emas=("ema200",), dedup_bars=15,
           note="Loose LONG C&H dedup=15"),
    Config(name="FT_LONG_loose_dedup5", pattern_type="classic_cnh", direction="long", tf="4h",
           cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
           handle_max_depth_frac=0.70, peak_tolerance=6,
           entry_emas=("ema24",), sl_atr_mult=2.0, regime_sl_mode="off",
           tp_emas=("ema200",), dedup_bars=5,
           note="Loose LONG C&H dedup=5"),
]


def _run_cfg(c: Config) -> dict:
    from tools.icnh_mega_sweep import run_config
    try:
        return run_config(c)
    except Exception as e:
        return {"config": asdict(c), "trades": 0, "win_rate": 0, "cum": 0,
                "sharpe": 0, "per_window": [], "trades_per_month": 0, "error": str(e)}


def main() -> int:
    print("=== Single-detector PUSH configs ===")
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_run_cfg, c): c for c in PUSH_CONFIGS}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            cfg = futs[fut]
            print(f"  {cfg.name:<30} tr={r['trades']:>4} "
                  f"({r.get('trades_per_month', 0):.1f}/mo)  "
                  f"WR={r['win_rate']*100:>5.1f}% cum={r['cum']*100:>+7.2f}% "
                  f"sh={r['sharpe']:>+5.2f}")

    print("\n=== HYBRID detector (DT + ICnH union) ===")
    hybrid_configs = [
        # 4h variants
        ("4h", 10, 1.5, ("ema100",), ("ema24",)),
        ("4h", 15, 1.5, ("ema100",), ("ema24",)),
        ("4h", 5, 1.5, ("ema100",), ("ema24",)),
        ("4h", 10, 1.5, ("ema100",), ("ema24", "ema50")),
        ("4h", 10, 2.0, ("ema100",), ("ema24",)),
        ("4h", 10, 1.5, ("ema50",), ("ema24",)),
        ("4h", 10, 1.5, ("ema200",), ("ema24",)),
        ("4h", 10, 1.5, ("ema100", "ema200"), ("ema24",)),
        # 1h
        ("1h", 10, 1.5, ("ema100",), ("ema24",)),
    ]
    hybrid_results = []
    for h in hybrid_configs:
        tf, dedup, sl, tp, ent = h
        r = run_hybrid(tf, dedup, sl, tp, ent)
        hybrid_results.append(r)
        cfg = r.get("config", {})
        print(f"  HYBRID {tf} dedup={dedup} atr={sl} tp={tp} ent={ent}:")
        print(f"    trades={r['trades']} ({r.get('trades_per_month', 0):.1f}/mo)  "
              f"WR={r['win_rate']*100:.1f}%  cum={r['cum']*100:+.2f}% "
              f"sh={r['sharpe']:+.2f}  (DT:{r.get('dt_count', 0)}/ICNH:{r.get('icnh_count', 0)})")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s")
    all_results = results + hybrid_results
    RESULTS.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Saved → {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
