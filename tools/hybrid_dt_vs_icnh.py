"""Experiment: DT vs ICnH contribution to HYBRID's edge.

The HYBRID detector is the UNION of two patterns:
  - Distribution-Top (DT)  — uptrend → chop → breakdown
  - Inverse Cup-and-Handle (ICnH) — concave-up parabola with handle

Run three variants over the same 2020 → 2026 window with dedup=15:
  1. HYBRID — both detectors (the deployed strategy)
  2. DT-only — disable ICnH (use a cup_len of 10000 so it never matches)
  3. ICnH-only — disable DT (use uptrend_bars 10000 so it never matches)

Compare per-trade WR, R-multiple, cum, and the regime where each detector
dominates. Tells us whether either detector could be dropped to simplify
the strategy.

Run:
    uv run python tools/hybrid_dt_vs_icnh.py
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

from strategy.cnh_detectors import (  # noqa: E402
    HybridConfig,
    attach_indicators,
    detect_distribution_top,
    detect_inverse_cnh,
    is_ema_breakdown,
)
from tools.icnh_mega_sweep import load_tf  # noqa: E402

RESULTS_PATH = ROOT / "data" / "hybrid_dt_vs_icnh_results.json"
SIM_START = pd.Timestamp("2020-01-01", tz="UTC")
SIM_END = pd.Timestamp("2026-05-23", tz="UTC")
TIME_STOP_BARS = 96
FRICTION_BPS_RT = 13.0
DEDUP_BARS = 15


def _admitted_filtered(
    df: pd.DataFrame, cfg: HybridConfig, up_to: int,
    enable_dt: bool, enable_icnh: bool,
) -> list[tuple[int, str]]:
    """Like _admitted_patterns but optionally suppresses one detector."""
    start = max(cfg.cup_len + cfg.handle_len,
                cfg.uptrend_bars + cfg.chop_bars + 1, 200)
    admitted: list[tuple[int, str]] = []
    last_idx = None
    for j in range(start, up_to + 1):
        dt_hit = detect_distribution_top(df, j, cfg) if enable_dt else None
        icnh_hit = detect_inverse_cnh(df, j, cfg) if enable_icnh else None
        if dt_hit is not None:
            if last_idx is None or (j - last_idx) >= DEDUP_BARS:
                admitted.append((j, "DT")); last_idx = j
        elif icnh_hit is not None:
            if last_idx is None or (j - last_idx) >= DEDUP_BARS:
                admitted.append((j, "ICNH")); last_idx = j
    return admitted


def _sim_variant(df: pd.DataFrame, enable_dt: bool, enable_icnh: bool) -> dict:
    cfg = HybridConfig(dedup_bars=DEDUP_BARS)
    admitted = _admitted_filtered(df, cfg, len(df) - 1, enable_dt, enable_icnh)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    ts = df.index

    candidates = []
    for idx, kind in admitted:
        if kind == "DT":
            signal_idx = idx
        else:
            signal_idx = None
            limit = min(idx + 1 + cfg.entry_max_bars_after_handle, len(df))
            for j in range(idx + 1, limit):
                if is_ema_breakdown(df, j, "ema24"):
                    signal_idx = j; break
            if signal_idx is None:
                continue
        atr_v = float(df["atr14"].iloc[signal_idx])
        entry_price = float(df["close"].iloc[signal_idx])
        ema100 = float(df["ema100"].iloc[signal_idx])
        if not (np.isfinite(atr_v) and atr_v > 0 and np.isfinite(ema100)
                and ema100 < entry_price):
            continue
        candidates.append({
            "signal_idx": signal_idx, "pattern": kind,
            "entry_price": entry_price, "atr": atr_v,
            "stop": entry_price + cfg.sl_atr_mult * atr_v,
            "tp": ema100,
        })

    trades = []
    cand_iter = iter(candidates)
    nxt = next(cand_iter, None)
    open_trade = None
    for i in range(250, len(df)):
        if open_trade is not None:
            o = open_trade
            hit = None
            if high[i] >= o["stop"]:
                hit = ("sl", o["stop"])
            elif low[i] <= o["tp"]:
                hit = ("tp", o["tp"])
            elif (i - o["entry_idx"]) >= TIME_STOP_BARS:
                hit = ("time", float(close[i]))
            if hit:
                reason, exit_price = hit
                gross = (o["entry_price"] - exit_price) / o["entry_price"]
                net = gross - FRICTION_BPS_RT / 10_000.0
                trades.append({
                    "entry_ts": str(o["entry_ts"]), "exit_ts": str(ts[i]),
                    "pattern": o["pattern"],
                    "net_pct": net, "exit_reason": reason,
                    "year": pd.Timestamp(o["entry_ts"]).year,
                })
                open_trade = None
        while nxt is not None and nxt["signal_idx"] < i:
            nxt = next(cand_iter, None)
        if nxt is not None and nxt["signal_idx"] == i and open_trade is None:
            open_trade = {
                "entry_ts": ts[i], "entry_idx": i,
                "entry_price": nxt["entry_price"],
                "stop": nxt["stop"], "tp": nxt["tp"],
                "pattern": nxt["pattern"],
            }
            nxt = next(cand_iter, None)

    if not trades:
        return {"trades": 0, "wins": 0, "cum": 0.0, "mean_R": 0.0,
                "win_rate": 0.0, "per_year": {}, "by_pattern": {}}
    nets = np.array([t["net_pct"] for t in trades])
    cum = float(np.prod(1.0 + nets) - 1.0)
    by_year: dict[int, list[float]] = {}
    for t in trades:
        by_year.setdefault(t["year"], []).append(t["net_pct"])
    per_year = {
        y: {
            "n": len(v),
            "cum": float(np.prod(1.0 + np.array(v)) - 1.0),
            "wr": float(sum(1 for x in v if x > 0) / len(v)),
        }
        for y, v in sorted(by_year.items())
    }
    by_pat: dict[str, list[float]] = {"DT": [], "ICNH": []}
    for t in trades:
        by_pat[t["pattern"]].append(t["net_pct"])
    by_pattern = {
        k: {
            "n": len(v),
            "cum": float(np.prod(1.0 + np.array(v)) - 1.0) if v else 0.0,
            "wr": float(sum(1 for x in v if x > 0) / len(v)) if v else 0.0,
            "mean_pct": float(np.mean(v)) if v else 0.0,
        }
        for k, v in by_pat.items()
    }
    return {
        "trades": len(trades),
        "wins": int((nets > 0).sum()),
        "win_rate": float((nets > 0).mean()),
        "cum": cum,
        "mean_pct_per_trade": float(nets.mean()),
        "per_year": per_year,
        "by_pattern": by_pattern,
    }


def main() -> int:
    print("=" * 78)
    print("HYBRID SHORT — DT vs ICnH contribution experiment")
    print(f"Window: {SIM_START.date()} → {SIM_END.date()}")
    print(f"Dedup:  {DEDUP_BARS} bars  Time-stop: {TIME_STOP_BARS} bars")
    print("=" * 78)

    t0 = time.time()
    df_raw = load_tf("4h").loc[SIM_START:SIM_END]
    cfg = HybridConfig(dedup_bars=DEDUP_BARS)
    df = attach_indicators(df_raw, cfg)
    print(f"\nIndicators ready ({len(df)} bars).")

    variants = [
        ("HYBRID (DT+ICnH)", True, True),
        ("DT-only",          True, False),
        ("ICnH-only",        False, True),
    ]
    results = {}
    for label, en_dt, en_ic in variants:
        print(f"\n→ {label}...", flush=True)
        t1 = time.time()
        r = _sim_variant(df, en_dt, en_ic)
        results[label] = r
        print(
            f"  n={r['trades']:>3}  WR={r['win_rate'] * 100:>5.1f}%  "
            f"cum={r['cum'] * 100:>+6.1f}%  "
            f"mean/trade={r['mean_pct_per_trade'] * 10_000:+6.1f} bps  "
            f"[{time.time() - t1:.0f}s]"
        )

    # Side-by-side
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'variant':<22}{'n':>6}{'WR':>9}{'cum':>10}{'/trade':>11}")
    print("-" * 78)
    for label, _, _ in variants:
        r = results[label]
        print(
            f"{label:<22}{r['trades']:>6}{r['win_rate'] * 100:>8.1f}%"
            f"{r['cum'] * 100:>+9.1f}%{r['mean_pct_per_trade'] * 10_000:>+8.1f}bp"
        )

    # Per-pattern breakdown within HYBRID
    h = results["HYBRID (DT+ICnH)"]["by_pattern"]
    print(f"\nWithin HYBRID, per-pattern contribution:")
    print(f"  DT   : n={h['DT']['n']:>3}  WR={h['DT']['wr'] * 100:>5.1f}%  "
          f"cum={h['DT']['cum'] * 100:>+6.1f}%  mean={h['DT']['mean_pct'] * 10_000:+6.1f}bp")
    print(f"  ICNH : n={h['ICNH']['n']:>3}  WR={h['ICNH']['wr'] * 100:>5.1f}%  "
          f"cum={h['ICNH']['cum'] * 100:>+6.1f}%  mean={h['ICNH']['mean_pct'] * 10_000:+6.1f}bp")

    # Per-year contributions of each variant
    print("\nPer-year cum returns:")
    years = sorted(set(y for v in results.values() for y in v["per_year"]))
    print(f"  {'year':<6}", end="")
    for label, _, _ in variants:
        print(f"{label:>22}", end="")
    print()
    print("  " + "-" * (6 + 22 * 3))
    for y in years:
        print(f"  {y:<6}", end="")
        for label, _, _ in variants:
            py = results[label]["per_year"].get(y, {})
            cell = (f"n={py.get('n', 0):>2} {py.get('cum', 0) * 100:>+6.1f}%"
                    if py else "—")
            print(f"{cell:>22}", end="")
        print()

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved → {RESULTS_PATH}")
    print(f"Total: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
