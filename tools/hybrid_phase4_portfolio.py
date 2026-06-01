"""Phase 4 portfolio simulation for cnh-hybrid-short-v1.

Combines three legs at equal capital weight (1/3 each):
  - v1            (from reports/full_history_*_v1_trades.csv)
  - Donchian-v3 cons (from reports/full_history_*_d3cons_trades.csv)
  - HYBRID short  (replayed from the LIVE evaluator, NOT the backtest's
                   filtered trade list — to honour Phase 3's documented
                   over-firing)

Builds daily P&L per leg, computes:
  - Sharpe(v1 + Donchian)              — baseline 2-leg
  - Sharpe(v1 + Donchian + HYBRID)     — proposed 3-leg
  - Sharpe lift = (3-leg) - (2-leg)
  - Daily-return correlation matrix    — for diversification check

Gate (per HYBRID_SHORT_PLAN.md Phase 4):
  - Combined Sharpe lift ≥ 0.1
  - HYBRID-v1 correlation < 0.3
  - HYBRID-Donchian correlation < 0.3

Run:
    uv run python tools/hybrid_phase4_portfolio.py
"""

from __future__ import annotations

import glob
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
    is_ema_breakdown,
)
from strategy.live_cnh_hybrid_short import (  # noqa: E402
    DEDUP_BARS,
    _admitted_patterns,
)
from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
from tools.icnh_mega_sweep import (  # noqa: E402
    Config as ToolsConfig,
    load_tf,
    simulate_trades,
)

RESULTS_PATH = ROOT / "data" / "hybrid_phase4_portfolio_results.json"

# Reuse the latest v1 + Donchian trade exports.
V1_GLOB = "reports/full_history_*_v1_trades.csv"
D3_GLOB = "reports/full_history_*_d3cons_trades.csv"

# Sim window — match the period the v1 + Donchian CSVs cover.
SIM_START = pd.Timestamp("2020-01-01", tz="UTC")
SIM_END = pd.Timestamp("2026-05-23", tz="UTC")

# Live HYBRID sim params (match Phase 1 winner).
TIME_STOP_BARS = 96   # 96 × 4h = 16 days
FRICTION_BPS_RT = 13.0   # live taker + 5 bps slippage stress

# Gate thresholds.
GATE_SHARPE_LIFT_MIN = 0.10
GATE_MAX_CORR = 0.30
TRADING_DAYS_PER_YEAR = 365.25   # crypto trades every day


def _latest(glob_pattern: str) -> Path:
    files = sorted(glob.glob(str(ROOT / glob_pattern)))
    if not files:
        raise FileNotFoundError(glob_pattern)
    return Path(files[-1])


def _load_trade_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["EntryTime", "ExitTime"])
    df = df.rename(columns={"EntryTime": "entry_ts", "ExitTime": "exit_ts",
                            "ReturnPct": "ret"})
    df = df[["entry_ts", "exit_ts", "ret"]].copy()
    # Normalise to UTC.
    for col in ("entry_ts", "exit_ts"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["leg"] = label
    return df


def _sim_hybrid_trades(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Compute live HYBRID trade set without bar-by-bar O(N²) evaluator calls.

    Logic mirrors the (now-stateful) live evaluator exactly:
      1. Compute admitted patterns once (Phase 3b stateful dedup).
      2. For each admission, derive a SIGNAL bar:
          - DT: signal bar = admission bar
          - ICnH: first EMA24 cross-down within entry_max_bars_after_handle
      3. Require TP slot (EMA100 < entry_price for SHORT) at signal bar.
      4. Enforce 'one open position' — drop signals that fire while a
         prior position is still open.
      5. Simulate forward to SL / TP / time-stop / eod.
    """
    cfg = HybridConfig()
    df = attach_indicators(df_4h, cfg)
    admitted = _admitted_patterns(df, cfg, len(df) - 1, DEDUP_BARS)

    # Step 1 → 3: collect candidate (signal_idx, entry_price, sl_dist, tp_dist).
    candidates: list[dict] = []
    for idx, kind in admitted:
        if kind == "DT":
            signal_idx = idx
        else:
            signal_idx = None
            limit = min(idx + 1 + cfg.entry_max_bars_after_handle, len(df))
            for j in range(idx + 1, limit):
                if is_ema_breakdown(df, j, "ema24"):
                    signal_idx = j
                    break
            if signal_idx is None:
                continue
        atr_v = float(df["atr14"].iloc[signal_idx])
        entry_price = float(df["close"].iloc[signal_idx])
        ema100 = float(df["ema100"].iloc[signal_idx])
        if not (np.isfinite(atr_v) and atr_v > 0 and np.isfinite(ema100)
                and ema100 < entry_price):
            continue
        sl_dist = cfg.sl_atr_mult * atr_v
        tp_dist = entry_price - ema100
        candidates.append({
            "signal_idx": signal_idx, "entry_price": entry_price,
            "stop": entry_price + sl_dist, "tp": entry_price - tp_dist,
        })

    # Step 4 + 5: walk forward, enforce one-position rule, simulate.
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    ts = df.index

    trades: list[dict] = []
    cand_iter = iter(candidates)
    next_cand = next(cand_iter, None)
    open_trade: dict | None = None

    for i in range(250, len(df)):
        # Resolve any open trade.
        if open_trade is not None:
            o = open_trade
            hit = None
            if high[i] >= o["stop"]:
                hit = ("sl", o["stop"])
            elif low[i] <= o["tp"]:
                hit = ("tp", o["tp"])
            elif (i - o["entry_idx"]) >= TIME_STOP_BARS:
                hit = ("time", float(close[i]))
            if hit is not None:
                reason, exit_price = hit
                gross = (o["entry_price"] - exit_price) / o["entry_price"]
                net = gross - FRICTION_BPS_RT / 10_000.0
                trades.append({
                    "entry_ts": o["entry_ts"], "exit_ts": ts[i],
                    "entry_price": o["entry_price"], "exit_price": exit_price,
                    "stop": o["stop"], "tp": o["tp"],
                    "gross_pct": gross, "ret": net, "exit_reason": reason,
                    "bars_held": int(i - o["entry_idx"]),
                })
                open_trade = None

        # Open a new trade if a candidate's signal_idx is now and we're flat.
        while next_cand is not None and next_cand["signal_idx"] < i:
            next_cand = next(cand_iter, None)
        if (next_cand is not None and next_cand["signal_idx"] == i
                and open_trade is None):
            open_trade = {
                "entry_ts": ts[i], "entry_idx": i,
                "entry_price": next_cand["entry_price"],
                "stop": next_cand["stop"], "tp": next_cand["tp"],
            }
            next_cand = next(cand_iter, None)

    if open_trade is not None:
        last_idx = len(df) - 1
        exit_price = float(close[last_idx])
        gross = (open_trade["entry_price"] - exit_price) / open_trade["entry_price"]
        net = gross - FRICTION_BPS_RT / 10_000.0
        trades.append({
            "entry_ts": open_trade["entry_ts"], "exit_ts": ts[last_idx],
            "entry_price": open_trade["entry_price"], "exit_price": exit_price,
            "stop": open_trade["stop"], "tp": open_trade["tp"],
            "gross_pct": gross, "ret": net, "exit_reason": "eod",
            "bars_held": int(last_idx - open_trade["entry_idx"]),
        })

    return pd.DataFrame(trades)


def _sim_hybrid_trades_ideal(df_4h: pd.DataFrame) -> pd.DataFrame:
    """The 'ideal' HYBRID — runs the backtest's find_hybrid_patterns (which
    has pattern-level dedup) plus simulate_trades. Represents what live
    behaviour would be IF the live evaluator gained stateful pattern dedup
    matching the backtest. Used for the apples-to-apples Phase 4 verdict
    against the over-firing live evaluator."""
    dt_cfg = ToolsConfig(
        name="hybrid_dt", pattern_type="distribution_top", direction="short", tf="4h",
        uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
        require_chop_at_top=True, breakdown_mode="chop_low_or_ema24",
        sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
        entry_emas=("ema24",), dedup_bars=15,
    )
    icnh_cfg = ToolsConfig(
        name="hybrid_icnh", pattern_type="inverse_cnh", direction="short", tf="4h",
        cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
        handle_max_depth_frac=0.70, peak_tolerance=6,
        entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
        tp_emas=("ema100",), dedup_bars=15,
    )
    hits = find_hybrid_patterns(df_4h, dt_cfg, icnh_cfg)
    dt_idxs = [h for h, src in hits if src == "DT"]
    icnh_idxs = [h for h, src in hits if src == "ICNH"]
    rows: list[dict] = []
    for t in simulate_trades(df_4h, dt_idxs, dt_cfg, "ALL"):
        rows.append({
            "entry_ts": pd.Timestamp(t["entry_ts"]).tz_convert("UTC")
            if pd.Timestamp(t["entry_ts"]).tzinfo else pd.Timestamp(t["entry_ts"], tz="UTC"),
            "exit_ts": pd.Timestamp(t["exit_ts"]).tz_convert("UTC")
            if pd.Timestamp(t["exit_ts"]).tzinfo else pd.Timestamp(t["exit_ts"], tz="UTC"),
            "ret": t["gross_pct"] - FRICTION_BPS_RT / 10_000.0,
            "exit_reason": t["exit_reason"],
        })
    for t in simulate_trades(df_4h, icnh_idxs, icnh_cfg, "ALL"):
        rows.append({
            "entry_ts": pd.Timestamp(t["entry_ts"]).tz_convert("UTC")
            if pd.Timestamp(t["entry_ts"]).tzinfo else pd.Timestamp(t["entry_ts"], tz="UTC"),
            "exit_ts": pd.Timestamp(t["exit_ts"]).tz_convert("UTC")
            if pd.Timestamp(t["exit_ts"]).tzinfo else pd.Timestamp(t["exit_ts"], tz="UTC"),
            "ret": t["gross_pct"] - FRICTION_BPS_RT / 10_000.0,
            "exit_reason": t["exit_reason"],
        })
    return pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)


def _daily_pnl(trades: pd.DataFrame, label: str,
               index: pd.DatetimeIndex) -> pd.Series:
    """Distribute trade returns across their hold period at a daily granularity.

    Approach: each trade contributes ret/D to each calendar day in
    [entry_ts.date(), exit_ts.date()] where D = number of days covered.
    Daily summed per leg. This linearises the daily-P&L so equity-curve
    correlations make sense even when trades span multiple days.
    """
    daily = pd.Series(0.0, index=index, name=label)
    if trades.empty:
        return daily
    for _, row in trades.iterrows():
        e = pd.Timestamp(row["entry_ts"]).tz_convert("UTC").normalize()
        x = pd.Timestamp(row["exit_ts"]).tz_convert("UTC").normalize()
        days = pd.date_range(e, x, freq="1D", tz="UTC")
        days = days.intersection(index)
        if len(days) == 0:
            continue
        per_day = row["ret"] / len(days)
        daily.loc[days] += per_day
    return daily


def _sharpe(daily: pd.Series) -> float:
    if daily.std() == 0 or len(daily) < 2:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _cum(daily: pd.Series) -> float:
    return float(np.prod(1.0 + daily.values) - 1.0)


def main() -> int:
    print("=" * 78)
    print("HYBRID SHORT — Phase 4 portfolio simulation")
    print(f"Window:    {SIM_START.date()} → {SIM_END.date()}")
    print(f"Friction:  {FRICTION_BPS_RT} bps round-trip (live + 5 bps slip)")
    print(f"HYBRID time-stop: {TIME_STOP_BARS} bars ({TIME_STOP_BARS * 4 / 24:.0f} days)")
    print("=" * 78)

    # ---- Load v1 + Donchian trade CSVs ----
    v1_path = _latest(V1_GLOB)
    d3_path = _latest(D3_GLOB)
    print(f"\nv1 trades:  {v1_path.name}")
    print(f"d3 trades:  {d3_path.name}")
    v1_trades = _load_trade_csv(v1_path, "v1")
    d3_trades = _load_trade_csv(d3_path, "donchian")
    # Clip to sim window.
    v1_trades = v1_trades[(v1_trades.entry_ts >= SIM_START) &
                          (v1_trades.exit_ts <= SIM_END)].reset_index(drop=True)
    d3_trades = d3_trades[(d3_trades.entry_ts >= SIM_START) &
                          (d3_trades.exit_ts <= SIM_END)].reset_index(drop=True)
    print(f"  v1: {len(v1_trades)} trades, total cum (compounded) "
          f"{(np.prod(1.0 + v1_trades.ret) - 1.0) * 100:+.1f}%")
    print(f"  d3: {len(d3_trades)} trades, total cum (compounded) "
          f"{(np.prod(1.0 + d3_trades.ret) - 1.0) * 100:+.1f}%")

    # ---- Sim HYBRID via live evaluator (current stateless behaviour) ----
    t0 = time.time()
    df_4h = load_tf("4h").loc[SIM_START:SIM_END]
    print(f"\nReplaying HYBRID live evaluator over {len(df_4h)} bars...")
    hybrid_trades = _sim_hybrid_trades(df_4h)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s — {len(hybrid_trades)} trades")
    if not hybrid_trades.empty:
        print(f"  cum (compounded) {(np.prod(1.0 + hybrid_trades.ret) - 1.0) * 100:+.1f}%, "
              f"WR {(hybrid_trades.ret > 0).mean() * 100:.1f}%, "
              f"exits: tp={int((hybrid_trades.exit_reason == 'tp').sum())} "
              f"sl={int((hybrid_trades.exit_reason == 'sl').sum())} "
              f"time={int((hybrid_trades.exit_reason == 'time').sum())} "
              f"eod={int((hybrid_trades.exit_reason == 'eod').sum())}")

    # ---- Sim 'ideal' HYBRID (backtest's pattern-level dedup) ----
    print(f"\nReplaying IDEAL HYBRID (backtest's stateful dedup)...")
    hybrid_ideal = _sim_hybrid_trades_ideal(df_4h)
    if not hybrid_ideal.empty:
        print(f"  {len(hybrid_ideal)} trades  "
              f"cum {(np.prod(1.0 + hybrid_ideal.ret) - 1.0) * 100:+.1f}%, "
              f"WR {(hybrid_ideal.ret > 0).mean() * 100:.1f}%")

    # ---- Build daily P&L per leg ----
    daily_idx = pd.date_range(SIM_START.normalize(), SIM_END.normalize(),
                              freq="1D", tz="UTC")
    daily_v1 = _daily_pnl(v1_trades, "v1", daily_idx)
    daily_d3 = _daily_pnl(d3_trades, "donchian", daily_idx)
    daily_hy = _daily_pnl(hybrid_trades, "hybrid", daily_idx)
    daily_hy_id = _daily_pnl(hybrid_ideal, "hybrid_ideal", daily_idx)

    # ---- Portfolio P&L (equal weight) ----
    port_2leg = 0.5 * daily_v1 + 0.5 * daily_d3
    port_3leg = (daily_v1 + daily_d3 + daily_hy) / 3.0
    port_3leg_ideal = (daily_v1 + daily_d3 + daily_hy_id) / 3.0

    sh_v1 = _sharpe(daily_v1)
    sh_d3 = _sharpe(daily_d3)
    sh_hy = _sharpe(daily_hy)
    sh_hy_id = _sharpe(daily_hy_id)
    sh_2 = _sharpe(port_2leg)
    sh_3 = _sharpe(port_3leg)
    sh_3_id = _sharpe(port_3leg_ideal)
    cum_v1 = _cum(daily_v1)
    cum_d3 = _cum(daily_d3)
    cum_hy = _cum(daily_hy)
    cum_hy_id = _cum(daily_hy_id)
    cum_2 = _cum(port_2leg)
    cum_3 = _cum(port_3leg)
    cum_3_id = _cum(port_3leg_ideal)
    sharpe_lift = sh_3 - sh_2
    sharpe_lift_ideal = sh_3_id - sh_2

    print("\n" + "-" * 78)
    print("Per-leg + portfolio (daily-P&L based, annualised)")
    print("-" * 78)
    print(f"  {'leg':<28}{'Sharpe':>10}{'cum':>10}")
    print(f"  {'v1':<28}{sh_v1:>10.2f}{cum_v1 * 100:>+9.1f}%")
    print(f"  {'donchian-v3 cons':<28}{sh_d3:>10.2f}{cum_d3 * 100:>+9.1f}%")
    print(f"  {'hybrid (live, over-fires)':<28}{sh_hy:>10.2f}{cum_hy * 100:>+9.1f}%")
    print(f"  {'hybrid (ideal, deduped)':<28}{sh_hy_id:>10.2f}{cum_hy_id * 100:>+9.1f}%")
    print(f"  {'2-leg baseline':<28}{sh_2:>10.2f}{cum_2 * 100:>+9.1f}%")
    print(f"  {'3-leg w/ live hybrid':<28}{sh_3:>10.2f}{cum_3 * 100:>+9.1f}%")
    print(f"  {'3-leg w/ ideal hybrid':<28}{sh_3_id:>10.2f}{cum_3_id * 100:>+9.1f}%")
    print(f"  Sharpe lift LIVE:  {sharpe_lift:+.3f}  (gate ≥ {GATE_SHARPE_LIFT_MIN})")
    print(f"  Sharpe lift IDEAL: {sharpe_lift_ideal:+.3f}  (gate ≥ {GATE_SHARPE_LIFT_MIN})")

    # ---- Correlation matrix (daily P&L, days where ≥1 leg traded) ----
    legs = pd.DataFrame({
        "v1": daily_v1, "donchian": daily_d3,
        "hybrid_live": daily_hy, "hybrid_ideal": daily_hy_id,
    })
    active = (legs != 0).any(axis=1)
    legs_active = legs[active]
    corr = legs_active.corr()
    print("\n" + "-" * 78)
    print("Daily-P&L correlation (days with ≥1 trade active)")
    print("-" * 78)
    print(corr.round(3).to_string())

    corr_hy_v1 = float(corr.loc["hybrid_live", "v1"])
    corr_hy_d3 = float(corr.loc["hybrid_live", "donchian"])
    corr_hi_v1 = float(corr.loc["hybrid_ideal", "v1"])
    corr_hi_d3 = float(corr.loc["hybrid_ideal", "donchian"])

    # ---- Verdict ----
    gate_sharpe_live = sharpe_lift >= GATE_SHARPE_LIFT_MIN
    gate_sharpe_ideal = sharpe_lift_ideal >= GATE_SHARPE_LIFT_MIN
    gate_corr_v1 = abs(corr_hy_v1) < GATE_MAX_CORR
    gate_corr_d3 = abs(corr_hy_d3) < GATE_MAX_CORR
    overall_live = gate_sharpe_live and gate_corr_v1 and gate_corr_d3
    overall_ideal = gate_sharpe_ideal and (abs(corr_hi_v1) < GATE_MAX_CORR) and (abs(corr_hi_d3) < GATE_MAX_CORR)

    print("\n" + "=" * 78)
    print("PHASE 4 VERDICT")
    print("=" * 78)
    print("  LIVE evaluator (stateless — current):")
    print(f"    Sharpe lift ≥ {GATE_SHARPE_LIFT_MIN}: {sharpe_lift:+.3f}  "
          f"→ {'PASS' if gate_sharpe_live else 'FAIL'}")
    print(f"    |corr(hybrid_live, v1)|       < {GATE_MAX_CORR}: "
          f"{abs(corr_hy_v1):.3f}  → {'PASS' if gate_corr_v1 else 'FAIL'}")
    print(f"    |corr(hybrid_live, donchian)| < {GATE_MAX_CORR}: "
          f"{abs(corr_hy_d3):.3f}  → {'PASS' if gate_corr_d3 else 'FAIL'}")
    print(f"    Overall: {'PASS' if overall_live else 'FAIL'}")
    print()
    print("  IDEAL HYBRID (stateful dedup — engineering required):")
    print(f"    Sharpe lift ≥ {GATE_SHARPE_LIFT_MIN}: {sharpe_lift_ideal:+.3f}  "
          f"→ {'PASS' if gate_sharpe_ideal else 'FAIL'}")
    print(f"    |corr(hybrid_ideal, v1)|       < {GATE_MAX_CORR}: "
          f"{abs(corr_hi_v1):.3f}")
    print(f"    |corr(hybrid_ideal, donchian)| < {GATE_MAX_CORR}: "
          f"{abs(corr_hi_d3):.3f}")
    print(f"    Overall: {'PASS' if overall_ideal else 'FAIL'}")
    print("=" * 78)
    overall = overall_live

    RESULTS_PATH.write_text(json.dumps({
        "window": [str(SIM_START.date()), str(SIM_END.date())],
        "n_trades": {
            "v1": len(v1_trades), "donchian": len(d3_trades),
            "hybrid_live": len(hybrid_trades), "hybrid_ideal": len(hybrid_ideal),
        },
        "sharpe": {
            "v1": sh_v1, "donchian": sh_d3,
            "hybrid_live": sh_hy, "hybrid_ideal": sh_hy_id,
            "two_leg": sh_2, "three_leg_live": sh_3, "three_leg_ideal": sh_3_id,
            "lift_live": sharpe_lift, "lift_ideal": sharpe_lift_ideal,
        },
        "cum_pct": {
            "v1": cum_v1, "donchian": cum_d3,
            "hybrid_live": cum_hy, "hybrid_ideal": cum_hy_id,
            "two_leg": cum_2, "three_leg_live": cum_3, "three_leg_ideal": cum_3_id,
        },
        "correlation": corr.round(4).to_dict(),
        "verdict_live": "PASS" if overall_live else "FAIL",
        "verdict_ideal": "PASS" if overall_ideal else "FAIL",
    }, indent=2, default=str))
    print(f"Saved → {RESULTS_PATH}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
