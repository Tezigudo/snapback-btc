"""Phase 5 head-to-head: compare dedup ∈ {5, 10, 15} on portfolio metrics.

Phase 1 (single-leg walk-forward) found dedup=15 best by OOS Sharpe/cum.
Phase 5 confirms that hold under the LIVE evaluator (post-3b stateful dedup)
in a 3-leg portfolio context, where a faster-firing variant might dilute or
diversify better than its standalone Sharpe suggests.

For each dedup, this tool:
  1. Replays the live HYBRID with that dedup over 2020 → 2026-05-23.
  2. Computes daily P&L per leg and the 3-leg portfolio.
  3. Reports Sharpe lift vs 2-leg baseline, correlation with v1 and
     Donchian, trade count, win rate.

No automatic pick — the user makes the call after seeing the numbers
(Phase 5 gate is explicit user choice).

Run:
    uv run python tools/hybrid_phase5_dedup_choice.py
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
from strategy.live_cnh_hybrid_short import _admitted_patterns  # noqa: E402
from tools.icnh_mega_sweep import load_tf  # noqa: E402

RESULTS_PATH = ROOT / "data" / "hybrid_phase5_dedup_choice_results.json"

SIM_START = pd.Timestamp("2020-01-01", tz="UTC")
SIM_END = pd.Timestamp("2026-05-23", tz="UTC")
TIME_STOP_BARS = 96
FRICTION_BPS_RT = 13.0
TRADING_DAYS_PER_YEAR = 365.25

DEDUPS_TO_TEST = [5, 10, 15]
V1_GLOB = "reports/full_history_*_v1_trades.csv"
D3_GLOB = "reports/full_history_*_d3cons_trades.csv"


def _latest(glob_pattern: str) -> Path:
    files = sorted(glob.glob(str(ROOT / glob_pattern)))
    return Path(files[-1])


def _load_trade_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["EntryTime", "ExitTime"])
    df = df.rename(columns={"EntryTime": "entry_ts", "ExitTime": "exit_ts",
                            "ReturnPct": "ret"})
    df = df[["entry_ts", "exit_ts", "ret"]].copy()
    for col in ("entry_ts", "exit_ts"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df[(df.entry_ts >= SIM_START) & (df.exit_ts <= SIM_END)].reset_index(drop=True)


def _sim_hybrid_for_dedup(df: pd.DataFrame, dedup_bars: int) -> pd.DataFrame:
    """Same logic as hybrid_phase4_portfolio._sim_hybrid_trades but with
    dedup_bars parameterised. df is expected to already have indicators
    attached (avoid recomputing across dedup variants)."""
    cfg = HybridConfig(dedup_bars=dedup_bars)
    admitted = _admitted_patterns(df, cfg, len(df) - 1, dedup_bars)

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
        candidates.append({
            "signal_idx": signal_idx, "entry_price": entry_price,
            "stop": entry_price + cfg.sl_atr_mult * atr_v,
            "tp": entry_price - (entry_price - ema100),
        })

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    ts = df.index

    trades: list[dict] = []
    cand_iter = iter(candidates)
    next_cand = next(cand_iter, None)
    open_trade: dict | None = None

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
            if hit is not None:
                reason, exit_price = hit
                gross = (o["entry_price"] - exit_price) / o["entry_price"]
                net = gross - FRICTION_BPS_RT / 10_000.0
                trades.append({
                    "entry_ts": o["entry_ts"], "exit_ts": ts[i],
                    "ret": net, "exit_reason": reason,
                })
                open_trade = None
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
            "ret": net, "exit_reason": "eod",
        })
    return pd.DataFrame(trades)


def _daily_pnl(trades: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    daily = pd.Series(0.0, index=index)
    if trades.empty:
        return daily
    for _, row in trades.iterrows():
        e = pd.Timestamp(row["entry_ts"]).tz_convert("UTC").normalize()
        x = pd.Timestamp(row["exit_ts"]).tz_convert("UTC").normalize()
        days = pd.date_range(e, x, freq="1D", tz="UTC").intersection(index)
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
    print("HYBRID SHORT — Phase 5 dedup choice")
    print(f"Window:    {SIM_START.date()} → {SIM_END.date()}")
    print(f"Friction:  {FRICTION_BPS_RT} bps round-trip")
    print(f"Time-stop: {TIME_STOP_BARS} bars (16 days)")
    print("=" * 78)

    # ---- Load v1 + Donchian baselines (constant across dedup variants) ----
    v1_trades = _load_trade_csv(_latest(V1_GLOB))
    d3_trades = _load_trade_csv(_latest(D3_GLOB))
    daily_idx = pd.date_range(SIM_START.normalize(), SIM_END.normalize(),
                              freq="1D", tz="UTC")
    daily_v1 = _daily_pnl(v1_trades, daily_idx)
    daily_d3 = _daily_pnl(d3_trades, daily_idx)
    port_2leg = 0.5 * daily_v1 + 0.5 * daily_d3
    sh_2 = _sharpe(port_2leg)
    cum_2 = _cum(port_2leg)
    print(f"\nBaseline: v1+Donchian Sharpe {sh_2:.2f}, cum {cum_2 * 100:+.1f}%")

    # ---- Precompute indicators once (constant across dedup variants) ----
    print("\nAttaching indicators to 4h series...")
    t_ind = time.time()
    df_4h_raw = load_tf("4h").loc[SIM_START:SIM_END]
    df_4h = attach_indicators(df_4h_raw, HybridConfig())
    print(f"  done in {time.time() - t_ind:.1f}s ({len(df_4h)} bars)")

    # ---- Per-dedup loop ----
    rows: list[dict] = []
    for dedup in DEDUPS_TO_TEST:
        print(f"\nDedup={dedup}...", flush=True)
        t0 = time.time()
        trades = _sim_hybrid_for_dedup(df_4h, dedup)
        elapsed = time.time() - t0
        if trades.empty:
            print(f"  no trades")
            continue
        daily_hy = _daily_pnl(trades, daily_idx)
        port_3leg = (daily_v1 + daily_d3 + daily_hy) / 3.0
        sh_hy = _sharpe(daily_hy)
        sh_3 = _sharpe(port_3leg)
        cum_hy = _cum(daily_hy)
        cum_3 = _cum(port_3leg)
        lift = sh_3 - sh_2
        wr = float((trades.ret > 0).mean())
        years = (SIM_END - SIM_START).days / 365.25
        per_year = len(trades) / years
        # Correlations
        active = (daily_hy != 0) | (daily_v1 != 0) | (daily_d3 != 0)
        legs = pd.DataFrame({"v1": daily_v1, "donchian": daily_d3, "hybrid": daily_hy})[active]
        corr_v1 = float(legs.corr().loc["hybrid", "v1"])
        corr_d3 = float(legs.corr().loc["hybrid", "donchian"])
        row = {
            "dedup": dedup,
            "trades": len(trades),
            "trades_per_year": per_year,
            "win_rate": wr,
            "cum_hybrid": cum_hy,
            "sharpe_hybrid": sh_hy,
            "cum_3leg": cum_3,
            "sharpe_3leg": sh_3,
            "sharpe_lift": lift,
            "corr_v1": corr_v1,
            "corr_donchian": corr_d3,
            "exit_tp": int((trades.exit_reason == "tp").sum()),
            "exit_sl": int((trades.exit_reason == "sl").sum()),
            "exit_time": int((trades.exit_reason == "time").sum()),
            "elapsed_s": elapsed,
        }
        rows.append(row)
        print(f"  {len(trades)} trades ({per_year:.1f}/yr), "
              f"WR {wr * 100:.1f}%, "
              f"hybrid Sharpe {sh_hy:.2f} cum {cum_hy * 100:+.1f}%, "
              f"lift {lift:+.3f}  [{elapsed:.0f}s]")

    # ---- Comparison table ----
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD")
    print("=" * 78)
    print(
        f"{'dedup':<8}{'trades':>8}{'/yr':>7}{'WR':>8}"
        f"{'hy cum':>10}{'hy Sh':>8}"
        f"{'3leg cum':>11}{'3leg Sh':>10}{'lift':>8}"
        f"{'corr v1':>9}{'corr d3':>9}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['dedup']:<8}{r['trades']:>8}{r['trades_per_year']:>7.1f}"
            f"{r['win_rate'] * 100:>7.1f}%"
            f"{r['cum_hybrid'] * 100:>+9.1f}%{r['sharpe_hybrid']:>8.2f}"
            f"{r['cum_3leg'] * 100:>+10.1f}%{r['sharpe_3leg']:>10.2f}"
            f"{r['sharpe_lift']:>+8.3f}"
            f"{r['corr_v1']:>+9.3f}{r['corr_donchian']:>+9.3f}"
        )

    # ---- Surface a recommendation (NOT an auto-pick) ----
    best_lift = max(rows, key=lambda r: r["sharpe_lift"])
    best_cum = max(rows, key=lambda r: r["cum_3leg"])
    best_wr = max(rows, key=lambda r: r["win_rate"])
    print()
    print("Suggested by metric (user picks consciously per HYBRID_SHORT_PLAN.md Phase 5):")
    print(f"  best Sharpe lift:  dedup={best_lift['dedup']}  (+{best_lift['sharpe_lift']:.3f})")
    print(f"  best 3-leg cum:    dedup={best_cum['dedup']}   ({best_cum['cum_3leg'] * 100:+.1f}%)")
    print(f"  best WR:           dedup={best_wr['dedup']}    ({best_wr['win_rate'] * 100:.1f}%)")
    print(f"  best frequency:    dedup={min(rows, key=lambda r: r['dedup'])['dedup']} "
          f"({min(rows, key=lambda r: r['dedup'])['trades_per_year']:.1f}/yr)")

    RESULTS_PATH.write_text(json.dumps({
        "window": [str(SIM_START.date()), str(SIM_END.date())],
        "baseline_sharpe_2leg": sh_2,
        "baseline_cum_2leg": cum_2,
        "variants": rows,
    }, indent=2, default=str))
    print(f"\nSaved → {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
