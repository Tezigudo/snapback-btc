"""Re-validate multifactor-v1 baseline (no 4H gate) post fractional-sizing refactor.

Runs 5 OOS windows at $1M cash, with PRICE_SCALE=0.001 fractional sizing,
aggregates per-trade pnl_pct, computes PSR, and writes:
  - reports/_postfrac_mf_baseline_<window>.csv  (per-window trades)
  - reports/postfrac_mf_baseline.json           (full result)

Uses LOCKED dict footing from tools/run_mf_deepening.py + {"use_mtf_4h_gate": False}.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
    equity_impact_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
FUND_PARQ = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

# LOCKED dict from tools/run_mf_deepening.py
LOCKED = {
    "rsi_period":                  14,
    "rsi_long_threshold":          35.0,
    "rsi_short_threshold":         70.0,
    "volume_ma_period":            20,
    "volume_multiple":             2.0,
    "mf_trend_ema_period":         200,
    "require_trend":               True,
    "require_candlestick":         False,
    "require_macd":                False,
    "require_funding_not_extreme": True,
    "funding_extreme_threshold":   0.0005,
    "sl_pct":                      0.015,
    "tp_pct":                      0.030,
    "max_hold_bars":               1344,
    "risk_per_trade_pct":          2.75,
    "leverage":                    20,
    "allow_shorts":                True,
}

# Disable the 4H gate — its 4H parquet is loaded UNSCALED inside strategy init(),
# which is apples-to-oranges with the scaled 15m feed (see tools/_fractional_run.py).
CONFIG = {**LOCKED, "use_mtf_4h_gate": False}

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame:
    """Load 15m BTC parquet, slice [start, end] inclusive, scale OHLC by PRICE_SCALE.

    Attaches Funding column (unscaled — funding rate is dimensionless ratio).
    """
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Attach funding (backward-asof merge, lookahead-safe)
    if FUND_PARQ.exists():
        fund = pd.read_parquet(FUND_PARQ)
        if fund.index.tz is not None:
            fund.index = fund.index.tz_localize(None)
        left = pd.DataFrame(index=df.index)
        right = pd.DataFrame({"Funding": fund["funding_rate"].values}, index=fund.index)
        merged = pd.merge_asof(
            left, right, left_index=True, right_index=True, direction="backward"
        )
        df["Funding"] = merged["Funding"].values
        df["Funding"] = df["Funding"].fillna(0.0)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sl) == 0:
        raise ValueError(f"Empty slice {start}..{end}")

    # Scale OHLC (volume + funding stay un-scaled — they are dimensionless/ratio)
    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * PRICE_SCALE
    return sl


def run_window(label: str, start: str, end: str) -> dict:
    df = _load_slice_scaled(start, end)
    bt = Backtest(
        df, DayTradeMultiFactorBTC,
        cash=CASH, commission=COMMISSION, margin=MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run(**CONFIG)

    n_trades = int(stats.get("# Trades", 0) or 0)
    ret_pct = float(stats.get("Return [%]", 0.0) or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)
    win_rate = float(stats.get("Win Rate [%]") or 0.0)
    equity_final = float(stats.get("Equity Final [$]", CASH) or CASH)

    trades_df = getattr(stats, "_trades", None)
    pnl_pct_list = []
    eq_impact_pnl_pct: list[float] = []
    if trades_df is not None and len(trades_df) > 0 and "ReturnPct" in trades_df.columns:
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
        # Save per-window CSV
        out_csv = ROOT / "reports" / f"_postfrac_mf_baseline_{label}.csv"
        out = pd.DataFrame({
            "pnl_pct": pnl_pct_list,
            "window_start": start,
            "window_end": end,
        })
        out.to_csv(out_csv, index=False)
        print(f"  [{label}] saved {len(out)} trades -> {out_csv.name}",
              file=sys.stderr)

    return {
        "label":             label,
        "start":             start,
        "end":               end,
        "trades":            n_trades,
        "return_pct":        round(ret_pct, 4),   # canonical: stats['Return [%]']
        "max_dd_pct":        round(max_dd, 4),
        "win_rate_pct":      round(win_rate, 4),
        "equity_final":      round(equity_final, 4),
        "pnl_pct":           pnl_pct_list,
        "eq_impact_pnl_pct": eq_impact_pnl_pct,
    }


def aggregate(per_window: list[dict]) -> dict:
    all_pnl: list[float] = []
    n_trades = 0
    n_pos = 0
    compounded = 1.0
    per_win_ret = []
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        if r["return_pct"] > 0:
            n_pos += 1
    return {
        "n_trades":              n_trades,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{len(per_window)}",
        "per_window_return_pct": per_win_ret,
        "all_pnl_pct":           all_pnl,
    }


def main() -> int:
    t0 = time.time()
    per_window = []
    print("[postfrac_mf_baseline] running 5 OOS windows ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        tw = time.time()
        r = run_window(label, start, end)
        print(
            f"  {label}  trades={r['trades']:4d}  ret={r['return_pct']:+8.2f}%  "
            f"dd={r['max_dd_pct']:+7.2f}%  win={r['win_rate_pct']:5.2f}%  "
            f"({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )
        per_window.append(r)

    agg = aggregate(per_window)
    all_pnl = agg.pop("all_pnl_pct")

    # LEGACY stitched-per-trade PSR (N-inflated; observability only).
    import numpy as np
    pnl_arr = np.asarray(all_pnl, dtype=float)
    legacy_psr_stitched = compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95) if len(pnl_arr) >= 2 else {
        "n_trades": int(len(pnl_arr)),
        "psr_vs_hurdle": 0.0,
        "interpretation": "insufficient_evidence",
    }

    # Aggregated CSV
    agg_csv = ROOT / "reports" / "_postfrac_mf_baseline_aggregated.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
    print(f"[postfrac_mf_baseline] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    # Headline PSR migrated from the N-inflated stitched per-trade ReturnPct
    # union to the equity-curve window-level aggregation (psr_walkforward).
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)
    psr = canon["psr_walkforward"]  # canonical headline PSR

    # --- bit-for-bit round-trip check (migration verification) --------------
    persisted = np.asarray(canon["per_window_return_pct"], dtype=float)
    recomputed = (
        compute_psr(persisted, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(persisted) >= 2
        else {"psr_vs_hurdle": 0.0}
    )
    assert recomputed.get("psr_vs_hurdle") == psr.get("psr_vs_hurdle"), (
        f"canonical PSR round-trip MISMATCH: recomputed="
        f"{recomputed.get('psr_vs_hurdle')} headline={psr.get('psr_vs_hurdle')}"
    )
    print(
        f"[postfrac_mf_baseline] canonical PSR round-trip OK: "
        f"{psr.get('psr_vs_hurdle')}",
        file=sys.stderr,
    )

    result = {
        "strategy_id":     "mf_baseline",
        "strategy_class":  "strategy.signals_multifactor:DayTradeMultiFactorBTC",
        "cash":            CASH,
        "commission":      COMMISSION,
        "margin":          MARGIN,
        "price_scale":     PRICE_SCALE,
        "config":          CONFIG,
        "windows":         [w[0] for w in WINDOWS],
        "per_window":      [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":         agg,
        "psr":             psr,                # canonical psr_walkforward
        "legacy_psr_stitched": legacy_psr_stitched,  # observability only
        "canonical":       canon,              # v2 dual-emit
        "aggregation_method": canon["aggregation_method"],
        "pre_fix": {
            "compounded_pct": 50.48,
            "trades":         168,
        },
        "locked_reference": {
            "v1_locked_compounded_pct": 55.73,
            "note": "Phase 2 re-baseline — v2 canonical should match v1 to rounding "
                    "because this runner already used stats['Return [%]'].",
        },
        "elapsed_sec":     round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_mf_baseline.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[postfrac_mf_baseline] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
