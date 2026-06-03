"""TODO_LEG validation runner: Funding-Extreme Contrarian.

Builds all auxiliary feeds (funding 90d-percentile arms, 1H ATR14 on the
scaled price plane, 4H EMA200 slope-pct) from FULL parquet history, then
runs 5 OOS windows at PRICE_SCALE=0.001 with the same harness footing as
tools/_postfrac_mf_baseline.py and tools/_postfrac_mf_4h_btc_run.py.

Trap guards (per advisor):
  - 1H ATR is computed on the SCALED 1H series so distances live on the
    same plane as self.data.Close in the strategy. Skipping this would make
    sl/tp distances ~1000x and degenerate exits to "hold until next print".
  - Funding percentile is computed on the NATIVE 8h series with
    rolling(270).quantile(0.95/0.05) THEN aligned to 15m via merge_asof
    backward. Computing on the forward-filled 15m series would weight by
    occupancy (~32 dupes per print) and use a wrong window length.
  - 4H EMA200 slope is expressed as % change over 10 bars: scale-invariant.
  - Warm-up from full parquet history then slice — first window is not
    NaN-starved.

Outputs:
    reports/postfrac_funding_extreme.json           — full result
    reports/_postfrac_funding_extreme_<window>.csv  — per-window trades
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr, ema  # noqa: E402
from strategy.signals_funding_extreme_contrarian import (  # noqa: E402
    FundingExtremeContrarian,
)
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
    equity_impact_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

PARQ_15m = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
PARQ_FUND = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
PARQ_1H = ROOT / "data" / "historical" / "BTC_USDT_USDT_1h.parquet"
PARQ_4H = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]

# 90 days * 3 prints/day = 270 prints
FUNDING_LOOKBACK_PRINTS = 270
PCT_HIGH = 0.95
PCT_LOW = 0.05

# 4H EMA200 slope: pct change over last 10 bars
EMA4H_SLOPE_LOOKBACK_BARS = 10
EMA4H_PERIOD = 200


def _tznaive(df: pd.DataFrame) -> pd.DataFrame:
    """Strip tz, harmonize to datetime64[us] for merge_asof safety."""
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index.astype("datetime64[us]"))
    return df


# ---------------------------------------------------------------------------
# Aux feed builders (run on FULL parquet history -> aligned to 15m index)
# ---------------------------------------------------------------------------

def _build_funding_arms(idx_15m: pd.DatetimeIndex) -> dict:
    """Return dict of three ndarrays aligned 1:1 to idx_15m:

        arm_short[i]   = True iff the 15m bar at idx_15m[i] is the FIRST 15m bar
                         at or after a funding print whose value was >= 90d q95.
        arm_long[i]    = mirror with <= 90d q05.
        print_bar[i]   = True iff this 15m bar is the FIRST one at or after any
                         funding print (used as time-exit / arm-expiry trigger).

    Lookahead safety:
      - Percentile is computed on the rolling-PAST 270 prints (rolling window).
      - The funding value AT a print is publicly known at print time, so a
        15m bar with timestamp >= print_time may legitimately use the print's
        own value as the most recent observation.
      - merge_asof(direction="backward") attaches each 15m bar to the most
        recent print at-or-before its timestamp.
      - print_bar / arm_* are emitted only on the FIRST 15m bar that crosses
        each new print (by diffing the attached print_time).
    """
    fund = _tznaive(pd.read_parquet(PARQ_FUND))
    s = fund["funding_rate"].astype(float).copy()
    # Rolling q95 / q05 over trailing 270 prints (~90d). min_periods=270 forces
    # warm-up; before that, arms are False everywhere.
    q_hi = s.rolling(window=FUNDING_LOOKBACK_PRINTS, min_periods=FUNDING_LOOKBACK_PRINTS).quantile(PCT_HIGH)
    q_lo = s.rolling(window=FUNDING_LOOKBACK_PRINTS, min_periods=FUNDING_LOOKBACK_PRINTS).quantile(PCT_LOW)
    print_arm_short = (s >= q_hi) & q_hi.notna()
    print_arm_long = (s <= q_lo) & q_lo.notna()

    # Build a per-print frame: print_time, value, arm_short, arm_long
    print_df = pd.DataFrame({
        "print_time": s.index,
        "arm_short": print_arm_short.values,
        "arm_long": print_arm_long.values,
    }, index=s.index)
    print_df = _tznaive(print_df)

    # Align to 15m via backward merge_asof.
    left = pd.DataFrame(index=idx_15m)
    left = _tznaive(left)
    merged = pd.merge_asof(
        left, print_df,
        left_index=True, right_index=True,
        direction="backward",
    )

    # print_bar = True only at the FIRST 15m bar after each new print_time.
    pt = merged["print_time"]
    is_new_print = pt != pt.shift(1)
    is_new_print = is_new_print & pt.notna()
    print_bar = is_new_print.values

    arm_short_15m = (merged["arm_short"].fillna(False).values & print_bar).astype(bool)
    arm_long_15m = (merged["arm_long"].fillna(False).values & print_bar).astype(bool)
    return {
        "arm_short": arm_short_15m,
        "arm_long": arm_long_15m,
        "print_bar": print_bar,
    }


def _build_atr_1h_scaled(idx_15m: pd.DatetimeIndex) -> np.ndarray:
    """1H ATR14 on SCALED price plane, aligned to 15m via merge_asof(backward)
    on bar-CLOSE timestamps (open_time + 1h)."""
    df1h = _tznaive(pd.read_parquet(PARQ_1H))
    # Scale OHLC by PRICE_SCALE BEFORE computing ATR — distances must be
    # on the same plane as self.data.Close in the strategy.
    for col in ("open", "high", "low", "close"):
        if col in df1h.columns:
            df1h[col] = df1h[col].astype(float) * PRICE_SCALE
    atr_1h = atr(df1h["high"], df1h["low"], df1h["close"], 14)
    close_times = pd.DatetimeIndex(
        (df1h.index + pd.Timedelta(hours=1)).astype("datetime64[us]")
    )
    right = pd.DataFrame({"atr": atr_1h.values}, index=close_times).sort_index()
    left = pd.DataFrame(index=idx_15m)
    left = _tznaive(left)
    merged = pd.merge_asof(
        left, right, left_index=True, right_index=True, direction="backward"
    )
    return merged["atr"].values


def _build_ema4h_slope_pct(idx_15m: pd.DatetimeIndex) -> np.ndarray:
    """4H EMA200 slope expressed as pct change over last 10 bars.

    Scale-invariant — we can compute on the unscaled 4H series and the slope
    in % is identical to what we'd get on the scaled series.
    """
    df4h = _tznaive(pd.read_parquet(PARQ_4H))
    close_col = "close" if "close" in df4h.columns else "Close"
    e = ema(df4h[close_col].astype(float), EMA4H_PERIOD)
    slope_pct = (e / e.shift(EMA4H_SLOPE_LOOKBACK_BARS) - 1.0) * 100.0
    close_times = pd.DatetimeIndex(
        (df4h.index + pd.Timedelta(hours=4)).astype("datetime64[us]")
    )
    right = pd.DataFrame({"slope": slope_pct.values}, index=close_times).sort_index()
    left = pd.DataFrame(index=idx_15m)
    left = _tznaive(left)
    merged = pd.merge_asof(
        left, right, left_index=True, right_index=True, direction="backward"
    )
    return merged["slope"].values


# ---------------------------------------------------------------------------
# 15m slice loader (matches _postfrac_mf_baseline footing)
# ---------------------------------------------------------------------------

def _load_full_15m_scaled() -> pd.DataFrame:
    df = pd.read_parquet(PARQ_15m)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df = _tznaive(df)
    # Funding column (forward-fill via backward merge_asof onto 15m bars).
    # NOT used for percentile (we use native 8h series), but the live bot
    # has this column and a few defensive Strategy reads may inspect it.
    if PARQ_FUND.exists():
        fund = _tznaive(pd.read_parquet(PARQ_FUND))
        left = pd.DataFrame(index=df.index)
        right = pd.DataFrame({"Funding": fund["funding_rate"].astype(float).values}, index=fund.index)
        merged = pd.merge_asof(
            left, right, left_index=True, right_index=True, direction="backward"
        )
        df["Funding"] = merged["Funding"].fillna(0.0).values
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] * PRICE_SCALE
    return df


def _attach_aux(df_full: pd.DataFrame) -> pd.DataFrame:
    """Compute aux feeds on full history then return df_full augmented."""
    idx = pd.DatetimeIndex(df_full.index)
    arms = _build_funding_arms(idx)
    df_full = df_full.copy()
    df_full["FundArmShort"] = arms["arm_short"]
    df_full["FundArmLong"] = arms["arm_long"]
    df_full["FundPrintBar"] = arms["print_bar"]
    df_full["AtrPriceScaled1h"] = _build_atr_1h_scaled(idx)
    df_full["Ema4hSlopePct"] = _build_ema4h_slope_pct(idx)
    return df_full


# ---------------------------------------------------------------------------
# Window runner
# ---------------------------------------------------------------------------

def _slice_window(df_full: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = df_full.loc[(df_full.index >= start_ts) & (df_full.index <= end_ts)].copy()
    if len(sl) == 0:
        raise ValueError(f"Empty slice {start}..{end}")
    return sl


def run_window(label: str, df: pd.DataFrame, start: str, end: str) -> dict:
    bt = Backtest(
        df, FundingExtremeContrarian,
        cash=CASH, commission=COMMISSION, margin=MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run()

    n_trades = int(stats.get("# Trades", 0) or 0)
    ret_pct = float(stats.get("Return [%]", 0.0) or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)
    win_rate = float(stats.get("Win Rate [%]") or 0.0)
    sharpe = float(stats.get("Sharpe Ratio") or 0.0)
    equity_final = float(stats.get("Equity Final [$]", CASH) or CASH)

    trades_df = getattr(stats, "_trades", None)
    pnl_pct_list: list[float] = []
    eq_impact_pnl_pct: list[float] = []
    if trades_df is not None and len(trades_df) > 0 and "ReturnPct" in trades_df.columns:
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        # CANONICAL (v2): sizing-aware equity-impact returns for per-window PSR.
        # No warm-prefix here — the slice IS the OOS window, so all trades count.
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
        out_csv = ROOT / "reports" / f"_postfrac_funding_extreme_{label}.csv"
        out = pd.DataFrame({"pnl_pct": pnl_pct_list})
        out["window_start"] = start
        out["window_end"] = end
        out.to_csv(out_csv, index=False)
        print(f"  [{label}] saved {len(out)} trades -> {out_csv.name}", file=sys.stderr)

    return {
        "label":             label,
        "start":             start,
        "end":               end,
        "trades":            n_trades,
        "return_pct":        round(ret_pct, 4),
        "max_dd_pct":        round(max_dd, 4),
        "win_rate_pct":      round(win_rate, 4),
        "sharpe":            round(sharpe, 4),
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
    print("[postfrac_funding_extreme] loading 15m + funding + 1h + 4h ...", file=sys.stderr)
    df_full = _load_full_15m_scaled()
    print(f"  15m rows total: {len(df_full):,}", file=sys.stderr)
    df_full = _attach_aux(df_full)

    # Sanity diagnostic: stop distance ~ 1.5 * ATR should be ~0.5-3% of price.
    sample_idx = df_full.index.get_indexer([pd.Timestamp("2024-01-15")], method="nearest")[0]
    if 0 <= sample_idx < len(df_full):
        atr_v = df_full["AtrPriceScaled1h"].iloc[sample_idx]
        close_v = df_full["Close"].iloc[sample_idx]
        dist = 1.5 * atr_v
        pct = (dist / close_v) * 100.0 if close_v > 0 else float("nan")
        print(
            f"  sanity 2024-01-15: close_scaled={close_v:.4f} "
            f"atr_scaled={atr_v:.4f} stop_dist={dist:.4f} ({pct:.2f}% of price)",
            file=sys.stderr,
        )

    # Aux occupancy diagnostics
    n_print = int(df_full["FundPrintBar"].sum())
    n_arm_short = int(df_full["FundArmShort"].sum())
    n_arm_long = int(df_full["FundArmLong"].sum())
    print(
        f"  funding prints aligned to 15m: {n_print:,}  "
        f"arm_short={n_arm_short:,}  arm_long={n_arm_long:,}",
        file=sys.stderr,
    )

    per_window = []
    print("[postfrac_funding_extreme] running 5 OOS windows ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        tw = time.time()
        df = _slice_window(df_full, start, end)
        r = run_window(label, df, start, end)
        print(
            f"  {label}  trades={r['trades']:4d}  ret={r['return_pct']:+8.2f}%  "
            f"dd={r['max_dd_pct']:+7.2f}%  win={r['win_rate_pct']:5.2f}%  "
            f"sharpe={r['sharpe']:+6.2f}  ({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )
        per_window.append(r)

    agg = aggregate(per_window)
    all_pnl = agg.pop("all_pnl_pct")

    pnl_arr = np.asarray(all_pnl, dtype=float)
    # LEGACY stitched-per-trade PSR (N-inflated; observability only, NOT the
    # headline). Canonical block below carries this internally as
    # legacy_psr_stitched too — kept here for backcompat of the `psr` key shape.
    legacy_psr_stitched = compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95) if len(pnl_arr) >= 2 else {
        "n_trades": int(len(pnl_arr)),
        "psr_vs_hurdle": 0.0,
        "interpretation": "insufficient_evidence",
    }

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    # PSR axis migrated from stitched per-trade ReturnPct to the equity-curve
    # window-level aggregation (psr_walkforward).
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)
    psr = canon["psr_walkforward"]  # canonical headline PSR

    # Aggregated CSV
    agg_csv = ROOT / "reports" / "_postfrac_funding_extreme_AGGREGATE.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)

    result = {
        "strategy_id":     "funding_extreme_contrarian",
        "strategy_class":  "strategy.signals_funding_extreme_contrarian:FundingExtremeContrarian",
        "cash":            CASH,
        "commission":      COMMISSION,
        "margin":          MARGIN,
        "price_scale":     PRICE_SCALE,
        "config": {
            "arm_expiry_bars": 32,
            "min_spacing_bars": 96,
            "volume_multiple": 1.5,
            "atr_sl_mult": 1.5,
            "atr_tp_mult": 2.5,
            "risk_per_trade_pct": 0.5,
            "leverage": 20,
            "slope_flat_threshold_pct": 0.05,
            "funding_lookback_prints": FUNDING_LOOKBACK_PRINTS,
            "pct_high": PCT_HIGH,
            "pct_low": PCT_LOW,
        },
        "windows":         [w[0] for w in WINDOWS],
        "per_window":      [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":         agg,
        "psr":                  psr,                  # canonical psr_walkforward
        "legacy_psr_stitched":  legacy_psr_stitched,  # observability only
        "canonical":            canon,                # v2 dual-emit block
        "aggregation_method":   canon["aggregation_method"],
        "aux_diagnostics": {
            "n_funding_prints_15m_aligned": n_print,
            "n_arm_short_15m":              n_arm_short,
            "n_arm_long_15m":               n_arm_long,
        },
        "elapsed_sec":     round(time.time() - t0, 2),
    }

    # --- bit-for-bit round-trip check (migration verification) --------------
    # Headline psr == compute_psr on the PERSISTED canonical per-window return
    # series (rounded as aggregate_windows stores it), contiguous=False.
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
        f"[postfrac_funding_extreme] canonical PSR round-trip OK: "
        f"{psr.get('psr_vs_hurdle')}",
        file=sys.stderr,
    )

    out_path = ROOT / "reports" / "postfrac_funding_extreme.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[postfrac_funding_extreme] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    print(json.dumps({
        "n_trades":       agg["n_trades"],
        "compounded_pct": agg["compounded_pct"],
        "psr_vs_hurdle":  psr.get("psr_vs_hurdle"),
        "interpretation": psr.get("interpretation"),
        "windows_pos":    agg["windows_positive"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
