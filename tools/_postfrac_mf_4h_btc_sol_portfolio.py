"""Post-fractional-sizing multifactor-v1 + 4H gate: BTC+SOL PORTFOLIO harness.

Composes per-coin runs into a portfolio under 4 ARMs (selected via env var
ARM_TAG):

    solo_15bps        - run BTC alone and SOL alone at 15bps RT comm; baseline.
    port_5050_noveto  - 50/50 portfolio, no cross-coin 4H veto.
    port_5050_veto    - 50/50 portfolio, mutual cross-coin 4H EMA200 veto on.
    port_6040_veto    - 60% BTC / 40% SOL, mutual cross-coin veto on.

Veto wiring (when active):
    BTC slice: cross_4h_parquet_path = SOL 4H parquet (scaled)
    SOL slice: cross_4h_parquet_path = BTC 4H parquet (scaled)
Both parquets are price-scaled by PRICE_SCALE=0.001 (same plane as 15m).

Portfolio synthesis (per OOS window):
    compounded_portfolio = w_btc * btc_compounded + w_sol * sol_compounded
        (linear weighting; first-cut approximation — capital is NOT rebalanced
         intra-window; each sleeve compounds independently then weighted-sum
         on terminal return.)
    trades_portfolio     = btc_trades + sol_trades
    win_rate_portfolio   = trade-weighted across both coins
    maxdd_portfolio      = drawdown of weighted-sum equity curve (BTC and SOL
                            equity curves each scaled by their allocation,
                            superimposed element-wise on a unified index).
    psr_portfolio        = compute_psr on the per-period returns of the
                            time-aligned weighted-sum portfolio equity curve
                            (default 1D resample). The legacy stitched-trade
                            stream is retained as ``psr_trade_pool_proxy``
                            for diff-ability against pre-fix JSONs. See
                            ``tools/portfolio_psr.py`` for methodology debt
                            #2 rationale.

Output: reports/postfrac_mf_4h_btc_sol_portfolio_${ARM_TAG}.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402
from tools.portfolio_psr import (  # noqa: E402
    aggregate_portfolio_psr,
    build_portfolio_equity_curve,
    equity_to_period_returns,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASH = 1_000_000.0
COMMISSION = 0.00075  # 15bps RT (7.5bps per side ≈ 15bps round-trip)
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
    ("2026_H1", "2026-01-01", "2026-06-30"),  # optional — included only if data exists
]

PARQ_15m = {
    "BTC": ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet",
    "SOL": ROOT / "data" / "historical" / "SOL_USDT_USDT_15m.parquet",
}
PARQ_4H_SRC = {
    "BTC": ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet",
    "SOL": ROOT / "data" / "historical" / "SOL_USDT_USDT_4h.parquet",
}
PARQ_FUND = {
    "BTC": ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet",
    "SOL": ROOT / "data" / "historical" / "SOL_USDT_USDT_funding.parquet",
}

TMP_DIR = ROOT / "reports" / "_tmp"


# ---------------------------------------------------------------------------
# Locked multifactor-v1 config (mirrors run_mf_4h_multi_coin.py LOCKED dict)
# ---------------------------------------------------------------------------

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
    "use_mtf_4h_gate":             True,
}


# ---------------------------------------------------------------------------
# ARMS
# ---------------------------------------------------------------------------

# (allocation_btc, allocation_sol, veto_on) per arm.
ARMS = {
    "solo_15bps":          (1.0, 1.0, False),  # both coins as solo books (reference)
    "port_5050_noveto":    (0.5, 0.5, False),
    "port_5050_veto":      (0.5, 0.5, True),
    "port_6040_veto":      (0.6, 0.4, True),
    # v2 re-runs (post PSR-weighting fix) — same configs, separate output files
    "port_5050_noveto_v2": (0.5, 0.5, False),
    "port_5050_veto_v2":   (0.5, 0.5, True),
    "port_6040_veto_v2":   (0.6, 0.4, True),
}


# ---------------------------------------------------------------------------
# 4H parquet scaling (idempotent; one per coin)
# ---------------------------------------------------------------------------

def _ensure_scaled_4h_parquet(coin: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    src = PARQ_4H_SRC[coin]
    dst = TMP_DIR / f"{coin}_USDT_USDT_4h_scaled_{PRICE_SCALE}.parquet"
    if dst.exists():
        return dst
    df = pd.read_parquet(src)
    for col in ("open", "high", "low", "close", "Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] * PRICE_SCALE
    df.to_parquet(dst)
    return dst


# ---------------------------------------------------------------------------
# 15m slice loader (scaled OHLC + Funding merge if available)
# ---------------------------------------------------------------------------

def _load_slice_scaled(coin: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(PARQ_15m[coin])
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    fund_path = PARQ_FUND.get(coin)
    if fund_path and fund_path.exists():
        fund = pd.read_parquet(fund_path)
        if fund.index.tz is not None:
            fund.index = fund.index.tz_localize(None)
        left = pd.DataFrame(index=df.index)
        right = pd.DataFrame({"Funding": fund["funding_rate"].values}, index=fund.index)
        merged = pd.merge_asof(left, right, left_index=True, right_index=True,
                               direction="backward")
        df["Funding"] = merged["Funding"].fillna(0.0).values
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sliced) == 0:
        raise ValueError(f"Empty slice {coin} {start}..{end}")
    for col in ("Open", "High", "Low", "Close"):
        if col in sliced.columns:
            sliced[col] = sliced[col] * PRICE_SCALE
    return sliced


# ---------------------------------------------------------------------------
# Per-coin window backtest
# ---------------------------------------------------------------------------

def _run_coin_window(coin: str, label: str, start: str, end: str,
                     veto_on: bool) -> dict | None:
    try:
        df = _load_slice_scaled(coin, start, end)
    except ValueError as exc:
        print(f"  {coin} {label}: SKIP ({exc})", file=sys.stderr)
        return None

    primary_4h = _ensure_scaled_4h_parquet(coin)
    config = {
        **LOCKED,
        "mtf_4h_parquet_path": str(primary_4h),
    }
    if veto_on:
        other_coin = "SOL" if coin == "BTC" else "BTC"
        config["cross_4h_parquet_path"] = str(_ensure_scaled_4h_parquet(other_coin))
    else:
        config["cross_4h_parquet_path"] = ""

    bt = Backtest(df, DayTradeMultiFactorBTC, cash=CASH, commission=COMMISSION,
                  margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run(**config)
    trades_df = getattr(stats, "_trades", None)
    pnl_pct: list[float] = []
    entry_times: list[pd.Timestamp] = []
    exit_times: list[pd.Timestamp] = []
    if trades_df is not None and len(trades_df):
        if "ReturnPct" in trades_df.columns:
            pnl_pct = (trades_df["ReturnPct"].values * 100.0).tolist()
        if "EntryTime" in trades_df.columns:
            entry_times = list(pd.to_datetime(trades_df["EntryTime"]))
        if "ExitTime" in trades_df.columns:
            exit_times = list(pd.to_datetime(trades_df["ExitTime"]))

    # equity curve: backtesting.py exposes via stats._equity_curve (DataFrame with
    # 'Equity' column indexed by datetime).
    eq_curve = getattr(stats, "_equity_curve", None)
    equity_series = None
    if eq_curve is not None and "Equity" in eq_curve.columns:
        equity_series = eq_curve["Equity"].copy()
        if equity_series.index.tz is not None:
            equity_series.index = equity_series.index.tz_localize(None)

    return {
        "label":         label,
        "start":         start,
        "end":           end,
        "trades":        int(stats.get("# Trades", 0)),
        "return_pct":    float(stats.get("Return [%]", 0.0) or 0.0),
        "max_dd_pct":    float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
        "win_rate_pct":  float(stats.get("Win Rate [%]") or 0.0),
        "equity_final":  float(stats.get("Equity Final [$]", CASH) or CASH),
        "sharpe":        float(stats.get("Sharpe Ratio") or 0.0),
        "pnl_pct":       pnl_pct,
        "entry_times":   entry_times,
        "exit_times":    exit_times,
        "equity_series": equity_series,  # pd.Series or None
    }


# ---------------------------------------------------------------------------
# Portfolio synthesis (single window)
# ---------------------------------------------------------------------------

def _synth_portfolio_window(btc_r: dict | None, sol_r: dict | None,
                            w_btc: float, w_sol: float) -> dict:
    btc_ret = (btc_r["return_pct"] / 100.0) if btc_r else 0.0
    sol_ret = (sol_r["return_pct"] / 100.0) if sol_r else 0.0
    # Linear-weighted terminal return (no intra-window rebalancing).
    port_ret_pct = (w_btc * btc_ret + w_sol * sol_ret) * 100.0

    btc_trades = btc_r["trades"] if btc_r else 0
    sol_trades = sol_r["trades"] if sol_r else 0
    port_trades = btc_trades + sol_trades

    # Trade-weighted win rate
    btc_wr = (btc_r["win_rate_pct"] if btc_r else 0.0)
    sol_wr = (sol_r["win_rate_pct"] if sol_r else 0.0)
    if port_trades > 0:
        port_wr = (btc_wr * btc_trades + sol_wr * sol_trades) / port_trades
    else:
        port_wr = 0.0

    # MaxDD: weighted sum of normalized equity curves
    def _norm_eq(r):
        if r is None or r["equity_series"] is None:
            return None
        eq = r["equity_series"]
        if len(eq) == 0:
            return None
        # Normalize to start=1.0 so weights apply on growth multiple
        start_eq = float(eq.iloc[0])
        if start_eq <= 0:
            return None
        return eq / start_eq

    btc_eq_raw = btc_r["equity_series"] if (
        btc_r is not None and "equity_series" in btc_r) else None
    sol_eq_raw = sol_r["equity_series"] if (
        sol_r is not None and "equity_series" in sol_r) else None

    # Time-aligned weighted-sum portfolio equity (reused for both maxDD and
    # the post-fix headline PSR -- see tools/portfolio_psr.py).
    port_eq = build_portfolio_equity_curve(btc_eq_raw, sol_eq_raw, w_btc, w_sol)

    port_max_dd_pct = 0.0
    if port_eq is not None and len(port_eq) > 0:
        running_max = port_eq.cummax()
        dd_series = (port_eq / running_max) - 1.0
        port_max_dd_pct = float(dd_series.min() * 100.0)  # negative

    # Legacy stitched per-trade stream: each coin's per-trade return scaled by
    # its allocation weight. Retained as ``pnl_pct_trade_pool_proxy`` so
    # historical JSONs remain diff-able; DO NOT use this for the headline PSR
    # (methodology debt #2 -- n-inflated, cross-coin correlation destroyed).
    btc_pnl = btc_r["pnl_pct"] if (btc_r and "pnl_pct" in btc_r) else []
    sol_pnl = sol_r["pnl_pct"] if (sol_r and "pnl_pct" in sol_r) else []
    pnl_pct_trade_pool_proxy = (
        [r * w_btc for r in btc_pnl] + [r * w_sol for r in sol_pnl]
    )

    return {
        "return_pct":   round(port_ret_pct, 4),
        "trades":       int(port_trades),
        "win_rate_pct": round(float(port_wr), 4),
        "max_dd_pct":   round(port_max_dd_pct, 4),
        "btc_return_pct": round((btc_ret) * 100.0, 4) if btc_r else None,
        "sol_return_pct": round((sol_ret) * 100.0, 4) if sol_r else None,
        # Pre-fix stream, KEPT for diff-ability but NOT the headline metric.
        "pnl_pct_trade_pool_proxy": pnl_pct_trade_pool_proxy,
        # Post-fix headline input: time-aligned weighted-sum portfolio equity
        # for this window. aggregate_portfolio_psr resamples + diffs across
        # all windows.
        "portfolio_equity_series": port_eq,
    }


# ---------------------------------------------------------------------------
# Aggregate across windows (per book — BTC solo, SOL solo, Portfolio)
# ---------------------------------------------------------------------------

def _aggregate_book(per_window: dict, label_for_returns: str) -> dict:
    """Aggregate per-window stats into a book-level summary.

    For BTC / SOL legs the ``psr_vs_hurdle`` field is computed on the trade-
    level pnl pool exactly as before (those PSRs are TRUSTWORTHY -- per-leg).

    For the PORT label, this function intentionally treats ``psr_vs_hurdle``
    as the LEGACY (and broken) trade-pool proxy. main() then OVERWRITES it
    with the equity-curve PSR from ``aggregate_portfolio_psr``. The proxy is
    preserved under ``psr_trade_pool_proxy`` for diff-ability.
    """
    n_pos = 0
    compounded = 1.0
    per_window_returns = []
    n_trades_total = 0
    all_pnl: list[float] = []
    for w, r in per_window.items():
        if r is None:
            continue
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_window_returns.append(round(r["return_pct"], 4))
        if r["return_pct"] > 0:
            n_pos += 1
        n_trades_total += r.get("trades", 0)
        if "pnl_pct" in r:
            # BTC / SOL leg path (legitimate per-leg trade pool).
            all_pnl.extend(r["pnl_pct"])
        elif "pnl_pct_trade_pool_proxy" in r:
            # PORT path (legacy proxy -- kept only for diff-ability).
            all_pnl.extend(r["pnl_pct_trade_pool_proxy"])
    n_total = len([r for r in per_window.values() if r is not None])
    pnl_arr = np.asarray(all_pnl) if all_pnl else np.asarray([])
    psr = compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95) if len(pnl_arr) >= 2 else {
        "n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
        "interpretation": "insufficient_evidence",
    }
    return {
        "n_trades_total":        n_trades_total,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{n_total}",
        "per_window_return_pct": per_window_returns,
        "psr_vs_hurdle":         psr.get("psr_vs_hurdle"),
        "psr_interpretation":    psr.get("interpretation"),
        "_all_pnl":              all_pnl,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    arm = os.environ.get("ARM_TAG", "port_5050_noveto")
    if arm not in ARMS:
        print(f"[port] ERROR: ARM_TAG={arm!r} not in {list(ARMS.keys())}",
              file=sys.stderr)
        return 2
    w_btc, w_sol, veto_on = ARMS[arm]

    # Override: optional single-window via env (smoke test).
    only_window = os.environ.get("ONLY_WINDOW", "").strip()

    # solo_15bps arm: keep weights at 1.0 (each is its own solo book).
    solo_mode = (arm == "solo_15bps")

    print(f"[port] arm={arm} w_btc={w_btc} w_sol={w_sol} veto_on={veto_on} "
          f"solo_mode={solo_mode} only_window={only_window or 'ALL'}",
          file=sys.stderr)

    btc_windows: dict = {}
    sol_windows: dict = {}
    port_windows: dict = {}

    for label, start, end in WINDOWS:
        if only_window and label != only_window:
            continue
        # Skip 2026_H1 if data slice empty
        print(f"[port] window={label} ({start} -> {end}) ...", file=sys.stderr)
        btc_r = _run_coin_window("BTC", label, start, end, veto_on)
        sol_r = _run_coin_window("SOL", label, start, end, veto_on)
        if btc_r is None and sol_r is None:
            print(f"  {label}: SKIP (no data for either coin)", file=sys.stderr)
            continue
        btc_windows[label] = btc_r
        sol_windows[label] = sol_r
        if solo_mode:
            # In solo mode we don't synthesize a 50/50 (or any) blend.
            port_windows[label] = None
        else:
            port_windows[label] = _synth_portfolio_window(btc_r, sol_r, w_btc, w_sol)
        if btc_r:
            print(f"  BTC: trades={btc_r['trades']} return={btc_r['return_pct']:.4f}% "
                  f"dd={btc_r['max_dd_pct']:.4f}%", file=sys.stderr)
        if sol_r:
            print(f"  SOL: trades={sol_r['trades']} return={sol_r['return_pct']:.4f}% "
                  f"dd={sol_r['max_dd_pct']:.4f}%", file=sys.stderr)
        if not solo_mode and port_windows[label] is not None:
            pw = port_windows[label]
            print(f"  PORT: trades={pw['trades']} return={pw['return_pct']:.4f}% "
                  f"dd={pw['max_dd_pct']:.4f}%", file=sys.stderr)

    btc_agg = _aggregate_book(btc_windows, "BTC")
    sol_agg = _aggregate_book(sol_windows, "SOL")
    if solo_mode:
        port_agg = {
            "n_trades_total":   btc_agg["n_trades_total"] + sol_agg["n_trades_total"],
            "compounded_pct":   None,
            "windows_positive": "n/a",
            "per_window_return_pct": [],
            "psr_vs_hurdle":    None,
            "psr_interpretation": "n/a (solo arm)",
            "_all_pnl":         btc_agg["_all_pnl"] + sol_agg["_all_pnl"],
        }
    else:
        port_agg = _aggregate_book(port_windows, "PORT")
        # Methodology debt #2 fix: replace the stitched-trade-pool PSR proxy
        # with the equity-curve PSR computed from the time-aligned weighted-
        # sum portfolio equity series (per window) resampled to '1D'. The
        # legacy proxy is retained as ``psr_trade_pool_proxy`` for diff-
        # ability against pre-fix JSONs.
        port_eq_per_window = {
            label: pw["portfolio_equity_series"]
            for label, pw in port_windows.items()
            if pw is not None and pw.get("portfolio_equity_series") is not None
        }
        port_psr_true = aggregate_portfolio_psr(
            port_eq_per_window,
            resample_period="1D",
            sr_hurdle=0.0,
            confidence=0.95,
        )
        # Persist the literal daily-return arrays compute_psr was fed, so the
        # headline PSR is recomputable bit-for-bit from the JSON. Built by
        # calling the canonical public helper (equity_to_period_returns) on
        # the SAME dict in the SAME iteration order aggregate_portfolio_psr
        # uses; empty windows contribute [] per-window but are excluded from
        # `combined` exactly as the aggregator skips empty pieces, so combined
        # reproduces psr_n_periods and the headline PSR identically. These are
        # plain lists -> serialize cleanly without default=str truncation.
        psr_per_window_returns = {
            label: equity_to_period_returns(eq, resample_period="1D").tolist()
            for label, eq in port_eq_per_window.items()
        }
        _combined_pieces = [
            np.asarray(v, dtype=float)
            for v in psr_per_window_returns.values()
            if len(v) > 0
        ]
        psr_combined_returns = (
            np.concatenate(_combined_pieces).tolist()
            if _combined_pieces else []
        )
        port_agg["psr_per_window_returns"] = psr_per_window_returns
        port_agg["psr_combined_returns"] = psr_combined_returns
        port_agg["psr_trade_pool_proxy"] = port_agg.pop("psr_vs_hurdle")
        port_agg["psr_trade_pool_proxy_interpretation"] = port_agg.pop(
            "psr_interpretation"
        )
        port_agg["psr_equity_curve"] = port_psr_true["psr_equity_curve"]
        port_agg["psr_interpretation"] = port_psr_true["psr_interpretation"]
        port_agg["psr_n_periods"] = port_psr_true["n_periods_total"]
        port_agg["psr_resample_period"] = port_psr_true["resample_period"]
        port_agg["psr_per_window_n_periods"] = port_psr_true[
            "per_window_n_periods"
        ]
        port_agg["point_sharpe_period"] = port_psr_true["point_sharpe_period"]
        port_agg["sharpe_units"] = port_psr_true["sharpe_units"]
        # Headline `psr_vs_hurdle` now refers to the equity-curve metric.
        port_agg["psr_vs_hurdle"] = port_psr_true["psr_equity_curve"]

    # Strip the heavy _all_pnl arrays before serializing
    btc_pnl_all = btc_agg.pop("_all_pnl")
    sol_pnl_all = sol_agg.pop("_all_pnl")
    port_pnl_all = port_agg.pop("_all_pnl") if "_all_pnl" in port_agg else []

    # Lift summary
    btc_solo_psr = btc_agg["psr_vs_hurdle"]
    sol_solo_psr = sol_agg["psr_vs_hurdle"]
    lift_over_btc = None
    lift_over_sol = None
    if not solo_mode and port_agg["compounded_pct"] is not None:
        lift_over_btc = round(port_agg["compounded_pct"] - btc_agg["compounded_pct"], 4)
        lift_over_sol = round(port_agg["compounded_pct"] - sol_agg["compounded_pct"], 4)

    # Strip equity_series / timestamps before json (non-serializable)
    _DROP_KEYS = (
        "pnl_pct",
        "entry_times",
        "exit_times",
        "equity_series",
        "pnl_pct_trade_pool_proxy",
        "portfolio_equity_series",
    )

    def _clean(per_window: dict) -> dict:
        out = {}
        for label, r in per_window.items():
            if r is None:
                out[label] = None
                continue
            clean = {k: v for k, v in r.items() if k not in _DROP_KEYS}
            out[label] = clean
        return out

    out = {
        "arm":           arm,
        "allocation_btc": w_btc,
        "allocation_sol": w_sol,
        "veto_on":       veto_on,
        "cash_per_book": CASH,
        "commission":    COMMISSION,
        "margin":        MARGIN,
        "price_scale":   PRICE_SCALE,
        "locked_config": LOCKED,
        "windows":       [w[0] for w in WINDOWS if (not only_window or w[0] == only_window)],
        "per_window": {
            "BTC":  _clean(btc_windows),
            "SOL":  _clean(sol_windows),
            # Route PORT through _clean too: it strips _DROP_KEYS (incl. the
            # live ``portfolio_equity_series`` pandas Series, which json.dumps
            # default=str would otherwise stringify into a truncated, ellipsis-
            # bearing, forensically-dead repr). _clean no-ops on None entries.
            "PORT": _clean(port_windows) if not solo_mode else {},
        },
        "summary": {
            "BTC_solo":  btc_agg,
            "SOL_solo":  sol_agg,
            "Portfolio": port_agg,
        },
        "cross_arm_lift": {
            "btc_solo_psr":       btc_solo_psr,
            "sol_solo_psr":       sol_solo_psr,
            # Post-fix headline portfolio PSR (equity-curve, per-period).
            "portfolio_psr":      port_agg.get("psr_vs_hurdle"),
            "portfolio_psr_equity_curve": port_agg.get("psr_equity_curve"),
            "portfolio_psr_trade_pool_proxy": port_agg.get(
                "psr_trade_pool_proxy"
            ),
            "portfolio_psr_n_periods": port_agg.get("psr_n_periods"),
            "portfolio_psr_resample_period": port_agg.get(
                "psr_resample_period"
            ),
            "portfolio_sharpe_units": port_agg.get("sharpe_units"),
            "portfolio_compounded": port_agg["compounded_pct"],
            "portfolio_maxdd_proxy": (
                # Average of per-window maxdd as a proxy (NOT a true through-time
                # number — true MDD across windows would require a continuous
                # equity curve; this is a per-window indicator only).
                round(float(np.mean([
                    pw["max_dd_pct"] for pw in port_windows.values()
                    if pw is not None
                ])), 4) if not solo_mode and port_windows else None
            ),
            "lift_over_btc_solo": lift_over_btc,
            "lift_over_sol_solo": lift_over_sol,
        },
    }

    out_path = ROOT / "reports" / f"postfrac_mf_4h_btc_sol_portfolio_{arm}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[port] wrote {out_path}", file=sys.stderr)

    # Headline to stdout
    print(json.dumps({
        "arm":              arm,
        "btc_compounded":   btc_agg["compounded_pct"],
        "sol_compounded":   sol_agg["compounded_pct"],
        "port_compounded":  port_agg["compounded_pct"],
        "btc_solo_psr":     btc_solo_psr,
        "sol_solo_psr":     sol_solo_psr,
        # Post-fix headline = equity-curve PSR.
        "port_psr":         port_agg.get("psr_vs_hurdle"),
        "port_psr_equity_curve": port_agg.get("psr_equity_curve"),
        "port_psr_trade_pool_proxy": port_agg.get("psr_trade_pool_proxy"),
        "port_psr_n_periods": port_agg.get("psr_n_periods"),
        "lift_over_btc":    lift_over_btc,
        "lift_over_sol":    lift_over_sol,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
