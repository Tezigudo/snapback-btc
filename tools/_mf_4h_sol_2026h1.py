"""SOL multifactor-v1 + 4H gate — 2026 H1 re-validation (GATE #1).

Runs the locked multifactor config + use_mtf_4h_gate=True on SOL across the
single window 2026-01-01..2026-06-30 (or as far as the parquet extends).
Uses commission=0.00075 (15bps RT — TODO_LEG gate #5).

Prior in-house finding (2022-2025): +41.93%, 5/5 wins, PSR 0.894.
This script tests whether that edge survives into 2026 H1.

Outputs: reports/mf_4h_sol_2026h1.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402

CASH = 1_000_000.0
COMMISSION = 0.00075  # 15 bps RT (7.5 bps per side)
MARGIN = 1.0 / 20

WINDOW_LABEL = "2026H1"
WINDOW_START = "2026-01-01"
WINDOW_END = "2026-06-30"

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

PARQ_15M = ROOT / "data" / "historical" / "SOL_USDT_USDT_15m.parquet"
PARQ_4H = ROOT / "data" / "historical" / "SOL_USDT_USDT_4h.parquet"


def _load_slice(parquet: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sliced) == 0:
        raise ValueError(f"Empty slice {start}..{end} from {parquet.name}")
    return sliced


def run_one(df: pd.DataFrame, config: dict) -> dict:
    bt = Backtest(df, DayTradeMultiFactorBTC, cash=CASH, commission=COMMISSION,
                  margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run(**config)
    trades_df = getattr(stats, "_trades", None)
    pnl_pct = []
    if trades_df is not None and len(trades_df):
        if "ReturnPct" in trades_df.columns:
            pnl_pct = (trades_df["ReturnPct"].values * 100.0).tolist()
    return {
        "trades":        int(stats.get("# Trades", 0)),
        "return_pct":    float(stats.get("Return [%]", 0.0) or 0.0),
        "max_dd_pct":    float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
        "win_rate_pct":  float(stats.get("Win Rate [%]") or 0.0),
        "equity_final":  float(stats.get("Equity Final [$]", CASH) or CASH),
        "sharpe_ratio":  float(stats.get("Sharpe Ratio", 0.0) or 0.0),
        "pnl_pct":       pnl_pct,
        "first_bar":     str(df.index.min()),
        "last_bar":      str(df.index.max()),
        "n_bars":        int(len(df)),
    }


def main() -> int:
    print(f"[mf_4h_sol_2026h1] window={WINDOW_LABEL} {WINDOW_START}..{WINDOW_END}", file=sys.stderr)
    print(f"  15m parquet : {PARQ_15M.name}", file=sys.stderr)
    print(f"  4H  parquet : {PARQ_4H.name}", file=sys.stderr)
    print(f"  cash={CASH} commission={COMMISSION} margin={MARGIN}", file=sys.stderr)

    config = {
        **LOCKED,
        "use_mtf_4h_gate":      True,
        "mtf_4h_parquet_path":  str(PARQ_4H),
    }

    try:
        df = _load_slice(PARQ_15M, WINDOW_START, WINDOW_END)
    except ValueError as exc:
        print(f"DATA MISSING: {exc}", file=sys.stderr)
        out = {"data_missing": True, "error": str(exc)}
        out_path = ROOT / "reports" / "mf_4h_sol_2026h1.json"
        out_path.write_text(json.dumps(out, indent=2, default=str))
        return 1

    result = run_one(df, config)
    pnl = np.asarray(result["pnl_pct"])

    # Point Sharpe + sign proxy (single window — true PSR needs multi-window).
    if len(pnl) >= 2:
        mu = float(pnl.mean())
        sd = float(pnl.std(ddof=1))
        point_sharpe = (mu / sd) if sd > 1e-12 else 0.0
        # Bailey-Lopez de Prado PSR for single-window pnl stream (trade-level
        # Sharpe vs zero hurdle).  Skew/kurt correction.
        if sd > 1e-12 and len(pnl) >= 4:
            skew = float(pd.Series(pnl).skew())
            kurt = float(pd.Series(pnl).kurt())  # excess kurtosis
            n = len(pnl)
            sr = point_sharpe
            # PSR vs 0
            denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt) / 4.0) * (sr ** 2)))
            z = (sr - 0.0) * math.sqrt(n - 1) / denom
            # Standard normal CDF
            psr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            psr_flag = "single_window_proxy"
        else:
            psr = 0.0
            psr_flag = "insufficient_trades"
    else:
        point_sharpe = 0.0
        psr = 0.0
        psr_flag = "insufficient_trades"

    summary = {
        "window":              WINDOW_LABEL,
        "window_start":        WINDOW_START,
        "window_end":          WINDOW_END,
        "first_bar":           result["first_bar"],
        "last_bar":            result["last_bar"],
        "n_bars":              result["n_bars"],
        "n_trades":            result["trades"],
        "compounded_pct":      round(result["return_pct"], 4),
        "win_rate_pct":        round(result["win_rate_pct"], 4),
        "max_dd_pct":          round(result["max_dd_pct"], 4),
        "equity_final":        round(result["equity_final"], 2),
        "sharpe_ratio_bt":     round(result["sharpe_ratio"], 4),
        "point_sharpe_trade":  round(point_sharpe, 4),
        "psr_single_window":   round(psr, 4),
        "psr_flag":            psr_flag,
        "commission":          COMMISSION,
    }

    out = {
        "strategy":  "multifactor-v1-4h-gate",
        "coin":      "SOL",
        "window":    WINDOW_LABEL,
        "config":    {**LOCKED, "use_mtf_4h_gate": True},
        "summary":   summary,
    }

    out_path = ROOT / "reports" / "mf_4h_sol_2026h1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[mf_4h_sol_2026h1] wrote {out_path}", file=sys.stderr)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
