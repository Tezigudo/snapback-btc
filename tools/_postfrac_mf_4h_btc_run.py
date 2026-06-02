"""Post-fractional-sizing re-validation: multifactor-v1 + 4H gate on BTC.

Mirrors tools/_postfrac_mf_baseline.py — applies PRICE_SCALE=0.001 to BOTH
the 15m OHLC slice AND the 4H parquet that the strategy loads internally.
Without scaling the 4H parquet, the 4H EMA200 stays on the unscaled price
plane while the 15m close lives on the scaled one — the gate would always
reject (long: scaled close ≪ unscaled EMA200) and produce zero trades.

The 4H parquet is rewritten ONCE to a temp path under reports/_tmp/ and
the strategy is pointed at it via LOCKED["mtf_4h_parquet_path"]. The
on-disk canonical parquet at data/historical/BTC_USDT_USDT_4h.parquet is
NOT touched (live bot reads it).

Output:
    reports/postfrac_mf_4h_btc.json
    reports/_postfrac_mf_4h_btc_<window>.csv

This patch fixes the "bit-exact match to pre-fix" artifact flagged in
fractional_sizing_refactor_verdict.md — the previous version of this
runner reran the OLD code path (no scaling) and trivially matched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]

PARQ_15m = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
PARQ_FUND = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
PARQ_4H_SRC = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
PARQ_4H_SCALED = ROOT / "reports" / "_tmp" / f"BTC_USDT_USDT_4h_scaled_{PRICE_SCALE}.parquet"


def _ensure_scaled_4h_parquet() -> Path:
    """Write a price-scaled copy of the 4H parquet to reports/_tmp/.

    Idempotent — if the temp file exists, returns its path without rewriting.
    The original canonical parquet is left untouched (live bot reads it).
    """
    PARQ_4H_SCALED.parent.mkdir(parents=True, exist_ok=True)
    if PARQ_4H_SCALED.exists():
        return PARQ_4H_SCALED
    df = pd.read_parquet(PARQ_4H_SRC)
    for col in ("open", "high", "low", "close", "Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] * PRICE_SCALE
    df.to_parquet(PARQ_4H_SCALED)
    return PARQ_4H_SCALED


# Locked params.yaml values (copied from tools/run_mf_deepening.py LOCKED dict)
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
    # Point at the SCALED 4H parquet so EMA200 lives on the same price plane
    # as the scaled 15m feed below.
    "mtf_4h_parquet_path":         str(_ensure_scaled_4h_parquet()),
}


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame:
    """Load 15m BTC parquet, slice [start, end], scale OHLC by PRICE_SCALE.

    Volume and Funding stay un-scaled (dimensionless / ratio).
    """
    df = pd.read_parquet(PARQ_15m)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if PARQ_FUND.exists():
        fund = pd.read_parquet(PARQ_FUND)
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
        raise ValueError(f"Empty slice {start}..{end}")
    for col in ("Open", "High", "Low", "Close"):
        if col in sliced.columns:
            sliced[col] = sliced[col] * PRICE_SCALE
    return sliced


def run_window(label: str, start: str, end: str, save_csv_path: Path) -> dict:
    df = _load_slice_scaled(start, end)
    bt = Backtest(df, DayTradeMultiFactorBTC, cash=CASH, commission=COMMISSION,
                  margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run(**LOCKED)
    trades_df = getattr(stats, "_trades", None)
    pnl_pct: list[float] = []
    if trades_df is not None and len(trades_df):
        if "ReturnPct" in trades_df.columns:
            pnl_pct = (trades_df["ReturnPct"].values * 100.0).tolist()
            # Save per-window CSV
            out = pd.DataFrame({"pnl_pct": pnl_pct})
            out["window_start"] = start
            out["window_end"] = end
            out.to_csv(save_csv_path, index=False)
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
        "csv_path":      str(save_csv_path),
    }


def main() -> int:
    out: dict = {
        "strategy_id":   "mf_4h_btc",
        "strategy_class": "strategy.signals_multifactor:DayTradeMultiFactorBTC",
        "cash":           CASH,
        "commission":     COMMISSION,
        "margin":         MARGIN,
        "price_scale":    PRICE_SCALE,
        "config":         LOCKED,
        "windows":        [w[0] for w in WINDOWS],
        "per_window":     {},
    }

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    all_pnl: list[float] = []
    n_trades_total = 0
    compounded = 1.0
    n_pos = 0
    per_window_returns: list[float] = []

    for label, start, end in WINDOWS:
        csv_path = reports / f"_postfrac_mf_4h_btc_{label}.csv"
        print(f"[postfrac] window={label} ({start} -> {end}) ...", file=sys.stderr)
        r = run_window(label, start, end, csv_path)
        out["per_window"][label] = {k: v for k, v in r.items() if k != "pnl_pct"}
        n_trades_total += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_window_returns.append(round(r["return_pct"], 4))
        if r["return_pct"] > 0:
            n_pos += 1
        print(f"  trades={r['trades']} return={r['return_pct']:.4f}% dd={r['max_dd_pct']:.4f}%",
              file=sys.stderr)

    # Aggregate CSV for psr_eval
    agg_csv = reports / "_postfrac_mf_4h_btc_AGGREGATE.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)

    psr = compute_psr(np.asarray(all_pnl), sr_hurdle=0.0, confidence=0.95) if len(all_pnl) >= 2 else {
        "n_trades": len(all_pnl), "psr_vs_hurdle": 0.0, "interpretation": "insufficient_evidence",
    }

    out["aggregate"] = {
        "n_trades_total":        n_trades_total,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{len(WINDOWS)}",
        "per_window_return_pct": per_window_returns,
        "aggregate_csv":         str(agg_csv),
    }
    out["psr"] = psr

    out_path = reports / "postfrac_mf_4h_btc.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[postfrac] wrote {out_path}", file=sys.stderr)
    print(json.dumps({
        "n_trades_total":  n_trades_total,
        "compounded_pct":  out["aggregate"]["compounded_pct"],
        "psr_vs_hurdle":   psr.get("psr_vs_hurdle"),
        "interpretation":  psr.get("interpretation"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
