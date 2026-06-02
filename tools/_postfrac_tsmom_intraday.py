"""Post-fractional-sizing validation: IntradayTSMOM_BTC on BTC.

5 OOS windows × CASH=$1M × PRICE_SCALE=0.001 × COMMISSION=0.0005 × MARGIN=1/20.

Pre-scales the 1H AND 4H parquets to reports/_tmp/ (same pattern as
tools/_postfrac_mf_4h_btc_run.py) so the ATR series live on the same price
plane as the scaled 15m feed.

Writes:
  reports/_postfrac_tsmom_intraday_<window>.csv (per-window trades)
  reports/_postfrac_tsmom_intraday_AGGREGATE.csv
  reports/postfrac_tsmom_intraday.json (full result + PSR)
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

from strategy.signals_tsmom_intraday import IntradayTSMOM_BTC  # noqa: E402
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
PARQ_1H_SRC = ROOT / "data" / "historical" / "BTC_USDT_USDT_1h.parquet"
PARQ_4H_SRC = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
PARQ_1H_SCALED = ROOT / "reports" / "_tmp" / f"BTC_USDT_USDT_1h_scaled_{PRICE_SCALE}.parquet"
PARQ_4H_SCALED = ROOT / "reports" / "_tmp" / f"BTC_USDT_USDT_4h_scaled_{PRICE_SCALE}.parquet"


def _ensure_scaled_parquet(src: Path, dst: Path) -> Path:
    """Write a PRICE_SCALE'd copy of a higher-TF parquet to reports/_tmp/.

    Idempotent.  Live-bot canonical parquet is NOT touched.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    df = pd.read_parquet(src)
    for col in ("open", "high", "low", "close", "Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] * PRICE_SCALE
    df.to_parquet(dst)
    return dst


# Locked configuration (per spec).
LOCKED = {
    "morning_hour":         12,
    "threshold_std_coef":   0.3,
    "outlier_std_coef":     5.0,
    "risk_per_trade_pct":   0.5,
    "leverage":             20,
    "atr_stop_x":           1.0,
    "atr_tp_x":             2.0,
    "use_low_vol_skip":     True,
    "low_vol_lookback_days": 90,
    "low_vol_bottom_pct":   0.10,
    "atr_1h_period":        14,
    "atr_4h_period":        14,
    "allow_longs":          True,
    "allow_shorts":         True,
    "atr1h_parquet_path":   str(_ensure_scaled_parquet(PARQ_1H_SRC, PARQ_1H_SCALED)),
    "atr4h_parquet_path":   str(_ensure_scaled_parquet(PARQ_4H_SRC, PARQ_4H_SCALED)),
}


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(PARQ_15m)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Funding column (kept un-scaled; not used by this strategy but harness-compatible).
    if PARQ_FUND.exists():
        fund = pd.read_parquet(PARQ_FUND)
        if fund.index.tz is not None:
            fund.index = fund.index.tz_localize(None)
        left = pd.DataFrame(index=df.index)
        right = pd.DataFrame({"Funding": fund["funding_rate"].values}, index=fund.index)
        merged = pd.merge_asof(left, right, left_index=True, right_index=True, direction="backward")
        df["Funding"] = merged["Funding"].fillna(0.0).values
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sl) == 0:
        raise ValueError(f"Empty slice {start}..{end}")
    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * PRICE_SCALE
    return sl


def run_window(label: str, start: str, end: str, csv_path: Path) -> dict:
    df = _load_slice_scaled(start, end)
    bt = Backtest(
        df, IntradayTSMOM_BTC,
        cash=CASH, commission=COMMISSION, margin=MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run(**LOCKED)
    trades_df = getattr(stats, "_trades", None)
    pnl_pct: list[float] = []
    if trades_df is not None and len(trades_df):
        if "ReturnPct" in trades_df.columns:
            pnl_pct = (trades_df["ReturnPct"].values * 100.0).tolist()
            out = pd.DataFrame({"pnl_pct": pnl_pct})
            out["window_start"] = start
            out["window_end"] = end
            out.to_csv(csv_path, index=False)
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
        "csv_path":      str(csv_path),
    }


def main() -> int:
    t0 = time.time()
    out: dict = {
        "strategy_id":    "tsmom_intraday",
        "strategy_class": "strategy.signals_tsmom_intraday:IntradayTSMOM_BTC",
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
        csv_path = reports / f"_postfrac_tsmom_intraday_{label}.csv"
        tw = time.time()
        print(f"[postfrac_tsmom_intraday] window={label} ({start} -> {end}) ...",
              file=sys.stderr)
        r = run_window(label, start, end, csv_path)
        out["per_window"][label] = {k: v for k, v in r.items() if k != "pnl_pct"}
        n_trades_total += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_window_returns.append(round(r["return_pct"], 4))
        if r["return_pct"] > 0:
            n_pos += 1
        print(
            f"  trades={r['trades']:3d}  ret={r['return_pct']:+8.2f}%  "
            f"dd={r['max_dd_pct']:+7.2f}%  win={r['win_rate_pct']:5.2f}%  "
            f"({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )

    agg_csv = reports / "_postfrac_tsmom_intraday_AGGREGATE.csv"
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
    out["elapsed_sec"] = round(time.time() - t0, 2)

    out_path = reports / "postfrac_tsmom_intraday.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[postfrac_tsmom_intraday] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    print(json.dumps({
        "n_trades_total":  n_trades_total,
        "compounded_pct":  out["aggregate"]["compounded_pct"],
        "windows_positive": out["aggregate"]["windows_positive"],
        "psr_vs_hurdle":   psr.get("psr_vs_hurdle"),
        "interpretation":  psr.get("interpretation"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
