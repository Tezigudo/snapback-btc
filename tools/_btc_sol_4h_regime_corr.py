"""BTC vs SOL 4H EMA200 regime correlation (GATE #3).

Loads BTC and SOL 4H parquets, computes EMA200(Close), derives a regime
indicator (+1 if Close > EMA200 else -1), aligns on common timestamps in
2022-01-01..2026-06-30 (or latest available), and reports:

  - Pearson correlation of the +1/-1 regime series
  - Spearman correlation
  - Regime agreement % (pct of bars where both regimes match)

A correlation >= 0.95 means the BTC×SOL cross-veto would never bind — kill
the portfolio idea before building the harness.

Outputs: reports/btc_sol_4h_regime_corr.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BTC_PARQ = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
SOL_PARQ = ROOT / "data" / "historical" / "SOL_USDT_USDT_4h.parquet"

WINDOW_START = "2022-01-01"
WINDOW_END = "2026-06-30"
EMA_PERIOD = 200


def load_4h(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def main() -> int:
    btc = load_4h(BTC_PARQ)
    sol = load_4h(SOL_PARQ)
    print(f"[corr] BTC 4h: {btc.index.min()} .. {btc.index.max()}  rows={len(btc)}", file=sys.stderr)
    print(f"[corr] SOL 4h: {sol.index.min()} .. {sol.index.max()}  rows={len(sol)}", file=sys.stderr)

    # Compute EMA200 BEFORE windowing so warmup is honored from each series' start.
    btc["ema200"] = ema(btc["Close"], EMA_PERIOD)
    sol["ema200"] = ema(sol["Close"], EMA_PERIOD)
    btc["regime"] = np.where(btc["Close"] > btc["ema200"], 1, -1)
    sol["regime"] = np.where(sol["Close"] > sol["ema200"], 1, -1)

    # Restrict to common window
    start_ts = pd.Timestamp(WINDOW_START)
    end_ts = pd.Timestamp(WINDOW_END) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    btc_w = btc.loc[(btc.index >= start_ts) & (btc.index <= end_ts)]
    sol_w = sol.loc[(sol.index >= start_ts) & (sol.index <= end_ts)]
    # Drop rows where EMA200 still NaN
    btc_w = btc_w.dropna(subset=["ema200"])
    sol_w = sol_w.dropna(subset=["ema200"])

    # Inner-join on timestamps
    merged = btc_w[["regime"]].rename(columns={"regime": "btc_regime"}).join(
        sol_w[["regime"]].rename(columns={"regime": "sol_regime"}),
        how="inner",
    )
    n = len(merged)
    print(f"[corr] aligned bars: {n}  range {merged.index.min()} .. {merged.index.max()}", file=sys.stderr)

    if n < 100:
        print("ERROR: insufficient aligned bars", file=sys.stderr)
        return 1

    btc_arr = merged["btc_regime"].astype(float).values
    sol_arr = merged["sol_regime"].astype(float).values

    pearson = float(np.corrcoef(btc_arr, sol_arr)[0, 1])
    # Spearman: Pearson of average-ranks. For a binary +1/-1 series Spearman
    # collapses to Pearson, but compute explicitly via ranks (no scipy).
    btc_ranks = pd.Series(btc_arr).rank().values
    sol_ranks = pd.Series(sol_arr).rank().values
    spearman = float(np.corrcoef(btc_ranks, sol_ranks)[0, 1])
    agree_pct = float((btc_arr == sol_arr).mean() * 100.0)

    # Regime distribution
    btc_long_pct = float((btc_arr > 0).mean() * 100.0)
    sol_long_pct = float((sol_arr > 0).mean() * 100.0)

    out = {
        "window_start":     str(merged.index.min()),
        "window_end":       str(merged.index.max()),
        "n_bars_aligned":   n,
        "ema_period":       EMA_PERIOD,
        "pearson":          round(pearson, 6),
        "spearman":         round(spearman, 6),
        "regime_agree_pct": round(agree_pct, 4),
        "btc_long_regime_pct": round(btc_long_pct, 4),
        "sol_long_regime_pct": round(sol_long_pct, 4),
        "gate_threshold":   0.95,
        "gate_pass":        pearson < 0.95,
    }

    out_path = ROOT / "reports" / "btc_sol_4h_regime_corr.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[corr] wrote {out_path}", file=sys.stderr)

    print("\n=== SUMMARY ===")
    print(json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
