"""Phase 1 of the vol-regime switcher: empirically test the hypothesis.

Claim: v1 (mean-reversion) outperforms in low-vol days; Donchian-v3 cons
(breakout) outperforms in high-vol days. If true, a switcher beats 50/50.

Method:
  1. Compute daily ATR(14) percentile rank (rolling 90d, shifted 1d to avoid
     lookahead) on BTC/USDT:USDT 1d klines.
  2. Load v1 and Donchian-cons daily equity curves from the 6.7-year backtest.
  3. Compute daily simple returns from normalised equity.
  4. Bin each day by ATR percentile decile.
  5. Per decile: mean v1 daily return, mean Donchian daily return, t-stat of
     difference.
  6. Verdict: does v1 dominate low deciles? Does Donchian dominate high
     deciles? If clean separation → proceed to Phase 2. If muddled → kill it.

Quick CLI: `uv run python tools/regime_switcher_phase1.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr  # noqa: E402

DATA = ROOT / "data" / "historical"
REPORTS = ROOT / "reports"

LOOKBACK = 90
N_BINS = 10


def load_1d_atr_pctile() -> pd.Series:
    """Daily ATR(14) percentile rank, shifted 1d so today uses YESTERDAY's
    rank — no lookahead."""
    # Use 1h klines and resample to daily — 4h has 6 bars/day which is fine
    # but 1h gives cleaner High/Low.
    k1h = pd.read_parquet(DATA / "BTC_USDT_USDT_1h.parquet")
    ts_col = next((c for c in k1h.columns if c.lower() in ("timestamp", "ts", "time", "datetime")), None)
    if ts_col:
        k1h["_ts"] = pd.to_datetime(k1h[ts_col], utc=True)
        k1h = k1h.set_index("_ts")
    k1h = k1h.sort_index()
    # tz-strip for daily resample alignment
    if k1h.index.tz is not None:
        k1h.index = k1h.index.tz_convert("UTC").tz_localize(None)

    daily = k1h[["high", "low", "close"]].resample("1D").agg(
        {"high": "max", "low": "min", "close": "last"}
    )
    a = atr(daily["high"], daily["low"], daily["close"], 14)
    pct = a.rolling(LOOKBACK, min_periods=30).apply(
        lambda s: s.rank(pct=True).iloc[-1] if len(s) else np.nan, raw=False
    )
    return pct.shift(1).dropna()  # no lookahead


def load_daily_returns(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    eq = df["equity_norm"]
    if eq.index.tz is not None:
        eq.index = eq.index.tz_convert("UTC").tz_localize(None)
    # Already daily, but be defensive
    eq_d = eq.resample("1D").last().ffill()
    return eq_d.pct_change().dropna()


def _latest_ts() -> str:
    # Find the latest full_history run
    cands = sorted(REPORTS.glob("full_history_*_v1_equity.csv"))
    if not cands:
        raise RuntimeError("no equity CSVs found")
    return cands[-1].stem.replace("full_history_", "").replace("_v1_equity", "")


def main() -> int:
    ts = _latest_ts()
    print(f"Using backtest equity from run {ts}")
    pct = load_1d_atr_pctile()
    v1_r = load_daily_returns(REPORTS / f"full_history_{ts}_v1_equity.csv")
    d3_r = load_daily_returns(REPORTS / f"full_history_{ts}_d3cons_equity.csv")

    # Align all three on common index
    df = pd.concat({"pct": pct, "v1": v1_r, "d3": d3_r}, axis=1).dropna()
    print(f"aligned days: {len(df):,} ({df.index.min().date()} → {df.index.max().date()})")

    # Bin by percentile
    df["bin"] = pd.cut(
        df["pct"],
        bins=np.linspace(0.0, 1.0, N_BINS + 1),
        labels=[f"D{i+1}" for i in range(N_BINS)],
        include_lowest=True,
    )

    grouped = df.groupby("bin", observed=True).agg(
        n=("v1", "count"),
        v1_mean=("v1", "mean"),
        v1_std=("v1", "std"),
        d3_mean=("d3", "mean"),
        d3_std=("d3", "std"),
    )
    grouped["v1_ann_pct"] = grouped["v1_mean"] * 365 * 100
    grouped["d3_ann_pct"] = grouped["d3_mean"] * 365 * 100
    grouped["edge_d3_vs_v1_bps"] = (grouped["d3_mean"] - grouped["v1_mean"]) * 10000

    # t-stat for per-bin difference of means
    def t_stat(g):
        diff = g["d3"] - g["v1"]
        if len(diff) < 5 or diff.std() == 0:
            return np.nan
        return diff.mean() / (diff.std() / np.sqrt(len(diff)))
    tstats = df.groupby("bin", observed=True).apply(t_stat, include_groups=False)
    grouped["t_d3_minus_v1"] = tstats

    print("\n=== Per-ATR-decile daily returns (v1 mean-reversion vs Donchian-v3 cons breakout) ===")
    print(grouped.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Verdict
    low_deciles = ["D1", "D2", "D3"]
    high_deciles = ["D8", "D9", "D10"]
    low_v1_wins = sum(1 for d in low_deciles if d in grouped.index and grouped.loc[d, "v1_mean"] > grouped.loc[d, "d3_mean"])
    high_d3_wins = sum(1 for d in high_deciles if d in grouped.index and grouped.loc[d, "d3_mean"] > grouped.loc[d, "v1_mean"])

    print("\n=== Hypothesis check ===")
    print(f"Low-vol deciles where v1 > Donchian: {low_v1_wins}/3")
    print(f"High-vol deciles where Donchian > v1: {high_d3_wins}/3")

    # Per-strategy where it wins
    v1_best_bin = grouped["v1_mean"].idxmax()
    d3_best_bin = grouped["d3_mean"].idxmax()
    print(f"\nv1 best decile: {v1_best_bin} ({grouped.loc[v1_best_bin, 'v1_ann_pct']:+.1f}% ann)")
    print(f"d3 best decile: {d3_best_bin} ({grouped.loc[d3_best_bin, 'd3_ann_pct']:+.1f}% ann)")

    # Theoretical max: take whichever strategy has higher mean per bin
    df["best_daily_ret"] = np.where(df["v1"] > df["d3"], df["v1"], df["d3"])
    # Realistic switcher: use whichever has higher EXPECTED mean for each bin
    bin_winners = grouped.idxmax(axis=1).map({"v1_mean": "v1", "d3_mean": "d3"})
    bin_winners = grouped[["v1_mean", "d3_mean"]].idxmax(axis=1).str.replace("_mean", "")
    df["regime_choice"] = df["bin"].map(bin_winners.to_dict())
    df["switched_ret"] = df.apply(lambda r: r["v1"] if r["regime_choice"] == "v1" else r["d3"], axis=1)

    # Cumulative returns of each strategy + the perfect-switcher
    cum_v1 = (1 + df["v1"]).prod() - 1
    cum_d3 = (1 + df["d3"]).prod() - 1
    cum_50_50 = (1 + 0.5 * df["v1"] + 0.5 * df["d3"]).prod() - 1
    cum_switched = (1 + df["switched_ret"]).prod() - 1

    print(f"\n=== Cumulative returns over {len(df):,} aligned days ===")
    print(f"v1 alone:                {cum_v1 * 100:+.1f}%")
    print(f"Donchian-cons alone:     {cum_d3 * 100:+.1f}%")
    print(f"50/50 combined:          {cum_50_50 * 100:+.1f}%")
    print(f"Perfect bin-switcher:    {cum_switched * 100:+.1f}%  (in-sample upper bound)")

    # Save grouped for next phase
    out = REPORTS / f"regime_phase1_{ts}.json"
    grouped_save = grouped.reset_index()
    grouped_save["bin"] = grouped_save["bin"].astype(str)
    out.write_text(grouped_save.to_json(orient="records", indent=2))
    print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
