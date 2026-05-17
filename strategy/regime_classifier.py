"""
Regime classifier — "is the market trending or choppy right now?"

Uses Kaufman's Efficiency Ratio (ER):

    ER = |close[t] - close[t-N]| / sum(|close[i] - close[i-1]| for i in (t-N, t])

ER = 1.0 when price moves in a perfect straight line (max efficient = trend)
ER = 0.0 when price oscillates a lot without going anywhere (max inefficient = chop)

A 30-day ER around 0.25-0.35 is a common "trending" threshold for daily bars.
On 4h bars, 30 days = 180 bars; tune the threshold empirically.

This module exposes pure pandas functions — no Strategy class, no
dependence on backtesting.py. Used as a gate by Donchian-v3 (and any
future strategy that wants regime-conditional execution).

Quick CLI to sanity-check the classifier on a known period:
    python -m strategy.regime_classifier --tf 4h --start 2022-01-01 --end 2022-06-30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman's ER over a rolling window of `period` bars.

    Returns a pandas Series same index as `close`, with NaN for the first
    `period` bars (not enough lookback).
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period, min_periods=period).sum()
    er = direction / volatility.replace(0, np.nan)
    return er


def is_trending(close: pd.Series, period: int, threshold: float) -> pd.Series:
    """Boolean series — True when ER is at or above the trending threshold."""
    return efficiency_ratio(close, period) >= threshold


def ema_slope_strength(close: pd.Series, ema_period: int, slope_window: int) -> pd.Series:
    """|EMA slope per bar| / price, expressed in % per bar.

    Measures how steeply the long EMA is moving. High = strong directional
    drift; near zero = sideways. Unlike ER, this isn't confused by
    intraday volatility because the EMA smooths over it.

    `ema_period` = how long-term the trend reference is
    `slope_window` = over how many bars to measure the slope
    """
    ema = close.ewm(span=ema_period, adjust=False).mean()
    slope_per_bar = (ema - ema.shift(slope_window)) / slope_window
    return (slope_per_bar / close).abs() * 100.0


def ema_slope_signed(close: pd.Series, ema_period: int, slope_window: int) -> pd.Series:
    """Signed EMA slope per bar / price (% per bar, with sign).

    Positive = EMA going up (uptrend), negative = EMA going down (downtrend).
    Use with a threshold like ±0.03% to gate directional entries.
    """
    ema = close.ewm(span=ema_period, adjust=False).mean()
    slope_per_bar = (ema - ema.shift(slope_window)) / slope_window
    return (slope_per_bar / close) * 100.0


def is_trending_by_slope(
    close: pd.Series,
    ema_period: int,
    slope_window: int,
    threshold_pct: float,
) -> pd.Series:
    """Boolean Series — True when EMA is sloping faster than the threshold."""
    return ema_slope_strength(close, ema_period, slope_window) >= threshold_pct


def regime_summary(close: pd.Series, period: int) -> dict:
    """Summarise the regime distribution of a price series.

    Returns both ER (Kaufman) and slope-strength metrics. Slope-strength
    works better on BTC perp because BTC's intraday volatility crushes the
    ER signal (a strongly-trending month with wiggly intraday bars looks
    'inefficient' to ER but obviously trending to the EMA slope).
    """
    er = efficiency_ratio(close, period).dropna()
    slope_30 = ema_slope_strength(close, ema_period=120, slope_window=30).dropna()
    slope_90 = ema_slope_strength(close, ema_period=200, slope_window=90).dropna()
    out = {"bars": int(len(er))}
    if not er.empty:
        out.update({
            "er_median": float(er.median()),
            "er_p90": float(er.quantile(0.9)),
        })
    if not slope_30.empty:
        out.update({
            "slope30_median_pct": float(slope_30.median()),
            "slope30_mean_pct": float(slope_30.mean()),
            "slope30_p10_pct": float(slope_30.quantile(0.1)),
            "slope30_p90_pct": float(slope_30.quantile(0.9)),
            "slope30_pct_trending_0.05": float((slope_30 >= 0.05).mean() * 100),
            "slope30_pct_trending_0.10": float((slope_30 >= 0.10).mean() * 100),
            "slope30_pct_trending_0.15": float((slope_30 >= 0.15).mean() * 100),
        })
    if not slope_90.empty:
        out.update({
            "slope90_median_pct": float(slope_90.median()),
            "slope90_pct_trending_0.05": float((slope_90 >= 0.05).mean() * 100),
            "slope90_pct_trending_0.10": float((slope_90 >= 0.10).mean() * 100),
        })
    return out


def _main() -> int:
    p = argparse.ArgumentParser(description="Inspect regime distribution on cached data.")
    p.add_argument("--tf", default="4h")
    p.add_argument("--start", required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--end", required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--period", type=int, default=120,
                   help="ER lookback in bars (default 120 = 20d at 4h, 30d at 1d, 30h at 15m)")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    args = p.parse_args()

    from exchange.data import load_klines  # local import keeps this file fast to import

    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    days_back = max((end - start).days + 10, 30)
    k = load_klines(args.symbol, args.tf, days_back=days_back, end=end)
    if k.index.tz is not None:
        k.index = k.index.tz_convert("UTC").tz_localize(None)
    naive_start = start.replace(tzinfo=None)
    naive_end = end.replace(tzinfo=None)
    window = k.loc[naive_start:naive_end]
    close = window["close"] if "close" in window.columns else window["Close"]

    s = regime_summary(close, args.period)
    print(f"Window: {window.index[0]} → {window.index[-1]}  ({s['bars']} bars at {args.tf})")
    print(f"  ER (period={args.period}): median {s.get('er_median', 0):.3f}  p90 {s.get('er_p90', 0):.3f}")
    print(f"  EMA-slope 30bar (% per bar): median {s.get('slope30_median_pct', 0):.4f}  "
          f"p10 {s.get('slope30_p10_pct', 0):.4f}  p90 {s.get('slope30_p90_pct', 0):.4f}")
    print(f"  EMA-slope 30bar % of bars TRENDING:")
    print(f"    threshold 0.05%/bar : {s.get('slope30_pct_trending_0.05', 0):5.1f}%")
    print(f"    threshold 0.10%/bar : {s.get('slope30_pct_trending_0.10', 0):5.1f}%")
    print(f"    threshold 0.15%/bar : {s.get('slope30_pct_trending_0.15', 0):5.1f}%")
    print(f"  EMA-slope 90bar median: {s.get('slope90_median_pct', 0):.4f}%/bar")
    print(f"  EMA-slope 90bar % >= 0.05%/bar: {s.get('slope90_pct_trending_0.05', 0):.1f}%")
    print(f"  EMA-slope 90bar % >= 0.10%/bar: {s.get('slope90_pct_trending_0.10', 0):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
