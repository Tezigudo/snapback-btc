"""
Indicators with no external dependency beyond pandas/numpy.

We intentionally avoid pandas-ta here because (a) it ships unstable across
numpy 1/2, (b) it's easy to misuse without realising it inserts lookahead via
fillna, and (c) writing these by hand documents the exact convention used by
the strategy. If you swap implementations later, update the tests in
tests/test_indicators.py to match.

All functions take a pandas Series indexed by time and return a Series of the
same shape. NaN-fill behaviour: leave NaNs at the head (warm-up). Never fill
NaNs in the middle of a series — that hides data gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI. Standard formula:
        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)
    Uses Wilder smoothing (EWM with alpha=1/period, adjust=False) — matches
    most charting platforms.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 (only gains), RSI = 100.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return out


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average via pandas .ewm(span=...)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's ATR: smoothed True Range.
        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = Wilder EMA(TR, period)
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return series.rolling(window=period, min_periods=period).mean()
