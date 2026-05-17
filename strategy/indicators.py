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


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classic MACD (Moving Average Convergence Divergence).

    Returns (macd_line, signal_line, histogram).
      macd_line = EMA(close, fast) - EMA(close, slow)
      signal_line = EMA(macd_line, signal)
      histogram = macd_line - signal_line

    Histogram > 0 = bullish momentum, < 0 = bearish.
    Histogram CROSSING zero is a tradeable signal (momentum direction change).
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bullish_engulfing(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Bullish engulfing pattern: today's GREEN body fully engulfs yesterday's RED body.

    Web research (multiple backtests on crypto): one of the highest win-rate
    reversal patterns when paired with volume confirmation (~65-75% in
    BTC/major pairs). Used here as a CONFIRMATION filter, not a sole entry
    signal.

    Returns a boolean Series — True on the bar where the engulfing completes.
    """
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    # Previous bar must be red (close < open).
    prev_red = prev_close < prev_open
    # Current bar must be green (close > open).
    cur_green = close > open_
    # Current body engulfs previous body (open below prev close, close above prev open).
    engulf = (open_ <= prev_close) & (close >= prev_open)
    return prev_red & cur_green & engulf


def bearish_engulfing(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Bearish engulfing: today's RED body fully engulfs yesterday's GREEN body."""
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_green = prev_close > prev_open
    cur_red = close < open_
    engulf = (open_ >= prev_close) & (close <= prev_open)
    return prev_green & cur_red & engulf


def hammer(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    body_max_pct: float = 0.3,
    lower_shadow_min_x: float = 2.0,
) -> pd.Series:
    """Hammer pattern (bullish reversal): small body at top, long lower shadow.

    body_max_pct: body size as fraction of total range (0.3 = small body)
    lower_shadow_min_x: lower shadow must be >= N × body size
    """
    body = (close - open_).abs()
    rng = high - low
    upper_shadow = high - close.where(close > open_, open_)
    lower_shadow = open_.where(close > open_, close) - low
    small_body = body <= body_max_pct * rng.replace(0, np.nan)
    long_lower = lower_shadow >= lower_shadow_min_x * body.replace(0, np.nan)
    short_upper = upper_shadow <= body  # rejection from below
    return small_body & long_lower & short_upper
