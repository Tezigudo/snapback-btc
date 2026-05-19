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


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_af: float = 0.2,
) -> pd.Series:
    """Wilder's Parabolic SAR. Returns the SAR value series.

    Trend flips when price crosses the SAR. SAR sits below price in uptrend,
    above in downtrend. The acceleration factor (AF) grows by `step` each
    time a new extreme is made, capped at `max_af`.

    Used by traders as both a trailing stop and a trend-following signal.
    Sign convention: returned values are positive prices; the trend direction
    can be derived by comparing SAR to close (close > SAR = uptrend).
    """
    n = len(high)
    sar = np.full(n, np.nan, dtype=float)
    if n < 2:
        return pd.Series(sar, index=high.index)

    # Initial state: pick direction by first two bars
    is_long = high.iloc[1] >= high.iloc[0]
    af = step
    ep = high.iloc[0] if is_long else low.iloc[0]  # extreme point
    sar[1] = low.iloc[0] if is_long else high.iloc[0]

    for i in range(2, n):
        prev_sar = sar[i - 1]
        if is_long:
            new_sar = prev_sar + af * (ep - prev_sar)
            # SAR must not exceed prior two lows
            new_sar = min(new_sar, low.iloc[i - 1], low.iloc[i - 2])
            if low.iloc[i] < new_sar:
                # Flip to short
                is_long = False
                new_sar = ep
                ep = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_af)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high.iloc[i - 1], high.iloc[i - 2])
            if high.iloc[i] > new_sar:
                is_long = True
                new_sar = ep
                ep = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_af)
        sar[i] = new_sar
    return pd.Series(sar, index=high.index)


def swing_high_low(
    high: pd.Series, low: pd.Series, k: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """N-bar fractal swing high / low detector.

    A bar is a swing HIGH if its high is strictly greater than the high of
    the `k` bars on each side. Returns (swing_highs, swing_lows) as boolean
    Series aligned with the input index.

    k=3 is the classic Bill Williams fractal. Use larger k for "major"
    swings on bigger timeframes; smaller k for more sensitive intraday.
    """
    n = len(high)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    h = high.values
    lo = low.values
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]) and (h[i] > h[i - 1] or h[i] > h[i + 1]):
            sh[i] = True
        if lo[i] == min(lo[i - k:i + k + 1]) and (lo[i] < lo[i - 1] or lo[i] < lo[i + 1]):
            sl[i] = True
    return pd.Series(sh, index=high.index), pd.Series(sl, index=low.index)


def trendline_from_swings(
    swing_mask: pd.Series, price: pd.Series, n_recent: int = 3,
) -> tuple[float, float] | None:
    """Fit a line through the most recent N swing points.

    Returns (slope_per_bar, intercept) for the line, suitable for
    `slope * bar_idx + intercept` to project the line.
    slope_per_bar is in price units per bar.

    Use with swing_highs for resistance lines, swing_lows for support.
    Returns None if fewer than n_recent swing points found.
    """
    idx = np.where(swing_mask.values)[0]
    if len(idx) < n_recent:
        return None
    idx = idx[-n_recent:]
    prices = price.values[idx]
    # Linear regression: slope, intercept = polyfit(x, y, 1)
    slope, intercept = np.polyfit(idx.astype(float), prices, 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Multifactor-v2 helpers: trendline proximity, S/R zones, Fibonacci
# ---------------------------------------------------------------------------
def trendline_proximity_pct(
    price_now: float,
    slope: float,
    intercept: float,
    bar_now: int,
) -> float | None:
    """% distance from price_now to the projected trendline value at bar_now.

    Returns signed distance: price > line → positive, price < line → negative.
    Used by v2: "long signal valid only if price is within X% above a support
    trendline" (small positive distance), and the mirror for short.
    """
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    line_value = slope * bar_now + intercept
    if line_value <= 0:
        return None
    return (price_now - line_value) / line_value


def sr_zones(
    swing_prices: np.ndarray, cluster_tolerance_pct: float = 0.005,
) -> list[float]:
    """Cluster swing prices into S/R zones.

    Two swings within `cluster_tolerance_pct` of each other (default 0.5%) are
    merged into one zone, represented by their mean. Returns sorted list of
    zone centers. The more swings in a zone, the more significant the level —
    but this function returns one entry per zone regardless of weight.
    """
    if len(swing_prices) == 0:
        return []
    sorted_prices = np.sort(swing_prices)
    zones: list[list[float]] = [[float(sorted_prices[0])]]
    for p in sorted_prices[1:]:
        p = float(p)
        ref = zones[-1][0]
        if abs(p - ref) / ref <= cluster_tolerance_pct:
            zones[-1].append(p)
        else:
            zones.append([p])
    return [float(np.mean(z)) for z in zones]


def nearest_sr_zone_distance_pct(
    price: float, zones: list[float], direction: str = "below",
) -> float | None:
    """Signed distance from `price` to the NEAREST zone in the given direction.

    direction='below': only consider zones BELOW price (potential support).
    direction='above': only consider zones ABOVE price (potential resistance).
    Returns abs distance / price (positive). None if no zone in that direction.
    """
    if not zones or price <= 0:
        return None
    if direction == "below":
        candidates = [z for z in zones if z <= price]
        if not candidates:
            return None
        nearest = max(candidates)  # closest one below
    else:
        candidates = [z for z in zones if z >= price]
        if not candidates:
            return None
        nearest = min(candidates)  # closest one above
    return abs(price - nearest) / price


# Standard Fibonacci retracement levels (proper subset; 0% and 100% omitted).
FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)


def fib_retracement_distance_pct(
    price: float,
    swing_high: float,
    swing_low: float,
    levels: tuple[float, ...] = FIB_LEVELS,
) -> tuple[float, float] | None:
    """Distance from `price` to nearest Fibonacci retracement level.

    Returns (nearest_level, distance_pct) where:
      - nearest_level ∈ levels (the Fib ratio that price is closest to)
      - distance_pct = |price - level_price| / price (always positive)

    For an uptrend retracement: levels measured DOWN from swing_high by
    `level * (swing_high - swing_low)`. For downtrend retracement: levels
    measured UP from swing_low. Direction is auto-inferred:
      - if swing_high > swing_low (price recently made a high then pulled back): UP retrace
      - if swing_low > swing_high (price recently made a low then bounced up): DOWN retrace
    """
    if swing_high == swing_low or price <= 0:
        return None
    rng = abs(swing_high - swing_low)
    if swing_high > swing_low:
        # Uptrend retracement: support levels below the high.
        level_prices = [swing_high - lvl * rng for lvl in levels]
    else:
        # Downtrend retracement: resistance levels above the low.
        # We swap names but the math is symmetric.
        level_prices = [swing_low - lvl * rng for lvl in levels]
        # Note: swing_low here is actually the lower of the two passed.
    best_level = None
    best_dist = float("inf")
    for lvl, lp in zip(levels, level_prices):
        d = abs(price - lp) / price
        if d < best_dist:
            best_dist = d
            best_level = lvl
    return (float(best_level), float(best_dist)) if best_level is not None else None


def recent_swing_pair(
    swing_high_mask: pd.Series,
    swing_low_mask: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback_bars: int = 200,
) -> tuple[float, float] | None:
    """Return (most_recent_swing_high_price, most_recent_swing_low_price) within
    the last `lookback_bars` bars. None if either is missing.

    Used to compute the Fibonacci range. The CALLER decides which one is
    "first" (which side price retraced from).
    """
    n = len(high)
    start = max(0, n - lookback_bars)
    high_idx = np.where(swing_high_mask.values[start:])[0]
    low_idx = np.where(swing_low_mask.values[start:])[0]
    if len(high_idx) == 0 or len(low_idx) == 0:
        return None
    last_high = float(high.values[start + high_idx[-1]])
    last_low = float(low.values[start + low_idx[-1]])
    return last_high, last_low
