"""
AdaptiveTrendV1 + H1 EMA50 multi-timeframe confirmation gate (research-only).

ONE feature added on top of AdaptiveTrendV1 (Algorithm 1 base):
    LONG entries require: most-recently-closed H1 bar has
        (a) EMA50 slope >= 0 over the last `h1_slope_lookback` H1 bars, AND
        (b) close > H1 EMA50.
    SHORT entries require the mirror: slope <= 0 AND close < H1 EMA50.
    Exits / trailing stop / sizing are unchanged.

Rationale
---------
AdaptiveTrendV1 fires on H6 MOM breakouts. The H6 cadence captures the
medium-term regime well but is silent about whether the H1-scale price
structure agrees. Adding a single intermediate-TF gate (between the 15m
carrier and the H6 signal) drops entries that fire against the 1-hour
trend — a textbook MTF confirmation pattern.

Specifically: requiring price ABOVE H1 EMA50 AND H1 EMA50 trending UP
filters chop and counter-trend pops. Strictly an entry-quality filter,
so it can only reduce trade count, never increase it.

Design notes
------------
- The H1 EMA50 is loaded from a SEPARATE H1 parquet (FULL history, so
  EMA(50) is warmed up before any OOS slice begins).
- Lookahead safety: the H1 parquet's index is bar OPEN time. The EMA of
  the bar opening at T is only available from T+1h onward. We therefore
  align the EMA series by CLOSE times = open + 1h, and merge_asof
  backward — each 15m bar receives the EMA of the most-recently CLOSED
  H1 bar. NO shift(1) is needed because the close-time re-index already
  enforces the "only after close" constraint. This mirrors
  strategy/signals_multifactor._build_4h_ema_aligned exactly.
- Slope = sign of (EMA[t] - EMA[t - lookback]) on the H1-CLOSE grid.
  Calculated on the H1 frame BEFORE re-indexing to 15m so the lookback
  is in H1 bars, not 15m bars.
- PRICE_SCALE: under the fractional-sizing harness OHLC on the 15m frame
  is multiplied by 0.001 (see tools/_fractional_run.py + tools/_postfrac_*).
  The H1 parquet read here must ALSO be scaled or the EMA50 will live
  on a different scale than self.data.Close. The runner pre-scales the
  H1 parquet to a temp file and passes that path via `h1_parquet_path`.

Authority: research-only. NOT wired to bot.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy.indicators import ema as wilder_ema
from strategy.signals_adaptive_trend import AdaptiveTrendV1

_REPO_ROOT = Path(__file__).resolve().parent.parent
_H1_PARQUET_DEFAULT = _REPO_ROOT / "data" / "historical" / "BTC_USDT_USDT_1h.parquet"


def _build_h1_ema_and_slope_aligned(
    dates_15m: pd.DatetimeIndex,
    parquet_path: Path,
    ema_period: int,
    slope_lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load H1 bars, compute EMA(N) + slope(lookback), align to 15m index.

    TIMING CONVENTION:
        The parquet index is bar-OPEN time. A 1H bar opening at T closes
        at T+1h. We may only use the EMA / slope of that bar at 15m
        timestamps >= T+1h.

    Implementation mirrors strategy/signals_multifactor._build_4h_ema_aligned:
        1. Load FULL H1 parquet (history → EMA(N) warmup is global).
        2. Compute EMA(N) on H1 closes.
        3. Compute slope = EMA[t] - EMA[t - slope_lookback] on the H1
           frame.
        4. Re-index both series by CLOSE timestamps (open_time + 1h).
        5. merge_asof(direction="backward") against the 15m index — each
           15m bar receives the EMA + slope of the most-recently CLOSED
           H1 bar.

    Returns (ema_arr, slope_arr), each ndarray aligned 1:1 to dates_15m.
    Warm-up rows are NaN (and the strategy treats NaN as "skip entry").
    """
    df1h = pd.read_parquet(parquet_path)
    if df1h.index.tz is not None:
        df1h.index = df1h.index.tz_localize(None)
    close_col = "close" if "close" in df1h.columns else "Close"

    ema_series = wilder_ema(df1h[close_col], ema_period)
    close_series = df1h[close_col]
    slope_series = ema_series - ema_series.shift(slope_lookback)

    # Close-time re-index = lookahead barrier.
    close_times = df1h.index + pd.Timedelta(hours=1)
    close_times_us = pd.DatetimeIndex(close_times.astype("datetime64[us]"))

    left_idx_raw = dates_15m
    if left_idx_raw.tz is not None:
        left_idx_raw = left_idx_raw.tz_localize(None)
    left_idx = pd.DatetimeIndex(left_idx_raw.astype("datetime64[us]"))

    right = pd.DataFrame(
        {
            "ema_h1":   ema_series.values,
            "slope_h1": slope_series.values,
            "close_h1": close_series.values,
        },
        index=close_times_us,
    ).sort_index()
    left = pd.DataFrame(index=left_idx)
    merged = pd.merge_asof(
        left, right,
        left_index=True, right_index=True,
        direction="backward",
    )
    return (
        merged["ema_h1"].values,
        merged["slope_h1"].values,
    )


class AdaptiveTrendV1_h1_confirmation(AdaptiveTrendV1):
    """AdaptiveTrendV1 with H1 EMA50 MTF trend-confirmation entry gate."""

    # Feature flag — default OFF for backward compat. The runner flips it on.
    use_h1_confirmation: bool = False
    h1_ema_period: int = 50
    h1_slope_lookback: int = 10   # H1 bars over which to measure EMA50 slope
    h1_parquet_path: str = str(_H1_PARQUET_DEFAULT)

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        super().init()
        if not self.use_h1_confirmation:
            self._h1_ema_arr = None
            self._h1_slope_arr = None
            return

        dates = pd.DatetimeIndex(self.data.index)
        ema_arr, slope_arr = _build_h1_ema_and_slope_aligned(
            dates,
            Path(self.h1_parquet_path),
            ema_period=self.h1_ema_period,
            slope_lookback=self.h1_slope_lookback,
        )
        self._h1_ema_arr = ema_arr
        self._h1_slope_arr = slope_arr

    # ------------------------------------------------------------------ helpers

    def _h1_confirms_long(self, i: int, close_v: float) -> bool:
        if self._h1_ema_arr is None:
            return True  # gate disabled => pass-through
        e = self._h1_ema_arr[i]
        s = self._h1_slope_arr[i]
        if not (np.isfinite(e) and np.isfinite(s)):
            return False  # warm-up / missing -> skip entry (safer than allow)
        return (s >= 0.0) and (close_v > e)

    def _h1_confirms_short(self, i: int, close_v: float) -> bool:
        if self._h1_ema_arr is None:
            return True
        e = self._h1_ema_arr[i]
        s = self._h1_slope_arr[i]
        if not (np.isfinite(e) and np.isfinite(s)):
            return False
        return (s <= 0.0) and (close_v < e)

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        # Position-management branch unchanged — delegate to base while open.
        if self.position:
            super().next()
            return

        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]
        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # Entry only at H6 close (mirror base).
        if not self._is_h6_close_bar(ts):
            return
        if self._last_h6_close_seen == ts:
            return
        self._last_h6_close_seen = ts

        if not np.isfinite(mom_v) or not np.isfinite(atr_v) or atr_v <= 0:
            return

        sl_dist = self.alpha * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if mom_v > self.theta_entry:
            if not self._h1_confirms_long(i, close_v):
                return
            self.buy(size=units, sl=close_v - sl_dist)
            self._entry_bar = i
            self._trail_level = close_v - sl_dist
        elif self.allow_shorts and mom_v < -self.theta_entry:
            if not self._h1_confirms_short(i, close_v):
                return
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._trail_level = close_v + sl_dist
