"""
IntradayTSMOM_BTC — clock-anchored intraday time-series momentum on BTC perp.

TODO_LEG candidate per
  /Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_tsmom_intraday.md

Hypothesis (Shen, Urquhart, Wang 2022, Financial Review):
  The sign of the first-half-of-day BTC return predicts the sign of the
  second-half-of-day return. We trade a half-DAY variant: at 12:00 UTC, if
  the morning return is large enough (|r_morn| > coef × 30-day realised
  daily-return std), open a position in the SAME DIRECTION as morning. Exit
  at 24:00 UTC (forced close), stop, or ATR target.

Bar convention (15m feed, bar-open timestamps, tz-naive UTC):
  - "Morning return" r_morn = Close(bar at 11:45) / Open(bar at 00:00) - 1
      Close of 11:45 bar IS the 12:00 UTC mark.
      Open of 00:00 bar IS the 00:00 UTC mark.
  - Signal evaluated and order placed on bar with index = (HH=11, MM=45).
      With backtesting.py trade_on_close=False the order fills at next bar OPEN,
      i.e. the 12:00 bar opens — which equals 12:00 UTC. Lookahead-safe: we use
      only data that finished at exactly 12:00 UTC.
  - Time exit fires on the 23:45 bar (closes at 24:00 UTC) → order closes at
      next bar OPEN = 00:00 of next day.

Volatility filter:
  realised_daily_std = rolling std (30) of daily close-to-close returns,
    computed on a DAILY series resampled from the 15m feed and SHIFTED by 1
    (so day d uses days [d-30 .. d-1], lookahead-safe). Held constant for the
    whole UTC day.

ATR(14) on 1H, ALIGNED into the 15m index via backward merge_asof on bar-CLOSE
  timestamps — identical convention to multifactor-v1's 4H EMA200 alignment
  in strategy/signals_multifactor.py. ATR of the still-open 1H bar is unreachable.

Filters (skip the day entirely if):
  - |r_morn| > 5 × realised_daily_std  (extreme outlier — mean-reversion risk)
  - 4H ATR is in the bottom 10% of trailing 90 days (no fuel for afternoon)

Position sizing: vol-targeted at risk_per_trade_pct of equity, sized so that
  a 1.0× ATR(1H) move equals that risk in dollars. Same integer-unit pattern
  as DayTradeMultiFactorBTC; 1 unit == 0.001 BTC after the harness PRICE_SCALE=0.001.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr

_REPO_ROOT = Path(__file__).resolve().parent.parent
_1H_PARQUET_DEFAULT = _REPO_ROOT / "data" / "historical" / "BTC_USDT_USDT_1h.parquet"
_4H_PARQUET_DEFAULT = _REPO_ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"


def _build_atr_aligned(
    dates_15m: pd.DatetimeIndex,
    parquet_path: Path,
    period: int,
    bar_hours: int,
) -> np.ndarray:
    """Compute ATR(period) on a higher-TF parquet, align lookahead-safe to 15m.

    Mirrors strategy.signals_multifactor._build_4h_ema_aligned.
        - Parquet index is bar-OPEN; a bar opening at T closes at T + bar_hours.
        - ATR of bar T may be used at 15m timestamps >= T + bar_hours.
        - backward merge_asof on close timestamps.
    """
    df = pd.read_parquet(parquet_path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    high_col = "high" if "high" in df.columns else "High"
    low_col = "low" if "low" in df.columns else "Low"
    close_col = "close" if "close" in df.columns else "Close"
    atr_series = atr(df[high_col], df[low_col], df[close_col], period).values

    close_times = df.index + pd.Timedelta(hours=bar_hours)
    close_times_us = pd.DatetimeIndex(close_times.astype("datetime64[us]"))

    left_idx_raw = dates_15m
    if left_idx_raw.tz is not None:
        left_idx_raw = left_idx_raw.tz_localize(None)
    left_idx = pd.DatetimeIndex(left_idx_raw.astype("datetime64[us]"))

    right = pd.DataFrame({"atr": atr_series}, index=close_times_us).sort_index()
    left = pd.DataFrame(index=left_idx)
    merged = pd.merge_asof(left, right, left_index=True, right_index=True, direction="backward")
    return merged["atr"].values


def _build_morning_anchor_arrays(dates_15m: pd.DatetimeIndex, opens_15m: np.ndarray) -> np.ndarray:
    """For each 15m bar, return the OPEN price of THIS UTC day's 00:00 bar.

    Lookahead-safe because every bar within day d already saw the 00:00 bar
    of day d (it occurred earlier). NaN for any bars before the first 00:00
    anchor appears.
    """
    n = len(dates_15m)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    # Identify each bar's calendar day, find each day's first bar with HH=0, MM=0,
    # take its Open as the anchor for that day.
    hours = dates_15m.hour
    minutes = dates_15m.minute
    is_midnight = (hours == 0) & (minutes == 0)
    day_floor = dates_15m.floor("D")
    # Build per-day anchor = Open at the midnight bar of that day.
    midnight_positions = np.where(is_midnight)[0]
    if len(midnight_positions) == 0:
        return out
    anchor_by_day: dict = {}
    for pos in midnight_positions:
        day = day_floor[pos]
        # Only first occurrence per day; subsequent skipped.
        if day not in anchor_by_day:
            anchor_by_day[day] = float(opens_15m[pos])
    # Map each bar to its day's anchor (only valid for bars on/after the day's midnight bar).
    for i in range(n):
        a = anchor_by_day.get(day_floor[i])
        if a is not None:
            out[i] = a
    return out


def _build_daily_std_anchor(
    dates_15m: pd.DatetimeIndex,
    closes_15m: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Per-bar value of the trailing-30-day realised daily-return std.

    Daily series: close-to-close returns of the LAST bar of each UTC day
    (bar with HH=23, MM=45 → closes at 24:00 UTC). Rolling std with window=30,
    THEN SHIFT(1) so day d sees the std using days [d-30 .. d-1] only.

    Returned ndarray is aligned 1:1 with dates_15m: each 15m bar receives the
    std valid for its calendar day. Bars before the first valid std are NaN.
    """
    n = len(dates_15m)
    out = np.full(n, np.nan, dtype=float)
    hours = dates_15m.hour
    minutes = dates_15m.minute
    is_last_bar = (hours == 23) & (minutes == 45)
    day_floor = dates_15m.floor("D")
    # Daily close series indexed by day.
    last_positions = np.where(is_last_bar)[0]
    if len(last_positions) < 2:
        return out
    day_to_close: dict = {}
    for pos in last_positions:
        d = day_floor[pos]
        if d not in day_to_close:
            day_to_close[d] = float(closes_15m[pos])
    days_sorted = sorted(day_to_close.keys())
    closes_daily = pd.Series([day_to_close[d] for d in days_sorted], index=pd.DatetimeIndex(days_sorted))
    daily_ret = closes_daily.pct_change()
    daily_std = daily_ret.rolling(window=window, min_periods=window).std()
    daily_std_shifted = daily_std.shift(1)  # day d uses [d-window .. d-1]
    # Map each 15m bar's day to that shifted std.
    std_by_day = daily_std_shifted.to_dict()
    for i in range(n):
        v = std_by_day.get(day_floor[i])
        if v is not None and np.isfinite(v):
            out[i] = float(v)
    return out


def _build_atr4h_pctile_flag(
    dates_15m: pd.DatetimeIndex,
    atr4h_aligned: np.ndarray,
    lookback_days: int = 90,
    bottom_pct: float = 0.10,
) -> np.ndarray:
    """Bool per 15m bar: True if current 4H ATR is in the bottom `bottom_pct`
    of the trailing `lookback_days`. NaN-safe.

    Computed on a per-DAY scale: for each day, take a representative ATR4H
    (the value at end-of-day's 23:45 bar — already lookahead-safe via prior
    alignment), build a rolling pctile over `lookback_days` days, SHIFT(1),
    broadcast back to the 15m index for that day.
    """
    n = len(dates_15m)
    flag = np.zeros(n, dtype=bool)
    hours = dates_15m.hour
    minutes = dates_15m.minute
    is_last_bar = (hours == 23) & (minutes == 45)
    day_floor = dates_15m.floor("D")
    last_positions = np.where(is_last_bar)[0]
    if len(last_positions) < lookback_days:
        return flag
    day_to_atr: dict = {}
    for pos in last_positions:
        d = day_floor[pos]
        v = atr4h_aligned[pos]
        if d not in day_to_atr and np.isfinite(v):
            day_to_atr[d] = float(v)
    days_sorted = sorted(day_to_atr.keys())
    atr_daily = pd.Series([day_to_atr[d] for d in days_sorted], index=pd.DatetimeIndex(days_sorted))
    # rolling quantile at `bottom_pct`. SHIFT(1) to remove lookahead.
    threshold_daily = atr_daily.rolling(window=lookback_days, min_periods=lookback_days).quantile(bottom_pct).shift(1)
    # is current ATR (today's value, shifted by 1 too) <= the threshold?
    atr_today_shifted = atr_daily.shift(1)
    skip_daily = atr_today_shifted <= threshold_daily
    skip_by_day = skip_daily.fillna(False).to_dict()
    for i in range(n):
        if skip_by_day.get(day_floor[i], False):
            flag[i] = True
    return flag


class IntradayTSMOM_BTC(Strategy):
    """Clock-anchored intraday TSMOM.

    Parameters can be overridden via bt.run(**kwargs).
    """

    # -- entry / clock --
    morning_hour = 12              # 12:00 UTC end-of-morning anchor
    eod_hour = 24                  # forced exit at 24:00 UTC of same day
    threshold_std_coef = 0.3       # |r_morn| > coef × realised daily std
    outlier_std_coef = 5.0         # skip if |r_morn| > coef × realised daily std

    # -- risk / sizing --
    risk_per_trade_pct = 0.5
    leverage = 20
    atr_stop_x = 1.0               # stop = entry ± atr_stop_x × ATR(1H)
    atr_tp_x = 2.0                 # tp = entry ± atr_tp_x × ATR(1H)

    # -- regime filter (4H ATR low-vol skip) --
    use_low_vol_skip = True
    low_vol_lookback_days = 90
    low_vol_bottom_pct = 0.10

    # -- aux parquet paths (can be overridden for scaled runs) --
    atr1h_parquet_path = str(_1H_PARQUET_DEFAULT)
    atr4h_parquet_path = str(_4H_PARQUET_DEFAULT)
    atr_1h_period = 14
    atr_4h_period = 14

    # -- direction toggle (debug) --
    allow_longs = True
    allow_shorts = True

    def init(self) -> None:
        dates = pd.DatetimeIndex(self.data.index)
        opens = np.asarray(self.data.Open, dtype=float)
        closes = np.asarray(self.data.Close, dtype=float)

        self._dates = dates
        self._hours = dates.hour.values
        self._minutes = dates.minute.values
        self._day_floor = dates.floor("D")

        # Per-bar morning anchor (Open of this day's 00:00 bar). Lookahead-safe:
        # within a day all bars have already seen the 00:00 bar of that day.
        self._morn_anchor = _build_morning_anchor_arrays(dates, opens)

        # Trailing 30-day realised daily-return std (lookahead-safe via shift(1)).
        self._daily_std = _build_daily_std_anchor(dates, closes, window=30)

        # 1H ATR(14) aligned to 15m.
        self._atr_1h = _build_atr_aligned(
            dates, Path(self.atr1h_parquet_path), self.atr_1h_period, bar_hours=1
        )

        # 4H ATR(14) aligned + bottom-pctile flag.
        if self.use_low_vol_skip:
            atr_4h_aligned = _build_atr_aligned(
                dates, Path(self.atr4h_parquet_path), self.atr_4h_period, bar_hours=4
            )
            self._skip_lowvol = _build_atr4h_pctile_flag(
                dates,
                atr_4h_aligned,
                lookback_days=self.low_vol_lookback_days,
                bottom_pct=self.low_vol_bottom_pct,
            )
        else:
            self._skip_lowvol = np.zeros(len(dates), dtype=bool)

        # Entry-bar bookkeeping for time-stop exit.
        self._entry_bar: int | None = None
        self._entry_day = None  # pd.Timestamp at day floor

    def _position_units(self, price: float, sl_distance: float) -> int:
        """Volatility-targeted integer units. Mirrors multifactor-v1.

        Under harness PRICE_SCALE=0.001 each "unit" equals 0.001 BTC.
        """
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_units = risk_amount / sl_distance
        max_units = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_units, max_units)), 0)

    def next(self) -> None:
        i = len(self.data) - 1
        hh = int(self._hours[i])
        mm = int(self._minutes[i])
        close_v = float(self.data.Close[-1])

        # Forced exit at 24:00 UTC == bar with HH=23, MM=45 (closes at 24:00).
        # If we have a position opened on this day, close it now. exclusive_orders
        # in the harness queues the close at next bar open (00:00 next day).
        if self.position:
            if hh == 23 and mm == 45:
                self.position.close()
                self._entry_bar = None
                self._entry_day = None
                return
            # Safety: if for any reason we're carrying a position from a prior
            # day (e.g. day boundary missed due to data gap), close it.
            if self._entry_day is not None and self._day_floor[i] != self._entry_day:
                self.position.close()
                self._entry_bar = None
                self._entry_day = None
                return
            return

        # Only consider entry on the bar that closes at 12:00 UTC: HH=11, MM=45.
        if not (hh == 11 and mm == 45):
            return

        # Low-vol skip filter.
        if self.use_low_vol_skip and bool(self._skip_lowvol[i]):
            return

        anchor = self._morn_anchor[i]
        if not np.isfinite(anchor) or anchor <= 0:
            return
        r_morn = close_v / anchor - 1.0

        sigma = self._daily_std[i]
        if not np.isfinite(sigma) or sigma <= 0:
            return

        abs_r = abs(r_morn)
        if abs_r > self.outlier_std_coef * sigma:
            return  # extreme outlier — mean-reversion risk
        if abs_r <= self.threshold_std_coef * sigma:
            return  # not enough morning move

        atr_1h_v = self._atr_1h[i]
        if not np.isfinite(atr_1h_v) or atr_1h_v <= 0:
            return

        sl_dist = self.atr_stop_x * atr_1h_v
        tp_dist = self.atr_tp_x * atr_1h_v

        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        going_long = r_morn > 0
        if going_long and not self.allow_longs:
            return
        if (not going_long) and not self.allow_shorts:
            return

        if going_long:
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
        else:
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
        self._entry_bar = i
        self._entry_day = self._day_floor[i]
