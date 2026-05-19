"""
DayTradeMultiFactorBTCv2 — v1 filters + classical-TA confirmation.

Per user request: "could it have more conditions? trendline, support and
resistance zone, fibo... do as another version of bot"

Built as PARALLEL to v1 (not a replacement). Same 4 base filters as v1, plus
3 new TA confirmations that look for price near a meaningful level:
  - near a detected trendline (support for long, resistance for short)
  - near a swing-clustered S/R zone
  - near a Fibonacci retracement level

The new confirmations are gated by `confirmations_required`:
  - 0 → behaves like v1 (TA confirmations disabled)
  - 1, 2, 3 → must satisfy at least N of {trendline, S/R, fib}

We expose TWO ready-made variants in backtest.py:
  - multifactor-v2-loose: confirmations_required=1 (more trades)
  - multifactor-v2-strict: confirmations_required=2 (fewer, higher conviction)

WARNING (read PATH2_RESULTS.html first): we previously added an MTF gate as
a 5th filter and it REDUCED returns. There is no guarantee adding TA filters
will improve outcomes. They could:
  - filter out genuine winners (correlated information loss)
  - shrink the trade count to statistical noise
  - overfit to the validation set
Always read V2_RESULTS.html for the empirical comparison BEFORE deploying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import (
    ema,
    fib_retracement_distance_pct,
    nearest_sr_zone_distance_pct,
    recent_swing_pair,
    rsi,
    sma,
    sr_zones,
    swing_high_low,
    trendline_from_swings,
    trendline_proximity_pct,
)


class DayTradeMultiFactorBTCv2(Strategy):
    # --- v1 base filters ---
    rsi_period = 14
    rsi_long_threshold = 40.0
    rsi_short_threshold = 70.0
    volume_ma_period = 20
    volume_multiple = 2.0
    mf_trend_ema_period = 200
    require_trend = True
    funding_extreme_threshold = 0.0005
    require_funding_not_extreme = True

    # --- v2 TA confirmation knobs ---
    # How many of {trendline, sr_zone, fib} must agree.
    # 0 = behave like v1. 1 = loose. 2 = balanced. 3 = strict.
    confirmations_required: int = 2
    # Swing detection
    swing_k: int = 3
    swing_lookback_bars: int = 200
    # Trendline proximity: long valid if price within X% above support line
    trendline_max_distance_pct: float = 0.015   # within 1.5% of the line
    # S/R zone proximity: long valid if support below price by < X%
    sr_max_distance_pct: float = 0.010          # within 1.0% of a zone
    sr_cluster_tolerance_pct: float = 0.005     # 0.5% clusters
    # Fibonacci proximity
    fib_max_distance_pct: float = 0.010         # within 1.0% of a fib level

    # --- exits (v1 geometry) ---
    sl_pct = 0.015
    tp_pct = 0.030
    max_hold_bars = 1344

    # --- sizing ---
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self._rsi = rsi(close, self.rsi_period).values
        self._vol_sma = sma(volume, self.volume_ma_period).values
        self._trend_ema = ema(close, self.mf_trend_ema_period).values

        # Pre-compute swing masks across the whole series. At each bar i we'll
        # slice these via [:i+1] to avoid lookahead — `swing_high_low` already
        # only marks bars where the neighbours are visible, but we still cap
        # the swings considered to those that are k bars in the past.
        self._sh_mask, self._sl_mask = swing_high_low(high, low, self.swing_k)
        # Save raw highs/lows for later swing-price lookup
        self._high = high
        self._low = low
        self._entry_bar: int | None = None

    def _position_units(self, price: float, sl_distance: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    # -----------------------------------------------------------------------
    # v1 base filters
    # -----------------------------------------------------------------------
    def _base_long_ok(self, i: int) -> bool:
        if not (np.isfinite(self._rsi[i]) and self._rsi[i] < self.rsi_long_threshold):
            return False
        vol_sma_v = self._vol_sma[i]
        if not (np.isfinite(vol_sma_v) and self.data.Volume[-1] > self.volume_multiple * vol_sma_v):
            return False
        if self.require_trend:
            t = self._trend_ema[i]
            if not (np.isfinite(t) and self.data.Close[-1] > t):
                return False
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f > self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        return True

    def _base_short_ok(self, i: int) -> bool:
        if not self.allow_shorts:
            return False
        if not (np.isfinite(self._rsi[i]) and self._rsi[i] > self.rsi_short_threshold):
            return False
        vol_sma_v = self._vol_sma[i]
        if not (np.isfinite(vol_sma_v) and self.data.Volume[-1] > self.volume_multiple * vol_sma_v):
            return False
        if self.require_trend:
            t = self._trend_ema[i]
            if not (np.isfinite(t) and self.data.Close[-1] < t):
                return False
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f < -self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        return True

    # -----------------------------------------------------------------------
    # v2 TA confirmations (return True/False; ALL use only data up to bar i)
    # -----------------------------------------------------------------------
    def _trendline_confirm(self, i: int, side: str) -> bool:
        """For LONG: price near (above) a recent support trendline.
        For SHORT: price near (below) a recent resistance trendline."""
        if i < self.swing_lookback_bars:
            return False
        start = i - self.swing_lookback_bars
        sub_high = self._high.iloc[start:i + 1]
        sub_low = self._low.iloc[start:i + 1]
        sh = self._sh_mask.iloc[start:i + 1]
        sl = self._sl_mask.iloc[start:i + 1]
        price = float(self.data.Close[-1])
        bar_idx_local = len(sub_high) - 1
        if side == "long":
            line = trendline_from_swings(sl, sub_low, n_recent=3)
            if line is None:
                return False
            d = trendline_proximity_pct(price, *line, bar_idx_local)
            # Long valid if price is just above support (small positive distance)
            return d is not None and 0 <= d <= self.trendline_max_distance_pct
        else:
            line = trendline_from_swings(sh, sub_high, n_recent=3)
            if line is None:
                return False
            d = trendline_proximity_pct(price, *line, bar_idx_local)
            # Short valid if price is just below resistance (small negative distance)
            return d is not None and -self.trendline_max_distance_pct <= d <= 0

    def _sr_zone_confirm(self, i: int, side: str) -> bool:
        if i < self.swing_lookback_bars:
            return False
        start = i - self.swing_lookback_bars
        sh = self._sh_mask.iloc[start:i + 1]
        sl = self._sl_mask.iloc[start:i + 1]
        sub_high = self._high.iloc[start:i + 1]
        sub_low = self._low.iloc[start:i + 1]
        price = float(self.data.Close[-1])
        if side == "long":
            sl_prices = sub_low.values[np.where(sl.values)[0]]
            zones = sr_zones(sl_prices, self.sr_cluster_tolerance_pct)
            d = nearest_sr_zone_distance_pct(price, zones, "below")
            return d is not None and d <= self.sr_max_distance_pct
        else:
            sh_prices = sub_high.values[np.where(sh.values)[0]]
            zones = sr_zones(sh_prices, self.sr_cluster_tolerance_pct)
            d = nearest_sr_zone_distance_pct(price, zones, "above")
            return d is not None and d <= self.sr_max_distance_pct

    def _fib_confirm(self, i: int, side: str) -> bool:
        if i < self.swing_lookback_bars:
            return False
        start = i - self.swing_lookback_bars
        sh = self._sh_mask.iloc[start:i + 1]
        sl = self._sl_mask.iloc[start:i + 1]
        sub_high = self._high.iloc[start:i + 1]
        sub_low = self._low.iloc[start:i + 1]
        price = float(self.data.Close[-1])
        pair = recent_swing_pair(sh, sl, sub_high, sub_low, self.swing_lookback_bars)
        if pair is None:
            return False
        sh_p, sl_p = pair
        # For LONG (uptrend retracement): pass high=swing_high, low=swing_low
        # For SHORT (downtrend retracement): pass high=swing_low, low=swing_high
        if side == "long":
            fib = fib_retracement_distance_pct(price, sh_p, sl_p)
        else:
            fib = fib_retracement_distance_pct(price, sl_p, sh_p)
        return fib is not None and fib[1] <= self.fib_max_distance_pct

    def _ta_confirmations_count(self, i: int, side: str) -> int:
        c = 0
        if self._trendline_confirm(i, side):
            c += 1
        if self._sr_zone_confirm(i, side):
            c += 1
        if self._fib_confirm(i, side):
            c += 1
        return c

    # -----------------------------------------------------------------------
    # Main next()
    # -----------------------------------------------------------------------
    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Position management — same as v1
        if self.position:
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold_bars:
                self.position.close()
                self._entry_bar = None
                return
            if self.require_trend:
                t = self._trend_ema[i]
                if np.isfinite(t):
                    if self.position.is_long and close_v < t:
                        self.position.close()
                        self._entry_bar = None
                        return
                    if self.position.is_short and close_v > t:
                        self.position.close()
                        self._entry_bar = None
                        return
            return

        sl_dist = self.sl_pct * close_v
        tp_dist = self.tp_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        # Entry: base filter + TA confirmations
        if self._base_long_ok(i):
            n_conf = self._ta_confirmations_count(i, "long")
            if n_conf >= self.confirmations_required:
                self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
                self._entry_bar = i
        elif self._base_short_ok(i):
            n_conf = self._ta_confirmations_count(i, "short")
            if n_conf >= self.confirmations_required:
                self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
                self._entry_bar = i


class DayTradeMultiFactorBTCv2Loose(DayTradeMultiFactorBTCv2):
    """Require at least 1 of {trendline, S/R, fib} to confirm."""
    confirmations_required: int = 1


class DayTradeMultiFactorBTCv2Strict(DayTradeMultiFactorBTCv2):
    """Require ALL 3 of {trendline, S/R, fib} to confirm.

    Best-performing variant in OOS testing (5 of 6 windows positive,
    +82.66% compounded across 2022 H1 – 2026 Q1). See V2_RESULTS.html.
    """
    confirmations_required: int = 3
