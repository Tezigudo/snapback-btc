"""
ADXDualRegimeV1 — regime-switched RSI MR + Donchian-20 breakout on 15m BTC.

Research basis (ADX_DUAL_REGIME_PLAN.md, authored 2026-06-02):
  - ADX(14) partitions every bar into one of two mutually exclusive regimes:
      ADX ≤ adx_chop_threshold → range/chop leg: RSI(2) mean-reversion (Larry Connors)
      ADX > adx_chop_threshold → trend leg: Donchian-20 channel breakout (Turtle)
  - Trend-EMA(200) filter on the *range* leg only: avoids fading strong directional
    moves with RSI(2) (lesson from FUTURE_DIRECTIONS divergence-v1 bug #3).
  - Trend leg is pure breakout: adding a second gate risks replicating
    divergence-v1's over-AND-gating failure.

Entry rules — LONG:
  1. ADX(14) is finite at bar i.
  2a. If ADX[i] ≤ adx_chop_threshold (range regime):
      RSI(2)[i] < range_rsi_long_threshold  AND  close[i] > EMA(200)[i]
  2b. If ADX[i] > adx_chop_threshold (trend regime):
      high[i] > donchian_upper.shift(1)[i]  (breakout above prior 20-bar channel)
  3. Exactly one of (2a), (2b) can fire per bar — ADX gate is mutually exclusive.
  4. ATR(14) is finite and > 0.

SHORT = mirror:
  2a. RSI(2)[i] > range_rsi_short_threshold  AND  close[i] < EMA(200)[i]
  2b. low[i] < donchian_lower.shift(1)[i]

Exit rules:
  - SL: entry ± sl_atr_multiple × ATR(14)  (initial stop, not trailing)
  - TP: entry ± tp_atr_multiple × ATR(14)
  - Time stop: position closed after max_hold_bars regardless

Sizing: identical to DivergenceV1._position_units (risk-based, leverage-capped).

Authority: ADX_DUAL_REGIME_PLAN.md.
Phase: smoke / research only — not wired to bot.py, not in live deploy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import adx, atr, donchian_channel, ema, rsi


class ADXDualRegimeV1(Strategy):
    # --- ADX regime gate ---
    adx_period: int = 14
    adx_chop_threshold: float = 25.0   # ADX ≤ → range leg, > → trend leg

    # --- Range leg: RSI(2) snap MR (Larry Connors style) ---
    range_rsi_period: int = 2
    range_rsi_long_threshold: float = 10.0
    range_rsi_short_threshold: float = 90.0

    # --- Trend leg: Donchian-20 channel breakout ---
    donchian_period: int = 20

    # --- Trend-EMA (range leg filter + precomputed for both) ---
    trend_ema_period: int = 200

    # --- Exits ---
    atr_period: int = 14
    sl_atr_multiple: float = 1.5
    tp_atr_multiple: float = 3.0       # 2:1 R:R (lower than divergence's 3:1 —
                                       # Donchian breakouts have higher hit rate than
                                       # divergence reversals, don't need the same R:R cushion)
    max_hold_bars: int = 96            # 24h at 15m

    # --- Sizing: STRICTER defaults than divergence-v1 (FUTURE_DIRECTIONS lesson) ---
    risk_per_trade_pct: float = 1.0
    leverage: int = 5                  # NOT 20 — see FUTURE_DIRECTIONS bug #3
    allow_shorts: bool = True

    # --- Donchian retest entry (break + retest + rejection) ---
    # When True (default), the trend leg waits for price to pull back and confirm
    # the break level before firing.  Set False for raw-breakout back-compat.
    use_donchian_retest: bool = True
    retest_window_bars: int = 10          # max bars after break to wait for retest
    retest_proximity_pct: float = 0.005   # 0.5% above/below break level counts as retest
    retest_invalidation_pct: float = 0.005  # 0.5% through break level → setup dead

    # ------------------------------------------------------------------ init

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        high  = pd.Series(self.data.High)
        low   = pd.Series(self.data.Low)
        open_ = pd.Series(self.data.Open)

        self._close_arr = close.values
        self._high_arr  = high.values
        self._low_arr   = low.values
        self._open_arr  = open_.values   # needed for rejection-candle check

        # ADX — double Wilder pass, warmup ~2×period bars
        self._adx = adx(high, low, close, period=self.adx_period).values

        # RSI(2) for the range/chop leg
        self._rsi2 = rsi(close, period=self.range_rsi_period).values

        # Donchian channel — shift(1) so breakout is judged against prior-period channel
        dc_upper, dc_lower = donchian_channel(high, low, period=self.donchian_period)
        self._donchian_upper_prior = dc_upper.shift(1).values
        self._donchian_lower_prior = dc_lower.shift(1).values

        # EMA(200) — trend filter for range leg
        self._ema200 = ema(close, period=self.trend_ema_period).values

        # ATR(14) — sizing and SL/TP distances
        self._atr = atr(high, low, close, period=self.atr_period).values

        self._entry_bar: int | None = None

        # Pending Donchian retest setups — each is a dict or None:
        #   {"break_level": float, "age_bars": int}
        # age_bars is incremented once per bar AFTER the break bar (so the break bar
        # itself is age 0 and is not checked for retest on that bar).
        self._pending_long_break: dict | None = None
        self._pending_short_break: dict | None = None

    # ------------------------------------------------------------------ sizing

    def _position_units(self, price: float, sl_distance: float) -> int:
        # NOTE: backtesting.py 0.6.5 only accepts integer units.
        # Fractional 0.001-BTC sizing is implemented via HARNESS-level price
        # scaling (see tools/_fractional_run.py). Under scaling: 1 returned
        # "unit" == 0.001 BTC, matching Binance USDT-M perp qty_step.
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    # ------------------------------------------------------------------ signals

    def _check_raw_donchian_break_long(self, i: int) -> bool:
        """Return True if this bar's high breaks above prior Donchian upper."""
        upper_prior = self._donchian_upper_prior[i]
        return np.isfinite(upper_prior) and self._high_arr[i] > upper_prior

    def _check_raw_donchian_break_short(self, i: int) -> bool:
        """Return True if this bar's low breaks below prior Donchian lower."""
        lower_prior = self._donchian_lower_prior[i]
        return np.isfinite(lower_prior) and self._low_arr[i] < lower_prior

    def _advance_retest_state(self, i: int) -> tuple[bool, bool]:
        """Advance the retest state machine for bar i.

        Called once per bar in next() before checking signals, regardless of
        whether a position is open (pending state must decay while in a position
        so setups don't linger forever — in practice, positions block new entries
        anyway, so this is about consistent state).

        Returns (long_fire, short_fire): True if the corresponding retest setup
        resolves as a valid rejection signal on this bar.
        """
        long_fire = False
        short_fire = False

        close_v = self._close_arr[i]
        open_v  = self._open_arr[i]
        low_v   = self._low_arr[i]
        high_v  = self._high_arr[i]

        prox  = self.retest_proximity_pct
        inval = self.retest_invalidation_pct
        win   = self.retest_window_bars

        # --- Long pending ---
        if self._pending_long_break is not None:
            p = self._pending_long_break
            bl = p["break_level"]
            age = p["age_bars"]

            expired      = age > win
            # Invalidated: low dropped more than inval_pct below break level
            invalidated  = low_v < bl * (1.0 - inval)

            if expired or invalidated:
                self._pending_long_break = None
            else:
                # Retest zone: low touched within proximity_pct ABOVE the break level
                in_zone = low_v <= bl * (1.0 + prox)
                # Rejection candle: bullish close >= break level
                rejection = (close_v > open_v) and (close_v >= bl)
                if in_zone and rejection:
                    long_fire = True
                    self._pending_long_break = None
                else:
                    p["age_bars"] += 1

        # --- Short pending ---
        if self._pending_short_break is not None:
            p = self._pending_short_break
            bl = p["break_level"]
            age = p["age_bars"]

            expired     = age > win
            # Invalidated: high pushed more than inval_pct above break level
            invalidated = high_v > bl * (1.0 + inval)

            if expired or invalidated:
                self._pending_short_break = None
            else:
                # Retest zone: high touched within proximity_pct BELOW the break level
                in_zone   = high_v >= bl * (1.0 - prox)
                # Rejection candle: bearish close <= break level
                rejection = (close_v < open_v) and (close_v <= bl)
                if in_zone and rejection:
                    short_fire = True
                    self._pending_short_break = None
                else:
                    p["age_bars"] += 1

        return long_fire, short_fire

    def _long_signal(self, i: int) -> bool:
        adx_v = self._adx[i]
        if not np.isfinite(adx_v):
            return False

        close_v = self._close_arr[i]

        if adx_v <= self.adx_chop_threshold:
            # Range regime: RSI(2) snap MR with trend-EMA filter
            rsi2_v = self._rsi2[i]
            ema200_v = self._ema200[i]
            return (
                np.isfinite(rsi2_v)
                and np.isfinite(ema200_v)
                and rsi2_v < self.range_rsi_long_threshold
                and close_v > ema200_v
            )
        else:
            # Trend regime: Donchian-20 breakout
            if not self.use_donchian_retest:
                # Raw breakout — back-compat path
                return self._check_raw_donchian_break_long(i)
            # Retest path: register break if detected; actual fire comes from
            # _advance_retest_state(), which is called before _long_signal().
            # Register a new break (overrides any existing pending setup).
            if self._check_raw_donchian_break_long(i):
                upper_prior = self._donchian_upper_prior[i]
                self._pending_long_break = {"break_level": upper_prior, "age_bars": 0}
            # The retest-fire is handled separately via _retest_long_fire flag
            # set on self before calling _long_signal (see next()).
            return self._retest_long_fire

    def _short_signal(self, i: int) -> bool:
        if not self.allow_shorts:
            return False

        adx_v = self._adx[i]
        if not np.isfinite(adx_v):
            return False

        close_v = self._close_arr[i]

        if adx_v <= self.adx_chop_threshold:
            # Range regime: RSI(2) overbought with trend-EMA filter (short below EMA)
            rsi2_v = self._rsi2[i]
            ema200_v = self._ema200[i]
            return (
                np.isfinite(rsi2_v)
                and np.isfinite(ema200_v)
                and rsi2_v > self.range_rsi_short_threshold
                and close_v < ema200_v
            )
        else:
            # Trend regime: Donchian-20 breakdown below prior channel
            if not self.use_donchian_retest:
                # Raw breakout — back-compat path
                return self._check_raw_donchian_break_short(i)
            # Retest path: register break if detected.
            if self._check_raw_donchian_break_short(i):
                lower_prior = self._donchian_lower_prior[i]
                self._pending_short_break = {"break_level": lower_prior, "age_bars": 0}
            return self._retest_short_fire

    # ------------------------------------------------------------------ loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Advance retest state machine unconditionally each bar (pending setups
        # must age/expire even while in a position, to avoid stale state).
        if self.use_donchian_retest:
            self._retest_long_fire, self._retest_short_fire = (
                self._advance_retest_state(i)
            )
        else:
            self._retest_long_fire = False
            self._retest_short_fire = False

        # Position management: time stop
        if self.position:
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold_bars:
                self.position.close()
                self._entry_bar = None
            return

        # ATR guard
        atr_v = self._atr[i]
        if not np.isfinite(atr_v) or atr_v <= 0:
            return

        sl_dist = self.sl_atr_multiple * atr_v
        tp_dist = self.tp_atr_multiple * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if self._long_signal(i):
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
            self._entry_bar = i
        elif self._short_signal(i):
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
            self._entry_bar = i
