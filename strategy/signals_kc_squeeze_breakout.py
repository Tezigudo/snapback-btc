"""KCSqueezeBreakoutBTC — volatility-contraction → expansion strategy on 15m BTC perp.

Hypothesis: when Bollinger Bands close *inside* Keltner Channels for ≥10
consecutive bars, realized volatility has been suppressed and the next
directional move tends to be outsized.

Entry logic (15m close-of-bar evaluation):
  1. Compute BB(20, 2σ) and KC(20, 1.5×ATR14) on close / high / low.
  2. "In squeeze" iff BB upper < KC upper AND BB lower > KC lower.
  3. ARM the breakout after ≥ ``squeeze_min_bars`` consecutive in-squeeze bars.
     Re-arming requires the squeeze state to first DROP and then re-form
     for ≥ squeeze_min_bars again (one-shot per squeeze episode).
  4. Direction trigger:
        long  := armed AND close > rolling_max(High, donchian_period).shift(1)
        short := armed AND close < rolling_min(Low,  donchian_period).shift(1)
  5. Volume confirmation: Volume[t] > volume_multiple * SMA(Volume, 20)[t].
  6. Stop = entry ± stop_atr_mult × ATR(14); Target = entry ∓ target_atr_mult × ATR(14).
  7. Sizing: risk_per_trade_pct (default 2.0%) at leverage (default 20×).

Lookahead handling:
  - ATR, BB, KC, ATR-based stops all use indicators computed on the CURRENT bar
    using current bar's OHLC. These are observable at bar close.
  - The 20-bar high / 20-bar low Donchian channel is SHIFTED by 1 so the
    breakout test compares the current close to the PRIOR 20-bar high/low
    (cannot include the current bar's own high/low).
  - Entry fires at bar i; backtesting.py with trade_on_close=False fills at
    bar i+1's open. This matches the convention used by multifactor-v1.

Why NOT use linear-regression slope as direction filter (departing from spec):
  The user's instructions explicitly specify "close > 20-bar high (long) or
  close < 20-bar low (short)" as the direction trigger, which is what's
  implemented here. This is structurally a vol-contraction → Donchian-confirmed
  break.

Backward-compat: this is a NEW class (not a subclass). `enabled = True` is fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import (
    atr,
    bollinger_bands,
    keltner_channel,
    sma,
)


class KCSqueezeBreakoutBTC(Strategy):
    # --- BB / KC parameters ---
    bb_period = 20
    bb_n_std = 2.0
    kc_ema_period = 20
    kc_atr_period = 20
    kc_mult = 1.5

    # --- squeeze gate ---
    squeeze_min_bars = 10           # require ≥ N consecutive in-squeeze bars to ARM

    # --- breakout / direction ---
    donchian_period = 20            # 20-bar high/low for breakout direction

    # --- volume confirm ---
    volume_ma_period = 20
    volume_multiple = 1.5

    # --- ATR stops/targets ---
    atr_period = 14
    stop_atr_mult = 1.5
    target_atr_mult = 2.5

    # --- sizing ---
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True

    # --- safety cap ---
    max_hold_bars = 1344            # 14 days × 96 bars/day at 15m (same as mf-v1)

    # --- enable flag (opt-in, but default True since this is new class) ---
    enabled = True

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands
        bb_up, bb_mid, bb_lo = bollinger_bands(close, self.bb_period, self.bb_n_std)
        self._bb_up = bb_up.values
        self._bb_lo = bb_lo.values

        # Keltner Channels
        kc_up, kc_mid, kc_lo = keltner_channel(
            high, low, close, self.kc_ema_period, self.kc_atr_period, self.kc_mult
        )
        self._kc_up = kc_up.values
        self._kc_lo = kc_lo.values

        # ATR for stops / targets
        self._atr = atr(high, low, close, self.atr_period).values

        # 20-bar Donchian channel — SHIFTED by 1 (prior-period high/low) to
        # avoid lookahead. close > prior 20-bar high → long breakout.
        donch_up = high.rolling(window=self.donchian_period, min_periods=self.donchian_period).max().shift(1)
        donch_lo = low.rolling(window=self.donchian_period, min_periods=self.donchian_period).min().shift(1)
        self._donch_up = donch_up.values
        self._donch_lo = donch_lo.values

        # Volume SMA
        self._vol_sma = sma(volume, self.volume_ma_period).values

        # Squeeze state machine — precompute the count of consecutive in-squeeze
        # bars at each index. Vectorised so it's cheap and equivalent to a loop.
        in_sq = (
            (np.asarray(self._bb_up) < np.asarray(self._kc_up))
            & (np.asarray(self._bb_lo) > np.asarray(self._kc_lo))
        )
        # NaN-safe: treat NaN as "not in squeeze"
        bb_nan = ~np.isfinite(self._bb_up) | ~np.isfinite(self._kc_up)
        in_sq = in_sq & ~bb_nan
        # Consecutive-True counter: resets to 0 on False, increments by 1 on True.
        counts = np.zeros(len(in_sq), dtype=int)
        run = 0
        for k, v in enumerate(in_sq):
            run = run + 1 if v else 0
            counts[k] = run
        self._squeeze_count = counts
        self._in_squeeze = in_sq

        # Track entry bar for max-hold timestop and re-arm hysteresis.
        # Once ARM fires and an entry happens, we wait for the squeeze to
        # release (squeeze_count drops to 0) before considering re-arm.
        self._entry_bar: int | None = None
        self._consumed_squeeze_end: int = -1  # bar index where we last consumed an armed squeeze

    def _position_units(self, price: float, sl_distance: float) -> int:
        # backtesting.py 0.6.5 only accepts integer units.
        # Under HARNESS-level price scaling (PRICE_SCALE=0.001 in the runner),
        # 1 returned unit == 0.001 BTC, matching Binance USDT-M perp qty_step.
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _armed(self, i: int) -> bool:
        """Armed iff most recent squeeze run >= squeeze_min_bars, and we have
        NOT already consumed a trade off this same squeeze episode.

        A "squeeze episode" is a contiguous run of in-squeeze bars. The end of
        the episode is when squeeze_count drops to 0. We record the bar index
        at which we last consumed an armed trigger and refuse to re-arm until
        the count has dropped to 0 since then.
        """
        if self._squeeze_count[i] < self.squeeze_min_bars:
            return False
        if self._consumed_squeeze_end >= 0:
            # Find the most recent bar where squeeze_count dropped to 0
            # AFTER the bar we consumed at. If none, we're still in the same
            # episode -> not re-armed.
            last_reset = -1
            for j in range(i, self._consumed_squeeze_end, -1):
                if self._squeeze_count[j] == 0:
                    last_reset = j
                    break
            if last_reset <= self._consumed_squeeze_end:
                return False
        return True

    def _vol_ok(self, i: int) -> bool:
        vol_sma_v = self._vol_sma[i]
        if not np.isfinite(vol_sma_v):
            return False
        return float(self.data.Volume[-1]) > self.volume_multiple * vol_sma_v

    def _long_break(self, i: int) -> bool:
        up = self._donch_up[i]
        if not np.isfinite(up):
            return False
        return float(self.data.Close[-1]) > up

    def _short_break(self, i: int) -> bool:
        lo = self._donch_lo[i]
        if not np.isfinite(lo):
            return False
        return float(self.data.Close[-1]) < lo

    def next(self) -> None:
        if not self.enabled:
            return

        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])

        # Position management: time stop
        if self.position:
            if self._entry_bar is not None:
                if (i - self._entry_bar) >= self.max_hold_bars:
                    self.position.close()
                    self._entry_bar = None
                    return
            return

        # Need ATR for sizing
        atr_v = self._atr[i]
        if not np.isfinite(atr_v) or atr_v <= 0:
            return

        sl_dist = self.stop_atr_mult * atr_v
        tp_dist = self.target_atr_mult * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if not self._armed(i):
            return
        if not self._vol_ok(i):
            return

        if self._long_break(i):
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
            self._entry_bar = i
            self._consumed_squeeze_end = i
        elif self.allow_shorts and self._short_break(i):
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
            self._entry_bar = i
            self._consumed_squeeze_end = i
