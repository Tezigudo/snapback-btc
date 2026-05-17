"""
DayTradeMultiFactorBTC — multi-factor day-trading-style strategy on 15m.

Built per user spec (P5):
  - 15m entry timeframe (back from 4h, user wants day-trade pace)
  - RSI(14) thresholds: OVS<40 (long candidate), OVB>70 (short candidate)
    (user's asymmetric thresholds — tighter on shorts than longs)
  - Volume confirmation (volume > X × SMA(20) — real interest)
  - Candlestick pattern (bullish/bearish engulfing or hammer)
  - Trend filter (close vs EMA(200) on 15m)
  - Funding-rate "sentiment" gate (refuse crowded-trade direction)
  - MACD histogram confirmation (momentum agreement)
  - Max hold = 14 days (2 weeks at user request) = 1344 bars at 15m
  - 1.5% SL / 3% TP geometry (2:1 R:R)

Research basis (from internet search, May 2026):
  - RSI+MACD+Volume+Bollinger hybrid is a well-documented combo for
    crypto day trading (Medium @redsword_23261)
  - Bullish engulfing + volume confirmation: 65-75% win rate in BTC
    pair backtests (altrady, altFINS)
  - Funding rate as sentiment proxy: extreme funding (>0.05% / 8h) =
    crowded trade vulnerable to squeeze (Phemex, CoinMarketCap docs)

Entry rules — ALL must be true for a LONG:
  1. RSI(14) < rsi_long_threshold (default 40)
  2. Volume(t) > volume_multiple × SMA(Volume, 20)
  3. Bullish candle: bullish engulfing OR hammer
  4. Close > EMA(200) (in uptrend)
  5. Funding rate NOT extremely positive (not entering a crowded long)
  6. MACD histogram > 0 (bullish momentum)

SHORT = mirror with RSI > 70, bearish engulfing, close < EMA(200), funding
not extremely negative, MACD histogram < 0.

Exit rules:
  - SL hit (1.5% adverse)
  - TP hit (3.0% favourable)
  - Time stop (max_hold_bars)
  - Adverse trend cross (close moves to wrong side of EMA(200))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import (
    bearish_engulfing,
    bullish_engulfing,
    ema,
    hammer,
    macd,
    rsi,
    sma,
)


class DayTradeMultiFactorBTC(Strategy):
    # --- entry thresholds (user-specified RSI(14) asymmetric) ---
    rsi_period = 14
    rsi_long_threshold = 40.0       # OVS < 40 (long candidate)
    rsi_short_threshold = 70.0      # OVB > 70 (short candidate)

    # --- volume confirmation ---
    volume_ma_period = 20
    volume_multiple = 1.5           # volume must be 1.5× SMA(20)

    # --- candlestick / trend ---
    require_candlestick = True       # require bullish/bearish engulfing or hammer
    mf_trend_ema_period = 200       # 200-EMA on 15m for trend filter
                                    # (name avoids collision with carry-v4 trend_ema_period
                                    #  which can be 0 in shared StrategyParams)
    require_trend = True            # require close on correct side of EMA(200)

    # --- MACD momentum filter ---
    require_macd = True             # require MACD histogram sign matches direction
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # --- funding sentiment proxy (refuse crowded trade) ---
    funding_extreme_threshold = 0.0005   # ±0.05% per 8h is "extreme"
    require_funding_not_extreme = True

    # --- exits ---
    sl_pct = 0.015                  # 1.5% stop loss
    tp_pct = 0.03                   # 3.0% take profit
    max_hold_bars = 1344            # 14 days × 96 bars/day at 15m

    # --- sizing ---
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        open_ = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self._rsi = rsi(close, self.rsi_period).values
        self._vol_sma = sma(volume, self.volume_ma_period).values
        self._trend_ema = ema(close, self.mf_trend_ema_period).values
        _, _, self._macd_hist = (s.values for s in
                                  macd(close, self.macd_fast, self.macd_slow, self.macd_signal))
        self._bull_engulf = bullish_engulfing(open_, high, low, close).values
        self._bear_engulf = bearish_engulfing(open_, high, low, close).values
        self._hammer = hammer(open_, high, low, close).values
        self._entry_bar: int | None = None

    def _position_units(self, price: float, sl_distance: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _long_signal(self, i: int) -> bool:
        # 1. RSI oversold
        if not (np.isfinite(self._rsi[i]) and self._rsi[i] < self.rsi_long_threshold):
            return False
        # 2. Volume confirmation
        vol_sma_v = self._vol_sma[i]
        if not (np.isfinite(vol_sma_v) and self.data.Volume[-1] > self.volume_multiple * vol_sma_v):
            return False
        # 3. Trend filter
        if self.require_trend:
            t = self._trend_ema[i]
            if not (np.isfinite(t) and self.data.Close[-1] > t):
                return False
        # 4. Candlestick pattern
        if self.require_candlestick:
            if not (bool(self._bull_engulf[i]) or bool(self._hammer[i])):
                return False
        # 5. MACD agreement
        if self.require_macd:
            h = self._macd_hist[i]
            if not (np.isfinite(h) and h > 0):
                return False
        # 6. Funding sentiment (refuse crowded long)
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f > self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        return True

    def _short_signal(self, i: int) -> bool:
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
        if self.require_candlestick:
            if not bool(self._bear_engulf[i]):
                return False
        if self.require_macd:
            h = self._macd_hist[i]
            if not (np.isfinite(h) and h < 0):
                return False
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f < -self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        return True

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Position management: time stop + adverse-trend exit
        if self.position:
            if self._entry_bar is not None:
                if (i - self._entry_bar) >= self.max_hold_bars:
                    self.position.close()
                    self._entry_bar = None
                    return
            # Adverse-trend exit
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

        # Entry
        sl_dist = self.sl_pct * close_v
        tp_dist = self.tp_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if self._long_signal(i):
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
            self._entry_bar = i
        elif self._short_signal(i):
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
            self._entry_bar = i
