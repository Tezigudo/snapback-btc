"""
CarryHarvester v4 — adds a trend-direction gate to fix the worst v2/v3
failure mode: carrying *against a sustained trend*.

The catastrophic folds in v2 / v3 walk-forward (fold 27: -47%, fold 17:
-21%, fold 19: -25%) all had the same shape — funding stayed persistently
ONE SIGN while price moved STRONGLY in the funding-direction:

  - Fold 27 (Oct→Nov 2024, Trump rally): funding > 0 (longs paying),
    so v2/v3 entered SHORT 38 times into a +35% rally. Got stopped 38
    times. Net −47%.
  - Fold 17 (Dec 2023→Jan 2024): same pattern, smaller scale.

The v2 `max_24h_change_pct` filter and v3 ATR-percentile / DD-breaker
gates are *reactive* — they fire after a move has already happened or
after equity has already dropped. None of them prevent the initial
"short into the rally" trades.

v4 adds a `trend_ema_period`-bar EMA on entry-TF close. Entry gate:
  - To SHORT (funding > 0): require close < trend_ema (BTC in downtrend)
  - To LONG  (funding < 0): require close > trend_ema (BTC in uptrend)

This refuses carry trades that would fight the prevailing trend. The
trend EMA is *long* (default 800 bars × 15m = 200h = 8.3 days) so it
filters by macro regime, not by intraday noise.

trend_ema_period=0 disables (v3 behaviour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

LOOKBACK_24H_15M = 96


class CarryHarvesterV4(Strategy):
    # --- v2 params ---
    funding_threshold = 0.0002
    funding_exit_threshold = 0.00005
    sl_pct = 0.01
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True
    max_24h_change_pct = 100.0

    # --- v3 gates ---
    atr_window_bars = 96
    atr_lookback_bars = 2880
    atr_percentile_threshold = 100.0
    dd_lookback_bars = 1920
    dd_halt_pct = 100.0

    # --- v4 NEW: trend gate ---
    # EMA period in entry-TF bars. At 15m: 800 bars ≈ 8.3 days.
    # 0 = gate OFF (v3 behaviour).
    trend_ema_period = 0

    def init(self) -> None:
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        self._atr_series = tr.rolling(
            self.atr_window_bars, min_periods=self.atr_window_bars
        ).mean().values

        if self.trend_ema_period > 0:
            self._trend_ema = close.ewm(
                span=self.trend_ema_period, adjust=False
            ).mean().values
        else:
            self._trend_ema = None

    def _position_units(self, price: float, sl_distance: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _fast_move_block(self) -> bool:
        if self.max_24h_change_pct >= 100.0:
            return False
        if len(self.data) <= LOOKBACK_24H_15M:
            return False
        ref = self.data.Close[-LOOKBACK_24H_15M - 1]
        now = self.data.Close[-1]
        if ref <= 0 or not np.isfinite(ref):
            return False
        change = abs(now / ref - 1.0) * 100.0
        return change > self.max_24h_change_pct

    def _atr_percentile_block(self) -> bool:
        if self.atr_percentile_threshold >= 100.0:
            return False
        i = len(self.data) - 1
        if i < self.atr_lookback_bars:
            return False
        cur = self._atr_series[i]
        if not np.isfinite(cur):
            return False
        lookback = self._atr_series[max(0, i - self.atr_lookback_bars):i]
        lookback = lookback[np.isfinite(lookback)]
        if len(lookback) < self.atr_lookback_bars // 2:
            return False
        pct = (lookback < cur).mean() * 100.0
        return pct > self.atr_percentile_threshold

    def _drawdown_block(self) -> bool:
        if self.dd_halt_pct >= 100.0:
            return False
        if not hasattr(self, "_recent_equity"):
            self._recent_equity = []
        self._recent_equity.append(self.equity)
        if len(self._recent_equity) > self.dd_lookback_bars:
            self._recent_equity = self._recent_equity[-self.dd_lookback_bars:]
        if len(self._recent_equity) < self.dd_lookback_bars // 4:
            return False
        peak = max(self._recent_equity)
        if peak <= 0:
            return False
        dd_pct = (peak - self.equity) / peak * 100.0
        return dd_pct > self.dd_halt_pct

    def _trend_block(self, want_short: bool) -> bool:
        """v4 NEW. Block entries that would fight the trend EMA.

        want_short=True if we're about to short (funding > 0).
        Block short  when close > trend (BTC in uptrend → don't short).
        Block long   when close < trend (BTC in downtrend → don't long).
        """
        if self._trend_ema is None or self.trend_ema_period <= 0:
            return False
        i = len(self.data) - 1
        trend = self._trend_ema[i]
        if not np.isfinite(trend) or trend <= 0:
            return False
        close = self.data.Close[-1]
        if want_short:
            return close > trend     # uptrend → refuse the short
        else:
            return close < trend     # downtrend → refuse the long

    def next(self) -> None:
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]
        dd_block = self._drawdown_block()

        if not np.isfinite(funding_v) or not np.isfinite(close_v):
            return

        if self.position:
            if abs(funding_v) < self.funding_exit_threshold:
                self.position.close()
                return
            if self.position.is_long and funding_v > 0:
                self.position.close()
                return
            if self.position.is_short and funding_v < 0:
                self.position.close()
                return
            return

        if abs(funding_v) < self.funding_threshold:
            return
        if self._fast_move_block():
            return
        if self._atr_percentile_block():
            return
        if dd_block:
            return

        want_short = funding_v > 0
        if self._trend_block(want_short):
            return

        sl_dist = self.sl_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if want_short:
            if not self.allow_shorts:
                return
            sl = close_v + sl_dist
            self.sell(size=units, sl=sl)
        else:
            sl = close_v - sl_dist
            self.buy(size=units, sl=sl)
