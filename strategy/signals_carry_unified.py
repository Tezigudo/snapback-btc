"""
Unified CarryHarvester — replaces v1/v2/v3/v4 with one configurable class.

All four carry versions did the same thing — short when funding is positive
(longs paying), long when negative (shorts paying), exit on funding revert
or SL — and stacked progressively more gates. This consolidates them.

Feature flags via class attrs (each "off" by default = original v1 behaviour):
  - max_24h_change_pct < 100      → v2 fast-move skip filter
  - atr_percentile_threshold < 100 → v3 vol-regime gate
  - dd_halt_pct < 100              → v3 drawdown circuit breaker
  - trend_ema_period > 0           → v4 trend-direction gate

Strategy aliases for backwards-compat / sweep configs:
  CarryHarvesterUnifiedV1 — all gates off (v1 behaviour)
  CarryHarvesterUnifiedV2 — fast-move filter only (v2 behaviour)
  CarryHarvesterUnifiedV3 — fast-move + vol-regime + DD-breaker (v3 behaviour)
  CarryHarvesterUnifiedV4 — all gates available (v4 behaviour)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

LOOKBACK_24H_15M = 96


class CarryHarvesterUnified(Strategy):
    # --- carry params ---
    funding_threshold = 0.0002
    funding_exit_threshold = 0.00005
    sl_pct = 0.01
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True

    # --- v2 gate (100 = off) ---
    max_24h_change_pct = 100.0

    # --- v3 gates ---
    atr_window_bars = 96
    atr_lookback_bars = 2880
    atr_percentile_threshold = 100.0   # 100 = off
    dd_lookback_bars = 1920
    dd_halt_pct = 100.0                # 100 = off

    # --- v4 gate ---
    trend_ema_period = 0               # 0 = off

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        if any(t < 100 for t in (self.atr_percentile_threshold,)):
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            self._atr_series = tr.rolling(
                self.atr_window_bars, min_periods=self.atr_window_bars
            ).mean().values
        else:
            self._atr_series = None

        if self.trend_ema_period > 0:
            self._trend_ema = close.ewm(
                span=self.trend_ema_period, adjust=False
            ).mean().values
        else:
            self._trend_ema = None

        self._recent_equity: list[float] = []

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
        return abs(now / ref - 1.0) * 100.0 > self.max_24h_change_pct

    def _atr_percentile_block(self) -> bool:
        if self._atr_series is None or self.atr_percentile_threshold >= 100.0:
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
        # Update rolling equity list every bar regardless of gate state, so
        # if the gate is turned on mid-run via class attr the history exists.
        self._recent_equity.append(self.equity)
        if len(self._recent_equity) > self.dd_lookback_bars:
            self._recent_equity = self._recent_equity[-self.dd_lookback_bars:]
        if self.dd_halt_pct >= 100.0:
            return False
        if len(self._recent_equity) < self.dd_lookback_bars // 4:
            return False
        peak = max(self._recent_equity)
        if peak <= 0:
            return False
        return (peak - self.equity) / peak * 100.0 > self.dd_halt_pct

    def _trend_block(self, want_short: bool) -> bool:
        if self._trend_ema is None or self.trend_ema_period <= 0:
            return False
        trend = self._trend_ema[len(self.data) - 1]
        if not np.isfinite(trend) or trend <= 0:
            return False
        close = self.data.Close[-1]
        return (close > trend) if want_short else (close < trend)

    def next(self) -> None:
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]
        dd_block = self._drawdown_block()  # call every bar to maintain history

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
        if self._fast_move_block() or self._atr_percentile_block() or dd_block:
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
            self.sell(size=units, sl=close_v + sl_dist)
        else:
            self.buy(size=units, sl=close_v - sl_dist)


# --- Back-compat aliases ---
# Each subclass exists so backtest.STRATEGIES can map a distinct class per
# strategy-name (the sweep machinery mutates class attrs via setattr; if two
# names shared a class they'd race). Defaults match the historical v1-v4
# behaviour for replay of old reports.
class CarryHarvesterUnifiedV1(CarryHarvesterUnified):
    max_24h_change_pct = 100.0
    atr_percentile_threshold = 100.0
    dd_halt_pct = 100.0
    trend_ema_period = 0


class CarryHarvesterUnifiedV2(CarryHarvesterUnified):
    atr_percentile_threshold = 100.0
    dd_halt_pct = 100.0
    trend_ema_period = 0


class CarryHarvesterUnifiedV3(CarryHarvesterUnified):
    trend_ema_period = 0


class CarryHarvesterUnifiedV4(CarryHarvesterUnified):
    pass
