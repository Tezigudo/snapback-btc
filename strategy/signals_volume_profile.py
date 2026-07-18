"""
VolumeProfilePOC — Previous-session POC retest bounce strategy on 15m BTC.

Research basis (FUTURE_DIRECTIONS.md §2, RESEARCH_PNL_FINDINGS.md §3):
  - Volume Profile Point of Control (POC): the price at which the most volume
    traded during a closed session. Institutional desks frequently defend HVNs
    (High-Volume Nodes), creating short-term mean-reversion bounces.
  - ADX < 20 gate: POC fades get stopped out repeatedly in trending markets
    (multiple source warnings in RESEARCH_PNL_FINDINGS.md §3). The chop/balanced
    regime is the only documented favorable environment for this pattern.

SPIKE intent: prove edge or shelf cleanly. This is NOT a production strategy.
Research explicitly warns "No peer-reviewed or vendor-published backtest reports
a Sharpe ratio for a BTC 15m POC-bounce system."

Entry rules — LONG:
  1. Prior-session POC is finite (session_poc returns a valid value).
  2. ADX(14)[i] < adx_chop_threshold (balanced/range day, default 20).
  3. POC retest zone: low[i] is within poc_proximity_pct of the POC level
     (i.e. poc*(1-pct) <= low[i] <= poc*(1+pct)).
  4. Rejection candle (when require_rejection_candle=True):
       close[i] > open[i]  AND  close[i] >= poc[i]
     Interpretation: price touched the POC and closed above it bullishly.

SHORT = mirror:
  3. high[i] is within poc_proximity_pct of the POC.
  4. close[i] < open[i]  AND  close[i] <= poc[i].

Exit rules:
  - Initial SL: entry ± sl_atr_multiple × ATR(14)
  - Take profit: entry ± tp_atr_multiple × ATR(14)
  - Time stop: position closed after max_hold_bars bars regardless

Sizing: verbatim copy of DivergenceV1._position_units (risk-based, leverage-capped).

Lookahead safety: session_poc() is shift(1)-safe — each bar sees only the POC
of the PRIOR completed session, never the current session's accumulating volume.

Authority: spike only — not wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import adx, atr, session_poc


class VolumeProfilePOC(Strategy):
    # --- Entry filter: ADX chop regime ---
    adx_period: int = 14
    adx_chop_threshold: float = 20.0   # only trade when ADX < 20 (range day)

    # --- POC computation ---
    poc_n_bins: int = 50
    poc_session: str = "UTC_day"

    # --- Entry: POC retest pattern ---
    poc_proximity_pct: float = 0.005   # 0.5% proximity band around POC
    require_rejection_candle: bool = True

    # --- Exits ---
    atr_period: int = 14
    sl_atr_multiple: float = 1.5
    tp_atr_multiple: float = 3.0
    max_hold_bars: int = 96            # 24h at 15m

    # --- Sizing ---
    risk_per_trade_pct: float = 1.0
    leverage: int = 5
    allow_shorts: bool = True

    # ------------------------------------------------------------------ init

    def init(self) -> None:
        # Build Series with the real DatetimeIndex so session_poc can group by day.
        idx = pd.DatetimeIndex(self.data.index)
        close  = pd.Series(self.data.Close,  index=idx)
        high   = pd.Series(self.data.High,   index=idx)
        low    = pd.Series(self.data.Low,    index=idx)
        open_  = pd.Series(self.data.Open,   index=idx)
        volume = pd.Series(self.data.Volume, index=idx)

        self._close_arr = close.values
        self._high_arr  = high.values
        self._low_arr   = low.values
        self._open_arr  = open_.values

        # POC: prior-session point of control for each bar
        self._poc = session_poc(
            high, low, close, volume,
            session=self.poc_session,
            n_bins=self.poc_n_bins,
        ).values

        # ADX for chop-regime gate
        self._adx = adx(high, low, close, period=self.adx_period).values

        # ATR for sizing and SL/TP
        self._atr = atr(high, low, close, period=self.atr_period).values

        self._entry_bar: int | None = None

    # ------------------------------------------------------------------ sizing

    def _position_units(self, price: float, sl_distance: float) -> int:
        """Integer-units sizing (verbatim mirror of DivergenceV1).

        NOTE: backtesting.py 0.6.5 only accepts integer units. Fractional
        0.001-BTC sizing is implemented via HARNESS-level price scaling
        (see tools/_fractional_run.py). Under scaling: 1 returned "unit"
        == 0.001 BTC, matching Binance USDT-M perp qty_step.
        """
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    # ------------------------------------------------------------------ signals

    def _long_signal(self, i: int) -> bool:
        # 1. POC is valid (prior session exists).
        poc_v = self._poc[i]
        if not np.isfinite(poc_v) or poc_v <= 0:
            return False

        # 2. ADX chop regime.
        adx_v = self._adx[i]
        if not np.isfinite(adx_v):
            return False
        if adx_v >= self.adx_chop_threshold:
            return False

        # 3. Low touches the POC zone from above.
        low_v = self._low_arr[i]
        poc_lo = poc_v * (1.0 - self.poc_proximity_pct)
        poc_hi = poc_v * (1.0 + self.poc_proximity_pct)
        if not (poc_lo <= low_v <= poc_hi):
            return False

        # 4. Rejection candle (bullish).
        if self.require_rejection_candle:
            close_v = self._close_arr[i]
            open_v  = self._open_arr[i]
            if not (close_v > open_v and close_v >= poc_v):
                return False

        return True

    def _short_signal(self, i: int) -> bool:
        if not self.allow_shorts:
            return False

        # 1. POC is valid.
        poc_v = self._poc[i]
        if not np.isfinite(poc_v) or poc_v <= 0:
            return False

        # 2. ADX chop regime.
        adx_v = self._adx[i]
        if not np.isfinite(adx_v):
            return False
        if adx_v >= self.adx_chop_threshold:
            return False

        # 3. High touches the POC zone from below.
        high_v = self._high_arr[i]
        poc_lo = poc_v * (1.0 - self.poc_proximity_pct)
        poc_hi = poc_v * (1.0 + self.poc_proximity_pct)
        if not (poc_lo <= high_v <= poc_hi):
            return False

        # 4. Rejection candle (bearish).
        if self.require_rejection_candle:
            close_v = self._close_arr[i]
            open_v  = self._open_arr[i]
            if not (close_v < open_v and close_v <= poc_v):
                return False

        return True

    # ------------------------------------------------------------------ loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

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
