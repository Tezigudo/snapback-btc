"""
AdaptiveTrendV2 + regime_gate_adx improvement.

ONE feature added on top of the AdaptiveTrendV2 (Algorithm 2) base:
    Only allow ENTRIES when ADX(14) on the most-recently-closed H6 bar > 25.
    Exits / trailing stop / monthly re-opt are unchanged.

Rationale
---------
AdaptiveTrend is a trend follower (H6 MOM breakout + ATR trail). Trend
followers typically bleed during chop. ADX(14) is the canonical
trend-strength filter — values above 25 mark "trend present", below 25
mark "range / chop". Gating entries on H6 ADX > 25 should suppress
entries during low-quality range regimes while keeping the high-quality
trend entries intact.

Design notes
------------
- The gate fires at H6 close (same cadence as the entry decision).
  We compute ADX on the FULL H6 frame in init() (same shape as MOM/ATR
  in the base class), shift(1), then forward-fill onto the 15m index.
- Strictly causal: we only ever read self._adx_h6[i] at the current 15m
  bar i; the shift(1) ensures the value reflects the just-closed H6 bar.
- The ATR_period_h6 default is 14 in the base, so we reuse 14 for ADX
  as well (paper convention; matches the existing adx() helper default).
- We do NOT touch _maybe_refit / _rebuild_live_signal / exit logic —
  exits MUST keep firing inside chop or we'd hold positions through
  regime breaks. The gate is entry-only.

Authority: research-only. Not wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import adx as wilder_adx
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2


class AdaptiveTrendV2_regime_gate_adx(AdaptiveTrendV2):
    """AdaptiveTrendV2 with H6 ADX(14) > 25 entry gate."""

    # Threshold for the trend-only regime. 25 is the canonical Wilder cutoff.
    adx_threshold: float = 25.0
    adx_period_h6: int = 14

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        # Base init builds self._h6, self._index, self._mom/_atr arrays, etc.
        super().init()

        # Compute ADX on the H6 frame, shift(1) to mirror the MOM/ATR pipeline
        # (so the value at H6 close ts is the value of the just-closed bar),
        # then forward-fill onto the 15m index. Pure read; no future leak.
        h6 = self._h6
        adx_series = wilder_adx(
            h6["High"], h6["Low"], h6["Close"], period=self.adx_period_h6
        )
        aligned = adx_series.shift(1).reindex(self._index, method="ffill")
        self._adx_h6_arr = aligned.values

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]

        # Monthly re-opt (unchanged).
        self._maybe_refit(ts)

        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management (UNCHANGED — exits must remain unconditional). ---
        if self.position:
            if not np.isfinite(atr_v) or atr_v <= 0:
                return

            if (
                self._entry_bar is not None
                and (i - self._entry_bar) >= self.max_hold_h6_bars * 24
            ):
                self.position.close()
                self._trail_level = None
                self._entry_bar = None
                return

            trade = self.trades[-1] if self.trades else None
            if trade is None:
                return

            if trade.is_long:
                candidate = close_v - self.alpha * atr_v
                if self._trail_level is None or candidate > self._trail_level:
                    self._trail_level = candidate
                if trade.sl is None or self._trail_level > trade.sl:
                    trade.sl = self._trail_level
                if close_v < self._trail_level:
                    self.position.close()
                    self._trail_level = None
                    self._entry_bar = None
            else:
                candidate = close_v + self.alpha * atr_v
                if self._trail_level is None or candidate < self._trail_level:
                    self._trail_level = candidate
                if trade.sl is None or self._trail_level < trade.sl:
                    trade.sl = self._trail_level
                if close_v > self._trail_level:
                    self.position.close()
                    self._trail_level = None
                    self._entry_bar = None
            return

        # --- Entry path: only at H6 close boundaries. ---
        if not self._is_h6_close_bar(ts):
            return
        if self._last_h6_close_seen == ts:
            return
        self._last_h6_close_seen = ts

        if not np.isfinite(mom_v) or not np.isfinite(atr_v) or atr_v <= 0:
            return

        # Prefix guard (unchanged).
        if self.trade_start_ns > 0 and ts.value < self.trade_start_ns:
            return

        # --- regime_gate_adx FEATURE: require H6 ADX(14) > 25 to enter. ---
        adx_v = self._adx_h6_arr[i]
        if not np.isfinite(adx_v) or adx_v <= self.adx_threshold:
            return

        sl_dist = self.alpha * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        theta = self._active_theta
        if mom_v > theta:
            self.buy(size=units, sl=close_v - sl_dist)
            self._entry_bar = i
            self._trail_level = close_v - sl_dist
        elif self.allow_shorts and mom_v < -theta:
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._trail_level = close_v + sl_dist
