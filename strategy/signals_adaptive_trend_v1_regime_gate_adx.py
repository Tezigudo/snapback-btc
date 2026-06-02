"""
AdaptiveTrendV1 + regime_gate_adx improvement (research-only).

ONE feature added on top of AdaptiveTrendV1 (Algorithm 1 base):
    Only allow ENTRIES when ADX(14) on the most-recently-closed H6 bar > 25.
    Exits / trailing stop / sizing are unchanged.

Rationale
---------
AdaptiveTrend V1 is a trend-following H6 MOM breakout with ATR trail.
Trend followers typically bleed during chop (the 2022_H1 -7% DD window in
our 5_OOS set has the lowest WR at 35.3% — a chop tell). ADX(14) above 25
is the canonical Wilder cutoff for "trend present". Gating ENTRIES on
H6 ADX > 25 should suppress entries during low-quality range regimes
while preserving the high-quality trend entries.

Design notes
------------
- Subclass, not modification. The base class default behavior is
  unchanged; this variant is opt-in via instantiation.
- ADX is computed on the FULL H6 frame in init() (same resample as
  the base's MOM/ATR pipeline), shifted by 1, and forward-filled onto
  the 15m index. Strictly causal — the value at 15m bar i reflects the
  most recently CLOSED H6 bar.
- The gate fires at H6 close boundaries (same cadence as the entry
  decision). It is entry-only — exits remain unconditional, otherwise
  we'd hold positions through regime breaks.
- Threshold and period are parameterizable for sweep work.

Authority: research-only. NOT wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import adx as wilder_adx
from strategy.signals_adaptive_trend import (
    AdaptiveTrendV1,
    _resample_h6,
)


class AdaptiveTrendV1_regime_gate_adx(AdaptiveTrendV1):
    """AdaptiveTrendV1 with H6 ADX(14) > 25 entry gate."""

    # Threshold for the trend-only regime. 25 is the canonical Wilder cutoff.
    adx_threshold: float = 25.0
    adx_period_h6: int = 14

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        # Base init builds self._mom, self._atr, self._close_h6, self._index
        # and trailing-stop state. We add the ADX series on top.
        super().init()

        # Re-derive the H6 frame the same way the base's compute_h6_signal
        # does, then compute ADX on it, shift(1) to keep the pipeline causal,
        # and forward-fill onto the 15m index.
        df_15m = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
            },
            index=self.data.index,
        )
        h6 = _resample_h6(df_15m)

        adx_series = wilder_adx(
            h6["High"], h6["Low"], h6["Close"], period=self.adx_period_h6
        )
        aligned = adx_series.shift(1).reindex(df_15m.index, method="ffill")
        self._adx_h6_arr = aligned.values

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        # We replicate the base's next() control flow but inject the ADX
        # check into the entry path. Exits (position-management branch) are
        # left to the base. To do this cleanly we copy the entry-path logic
        # here; the position-management branch is identical to the base, so
        # we delegate to super() when self.position is truthy.
        if self.position:
            super().next()
            return

        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]
        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Entry: only at H6 close boundaries. ---
        if not self._is_h6_close_bar(ts):
            return
        if self._last_h6_close_seen == ts:
            return
        self._last_h6_close_seen = ts

        if not np.isfinite(mom_v) or not np.isfinite(atr_v) or atr_v <= 0:
            return

        # --- regime_gate_adx FEATURE: require H6 ADX(14) > threshold. ---
        adx_v = self._adx_h6_arr[i]
        if not np.isfinite(adx_v) or adx_v <= self.adx_threshold:
            return

        # Initial stop seed = entry - alpha * ATR (paper line 6 of Alg 1).
        sl_dist = self.alpha * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if mom_v > self.theta_entry:
            self.buy(size=units, sl=close_v - sl_dist)
            self._entry_bar = i
            self._trail_level = close_v - sl_dist
        elif self.allow_shorts and mom_v < -self.theta_entry:
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._trail_level = close_v + sl_dist
