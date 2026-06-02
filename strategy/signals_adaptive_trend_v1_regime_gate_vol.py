"""
AdaptiveTrendV1 + regime_gate_vol improvement (research-only).

ONE feature added on top of AdaptiveTrendV1 (Algorithm 1 base):
    Only allow ENTRIES when the most-recently-closed H6 bar's ATR/Close
    ratio is STRICTLY ABOVE the 60th percentile of its trailing 60-day
    distribution.  Exits / trailing stop / sizing are unchanged.

Rationale
---------
ADX as a regime classifier failed (sibling _postfrac_adaptrend_v1_adx
ablation SHELVED).  Per the handoff: "different feature class per the H1
confirmation insight" — instead of measuring directional strength, gate on
realized volatility magnitude.  Trend followers profit from sustained
moves; compressed-vol regimes generate noise breakouts that whipsaw the
ATR trail.  A vol-percentile gate suppresses entries when vol is in the
bottom 60% of the recent distribution.

Vol estimator (causal, documented choice)
-----------------------------------------
We use ATR/Close (normalised range) on the H6 frame the base already
maintains.  ATR is Wilder, period = atr_period_h6 = 14 (same as the base's
trailing-stop ATR — no new tuning surface).  ATR/Close is dimensionless
and robust across regime price levels (BTC at $20k vs $70k); a raw ATR
percentile would bias the gate toward the prices' absolute level.

Distribution
------------
Rolling 60-day window of ATR/Close at H6 granularity = 60 * 4 = 240 H6
bars.  At each H6 close ts we compute the 60th-percentile of the trailing
240 H6 ratios INCLUDING the just-closed bar (i.e. on the shift(1)'d series
that the base already uses).  Gate opens when ratio[t] > q60[t].

Strictly causal: the H6 frame is shifted by 1 before the rolling window
is taken (so at 15m bar i the value reflects the most recently CLOSED H6
bar), exactly the pattern the base uses for MOM/ATR.

Gate semantics
--------------
The gate fires at H6 close boundaries (same cadence as the entry
decision).  It is ENTRY-ONLY — exits remain unconditional, matching the
sibling ADX gate (otherwise we'd hold positions through regime breaks).

Authority: research-only.  NOT wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import atr as wilder_atr
from strategy.signals_adaptive_trend import (
    AdaptiveTrendV1,
    _resample_h6,
)


class AdaptiveTrendV1_regime_gate_vol(AdaptiveTrendV1):
    """AdaptiveTrendV1 with rolling-60d 60th-pct ATR/Close entry gate."""

    # Gate config — defaults match the handoff hypothesis.
    # vol_lookback_days * 4 = number of H6 bars in the trailing distribution.
    vol_lookback_days: int = 60
    vol_quantile: float = 0.60
    # ATR period for the vol estimator — re-use the base's H6 ATR period so
    # there's no extra tuning surface and ATR is already computed.
    vol_atr_period_h6: int = 14

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        # Base init builds self._mom, self._atr, self._close_h6, self._index
        # and trailing-stop state.  We add the vol-gate series on top.
        super().init()
        self._build_vol_gate()

    # ------------------------------------------------------------------ gate build

    def _build_vol_gate(self) -> None:
        """Precompute the gate boolean array aligned to the 15m index.

        Steps (all causal):
          1. Resample 15m -> H6 (same _resample_h6 the base uses).
          2. ATR(14) on H6 via Wilder.
          3. ratio = ATR / Close.
          4. shift(1) — value at H6 close ts is the just-closed bar's ratio.
          5. rolling 60d (240 H6-bar) 60th-percentile on the shifted series.
          6. boolean: ratio > q60.  Warmup bars (NaN) -> blocked (0).
          7. Reindex / ffill onto the 15m index.
        """
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

        atr_h6 = wilder_atr(
            h6["High"], h6["Low"], h6["Close"], period=self.vol_atr_period_h6
        )
        ratio = atr_h6 / h6["Close"]
        # Causal shift — value at H6 close ts is the just-closed bar's ratio.
        ratio_shift = ratio.shift(1)

        window_bars = int(self.vol_lookback_days * 4)  # 4 H6 bars per day
        q_series = ratio_shift.rolling(
            window_bars, min_periods=window_bars
        ).quantile(self.vol_quantile)

        gate_h6 = (ratio_shift > q_series).astype(float)  # 1.0 = pass, 0.0 = block
        warmup_mask = ratio_shift.isna() | q_series.isna()
        gate_h6 = gate_h6.where(~warmup_mask, other=0.0)

        # Reindex onto 15m grid (ffill — gate persists between H6 closes).
        aligned = gate_h6.reindex(df_15m.index, method="ffill")
        self._vol_gate = aligned.fillna(0.0).values

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        # We replicate the base's next() control flow but inject the vol
        # check into the entry path.  Exits (position-management branch) are
        # left to the base — delegate when self.position is truthy.
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

        # --- regime_gate_vol FEATURE: require ATR/Close > 60th pct (60d). ---
        if not bool(self._vol_gate[i] > 0.5):
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
