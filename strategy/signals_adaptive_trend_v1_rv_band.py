"""
AdaptiveTrendV1 + RV-band gate improvement (research-only).

ONE feature added on top of AdaptiveTrendV1 (Algorithm 1 base):
    Only allow ENTRIES when the realised-volatility (30-day, annualised)
    percentile rank — measured in a trailing 365-day rolling window — sits
    inside the [0.25, 0.75] band.  Block when:
      - RV percentile < 0.25  (dead-vol regime, breakouts whipsaw)
      - RV percentile > 0.75  (vol-crush regime, mean-reversion kicks in)

Rationale
---------
Sibling regime_gate_vol (ATR/Close 60th-pct, one-sided gate) was SHELVED
(-30.6pp / -178 trades).  That gate filtered the bottom 60% of vol.  THIS
variant tests a DIFFERENT hypothesis: trend-following pays best in the
MIDDLE of the vol distribution.  At very low vol, signals are noise.  At
very high vol, regime transitions / vol-crush events dominate and trend
following whipsaws on the way down.  We therefore want a two-sided
"sweet-spot" gate rather than a one-sided "high vol = good" gate.

Vol estimator (causal, documented choice)
-----------------------------------------
Annualised realised volatility computed on the 1-hour frame:
    log_ret_1h = ln(C_t / C_{t-1})
    RV_per_hour = stdev(log_ret_1h over trailing 720h) * sqrt(8760)
We use 1h closes (last 15m bar of each hour) which gives more granular RV
than the H6 frame used by the base.  Annualisation constant 8760 = 24*365.

Distribution
------------
Rolling 365-day window of RV_per_hour (8760 hours) → percentile rank.
Gate opens when 0.25 <= percentile <= 0.75.

Strictly causal: the 1h RV series is shift(1)'d before the rank window so
the value at bar i reflects only the most recently CLOSED hour and prior.
This matches the base's H6 shift(1) discipline.

Gate semantics
--------------
The gate is sampled at H6 close boundaries via ffill onto the 15m grid —
identical alignment pattern to the sibling vol-gate.  Entry-only: exits
remain unconditional so we never hold through a regime break.

Authority: research-only.  NOT wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals_adaptive_trend import AdaptiveTrendV1


class AdaptiveTrendV1_rv_band(AdaptiveTrendV1):
    """AdaptiveTrendV1 with two-sided RV-percentile entry gate."""

    # Gate config — defaults match the pinned hypothesis.
    # rv_window_hours is FIXED at 720 (30d, annualised via sqrt(8760)).  Only
    # the band edges and rank-lookback are tuned across arms in the runner.
    rv_window_hours: int = 720
    rv_rank_lookback_days: int = 365
    rv_band_lo: float = 0.25
    rv_band_hi: float = 0.75

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        # Base init builds self._mom, self._atr, self._close_h6, self._index
        # and trailing-stop state.  We add the RV-band gate on top.
        super().init()
        self._build_rv_band_gate()

    # ------------------------------------------------------------------ gate build

    def _build_rv_band_gate(self) -> None:
        """Precompute the gate boolean array aligned to the 15m index.

        Steps (all causal):
          1. Resample 15m -> 1h (last close per hour).
          2. log_ret_1h = ln(C / C.shift(1)).
          3. RV_per_hour = stdev(log_ret_1h, 720) * sqrt(8760).  Annualised.
          4. RV_pct = rolling-(365*24)-hour percentile rank of RV_per_hour.
          5. shift(1) — value at hour t uses bars t-1 and prior (causal).
          6. boolean: lo <= RV_pct_shift <= hi.  Warmup (NaN) -> blocked.
          7. Reindex onto 15m grid via ffill (gate persists between H closes).
        """
        df_15m = pd.DataFrame(
            {
                "Close": self.data.Close,
            },
            index=self.data.index,
        )

        # 1. Resample 15m -> 1h.  Take the LAST close per hour (the close at
        #    the hour boundary; matches how a live bot sees the hourly close).
        close_1h = df_15m["Close"].resample("1h", label="right", closed="right").last()
        close_1h = close_1h.dropna()

        # 2. log returns at hourly frequency.
        log_ret_1h = np.log(close_1h / close_1h.shift(1))

        # 3. Annualised realised vol via 720h stdev * sqrt(8760).
        rv_per_hour = (
            log_ret_1h.rolling(self.rv_window_hours, min_periods=self.rv_window_hours)
            .std()
            * np.sqrt(8760.0)
        )

        # 4. Percentile rank in trailing 365d (8760h) window.
        rank_window = int(self.rv_rank_lookback_days * 24)
        rv_pct = rv_per_hour.rolling(rank_window, min_periods=rank_window).rank(pct=True)

        # 5. Causal shift — value at hour t reflects bars t-1 and prior.
        rv_pct_shift = rv_pct.shift(1)

        # 6. Two-sided band gate; warmup (NaN) -> blocked.
        gate_1h = (
            (rv_pct_shift >= self.rv_band_lo) & (rv_pct_shift <= self.rv_band_hi)
        ).astype(float)
        warmup_mask = rv_pct_shift.isna()
        gate_1h = gate_1h.where(~warmup_mask, other=0.0)

        # 7. Reindex onto the 15m grid via ffill (gate persists between hourly
        #    closes — same alignment pattern the sibling vol-gate uses).
        aligned = gate_1h.reindex(df_15m.index, method="ffill")
        self._rv_band_gate = aligned.fillna(0.0).values

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        # Replicate the base's next() control flow but inject the RV-band
        # check into the ENTRY path.  Exits (position-management branch) are
        # delegated to the base — exits unconditional, matching the sibling
        # vol-gate pattern (we never hold through a regime break).
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

        # --- rv_band FEATURE: require lo <= RV_pct <= hi. ---
        if not bool(self._rv_band_gate[i] > 0.5):
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
