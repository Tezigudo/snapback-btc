"""
AdaptiveTrendV1 + session_volume_filter improvement (research-only).

ONE feature added on top of AdaptiveTrendV1 (Algorithm 1 base):
    Only allow ENTRIES when the H6 bar's volume is > `volume_multiplier` x
    its rolling hour-of-day bucket mean (60-day window). Exits, trailing
    stop, sizing are unchanged.

Rationale
---------
AdaptiveTrend V1 is an H6 momentum breakout. The intuition behind the
session_volume_filter is microstructural: H6 boundaries (00, 06, 12, 18
UTC) coincide with Asia / Europe / US session opens. Volume regimes
differ markedly across these sessions; entries during a session's
quiet-volume window are noisier and more vulnerable to mean reversion.
A 1.2x bucket-mean filter only fires on "above-typical" session volume,
which is a coarse proxy for institutional participation.

Design notes
------------
- Subclass, not modification. Base default behaviour unchanged; opt-in
  only via the test-arm config (volume_multiplier supplied; gate active).
- Volume is aggregated on the same H6 frame the base uses (sum of 15m
  volumes inside each right-closed 6h window).
- Bucket: hour-of-day taken from the H6 right-aligned timestamp. With
  H6 right-closed 00/06/12/18 boundaries there are exactly 4 buckets
  (0, 6, 12, 18). For each H6 bar we compute the trailing 60-day mean
  of volume within that same bucket, shift(1) to keep causal, and
  forward-fill onto the 15m grid.
- The gate fires at H6 close boundaries (same cadence as the entry
  decision). Entry-only — exits remain unconditional.
- 60-day window is 60 H6 bars/day worth of bucket samples =>
  60 samples per bucket (one per day). We use min_periods=10 to allow
  the first ~10 calendar days of warmup before the filter activates;
  before that the gate is permissive (passes) so the base behaviour is
  preserved early in any window. This avoids the gate silently
  zeroing out entries in the first 10 days of every OOS window which
  would inflate the apparent filter effect.

Authority: research-only. NOT wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals_adaptive_trend import (
    AdaptiveTrendV1,
    _resample_h6,
)


class AdaptiveTrendV1_session_volume(AdaptiveTrendV1):
    """AdaptiveTrendV1 with hour-of-day session-volume entry gate."""

    # Multiplier vs the trailing bucket mean. 1.2 = entries only when
    # current H6 volume is >= 120% of the 60-day same-bucket average.
    volume_multiplier: float = 1.2
    # Trailing window in days for the bucket-mean calc.
    volume_lookback_days: int = 60
    # Minimum samples (in the bucket) before the gate activates. Below
    # this, the gate passes (base behaviour) to avoid warmup artifacts.
    volume_min_samples: int = 10

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        # Base init builds self._mom, self._atr, self._close_h6, self._index
        # and trailing-stop state. We add the session-volume series on top.
        super().init()

        # Pull 15m OHLCV so we can resample volume on the same H6 grid.
        df_15m = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
                "Volume": self.data.Volume,
            },
            index=self.data.index,
        )
        # H6 OHLC (same grid as base, right-closed 6h windows anchored
        # on 00 UTC). Reuse base's resample to keep the grid identical.
        h6_ohlc = _resample_h6(df_15m)
        # H6 volume on the same grid: sum of 15m volumes inside the
        # right-closed 6h window. dropna() afterwards to align indices.
        h6_vol = (
            df_15m["Volume"]
            .resample("6h", label="right", closed="right")
            .sum()
        )
        # Align: keep only the H6 rows that survived dropna() in
        # _resample_h6 (i.e. where OHLC is non-null).
        h6_vol = h6_vol.reindex(h6_ohlc.index)

        # Bucket = hour-of-day from the H6 right-aligned timestamp.
        bucket = h6_ohlc.index.hour

        # Per-bucket rolling 60-day mean of volume. With 4 H6 bars/day
        # and 4 buckets, the bucket sees one sample per day. We
        # therefore use a window-of-60 rolling mean over the bucket
        # subseries (not over H6 bars).
        bucket_means: dict[int, pd.Series] = {}
        for b in sorted(set(bucket.tolist())):
            sub = h6_vol[bucket == b]
            roll = sub.rolling(
                window=self.volume_lookback_days,
                min_periods=self.volume_min_samples,
            ).mean()
            bucket_means[b] = roll

        # Reassemble bucket_mean aligned to H6 index. Each H6 row gets
        # the trailing bucket mean from its own bucket subseries.
        bm_full = pd.Series(np.nan, index=h6_ohlc.index, dtype=float)
        for b, ser in bucket_means.items():
            bm_full.loc[ser.index] = ser.values

        # shift(1) to ensure the H6 bar we're DECIDING on uses ONLY
        # buckets that closed BEFORE the current one. Without shift,
        # the bucket sample at index i includes h6_vol[i] which is
        # the volume of the bar we are entering on — borderline
        # look-ahead. With shift(1) the gate uses the trailing mean
        # of samples strictly before now.
        bm_full = bm_full.shift(1)
        vol_shift = h6_vol.shift(1)

        # Forward-fill onto the 15m index for next() lookups. Because
        # the gate only fires on H6 close boundaries (where the value
        # is the just-shifted previous bucket mean and previous-H6
        # volume), the ffill in between H6 boundaries is harmless.
        bm_15m = bm_full.reindex(df_15m.index, method="ffill")
        vol_15m = vol_shift.reindex(df_15m.index, method="ffill")

        self._bucket_mean_arr = bm_15m.values
        self._h6_vol_prev_arr = vol_15m.values

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        # Position-management path is identical to base; delegate.
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

        # --- session_volume_filter FEATURE ---
        # When the bucket mean is NaN (warmup), we let entries through
        # so we don't artifically suppress the first ~10 trading days of
        # each window. After warmup, require previous-H6 volume >=
        # multiplier * trailing bucket mean.
        bm_v = self._bucket_mean_arr[i]
        vol_prev = self._h6_vol_prev_arr[i]
        if (
            np.isfinite(bm_v)
            and np.isfinite(vol_prev)
            and bm_v > 0
        ):
            if vol_prev < self.volume_multiplier * bm_v:
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
