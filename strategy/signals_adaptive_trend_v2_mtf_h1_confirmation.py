"""
AdaptiveTrendV2 + mtf_h1_confirmation improvement.

ONE feature added on top of the AdaptiveTrendV2 (Algorithm 2) base:
    Only allow ENTRIES when H1 RSI(14) is NOT extreme (30 < RSI < 70)
    on the most-recently-closed H1 bar.
    Exits / trailing stop / monthly re-opt are unchanged.

Rationale
---------
AdaptiveTrend enters on H6 momentum breakouts (MOM > theta). When the
H1 RSI is extreme (>70 overbought, <30 oversold), mean-reversion is
more likely than continuation. Gating entries on H1 RSI being in the
neutral band (30, 70) avoids buying tops / shorting bottoms where
H1-scale momentum is overextended.

Design notes
------------
- The gate fires at H6 close (same cadence as the entry decision).
  We compute RSI on the FULL H1 frame in init(), shift(1), then
  forward-fill onto the 15m index.
- Strictly causal: we only ever read self._rsi_h1_arr[i] at the current
  15m bar i; the shift(1) ensures the value reflects the just-closed
  H1 bar.
- RSI period 14 is the canonical Wilder cutoff (paper convention;
  matches the existing rsi() helper).
- The neutral band [30, 70] is the textbook Wilder convention for
  "not overbought / not oversold".
- We do NOT touch _maybe_refit / _rebuild_live_signal / exit logic —
  exits MUST keep firing regardless of RSI regime or we'd hold
  positions across extreme moves. The gate is entry-only.

Authority: research-only. Not wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import rsi as wilder_rsi
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2

_H1_RULE = "1h"


def _resample_h1(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample a (capitalised-column) 15m OHLC frame to H1.

    Returns a frame indexed by H1 close timestamps with columns
    [Open, High, Low, Close]. Mirrors the H6 resampler convention
    (right-labelled, right-closed).
    """
    o = df_15m["Open"].resample(_H1_RULE, label="right", closed="right").first()
    h = df_15m["High"].resample(_H1_RULE, label="right", closed="right").max()
    lo = df_15m["Low"].resample(_H1_RULE, label="right", closed="right").min()
    c = df_15m["Close"].resample(_H1_RULE, label="right", closed="right").last()
    h1 = pd.concat({"Open": o, "High": h, "Low": lo, "Close": c}, axis=1).dropna()
    return h1


class AdaptiveTrendV2_mtf_h1_confirmation(AdaptiveTrendV2):
    """AdaptiveTrendV2 with H1 RSI(14) in (30, 70) entry gate."""

    rsi_period_h1: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        # Base init builds self._h6, self._index, self._mom/_atr arrays, etc.
        super().init()

        # Build the H1 frame from the same 15m OHLC the strategy was
        # constructed against, then compute RSI(14) on H1 closes.
        df_15m = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
            },
            index=self.data.index,
        )
        h1 = _resample_h1(df_15m)
        rsi_series = wilder_rsi(h1["Close"], period=self.rsi_period_h1)
        # shift(1) so the value at the H1 close ts is the value of the
        # just-closed bar (mirrors v2's MOM/ATR pipeline).
        aligned = rsi_series.shift(1).reindex(self._index, method="ffill")
        self._rsi_h1_arr = aligned.values

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

        # --- mtf_h1_confirmation FEATURE: require H1 RSI(14) in (30, 70). ---
        rsi_v = self._rsi_h1_arr[i]
        if not np.isfinite(rsi_v):
            return
        if rsi_v <= self.rsi_low or rsi_v >= self.rsi_high:
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
