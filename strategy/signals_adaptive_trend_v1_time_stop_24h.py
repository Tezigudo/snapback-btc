"""
AdaptiveTrendV1 + time_stop_24h improvement (research-only).

ONE feature added on top of AdaptiveTrendV1 (Algorithm 1 base):
    Force-close any open position 96 fifteen-minute bars (= 24 hours)
    after entry if it has not already exited via trailing stop or
    initial SL. Tests whether time-decay justifies a hard exit before
    the natural trail-out.

Rationale
---------
AdaptiveTrendV1's trailing stop is patient — alpha=2.0 keeps it ~2.0 H6-ATR
away from price, so a position that goes nowhere bleeds funding + holds
capital that could be redeployed to fresher MOM signals. The base has a
30-day max-hold belt (`max_hold_h6_bars=120` → `120*24=2880` 15m bars), but
that's a safety belt, not a working filter. A 24-hour ceiling forces the
trade either to make its move quickly or get cycled out.

24h corresponds to:
  - 96 fifteen-minute bars (one full day of 15m candles).
  - 4 H6 bars (one full day of the strategy's native cadence).
  - The same horizon the MOM signal looks back over by default
    (momentum_lookback_h6=4), so a stale signal has had a full lookback
    window to play out.

Design notes
------------
- Subclass, not modification. The base class default behavior is
  unchanged; this variant is opt-in via instantiation.
- Opt-in via the `time_stop_24h` class flag (default False). When False,
  next() falls through to identical base behavior — strict
  backward compatibility, matching the half_out_1r pattern.
- The time check uses 15m bar indices: `(i - self._entry_bar) >= time_stop_bars`
  where `time_stop_bars=96` by default. Same machinery as the existing
  base max-hold belt, just a tighter cap.
- The time stop fires in the position-management branch (every 15m bar,
  same cadence as the trailing-stop ratchet). Exits are evaluated before
  the trailing logic so a forced close doesn't get pre-empted by a trail.
- Threshold is parameterizable for sweep work.

CAUTION: this is NOT the same lever as the V2 ablation
`tools/_adaptrend_v2_imp_time_stop_24h.py`, which set `max_hold_h6_bars=96`
on the V2 base — that meant 96 H6 BARS = 24 DAYS, three orders of
magnitude longer than this 24-HOUR cap. Same name, completely different
mechanism. Do not confuse the two when reading prior reports.

Authority: research-only. NOT wired to bot.py.
"""

from __future__ import annotations

import numpy as np

from strategy.signals_adaptive_trend import AdaptiveTrendV1


class AdaptiveTrendV1_time_stop_24h(AdaptiveTrendV1):
    """V1 with a 24-hour hard time stop on open positions."""

    # Opt-in flag. Default OFF so this class is a no-op subclass unless
    # the runner explicitly enables it. Backward-compat per task spec.
    time_stop_24h: bool = False

    # 24h on the 15m carrier = 96 bars. Parameterised for sweep work.
    time_stop_bars: int = 96

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        # Flag OFF -> identical to base.
        if not self.time_stop_24h:
            super().next()
            return

        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]
        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management ---
        if self.position:
            if not np.isfinite(atr_v) or atr_v <= 0:
                return

            # ---- NEW: 24-hour hard time stop ----
            # Fires BEFORE the base's 30-day max-hold belt and BEFORE the
            # trailing stop, because once we've decided the trade is stale
            # we want it out at the next bar's open.
            if (
                self._entry_bar is not None
                and (i - self._entry_bar) >= self.time_stop_bars
            ):
                self.position.close()
                self._trail_level = None
                self._entry_bar = None
                return

            # Hard max-hold belt (unchanged from base; only fires if the
            # time_stop_bars cap is set higher than `max_hold_h6_bars * 24`,
            # which doesn't happen at defaults but keeps the safety belt
            # semantically intact).
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

            # ---- Trailing stop (unchanged from base) ----
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

        # --- Entry: only at H6 close boundaries (unchanged from base). ---
        if not self._is_h6_close_bar(ts):
            return
        if self._last_h6_close_seen == ts:
            return
        self._last_h6_close_seen = ts

        if not np.isfinite(mom_v) or not np.isfinite(atr_v) or atr_v <= 0:
            return

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
