"""
AdaptiveTrendV2_half_out_at_1R — V2 base + 50% scale-out at +1R.

Improvement under test: scale out 50% of the position when price moves +1R
(= alpha * ATR_entry = the SL distance at entry) in our favor.  Let the
remaining 50% ride with the original ATR trailing stop.

Rationale: trend systems typically have a thin per-trade edge bloated by a
few fat-tailed winners.  Scaling out at +1R locks in a small partial profit
on the median trade (where the trail later gives back); the runner half keeps
the tail exposure.  Net effect on per-trade Sharpe is the empirical question.

What changes vs AdaptiveTrendV2 (single feature toggle)
------------------------------------------------------
- On entry: record `_entry_price`, `_entry_sl_dist = alpha * ATR_entry`, and
  a `_half_taken` flag set to False.
- In next(): before the trail logic, check if the current bar's intra-bar
  range reached the +1R level (`high >= entry + sl_dist` for longs,
  `low <= entry - sl_dist` for shorts).  If yes AND we have not already
  half-closed, call `trade.close(portion=0.5)` — backtesting.py schedules
  this for next bar's open (consistent with how other exits flow through
  the engine; see Trade.close source).
- After half-close, the original trade reference is consumed; the remaining
  50% becomes a new trade entry inside backtesting.py.  We must rebind the
  trail's "current trade" reference (`self.trades[-1]`) next bar — already
  what v2 does each iteration, so no extra plumbing needed.
- Trailing stop, max-hold, and re-opt logic are untouched.

Lookahead safety
----------------
- The +1R check uses bar i's high/low (not look-ahead — we know the current
  bar's extrema at bar close, same as how the trail uses `low[i]` /
  `high[i]`).
- backtesting.py executes the close order at bar i+1's open, NOT i's high —
  so the fill price for the partial close is realistic (no perfect-fill
  cheating).
- We DO NOT overwrite the half-take threshold after re-binding; the entry
  price and ATR are stored at entry, not recomputed.

Authority: research-only. Not wired to live bot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2


class AdaptiveTrendV2_half_out_at_1R(AdaptiveTrendV2):
    """V2 with 50% scale-out at +1R take-profit, runner trails the rest."""

    # No new config knobs — the half-out is hard-wired to portion=0.5 at +1R.
    # If we want to A/B different portions or R-multiples later, add them here.

    def init(self) -> None:  # noqa: D401
        super().init()
        # Per-trade half-out bookkeeping.  Reset on every fresh entry.
        self._entry_price: float | None = None
        self._entry_sl_dist: float | None = None
        self._half_taken: bool = False

    def next(self) -> None:  # noqa: C901 — mirrors v2.next() with one inserted block
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        high_v = float(self.data.High[-1])
        low_v = float(self.data.Low[-1])
        ts = self._index[i]

        # Algorithm 2 re-fit (unchanged).
        self._maybe_refit(ts)

        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management: half-out THEN trail. ---
        if self.position:
            if not np.isfinite(atr_v) or atr_v <= 0:
                return

            # Max-hold guard (unchanged from v2).
            if (
                self._entry_bar is not None
                and (i - self._entry_bar) >= self.max_hold_h6_bars * 24
            ):
                self.position.close()
                self._trail_level = None
                self._entry_bar = None
                self._entry_price = None
                self._entry_sl_dist = None
                self._half_taken = False
                return

            trade = self.trades[-1] if self.trades else None
            if trade is None:
                return

            # ---- NEW: 50% scale-out at +1R ----
            # Threshold is locked in at entry (alpha * ATR_entry == initial SL distance).
            # We require finite entry_price/sl_dist (set on the entry path below).
            if (
                not self._half_taken
                and self._entry_price is not None
                and self._entry_sl_dist is not None
            ):
                if trade.is_long:
                    tp1_level = self._entry_price + self._entry_sl_dist
                    # Use bar HIGH for intra-bar hit detection (matches trail logic
                    # which uses bar LOW for stop hits).
                    if high_v >= tp1_level:
                        trade.close(portion=0.5)
                        self._half_taken = True
                else:
                    tp1_level = self._entry_price - self._entry_sl_dist
                    if low_v <= tp1_level:
                        trade.close(portion=0.5)
                        self._half_taken = True

            # ---- Trailing stop (unchanged from v2) ----
            # Re-read trade in case half-out consumed it (defensive — partial
            # close does not zero out self.trades; the remaining position
            # stays as the same Trade with updated Size).
            trade = self.trades[-1] if self.trades else None
            if trade is None:
                # Whole position closed via half-out edge case (shouldn't happen
                # at portion=0.5 unless size was 1 and round(0.5*1)=1).  Reset.
                self._trail_level = None
                self._entry_bar = None
                self._entry_price = None
                self._entry_sl_dist = None
                self._half_taken = False
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
                    self._entry_price = None
                    self._entry_sl_dist = None
                    self._half_taken = False
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
                    self._entry_price = None
                    self._entry_sl_dist = None
                    self._half_taken = False
            return

        # --- Entry: only at H6 close boundaries (unchanged from v2). ---
        if not self._is_h6_close_bar(ts):
            return
        if self._last_h6_close_seen == ts:
            return
        self._last_h6_close_seen = ts

        if not np.isfinite(mom_v) or not np.isfinite(atr_v) or atr_v <= 0:
            return

        # Prefix guard.
        if self.trade_start_ns > 0 and ts.value < self.trade_start_ns:
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
            self._entry_price = close_v
            self._entry_sl_dist = sl_dist
            self._half_taken = False
        elif self.allow_shorts and mom_v < -theta:
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._trail_level = close_v + sl_dist
            self._entry_price = close_v
            self._entry_sl_dist = sl_dist
            self._half_taken = False
