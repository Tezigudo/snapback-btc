"""
AdaptiveTrendV1 + half_out_at_1R improvement (research-only).

ONE feature added on top of AdaptiveTrendV1 (Algorithm 1 base):
    When an open position's intra-bar range reaches +1R (where 1R is the
    initial SL distance = alpha * ATR_entry), scale out 50% of the position.
    The remaining 50% rides with the original ATR trailing stop.

Rationale
---------
Trend systems have right-skewed per-trade distributions: a few large winners
fund a steady stream of small losers/breakeven trades.  Many trades show
+1R unrealised profit before the trail later eats them.  Locking in half at
+1R secures a guaranteed-positive partial on those trades, while the runner
half keeps tail exposure for the genuine trend wins.  Net effect on
per-trade Sharpe is empirical.

Design notes
------------
- Subclass, not modification.  Base behaviour unchanged.
- Opt-in via the `half_out_at_1r` class flag (default False).  When False,
  next() falls through to identical base behaviour — strict backward
  compatibility.
- The +1R level is anchored at ENTRY (entry_price ± alpha*ATR_entry).
  Locking the threshold at entry — not recomputing it as ATR drifts —
  matches the paper's "initial SL = alpha*ATR_entry" interpretation
  and gives a stable, well-defined R-multiple.
- Intra-bar hit detection uses bar HIGH for longs / LOW for shorts.
  backtesting.py schedules trade.close(portion=0.5) at the NEXT bar's
  open, so no perfect-fill cheating (consistent with how the trail's
  stop hits flow through the broker).
- Trailing stop + max-hold + entry logic are unchanged.

PRICE_SCALE/sizing safety
-------------------------
backtesting.py's Trade.close(portion=0.5) computes
  size = max(1, int(round(abs(self.size) * 0.5)))
With PRICE_SCALE=0.001 in the postfrac harness, the V1 base typically
opens 1000+ units per entry, so the half-close cleanly splits to ~500/500
without collapsing to a full close.  If a tiny entry of size=1 ever occurs
the half-close would close the entire trade — we defensively reset state
in that path.

Authority: research-only.  NOT wired to bot.py.
"""

from __future__ import annotations

import numpy as np

from strategy.signals_adaptive_trend import AdaptiveTrendV1


class AdaptiveTrendV1_half_out_1r(AdaptiveTrendV1):
    """V1 with 50% scale-out at +1R, runner trails the rest."""

    # Opt-in flag.  Default OFF so this class is a no-op subclass unless
    # the runner explicitly enables it.  Backward-compat per task spec.
    half_out_at_1r: bool = False

    # ------------------------------------------------------------------ init

    def init(self) -> None:  # type: ignore[override]
        super().init()
        # Per-trade half-out bookkeeping.  Reset on every fresh entry.
        self._entry_price: float | None = None
        self._entry_sl_dist: float | None = None
        self._half_taken: bool = False

    # ------------------------------------------------------------------ loop

    def next(self) -> None:  # type: ignore[override]
        # Flag OFF -> identical to base.
        if not self.half_out_at_1r:
            super().next()
            return

        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        high_v = float(self.data.High[-1])
        low_v = float(self.data.Low[-1])
        ts = self._index[i]
        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management ---
        if self.position:
            if not np.isfinite(atr_v) or atr_v <= 0:
                return

            # Hard max-hold belt (unchanged from base).
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
            if (
                not self._half_taken
                and self._entry_price is not None
                and self._entry_sl_dist is not None
            ):
                if trade.is_long:
                    tp1_level = self._entry_price + self._entry_sl_dist
                    if high_v >= tp1_level:
                        trade.close(portion=0.5)
                        self._half_taken = True
                else:
                    tp1_level = self._entry_price - self._entry_sl_dist
                    if low_v <= tp1_level:
                        trade.close(portion=0.5)
                        self._half_taken = True

            # Re-read trade in case half-out edge-case consumed it (size=1
            # collapses to a full close).  Should not happen at PRICE_SCALE
            # sizing but defensive.
            trade = self.trades[-1] if self.trades else None
            if trade is None:
                self._trail_level = None
                self._entry_bar = None
                self._entry_price = None
                self._entry_sl_dist = None
                self._half_taken = False
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
            self._entry_price = close_v
            self._entry_sl_dist = sl_dist
            self._half_taken = False
        elif self.allow_shorts and mom_v < -self.theta_entry:
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._trail_level = close_v + sl_dist
            self._entry_price = close_v
            self._entry_sl_dist = sl_dist
            self._half_taken = False
