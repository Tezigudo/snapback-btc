"""
AdaptiveTrendV2 + session_volume_filter improvement.

ONE feature added on top of the AdaptiveTrendV2 (Algorithm 2) base:
    Only allow ENTRIES when the H6-close timestamp falls inside the
    high-volume session window [12:00, 22:00) UTC (London + NY overlap).
    Asia + late-NY closes (00:00, 06:00) are skipped on the entry path.
    Exits / trailing stop / monthly re-opt are unchanged.

Rationale
---------
The base strategy enters at H6 close boundaries: 00, 06, 12, 18 UTC.
- 00 UTC = early Asia open, thin book, wider spreads
- 06 UTC = pre-London, mid-Asia, lower participation
- 12 UTC = London open + pre-NY (HIGH liquidity)
- 18 UTC = NY afternoon (HIGH liquidity)
Filtering on the 12 / 18 UTC window keeps the two H6 closes that sit
inside London+NY active hours and drops the two that don't — cutting
~50% of candidate entries but skewing them to lower-slippage regimes.

Per the brief: "Avoid Asia/late-NY low-liquidity entries that bleed via
slippage."  We don't model slippage explicitly in the harness, but the
realised funding + commission cost differential should still favour the
high-volume window because trades fired into thin liquidity tend to be
weaker MOM signals that fail more often.

Design notes
------------
- Pure entry-time gate. No new indicators, no extra series, no I/O.
- The strategy already only considers entries at `_is_h6_close_bar(ts)`
  (ts.hour % 6 == 0, ts.minute == 0).  We add a second predicate:
  ts.hour in {12, 18}.
- Exit path, trailing-stop ratchet, and monthly re-opt are UNCHANGED.
  The gate must not block exits — that would carry losers through Asia
  and amplify drawdowns.
- The gate fires AFTER the prefix guard and AFTER mom/atr validity
  checks, mirroring the regime_gate_adx pattern.
- We intentionally keep [12:00, 22:00) UTC framing in the docstring even
  though only 12 and 18 fall on H6 closes inside that band; this is the
  same semantic, just expressed in the H6-close grid.

Authority: research-only. Not wired to bot.py.
"""

from __future__ import annotations

import numpy as np

from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2


# H6 close hours that fall inside the high-volume session window
# [12:00, 22:00) UTC = London + NY overlap.
# Of {0, 6, 12, 18}, only {12, 18} sit inside that band.
_HIGH_VOL_H6_HOURS: frozenset[int] = frozenset({12, 18})


class AdaptiveTrendV2_session_volume_filter(AdaptiveTrendV2):
    """AdaptiveTrendV2 with entry restricted to high-volume sessions."""

    # Configurable for diagnostics; default is {12, 18} UTC.
    session_entry_hours: tuple = (12, 18)

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

        # --- session_volume_filter FEATURE: only enter at high-volume H6 closes. ---
        # Of {0, 6, 12, 18}, we keep {12, 18} (inside [12:00, 22:00) UTC).
        if ts.hour not in self.session_entry_hours:
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
