"""
AdaptiveTrendV2_funding_skip — single-feature improvement on AdaptiveTrendV2.

Feature: SKIP entries within 30 minutes (2 x 15m bars) of a Binance perp
funding event time (00:00, 08:00, 16:00 UTC) IF the most recent observed
funding rate (strictly BEFORE the current bar timestamp) has absolute value
greater than `funding_skip_threshold` (default 0.0005 = 5 bps per 8h).

Why
---
v1 + v2 hold positions across funding boundaries by design (MOM-driven entry,
ATR trailing-stop exit).  We've already accounted for funding as a post-process
cost on closed trades.  But OPENING into a hostile funding regime stacks two
known biases:
  - the bar after a funding event tends to mean-revert briefly,
  - the position will eat the SIGNED funding cost on the very next funding tick.
"Hostile" here means |rate| > 5 bps — Binance's own kink-point in the funding
formula where the premium index dominates the index price.  Empirically these
spikes are the cases where post-funding price gaps eat 30-50 bps before a
trend-follow entry can even start trailing.  Skipping entries in the 30-min
neighbourhood removes those low-conviction openings without touching exits.

What's NOT changed
------------------
- Position management (trailing stop, max-hold, exit logic) is untouched —
  we only veto NEW entries during the skip window.
- The Algorithm-2 monthly re-opt still runs as in v2 base.
- alpha (fixed 2.0), sizing, leverage, prefix guard — all inherited verbatim.

Wiring contract
---------------
The runner must set the strategy-class attribute `funding_series` to a
pandas Series indexed by tz-naive UTC timestamps with float `funding_rate`
values BEFORE constructing the Backtest.  An empty series disables the
filter (graceful no-op).  Without the attribute the parent class behaviour
applies.

Lookahead safety
----------------
We use `searchsorted(side='right')` on the funding-event index to find the
LAST funding rate observed STRICTLY BEFORE the current 15m bar timestamp.
The current bar's open is `ts` (backtesting.py convention: data.index[-1]
is the just-CLOSED bar's start).  Funding events at the same timestamp as
the current bar are NOT used — we only consult funding ticks already in
the public tape.

Authority: research-only.  Not wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2


# ---------------------------------------------------------------------------
# Helper: build the per-15m-bar "should skip" mask once at init().
# ---------------------------------------------------------------------------


def _build_skip_mask(
    bar_index: pd.DatetimeIndex,
    funding: pd.Series,
    threshold: float,
    window_minutes: int = 30,
) -> np.ndarray:
    """Return a bool array of len(bar_index): True = skip entry at this bar.

    A bar is skipped iff:
      (a) it lies within `window_minutes` of any funding event time
          (00:00, 08:00, 16:00 UTC), AND
      (b) the LAST observed funding rate strictly BEFORE the bar's timestamp
          has absolute value > `threshold`.

    Pure-numpy, vectorised — built once at init() so next() is O(1).
    """
    n = len(bar_index)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Condition (a): proximity to funding event clocks.
    # Bars at 23:30, 23:45, 00:00, 00:15, 00:30 are near 00 UTC funding.
    # Similarly for 08 UTC and 16 UTC.  At 15m granularity, "within 30 min"
    # = 2 bars before, the event bar itself, 2 bars after = 5 bars.
    hours = bar_index.hour.values
    mins = bar_index.minute.values
    minutes_in_day = hours * 60 + mins
    event_minutes = np.array([0, 8 * 60, 16 * 60], dtype=int)
    # Distance to nearest funding event, accounting for day wrap (00 UTC is
    # also reachable from late prior-day bars).
    # Build (n, 3) distance matrix then take min.
    diff_to_event = np.abs(minutes_in_day[:, None] - event_minutes[None, :])
    # Also allow wrap: 23:45 is 15 min before next 00 UTC.
    diff_wrap = np.abs((minutes_in_day[:, None] + 24 * 60) - event_minutes[None, :])
    diff = np.minimum(diff_to_event, diff_wrap)
    near_event = diff.min(axis=1) <= window_minutes

    if funding is None or len(funding) == 0:
        return np.zeros(n, dtype=bool)

    # Condition (b): magnitude of last-observed funding rate.
    # Drop tz on funding index to match bar index.
    f = funding.copy()
    if isinstance(f.index, pd.DatetimeIndex) and f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    f = f.sort_index()
    f_ts = f.index.to_numpy()  # datetime64[ns]
    f_vals = np.abs(f.to_numpy().astype(float))

    bar_ts = bar_index.to_numpy()
    # searchsorted(side='left') returns the index where bar_ts would be inserted
    # to keep f_ts sorted — so f_ts[idx-1] is the last funding event STRICTLY
    # before bar_ts (side='left' on equal puts ties at idx, which we then -1
    # away from).  This is the lookahead-safe choice.
    idx = np.searchsorted(f_ts, bar_ts, side="left") - 1
    valid = idx >= 0
    last_abs = np.full(n, 0.0, dtype=float)
    last_abs[valid] = f_vals[idx[valid]]

    hostile = last_abs > threshold

    return near_event & hostile


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class AdaptiveTrendV2_funding_skip(AdaptiveTrendV2):
    """v2 + skip-entry-on-hostile-funding filter.  See module docstring."""

    # Filter knobs (overridable via bt.run(**config)).
    funding_skip_threshold: float = 0.0005   # absolute funding rate
    funding_skip_window_min: int = 30        # minutes around event time

    # The runner sets this CLASS attribute before bt.run().  We don't try
    # to pass a pandas Series through bt.run(**kwargs) because backtesting.py
    # setattr's kwargs on the class — works fine but conceptually awkward.
    # Keeping it as a class-attribute "injection" is the standard pattern.
    funding_series: pd.Series | None = None

    def init(self) -> None:
        super().init()
        # Build skip mask aligned to the 15m bar index.
        bar_idx = self._index
        if not isinstance(bar_idx, pd.DatetimeIndex):
            bar_idx = pd.DatetimeIndex(bar_idx)
        funding = type(self).funding_series  # read from class, not instance
        self._skip_mask = _build_skip_mask(
            bar_index=bar_idx,
            funding=funding,
            threshold=float(self.funding_skip_threshold),
            window_minutes=int(self.funding_skip_window_min),
        )
        # Diagnostic counters.
        self._n_skipped_by_funding = 0

    def next(self) -> None:
        # Parent's next() handles refit + position management + entry.  We
        # short-circuit ONLY when we're about to attempt an entry AND the
        # filter says skip.  Easiest insertion point: monkey-patch our own
        # next() to mirror v2's flow with the extra veto check.
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]

        # Algorithm 2 refit (unchanged).
        self._maybe_refit(ts)

        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management (unchanged — copied verbatim from v2). ---
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

        # --- Entry gate (unchanged H6-close requirement). ---
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

        # *** THE ONE NEW LINE ***
        if self._skip_mask[i]:
            self._n_skipped_by_funding += 1
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
