"""
AdaptiveTrendV1_funding_skip — single-feature improvement on AdaptiveTrendV1.

Mirrors strategy/signals_adaptive_trend_v2_funding_skip.py verbatim, but the
base class is the BARE V1 signal (no Algorithm-2 monthly re-opt, no
trade_start_ns prefix guard, no _maybe_refit). The veto check is inserted at
the same logical point — AFTER the H6-close gate + finite checks, BEFORE
computing position units / placing the order.

Feature
-------
SKIP entries within 30 minutes (2 x 15m bars) of a Binance perp funding event
time (00:00, 08:00, 16:00 UTC) IF the most recent observed funding rate
(strictly BEFORE the current bar timestamp) has absolute value greater than
`funding_skip_threshold` (default 0.0005 = 5 bps per 8h).

Wiring contract
---------------
The runner must set the strategy-class attribute `funding_series` to a pandas
Series indexed by tz-naive UTC timestamps with float `funding_rate` values
BEFORE constructing the Backtest. An empty/None series disables the filter
(graceful no-op).

Authority: research-only.  Not wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals_adaptive_trend import AdaptiveTrendV1


# ---------------------------------------------------------------------------
# Helper: per-15m-bar skip mask (verbatim from v2_funding_skip).
# ---------------------------------------------------------------------------


def _build_skip_mask(
    bar_index: pd.DatetimeIndex,
    funding: pd.Series,
    threshold: float,
    window_minutes: int = 30,
) -> np.ndarray:
    n = len(bar_index)
    if n == 0:
        return np.zeros(0, dtype=bool)

    hours = bar_index.hour.values
    mins = bar_index.minute.values
    minutes_in_day = hours * 60 + mins
    event_minutes = np.array([0, 8 * 60, 16 * 60], dtype=int)
    diff_to_event = np.abs(minutes_in_day[:, None] - event_minutes[None, :])
    diff_wrap = np.abs((minutes_in_day[:, None] + 24 * 60) - event_minutes[None, :])
    diff = np.minimum(diff_to_event, diff_wrap)
    near_event = diff.min(axis=1) <= window_minutes

    if funding is None or len(funding) == 0:
        return np.zeros(n, dtype=bool)

    f = funding.copy()
    if isinstance(f.index, pd.DatetimeIndex) and f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    f = f.sort_index()
    f_ts = f.index.to_numpy()
    f_vals = np.abs(f.to_numpy().astype(float))

    bar_ts = bar_index.to_numpy()
    idx = np.searchsorted(f_ts, bar_ts, side="left") - 1
    valid = idx >= 0
    last_abs = np.full(n, 0.0, dtype=float)
    last_abs[valid] = f_vals[idx[valid]]

    hostile = last_abs > threshold
    return near_event & hostile


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class AdaptiveTrendV1_funding_skip(AdaptiveTrendV1):
    """V1 + skip-entry-on-hostile-funding filter. See module docstring."""

    # Filter knobs (overridable via bt.run(**config)).
    funding_skip_threshold: float = 0.0005   # absolute funding rate
    funding_skip_window_min: int = 30        # minutes around event time

    # CLASS-level injection slot for the funding series. Runner sets this
    # before bt.run().
    funding_series: pd.Series | None = None

    # ------------------------------------------------------------------ init
    def init(self) -> None:
        super().init()
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
        self._n_skipped_by_funding = 0

    # ------------------------------------------------------------------ loop
    def next(self) -> None:
        # Verbatim copy of V1.next() with the ONE NEW LINE for the skip veto.
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]
        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management (verbatim from V1). ---
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

        # --- Entry: only at H6 close boundaries. ---
        if not self._is_h6_close_bar(ts):
            return
        if self._last_h6_close_seen == ts:
            return
        self._last_h6_close_seen = ts

        if not np.isfinite(mom_v) or not np.isfinite(atr_v) or atr_v <= 0:
            return

        # *** THE ONE NEW LINE *** — funding-skip veto, before sizing/entry.
        if self._skip_mask[i]:
            self._n_skipped_by_funding += 1
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
