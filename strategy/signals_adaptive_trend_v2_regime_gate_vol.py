"""
AdaptiveTrendV2 + regime_gate_vol — entries blocked in compressed-vol regimes.

Improvement under test
----------------------
Only allow LIVE entries when realized vol > 50th percentile of trailing
60-day vol distribution.  Skip compressed-vol regimes.

Realized vol estimator (causal, documented choice)
--------------------------------------------------
We need a concrete definition of "realized vol" — the task spec leaves it
open.  Choice: rolling std of H6 log returns over `vol_window_h6=14` H6 bars
(mirrors `atr_period_h6=14` granularity).  Computed on the FULL H6 frame
(same frame the strategy already uses for MOM/ATR), then **shifted by 1**
so the value at any H6 close is the value of the JUST-CLOSED bar — same
causal pattern as `_rebuild_live_signal`.

The trailing 60-day "vol distribution" is the rolling MEDIAN (50th pct) of
that vol series over the last 60 days.  At H6 granularity that's
60 * 4 = 240 H6 bars.  Median is computed on the shifted series, so it
sees only past H6 closes — no lookahead.

Gate semantics
--------------
At an H6-close entry boundary in live next(): if vol[t] <= median60d_vol[t],
return WITHOUT entering.  Position management (trailing stop) and monthly
re-opt are NOT gated — they run every bar as before.  The inner H6 fit
simulator (`_simulate_h6_fit`) is also NOT gated; the spec describes a
LIVE entry rule, not a re-opt selection rule, and gating the fit sim would
conflate the L/theta search with a regime filter.

Prefix usage
------------
The runner already prepends `fit_window_months=6` of history.  60 trailing
days fits comfortably within that prefix, so the gate is warm by the first
OOS H6 close — no additional history needed.

Authority: research-only.  Not wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2


class AdaptiveTrendV2_regime_gate_vol(AdaptiveTrendV2):
    """V2 with a vol-regime entry gate."""

    # --- Gate config ---
    vol_window_h6: int = 14         # H6 bars for realized-vol estimate
    vol_lookback_days: int = 60     # trailing window for the percentile distribution
    vol_quantile: float = 0.50      # 50th percentile (median) threshold

    # ------------------------------------------------------------------ init

    def init(self) -> None:
        super().init()
        self._build_vol_gate()

    # ------------------------------------------------------------------ gate build

    def _build_vol_gate(self) -> None:
        """Precompute the gate boolean array aligned to the 15m index.

        Steps (all causal):
          1. realized vol = rolling std of H6 log returns over `vol_window_h6`
          2. shift(1) — value at H6 close is just-closed-bar's vol
          3. rolling-median over `vol_lookback_days * 4` H6 bars
          4. boolean: vol > rolling-median
          5. reindex / ffill onto the 15m index
        """
        h6 = self._h6
        h6_close = h6["Close"]
        # Log returns at H6 granularity.
        log_ret = np.log(h6_close / h6_close.shift(1))
        # Realized vol: rolling std of log returns.
        rvol = log_ret.rolling(self.vol_window_h6, min_periods=self.vol_window_h6).std()
        # Trailing 60-day median of that vol series.
        window_bars = int(self.vol_lookback_days * 4)  # 4 H6 bars/day
        rvol_median = rvol.rolling(window_bars, min_periods=window_bars).quantile(
            self.vol_quantile
        )
        # Causal shift — value at H6 close ts is for the just-closed bar.
        gate_h6 = (rvol > rvol_median).astype(float)  # 1.0 = pass, 0.0 = block
        # Keep NaN where rvol or median is NaN (warmup) — those bars should not pass.
        warmup_mask = rvol.isna() | rvol_median.isna()
        gate_h6 = gate_h6.where(~warmup_mask, other=0.0)
        gate_h6 = gate_h6.shift(1)
        # Reindex onto 15m index (ffill — gate persists between H6 closes).
        aligned = gate_h6.reindex(self._index, method="ffill")
        # Default to blocked (0) anywhere still NaN at the head.
        self._vol_gate = aligned.fillna(0.0).values

    # ------------------------------------------------------------------ loop

    def next(self) -> None:
        # If we're not flat at an H6-close, parent's logic is unchanged
        # (trailing stop, refit, etc.).  We only suppress NEW entries.
        i = len(self.data) - 1
        ts = self._index[i]

        # Refit must still run (keeps params warm even when gated).
        # Parent's next() runs _maybe_refit unconditionally — let it.
        if self.position:
            super().next()
            return

        # Flat, NOT at H6 close → no entry possible anyway, delegate.
        if not self._is_h6_close_bar(ts):
            super().next()
            return

        # Flat AT an H6 close.  Check the gate.
        gate_open = bool(self._vol_gate[i] > 0.5)
        if gate_open:
            super().next()
            return

        # Gate closed: suppress entry, but still let _maybe_refit run.
        self._maybe_refit(ts)
        # Do not call super().next() — that would attempt entry.
        return
