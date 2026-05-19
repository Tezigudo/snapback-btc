"""
Tuned variants of multifactor-v1 — testbeds for trend-break behavior.

Investigation of low win rate (24.8% on v1) revealed that 73% of exits are
`adverse-trend-cross` events: price crosses EMA(200) against the position
and the position is closed immediately. Those exits average -0.43% each and
drag the strategy from ~+11% theoretical to ~+0.3% realized.

Two specific improvements to test (independently and combined):

  A) Debounce: require N consecutive bars on the wrong side of EMA before
     closing. With N=1 we reproduce the original behavior; with N=2..4 we
     filter single-bar whipsaws.

  C) Loss-exit floor: only close on trend-cross if current trade PnL > floor.
     This says "take small profits on trend flip, but don't capitulate on
     small losses via trend-flip — let SL/TP handle those."

Combined: AC variant uses both gates.

Both knobs are CLASS-LEVEL attributes (NOT in StrategyParams), so
_apply_params_to_class won't clobber them between runs.
"""

from __future__ import annotations

import numpy as np

from strategy.signals_multifactor import DayTradeMultiFactorBTC


class _TunedBase(DayTradeMultiFactorBTC):
    """Shared base for tuned trend-exit variants."""

    # A) consecutive-bar debounce on trend-cross exit
    trend_break_lookback: int = 1

    # C) only allow trend-cross exit when current PnL > floor (as fraction)
    trend_break_min_pnl_pct: float = -1.0  # -1.0 = no floor (always allow)

    def _bars_against_trend(self, side: str, i: int, n: int) -> bool:
        """True iff the last `n` bars all closed on the wrong side of EMA."""
        if n <= 0:
            return True
        if i - (n - 1) < 0:
            return False
        for k in range(n):
            c = self.data.Close[-(k + 1)]
            e = self._trend_ema[i - k]
            if not (np.isfinite(c) and np.isfinite(e)):
                return False
            if side == "long" and not (c < e):
                return False
            if side == "short" and not (c > e):
                return False
        return True

    def _trend_exit_allowed(self) -> bool:
        """Check the loss-floor gate against current trade PnL."""
        if self.trend_break_min_pnl_pct <= -1.0:
            return True  # floor disabled
        try:
            pl_pct = float(self.position.pl_pct)  # signed fraction
        except Exception:
            return True
        return pl_pct > self.trend_break_min_pnl_pct

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        if self.position:
            # time stop (unchanged from parent)
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold_bars:
                self.position.close()
                self._entry_bar = None
                return

            # tuned trend-cross exit
            if self.require_trend and self._trend_exit_allowed():
                t = self._trend_ema[i]
                if np.isfinite(t):
                    n = max(1, int(self.trend_break_lookback))
                    if self.position.is_long and close_v < t:
                        if self._bars_against_trend("long", i, n):
                            self.position.close()
                            self._entry_bar = None
                            return
                    if self.position.is_short and close_v > t:
                        if self._bars_against_trend("short", i, n):
                            self.position.close()
                            self._entry_bar = None
                            return
            return  # in-position → no entry attempts

        # No position → delegate entry logic to parent
        super().next()


# --- A) debounce variants ----------------------------------------------------
class V1Debounce2(_TunedBase):
    trend_break_lookback: int = 2


class V1Debounce3(_TunedBase):
    trend_break_lookback: int = 3


class V1Debounce4(_TunedBase):
    trend_break_lookback: int = 4


# --- C) loss-floor variants --------------------------------------------------
class V1Floor005(_TunedBase):
    """Loss floor at -0.5% (don't capitulate on small losses)."""
    trend_break_min_pnl_pct: float = -0.005


class V1Floor010(_TunedBase):
    """Loss floor at -1.0%."""
    trend_break_min_pnl_pct: float = -0.010


# --- Combined A+C ------------------------------------------------------------
class V1Deluxe(_TunedBase):
    """N=2 debounce + -0.5% loss floor."""
    trend_break_lookback: int = 2
    trend_break_min_pnl_pct: float = -0.005
