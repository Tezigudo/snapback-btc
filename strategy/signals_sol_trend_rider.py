"""
SolTrendRider — native-4h SOL long-only Supertrend rider. No take-profit.

Why this exists
---------------
`tools/sol_leg_return_search.py` ran a 15-candidate, 956-combo walk-forward on
SOL/USDT:USDT 4h ranked by return. The winning geometry was the `st-donchexit`
variant restricted to longs — but investigation showed *why* it won, and it is
not the reason its name suggests:

  `SupertrendDonchExit` reads `self.data.STDonchLower[-1]` unshifted. Because
  `donchian_channel()` includes the current bar (its docstring says the caller
  must shift by 1), the long exit test `close <= rolling_min(low, N)` can only
  be true when the close equals the window's lowest low — it fired **0 times
  in 9,457 SOL 4h bars**, at every period from 3 to 55. The Donchian exit is
  dead code, and `st_tp_atr` is deliberately unused in that class.

So `st-donchexit` long-only is really: *enter on Supertrend flip, stop at
1×ATR, no take-profit, exit only when the trend flips back*. That accidental
configuration beat every deliberately-parameterised variant on SOL, because
removing the take-profit is what lets a SOL trend run.

This module states that geometry explicitly instead of relying on a bug, and
adds the two exits that were *meant* to be tested, correctly shifted:

  * `sol_donch_exit_period > 0` — exit long when close < the N-bar low formed
    by bars [i-N, i-1] (shifted at attach time, so `next()` cannot peek).
  * `sol_trail_atr > 0` — chandelier trail off the running high since entry.

Both default OFF, so the default class reproduces the walk-forward winner.

Entry (long): STDir flips -1 → +1 on the last closed bar
Stop:         entry close - sl_atr × ATR(atr_period)   (hard backstop)
Exit:         STDir flips +1 → -1  (plus optional donch / trail / time stop)
Take-profit:  none, by design

Shorts are available (`allow_shorts=True`) but default OFF: the search found
SOL short-side trend-following gave back most of the long-side edge, and wide
short targets are unreachable on cheap/volatile early SOL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr, donchian_channel, supertrend


def attach_sol_trend_rider(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
    donch_period: int = 0,
) -> pd.DataFrame:
    """Attach STLine/STDir/STAtr (+ SolDonchLow/SolDonchHigh when
    donch_period > 0) to a capitalised, tz-naive native-4h frame.

    SolDonchLow/High are **shifted by 1** here, at attach time, so that bar i
    carries the channel formed by bars [i-period, i-1]. `next()` therefore
    reads `[-1]` like every other indicator in this repo and still cannot see
    the current bar's own extreme — the mistake that made the `st-donchexit`
    exit dead code.
    """
    out = df.copy()
    st = supertrend(out["High"], out["Low"], out["Close"],
                    period=period, multiplier=multiplier)
    out["STLine"] = st["supertrend"]
    out["STDir"] = st["direction"]
    out["STAtr"] = atr(out["High"], out["Low"], out["Close"], atr_period)
    if donch_period and donch_period > 0:
        upper, lower = donchian_channel(out["High"], out["Low"], donch_period)
        out["SolDonchHigh"] = upper.shift(1)
        out["SolDonchLow"] = lower.shift(1)
    else:
        out["SolDonchHigh"] = np.nan
        out["SolDonchLow"] = np.nan
    return out


class SolTrendRider(Strategy):
    """SOL 4h long-only Supertrend rider — ATR stop, no TP, exit on flip.

    Sizing mirrors donchian-v3 / rider-v1 / supertrend: risk a fixed % of
    equity per trade against the stop distance, capped by leverage.
    """

    st_period: int = 10
    st_multiplier: float = 3.0
    st_atr_period: int = 14
    st_sl_atr: float = 1.0
    sol_risk_per_trade_pct: float = 1.25
    allow_shorts: bool = False

    # Optional exits — both OFF by default (0 disables).
    sol_donch_exit_period: int = 0
    sol_trail_atr: float = 0.0
    sol_time_stop_bars: int = 0

    # Set by run_backtest via STRATEGIES[name].leverage = eff_leverage.
    leverage: int = 3

    def init(self) -> None:
        self._entry_bar: int | None = None
        self._trail_extreme: float | None = None

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.sol_risk_per_trade_pct / 100.0)
        target = risk_amount / sl_distance
        cap = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target, cap)), 0)

    def _should_exit_long(self, close_v: float, high_v: float, atr_v: float,
                          direction: float) -> bool:
        if direction == -1.0:
            return True
        if self.sol_donch_exit_period > 0:
            donch_low = self.data.SolDonchLow[-1]
            if np.isfinite(donch_low) and close_v < donch_low:
                return True
        if self.sol_trail_atr > 0:
            self._trail_extreme = max(self._trail_extreme or high_v, high_v)
            if close_v < self._trail_extreme - self.sol_trail_atr * atr_v:
                return True
        if self.sol_time_stop_bars > 0 and self._entry_bar is not None:
            if len(self.data) - self._entry_bar >= self.sol_time_stop_bars:
                return True
        return False

    def _should_exit_short(self, close_v: float, low_v: float, atr_v: float,
                           direction: float) -> bool:
        if direction == 1.0:
            return True
        if self.sol_donch_exit_period > 0:
            donch_high = self.data.SolDonchHigh[-1]
            if np.isfinite(donch_high) and close_v > donch_high:
                return True
        if self.sol_trail_atr > 0:
            self._trail_extreme = min(self._trail_extreme or low_v, low_v)
            if close_v > self._trail_extreme + self.sol_trail_atr * atr_v:
                return True
        if self.sol_time_stop_bars > 0 and self._entry_bar is not None:
            if len(self.data) - self._entry_bar >= self.sol_time_stop_bars:
                return True
        return False

    def next(self) -> None:
        close_v = self.data.Close[-1]
        direction = self.data.STDir[-1]
        atr_v = self.data.STAtr[-1]

        if any(v is None or not np.isfinite(v) for v in (direction, atr_v)):
            return
        if len(self.data) < 2:
            return
        prev_direction = self.data.STDir[-2]
        if prev_direction is None or not np.isfinite(prev_direction):
            return

        if self.position:
            if self.position.is_long:
                if self._should_exit_long(close_v, self.data.High[-1], atr_v, direction):
                    self.position.close()
                    self._entry_bar = None
                    self._trail_extreme = None
            elif self._should_exit_short(close_v, self.data.Low[-1], atr_v, direction):
                self.position.close()
                self._entry_bar = None
                self._trail_extreme = None
            return

        flipped_long = prev_direction == -1.0 and direction == 1.0
        flipped_short = prev_direction == 1.0 and direction == -1.0
        sl_dist = self.st_sl_atr * atr_v

        if flipped_long:
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._trail_extreme = self.data.High[-1]
        elif self.allow_shorts and flipped_short:
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._trail_extreme = self.data.Low[-1]
