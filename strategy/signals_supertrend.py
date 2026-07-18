"""
SupertrendBTC — native 4h Supertrend trend-follower, long + short.

Head-to-head bake-off counterpart to donchian-v3. Same family of idea
(trend-following on the channel/band trailing the price), different
construction: Supertrend flips direction on a close-cross of its ATR band
rather than waiting for an N-bar Donchian breakout.

Entry (long):  STDir flips from -1 to +1 (close crosses above the band)
Entry (short): STDir flips from +1 to -1 (close crosses below the band)
Exit:          opposite STDir flip, OR fixed ATR take-profit, whichever first
SL:            entry close ∓ sl_atr_multiple * ATR(atr_period)
TP:            entry close ± tp_atr_multiple * ATR(atr_period)

STLine/STDir are computed by `supertrend()` (strategy/indicators.py), which
is itself causal — bar i's value uses data through bar i only. The strategy
reads `[-1]` in `next()`, i.e. the last CLOSED bar, matching every other
indicator convention in this repo (signals_donchian.py:47-51).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr, supertrend


def attach_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Attach STLine/STDir/STAtr columns to a capitalised, tz-naive DataFrame
    (native 4h bars).

    STLine = supertrend line value (the trailing band, switches sides on flip).
    STDir  = +1 (uptrend/long) / -1 (downtrend/short), NaN during warm-up.
    STAtr  = Wilder ATR(atr_period) on High/Low/Close — used for SL/TP sizing.

    Prefixed names avoid collisions with DonchianUpper/RiderDonHi/etc columns
    used by other strategies (signals_donchian.py:340-342 convention).
    """
    out = df.copy()
    st = supertrend(out["High"], out["Low"], out["Close"], period=period, multiplier=multiplier)
    out["STLine"] = st["supertrend"]
    out["STDir"] = st["direction"]
    out["STAtr"] = atr(out["High"], out["Low"], out["Close"], atr_period)
    return out


class SupertrendBTC(Strategy):
    """4h Supertrend trend-follower: long + short, ATR stop + ATR TP bracket.

    Entry:  STDir flips to +1 (long) or -1 (short) on the last closed bar.
    Stop:   sl = close_v -/+ st_sl_atr * ATR   (anchored to signal-bar close)
    Target: tp = close_v +/- st_tp_atr * ATR   (fixed bracket)
    Exit:   opposite STDir flip closes the position even if SL/TP haven't hit.

    Sizing mirrors donchian-v3/rider-v1: risk_per_trade_pct of equity /
    sl_distance, capped at leverage * equity * 0.95 / price.
    """

    st_period: int = 10
    st_multiplier: float = 3.0
    st_atr_period: int = 14
    st_sl_atr: float = 1.5
    st_tp_atr: float = 5.0
    st_risk_per_trade_pct: float = 2.0
    allow_shorts: bool = True
    # leverage IS set by run_backtest via STRATEGIES[name].leverage = eff_leverage
    leverage: int = 3

    def init(self) -> None:
        self._entry_bar: int | None = None

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.st_risk_per_trade_pct / 100.0)
        target = risk_amount / sl_distance
        cap = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target, cap)), 0)

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
            # Exit on opposite flip.
            if self.position.is_long and direction == -1.0:
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and direction == 1.0:
                self.position.close()
                self._entry_bar = None
            return

        flipped_long = prev_direction == -1.0 and direction == 1.0
        flipped_short = prev_direction == 1.0 and direction == -1.0

        sl_dist = self.st_sl_atr * atr_v

        if flipped_long:
            sl = close_v - sl_dist
            tp = close_v + self.st_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
        elif self.allow_shorts and flipped_short:
            sl = close_v + sl_dist
            tp = close_v - self.st_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
