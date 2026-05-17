"""
DonchianBreakoutBTC — classic turtle-style trend-following on 1h Donchian
channels, evaluated on 15m bars.

OPPOSITE hypothesis vs snapback. snapback assumes BTC mean-reverts on 15m
inside an EMA-defined trend; Donchian assumes that when price punches
through a multi-day high/low it keeps going. If both fail OOS, the
strategy family is unrelated to whether you mean-revert or trend-follow —
it's the timeframe / asset combination that has no easy edge.

Entry (long):  15m close > rolling Donchian-upper of last N 1h closes
Entry (short): 15m close < rolling Donchian-lower of last N 1h closes
Exit:          opposite direction's M-bar Donchian channel (M < N)
SL:            entry_price ± atr_sl_multiple × ATR(20, 1h)
No TP — let winners run; this is trend-following's whole pitch.

All 1h-derived columns are computed at 1h close and SHIFTED BY ONE 1h bar
before reindexing onto 15m, so a 15m bar at time T only sees a Donchian
channel computed from 1h bars strictly before T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr


def attach_donchian(
    df_15m: pd.DataFrame,
    klines_1h: pd.DataFrame,
    period_entry: int = 20,
    period_exit: int = 10,
    atr_period: int = 20,
) -> pd.DataFrame:
    """Attach DonchianUpper/Lower/ExitUpper/ExitLower/ATR_1h columns to df_15m.

    df_15m should already be capitalised + tz-naive (from prepare_strategy_data).
    """
    one_h = klines_1h.copy()
    one_h.columns = [c.capitalize() for c in one_h.columns]
    if one_h.index.tz is not None:
        one_h.index = one_h.index.tz_convert("UTC").tz_localize(None)

    upper = one_h["Close"].rolling(period_entry, min_periods=period_entry).max().shift(1)
    lower = one_h["Close"].rolling(period_entry, min_periods=period_entry).min().shift(1)
    exit_u = one_h["Close"].rolling(period_exit, min_periods=period_exit).max().shift(1)
    exit_l = one_h["Close"].rolling(period_exit, min_periods=period_exit).min().shift(1)
    atr_v = atr(one_h["High"], one_h["Low"], one_h["Close"], atr_period).shift(1)

    out = df_15m.copy()
    out["DonchianUpper"] = upper.reindex(out.index, method="ffill")
    out["DonchianLower"] = lower.reindex(out.index, method="ffill")
    out["DonchianExitUpper"] = exit_u.reindex(out.index, method="ffill")
    out["DonchianExitLower"] = exit_l.reindex(out.index, method="ffill")
    if "ATR_1h" not in out.columns:
        out["ATR_1h"] = atr_v.reindex(out.index, method="ffill")
    return out


class DonchianBreakoutBTC(Strategy):
    # Sweepable via the same params_override / class-attr injection used elsewhere.
    donchian_period_entry = 20
    donchian_period_exit = 10
    atr_sl_multiple = 2.0   # wider stop than snapback — trend-following needs room
    risk_per_trade_pct = 2.0
    leverage = 3
    allow_shorts = True

    def init(self) -> None:
        self._entry_bar: int | None = None

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def next(self) -> None:
        upper = self.data.DonchianUpper[-1]
        lower = self.data.DonchianLower[-1]
        exit_upper = self.data.DonchianExitUpper[-1]
        exit_lower = self.data.DonchianExitLower[-1]
        atr_v = self.data.ATR_1h[-1]
        close_v = self.data.Close[-1]

        if any(
            v is None or not np.isfinite(v)
            for v in (upper, lower, exit_upper, exit_lower, atr_v)
        ):
            return

        # Exit via opposite N-bar channel (Donchian/turtle classic).
        if self.position:
            if self.position.is_long and close_v < exit_lower:
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and close_v > exit_upper:
                self.position.close()
                self._entry_bar = None
            return

        sl_dist = self.atr_sl_multiple * atr_v

        if close_v > upper:  # bullish breakout
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
        elif self.allow_shorts and close_v < lower:  # bearish breakout
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)
