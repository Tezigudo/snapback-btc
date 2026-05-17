"""
FundingMomentum — hypothesis opposite to carry.

Carry trades AGAINST funding: when longs are paying (funding > 0), go
SHORT. Bet that the squeezed-side eventually capitulates and price reverts.

Funding momentum trades WITH funding *direction-change*: when funding is
rising (more positive or less negative), it means longs are getting more
aggressive — GO LONG. When funding is falling, shorts are getting more
aggressive — GO SHORT.

Why this might work where carry doesn't:
  - Carry assumes mean-reversion at the funding-cycle scale (8h). BTC is
    in bull markets often; mean-reversion doesn't dominate.
  - Funding momentum assumes that the side paying *more aggressively* is
    correct about direction — leveraged crowd flow as a leading signal.
  - Bonus: when we're WITH the crowd, we pay funding instead of collecting
    it. So this is a price-edge strategy, not a yield strategy. Funding
    is a COST, not income.

Entry:
  - Compute funding momentum = funding[now] - funding[N_bars_ago]
  - If momentum > +threshold (longs paying MORE aggressively): LONG
  - If momentum < -threshold (shorts paying MORE aggressively): SHORT
  - Optional trend confirmation: only long if close > trend_ema, etc.

Exit:
  - SL at fixed % adverse
  - TP at fixed % favourable (this is a price-edge play)
  - Time stop (don't hold forever waiting for momentum to play out)
  - Funding momentum reversal (if direction flips, exit)

Position sizing: standard risk-per-trade with leverage cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


class FundingMomentumBTC(Strategy):
    # --- entry params ---
    momentum_lookback_bars = 96       # 24h at 15m — funding momentum window
    momentum_threshold = 0.0002       # funding must change by this much
    require_trend_align = False       # if True, only enter aligned with EMA

    # --- exit params ---
    sl_pct = 0.01                     # 1% adverse stop
    tp_pct = 0.02                     # 2% favourable take-profit (2:1 R:R)
    time_stop_bars = 96               # close if held this long

    # --- trend confirmation (optional) ---
    trend_ema_period = 0              # 0 = trend filter OFF

    # --- sizing ---
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True

    def init(self) -> None:
        # Funding-momentum series, sampled at entry-TF cadence.
        funding = pd.Series(self.data.Funding)
        self._mom = (funding - funding.shift(self.momentum_lookback_bars)).values
        if self.trend_ema_period > 0:
            close = pd.Series(self.data.Close)
            self._trend = close.ewm(span=self.trend_ema_period, adjust=False).mean().values
        else:
            self._trend = None
        self._entry_bar: int | None = None

    def _position_units(self, price: float, sl_distance: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _trend_allows(self, want_long: bool) -> bool:
        if not self.require_trend_align or self._trend is None:
            return True
        trend = self._trend[len(self.data) - 1]
        if not np.isfinite(trend) or trend <= 0:
            return True   # fail-open if no trend yet
        close = self.data.Close[-1]
        return (close > trend) if want_long else (close < trend)

    def next(self) -> None:
        i = len(self.data) - 1
        if i < self.momentum_lookback_bars + 1:
            return

        funding_v = self.data.Funding[-1]
        mom = self._mom[i]
        close_v = self.data.Close[-1]
        if not np.isfinite(funding_v) or not np.isfinite(mom) or not np.isfinite(close_v):
            return

        if self.position:
            # Time stop
            if self._entry_bar is not None and (i - self._entry_bar) >= self.time_stop_bars:
                self.position.close()
                self._entry_bar = None
                return
            # Momentum reversal exit
            if self.position.is_long and mom < -self.momentum_threshold / 2:
                self.position.close()
                self._entry_bar = None
                return
            if self.position.is_short and mom > self.momentum_threshold / 2:
                self.position.close()
                self._entry_bar = None
                return
            return

        sl_dist = self.sl_pct * close_v
        tp_dist = self.tp_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if mom > self.momentum_threshold and self._trend_allows(True):
            sl = close_v - sl_dist
            tp = close_v + tp_dist
            self.buy(size=units, sl=sl, tp=tp)
            self._entry_bar = i
        elif mom < -self.momentum_threshold and self.allow_shorts and self._trend_allows(False):
            sl = close_v + sl_dist
            tp = close_v - tp_dist
            self.sell(size=units, sl=sl, tp=tp)
            self._entry_bar = i
