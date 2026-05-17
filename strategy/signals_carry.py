"""
CarryHarvester — funding-rate carry trade. Not a price-prediction strategy.

Thesis: when perp funding is significantly off zero, holders of one side
are paying the other side every 8h. By taking the position OPPOSITE to the
funding sign, we collect funding payments. We accept directional price
risk to do this, capped by a fixed stop loss.

Entry:
    abs(funding_rate) > funding_threshold
    direction = -sign(funding_rate)
    SL at entry_price ± sl_pct
    size from 2% equity-at-risk over the SL distance

Exit (any of):
    - funding rate has reverted to within funding_threshold/2 of zero
    - funding rate has flipped sign (incompatible with current position)
    - SL hit (price moved against us by sl_pct)

No TP. We're collecting funding payments at every 8h event while held;
the longer we can stay in (within risk), the better the carry harvest.

backtest.py's per-trade funding accounting (funding_cost_for_trades)
already calculates per-trade funding income with signed position size, so
the "after funding" metric in the result dict reflects the actual carry
income. For this strategy, "after_funding_pct" is the headline number.
"""

from __future__ import annotations

import numpy as np
from backtesting import Strategy


class CarryHarvester(Strategy):
    funding_threshold = 0.0002        # 0.02% per 8h = ~22% annualised — enter
    funding_exit_threshold = 0.00005  # quit when funding settles below this
    sl_pct = 0.01                     # 1% stop loss caps the price-risk per cycle
    risk_per_trade_pct = 2.0
    leverage = 3
    allow_shorts = True

    def init(self) -> None:
        pass

    def _position_units(self, price: float, sl_distance: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def next(self) -> None:
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]
        if not np.isfinite(funding_v) or not np.isfinite(close_v):
            return

        if self.position:
            # Exit on funding normalisation.
            if abs(funding_v) < self.funding_exit_threshold:
                self.position.close()
                return
            # Exit on sign flip — carry direction reversed.
            if self.position.is_long and funding_v > 0:
                self.position.close()
                return
            if self.position.is_short and funding_v < 0:
                self.position.close()
                return
            return

        if abs(funding_v) < self.funding_threshold:
            return

        sl_dist = self.sl_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if funding_v > 0:
            # Longs are paying. Short to collect.
            if not self.allow_shorts:
                return
            sl = close_v + sl_dist
            self.sell(size=units, sl=sl)
        else:
            # Shorts are paying. Long to collect.
            sl = close_v - sl_dist
            self.buy(size=units, sl=sl)
