"""
CarryHarvester v2 — adds a fast-move skip filter.

v1 collected funding but bled on the rare candle where price ran several
percent against us before the 8h funding event could compound enough to
offset. v2 refuses to open a new carry trade when the trailing 24h price
change exceeds `max_24h_change_pct` — typical liquidation cascades and
news spikes happen inside that window and resolve within a day; sitting
out flat for a day saves the worst tails.

Also exposes a tighter `sl_pct` grid in the sweep so the walk-forward can
pick a smaller risk envelope per fold.

Everything else (entry direction, exit conditions) matches v1 exactly.
"""

from __future__ import annotations

import numpy as np
from backtesting import Strategy

LOOKBACK_24H_15M = 96  # 96 × 15m = 24h


class CarryHarvesterV2(Strategy):
    funding_threshold = 0.0002
    funding_exit_threshold = 0.00005
    sl_pct = 0.01
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True
    max_24h_change_pct = 100.0  # 100 = disabled (matches v1 behaviour)

    def init(self) -> None:
        pass

    def _position_units(self, price: float, sl_distance: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _fast_move_block(self) -> bool:
        """Return True if last-24h abs return exceeds the filter threshold."""
        if self.max_24h_change_pct >= 100.0:
            return False
        if len(self.data) <= LOOKBACK_24H_15M:
            return False
        ref = self.data.Close[-LOOKBACK_24H_15M - 1]
        now = self.data.Close[-1]
        if ref <= 0 or not np.isfinite(ref):
            return False
        change = abs(now / ref - 1.0) * 100.0
        return change > self.max_24h_change_pct

    def next(self) -> None:
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]
        if not np.isfinite(funding_v) or not np.isfinite(close_v):
            return

        if self.position:
            if abs(funding_v) < self.funding_exit_threshold:
                self.position.close()
                return
            if self.position.is_long and funding_v > 0:
                self.position.close()
                return
            if self.position.is_short and funding_v < 0:
                self.position.close()
                return
            return

        if abs(funding_v) < self.funding_threshold:
            return

        # v2 gate: skip new entries during fast-move regimes.
        if self._fast_move_block():
            return

        sl_dist = self.sl_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if funding_v > 0:
            if not self.allow_shorts:
                return
            sl = close_v + sl_dist
            self.sell(size=units, sl=sl)
        else:
            sl = close_v - sl_dist
            self.buy(size=units, sl=sl)
