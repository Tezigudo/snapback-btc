"""
SnapbackBTCv2 — regime-replaced (not regime-stacked) snapback.

v1 stacked filters and starved itself: the +/-0.005% funding gate alone
produced ~2 signals per 60 days. v2 REPLACES v1's funding-magnitude gate
with a regime SIGN gate: only long when shorts are paying funding
("squeeze the shorts"), only short when longs are paying ("punish chasers").

Pattern borrowed from AgentQuant's regime-aware param selection — implemented
deterministically here (no LLM, no API). The funding regime is the
deterministic analog of AgentQuant's "VIX/Momentum" regime detection from
their LLM proposer.

LONG entry:
    RSI(2, 15m) < rsi_long_threshold
    close > EMA(200, 1h)                 ← still trend-with
    volume > volume_multiple * SMA(20)   ← still volume filter
    funding_regime == -1                 ← shorts are paying (active + negative)
    [optional] volatility_regime == 1    ← high vol only (defaults ON)

SHORT entry: mirror with funding_regime == +1.

Exits: same as v1 (ATR-based TP/SL, time-stop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.regime import attach_regimes
from strategy.signals import (
    FRACTIONAL_UNIT,
    SnapbackBTC,
    StrategyParams,
    prepare_strategy_data,
)


def prepare_strategy_data_v2(
    klines_15m: pd.DataFrame,
    klines_1h: pd.DataFrame,
    funding: pd.DataFrame,
    params: StrategyParams,
) -> pd.DataFrame:
    """v2 = v1 prep + regime columns attached. No lookahead introduced."""
    base = prepare_strategy_data(klines_15m, klines_1h, funding, params)
    return attach_regimes(
        base,
        funding=base["Funding"],
        ema_1h=base["EMA_1h"],
        atr_1h=base["ATR_1h"],
    )


class SnapbackBTCv2(SnapbackBTC):
    """v1 signals but with regime SIGN replacing v1's magnitude funding gate.

    Defaults:
      require_high_volatility = True    (on; mean-reversion needs movement)
      require_matched_trend   = False   (off; trend filter is already on close vs EMA)
    """

    require_high_volatility = True
    require_matched_trend = False

    def next(self) -> None:
        rsi_v = self.data.RSI[-1]
        ema_v = self.data.EMA_1h[-1]
        atr_v = self.data.ATR_1h[-1]
        vol_sma_v = self.data.VolSMA[-1]
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]
        vol_v = self.data.Volume[-1]
        fr_v = self.data.funding_regime[-1]
        vr_v = self.data.volatility_regime[-1]
        tr_v = self.data.trend_regime[-1]

        if any(
            v is None or not np.isfinite(v)
            for v in (rsi_v, ema_v, atr_v, vol_sma_v, funding_v, fr_v, vr_v, tr_v)
        ):
            return

        if self.position:
            if self._entry_bar is not None:
                age = len(self.data) - self._entry_bar
                if age >= self.time_stop_bars:
                    self.position.close()
                    self._entry_bar = None
            return

        # Universal regime gate(s)
        if self.require_high_volatility and vr_v <= 0:
            return

        volume_ok = vol_v > self.volume_multiple * vol_sma_v

        # v2 LONG: shorts are paying (funding_regime == -1)
        long_filters = (
            rsi_v < self.rsi_long_threshold
            and close_v > ema_v
            and volume_ok
            and fr_v < 0  # regime sign instead of magnitude
            and (not self.require_matched_trend or tr_v > 0)
        )
        # v2 SHORT: longs are paying (funding_regime == +1)
        short_filters = (
            self.allow_shorts
            and rsi_v > self.rsi_short_threshold
            and close_v < ema_v
            and volume_ok
            and fr_v > 0
            and (not self.require_matched_trend or tr_v < 0)
        )

        if long_filters:
            sl_dist = self.atr_sl_multiple * atr_v
            sl = close_v - sl_dist
            tp = close_v + self.atr_tp_multiple * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v < tp:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
        elif short_filters:
            sl_dist = self.atr_sl_multiple * atr_v
            sl = close_v + sl_dist
            tp = close_v - self.atr_tp_multiple * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and tp < close_v < sl:
                self.sell(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
