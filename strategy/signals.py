"""
Strategy v1: "RSI Extreme + EMA Confluence + Volume + Funding"

Multi-timeframe deterministic logic. Entry timeframe = 15m, trend filter +
ATR-based exits come from 1h. Funding rate (8h cadence on Binance perps) is
forward-filled to the entry timeframe and used as a filter / cost.

Two pieces:

  1. `prepare_strategy_data()` — pure-pandas, no backtesting.py.
     Takes raw OHLCV at 15m + 1h plus funding history, returns ONE DataFrame
     indexed at 15m with all indicators attached. Crucially, ALL 1h indicators
     are computed at the close of each 1h bar and then shifted by one bar
     before reindexing so that a 15m bar T sees only the last FULLY CLOSED
     1h bar's value — no lookahead.

  2. `SnapbackBTC` — backtesting.py Strategy that reads the prepared columns.
     Order placed in next() at bar T fills at open of T+1 (trade_on_close=False
     in backtest.py), so the strategy never trades on information it shouldn't
     have at decision time.

LONG entry, ALL of:
    RSI(2, 15m) < rsi_long_threshold (default 10)
    close > EMA(200, 1h)
    volume(15m) > volume_multiple * SMA(20, vol)
    funding_rate <= funding_long_max  (default -0.0003 per 8h)

SHORT is the mirror.

Exit:
    TP at +/- atr_tp_multiple * ATR(20, 1h)
    SL at -/+ atr_sl_multiple * ATR(20, 1h)
    Time-stop at time_stop_bars (default 48 * 15m = 12h)

Sizing:
    risk_per_trade_pct * equity / sl_distance = BTC quantity
    Expressed as integer "fractional units" (1 unit = FRACTIONAL_UNIT BTC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from backtesting import Strategy

from exchange.env import REPO_ROOT

from .indicators import atr, ema, rsi, sma

FRACTIONAL_UNIT = 1e-6  # matches FractionalBacktest config in backtest.py


@dataclass(frozen=True)
class StrategyParams:
    rsi_period: int = 2
    rsi_long_threshold: float = 10.0
    rsi_short_threshold: float = 90.0
    ema_period: int = 200
    volume_ma_period: int = 20
    volume_multiple: float = 1.5
    funding_long_max: float = -0.0003
    funding_short_min: float = 0.0003
    atr_period: int = 20
    atr_tp_multiple: float = 1.5
    atr_sl_multiple: float = 1.0
    time_stop_bars: int = 48
    risk_per_trade_pct: float = 2.0
    leverage: int = 20
    # Donchian-specific (ignored by snapback strategies)
    donchian_period_entry: int = 20
    donchian_period_exit: int = 10
    atr_trail_multiple: float = 0.0       # 0 = no trailing
    # Carry-specific (ignored by other strategies)
    funding_threshold: float = 0.0002
    funding_exit_threshold: float = 0.00005
    sl_pct: float = 0.01
    max_24h_change_pct: float = 100.0     # 100 = disabled (no filter)
    # Carry-v3 NEW tail-risk gates (100 / 100 = both disabled = v2 behaviour)
    atr_percentile_threshold: float = 100.0
    dd_halt_pct: float = 100.0
    # Carry-v4 NEW trend gate (0 = disabled = v3 behaviour)
    trend_ema_period: int = 0

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "StrategyParams":
        path = path or str(REPO_ROOT / "config" / "params.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        s = cfg["strategy"]
        z = cfg["sizing"]
        return cls(
            rsi_period=s["rsi_period"],
            rsi_long_threshold=float(s["rsi_long_threshold"]),
            rsi_short_threshold=float(s["rsi_short_threshold"]),
            ema_period=s["ema_period"],
            volume_ma_period=s["volume_ma_period"],
            volume_multiple=float(s["volume_multiple"]),
            funding_long_max=float(s["funding_long_max"]),
            funding_short_min=float(s["funding_short_min"]),
            atr_period=s["atr_period"],
            atr_tp_multiple=float(s["atr_tp_multiple"]),
            atr_sl_multiple=float(s["atr_sl_multiple"]),
            time_stop_bars=s["time_stop_bars"],
            risk_per_trade_pct=float(z["risk_per_trade_pct"]),
            leverage=int(z["leverage"]),
        )


def prepare_strategy_data(
    klines_15m: pd.DataFrame,
    klines_1h: pd.DataFrame,
    funding: pd.DataFrame,
    params: StrategyParams,
) -> pd.DataFrame:
    """Attach precomputed indicators to the 15m bars. No lookahead.

    Expects 15m and 1h DataFrames with columns open/high/low/close/volume
    (lowercase) indexed by tz-naive UTC time. Funding DataFrame has a
    `funding_rate` column indexed by funding_time.

    Returns a 15m DataFrame with capitalised OHLCV columns plus:
      RSI         : RSI(period) on 15m close
      VolSMA      : SMA(volume_ma_period) on 15m volume
      EMA_1h      : EMA(ema_period) on 1h close, shifted one 1h bar then
                    reindexed to 15m via ffill (so 15m at T uses last
                    fully-closed 1h bar)
      ATR_1h      : ATR(atr_period) on 1h H/L/C, same shift-then-ffill
      Funding     : funding_rate forward-filled from last known event
    """
    if klines_15m.empty:
        raise ValueError("klines_15m is empty")
    if klines_1h.empty:
        raise ValueError("klines_1h is empty")

    df = klines_15m.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    one_h = klines_1h.copy()
    one_h.columns = [c.capitalize() for c in one_h.columns]
    if one_h.index.tz is not None:
        one_h.index = one_h.index.tz_convert("UTC").tz_localize(None)

    # 15m indicators — directly usable since RSI[T] reflects close at T+15m,
    # and backtesting.py with trade_on_close=False fills at open of T+1.
    df["RSI"] = rsi(df["Close"], params.rsi_period)
    df["VolSMA"] = sma(df["Volume"], params.volume_ma_period)

    # 1h indicators — shift by 1 so the value at bar T represents the
    # information that became available at T-1's close (= T's open).
    # Then reindex to 15m with ffill.
    ema_1h = ema(one_h["Close"], params.ema_period).shift(1)
    atr_1h = atr(one_h["High"], one_h["Low"], one_h["Close"], params.atr_period).shift(1)
    df["EMA_1h"] = ema_1h.reindex(df.index, method="ffill")
    df["ATR_1h"] = atr_1h.reindex(df.index, method="ffill")

    # Funding rate — most recent known event, ffilled. No shift: the rate is
    # published at fundingTime; a 15m bar at or after fundingTime can see it.
    if funding is None or funding.empty:
        df["Funding"] = np.nan
    else:
        f = funding.copy()
        if f.index.tz is not None:
            f.index = f.index.tz_convert("UTC").tz_localize(None)
        df["Funding"] = f["funding_rate"].reindex(df.index, method="ffill")

    return df


class SnapbackBTC(Strategy):
    """v1 strategy. Params come from class attributes — backtest.py overrides
    them from `config/params.yaml` before instantiating."""

    # Filled by backtest.py before Backtest() is constructed.
    rsi_long_threshold = 10.0
    rsi_short_threshold = 90.0
    volume_multiple = 1.5
    funding_long_max = -0.0003
    funding_short_min = 0.0003
    atr_tp_multiple = 1.5
    atr_sl_multiple = 1.0
    time_stop_bars = 48
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True

    def init(self) -> None:
        # Indicators are read directly from self.data in next(). We deliberately
        # do NOT register them via self.I() — FractionalBacktest would scale
        # every registered indicator by 1/fractional_unit. We use plain Backtest
        # with a large starting cash instead so 1+ BTC fits naturally.
        self._entry_bar: int | None = None

    def _position_units(self, sl_distance: float, price: float) -> int:
        """Risk-based sizing in integer BTC units, clamped to broker margin.

        Backtest size = int N means N units of the asset. We compute N from the
        risk-per-trade target, then clamp to 95% of max-margin notional so the
        broker won't reject for insufficient margin.
        """
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        units = int(min(target_btc, max_btc))
        return max(units, 0)

    def next(self) -> None:
        # Skip until all warm-up indicators are ready.
        rsi_v = self.data.RSI[-1]
        ema_v = self.data.EMA_1h[-1]
        atr_v = self.data.ATR_1h[-1]
        vol_sma_v = self.data.VolSMA[-1]
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]
        vol_v = self.data.Volume[-1]

        if any(
            v is None or not np.isfinite(v)
            for v in (rsi_v, ema_v, atr_v, vol_sma_v, funding_v)
        ):
            return

        # Position management — handle time-stop while in a trade.
        if self.position:
            if self._entry_bar is not None:
                age = len(self.data) - self._entry_bar
                if age >= self.time_stop_bars:
                    self.position.close()
                    self._entry_bar = None
            return

        # Entry filters
        volume_ok = vol_v > self.volume_multiple * vol_sma_v
        long_filters = (
            rsi_v < self.rsi_long_threshold
            and close_v > ema_v
            and volume_ok
            and funding_v <= self.funding_long_max
        )
        short_filters = (
            self.allow_shorts
            and rsi_v > self.rsi_short_threshold
            and close_v < ema_v
            and volume_ok
            and funding_v >= self.funding_short_min
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
