"""
DayTradeMultiFactorBTC — multi-factor day-trading-style strategy on 15m.

Built per user spec (P5):
  - 15m entry timeframe (back from 4h, user wants day-trade pace)
  - RSI(14) thresholds: OVS<40 (long candidate), OVB>70 (short candidate)
    (user's asymmetric thresholds — tighter on shorts than longs)
  - Volume confirmation (volume > X × SMA(20) — real interest)
  - Candlestick pattern (bullish/bearish engulfing or hammer)
  - Trend filter (close vs EMA(200) on 15m)
  - Funding-rate "sentiment" gate (refuse crowded-trade direction)
  - MACD histogram confirmation (momentum agreement)
  - Max hold = 14 days (2 weeks at user request) = 1344 bars at 15m
  - 1.5% SL / 3% TP geometry (2:1 R:R)

Research basis (from internet search, May 2026):
  - RSI+MACD+Volume+Bollinger hybrid is a well-documented combo for
    crypto day trading (Medium @redsword_23261)
  - Bullish engulfing + volume confirmation: 65-75% win rate in BTC
    pair backtests (altrady, altFINS)
  - Funding rate as sentiment proxy: extreme funding (>0.05% / 8h) =
    crowded trade vulnerable to squeeze (Phemex, CoinMarketCap docs)

Entry rules — ALL must be true for a LONG:
  1. RSI(14) < rsi_long_threshold (default 40)
  2. Volume(t) > volume_multiple × SMA(Volume, 20)
  3. Bullish candle: bullish engulfing OR hammer
  4. Close > EMA(200) (in uptrend)
  5. Funding rate NOT extremely positive (not entering a crowded long)
  6. MACD histogram > 0 (bullish momentum)

SHORT = mirror with RSI > 70, bearish engulfing, close < EMA(200), funding
not extremely negative, MACD histogram < 0.

Exit rules:
  - SL hit (1.5% adverse)
  - TP hit (3.0% favourable)
  - Time stop (max_hold_bars)
  - Adverse trend cross (close moves to wrong side of EMA(200))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import (
    bearish_engulfing,
    bullish_engulfing,
    ema,
    hammer,
    macd,
    rsi,
    sma,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_4H_PARQUET_DEFAULT = _REPO_ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"


def _build_4h_ema_aligned(
    dates_15m: pd.DatetimeIndex,
    parquet_path: Path,
    ema_period: int = 200,
) -> np.ndarray:
    """Load 4H bars, compute EMA(N), align to 15m timestamps with lookahead safety.

    TIMING CONVENTION (critical):
        The parquet index is bar-OPEN time. A 4H bar opening at T closes at T+4h.
        We may only use the EMA of that bar at 15m timestamps >= T+4h.

    Implementation mirrors strategy/signals_divergence_v2._build_4h_ema200_aligned:
        1. Load FULL 4H parquet (all history → ensures EMA(200) is warmed up
           before any OOS window begins).
        2. Compute EMA(N) on 4H closes.
        3. Re-index the EMA series by CLOSE timestamps (open_time + 4h).
        4. pd.merge_asof(direction="backward") against the 15m index — each 15m
           bar receives the EMA of the most-recently CLOSED 4H bar. The EMA of
           the still-open 4H bar is unreachable, hence no lookahead.
        5. Both indices stripped to tz-naive + datetime64[us] precision (pandas
           refuses merges across mixed precision / tz).

    Returns an ndarray aligned 1:1 to dates_15m (NaN during warm-up).
    """
    df4h = pd.read_parquet(parquet_path)
    if df4h.index.tz is not None:
        df4h.index = df4h.index.tz_localize(None)
    # parquet columns are lowercased on disk; backtesting.py capitalises for
    # the 15m frame but we read 4H directly here so use the on-disk name.
    close_col = "close" if "close" in df4h.columns else "Close"
    ema_4h = ema(df4h[close_col], ema_period).values

    close_times = df4h.index + pd.Timedelta(hours=4)
    close_times_us = pd.DatetimeIndex(close_times.astype("datetime64[us]"))

    left_idx_raw = dates_15m
    if left_idx_raw.tz is not None:
        left_idx_raw = left_idx_raw.tz_localize(None)
    left_idx = pd.DatetimeIndex(left_idx_raw.astype("datetime64[us]"))

    right = pd.DataFrame({"ema4h": ema_4h}, index=close_times_us).sort_index()
    left = pd.DataFrame(index=left_idx)
    merged = pd.merge_asof(
        left, right,
        left_index=True, right_index=True,
        direction="backward",
    )
    return merged["ema4h"].values


class DayTradeMultiFactorBTC(Strategy):
    # --- entry thresholds (user-specified RSI(14) asymmetric) ---
    rsi_period = 14
    rsi_long_threshold = 40.0       # OVS < 40 (long candidate)
    rsi_short_threshold = 70.0      # OVB > 70 (short candidate)

    # --- volume confirmation ---
    volume_ma_period = 20
    volume_multiple = 1.5           # volume must be 1.5× SMA(20)

    # --- candlestick / trend ---
    require_candlestick = True       # require bullish/bearish engulfing or hammer
    mf_trend_ema_period = 200       # 200-EMA on 15m for trend filter
                                    # (name avoids collision with carry-v4 trend_ema_period
                                    #  which can be 0 in shared StrategyParams)
    require_trend = True            # require close on correct side of EMA(200)

    # --- MACD momentum filter ---
    require_macd = True             # require MACD histogram sign matches direction
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # --- funding sentiment proxy (refuse crowded trade) ---
    funding_extreme_threshold = 0.0005   # ±0.05% per 8h is "extreme"
    require_funding_not_extreme = True

    # --- exits ---
    sl_pct = 0.015                  # 1.5% stop loss
    tp_pct = 0.03                   # 3.0% take profit
    max_hold_bars = 1344            # 14 days × 96 bars/day at 15m

    # --- sizing ---
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True

    # --- 4H EMA200 regime gate (additive; deepening 2026-06-02 → +27.45pp) ---
    # Lookahead-safe alignment: see _build_4h_ema_aligned() module docstring.
    # Long entries require 15m close > 4H EMA(200); shorts require close <.
    # Disable by setting use_mtf_4h_gate=False (or passing False via bt.run).
    use_mtf_4h_gate = True
    mtf_4h_ema_period = 200
    mtf_4h_parquet_path = str(_4H_PARQUET_DEFAULT)

    # --- Cross-coin 4H EMA200 veto (additive; portfolio harness 2026-06-02) ---
    # Optional SECOND 4H gate using a DIFFERENT coin's 4H parquet (e.g. BTC's
    # 4H trend gating SOL longs/shorts). Empty string = disabled. When set,
    # entries must clear BOTH the primary mtf_4h gate AND this cross gate.
    # Same lookahead-safe alignment via _build_4h_ema_aligned. NaN-block
    # treated as fail (warm-up bars cannot enter).
    cross_4h_parquet_path = ""

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        open_ = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self._rsi = rsi(close, self.rsi_period).values
        self._vol_sma = sma(volume, self.volume_ma_period).values
        self._trend_ema = ema(close, self.mf_trend_ema_period).values
        _, _, self._macd_hist = (s.values for s in
                                  macd(close, self.macd_fast, self.macd_slow, self.macd_signal))
        self._bull_engulf = bullish_engulfing(open_, high, low, close).values
        self._bear_engulf = bearish_engulfing(open_, high, low, close).values
        self._hammer = hammer(open_, high, low, close).values
        self._entry_bar: int | None = None

        # 4H EMA(N) regime filter (additive). Loaded from the FULL 4H parquet so
        # the EMA is warmed-up before any backtest slice begins. Aligned to the
        # 15m index via backward merge_asof on bar-CLOSE timestamps — same
        # convention as strategy/signals_divergence_v2.py.
        if self.use_mtf_4h_gate:
            dates = pd.DatetimeIndex(self.data.index)
            self._ema_4h_200 = _build_4h_ema_aligned(
                dates,
                Path(self.mtf_4h_parquet_path),
                self.mtf_4h_ema_period,
            )
        else:
            self._ema_4h_200 = None

        # Cross-coin 4H EMA veto (additive). Reuses the same lookahead-safe
        # alignment helper so warm-up + bar-close timing match the primary gate.
        if self.cross_4h_parquet_path:
            dates_x = pd.DatetimeIndex(self.data.index)
            self._ema_4h_cross = _build_4h_ema_aligned(
                dates_x,
                Path(self.cross_4h_parquet_path),
                self.mtf_4h_ema_period,
            )
        else:
            self._ema_4h_cross = None

    def _position_units(self, price: float, sl_distance: float) -> int:
        # NOTE: backtesting.py 0.6.5 only accepts integer units.
        # Fractional 0.001-BTC sizing is implemented via HARNESS-level price
        # scaling (see tools/_fractional_run.py). Under scaling: 1 returned
        # "unit" == 0.001 BTC, matching Binance USDT-M perp qty_step.
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _long_signal(self, i: int) -> bool:
        # 1. RSI oversold
        if not (np.isfinite(self._rsi[i]) and self._rsi[i] < self.rsi_long_threshold):
            return False
        # 2. Volume confirmation
        vol_sma_v = self._vol_sma[i]
        if not (np.isfinite(vol_sma_v) and self.data.Volume[-1] > self.volume_multiple * vol_sma_v):
            return False
        # 3. Trend filter
        if self.require_trend:
            t = self._trend_ema[i]
            if not (np.isfinite(t) and self.data.Close[-1] > t):
                return False
        # 4. Candlestick pattern
        if self.require_candlestick:
            if not (bool(self._bull_engulf[i]) or bool(self._hammer[i])):
                return False
        # 5. MACD agreement
        if self.require_macd:
            h = self._macd_hist[i]
            if not (np.isfinite(h) and h > 0):
                return False
        # 6. Funding sentiment (refuse crowded long)
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f > self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        # 7. 4H EMA(N) regime gate — added 2026-06-02. Closes the loop on the
        # "trend gate" already present (15m EMA200) by also requiring the higher
        # timeframe regime to agree. Long requires 15m close > 4H EMA(200).
        if self.use_mtf_4h_gate and self._ema_4h_200 is not None:
            ema_4h_v = self._ema_4h_200[i]
            if not (np.isfinite(ema_4h_v) and self.data.Close[-1] > ema_4h_v):
                return False
            # 7b. Cross-coin 4H EMA veto (optional). Requires the OTHER coin's
            # 4H trend to also agree before allowing the long entry.
            if self._ema_4h_cross is not None:
                ema_4h_x = self._ema_4h_cross[i]
                if not (np.isfinite(ema_4h_x) and self.data.Close[-1] > ema_4h_x):
                    return False
        return True

    def _short_signal(self, i: int) -> bool:
        if not self.allow_shorts:
            return False
        if not (np.isfinite(self._rsi[i]) and self._rsi[i] > self.rsi_short_threshold):
            return False
        vol_sma_v = self._vol_sma[i]
        if not (np.isfinite(vol_sma_v) and self.data.Volume[-1] > self.volume_multiple * vol_sma_v):
            return False
        if self.require_trend:
            t = self._trend_ema[i]
            if not (np.isfinite(t) and self.data.Close[-1] < t):
                return False
        if self.require_candlestick:
            if not bool(self._bear_engulf[i]):
                return False
        if self.require_macd:
            h = self._macd_hist[i]
            if not (np.isfinite(h) and h < 0):
                return False
        if self.require_funding_not_extreme:
            try:
                f = self.data.Funding[-1]
                if np.isfinite(f) and f < -self.funding_extreme_threshold:
                    return False
            except (AttributeError, IndexError):
                pass
        # 7. 4H EMA(N) regime gate (short mirror). Short requires 15m close < EMA.
        if self.use_mtf_4h_gate and self._ema_4h_200 is not None:
            ema_4h_v = self._ema_4h_200[i]
            if not (np.isfinite(ema_4h_v) and self.data.Close[-1] < ema_4h_v):
                return False
            # 7b. Cross-coin 4H EMA veto (short mirror). Requires the OTHER
            # coin's 4H trend to also agree (close on bear side) before short.
            if self._ema_4h_cross is not None:
                ema_4h_x = self._ema_4h_cross[i]
                if not (np.isfinite(ema_4h_x) and self.data.Close[-1] < ema_4h_x):
                    return False
        return True

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Position management: time stop + adverse-trend exit
        if self.position:
            if self._entry_bar is not None:
                if (i - self._entry_bar) >= self.max_hold_bars:
                    self.position.close()
                    self._entry_bar = None
                    return
            # Adverse-trend exit
            if self.require_trend:
                t = self._trend_ema[i]
                if np.isfinite(t):
                    if self.position.is_long and close_v < t:
                        self.position.close()
                        self._entry_bar = None
                        return
                    if self.position.is_short and close_v > t:
                        self.position.close()
                        self._entry_bar = None
                        return
            return

        # Entry
        sl_dist = self.sl_pct * close_v
        tp_dist = self.tp_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if self._long_signal(i):
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
            self._entry_bar = i
        elif self._short_signal(i):
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
            self._entry_bar = i
