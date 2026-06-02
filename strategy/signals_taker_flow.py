"""
TODO_LEG candidate: Taker-flow imbalance (candle-level CVD proxy).

Reads `taker_buy_base_volume` (TBBV) from each 15m Binance kline. Computes:
    taker_imb(t) = (2 * taker_buy_base - volume) / volume      in [-1, 1]
    tibs(t)      = EMA(taker_imb, 4 bars)                      ~1h smoothing

Entries (when no position open):
    LONG  if tibs > +THR  AND  Close > 1H EMA20
    SHORT if tibs < -THR  AND  Close < 1H EMA20

Exits (whichever first):
    1) 2.0x ATR14 target hit
    2) 1.0x ATR14 stop hit
    3) tibs flips sign through zero (proxy: cross of 0)
    4) Time stop: 4 hours (= 16 bars at 15m)

Sizing: backtesting.py fraction-of-equity. risk_pct / stop_distance_pct, clipped.

Required Data column on the DataFrame passed to Backtest:
    `TakerBuyBase`  — same length as OHLCV; raw base-asset taker-buy volume.

If the column is missing, init() raises so the harness fails loudly rather
than silently regressing to a noise strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


def _ema_np(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple EMA on a 1-D numpy array, NaN-safe for warm-up."""
    if period <= 1:
        return arr.astype(float).copy()
    out = np.full_like(arr, np.nan, dtype=float)
    alpha = 2.0 / (period + 1.0)
    # Seed at first non-NaN sample
    first_idx = None
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            first_idx = i
            break
    if first_idx is None:
        return out
    val = float(arr[first_idx])
    out[first_idx] = val
    for i in range(first_idx + 1, len(arr)):
        x = arr[i]
        if np.isnan(x):
            out[i] = val
            continue
        val = alpha * x + (1.0 - alpha) * val
        out[i] = val
    return out


def _atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR via simple recursive smoothing on numpy arrays."""
    n = len(close)
    tr = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            a = high[i] - low[i]
            b = abs(high[i] - close[i - 1])
            c = abs(low[i] - close[i - 1])
            tr[i] = max(a, b, c)
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class TakerFlowImbalance(Strategy):
    """Candle-level taker-flow imbalance reversal/initiation signal."""

    # Sweepable params
    tibs_period = 4          # EMA smoothing on taker_imb
    tibs_threshold = 0.25    # entry threshold magnitude
    ema_1h_period = 20       # 1H EMA filter (20 1H-bars = 80 15m-bars)
    atr_period = 14
    atr_sl_mult = 1.0
    atr_tp_mult = 2.0
    max_hold_bars = 16       # 4h = 16 x 15m
    risk_per_trade_pct = 0.5  # 0.5% per-trade risk
    allow_shorts = True
    leverage = 20

    def init(self):
        df = self.data.df
        # Hard guard: the data column must be present and non-empty
        if "TakerBuyBase" not in df.columns:
            raise RuntimeError(
                "TakerFlowImbalance: 'TakerBuyBase' column is REQUIRED. "
                "Patch exchange/data.py::load_klines to preserve taker_buy_base "
                "and re-cache the parquet."
            )

        close = df["Close"].values.astype(float)
        high = df["High"].values.astype(float)
        low = df["Low"].values.astype(float)
        vol = df["Volume"].values.astype(float)
        tbb = df["TakerBuyBase"].values.astype(float)

        # taker_imb in [-1, 1]; guarded against zero-volume bars
        with np.errstate(divide="ignore", invalid="ignore"):
            taker_imb = np.where(vol > 0.0, (2.0 * tbb - vol) / vol, 0.0)
        taker_imb = np.clip(taker_imb, -1.0, 1.0)

        tibs = _ema_np(taker_imb, int(self.tibs_period))

        # 1H EMA(20) on Close — at 15m bars that's EMA span 80
        ema_1h = _ema_np(close, int(self.ema_1h_period) * 4)

        atr14 = _atr_np(high, low, close, int(self.atr_period))

        self.tibs = self.I(lambda: tibs, name="tibs")
        self.ema_1h = self.I(lambda: ema_1h, name="ema_1h")
        self.atr = self.I(lambda: atr14, name="atr")

        self._bars_in_trade = 0
        self._tibs_at_entry = 0.0

    def _size_from_risk(self, stop_distance_price: float) -> float:
        """Convert risk_pct + stop_distance to a backtesting.py size in (0, 1)."""
        price = float(self.data.Close[-1])
        if stop_distance_price <= 0.0 or price <= 0.0:
            return 0.0
        stop_pct = stop_distance_price / price
        if stop_pct <= 0.0:
            return 0.0
        risk_frac = float(self.risk_per_trade_pct) / 100.0
        # size = (risk_frac / stop_pct) but bounded to (0, 0.999) — fraction of equity
        size = risk_frac / stop_pct
        size *= float(self.leverage)  # leverage amplifies the notional
        size = max(0.0, min(size, 0.999))
        return size

    def next(self):
        # warm-up checks
        if (
            len(self.data) < max(int(self.ema_1h_period) * 4, int(self.atr_period), int(self.tibs_period)) + 2
            or np.isnan(self.atr[-1])
            or np.isnan(self.ema_1h[-1])
            or np.isnan(self.tibs[-1])
        ):
            return

        price = float(self.data.Close[-1])
        ema1h = float(self.ema_1h[-1])
        a = float(self.atr[-1])
        t = float(self.tibs[-1])
        t_prev = float(self.tibs[-2]) if len(self.tibs) >= 2 else t

        # Manage open position
        if self.position:
            self._bars_in_trade += 1

            # Time stop
            if self._bars_in_trade >= int(self.max_hold_bars):
                self.position.close()
                self._bars_in_trade = 0
                return

            # Zero-cross of tibs flips → exit
            if self.position.is_long and t_prev > 0.0 and t <= 0.0:
                self.position.close()
                self._bars_in_trade = 0
                return
            if self.position.is_short and t_prev < 0.0 and t >= 0.0:
                self.position.close()
                self._bars_in_trade = 0
                return
            return

        # Flat — look for entry
        thr = float(self.tibs_threshold)
        size = self._size_from_risk(a * float(self.atr_sl_mult))
        if size <= 0.0:
            return

        if t > thr and price > ema1h:
            sl = price - a * float(self.atr_sl_mult)
            tp = price + a * float(self.atr_tp_mult)
            if sl > 0 and tp > price:
                self.buy(size=size, sl=sl, tp=tp)
                self._bars_in_trade = 0
                self._tibs_at_entry = t
        elif self.allow_shorts and t < -thr and price < ema1h:
            sl = price + a * float(self.atr_sl_mult)
            tp = price - a * float(self.atr_tp_mult)
            if tp > 0 and sl > price:
                self.sell(size=size, sl=sl, tp=tp)
                self._bars_in_trade = 0
                self._tibs_at_entry = t
