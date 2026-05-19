"""
DayTradeMultiFactorBTCv3 — v2 + adaptive risk management.

Built to address the 2024 H1 underperformance (see INVESTIGATE_2024H1.html).
The diagnosis: v1/v2 buy dips during parabolic moves and during topping/chop
because they have no margin-of-safety check against price extension or
volatility regime.

v3 adds 3 INDEPENDENT, INDIVIDUALLY-TOGGLABLE gates on top of v2's TA
confirmations:

  1. Distance-from-EMA filter
     - Block longs when close > (1 + max_distance_above_ema_pct) × EMA(200).
     - Block shorts when close < (1 − max_distance_above_ema_pct) × EMA(200).
     - Default: 0.20 → skip when price is more than 20% extended.
     - Rationale: at 20%+ extension from EMA, dip-buying often catches a
       knife. Stays on the sideline during parabolic moves.

  2. ATR-based stops (replaces fixed 1.5% SL / 3.0% TP)
     - sl_distance = atr_sl_k * ATR(atr_period)
     - tp_distance = atr_tp_k * ATR(atr_period)
     - Position size still follows risk-based formula: qty = risk_$ / sl_distance.
     - Default: atr_sl_k=1.5, atr_tp_k=3.0 — same 2:1 R:R but vol-adaptive.
     - Rationale: in high-vol regimes a fixed 1.5% stop is too tight; ATR
       widens stops automatically without changing risk budget.

  3. Vol-regime gate (pause entries when volatility is in top percentile)
     - Daily ATR(14) rolling 30-day percentile rank.
     - If current daily ATR's rank > vol_regime_max_pctile (default 0.85):
       skip new entries. Existing positions managed normally.
     - Rationale: parabolic rallies and crashes both spike ATR. Sitting out
       avoids the worst clusters of stop-outs.

Each gate has an `enable_*` boolean for ablation testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import atr
from strategy.signals_multifactor_v2 import DayTradeMultiFactorBTCv2


class DayTradeMultiFactorBTCv3(DayTradeMultiFactorBTCv2):
    # Inherit v2's confirmations_required default (=2). For the deployable
    # multifactor-v3 we want the same all-3-TA gate as v2-strict.
    confirmations_required: int = 3

    # --- v3 gate switches (CLASS-LEVEL — variant subclasses override) ---
    enable_dist_ema_filter: bool = True
    enable_atr_stops: bool = True
    enable_vol_regime_gate: bool = True

    # --- 1. distance-from-EMA ---
    max_distance_above_ema_pct: float = 0.20   # skip longs when price > 20% above EMA200
    max_distance_below_ema_pct: float = 0.20   # skip shorts when price > 20% below EMA200

    # --- 2. ATR stops ---
    atr_period: int = 14
    atr_sl_k: float = 1.5    # SL distance = 1.5 × ATR
    atr_tp_k: float = 3.0    # TP distance = 3.0 × ATR (keeps 2:1 R:R)

    # --- 3. vol-regime gate ---
    vol_regime_lookback_days: int = 30          # rolling window for percentile
    vol_regime_max_pctile: float = 0.85         # skip when current ATR > 85th pctile

    def init(self) -> None:
        super().init()
        # ATR on 15m for stops
        self._atr15m = atr(
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
            pd.Series(self.data.Close),
            self.atr_period,
        ).values

        # Daily ATR rolling percentile — precompute for cheap lookup at next()
        close = pd.Series(self.data.Close, index=pd.to_datetime(self.data.index))
        high = pd.Series(self.data.High, index=close.index)
        low = pd.Series(self.data.Low, index=close.index)
        # Resample to 1D for stability
        daily = pd.DataFrame({"High": high, "Low": low, "Close": close}).resample("1D").agg({
            "High": "max", "Low": "min", "Close": "last",
        })
        daily_atr = atr(daily["High"], daily["Low"], daily["Close"], self.atr_period)
        # Rolling 30-day percentile rank: where does today's ATR sit vs last N days?
        rolling_rank = daily_atr.rolling(window=self.vol_regime_lookback_days, min_periods=10).apply(
            lambda s: (s.rank(pct=True).iloc[-1]) if len(s) else np.nan, raw=False
        )
        # Map each 15m bar to its day's percentile rank (use rank from the PRIOR
        # closed day to avoid lookahead).
        rank_series = rolling_rank.shift(1)
        # ffill onto the 15m index
        full_idx = pd.DatetimeIndex(self.data.index)
        self._daily_atr_rank = rank_series.reindex(
            full_idx.normalize(), method="ffill",
        ).values

    # --------------------------------------------------------------------
    # Gates
    # --------------------------------------------------------------------
    def _dist_ema_ok(self, side: str, i: int) -> bool:
        if not self.enable_dist_ema_filter:
            return True
        ema_v = self._trend_ema[i]
        close_v = self.data.Close[-1]
        if not np.isfinite(ema_v) or ema_v <= 0:
            return False
        ratio = close_v / ema_v - 1.0
        if side == "long":
            return ratio <= self.max_distance_above_ema_pct
        return ratio >= -self.max_distance_below_ema_pct

    def _vol_regime_ok(self, i: int) -> bool:
        if not self.enable_vol_regime_gate:
            return True
        try:
            rank = self._daily_atr_rank[i]
        except (IndexError, AttributeError):
            return True
        if not np.isfinite(rank):
            return True  # not enough history yet → allow
        return rank <= self.vol_regime_max_pctile

    # --------------------------------------------------------------------
    # Sizing using ATR (when enabled)
    # --------------------------------------------------------------------
    def _sl_tp_distances(self, i: int, price: float) -> tuple[float, float]:
        """Return (sl_distance, tp_distance) in PRICE units."""
        if self.enable_atr_stops and np.isfinite(self._atr15m[i]) and self._atr15m[i] > 0:
            atr_v = float(self._atr15m[i])
            return self.atr_sl_k * atr_v, self.atr_tp_k * atr_v
        return self.sl_pct * price, self.tp_pct * price

    # --------------------------------------------------------------------
    # Main next() — copy v2 structure but use gates + ATR distances
    # --------------------------------------------------------------------
    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Position management — same shape as v2
        if self.position:
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold_bars:
                self.position.close()
                self._entry_bar = None
                return
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

        # Vol-regime gate applies to ENTRIES only (not exits)
        if not self._vol_regime_ok(i):
            return

        # Compute stop / target distances + position size
        sl_dist, tp_dist = self._sl_tp_distances(i, close_v)
        if sl_dist <= 0 or tp_dist <= 0:
            return
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        # Entry: v2's base + TA filter, plus v3's distance-from-EMA + vol-regime
        if self._base_long_ok(i) and self._dist_ema_ok("long", i):
            n_conf = self._ta_confirmations_count(i, "long")
            if n_conf >= self.confirmations_required:
                self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
                self._entry_bar = i
        elif self._base_short_ok(i) and self._dist_ema_ok("short", i):
            n_conf = self._ta_confirmations_count(i, "short")
            if n_conf >= self.confirmations_required:
                self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
                self._entry_bar = i


# Ablation variants for testing which gate(s) actually help
class V3DistEmaOnly(DayTradeMultiFactorBTCv3):
    """Only the distance-from-EMA filter (other gates off)."""
    enable_dist_ema_filter: bool = True
    enable_atr_stops: bool = False
    enable_vol_regime_gate: bool = False


class V3VolRegimeOnly(DayTradeMultiFactorBTCv3):
    """Only the vol-regime gate."""
    enable_dist_ema_filter: bool = False
    enable_atr_stops: bool = False
    enable_vol_regime_gate: bool = True


class V3AtrStopsOnly(DayTradeMultiFactorBTCv3):
    """Only ATR-based stops."""
    enable_dist_ema_filter: bool = False
    enable_atr_stops: bool = True
    enable_vol_regime_gate: bool = False


class V3All(DayTradeMultiFactorBTCv3):
    """All three v3 gates enabled."""
    enable_dist_ema_filter: bool = True
    enable_atr_stops: bool = True
    enable_vol_regime_gate: bool = True


# --- Friction-tolerant variants (wider ATR stops) ----------------------------
# Original V3All has atr_sl_k=1.5, atr_tp_k=3.0 — narrow stops that get eaten
# by 7-10 bps friction. Wider variants trade SL hit rate for more room to absorb
# fees + slippage. Same 2:1 R:R maintained.
class V3AllWider2(V3All):
    """V3All with 2.0×ATR SL / 4.0×ATR TP — ~33% wider stops."""
    atr_sl_k: float = 2.0
    atr_tp_k: float = 4.0


class V3AllWider3(V3All):
    """V3All with 3.0×ATR SL / 6.0×ATR TP — 2× wider stops."""
    atr_sl_k: float = 3.0
    atr_tp_k: float = 6.0


class V3AllWider4(V3All):
    """V3All with 4.0×ATR SL / 8.0×ATR TP — most defensive, biggest absorption of friction."""
    atr_sl_k: float = 4.0
    atr_tp_k: float = 8.0


# Intermediate / finer granularity for optimum search
class V3AllK_2_5(V3All):
    atr_sl_k: float = 2.5
    atr_tp_k: float = 5.0


class V3AllK_3_5(V3All):
    atr_sl_k: float = 3.5
    atr_tp_k: float = 7.0


class V3AllK_5_0(V3All):
    atr_sl_k: float = 5.0
    atr_tp_k: float = 10.0


class V3AllK_6_0(V3All):
    atr_sl_k: float = 6.0
    atr_tp_k: float = 12.0
