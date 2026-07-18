"""
Supertrend variant strategies — additive subclasses of SupertrendBTC.

Each variant reuses SupertrendBTC's `next()` entry/exit/sizing logic where
possible and only overrides the pieces needed for its filter or exit change.
All variants are tf-agnostic (native 4h, same as the base supertrend) and
read `[-1]` for the last CLOSED bar, matching the causal convention of
strategy/indicators.py.

2026-06-14: built for walk-forward bake-off (experiments/walkforward.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import adx, atr, donchian_channel, ema, supertrend
from strategy.signals_supertrend import SupertrendBTC, attach_supertrend


# ---------------------------------------------------------------------------
# 1. SupertrendADX — ADX trend-strength filter on entries.
# ---------------------------------------------------------------------------
def attach_supertrend_adx(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
    adx_period: int = 14,
) -> pd.DataFrame:
    """attach_supertrend() + STADX column (Wilder ADX)."""
    out = attach_supertrend(df, period=period, multiplier=multiplier, atr_period=atr_period)
    out["STADX"] = adx(out["High"], out["Low"], out["Close"], adx_period)
    return out


class SupertrendADX(SupertrendBTC):
    """Base Supertrend entries, gated on ADX[-1] >= st_adx_min (trend strength)."""

    st_adx_min: float = 20
    st_adx_period: int = 14

    def next(self) -> None:
        close_v = self.data.Close[-1]
        direction = self.data.STDir[-1]
        atr_v = self.data.STAtr[-1]
        adx_v = self.data.STADX[-1]

        if any(v is None or not np.isfinite(v) for v in (direction, atr_v, adx_v)):
            return
        if len(self.data) < 2:
            return
        prev_direction = self.data.STDir[-2]
        if prev_direction is None or not np.isfinite(prev_direction):
            return

        if self.position:
            if self.position.is_long and direction == -1.0:
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and direction == 1.0:
                self.position.close()
                self._entry_bar = None
            return

        flipped_long = prev_direction == -1.0 and direction == 1.0
        flipped_short = prev_direction == 1.0 and direction == -1.0

        if adx_v < self.st_adx_min:
            return

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


# ---------------------------------------------------------------------------
# 2. SupertrendEMA — long-only above EMA, short-only below EMA (regime gate).
# ---------------------------------------------------------------------------
def attach_supertrend_ema(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
    ema_period: int = 200,
) -> pd.DataFrame:
    """attach_supertrend() + STEMA column."""
    out = attach_supertrend(df, period=period, multiplier=multiplier, atr_period=atr_period)
    out["STEMA"] = ema(out["Close"], ema_period)
    return out


class SupertrendEMA(SupertrendBTC):
    """Base Supertrend entries, gated by an EMA regime filter:
    longs only when close[-1] > EMA[-1], shorts only when close[-1] < EMA[-1].
    """

    st_ema_period: int = 200

    def next(self) -> None:
        close_v = self.data.Close[-1]
        direction = self.data.STDir[-1]
        atr_v = self.data.STAtr[-1]
        ema_v = self.data.STEMA[-1]

        if any(v is None or not np.isfinite(v) for v in (direction, atr_v, ema_v)):
            return
        if len(self.data) < 2:
            return
        prev_direction = self.data.STDir[-2]
        if prev_direction is None or not np.isfinite(prev_direction):
            return

        if self.position:
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

        if flipped_long and close_v > ema_v:
            sl = close_v - sl_dist
            tp = close_v + self.st_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
        elif self.allow_shorts and flipped_short and close_v < ema_v:
            sl = close_v + sl_dist
            tp = close_v - self.st_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)


# ---------------------------------------------------------------------------
# 3. SupertrendDual — fast + slow supertrend agreement filter.
# ---------------------------------------------------------------------------
def attach_supertrend_dual(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
    slow_period: int = 20,
    slow_multiplier: float = 4.0,
) -> pd.DataFrame:
    """attach_supertrend() (fast) + STDir2/STLine2 (slow supertrend)."""
    out = attach_supertrend(df, period=period, multiplier=multiplier, atr_period=atr_period)
    st_slow = supertrend(out["High"], out["Low"], out["Close"], period=slow_period, multiplier=slow_multiplier)
    out["STLine2"] = st_slow["supertrend"]
    out["STDir2"] = st_slow["direction"]
    return out


class SupertrendDual(SupertrendBTC):
    """Enter long only when both fast (STDir) and slow (STDir2) supertrends
    agree on +1 (mirror for short). Exit still on fast STDir opposite flip.
    """

    st_slow_period: int = 20
    st_slow_multiplier: float = 4.0

    def next(self) -> None:
        close_v = self.data.Close[-1]
        direction = self.data.STDir[-1]
        slow_direction = self.data.STDir2[-1]
        atr_v = self.data.STAtr[-1]

        if any(v is None or not np.isfinite(v) for v in (direction, slow_direction, atr_v)):
            return
        if len(self.data) < 2:
            return
        prev_direction = self.data.STDir[-2]
        if prev_direction is None or not np.isfinite(prev_direction):
            return

        if self.position:
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

        if flipped_long and slow_direction == 1.0:
            sl = close_v - sl_dist
            tp = close_v + self.st_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
        elif self.allow_shorts and flipped_short and slow_direction == -1.0:
            sl = close_v + sl_dist
            tp = close_v - self.st_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)


# ---------------------------------------------------------------------------
# 4. SupertrendDonchExit — Donchian-channel exit instead of fixed ATR TP.
# ---------------------------------------------------------------------------
def attach_supertrend_donchexit(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
    donch_period: int = 20,
) -> pd.DataFrame:
    """attach_supertrend() + STDonchUpper/STDonchLower columns."""
    out = attach_supertrend(df, period=period, multiplier=multiplier, atr_period=atr_period)
    upper, lower = donchian_channel(out["High"], out["Low"], donch_period)
    out["STDonchUpper"] = upper
    out["STDonchLower"] = lower
    return out


class SupertrendDonchExit(SupertrendBTC):
    """Entry as base (Supertrend flip). Exit on opposite Donchian channel
    breach (close crosses the opposite-side Donchian band) instead of the
    fixed ATR take-profit; the ATR stop-loss is still set on entry as a
    hard backstop. Opposite STDir flip still closes the position too.
    """

    st_donch_period: int = 20

    def next(self) -> None:
        close_v = self.data.Close[-1]
        direction = self.data.STDir[-1]
        atr_v = self.data.STAtr[-1]
        donch_upper = self.data.STDonchUpper[-1]
        donch_lower = self.data.STDonchLower[-1]

        if any(v is None or not np.isfinite(v) for v in (direction, atr_v, donch_upper, donch_lower)):
            return
        if len(self.data) < 2:
            return
        prev_direction = self.data.STDir[-2]
        if prev_direction is None or not np.isfinite(prev_direction):
            return

        if self.position:
            if self.position.is_long and (direction == -1.0 or close_v <= donch_lower):
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and (direction == 1.0 or close_v >= donch_upper):
                self.position.close()
                self._entry_bar = None
            return

        flipped_long = prev_direction == -1.0 and direction == 1.0
        flipped_short = prev_direction == 1.0 and direction == -1.0

        sl_dist = self.st_sl_atr * atr_v

        # No `tp=` here — the Donchian breach exit above (and opposite STDir
        # flip) replace the fixed ATR take-profit. SL remains as a hard stop.
        if flipped_long:
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
        elif self.allow_shorts and flipped_short:
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)


# ---------------------------------------------------------------------------
# 5. SupertrendVolAdaptive — multiplier adapts to a volatility regime.
# ---------------------------------------------------------------------------
def attach_supertrend_voladapt(
    df: pd.DataFrame,
    period: int = 10,
    multiplier_low: float = 2.0,
    multiplier_high: float = 4.0,
    atr_period: int = 14,
    vol_lookback: int = 100,
) -> pd.DataFrame:
    """Approximate a per-bar-adaptive-multiplier Supertrend by blending two
    precomputed supertrends (mult_low / mult_high) per-bar, weighted by the
    rolling percentile rank of ATR/Close (the volatility regime).

    Implementation choice: a true per-bar-multiplier Supertrend requires
    rewriting the band-carry recursion in supertrend() (final_upper/lower
    depend on the prior bar's multiplier-dependent bands), which would mean
    duplicating/modifying the shared indicator. Instead we compute two full
    Supertrend series with fixed multipliers (mult_low, mult_high) — both
    using the *existing*, unmodified supertrend() — and at each bar select
    whichever series' STDir/STLine corresponds to the bar's vol regime:
    rolling ATR%/Close percentile rank over `vol_lookback` bars >= 0.5 picks
    the wide (mult_high) band (high-vol regime), else the tight (mult_low)
    band (low-vol regime). This is causal (percentile rank at bar i only
    uses bars <= i) and reuses the unmodified, validated supertrend().

    Columns added: STLine, STDir, STAtr (selected/blended series — same
    names as attach_supertrend so SupertrendVolAdaptive.next() can reuse the
    base class entry/exit logic unchanged), plus STVolRegime (0=low, 1=high)
    for inspection.
    """
    out = df.copy()
    atr_v = atr(out["High"], out["Low"], out["Close"], atr_period)
    vol_ratio = atr_v / out["Close"]
    vol_pctile = vol_ratio.rolling(window=vol_lookback, min_periods=vol_lookback).rank(pct=True)

    st_low = supertrend(out["High"], out["Low"], out["Close"], period=period, multiplier=multiplier_low)
    st_high = supertrend(out["High"], out["Low"], out["Close"], period=period, multiplier=multiplier_high)

    regime_high = vol_pctile >= 0.5  # bool Series, NaN -> False during warm-up

    out["STLine"] = st_low["supertrend"].where(~regime_high, st_high["supertrend"])
    out["STDir"] = st_low["direction"].where(~regime_high, st_high["direction"])
    out["STAtr"] = atr_v
    out["STVolRegime"] = regime_high.astype(float).where(vol_pctile.notna())
    return out


class SupertrendVolAdaptive(SupertrendBTC):
    """Supertrend whose effective multiplier widens in high-vol regimes and
    tightens in low-vol regimes (see attach_supertrend_voladapt for the
    blend implementation). Entry/exit logic is identical to the base class —
    STDir/STLine/STAtr are the regime-selected series.
    """

    st_vol_lookback: int = 100
    st_mult_low: float = 2.0
    st_mult_high: float = 4.0


# ---------------------------------------------------------------------------
# 6. SupertrendADXDonchExit — round-1 winners combined: ADX entry gate +
#    Donchian-channel exit.
# ---------------------------------------------------------------------------
def attach_supertrend_adx_donchexit(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
    adx_period: int = 14,
    donch_period: int = 20,
) -> pd.DataFrame:
    """attach_supertrend() + STADX column + STDonchUpper/STDonchLower columns."""
    out = attach_supertrend(df, period=period, multiplier=multiplier, atr_period=atr_period)
    out["STADX"] = adx(out["High"], out["Low"], out["Close"], adx_period)
    upper, lower = donchian_channel(out["High"], out["Low"], donch_period)
    out["STDonchUpper"] = upper
    out["STDonchLower"] = lower
    return out


class SupertrendADXDonchExit(SupertrendBTC):
    """Combines the two round-1 winners: ADX entry gate (enter only when
    ADX[-1] >= st_adx_min) AND exit on opposite Donchian channel breach
    (mirrors SupertrendDonchExit's exit logic, replacing the fixed ATR TP).
    The ATR stop-loss remains a hard backstop on entry, as in SupertrendDonchExit.
    """

    st_adx_min: float = 20
    st_adx_period: int = 14
    st_donch_period: int = 20

    def next(self) -> None:
        close_v = self.data.Close[-1]
        direction = self.data.STDir[-1]
        atr_v = self.data.STAtr[-1]
        adx_v = self.data.STADX[-1]
        donch_upper = self.data.STDonchUpper[-1]
        donch_lower = self.data.STDonchLower[-1]

        if any(v is None or not np.isfinite(v)
               for v in (direction, atr_v, adx_v, donch_upper, donch_lower)):
            return
        if len(self.data) < 2:
            return
        prev_direction = self.data.STDir[-2]
        if prev_direction is None or not np.isfinite(prev_direction):
            return

        if self.position:
            if self.position.is_long and (direction == -1.0 or close_v <= donch_lower):
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and (direction == 1.0 or close_v >= donch_upper):
                self.position.close()
                self._entry_bar = None
            return

        flipped_long = prev_direction == -1.0 and direction == 1.0
        flipped_short = prev_direction == 1.0 and direction == -1.0

        if adx_v < self.st_adx_min:
            return

        sl_dist = self.st_sl_atr * atr_v

        # No `tp=` here — the Donchian breach exit above (and opposite STDir
        # flip) replace the fixed ATR take-profit. SL remains as a hard stop.
        if flipped_long:
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
        elif self.allow_shorts and flipped_short:
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)


# ---------------------------------------------------------------------------
# 7. SupertrendTrail — Supertrend flip entry, optional ADX gate, chandelier
#    trailing-stop exit (replaces the fixed ATR TP).
# ---------------------------------------------------------------------------
def attach_supertrend_trail(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_period: int = 14,
    adx_period: int = 14,
) -> pd.DataFrame:
    """attach_supertrend() + STADX column (ADX gate is optional — st_adx_min=0 disables it)."""
    out = attach_supertrend(df, period=period, multiplier=multiplier, atr_period=atr_period)
    out["STADX"] = adx(out["High"], out["Low"], out["Close"], adx_period)
    return out


class SupertrendTrail(SupertrendBTC):
    """Supertrend flip entry with an optional ADX trend-strength gate
    (st_adx_min=0 disables the gate). Exit via a chandelier trailing stop
    instead of the fixed 5x ATR take-profit:

      long:  trail = (highest High since entry) - st_trail_atr * ATR; exit when Close < trail
      short: trail = (lowest Low since entry)   + st_trail_atr * ATR; exit when Close > trail

    The initial ATR stop-loss from the base class is still set on entry as a
    hard backstop (whichever of SL / trailing exit fires first).

    Running-extreme tracking: `self._trail_extreme` is updated each bar (on
    the last CLOSED High/Low, i.e. data.High[-1]/data.Low[-1]) while a
    position is open, seeded from the entry bar's High/Low. Because it only
    ever reads `[-1]` (the bar that just closed) and is updated once per
    `next()` call before being used to compute the trail/exit decision for
    *this* bar, there is no look-ahead — the exit check for bar i uses the
    running extreme through bar i, never bar i+1.
    """

    st_trail_atr: float = 3.0
    st_adx_min: float = 0  # 0 = gate off
    st_adx_period: int = 14

    def init(self) -> None:
        super().init()
        self._trail_extreme: float | None = None

    def next(self) -> None:
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        low_v = self.data.Low[-1]
        direction = self.data.STDir[-1]
        atr_v = self.data.STAtr[-1]
        adx_v = self.data.STADX[-1]

        if any(v is None or not np.isfinite(v) for v in (direction, atr_v, adx_v)):
            return
        if len(self.data) < 2:
            return
        prev_direction = self.data.STDir[-2]
        if prev_direction is None or not np.isfinite(prev_direction):
            return

        if self.position:
            # Update the running extreme with this closed bar, then check the
            # chandelier trail (and the opposite STDir flip) for an exit.
            if self.position.is_long:
                self._trail_extreme = max(self._trail_extreme, high_v)
                trail = self._trail_extreme - self.st_trail_atr * atr_v
                if direction == -1.0 or close_v < trail:
                    self.position.close()
                    self._entry_bar = None
                    self._trail_extreme = None
            elif self.position.is_short:
                self._trail_extreme = min(self._trail_extreme, low_v)
                trail = self._trail_extreme + self.st_trail_atr * atr_v
                if direction == 1.0 or close_v > trail:
                    self.position.close()
                    self._entry_bar = None
                    self._trail_extreme = None
            return

        flipped_long = prev_direction == -1.0 and direction == 1.0
        flipped_short = prev_direction == 1.0 and direction == -1.0

        if adx_v < self.st_adx_min:
            return

        sl_dist = self.st_sl_atr * atr_v

        # No `tp=` here — the chandelier trailing stop (tracked via
        # self._trail_extreme above) replaces the fixed ATR take-profit.
        if flipped_long:
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._trail_extreme = high_v
        elif self.allow_shorts and flipped_short:
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._trail_extreme = low_v
