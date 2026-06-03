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

from strategy.indicators import atr, ema


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
    """v1 + ATR trailing stop and high-water-mark exit.

    The Donchian exit channel is *itself* a trailing mechanism, but it only
    updates once per 1h bar and only reflects the lowest of the last M
    closes. A separate ATR-distance trailing stop catches faster reversals
    that happen within a 1h bar (or that don't quite breach the channel
    but blow through the entry's risk envelope). Both exits coexist —
    whichever fires first wins.
    """

    donchian_period_entry = 20
    donchian_period_exit = 10
    atr_sl_multiple = 2.0      # initial stop distance
    atr_trail_multiple = 0.0   # 0 = no trailing; >0 = trail SL at high - K*ATR
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True
    # Max hold time. At 4h entry: 180 bars = 30 days = 1 month.
    # At 15m entry: would be 2880 bars. Default 180 = right for our 4h use.
    # Set 0 to disable.
    time_stop_bars = 180

    def init(self) -> None:
        self._entry_bar: int | None = None
        self._high_water: float = 0.0   # for long trailing
        self._low_water: float = 0.0    # for short trailing

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _maybe_trail(self, atr_v: float) -> None:
        """Ratchet the trade SL toward high_water - K*ATR (long) or low_water + K*ATR (short)."""
        if self.atr_trail_multiple <= 0 or not self.trades:
            return
        trade = self.trades[-1]
        if trade.is_long:
            new_sl = self._high_water - self.atr_trail_multiple * atr_v
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl
        else:
            new_sl = self._low_water + self.atr_trail_multiple * atr_v
            if trade.sl is None or new_sl < trade.sl:
                trade.sl = new_sl

    def next(self) -> None:
        upper = self.data.DonchianUpper[-1]
        lower = self.data.DonchianLower[-1]
        exit_upper = self.data.DonchianExitUpper[-1]
        exit_lower = self.data.DonchianExitLower[-1]
        atr_v = self.data.ATR_1h[-1]
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        low_v = self.data.Low[-1]

        if any(
            v is None or not np.isfinite(v)
            for v in (upper, lower, exit_upper, exit_lower, atr_v)
        ):
            return

        if self.position:
            if self.position.is_long:
                self._high_water = max(self._high_water, high_v)
            else:
                self._low_water = min(self._low_water if self._low_water > 0 else low_v, low_v)
            self._maybe_trail(atr_v)

            # Time stop — max-hold safety. Prevents indefinite positions when
            # the Donchian exit channel never fires (e.g. a sustained micro-trend
            # that doesn't pierce the opposite N-bar channel for weeks).
            if self.time_stop_bars > 0 and self._entry_bar is not None:
                if (len(self.data) - self._entry_bar) >= self.time_stop_bars:
                    self.position.close()
                    self._entry_bar = None
                    return

            # Donchian channel exit
            if self.position.is_long and close_v < exit_lower:
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and close_v > exit_upper:
                self.position.close()
                self._entry_bar = None
            return

        sl_dist = self.atr_sl_multiple * atr_v

        if close_v > upper:
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = high_v
                self._low_water = 0.0
        elif self.allow_shorts and close_v < lower:
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = 0.0
                self._low_water = low_v


class DonchianBreakoutBTCv2(DonchianBreakoutBTC):
    """Same behaviour as v1 — separate class so the sweep machinery can mutate
    class attributes without touching the v1 baseline. v2 enables ATR trailing
    via the wider sweep grid in `config/sweep_donchian_v2.yaml`."""


class DonchianBreakoutBTCv3(DonchianBreakoutBTC):
    """v2 + DIRECTIONAL regime gate.

    Uses signed EMA-slope: positive = uptrend, negative = downtrend.
    Entry rules:
      - LONG breakout requires slope >= +slope_trend_threshold_pct
      - SHORT breakout requires slope <= -slope_trend_threshold_pct
      - Chop (|slope| < threshold) = no entry in either direction

    Two reasons this is better than the |slope| gate:
      1. It refuses to short during an uptrend (and vice versa), which is
         the most common Donchian failure in chop — a small downtick that
         technically pierces the lower channel but is just a pullback in
         a larger uptrend.
      2. The |slope| gate's "high threshold blocks early-trend entries"
         pathology is avoided — we don't gate on magnitude alone.

    slope_trend_threshold_pct=0 disables the gate (v2 behaviour)."""

    regime_ema_period: int = 120
    regime_slope_window: int = 30
    slope_trend_threshold_pct: float = 0.0   # 0 = gate OFF

    # debt #3 variant E — opt-in EMA-direction binary filter
    # (Faber 2007 / AQR Trends Everywhere — replace magnitude with sign).
    # Default OFF so the locked baseline (slope-gate path) is byte-for-byte
    # unchanged. When enabled, slope_trend_threshold_pct should be 0 so the
    # two gates don't overlap.
    use_ema_direction_filter: bool = False
    ema_direction_period: int = 200

    # debt #3 placeholder — all variants pin this to 0; declared so that
    # bt.run(**variant_config) doesn't AttributeError. No-op when 0.
    # Reserved for a follow-up sweep that adds an N×ATR buffer on top of
    # the Donchian breakout (Wilder breakout convention).
    atr_breakout_buffer_mult: float = 0.0

    def init(self) -> None:
        super().init()
        if self.slope_trend_threshold_pct > 0:
            import pandas as pd_
            from strategy.regime_classifier import ema_slope_signed
            close = pd_.Series(self.data.Close)
            self._regime_slope = ema_slope_signed(
                close,
                ema_period=self.regime_ema_period,
                slope_window=self.regime_slope_window,
            ).values
        else:
            self._regime_slope = None

        if self.use_ema_direction_filter:
            import pandas as pd_
            from strategy.indicators import ema as _ema
            close = pd_.Series(self.data.Close)
            self._direction_ema = _ema(close, self.ema_direction_period).values
        else:
            self._direction_ema = None

    def _slope_now(self) -> float | None:
        if self._regime_slope is None:
            return None
        s = self._regime_slope[len(self.data) - 1]
        import numpy as _np
        return float(s) if _np.isfinite(s) else None

    def next(self) -> None:
        upper = self.data.DonchianUpper[-1]
        lower = self.data.DonchianLower[-1]
        exit_upper = self.data.DonchianExitUpper[-1]
        exit_lower = self.data.DonchianExitLower[-1]
        atr_v = self.data.ATR_1h[-1]
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        low_v = self.data.Low[-1]

        import numpy as _np
        if any(
            v is None or not _np.isfinite(v)
            for v in (upper, lower, exit_upper, exit_lower, atr_v)
        ):
            return

        if self.position:
            if self.position.is_long:
                self._high_water = max(self._high_water, high_v)
            else:
                self._low_water = min(self._low_water if self._low_water > 0 else low_v, low_v)
            self._maybe_trail(atr_v)
            if self.position.is_long and close_v < exit_lower:
                self.position.close()
                self._entry_bar = None
            elif self.position.is_short and close_v > exit_upper:
                self.position.close()
                self._entry_bar = None
            return

        sl_dist = self.atr_sl_multiple * atr_v
        # Optional ATR buffer added on top of the Donchian breakout level.
        # All five debt #3 variants pin this to 0 → no-op; kept declared so
        # bt.run(**config) doesn't AttributeError on the attribute name.
        buf = self.atr_breakout_buffer_mult * atr_v if self.atr_breakout_buffer_mult > 0 else 0.0
        slope = self._slope_now()
        gate_on = self.slope_trend_threshold_pct > 0

        # Variant E — close-vs-EMA(period) on the entry TF.
        # When use_ema_direction_filter=True, long requires close>EMA, short
        # requires close<EMA. Independent of slope gate; both may be on but
        # the debt #3 plan pins slope_trend_threshold_pct=0 for variant E.
        if self._direction_ema is not None:
            ema_now = self._direction_ema[len(self.data) - 1]
            if not _np.isfinite(ema_now):
                return
        else:
            ema_now = None

        if close_v > upper + buf:
            if gate_on and (slope is None or slope < self.slope_trend_threshold_pct):
                return  # don't long unless in confirmed uptrend
            if ema_now is not None and close_v <= ema_now:
                return  # variant E — refuse longs when entry TF is below EMA
            sl = close_v - sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = high_v
                self._low_water = 0.0
        elif self.allow_shorts and close_v < lower - buf:
            if gate_on and (slope is None or slope > -self.slope_trend_threshold_pct):
                return  # don't short unless in confirmed downtrend
            if ema_now is not None and close_v >= ema_now:
                return  # variant E — refuse shorts when entry TF is above EMA
            sl = close_v + sl_dist
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl > close_v:
                self.sell(size=units, sl=sl)
                self._entry_bar = len(self.data)
                self._high_water = 0.0
                self._low_water = low_v


# ---------------------------------------------------------------------------
# Donchian Trend-Rider v1  (native 4h, LONG-only, fixed TP bracket)
# ---------------------------------------------------------------------------

def attach_rider(
    df: pd.DataFrame,
    donchian_n: int = 55,
    atr_period: int = 14,
    ema_period: int = 200,
) -> pd.DataFrame:
    """Attach RiderDonHi / RiderEma / RiderAtr columns to a capitalised, tz-naive
    DataFrame (native 4h bars).

    RiderDonHi = rolling max of HIGH over `donchian_n` bars, shifted 1 (channel
    excludes the current bar — mirrors hiwr build_breakout and rider_port_validate).
    RiderEma   = EMA(close, ema_period).
    RiderAtr   = Wilder ATR(atr_period) on High/Low/Close.

    Prefixed names avoid any collision with the existing DonchianUpper/ATR_1h
    columns used by donchian-v3.
    """
    out = df.copy()
    out["RiderDonHi"] = out["High"].rolling(donchian_n, min_periods=donchian_n).max().shift(1)
    out["RiderEma"] = ema(out["Close"], ema_period)
    out["RiderAtr"] = atr(out["High"], out["Low"], out["Close"], atr_period)
    return out


class DonchianRiderV1(Strategy):
    """4h Donchian trend-rider: long-only, small ATR stop, large fixed ATR TP.

    Entry:    close > RiderDonHi  AND  close > RiderEma
    Stop:     sl = close_v - rider_sl_atr * ATR(14)   (anchored to signal-bar close)
    Target:   tp = close_v + rider_tp_atr * ATR(14)   (fixed bracket — no trailing exit)
    Trail:    optional chandelier (rider_trail_atr >= 5); default off (0.0)
    Time-stop: rider_time_stop_bars bars

    Sizing mirrors donchian-v3: risk_per_trade_pct of equity / sl_distance,
    capped at leverage * equity * 0.95 / price.

    Default leverage=3 — this strategy targets -29% maxDD at 3x; it is NOT a
    20x strategy regardless of the global leverage ceiling.
    """

    # All params are rider-prefixed so _apply_params_to_class cannot
    # accidentally clobber them with values from params.yaml.
    rider_sl_atr: float = 1.0
    rider_tp_atr: float = 8.0
    rider_trail_atr: float = 0.0       # 0 = no trail; >=5 = chandelier
    rider_time_stop_bars: int = 200
    rider_donchian_n: int = 55
    rider_atr_period: int = 14
    rider_ema_period: int = 200
    rider_risk_per_trade_pct: float = 2.0  # prefixed to avoid apply-list clobber
    # leverage IS set by run_backtest via STRATEGIES[name].leverage = eff_leverage
    leverage: int = 3

    def init(self) -> None:
        self._entry_bar: int | None = None
        self._high_water: float = 0.0

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.rider_risk_per_trade_pct / 100.0)
        target = risk_amount / sl_distance
        cap = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target, cap)), 0)

    def next(self) -> None:
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        don_hi = self.data.RiderDonHi[-1]
        ema_v = self.data.RiderEma[-1]
        atr_v = self.data.RiderAtr[-1]

        if any(v is None or not np.isfinite(v) for v in (don_hi, ema_v, atr_v)):
            return

        if self.position:
            if self.rider_time_stop_bars > 0 and self._entry_bar is not None:
                if (len(self.data) - self._entry_bar) >= self.rider_time_stop_bars:
                    self.position.close()
                    self._entry_bar = None
                    return
            if self.rider_trail_atr > 0 and self.trades:
                self._high_water = max(self._high_water, high_v)
                new_sl = self._high_water - self.rider_trail_atr * atr_v
                tr = self.trades[-1]
                if tr.sl is None or new_sl > tr.sl:
                    tr.sl = new_sl
            return

        if close_v > don_hi and close_v > ema_v:
            sl_dist = self.rider_sl_atr * atr_v
            sl = close_v - sl_dist
            tp = close_v + self.rider_tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
                self._high_water = high_v
