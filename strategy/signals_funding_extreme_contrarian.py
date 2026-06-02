"""
FundingExtremeContrarian — directional fade of extreme funding-rate prints.

Hypothesis (per TODO_LEG memo, BIS WP 1087 backed):
    When 8h perpetual funding rate prints at >=95th percentile of trailing
    90d distribution, the perp is structurally over-long. Take the short side.
    Symmetric on the <=5th percentile (take long).

Signal logic (built per spec):
    1. At each new 8h funding print, compute trailing 90d percentile of funding.
       - new_funding >= q95 over 270 prints (~90d) -> ARM_SHORT
       - new_funding <= q05 over 270 prints (~90d) -> ARM_LONG
    2. 4H EMA200 SLOPE filter (scale-invariant pct change over 10 bars):
       - ARM_SHORT only acted if slope < +0.05% per bar (flat or down)
       - ARM_LONG  only acted if slope > -0.05% per bar (flat or up)
    3. Entry trigger (first 15m bar after print where ALL true):
       - direction-matched engulfing pattern
       - Volume > 1.5 * SMA(Volume, 20)
    4. ARM EXPIRY: arm dies at the next funding print (8h / 32 bars at 15m).
       This prevents stale arms from firing days later in unrelated price
       structure.
    5. Stop = 1.5 * 1H ATR14, opposite of entry.
    6. Target = 2.5 * 1H ATR14 in entry direction.
    7. Time exit at next funding print after entry (whichever first).
    8. Min spacing 24h (96 bars at 15m) between entries.
    9. Max one position at a time.
   10. Risk per trade = 0.5% of equity.

Data inputs (attached by the runner / harness):
    - self.data.Close/High/Low/Open: 15m OHLC (price-SCALED by PRICE_SCALE)
    - self.data.Volume: 15m volume (un-scaled)
    - self.data.Funding: 8h funding rate forward-filled onto 15m (un-scaled
        ratio — same as multifactor harness convention)
    - self.data.FundArmShort: bool array, True at the 15m timestamp of the
        most-recently CLOSED 8h funding print whose value >= 90d q95.
        (Pre-computed in the runner on the NATIVE 8h series with rolling
        quantile — NOT on the forward-filled 15m series. Lookahead-safe via
        merge_asof(backward).)
    - self.data.FundArmLong: bool array, True at funding-print 15m bar when
        funding <= 90d q05.
    - self.data.FundPrintBar: bool array, True at every 15m bar that is the
        first one *at or after* an 8h funding print (used as time-exit and
        arm-expiry trigger).
    - self.data.AtrPriceScaled1h: float array, 1H ATR14 ALIGNED to 15m,
        already on the SCALED price plane (so distances match self.data.Close).
        Lookahead-safe (backward merge on bar-close timestamps).
    - self.data.Ema4hSlopePct: float array, 4H EMA200 slope expressed as
        percent change over the trailing 10 bars (scale-invariant).
        Aligned to 15m via backward merge.

CRITICAL: All aux arrays are pre-computed in the runner from the FULL parquet
history (warm-up before window start) then aligned to the 15m index via
merge_asof(direction="backward"). The strategy itself does not load files —
it only consumes columns attached to self.data.

PRICE_SCALE convention (per fractional sizing refactor):
    - Close / High / Low / Open and the 1H ATR distance live on the SCALED
      plane (real_price * 0.001). Position sizing in integer "units" then
      corresponds to 0.001 BTC each (matches Binance qty_step).
    - Volume, Funding, slope-pct, and engulfing booleans are dimensionless
      and stay un-scaled.

Search: funding rate extreme contrarian, fade carry decay, BIS 1087, percentile
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import bearish_engulfing, bullish_engulfing, sma


class FundingExtremeContrarian(Strategy):
    # --- arm window ---
    arm_expiry_bars = 32          # 8h / 15m = 32 bars; arm dies at next print
    min_spacing_bars = 96         # 24h between entries

    # --- volume confirmation ---
    volume_ma_period = 20
    volume_multiple = 1.5

    # --- 4H EMA200 slope filter ---
    require_slope_filter = True
    slope_flat_threshold_pct = 0.05   # |slope| < this counts as "flat"
                                      # short OK if slope < +threshold (flat or down)
                                      # long  OK if slope > -threshold (flat or up)

    # --- ATR-based stops/targets (1H ATR14, on the SCALED plane) ---
    atr_sl_mult = 1.5
    atr_tp_mult = 2.5

    # --- sizing ---
    risk_per_trade_pct = 0.5          # 0.5% per trade (spec)
    leverage = 20
    allow_shorts = True
    allow_longs = True

    # --- direction toggle (for ablation: test long-bias and short-bias separately) ---
    # Default: both legs on. Set to "long_only" / "short_only" for asymmetry test.
    direction_mode = "both"

    def init(self) -> None:
        open_ = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)

        self._bull_engulf = bullish_engulfing(open_, high, low, close).values
        self._bear_engulf = bearish_engulfing(open_, high, low, close).values
        self._vol_sma = sma(volume, self.volume_ma_period).values

        # Aux columns attached by the runner. We accept missing columns
        # gracefully (filled as no-op) for unit tests but in real runs they
        # MUST be present and well-formed.
        self._arm_short = np.asarray(getattr(self.data, "FundArmShort", np.zeros(len(self.data))), dtype=bool)
        self._arm_long = np.asarray(getattr(self.data, "FundArmLong", np.zeros(len(self.data))), dtype=bool)
        self._print_bar = np.asarray(getattr(self.data, "FundPrintBar", np.zeros(len(self.data))), dtype=bool)
        self._atr_1h_scaled = np.asarray(getattr(self.data, "AtrPriceScaled1h", np.full(len(self.data), np.nan)), dtype=float)
        self._ema4h_slope_pct = np.asarray(getattr(self.data, "Ema4hSlopePct", np.full(len(self.data), 0.0)), dtype=float)

        # Arm state: when a print fires an arm, it stays "live" for the next
        # `arm_expiry_bars` bars (i.e., until the next print).
        self._short_armed_until: int = -1
        self._long_armed_until: int = -1
        # Last entry bar (for min-spacing gate)
        self._last_entry_bar: int = -10_000
        # Bar at which the current open position must time-exit (next print)
        self._exit_at_print_after: int = -1

    # ----------------- helpers -----------------

    def _position_units(self, price: float, sl_distance: float) -> int:
        """Integer units. Under PRICE_SCALE=0.001 harness, 1 unit == 0.001 BTC.

        Identical convention to multifactor / adaptrend strategies.
        """
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_units = risk_amount / sl_distance
        max_units = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_units, max_units)), 0)

    def _slope_allows_short(self, i: int) -> bool:
        if not self.require_slope_filter:
            return True
        s = self._ema4h_slope_pct[i]
        if not np.isfinite(s):
            return False  # safer: deny if slope unknown
        return s < self.slope_flat_threshold_pct  # flat or down

    def _slope_allows_long(self, i: int) -> bool:
        if not self.require_slope_filter:
            return True
        s = self._ema4h_slope_pct[i]
        if not np.isfinite(s):
            return False
        return s > -self.slope_flat_threshold_pct  # flat or up

    def _volume_ok(self, i: int) -> bool:
        v = self.data.Volume[-1]
        sma_v = self._vol_sma[i]
        return bool(np.isfinite(sma_v) and v > self.volume_multiple * sma_v)

    # ----------------- main step -----------------

    def next(self) -> None:
        i = len(self.data) - 1

        # ARM REFRESH on funding-print bar.
        # NOTE: arm_short / arm_long indicate the print's funding fell in the
        # extreme tail of the 90d distribution. They are pre-computed.
        if self._print_bar[i]:
            if self._arm_short[i]:
                self._short_armed_until = i + self.arm_expiry_bars
            if self._arm_long[i]:
                self._long_armed_until = i + self.arm_expiry_bars

        # POSITION MGMT: time-exit at next funding print after entry
        if self.position:
            if self._exit_at_print_after >= 0 and self._print_bar[i] and i > self._exit_at_print_after:
                self.position.close()
                self._exit_at_print_after = -1
                return
            # No mid-bar logic — SL/TP are managed by backtesting.py
            return

        # ENTRY GUARDS
        if (i - self._last_entry_bar) < self.min_spacing_bars:
            return

        close_v = float(self.data.Close[-1])
        atr_v = self._atr_1h_scaled[i]
        if not (np.isfinite(atr_v) and atr_v > 0):
            return

        sl_dist = self.atr_sl_mult * atr_v
        tp_dist = self.atr_tp_mult * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        # SHORT ENTRY: armed and within window, slope allows, engulfing+volume
        if self.allow_shorts and self.direction_mode in ("both", "short_only"):
            if i <= self._short_armed_until and self._slope_allows_short(i):
                if bool(self._bear_engulf[i]) and self._volume_ok(i):
                    sl = close_v + sl_dist
                    tp = close_v - tp_dist
                    if tp > 0:
                        self.sell(size=units, sl=sl, tp=tp)
                        self._last_entry_bar = i
                        self._exit_at_print_after = i
                        # Consume arm so it doesn't re-fire
                        self._short_armed_until = -1
                        return

        # LONG ENTRY: armed and within window, slope allows, engulfing+volume
        if self.allow_longs and self.direction_mode in ("both", "long_only"):
            if i <= self._long_armed_until and self._slope_allows_long(i):
                if bool(self._bull_engulf[i]) and self._volume_ok(i):
                    sl = close_v - sl_dist
                    tp = close_v + tp_dist
                    if sl > 0:
                        self.buy(size=units, sl=sl, tp=tp)
                        self._last_entry_bar = i
                        self._exit_at_print_after = i
                        self._long_armed_until = -1
                        return
