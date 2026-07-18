"""KCSqueezeBreakout — Bollinger/Keltner squeeze → directional breakout on 1H BTC.

Hypothesis (pinned, see TODO_LEG): when Bollinger Bands close *inside* Keltner
Channels for >= squeeze_min_bars consecutive 1H bars, realised volatility has
been suppressed. The breakout that follows tends to be outsized in the
direction of the prevailing trend (linreg slope of close).

Why 1H (not 15m / not 4H)
-------------------------
Pinned by task spec:
  - 15m: too noisy, micro-squeezes fire constantly; correlates with v1
    chop-fade.
  - 4H: shares the clock with donchian-v3 → corr > 0.40 expected; also too
    sparse (~9k bars 2020-2026).
  - 1H: ~36k bars (sufficient sample for PSR), orthogonal to v1 (15m) and
    donchian-v3 (4H). Complements the 4H trend leg by reacting to the
    SAME breakout faster.

Data path
---------
The strategy receives the same 15m parquet that every other strategy uses
(so the runner/harness machinery is unchanged), and internally resamples
to 1H right-aligned bars (label='right', closed='right' on the standard
Pandas '1h' rule, so the 1H close at HH:00:00 represents the just-closed
hour [HH-1:00, HH:00)).

For each 15m bar `i` we look up the 1H row corresponding to the most
recent FULLY CLOSED 1H candle (shift(1) on the 1H frame) and ffill onto
the 15m grid. Entry decisions are sampled ONCE per 1H boundary (the 15m
bar with minute==0). Exits (SL / re-squeeze) are evaluated every 15m bar
for responsiveness — strictly more conservative than the 1H clock.

Signal (per the pinned params)
------------------------------
On each 1H close, using 1H OHLCV with all indicators causal (shift(1)
already applied at alignment):

  BB(bb_period=20, bb_k=2)         on Close
  KC(kc_period=20, kc_atr_mult=1.5) on (High, Low, Close, ATR(atr_period=14))
  squeeze_state = (BB_upper < KC_upper) AND (BB_lower > KC_lower)
  squeeze_run   = consecutive in-squeeze 1H bars ending at the prior bar
  direction     = sign(linreg_slope(Close, lr_slope_window=20))
  volume_ok     = Volume_prev > volume_mult * SMA(Volume, 20)_prev

Entries (at 1H close boundary, no current position):
  LONG  := prev_bar in-squeeze (run >= squeeze_min_bars)
           AND current Close > BB_upper_prev (breakout above BB)
           AND direction > 0
           AND volume_ok
  SHORT := prev_bar in-squeeze (run >= squeeze_min_bars)
           AND current Close < BB_lower_prev (breakout below BB)
           AND direction < 0
           AND volume_ok

On entry:
  initial SL = entry +/- atr_sl_mult * ATR_prev  (opposite side)
  Trail ARM: when price has moved +1 * ATR favourable
    (min_move_to_arm_trail_atr), switch SL to a ratcheted trail at
    atr_trail_mult * ATR distance.
  Exit: SL fill OR a new squeeze forms (`exit_on_resqueeze=True`).

Sizing (repo convention; risk-based, leverage-capped):
  risk_per_trade_pct = 0.5 %   (pinned)
  leverage           = 5       (pinned)

Costs / harness
---------------
commission_rt = 0.00075 → 7.5 bps per side (15 bps round-trip) is passed
to the Backtest object by the runner. trade_on_close=False matches multi-
factor-v1 / AdaptiveTrendV1: signals fire at bar `i`, fill at bar `i+1`'s
open.

Fractional sizing under PRICE_SCALE=0.001 is handled at the HARNESS
level (see tools/_postfrac_kc_squeeze.py): integer units returned by
_position_units are interpreted as 0.001 BTC under price scaling, matching
Binance USDT-M perp qty_step.

Authority: research-only. NOT wired to bot.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr as wilder_atr
from strategy.indicators import bollinger_bands, keltner_channel, sma


_H1_RULE = "1h"


def _resample_h1(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample 15m OHLCV to 1H right-aligned bars (matches Pandas .resample('1h')).

    Returns a frame indexed by the 1H close timestamps with columns
    [Open, High, Low, Close, Volume].
    """
    o = df_15m["Open"].resample(_H1_RULE, label="right", closed="right").first()
    h = df_15m["High"].resample(_H1_RULE, label="right", closed="right").max()
    lo = df_15m["Low"].resample(_H1_RULE, label="right", closed="right").min()
    c = df_15m["Close"].resample(_H1_RULE, label="right", closed="right").last()
    v = df_15m["Volume"].resample(_H1_RULE, label="right", closed="right").sum()
    h1 = pd.concat({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": v}, axis=1)
    return h1.dropna()


def _rolling_linreg_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope over `window` bars (x = 0..window-1).

    Returns a Series aligned to `series.index` with the slope at bar t
    computed from the most recent `window` observations ENDING at t
    (inclusive). The shift to causal use happens at the call site.

    Closed-form rolling slope:
        slope = ( N * sum(x*y) - sum(x) * sum(y) )
                / ( N * sum(x^2) - sum(x)^2 )
    where x = [0,1,...,N-1] inside each window. N, sum(x), sum(x^2) are
    constants — so only sum(y) and sum(x*y) need to roll.
    """
    if window <= 1:
        raise ValueError("lr_slope_window must be > 1")

    n = window
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x

    y = series.astype(float)
    # sum(y) over rolling window
    sum_y = y.rolling(window=n, min_periods=n).sum()
    # sum(x*y) -- weight earliest bar in the window with x=0, latest with x=n-1.
    # The trick: for any window ending at t, sum_{k=0}^{n-1} k * y_{t-(n-1)+k}.
    # Equivalent to: convolve y with kernel x reversed. Use rolling.apply for
    # clarity; n=20 so cost is trivial vs the warmup-only one-time call.
    sum_xy = y.rolling(window=n, min_periods=n).apply(
        lambda w: float(np.dot(x, w)), raw=True
    )
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope


class KCSqueezeBreakout(Strategy):
    """BB/KC squeeze → linreg-direction breakout on 1H BTC perp.

    Receives 15m bars (carrier), resamples internally to 1H, evaluates
    entries at 1H close boundaries, manages exits every 15m bar.
    """

    # --- BB / KC parameters (PINNED) ---
    bb_period: int = 20
    bb_k: float = 2.0
    kc_period: int = 20
    kc_atr_mult: float = 1.5

    # --- squeeze gate (PINNED) ---
    squeeze_min_bars: int = 10

    # --- direction filter (PINNED) ---
    lr_slope_window: int = 20

    # --- volume confirm (PINNED) ---
    volume_mult: float = 1.5
    volume_ma_period: int = 20

    # --- ATR stops/trail (PINNED) ---
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_trail_mult: float = 2.0
    min_move_to_arm_trail_atr: float = 1.0

    # --- exit (PINNED) ---
    exit_on_resqueeze: bool = True

    # --- direction control (PINNED) ---
    allow_shorts: bool = True

    # --- sizing (PINNED — risk-based, leverage-capped) ---
    risk_per_trade_pct: float = 0.5
    leverage: int = 5

    # --- safety belt: hard time-stop (NOT in pinned params; defensive).
    # Set high enough never to bind for a normal post-squeeze move; only
    # there to prevent pathological no-exit holds. 1H clock × 30d = 720 bars
    # → 720 * 4 = 2880 15m bars.
    max_hold_15m_bars: int = 2880

    # ------------------------------------------------------------------ init

    def init(self) -> None:
        df_15m = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
                "Volume": self.data.Volume,
            },
            index=self.data.index,
        )

        # 1. Resample 15m → 1H.
        h1 = _resample_h1(df_15m)

        close_h1 = h1["Close"]
        high_h1 = h1["High"]
        low_h1 = h1["Low"]
        vol_h1 = h1["Volume"]

        # 2. BB(bb_period, bb_k) on 1H closes.
        bb_up, _bb_mid, bb_lo = bollinger_bands(close_h1, self.bb_period, self.bb_k)

        # 3. KC(kc_period, kc_atr_mult) using ATR(atr_period) on 1H.
        #    Note: kc_period drives BOTH the EMA midline AND the ATR period of
        #    the KC bands; the pinned param dict gives a single kc_period and
        #    kc_atr_mult so we use kc_period for both, matching the canonical
        #    LazyBear KC squeeze formulation.
        kc_up, _kc_mid, kc_lo = keltner_channel(
            high_h1, low_h1, close_h1,
            ema_period=self.kc_period,
            atr_period=self.kc_period,
            mult=self.kc_atr_mult,
        )

        # 4. Wilder ATR for stops/trail (atr_period bars on 1H).
        atr_h1 = wilder_atr(high_h1, low_h1, close_h1, self.atr_period)

        # 5. Linear-regression slope of close over lr_slope_window 1H bars.
        slope_h1 = _rolling_linreg_slope(close_h1, self.lr_slope_window)

        # 6. Volume SMA on 1H (volume_ma_period = 20 per pin).
        vol_ma_h1 = sma(vol_h1, self.volume_ma_period)

        # 7. Squeeze state and consecutive-run counter on 1H.
        in_sq_raw = (
            (bb_up < kc_up) & (bb_lo > kc_lo)
            & np.isfinite(bb_up) & np.isfinite(kc_up)
            & np.isfinite(bb_lo) & np.isfinite(kc_lo)
        )
        in_sq_arr = in_sq_raw.fillna(False).values
        run_counter = np.zeros(len(in_sq_arr), dtype=int)
        run = 0
        for k, v in enumerate(in_sq_arr):
            run = run + 1 if v else 0
            run_counter[k] = run
        squeeze_run = pd.Series(run_counter, index=close_h1.index)

        # 8. Shift(1) everything → strictly causal: at 1H bar t we use values
        #    that reflect bars t-1 and prior (i.e. the PRIOR bar's squeeze run,
        #    BB bands, direction, volume — current Close is the breakout test).
        bb_up_prev = bb_up.shift(1)
        bb_lo_prev = bb_lo.shift(1)
        squeeze_run_prev = squeeze_run.shift(1)
        in_sq_prev = in_sq_raw.astype(float).shift(1)
        slope_prev = slope_h1.shift(1)
        atr_prev = atr_h1.shift(1)
        vol_prev = vol_h1.shift(1)
        vol_ma_prev = vol_ma_h1.shift(1)
        # in_sq AT 1H bar t (after the bar closes) — used for re-squeeze exit
        # check. Sampled at the next 1H boundary onto the 15m grid.
        in_sq_curr = in_sq_raw.astype(float)

        # Bundle into one frame so reindex/ffill align consistently.
        h1_aligned_src = pd.concat(
            {
                "bb_up_prev":       bb_up_prev,
                "bb_lo_prev":       bb_lo_prev,
                "squeeze_run_prev": squeeze_run_prev,
                "in_sq_prev":       in_sq_prev,
                "in_sq_curr":       in_sq_curr,
                "slope_prev":       slope_prev,
                "atr_prev":         atr_prev,
                "vol_prev":         vol_prev,
                "vol_ma_prev":      vol_ma_prev,
            },
            axis=1,
        )

        # 9. Reindex onto the 15m grid via ffill — the 1H values persist until
        #    the next 1H boundary closes.
        aligned = h1_aligned_src.reindex(df_15m.index, method="ffill")

        self._bb_up_prev = aligned["bb_up_prev"].values
        self._bb_lo_prev = aligned["bb_lo_prev"].values
        self._squeeze_run_prev = aligned["squeeze_run_prev"].values
        self._in_sq_prev = aligned["in_sq_prev"].values
        self._in_sq_curr = aligned["in_sq_curr"].values
        self._slope_prev = aligned["slope_prev"].values
        self._atr_prev = aligned["atr_prev"].values
        self._vol_prev = aligned["vol_prev"].values
        self._vol_ma_prev = aligned["vol_ma_prev"].values

        self._index = df_15m.index

        # Per-trade state
        self._entry_bar: int | None = None
        self._entry_price: float | None = None
        self._entry_atr: float | None = None
        self._trail_armed: bool = False
        self._trail_level: float | None = None
        # Track the 1H boundary on which we last evaluated an entry so we
        # only act on 1H close events, not every 15m bar.
        self._last_h1_close_seen: pd.Timestamp | None = None

    # ------------------------------------------------------------------ sizing

    def _position_units(self, price: float, sl_distance: float) -> int:
        # backtesting.py 0.6.5 only accepts integer units. Under HARNESS-level
        # price scaling (PRICE_SCALE=0.001 in the runner), 1 returned unit
        # == 0.001 BTC, matching Binance USDT-M perp qty_step.
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    # ------------------------------------------------------------------ helpers

    def _is_h1_close_bar(self, ts: pd.Timestamp) -> bool:
        """True if this 15m bar IS a 1H close boundary (HH:00 UTC).

        Right-aligned 1H resample puts the close at minute=0 of the next
        hour. At 15m granularity that's exactly the bar with minute == 0.
        """
        return ts.minute == 0

    def _volume_ok(self, i: int) -> bool:
        vp = self._vol_prev[i]
        vm = self._vol_ma_prev[i]
        if not np.isfinite(vp) or not np.isfinite(vm) or vm <= 0:
            return False
        return vp > self.volume_mult * vm

    # ------------------------------------------------------------------ loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]

        # --- Position management (evaluated every 15m bar) ---
        if self.position:
            atr_v = self._atr_prev[i]
            # Hard time-stop (defensive belt; not in pinned params).
            if (
                self._entry_bar is not None
                and (i - self._entry_bar) >= self.max_hold_15m_bars
            ):
                self.position.close()
                self._reset_trade_state()
                return

            trade = self.trades[-1] if self.trades else None
            if trade is None:
                return

            # Re-squeeze exit (pinned: exit_on_resqueeze=True).
            if self.exit_on_resqueeze:
                in_sq_now = self._in_sq_curr[i]
                if np.isfinite(in_sq_now) and in_sq_now > 0.5:
                    self.position.close()
                    self._reset_trade_state()
                    return

            # Trailing-stop logic: ARM after price has moved
            # min_move_to_arm_trail_atr * ATR_prev favourable; then ratchet
            # at atr_trail_mult * ATR_prev distance.
            if (
                np.isfinite(atr_v) and atr_v > 0
                and self._entry_price is not None
            ):
                entry_px = self._entry_price
                if trade.is_long:
                    favourable = close_v - entry_px
                    if (
                        not self._trail_armed
                        and favourable >= self.min_move_to_arm_trail_atr * atr_v
                    ):
                        self._trail_armed = True
                    if self._trail_armed:
                        candidate = close_v - self.atr_trail_mult * atr_v
                        if self._trail_level is None or candidate > self._trail_level:
                            self._trail_level = candidate
                        if trade.sl is None or self._trail_level > trade.sl:
                            trade.sl = self._trail_level
                        if close_v < self._trail_level:
                            self.position.close()
                            self._reset_trade_state()
                            return
                else:
                    favourable = entry_px - close_v
                    if (
                        not self._trail_armed
                        and favourable >= self.min_move_to_arm_trail_atr * atr_v
                    ):
                        self._trail_armed = True
                    if self._trail_armed:
                        candidate = close_v + self.atr_trail_mult * atr_v
                        if self._trail_level is None or candidate < self._trail_level:
                            self._trail_level = candidate
                        if trade.sl is None or self._trail_level < trade.sl:
                            trade.sl = self._trail_level
                        if close_v > self._trail_level:
                            self.position.close()
                            self._reset_trade_state()
                            return
            return

        # --- Entry: only at 1H close boundaries, once per boundary ---
        if not self._is_h1_close_bar(ts):
            return
        if self._last_h1_close_seen == ts:
            return
        self._last_h1_close_seen = ts

        bb_up = self._bb_up_prev[i]
        bb_lo = self._bb_lo_prev[i]
        sq_run = self._squeeze_run_prev[i]
        in_sq_p = self._in_sq_prev[i]
        slope_v = self._slope_prev[i]
        atr_v = self._atr_prev[i]

        # Warmup / NaN safety: every input must be valid.
        if not (
            np.isfinite(bb_up)
            and np.isfinite(bb_lo)
            and np.isfinite(sq_run)
            and np.isfinite(in_sq_p)
            and np.isfinite(slope_v)
            and np.isfinite(atr_v)
            and atr_v > 0
        ):
            return

        # Squeeze gate: PRIOR 1H bar was in-squeeze AND prior consecutive run
        # >= squeeze_min_bars.
        if not (in_sq_p > 0.5 and sq_run >= self.squeeze_min_bars):
            return

        # Volume confirmation.
        if not self._volume_ok(i):
            return

        # Sizing.
        sl_dist = self.atr_sl_mult * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        # Long breakout: close > prior BB upper AND slope > 0.
        if close_v > bb_up and slope_v > 0:
            self.buy(size=units, sl=close_v - sl_dist)
            self._entry_bar = i
            self._entry_price = close_v
            self._entry_atr = atr_v
            self._trail_armed = False
            self._trail_level = close_v - sl_dist
            return

        # Short breakout: close < prior BB lower AND slope < 0.
        if self.allow_shorts and close_v < bb_lo and slope_v < 0:
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._entry_price = close_v
            self._entry_atr = atr_v
            self._trail_armed = False
            self._trail_level = close_v + sl_dist

    # ------------------------------------------------------------------ state

    def _reset_trade_state(self) -> None:
        self._entry_bar = None
        self._entry_price = None
        self._entry_atr = None
        self._trail_armed = False
        self._trail_level = None
