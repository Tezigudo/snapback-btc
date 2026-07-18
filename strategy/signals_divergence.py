"""
DivergenceV1 — RSI + OBV divergence strategy on 15m.

Research basis (DIVERGENCE_PLAN.md, authored 2026-06-01):
  - RSI divergence: canonical momentum-exhaustion signal — price makes a new
    extreme while RSI fails to confirm, signalling trend hollowness.
  - OBV divergence: same geometry on cumulative signed volume — when BOTH
    RSI and OBV diverge the same direction simultaneously, price + momentum
    + volume all agree the move is exhausted. Substantially stronger than
    RSI alone (which fires on minor wiggles) and replaces a crude volume-SMA
    gate with a directional flow check.
  - v1 ships regular bullish + bearish only. Hidden divergences are a
    continuation pattern requiring an explicit trend regime; deferred to v2.

Lookahead safety (critical — see DIVERGENCE_PLAN.md §Lookahead):
  swing_high_low() is a centred k-bar fractal: a swing at bar i is not
  knowable until bar i+k. find_divergence() bakes in the +k shift internally,
  firing exactly once at bar b2+k (the confirmation bar of the more-recent
  swing). Precomputing find_divergence() in init() and indexing by i in next()
  is therefore safe: the value at index i depends only on data ≤ i. Do NOT
  re-shift the output.

Indicator toggles:
  use_rsi_divergence  (default True)  — include RSI divergence in AND-gate
  use_obv_divergence  (default True)  — include OBV divergence in AND-gate (now uses OBV slope, not level)
  use_macd_divergence (default False) — include MACD-histogram divergence in AND-gate
  At least one toggle must be True (validated in init()).

Entry rules — ALL enabled divergence indicators must fire, PLUS:
  3. rsi[b2] < rsi_oversold_zone  (divergence is in the oversold zone)
     b2 = i - k (the actual swing bar that just registered)
     Note: this gate applies even when use_rsi_divergence is False, because
     the zone filter is a price-regime check, not an indicator-specific check.
  4. close[i] > max(high[b2:i]) + 0.25 * atr[i]
     (close cleared the recovery-window high (b2 to i-1, exclusive of bar i)
     plus a fractional-ATR buffer; high[i] is excluded because close[i] ≤ high[i])
  5. Trend filter (if enabled): close[i] > ema_trend[i]
  6. ATR/close ratio veto: if atr[i] / close[i] > atr_close_ratio_veto, skip entry
     (extreme realized vol — cascade-veto from FUTURE_DIRECTIONS Bug 4)

SHORT = mirror: enabled divergence signals + rsi[b2] > rsi_overbought_zone
        + close < min(low[b2:i+1]) - 0.25*atr[i] + (if enabled) close < ema_trend.

Fix 1 (Phase-5): OBV divergence gate now checks the windowed OBV SLOPE across
  b1..b2 via linear regression, not the cumulative OBV level at b1 vs b2.
  Bullish: slope > 0 (accumulation despite price LL). Bearish: slope < 0.
  This is computed inline in _long_signal/_short_signal using the swing-pair
  indices recovered from the precomputed swing mask.

Fix 2 (Phase-5): Confirmation gate strengthened from close > high[b2] to
  close > max(high[b2:i+1]) + 0.25*atr[i]. Option (a) from FUTURE_DIRECTIONS.
  Rationale: option (a) is implementable without a secondary swing-high scan;
  it closes the trivially-satisfied loophole while remaining parameter-light.

Fix 3 (Phase-5): Live-safe defaults — trend_filter_enabled=True, leverage=5,
  rsi_oversold_zone=30, rsi_overbought_zone=70, plus atr_close_ratio_veto=0.015.

Exit rules:
  - Initial SL: entry ± sl_atr_multiple × ATR(14)
  - Take profit: entry ± tp_atr_multiple × ATR(14)
  - Time stop: position closed after max_hold_bars regardless
  - No trailing stop in v1 (adds variance that obscures signal quality)

Authority: DIVERGENCE_PLAN.md is the design contract. Phase 2 smoke-check
only; not wired to bot.py, not in live deploy.

Phase-2 design choices and deviations (flagged for orchestrator):
  - DIVERGENCE_PLAN.md condition 7 says current_bar > b2+k (strict >), but
    find_divergence() fires at exactly b2+k and is non-sticky. This
    implementation enters at bar b2+k (i.e. the confirmation bar itself).
    Strict > would mean the signal always fires one bar too early relative
    to itself — i.e. never fires at all. This is the Phase-2 simplification;
    Phase 5's live evaluator can add a one-bar delay gate if desired.
  - SL floor (0.1% below swing low) mentioned in DIVERGENCE_PLAN.md §Exits
    is omitted per the brief's "fixed ATR multiples only" instruction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import (
    atr,
    ema,
    find_divergence,
    macd,
    obv,
    rsi,
    swing_high_low,
)


class DivergenceV1(Strategy):
    # --- swing detection ---
    swing_k: int = 3
    min_swing_separation_bars: int = 5
    max_swing_separation_bars: int = 60

    # --- RSI divergence ---
    rsi_period: int = 14
    rsi_oversold_zone: float = 30.0    # Fix 3: tightened from 35 → 30 (live-safe)
    rsi_overbought_zone: float = 70.0  # Fix 3: tightened from 65 → 70 (live-safe)

    # --- exits ---
    atr_period: int = 14
    sl_atr_multiple: float = 1.5
    tp_atr_multiple: float = 4.5
    max_hold_bars: int = 96        # 24 h on 15m

    # --- sizing ---
    risk_per_trade_pct: float = 1.0
    leverage: int = 5              # Fix 3: lowered from 20 → 5 (live-safe)
    allow_shorts: bool = True

    # --- indicator toggles (at least one must be True) ---
    use_rsi_divergence: bool = True
    use_obv_divergence: bool = True
    use_macd_divergence: bool = False

    # --- MACD params (used only when use_macd_divergence=True) ---
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- trend filter (Fix 3: default ON — live-safe) ---
    trend_filter_enabled: bool = True  # Fix 3: was False
    trend_ema_period: int = 200

    # --- cascade veto (Fix 3 / FUTURE_DIRECTIONS Bug 4) ---
    atr_close_ratio_veto: float = 0.015  # skip entry when ATR/close > this (extreme vol)

    # --- Fix 2 toggle (for ablation testing only; default True = strengthened confirmation) ---
    # When False, reverts to original close > high[b2] / close < low[b2] check.
    strengthened_confirmation: bool = True

    def init(self) -> None:
        if not (self.use_rsi_divergence or self.use_obv_divergence or self.use_macd_divergence):
            raise ValueError(
                "DivergenceV1: at least one of use_rsi_divergence, "
                "use_obv_divergence, use_macd_divergence must be True"
            )

        close  = pd.Series(self.data.Close)
        high   = pd.Series(self.data.High)
        low    = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self._close_arr = close.values
        self._high_arr  = high.values
        self._low_arr   = low.values

        # --- core indicators ---
        self._rsi = rsi(close, self.rsi_period).values
        self._atr = atr(high, low, close, self.atr_period).values

        # --- OBV raw values (Fix 1: used for windowed slope check inline) ---
        self._obv_arr = obv(close, volume).values

        # --- trend EMA (only consumed when trend_filter_enabled) ---
        if self.trend_filter_enabled:
            self._ema_trend = ema(close, self.trend_ema_period).values
        else:
            self._ema_trend = None

        # --- swing masks (unshifted — find_divergence shifts internally) ---
        swing_highs, swing_lows = swing_high_low(high, low, k=self.swing_k)

        # Store swing positions for inline OBV slope check (Fix 1).
        # swing_low_positions[j] = the most-recent swing-low bar index b1 that
        # pairs with b2 = j - k at firing bar j, or -1 if none.
        n = len(close)
        self._swing_low_b1 = np.full(n, -1, dtype=np.intp)   # b1 for each firing bar j
        self._swing_low_b2 = np.full(n, -1, dtype=np.intp)   # b2 for each firing bar j
        self._swing_high_b1 = np.full(n, -1, dtype=np.intp)
        self._swing_high_b2 = np.full(n, -1, dtype=np.intp)

        sl_positions = np.where(swing_lows.values)[0]
        for pos in range(1, len(sl_positions)):
            b2 = sl_positions[pos]
            b1 = sl_positions[pos - 1]
            sep = b2 - b1
            if self.min_swing_separation_bars <= sep <= self.max_swing_separation_bars:
                j = b2 + self.swing_k
                if j < n:
                    self._swing_low_b1[j] = b1
                    self._swing_low_b2[j] = b2

        sh_positions = np.where(swing_highs.values)[0]
        for pos in range(1, len(sh_positions)):
            b2 = sh_positions[pos]
            b1 = sh_positions[pos - 1]
            sep = b2 - b1
            if self.min_swing_separation_bars <= sep <= self.max_swing_separation_bars:
                j = b2 + self.swing_k
                if j < n:
                    self._swing_high_b1[j] = b1
                    self._swing_high_b2[j] = b2

        # --- build AND-gate signals from enabled indicators ---
        # OBV divergence is no longer computed via find_divergence (Fix 1).
        # Instead the slope check runs inline in _long_signal/_short_signal.
        # We keep use_obv_divergence as a toggle for that inline check.
        all_true = pd.Series(True, index=close.index)
        long_sig  = all_true.copy()
        short_sig = all_true.copy()

        if self.use_rsi_divergence:
            rsi_s = pd.Series(self._rsi, index=close.index)
            long_sig  &= find_divergence(
                price=low, indicator=rsi_s, swing_mask=swing_lows,
                kind="regular_bullish", k=self.swing_k,
                min_separation=self.min_swing_separation_bars,
                max_separation=self.max_swing_separation_bars,
            )
            short_sig &= find_divergence(
                price=high, indicator=rsi_s, swing_mask=swing_highs,
                kind="regular_bearish", k=self.swing_k,
                min_separation=self.min_swing_separation_bars,
                max_separation=self.max_swing_separation_bars,
            )

        if self.use_macd_divergence:
            _, _, macd_hist = macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
            macd_s = pd.Series(macd_hist.values, index=close.index)
            long_sig  &= find_divergence(
                price=low, indicator=macd_s, swing_mask=swing_lows,
                kind="regular_bullish", k=self.swing_k,
                min_separation=self.min_swing_separation_bars,
                max_separation=self.max_swing_separation_bars,
            )
            short_sig &= find_divergence(
                price=high, indicator=macd_s, swing_mask=swing_highs,
                kind="regular_bearish", k=self.swing_k,
                min_separation=self.min_swing_separation_bars,
                max_separation=self.max_swing_separation_bars,
            )

        self._long_sig  = long_sig.values
        self._short_sig = short_sig.values

        self._entry_bar: int | None = None

    # ---------------------------------------------------------------- OBV slope

    def _obv_slope(self, b1: int, b2: int) -> float:
        """Linear regression slope of OBV over [b1, b2] inclusive.

        Returns the slope coefficient (price units per bar). A positive slope
        means OBV is trending up across the window (accumulation); negative
        means distribution. Used by Fix 1: replaces the near-tautological
        level comparison (obv[b2] > obv[b1]) with a directional flow check.
        """
        window = self._obv_arr[b1: b2 + 1]
        n = len(window)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        # Two-element polyfit equivalent (faster than np.polyfit for long windows):
        x_mean = (n - 1) / 2.0
        y_mean = window.mean()
        slope = np.dot(x - x_mean, window - y_mean) / np.dot(x - x_mean, x - x_mean)
        return float(slope)

    # ------------------------------------------------------------------ sizing

    def _position_units(self, price: float, sl_distance: float) -> int:
        # NOTE: backtesting.py 0.6.5 only accepts integer units (or 0<size<1
        # fraction-of-equity). Fractional BTC sizing is implemented at the
        # HARNESS layer via 0.001 price scaling — see tools/_fractional_run.py.
        # When run under that scaling: 1 returned "unit" == 0.001 BTC,
        # which matches the live Binance USDT-M perp qty_step.
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    # ---------------------------------------------------------------- signals

    def _long_signal(self, i: int) -> bool:
        # Guard: need at least k bars of history so b2 = i-k >= 0
        if i < self.swing_k:
            return False
        if not self._long_sig[i]:
            return False

        b2 = i - self.swing_k  # actual swing bar that just registered

        # Oversold gate: RSI at the swing bar must be in the oversold zone
        rsi_b2 = self._rsi[b2]
        if not (np.isfinite(rsi_b2) and rsi_b2 < self.rsi_oversold_zone):
            return False

        # ATR available
        atr_i = self._atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            return False

        close_i = self._close_arr[i]

        # Fix 2: Strengthened confirmation (when strengthened_confirmation=True).
        # close must clear the highest high in the recovery window [b2, i)
        # (exclusive of bar i itself, since close[i] <= high[i] by definition)
        # plus a fractional-ATR buffer. Replaces the trivially-satisfied
        # close > high[b2] check (b2 is a swing low so high[b2] is structurally low).
        # When strengthened_confirmation=False: original close > high[b2] (ablation only).
        if self.strengthened_confirmation:
            if b2 < i:
                recovery_high = np.max(self._high_arr[b2: i])
            else:
                recovery_high = self._high_arr[b2]
            if not (np.isfinite(recovery_high) and close_i > recovery_high + 0.25 * atr_i):
                return False
        else:
            high_b2 = self._high_arr[b2]
            if not (np.isfinite(high_b2) and close_i > high_b2):
                return False

        # Fix 1: OBV slope gate — require positive slope across b1..b2 window.
        # (replaces the near-tautological cumulative OBV level comparison)
        if self.use_obv_divergence:
            b1 = int(self._swing_low_b1[i])
            b2_stored = int(self._swing_low_b2[i])
            if b1 < 0 or b2_stored < 0:
                return False  # no valid swing pair recorded for this bar
            slope = self._obv_slope(b1, b2_stored)
            if slope <= 0:
                return False  # OBV trending down: distribution, not accumulation

        # Trend filter
        if self.trend_filter_enabled and self._ema_trend is not None:
            t = self._ema_trend[i]
            if not (np.isfinite(t) and close_i > t):
                return False

        return True

    def _short_signal(self, i: int) -> bool:
        if not self.allow_shorts:
            return False
        if i < self.swing_k:
            return False
        if not self._short_sig[i]:
            return False

        b2 = i - self.swing_k

        # Overbought gate
        rsi_b2 = self._rsi[b2]
        if not (np.isfinite(rsi_b2) and rsi_b2 > self.rsi_overbought_zone):
            return False

        # ATR available
        atr_i = self._atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            return False

        close_i = self._close_arr[i]

        # Fix 2 (short mirror): when strengthened_confirmation=True, close must
        # be below the lowest low in the recovery window [b2, i) minus a buffer.
        if self.strengthened_confirmation:
            if b2 < i:
                recovery_low = np.min(self._low_arr[b2: i])
            else:
                recovery_low = self._low_arr[b2]
            if not (np.isfinite(recovery_low) and close_i < recovery_low - 0.25 * atr_i):
                return False
        else:
            low_b2 = self._low_arr[b2]
            if not (np.isfinite(low_b2) and close_i < low_b2):
                return False

        # Fix 1 (short mirror): OBV slope < 0 across b1..b2 (distribution)
        if self.use_obv_divergence:
            b1 = int(self._swing_high_b1[i])
            b2_stored = int(self._swing_high_b2[i])
            if b1 < 0 or b2_stored < 0:
                return False
            slope = self._obv_slope(b1, b2_stored)
            if slope >= 0:
                return False  # OBV trending up: accumulation, not distribution

        # Trend filter
        if self.trend_filter_enabled and self._ema_trend is not None:
            t = self._ema_trend[i]
            if not (np.isfinite(t) and close_i < t):
                return False

        return True

    # ---------------------------------------------------------------- loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Position management: time stop
        if self.position:
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold_bars:
                self.position.close()
                self._entry_bar = None
            return

        # Entry
        atr_v = self._atr[i]
        if not np.isfinite(atr_v) or atr_v <= 0:
            return

        # Fix 3 / FUTURE_DIRECTIONS Bug 4: cascade veto — skip entry when
        # realized vol is extreme (ATR/close > threshold).
        if close_v > 0 and (atr_v / close_v) > self.atr_close_ratio_veto:
            return

        sl_dist = self.sl_atr_multiple * atr_v
        tp_dist = self.tp_atr_multiple * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if self._long_signal(i):
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
            self._entry_bar = i
        elif self._short_signal(i):
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
            self._entry_bar = i


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd
    from backtesting import Backtest

    HERE = Path(__file__).resolve().parent.parent  # repo root
    PARQUET = HERE / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

    if not PARQUET.exists():
        print(f"ERROR: parquet not found at {PARQUET}")
        sys.exit(1)

    df = pd.read_parquet(PARQUET)

    # backtesting.py requires capitalised column names
    df = df.rename(columns={c: c.capitalize() for c in df.columns})

    # Slice the last 6 months
    cutoff = df.index[-1] - pd.Timedelta(days=180)
    df_slice = df[df.index >= cutoff].copy()

    # Strip timezone so backtesting.py doesn't complain
    if df_slice.index.tz is not None:
        df_slice.index = df_slice.index.tz_localize(None)

    print(f"Smoke check: {len(df_slice)} bars from {df_slice.index[0]} to {df_slice.index[-1]}")

    COMMISSION = 0.0005   # 0.04% taker + 0.01% slippage proxy
    CASH       = 1_000_000.0
    MARGIN     = 1.0 / 20  # leverage 20x

    bt = Backtest(
        df_slice,
        DivergenceV1,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()

    n_trades  = int(stats["# Trades"])
    ret_pct   = float(stats["Return [%]"])
    win_pct   = float(stats.get("Win Rate [%]") or 0.0)
    max_dd    = float(stats.get("Max. Drawdown [%]") or 0.0)

    print()
    print("=== DivergenceV1 smoke check (last 6 months, 15m BTC/USDT) ===")
    print(f"  Trades          : {n_trades}")
    print(f"  Total return    : {ret_pct:+.2f}%")
    print(f"  Win rate        : {win_pct:.1f}%")
    print(f"  Max drawdown    : {max_dd:.2f}%")
