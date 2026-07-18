"""
DivergenceV2 — research-redesign of DivergenceV1, applying the top-3 changes
from RESEARCH_PNL_FINDINGS.md (2026-06-02).

Changes vs v1:
  1. MFI(14) replaces cumulative OBV as volume indicator (RESEARCH §3).
     Bounded 0–100, RSI-like, no drift. Goes through find_divergence() directly —
     no slope workaround needed (the boundedness makes level comparison meaningful
     at swing pivots).

  2. 4H EMA200 regime gate (RESEARCH §2 — QuantPedia: Sharpe 0.33→1.07 on BTC).
     For longs: 15m close must be > 4H EMA200 at entry bar.
     For shorts: 15m close must be < 4H EMA200.
     LOOKAHEAD SAFETY (critical): the parquet index is bar-OPEN time. A 4H bar
     that opens at T has its close time at T+4h. We compute the 4H EMA on bar-open
     timestamps, then build a forward-fill Series whose INDEX is the bar-CLOSE time
     (open + 4h). When we merge_asof (direction="backward") against the 15m close
     timestamps, each 15m bar gets the EMA of the most-recently CLOSED 4H bar —
     never the bar whose 4H window is still open. This is the only lookahead-safe
     alignment for open-indexed parquets.

  3. Fix 2 option-b — proper swing-high confirmation (RESEARCH §4 note on option-b):
     Replace the v1 "close > recovery_window_high + ATR buffer" with:
     find the most-recent REGISTERED swing-high (from the precomputed shifted mask)
     at or before b2 (index h_recent); require close[j] > high[h_recent] AND
     close[j-1] > high[h_recent] (N=2 consecutive closes above the reference
     swing-high). Mirror for shorts.

  4. ATR compressed-regime skip (RESEARCH §5 / Nagel 2012):
     Reversal Sharpe INCREASES with vol; compressed-vol entries are the weak ones.
     Skip entry when atr[i] < 0.5 * mean(atr[i-50:i]).

  5. Inherits Fix 3 live-safe defaults from v1: trend_filter_enabled=True,
     leverage=5, rsi_oversold_zone=30, rsi_overbought_zone=70,
     atr_close_ratio_veto=0.015.

Lookahead safety:
  find_divergence() shifts the swing mask by +k internally — fires exactly at
  bar b2+k. Precomputing in init() and indexing by i in next() is safe.
  The 4H EMA gate uses bar-close timestamps with backward merge_asof — safe.

NOT changed from v1:
  - RSI divergence detection (same find_divergence path, AND-gated with MFI).
  - Swing fractal (k=3), separation params, exit logic (SL/TP/time stop).
  - signals_divergence.py (v1 preserved as historical artifact).

Authority: RESEARCH_PNL_FINDINGS.md + DIVERGENCE_PLAN.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import (
    atr,
    ema,
    find_divergence,
    mfi,
    rsi,
    swing_high_low,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_4H_PARQUET = _REPO_ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"


def _build_4h_ema200_aligned(dates_15m: pd.DatetimeIndex, ema_period: int = 200) -> np.ndarray:
    """Load 4H bars, compute EMA200, align to 15m timestamps with lookahead safety.

    TIMING CONVENTION (critical for lookahead safety):
        The parquet index is bar-OPEN time. A 4H bar opening at T is not closed
        until T + 4h. Therefore we can only use the EMA of bar T at 15m timestamps
        >= T + 4h (the bar's close time).

    Implementation:
        1. Load full 4H parquet (all history — ensures EMA(200) warm-up before
           any OOS window; ~5000 4H bars precede 2022-01-01).
        2. Compute EMA(200) on 4H close prices.
        3. Build a Series indexed by bar-CLOSE timestamps (open_time + 4h).
        4. Use pd.merge_asof(direction="backward") against the 15m timestamps to
           forward-fill: each 15m bar gets the EMA of the most recently CLOSED 4H bar.
        5. Strip timezone to match the tz-naive 15m index used by backtesting.py.

    Returns an ndarray aligned to dates_15m (NaN during the EMA warm-up period).
    """
    df4h = pd.read_parquet(_4H_PARQUET)
    # df4h.index = open_time (UTC-aware). Close time = open_time + 4h.
    ema4h = ema(df4h["close"], ema_period)

    # Build (close_time → EMA) Series
    close_times = df4h.index + pd.Timedelta(hours=4)
    ema_at_close = pd.Series(ema4h.values, index=close_times, name="ema4h")

    # The 15m index is tz-naive after backtesting.py strips tz (_load_slice).
    # For the merge we need matching tz and datetime precision.
    # Strategy: work in tz-naive UTC by stripping tz from both sides.
    close_times_naive = ema_at_close.index.tz_localize(None)
    ema_right = pd.Series(ema_at_close.values, index=close_times_naive, name="ema4h")

    # dates_15m is already tz-naive (backtesting.py stripped it); ensure same precision.
    dates_15m_naive = pd.DatetimeIndex(
        dates_15m.astype("datetime64[us]") if dates_15m.tz is None else dates_15m.tz_localize(None).astype("datetime64[us]")
    )
    ema_right.index = pd.DatetimeIndex(ema_right.index.astype("datetime64[us]"))

    left = pd.DataFrame(index=dates_15m_naive)
    left.index.name = "ts"
    right = pd.DataFrame({"ema4h": ema_right.values}, index=ema_right.index)
    right.index.name = "ts"

    merged = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged["ema4h"].values


class DivergenceV2(Strategy):
    """RSI + MFI divergence with 4H EMA200 regime gate.

    New vs v1: MFI replaces OBV, 4H EMA200 gate added, Fix-2 option-b
    confirmation, ATR compressed-regime skip. See module docstring.
    """

    # --- swing detection ---
    swing_k: int = 3
    min_swing_separation_bars: int = 5
    max_swing_separation_bars: int = 60

    # --- RSI divergence ---
    rsi_period: int = 14
    rsi_oversold_zone: float = 30.0
    rsi_overbought_zone: float = 70.0

    # --- MFI divergence ---
    mfi_period: int = 14

    # --- exits ---
    atr_period: int = 14
    sl_atr_multiple: float = 1.5
    tp_atr_multiple: float = 4.5
    max_hold_bars: int = 96  # 24h on 15m

    # --- sizing ---
    risk_per_trade_pct: float = 1.0
    leverage: int = 5  # Fix 3 live-safe default
    allow_shorts: bool = True

    # --- trend filter (15m EMA200, Fix 3 live-safe ON by default) ---
    trend_filter_enabled: bool = True
    trend_ema_period: int = 200

    # --- 4H EMA200 regime gate ---
    use_4h_regime_gate: bool = True
    ema_4h_period: int = 200

    # --- cascade veto (Fix 3 / FUTURE_DIRECTIONS Bug 4) ---
    atr_close_ratio_veto: float = 0.015

    # --- ATR compressed-regime skip (Nagel 2012 — skip low-vol, keep high-vol) ---
    atr_compressed_veto: bool = True
    atr_compressed_lookback: int = 50
    atr_compressed_threshold: float = 0.5  # skip when atr < threshold * atr_mean

    # --- gate / confirmation variant toggles (default=False = v2 original behavior) ---
    use_or_gate: bool = False          # True → rsi_div OR mfi_div; False → AND
    use_v1_confirmation: bool = False  # True → close > high[b2] (v1-original, looser)

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self._close_arr = close.values
        self._high_arr = high.values
        self._low_arr = low.values

        # --- core indicators ---
        self._rsi = rsi(close, self.rsi_period).values
        self._mfi = mfi(high, low, close, volume, self.mfi_period).values
        self._atr = atr(high, low, close, self.atr_period).values

        # --- ATR rolling mean (for compressed-regime veto) ---
        lb = self.atr_compressed_lookback
        atr_s = pd.Series(self._atr)
        self._atr_mean = atr_s.rolling(window=lb, min_periods=lb).mean().values

        # --- 15m trend EMA ---
        if self.trend_filter_enabled:
            self._ema_trend = ema(close, self.trend_ema_period).values
        else:
            self._ema_trend = None

        # --- 4H EMA200 regime gate ---
        if self.use_4h_regime_gate:
            # data index is tz-naive after backtesting.py's _load_slice strips tz.
            dates = pd.DatetimeIndex(self.data.index)
            self._ema_4h = _build_4h_ema200_aligned(dates, self.ema_4h_period)
        else:
            self._ema_4h = None

        # --- swing masks (unshifted — find_divergence shifts internally) ---
        swing_highs, swing_lows = swing_high_low(high, low, k=self.swing_k)

        # --- precompute registered swing positions for Fix-2 option-b ---
        # registered_swing_lows[j] = True iff bar j-k was a swing low (knowable at j).
        # We store positions for lookup in _long_signal/_short_signal.
        k = self.swing_k
        n = len(close)

        sl_positions = np.where(swing_lows.values)[0]
        sh_positions = np.where(swing_highs.values)[0]

        # registered_sl_positions[j] = the latest swing-low bar b such that b+k <= j.
        # We store as an array: for each j, the most-recent registered swing-low bar.
        # Same for swing-highs.
        # Build lookup arrays: _reg_sl_before[j] = largest b in sl_positions with b+k <= j.
        self._reg_sl_before = np.full(n, -1, dtype=np.intp)
        ptr = 0
        for j in range(n):
            while ptr < len(sl_positions) and sl_positions[ptr] + k <= j:
                ptr += 1
            if ptr > 0:
                self._reg_sl_before[j] = sl_positions[ptr - 1]

        self._reg_sh_before = np.full(n, -1, dtype=np.intp)
        ptr = 0
        for j in range(n):
            while ptr < len(sh_positions) and sh_positions[ptr] + k <= j:
                ptr += 1
            if ptr > 0:
                self._reg_sh_before[j] = sh_positions[ptr - 1]

        # --- AND-gate signals (RSI + MFI divergence both required) ---
        rsi_s = pd.Series(self._rsi, index=close.index)
        mfi_s = pd.Series(self._mfi, index=close.index)

        rsi_long = find_divergence(
            price=low, indicator=rsi_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k,
            min_separation=self.min_swing_separation_bars,
            max_separation=self.max_swing_separation_bars,
        )
        mfi_long = find_divergence(
            price=low, indicator=mfi_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k,
            min_separation=self.min_swing_separation_bars,
            max_separation=self.max_swing_separation_bars,
        )

        rsi_short = find_divergence(
            price=high, indicator=rsi_s, swing_mask=swing_highs,
            kind="regular_bearish", k=k,
            min_separation=self.min_swing_separation_bars,
            max_separation=self.max_swing_separation_bars,
        )
        mfi_short = find_divergence(
            price=high, indicator=mfi_s, swing_mask=swing_highs,
            kind="regular_bearish", k=k,
            min_separation=self.min_swing_separation_bars,
            max_separation=self.max_swing_separation_bars,
        )

        if self.use_or_gate:
            self._long_sig = (rsi_long | mfi_long).values
            self._short_sig = (rsi_short | mfi_short).values
        else:
            self._long_sig = (rsi_long & mfi_long).values
            self._short_sig = (rsi_short & mfi_short).values

        self._entry_bar: int | None = None

    # -------------------------------------------------------------- sizing

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

    # -------------------------------------------------------------- shared gates

    def _shared_gates_pass(self, i: int) -> bool:
        """Common pre-checks: ATR available, cascade veto, compressed-vol veto."""
        atr_i = self._atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            return False
        close_i = self._close_arr[i]
        # Cascade veto: extreme realized vol
        if close_i > 0 and (atr_i / close_i) > self.atr_close_ratio_veto:
            return False
        # ATR compressed-regime veto (Nagel 2012: skip compressed vol, keep high vol)
        if self.atr_compressed_veto:
            atr_mean = self._atr_mean[i]
            if np.isfinite(atr_mean) and atr_mean > 0:
                if atr_i < self.atr_compressed_threshold * atr_mean:
                    return False
        return True

    # -------------------------------------------------------------- signals

    def _long_signal(self, i: int) -> bool:
        if i < self.swing_k:
            return False
        if not self._long_sig[i]:
            return False

        b2 = i - self.swing_k  # most-recent swing bar that just confirmed

        # RSI oversold gate
        rsi_b2 = self._rsi[b2]
        if not (np.isfinite(rsi_b2) and rsi_b2 < self.rsi_oversold_zone):
            return False

        atr_i = self._atr[i]
        close_i = self._close_arr[i]

        # Confirmation gate: v1-original (loose) OR v2 option-b (strict).
        if self.use_v1_confirmation:
            # v1 original: close[j] > high[b2] — looser, uses swing bar's high directly.
            high_b2 = self._high_arr[b2]
            if not (np.isfinite(high_b2) and close_i > high_b2):
                return False
        else:
            # Fix 2 option-b: N=2 consecutive closes above h_recent (most-recent
            # registered swing-high at or before b2).
            # By definition: any swing at bar p registers at p+k. At bar j=b2+k,
            # the most-recent registered swing-high AT OR BEFORE b2 is the one
            # at index self._reg_sh_before[b2] (since that lookup includes all
            # bars p where p+k <= b2, i.e. p <= b2-k).
            h_recent_bar = int(self._reg_sh_before[b2])
            if h_recent_bar < 0:
                return False  # no registered swing-high before b2 to confirm against
            h_recent_price = self._high_arr[h_recent_bar]
            if not np.isfinite(h_recent_price):
                return False
            # Require 2 consecutive closes above h_recent (bars i-1 and i)
            if i < 1:
                return False
            if not (close_i > h_recent_price and self._close_arr[i - 1] > h_recent_price):
                return False

        # 4H regime gate
        if self.use_4h_regime_gate and self._ema_4h is not None:
            e4h = self._ema_4h[i]
            if not (np.isfinite(e4h) and close_i > e4h):
                return False

        # 15m trend filter
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

        # RSI overbought gate
        rsi_b2 = self._rsi[b2]
        if not (np.isfinite(rsi_b2) and rsi_b2 > self.rsi_overbought_zone):
            return False

        atr_i = self._atr[i]
        close_i = self._close_arr[i]

        # Confirmation gate (short mirror): v1-original (loose) OR v2 option-b (strict).
        if self.use_v1_confirmation:
            # v1 original: close[j] < low[b2] — looser, uses swing bar's low directly.
            low_b2 = self._low_arr[b2]
            if not (np.isfinite(low_b2) and close_i < low_b2):
                return False
        else:
            # Fix 2 option-b (short mirror): N=2 consecutive closes below l_recent
            l_recent_bar = int(self._reg_sl_before[b2])
            if l_recent_bar < 0:
                return False
            l_recent_price = self._low_arr[l_recent_bar]
            if not np.isfinite(l_recent_price):
                return False
            if i < 1:
                return False
            if not (close_i < l_recent_price and self._close_arr[i - 1] < l_recent_price):
                return False

        # 4H regime gate
        if self.use_4h_regime_gate and self._ema_4h is not None:
            e4h = self._ema_4h[i]
            if not (np.isfinite(e4h) and close_i < e4h):
                return False

        # 15m trend filter
        if self.trend_filter_enabled and self._ema_trend is not None:
            t = self._ema_trend[i]
            if not (np.isfinite(t) and close_i < t):
                return False

        return True

    # -------------------------------------------------------------- loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        # Time stop
        if self.position:
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold_bars:
                self.position.close()
                self._entry_bar = None
            return

        if not self._shared_gates_pass(i):
            return

        atr_v = self._atr[i]
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
# DivergenceV2Loose — OR-gate + v1 confirmation + widened RSI zones
# ---------------------------------------------------------------------------

class DivergenceV2Loose(DivergenceV2):
    """Looser variant of DivergenceV2, intended to trade more while keeping 4H regime gate.

    Changes vs DivergenceV2:
      - OR-gate: rsi_div OR mfi_div fires the signal (instead of AND).
      - v1 confirmation: close[j] > high[b2] (long) / close[j] < low[b2] (short).
        The 4H EMA200 gate now provides the regime context the original
        confirmation lacked, so the stricter option-b is unnecessary.
      - RSI zones widened to 35/65 (v1 original) from v2's 30/70.
      - ATR compressed-vol veto dropped (low evidence; may suppress trades).
      - Everything else inherited from DivergenceV2: 4H EMA200 gate (KEPT),
        MFI(14), leverage=5, atr_close_ratio_veto=0.015, trend_filter_enabled.
    """

    # OR-gate: either RSI or MFI divergence fires the signal
    use_or_gate: bool = True
    # v1-original confirmation: close > high[b2] / close < low[b2]
    use_v1_confirmation: bool = True
    # Widened RSI zones (back to v1 defaults)
    rsi_oversold_zone: float = 35.0
    rsi_overbought_zone: float = 65.0
    # Drop ATR compressed-vol veto (low evidence; may suppress trades)
    atr_compressed_veto: bool = False


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    import pandas as pd
    from backtesting import Backtest

    HERE = Path(__file__).resolve().parent.parent
    PARQUET = HERE / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

    if not PARQUET.exists():
        print(f"ERROR: parquet not found at {PARQUET}")
        sys.exit(1)

    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})

    cutoff = df.index[-1] - pd.Timedelta(days=180)
    df_slice = df[df.index >= cutoff].copy()

    if df_slice.index.tz is not None:
        df_slice.index = df_slice.index.tz_localize(None)

    print(f"Smoke check: {len(df_slice)} bars from {df_slice.index[0]} to {df_slice.index[-1]}")

    bt = Backtest(
        df_slice,
        DivergenceV2,
        cash=100_000.0,
        commission=0.0005,
        margin=1.0 / 20,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()

    print()
    print("=== DivergenceV2 smoke check (last 6 months, 15m BTC/USDT) ===")
    print(f"  Trades          : {int(stats['# Trades'])}")
    print(f"  Total return    : {float(stats['Return [%]']):+.2f}%")
    print(f"  Win rate        : {float(stats.get('Win Rate [%]') or 0.0):.1f}%")
    print(f"  Max drawdown    : {float(stats.get('Max. Drawdown [%]') or 0.0):.2f}%")
