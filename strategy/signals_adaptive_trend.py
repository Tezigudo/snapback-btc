"""
AdaptiveTrendV1 — bare-signal port of the AdaptiveTrend paper to BTC perp.

Paper: Bui & Nguyen, "Systematic Trend-Following with Adaptive Portfolio
Construction: Enhancing Risk-Adjusted Alpha in Cryptocurrency Markets"
(arXiv:2602.11708v1, Feb 2026).  Reported portfolio Sharpe 2.41 across 150+
crypto perps; bare-signal single-asset performance is expected to be much
lower (the paper's ablation has "Fixed Parameters (no opt.)" at Sharpe 1.34
on the same universe, and the portfolio machinery — market-cap filter,
Sharpe-selection, 70/30 allocation, monthly re-opt — contributes most of
the reported Sharpe).

Core signal (Algorithm 1, Eq. 2 & 3):
  MOM_t = (P_t - P_{t-L}) / P_{t-L}            (Eq. 2, on H6 candles)
  Entry long  if no position and MOM_t > +theta_entry
  Entry short if no position and MOM_t < -theta_entry
  S_t       = max(S_{t-1}, P_t - alpha * ATR_t)   (long; Eq. 3, ratcheted up)
  S_t       = min(S_{t-1}, P_t + alpha * ATR_t)   (short; mirrored)
  Close when P_t < S_t (long) / P_t > S_t (short)

What is and isn't implemented from the paper
--------------------------------------------
Implemented:
  - MOM signal on H6 (paper's Table 5 picks H6 over H1/H4/D1 explicitly).
  - Dynamic ATR trailing stop with alpha=2.5 (paper's plateau centre).
  - Position sizing via risk-per-trade-pct & leverage cap (repo convention).

NOT implemented (deliberate, scope-limited):
  - Monthly theta_entry / L re-optimisation (portfolio-level, not bar-level).
  - Market-cap filter + Sharpe-ratio asset selection (multi-asset only).
  - 70/30 long/short capital allocation (multi-asset only).
The spike measures whether the BARE signal has any post-funding edge on
BTC perp — not whether it reproduces 2.41.

Translation choices for 15m carrier -> H6 native
------------------------------------------------
The strategy receives the same 15m parquet every other strategy uses (so the
run_strategy_experiment runner works unchanged), but internally resamples
to H6 right-aligned 00/06/12/18 UTC bars and computes MOM(L), ATR(k) only
on the H6 series.  For each 15m bar `i`, we look up the H6 row corresponding
to the most recent FULLY CLOSED H6 candle and use its (shifted-by-one) MOM
and ATR.  This avoids:
  - The paper's anti-recommendation of sub-H6 sampling (Table 5: H1 Sharpe
    1.54 vs H6 2.41 — turnover/cost kills high-freq adaptations).
  - Look-ahead from peeking at the current incomplete H6 bar.

Funding model
-------------
Funding is NOT deducted inside next() — backtesting.py can't mutate equity
mid-run cleanly, and a synthetic offset-trade hack would corrupt trade
statistics.  Instead the harness post-processes realised trades using
backtest.funding_cost_for_trades(trades_df, data, funding_df), which uses
the actual Binance funding parquet and the signed Size convention (long
pays positive funding, short receives positive funding).  See
tools/_adaptrend_run.py for the wrapper.

Authority: this file is research-only.  Not wired to bot.py, not in live
deploy, not in backtest.py STRATEGIES dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr as wilder_atr


# Resample rule: right-closed 6-hour candles anchored on 00:00 UTC so the
# H6 boundaries are 00, 06, 12, 18 — matching Binance perpetual funding cadence.
_H6_RULE = "6h"


def _resample_h6(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample a (capitalised-column) 15m OHLC frame to H6.

    The index is naive UTC (run_strategy_experiment strips tz).  Returns a
    frame indexed by the H6 close timestamps with columns
    [Open, High, Low, Close].
    """
    o = df_15m["Open"].resample(_H6_RULE, label="right", closed="right").first()
    h = df_15m["High"].resample(_H6_RULE, label="right", closed="right").max()
    lo = df_15m["Low"].resample(_H6_RULE, label="right", closed="right").min()
    c = df_15m["Close"].resample(_H6_RULE, label="right", closed="right").last()
    h6 = pd.concat({"Open": o, "High": h, "Low": lo, "Close": c}, axis=1).dropna()
    return h6


def compute_h6_signal(
    df_15m: pd.DataFrame,
    momentum_lookback_h6: int,
    atr_period_h6: int,
) -> pd.DataFrame:
    """Build the H6-native signal frame and align it back onto the 15m index.

    Returns a frame indexed identically to ``df_15m`` with columns:
      - mom_h6        — most recently CLOSED H6 bar's MOM (shifted forward).
      - atr_h6        — most recently CLOSED H6 bar's ATR (shifted forward).
      - close_h6      — most recently CLOSED H6 close price.
      - h6_close_idx  — int index into the H6 frame of the most recent bar.

    "Most recently CLOSED" means: at 15m bar `i` whose timestamp is t, we
    use the H6 row whose right-aligned timestamp is the largest right-aligned
    6h boundary <= t MINUS one (i.e. shift(1) on H6 frame).  This is causal.
    """
    h6 = _resample_h6(df_15m)

    # Eq. 2: MOM_t = (P_t - P_{t-L}) / P_{t-L}.  Computed on closes.
    mom = (h6["Close"] - h6["Close"].shift(momentum_lookback_h6)) / h6["Close"].shift(
        momentum_lookback_h6
    )

    # ATR(k) on H6 via the repo's Wilder ATR.
    atr_h6 = wilder_atr(h6["High"], h6["Low"], h6["Close"], period=atr_period_h6)

    h6_signal = pd.concat(
        {"mom_h6": mom, "atr_h6": atr_h6, "close_h6": h6["Close"]}, axis=1
    )
    # shift(1): expose to 15m only AFTER the H6 bar has closed.
    h6_signal = h6_signal.shift(1)

    # Forward-fill onto the 15m grid.  ffill is correct because the H6 signal
    # is valid until the next H6 boundary closes.
    aligned = h6_signal.reindex(df_15m.index, method="ffill")
    return aligned


class AdaptiveTrendV1(Strategy):
    """Bare AdaptiveTrend signal on H6 (resampled internally from 15m)."""

    # --- Signal parameters (paper plateau centres / sane defaults) ---
    # L is optimised monthly in the paper; we pick a reasonable middle ground.
    # 4 H6 bars = 24h momentum.  This is also the lookback used by classical
    # 1-day TSMOM ports to intraday.
    momentum_lookback_h6: int = 4
    # theta_entry has no static default in the paper (also monthly-optimised).
    # 0.02 = 2% over the lookback window — strong enough to require a real
    # directional move, weak enough to fire >5 trades in a 6-month BTC window.
    theta_entry: float = 0.02
    # ATR period for the trailing stop.  Paper uses generic "k periods"; we
    # mirror Donchian-v3's 14 (Wilder default).
    atr_period_h6: int = 14
    # Plateau centre per Figure 2 (alpha in [2.0, 3.5], optimum ~2.5).
    alpha: float = 2.5

    # --- Direction control ---
    allow_shorts: bool = True

    # --- Sizing (repo convention; risk-based, leverage-capped) ---
    risk_per_trade_pct: float = 1.0
    leverage: int = 5

    # --- Safety belt: hard time-stop on a single trade.  The paper has none
    # (the trailing stop is the only exit), but a hard ceiling prevents
    # pathological no-exit holds in regimes where ATR collapses below noise.
    # 1 month at H6 = 120 bars = 30 days.  Decision-relevant on backtests
    # of 6mo+ where ATR can flatline for weeks.
    max_hold_h6_bars: int = 120

    # ------------------------------------------------------------------ init

    def init(self) -> None:
        df_15m = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
            },
            index=self.data.index,
        )
        aligned = compute_h6_signal(
            df_15m,
            momentum_lookback_h6=self.momentum_lookback_h6,
            atr_period_h6=self.atr_period_h6,
        )
        self._mom = aligned["mom_h6"].values
        self._atr = aligned["atr_h6"].values
        self._close_h6 = aligned["close_h6"].values
        self._index = df_15m.index

        # Trailing-stop ratchet state (S_t in Eq. 3).
        self._trail_level: float | None = None
        # Bar of last entry (15m bar index) — used for max-hold safety.
        self._entry_bar: int | None = None
        # Track the H6 boundary on which we last entered so we only act on
        # H6 close events, not every 15m bar — matches the paper's
        # "for each 6-hour candle" loop in Algorithm 1.
        self._last_h6_close_seen: pd.Timestamp | None = None

    # ------------------------------------------------------------------ sizing

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

    # ------------------------------------------------------------------ helpers

    def _is_h6_close_bar(self, ts: pd.Timestamp) -> bool:
        """True if this 15m bar IS an H6 close boundary (00/06/12/18 UTC).

        We only EVALUATE entry signals at H6 close to match Algorithm 1's
        "for each 6-hour candle" loop.  Exits (trailing stop) are evaluated
        every 15m bar for responsiveness in fast moves — strictly more
        conservative than paper.
        """
        # Right-aligned resample puts the H6 close at minute=0 of the next
        # 6h window: 06:00, 12:00, 18:00, 00:00.  At 15m granularity that's
        # exactly the bar with hour % 6 == 0 and minute == 0.
        return ts.hour % 6 == 0 and ts.minute == 0

    # ------------------------------------------------------------------ loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]
        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management: trailing stop, every 15m bar. ---
        if self.position:
            if not np.isfinite(atr_v) or atr_v <= 0:
                # No valid ATR (shouldn't happen mid-trade — bail safely).
                return

            # Hard max-hold belt (not in paper; prevents pathological holds).
            if (
                self._entry_bar is not None
                and (i - self._entry_bar) >= self.max_hold_h6_bars * 24  # H6 bars -> 15m
            ):
                self.position.close()
                self._trail_level = None
                self._entry_bar = None
                return

            trade = self.trades[-1] if self.trades else None
            if trade is None:
                return

            if trade.is_long:
                # S_t = max(S_{t-1}, P_t - alpha * ATR_t)
                candidate = close_v - self.alpha * atr_v
                if self._trail_level is None or candidate > self._trail_level:
                    self._trail_level = candidate
                # Push the broker stop up (ratchet, never down).
                if trade.sl is None or self._trail_level > trade.sl:
                    trade.sl = self._trail_level
                # Explicit close path (paper Eq.: close when P_t < S_t).
                if close_v < self._trail_level:
                    self.position.close()
                    self._trail_level = None
                    self._entry_bar = None
            else:
                # Short: mirror.  S_t = min(S_{t-1}, P_t + alpha * ATR_t)
                candidate = close_v + self.alpha * atr_v
                if self._trail_level is None or candidate < self._trail_level:
                    self._trail_level = candidate
                if trade.sl is None or self._trail_level < trade.sl:
                    trade.sl = self._trail_level
                if close_v > self._trail_level:
                    self.position.close()
                    self._trail_level = None
                    self._entry_bar = None
            return

        # --- Entry: only at H6 close boundaries. ---
        if not self._is_h6_close_bar(ts):
            return
        if self._last_h6_close_seen == ts:
            return
        self._last_h6_close_seen = ts

        if not np.isfinite(mom_v) or not np.isfinite(atr_v) or atr_v <= 0:
            return

        # Initial stop = entry - alpha * ATR (paper line 6 of Algorithm 1).
        # We use this as both the broker SL and the seed of S_t.
        sl_dist = self.alpha * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if mom_v > self.theta_entry:
            self.buy(size=units, sl=close_v - sl_dist)
            self._entry_bar = i
            self._trail_level = close_v - sl_dist
        elif self.allow_shorts and mom_v < -self.theta_entry:
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._trail_level = close_v + sl_dist
