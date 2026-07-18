"""
AdaptiveTrendV2 — Algorithm 2 (monthly re-optimisation layer) port.

Paper: Bui & Nguyen, "Systematic Trend-Following with Adaptive Portfolio
Construction" (arXiv:2602.11708, Feb 2026). v1 implements Algorithm 1 (bare
H6 MOM + ATR trailing stop). v2 wraps v1 with the paper's Algorithm 2:

    For each new month M:
      - Look back fit_window_months of H6 bars STRICTLY BEFORE month-M start.
      - Sweep (L, theta) over a small grid; replay the bare H6 signal on the
        fit slice; rank by per-trade Sharpe.
      - Trade month M with the winning (L*, theta*). alpha is held fixed
        (empirically validated plateau optimum, see ADAPTIVE_TREND_EXTENDED_VERDICT).

Why the H6-only inner sim (not nested backtesting.py)
-----------------------------------------------------
A nested Backtest() per candidate per month is ~12*6*5=360 full bt setups
per OOS sweep — slow, and it would force funding plumbing into the inner loop.
Instead we use a pure-numpy simulator (`_simulate_h6_fit`) that operates on
the same H6 closes/highs/lows the live strategy uses, with the same MOM/ATR/
trailing-stop rules but evaluated bar-by-bar at H6 (not 15m). The inner sim
returns gross per-trade returns; we rank candidates on the unannualised
Sharpe of those returns. This matches the paper's "select on Sharpe ratio
of the fit window" framing and the v1 harness's gross-then-funding split.

The H6-only sim diverges very slightly from the live 15m-granularity exits
(intra-H6 stop hits are bunched at the H6 close). That's acceptable for
RANKING candidates inside the fit window; it is NOT used to compute live PnL.

Lookahead safety
----------------
- The H6 frame used for fitting month M is sliced from raw OHLC with
  index < month-M start (strict). MOM and ATR are recomputed within that
  slice — we never index into a precomputed series whose tail was built
  with future data.
- Live next() uses only self._mom_h6[h6_idx] / self._atr_h6[h6_idx] where
  h6_idx is the index of the most recently CLOSED H6 bar (mirrors v1).
- For the param chosen at month-M start: we hold it for the WHOLE month,
  so re-opt happens exactly at the first H6 close of the new month.

What is NOT in v2 (deliberate)
------------------------------
- Multi-asset selection / market-cap filter / 70/30 long-short allocation
  (single-asset port, like v1).
- alpha is fixed to the v1 plateau optimum (2.0) — the extended sweep
  showed alpha monotonically increases with comp return up to 2.0 across
  every (L, theta) cell, so re-opting alpha would burn turns without
  upside.

Authority: research-only. Not wired to bot.py, not in live deploy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from backtesting import Strategy

from strategy.indicators import atr as wilder_atr
from strategy.signals_adaptive_trend import _H6_RULE, _resample_h6


# ---------------------------------------------------------------------------
# H6 pure-numpy simulator for fit-window ranking
# ---------------------------------------------------------------------------


def _simulate_h6_fit(
    h6: pd.DataFrame,
    L: int,
    theta: float,
    alpha: float,
    atr_period: int,
    allow_shorts: bool = True,
) -> np.ndarray:
    """Replay the AdaptiveTrend rule on an H6 frame, return per-trade gross returns.

    Pure numpy / no broker. Used ONLY for ranking candidates in the fit window.
    Mirrors v1's entry/exit logic at H6 granularity (intra-H6 stop hits collapsed
    to the H6 close — fine for ranking, NOT used for live PnL).

    Returns
    -------
    np.ndarray of per-trade gross returns (signed, fractional, e.g. 0.012 = +1.2%).
    Empty array if no trades fired.
    """
    if len(h6) < max(L, atr_period) + 5:
        return np.array([], dtype=float)

    close = h6["Close"].to_numpy()
    high = h6["High"].to_numpy()
    low = h6["Low"].to_numpy()

    # MOM(L) and ATR(period) computed within this slice.  No external state.
    mom_series = (h6["Close"] - h6["Close"].shift(L)) / h6["Close"].shift(L)
    atr_series = wilder_atr(h6["High"], h6["Low"], h6["Close"], period=atr_period)
    mom = mom_series.to_numpy()
    atr = atr_series.to_numpy()

    pos = 0  # 0 flat, +1 long, -1 short
    entry_price = 0.0
    trail = 0.0
    returns: list[float] = []

    for i in range(1, len(h6)):
        # Signals at bar i use mom[i-1], atr[i-1] (shift(1) — bar just closed).
        # Mirrors v1's "use most recently CLOSED H6 candle".
        m = mom[i - 1]
        a = atr[i - 1]
        c = close[i]

        if pos == 0:
            if not np.isfinite(m) or not np.isfinite(a) or a <= 0:
                continue
            if m > theta:
                pos = +1
                entry_price = c
                trail = c - alpha * a
            elif allow_shorts and m < -theta:
                pos = -1
                entry_price = c
                trail = c + alpha * a
            continue

        # In a position: ratchet trail + exit check.
        if not np.isfinite(a) or a <= 0:
            # Defensive: close at current price (no edge info).
            ret = (c - entry_price) / entry_price * pos
            returns.append(ret)
            pos = 0
            continue

        if pos > 0:
            candidate = c - alpha * a
            if candidate > trail:
                trail = candidate
            # Exit if either the bar's LOW pierced trail (intra-bar) or close < trail.
            if low[i] <= trail or c < trail:
                exit_px = min(c, trail) if low[i] <= trail else c
                ret = (exit_px - entry_price) / entry_price
                returns.append(ret)
                pos = 0
        else:
            candidate = c + alpha * a
            if candidate < trail:
                trail = candidate
            if high[i] >= trail or c > trail:
                exit_px = max(c, trail) if high[i] >= trail else c
                ret = (entry_price - exit_px) / entry_price
                returns.append(ret)
                pos = 0

    return np.asarray(returns, dtype=float)


def _per_trade_sharpe(returns: np.ndarray) -> float:
    """Unannualised per-trade Sharpe (mean/std).  Returns -inf on degenerate input."""
    if len(returns) < 2:
        return float("-inf")
    std = float(np.std(returns, ddof=1))
    if std <= 0 or not np.isfinite(std):
        return float("-inf")
    return float(np.mean(returns)) / std


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class _MonthlyChoice:
    """One row of the per-month decision log."""

    month_start: pd.Timestamp
    L: int
    theta: float
    n_fit_trades: int
    fit_sharpe: float
    reason: str  # "refit_ok" | "fallback_insufficient_trades" | "warmup_initial"


class AdaptiveTrendV2(Strategy):
    """AdaptiveTrend + monthly L/theta re-optimisation (Algorithm 2)."""

    # --- Initial / warmup params (used until first re-opt fires) ---
    momentum_lookback_h6: int = 4
    theta_entry: float = 0.02
    atr_period_h6: int = 14
    alpha: float = 2.0  # fixed (v1 plateau optimum)

    # --- Re-opt config ---
    fit_window_months: int = 6
    fit_param_grid_L: tuple = (3, 4, 5, 6)
    fit_param_grid_theta: tuple = (0.015, 0.02, 0.025)
    fit_metric: str = "per_trade_sharpe"
    min_trades_for_fit: int = 20

    # --- Direction + sizing (same as v1) ---
    allow_shorts: bool = True
    risk_per_trade_pct: float = 1.0
    leverage: int = 5
    max_hold_h6_bars: int = 120

    # --- Prefix-buffer trade guard ---
    # When the runner prepends `fit_window_months` of prior history (so the
    # first month's re-opt has data), entries must NOT fire during the prefix
    # — otherwise the in-window backtest carries a position over the oos_start
    # boundary, contaminating the v1 comparison.  The runner sets this to the
    # actual oos_start timestamp (pandas.Timestamp ns) at strategy spawn time
    # via a `trade_start_ns` config kwarg.  0 = no guard (run from bar 0).
    trade_start_ns: int = 0

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
        # Full H6 frame — built once.  We RE-COMPUTE MOM/ATR per fit slice,
        # so this frame only provides the raw OHLC.  Indexed by H6 close ts.
        self._h6 = _resample_h6(df_15m)
        # H6 close timestamps as a numpy datetime64 array for fast searchsorted.
        self._h6_ts = self._h6.index.to_numpy()

        # Precompute live signal arrays under INITIAL params; we will rebuild
        # these whenever the chosen (L, theta) changes.
        self._index = df_15m.index
        self._index_arr = df_15m.index.to_numpy()
        self._rebuild_live_signal(self.momentum_lookback_h6, self.theta_entry)

        # Currently active params (updated by re-opt).
        self._active_L = int(self.momentum_lookback_h6)
        self._active_theta = float(self.theta_entry)

        # Trailing-stop ratchet state.
        self._trail_level: float | None = None
        self._entry_bar: int | None = None
        self._last_h6_close_seen: pd.Timestamp | None = None

        # Re-opt bookkeeping: month key (year, month) we have already fit for.
        self._last_refit_month: tuple[int, int] | None = None
        # Per-month decision log — exposed for diagnostics.
        self.monthly_choices: list[_MonthlyChoice] = []

    # ------------------------------------------------------------------ signal cache

    def _rebuild_live_signal(self, L: int, theta: float) -> None:
        """Recompute the 15m-aligned MOM/ATR arrays under (L, theta).

        Mirrors v1's compute_h6_signal: MOM and ATR on the FULL H6 frame,
        shifted(1) so the value at H6 close ts is the value of the just-closed
        bar, then forward-filled onto the 15m index.

        This rebuild touches arrays only — we never reindex into the future
        because next() reads index `i = len(self.data) - 1`, which inside
        backtesting.py is always the current bar.  Holding the FULL series
        in arrays is identical to holding only the visible prefix, since
        next() reads strictly causally.
        """
        h6 = self._h6
        mom = (h6["Close"] - h6["Close"].shift(L)) / h6["Close"].shift(L)
        atr_h6 = wilder_atr(h6["High"], h6["Low"], h6["Close"], period=self.atr_period_h6)
        sig = pd.concat({"mom_h6": mom, "atr_h6": atr_h6, "close_h6": h6["Close"]}, axis=1)
        sig = sig.shift(1)
        aligned = sig.reindex(self._index, method="ffill")
        self._mom = aligned["mom_h6"].values
        self._atr = aligned["atr_h6"].values
        self._close_h6 = aligned["close_h6"].values
        self._active_theta = float(theta)
        self._active_L = int(L)

    # ------------------------------------------------------------------ re-opt

    def _maybe_refit(self, ts: pd.Timestamp) -> None:
        """If ts is the first bar of a new month, run Algorithm 2 to pick (L*, theta*)."""
        month_key = (ts.year, ts.month)
        if self._last_refit_month == month_key:
            return
        self._last_refit_month = month_key

        # Fit slice: H6 bars with index < month start.
        month_start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
        fit_start = month_start - pd.DateOffset(months=self.fit_window_months)

        # Strict-less-than on month_start ensures lookahead safety.
        h6_idx = self._h6.index
        fit_mask = (h6_idx >= fit_start) & (h6_idx < month_start)
        fit_h6 = self._h6.loc[fit_mask]

        # Need at least max(L)+atr_period buffer for valid MOM/ATR + a few trades.
        min_bars_needed = max(self.fit_param_grid_L) + self.atr_period_h6 + 20
        if len(fit_h6) < min_bars_needed:
            # Warmup: stick with initial / prior-month params.
            self.monthly_choices.append(_MonthlyChoice(
                month_start=month_start,
                L=self._active_L,
                theta=self._active_theta,
                n_fit_trades=0,
                fit_sharpe=float("nan"),
                reason="warmup_initial",
            ))
            return

        # Sweep the small grid.
        best = None
        best_sharpe = float("-inf")
        best_n = 0
        for L in self.fit_param_grid_L:
            for theta in self.fit_param_grid_theta:
                rets = _simulate_h6_fit(
                    fit_h6,
                    L=int(L),
                    theta=float(theta),
                    alpha=float(self.alpha),
                    atr_period=int(self.atr_period_h6),
                    allow_shorts=bool(self.allow_shorts),
                )
                if len(rets) < self.min_trades_for_fit:
                    continue
                sr = _per_trade_sharpe(rets)
                if sr > best_sharpe:
                    best_sharpe = sr
                    best = (int(L), float(theta))
                    best_n = len(rets)

        if best is None:
            # Small-sample fallback: keep the prior month's params.
            self.monthly_choices.append(_MonthlyChoice(
                month_start=month_start,
                L=self._active_L,
                theta=self._active_theta,
                n_fit_trades=0,
                fit_sharpe=float("nan"),
                reason="fallback_insufficient_trades",
            ))
            return

        L_star, theta_star = best
        if (L_star, theta_star) != (self._active_L, self._active_theta):
            self._rebuild_live_signal(L_star, theta_star)
        self.monthly_choices.append(_MonthlyChoice(
            month_start=month_start,
            L=L_star,
            theta=theta_star,
            n_fit_trades=best_n,
            fit_sharpe=best_sharpe,
            reason="refit_ok",
        ))

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

    @staticmethod
    def _is_h6_close_bar(ts: pd.Timestamp) -> bool:
        return ts.hour % 6 == 0 and ts.minute == 0

    # ------------------------------------------------------------------ loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])
        ts = self._index[i]

        # Algorithm 2: refit at the first 15m bar whose month differs from the
        # last-refit month.  Cheap check; the _maybe_refit early-outs on match.
        self._maybe_refit(ts)

        atr_v = self._atr[i]
        mom_v = self._mom[i]

        # --- Position management: trailing stop, every 15m bar. ---
        if self.position:
            if not np.isfinite(atr_v) or atr_v <= 0:
                return

            if (
                self._entry_bar is not None
                and (i - self._entry_bar) >= self.max_hold_h6_bars * 24
            ):
                self.position.close()
                self._trail_level = None
                self._entry_bar = None
                return

            trade = self.trades[-1] if self.trades else None
            if trade is None:
                return

            if trade.is_long:
                candidate = close_v - self.alpha * atr_v
                if self._trail_level is None or candidate > self._trail_level:
                    self._trail_level = candidate
                if trade.sl is None or self._trail_level > trade.sl:
                    trade.sl = self._trail_level
                if close_v < self._trail_level:
                    self.position.close()
                    self._trail_level = None
                    self._entry_bar = None
            else:
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

        # PREFIX GUARD: in prefix-buffered runs, block entries until oos_start.
        # Re-opt still ran above (keeps params/ATR warmed); we just don't trade.
        # This is what makes v2 start FLAT at oos_start — identical to v1.
        if self.trade_start_ns > 0 and ts.value < self.trade_start_ns:
            return

        sl_dist = self.alpha * atr_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        theta = self._active_theta
        if mom_v > theta:
            self.buy(size=units, sl=close_v - sl_dist)
            self._entry_bar = i
            self._trail_level = close_v - sl_dist
        elif self.allow_shorts and mom_v < -theta:
            self.sell(size=units, sl=close_v + sl_dist)
            self._entry_bar = i
            self._trail_level = close_v + sl_dist
