"""
CarryHarvester v3 — adds two tail-risk gates on top of v2.

v2 already had:
  - funding_threshold gate (only enter when |funding| > threshold)
  - max_24h_change_pct skip filter (refuse entries during fast moves)
  - SL exit + funding-normalisation exit + funding-flip exit

The P3.4 phase C result showed v2 still bleeds in vol regimes the filter
doesn't catch — 3 tail folds in 2022-12, 2024-02, and 2024-10 wiped out
25+ small-winning folds (compounded $100 → $62.30, CAGR −18.6%/yr).

v3 adds two NEW gates:

1. **ATR-percentile gate** — compute realised volatility on the last
   `atr_window_bars` bars; if the current ATR is above the 80th
   percentile over `atr_lookback_bars`, refuse new entries. This catches
   regimes where price is moving too far per bar for any 1.5% SL to be
   meaningful. The max_24h_change filter is reactive (already moved);
   this one is *current vol regime*-based and stays "off" until the
   regime calms.

2. **Drawdown circuit breaker** — track trailing max equity over the
   last `dd_lookback_bars` bars; if current equity has dropped more
   than `dd_halt_pct` from that high, refuse new entries. Existing
   positions stay managed normally. The breaker stops the strategy
   from compounding losses during a bad regime.

Both gates default to OFF (atr_percentile_threshold=100,
dd_halt_pct=100), so v3 with default params == v2 with default params.
The walk-forward sweep turns them on.
"""

from __future__ import annotations

import numpy as np
from backtesting import Strategy

LOOKBACK_24H_15M = 96  # 96 × 15m = 24h (re-exported from v2 for clarity)


class CarryHarvesterV3(Strategy):
    # --- v2-equivalent params ----------------------------------------------
    funding_threshold = 0.0002
    funding_exit_threshold = 0.00005
    sl_pct = 0.01
    risk_per_trade_pct = 2.0
    leverage = 20
    allow_shorts = True
    max_24h_change_pct = 100.0

    # --- v3 NEW: ATR-percentile vol gate -----------------------------------
    # ATR window (bars) — how many bars to average for "current vol"
    atr_window_bars = 96            # 24h at 15m
    # Rolling lookback (bars) — sample of "recent ATRs" to percentile against
    atr_lookback_bars = 2880        # 30 days at 15m
    # Skip entry if current ATR is above this percentile of the lookback.
    # 100 = gate OFF (current ATR can never exceed 100th percentile).
    atr_percentile_threshold = 100.0

    # --- v3 NEW: drawdown circuit breaker ----------------------------------
    # Trailing-high lookback (bars). If equity has dropped > dd_halt_pct
    # from this rolling high, refuse new entries.
    dd_lookback_bars = 1920         # 20 days at 15m
    dd_halt_pct = 100.0             # 100 = breaker OFF

    def init(self) -> None:
        # Pre-compute rolling ATR proxy: average true range over atr_window_bars.
        # Vectorised pandas avoids per-bar Python loops in next().
        import pandas as pd
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        self._atr_series = tr.rolling(self.atr_window_bars, min_periods=self.atr_window_bars).mean().values

    def _position_units(self, price: float, sl_distance: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance
        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)

    def _fast_move_block(self) -> bool:
        """v2's 24h-change filter."""
        if self.max_24h_change_pct >= 100.0:
            return False
        if len(self.data) <= LOOKBACK_24H_15M:
            return False
        ref = self.data.Close[-LOOKBACK_24H_15M - 1]
        now = self.data.Close[-1]
        if ref <= 0 or not np.isfinite(ref):
            return False
        change = abs(now / ref - 1.0) * 100.0
        return change > self.max_24h_change_pct

    def _atr_percentile_block(self) -> bool:
        """v3 ATR-percentile vol regime gate. True if vol is too high right now."""
        if self.atr_percentile_threshold >= 100.0:
            return False
        i = len(self.data) - 1
        if i < self.atr_lookback_bars:
            return False
        cur = self._atr_series[i]
        if not np.isfinite(cur):
            return False
        # Percentile of cur ATR among the last `atr_lookback_bars` ATRs.
        lookback = self._atr_series[max(0, i - self.atr_lookback_bars):i]
        lookback = lookback[np.isfinite(lookback)]
        if len(lookback) < self.atr_lookback_bars // 2:
            return False
        pct = (lookback < cur).mean() * 100.0
        return pct > self.atr_percentile_threshold

    def _drawdown_block(self) -> bool:
        """v3 drawdown circuit breaker. True if we've drawn down too much recently."""
        if self.dd_halt_pct >= 100.0:
            return False
        # backtesting.py exposes self.equity each bar; track via self._equity_series.
        # We compute trailing max from the recent equity curve, which
        # backtesting.py builds incrementally — but it isn't directly exposed
        # mid-run. Use a self-maintained rolling list instead.
        if not hasattr(self, "_recent_equity"):
            self._recent_equity = []
        self._recent_equity.append(self.equity)
        if len(self._recent_equity) > self.dd_lookback_bars:
            self._recent_equity = self._recent_equity[-self.dd_lookback_bars:]
        if len(self._recent_equity) < self.dd_lookback_bars // 4:
            return False  # not enough history
        peak = max(self._recent_equity)
        if peak <= 0:
            return False
        dd_pct = (peak - self.equity) / peak * 100.0
        return dd_pct > self.dd_halt_pct

    def next(self) -> None:
        funding_v = self.data.Funding[-1]
        close_v = self.data.Close[-1]

        # Note: _drawdown_block updates _recent_equity as a side effect; call
        # it every bar so the rolling list stays current even when no trade
        # decision is needed.
        dd_block = self._drawdown_block()

        if not np.isfinite(funding_v) or not np.isfinite(close_v):
            return

        if self.position:
            # Existing position management — gates only affect NEW entries.
            if abs(funding_v) < self.funding_exit_threshold:
                self.position.close()
                return
            if self.position.is_long and funding_v > 0:
                self.position.close()
                return
            if self.position.is_short and funding_v < 0:
                self.position.close()
                return
            return

        if abs(funding_v) < self.funding_threshold:
            return

        if self._fast_move_block():
            return
        if self._atr_percentile_block():
            return
        if dd_block:
            return

        sl_dist = self.sl_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if funding_v > 0:
            if not self.allow_shorts:
                return
            sl = close_v + sl_dist
            self.sell(size=units, sl=sl)
        else:
            sl = close_v - sl_dist
            self.buy(size=units, sl=sl)
