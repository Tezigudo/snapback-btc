"""
AdaptiveTrendV2 + vol-scaled position sizing — one-feature subclass.

Improvement under test: target annualised vol = 15%; scale base position size by
(target_vol / realised_vol).  realised_vol = std(close.pct_change(), 20*4=80 H6
bars) * sqrt(365*4=1460 H6 bars/year).

Why subclass (not toggle on v2):
- Constraint: do not modify v2-base file.
- Realised-vol precompute is one extra array + one override on _position_units;
  cleanly isolatable.

Lookahead safety:
- realised vol is computed on the FULL H6 frame, then .shift(1) (so the value
  at H6 close ts is the value of the just-closed bar), then forward-filled
  onto the 15m index — mirrors the v2-base treatment of MOM/ATR.  Built once
  in init(); does not depend on L/theta, so no rebuild on re-opt.

Per-trade Sharpe / PSR caveat:
- backtesting.py Trade.ReturnPct is (exit - entry) / entry — purely price-based
  and independent of size.  Vol-scaled sizing changes how many BTC we hold,
  not which trades fire (entry gates on mom > theta) and not their pnl_pct.
  Expectation: per-trade Sharpe and PSR essentially unchanged.  The effect of
  vol targeting lands entirely in COMPOUNDED RETURN and MAX DD (dollar PnL
  reweighted: smaller in high-vol regimes, larger in low-vol regimes).

Authority: research-only. Not wired to bot.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from strategy.indicators import atr as wilder_atr  # noqa: F401  (imported by base, kept for parity)
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2


# H6 bars/year = 365 * 4 = 1460. 20-day window = 20 * 4 = 80 H6 bars.
_VOL_LOOKBACK_H6_BARS = 80
_H6_BARS_PER_YEAR = 1460
_ANN_SQRT = math.sqrt(_H6_BARS_PER_YEAR)


class AdaptiveTrendV2_vol_scaled_sizing(AdaptiveTrendV2):
    """AdaptiveTrendV2 with target-vol position sizing.

    Adds ONE behaviour:
      target_vol = 0.15 (annualised, ~15%)
      scale = target_vol / realized_vol_annualised
      target_btc = base_target_btc * scale
    where realised vol is std(H6 close.pct_change(), 80 bars) * sqrt(1460).
    The leverage ceiling (`max_btc`) still binds via `min(target_btc, max_btc)`.
    """

    # --- vol targeting params ---
    target_vol_annualised: float = 0.15  # 15% target
    use_vol_scaled_sizing: bool = True   # toggle; flip off to fall back to base

    # ------------------------------------------------------------------ init
    def init(self) -> None:
        super().init()
        # Build realised-vol array on the SAME H6 frame the base built.
        # Mirror v2-base's shift+ffill pattern so we never read future data.
        h6 = self._h6
        rv_h6 = h6["Close"].pct_change().rolling(_VOL_LOOKBACK_H6_BARS).std() * _ANN_SQRT
        sig = pd.concat({"rv": rv_h6}, axis=1).shift(1).reindex(self._index, method="ffill")
        self._realized_vol = sig["rv"].values  # 15m-aligned, causal

    # ------------------------------------------------------------------ sizing
    def _position_units(self, price: float, sl_distance: float) -> int:
        """Vol-scaled position sizing (integer units).

        Falls back to base sizing if:
          - use_vol_scaled_sizing is False
          - realised vol is NaN / non-positive (warmup, degenerate)

        NOTE: backtesting.py 0.6.5 only accepts integer units. Fractional
        0.001-BTC sizing is implemented via HARNESS-level price scaling
        (tools/_fractional_run.py). Under that scaling, the vol-scale
        multiplier lands on milli-BTC units, so e.g. scale=0.7, target=
        2.3 BTC -> int(2.3 * 0.7 * 1000) = 1610 milli-BTC = 1.610 BTC,
        instead of int(1.61) = 1 BTC under the prior bug.
        """
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0

        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance

        if self.use_vol_scaled_sizing:
            i = len(self.data) - 1
            rv = self._realized_vol[i] if 0 <= i < len(self._realized_vol) else float("nan")
            if np.isfinite(rv) and rv > 0:
                scale = self.target_vol_annualised / rv
                target_btc *= scale
            # else: warmup / degenerate -> keep base target_btc (scale = 1)

        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)
