"""
AdaptiveTrendV1 + vol-scaled position sizing — one-feature subclass.

Re-validation: PSR=0.983 hint came from V2+vol_scaled (see
reports/adaptrend_v2_imp_vol_scaled_sizing.json). This file applies the
same vol-targeting overlay to the BARE V1 signal to test whether the
effect survives without V2's regime gates.

Implementation mirrors strategy/signals_adaptive_trend_v2_vol_scaled_sizing.py
verbatim except for the base class:
  target_vol = 0.15 (annualised)
  scale      = target_vol / realised_vol_annualised
  target_btc = base_target_btc * scale
  realised vol = std(H6 close.pct_change(), 80 bars) * sqrt(1460)

Lookahead safety: H6 realised-vol computed on FULL H6 frame, shift(1),
ffill onto 15m index. Identical pattern to V1's MOM/ATR alignment in
strategy/signals_adaptive_trend.compute_h6_signal().

Authority: research-only. Not wired to bot.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from strategy.signals_adaptive_trend import AdaptiveTrendV1, _resample_h6


_VOL_LOOKBACK_H6_BARS = 80          # 20-day window at H6
_H6_BARS_PER_YEAR = 1460             # 365 * 4
_ANN_SQRT = math.sqrt(_H6_BARS_PER_YEAR)


class AdaptiveTrendV1_vol_scaled_sizing(AdaptiveTrendV1):
    """AdaptiveTrendV1 with target-vol position sizing overlay."""

    # --- vol targeting params (match V2 vol_scaled defaults) ---
    target_vol_annualised: float = 0.15  # 15% target
    use_vol_scaled_sizing: bool = True   # toggle; flip off to fall back to base

    # ------------------------------------------------------------------ init
    def init(self) -> None:
        super().init()

        # Rebuild realised-vol array on the SAME H6 frame the base built.
        # We re-resample here (cheap, single pass) rather than threading
        # the H6 frame back through compute_h6_signal — keeps the V1 base
        # untouched and the override fully self-contained.
        df_15m = pd.DataFrame(
            {
                "Open":  self.data.Open,
                "High":  self.data.High,
                "Low":   self.data.Low,
                "Close": self.data.Close,
            },
            index=self.data.index,
        )
        h6 = _resample_h6(df_15m)
        rv_h6 = (
            h6["Close"].pct_change().rolling(_VOL_LOOKBACK_H6_BARS).std() * _ANN_SQRT
        )
        sig = (
            pd.concat({"rv": rv_h6}, axis=1)
            .shift(1)
            .reindex(self._index, method="ffill")
        )
        self._realized_vol = sig["rv"].values  # 15m-aligned, causal

    # ------------------------------------------------------------------ sizing
    def _position_units(self, price: float, sl_distance: float) -> int:
        """Vol-scaled position sizing (integer units, harness-scaled milli-BTC).

        Falls back to base sizing if:
          - use_vol_scaled_sizing is False
          - realised vol is NaN / non-positive (warmup, degenerate)
        """
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0

        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target_btc = risk_amount / sl_distance

        if self.use_vol_scaled_sizing:
            i = len(self.data) - 1
            rv = (
                self._realized_vol[i]
                if 0 <= i < len(self._realized_vol)
                else float("nan")
            )
            if np.isfinite(rv) and rv > 0:
                scale = self.target_vol_annualised / rv
                target_btc *= scale
            # else: warmup/degenerate -> keep base target_btc (scale = 1)

        max_btc = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target_btc, max_btc)), 0)
