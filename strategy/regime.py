"""
Deterministic regime detection — no LLM, pure features.

Borrowed *pattern* from AgentQuant (regime-aware parameter selection), but
implemented without their LLM-proposer step. We compute three regime
features per 15m bar from already-cached data:

  - funding_regime ∈ {-1, 0, +1}
      sign of the rolling-median funding rate; magnitude must exceed the
      rolling 75th-percentile of |funding| to count as +/-1, else 0
      (no edge from carry).

  - volatility_regime ∈ {0, 1}
      1 when current ATR(20,1h) > 60th percentile of trailing 200 1h-ATR
      values; 0 in quiet/dead-tape conditions where mean-reversion edges
      collapse into noise.

  - trend_regime ∈ {-1, 0, +1}
      sign of EMA(200,1h) slope over the last 24 1h-bars, with a magnitude
      threshold; 0 = ranging.

These features are attached as extra columns on the prepared DataFrame.
Strategies can gate entries on any subset. Computation is vectorised
pandas — no per-bar Python.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def funding_regime(
    funding: pd.Series,
    lookback_events: int = 30,
) -> pd.Series:
    """Return -1 / 0 / +1 per index based on rolling-median signed magnitude.

    Active iff |current rate| > rolling 75th percentile of |rate|. Inputs
    indexed at funding cadence (8h). Caller is responsible for reindexing.
    """
    abs_rate = funding.abs()
    rolling_p75 = abs_rate.rolling(window=lookback_events, min_periods=lookback_events // 2).quantile(0.75)
    active = abs_rate > rolling_p75
    sign = np.sign(funding)
    return (sign.where(active, 0.0)).astype(float).rename("funding_regime")


def volatility_regime(
    atr_series: pd.Series,
    lookback: int = 200,
    quantile: float = 0.60,
) -> pd.Series:
    """1 when current ATR is above the trailing q-th percentile, else 0."""
    threshold = atr_series.rolling(window=lookback, min_periods=lookback // 2).quantile(quantile)
    return (atr_series > threshold).astype(float).rename("volatility_regime")


def trend_regime(
    ema_series: pd.Series,
    lookback_bars: int = 24,
    min_slope_pct: float = 0.002,
) -> pd.Series:
    """Return -1 / 0 / +1 based on EMA slope over `lookback_bars` bars.

    Slope = (ema[t] - ema[t - lookback]) / ema[t - lookback].
    Magnitude must exceed `min_slope_pct` to count as trending.
    """
    past = ema_series.shift(lookback_bars)
    slope = (ema_series - past) / past
    sign = np.sign(slope)
    active = slope.abs() > min_slope_pct
    return sign.where(active, 0.0).astype(float).rename("trend_regime")


def attach_regimes(
    df: pd.DataFrame,
    funding: pd.Series | None = None,
    ema_1h: pd.Series | None = None,
    atr_1h: pd.Series | None = None,
    funding_lookback: int = 30,
    vol_lookback: int = 200,
    vol_quantile: float = 0.60,
    trend_lookback_bars: int = 24,
    trend_min_slope_pct: float = 0.002,
) -> pd.DataFrame:
    """Add funding_regime / volatility_regime / trend_regime columns to df.

    `df` must be at 15m cadence with EMA_1h, ATR_1h, Funding columns already
    attached by strategy.signals.prepare_strategy_data. The original series
    args are optional fallbacks; by default we use the columns on df.
    """
    out = df.copy()
    fund_src = funding if funding is not None else out["Funding"]
    ema_src = ema_1h if ema_1h is not None else out["EMA_1h"]
    atr_src = atr_1h if atr_1h is not None else out["ATR_1h"]

    # Funding regime computed on funding events, then ffilled to 15m.
    fund_events = fund_src.dropna().drop_duplicates()
    fr = funding_regime(fund_events, lookback_events=funding_lookback)
    out["funding_regime"] = fr.reindex(out.index, method="ffill")

    # Vol and trend computed on the already-aligned 15m columns (ffilled
    # from 1h source). The shifts in prepare_strategy_data prevent lookahead.
    out["volatility_regime"] = volatility_regime(atr_src, vol_lookback, vol_quantile)
    out["trend_regime"] = trend_regime(ema_src, trend_lookback_bars, trend_min_slope_pct)

    return out
