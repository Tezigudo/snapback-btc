"""
Live signal port of v3-all-wider-4 for use in bot.py.

This is a pure function port of:
  - DayTradeMultiFactorBTC (v1) — base RSI/volume/EMA/funding filters
  - DayTradeMultiFactorBTCv2 (v2) — TA confirmations (trendline, S/R, Fibonacci)
  - DayTradeMultiFactorBTCv3 (v3) — dist-ema filter + vol-regime gate + ATR stops
  - V3AllWider4 — atr_sl_k=4.0, atr_tp_k=8.0

`confirmations_required = 3` (strict) — must satisfy all TA confirmations.

The function operates on a recent slice of 15m bars (>= 250 recommended).
The bot calls this every poll tick; it evaluates the LAST CLOSED bar and
returns (side, sl_distance, tp_distance, debug) where sl/tp are in price units
(adapted to current ATR — NOT a fixed %).

Returns:
  side: 'long' | 'short' | None
  sl_distance: price distance for stop loss (current ATR × atr_sl_k)
  tp_distance: price distance for take profit (current ATR × atr_tp_k)
  debug: dict of gate states (for logging)

If side is None, sl/tp are NaN — caller should not place an order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import (
    atr,
    ema,
    fib_retracement_distance_pct,
    nearest_sr_zone_distance_pct,
    recent_swing_pair,
    rsi,
    sma,
    sr_zones,
    swing_high_low,
    trendline_from_swings,
    trendline_proximity_pct,
)

# Strategy constants — must match V3AllWider4 in signals_multifactor_v3.py.
# Centralised here so bot.py reads them from one source.
ATR_PERIOD = 14
ATR_SL_K = 4.0
ATR_TP_K = 8.0

MAX_DISTANCE_ABOVE_EMA_PCT = 0.20   # dist-ema filter (skip longs when >20% above EMA)
MAX_DISTANCE_BELOW_EMA_PCT = 0.20   # skip shorts when >20% below EMA

VOL_REGIME_LOOKBACK_DAYS = 30
VOL_REGIME_MAX_PCTILE = 0.85        # skip top 15% vol periods

# TA confirmation knobs
CONFIRMATIONS_REQUIRED = 3
SWING_K = 3
SWING_LOOKBACK_BARS = 200
TRENDLINE_MAX_DIST_PCT = 0.015
SR_MAX_DIST_PCT = 0.010
SR_CLUSTER_TOLERANCE_PCT = 0.005
FIB_MAX_DIST_PCT = 0.010

# Base v1 filter constants — driven by params.yaml at runtime
# (this module reads them via the params dict passed in).


def _daily_atr_percentile(bars_15m: pd.DataFrame, atr_period: int,
                          lookback_days: int) -> float:
    """Today's daily ATR vs prior `lookback_days` rolling percentile.

    Returns NaN if not enough history. Otherwise a float in [0, 1].
    """
    if len(bars_15m) < lookback_days * 96:
        return float("nan")
    idx = pd.to_datetime(bars_15m.index)
    df = pd.DataFrame({
        "High": bars_15m["High"].values,
        "Low": bars_15m["Low"].values,
        "Close": bars_15m["Close"].values,
    }, index=idx)
    daily = df.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(daily) < lookback_days + 2:
        return float("nan")
    d_atr = atr(daily["High"], daily["Low"], daily["Close"], atr_period)
    window = d_atr.iloc[-(lookback_days + 1):-1]  # PRIOR closed days (no lookahead)
    today = d_atr.iloc[-1]
    if not np.isfinite(today) or window.dropna().empty:
        return float("nan")
    # Where does today's ATR sit vs the recent distribution?
    rank = (window < today).sum() / len(window.dropna())
    return float(rank)


def _dist_ema_ok(side: str, close: float, ema_v: float) -> bool:
    if not np.isfinite(ema_v) or ema_v <= 0:
        return False
    ratio = close / ema_v - 1.0
    if side == "long":
        return ratio <= MAX_DISTANCE_ABOVE_EMA_PCT
    return ratio >= -MAX_DISTANCE_BELOW_EMA_PCT


# ---- v2 TA confirmations (pure, operate on a slice) -----------------------
def _trendline_confirm(side: str, high: pd.Series, low: pd.Series,
                       sh: pd.Series, sl: pd.Series, price: float) -> bool:
    # Match strategy class slice exactly: iloc[i-200 : i+1] = 201 bars (inclusive)
    sub_high = high.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_low = low.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_sh = sh.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_sl = sl.iloc[-(SWING_LOOKBACK_BARS + 1):]
    bar_idx = len(sub_high) - 1
    if side == "long":
        line = trendline_from_swings(sub_sl, sub_low, n_recent=3)
        if line is None:
            return False
        d = trendline_proximity_pct(price, *line, bar_idx)
        return d is not None and 0 <= d <= TRENDLINE_MAX_DIST_PCT
    line = trendline_from_swings(sub_sh, sub_high, n_recent=3)
    if line is None:
        return False
    d = trendline_proximity_pct(price, *line, bar_idx)
    return d is not None and -TRENDLINE_MAX_DIST_PCT <= d <= 0


def _sr_confirm(side: str, high: pd.Series, low: pd.Series,
                sh: pd.Series, sl: pd.Series, price: float) -> bool:
    # Match strategy class slice exactly: iloc[i-200 : i+1] = 201 bars (inclusive)
    sub_high = high.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_low = low.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_sh = sh.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_sl = sl.iloc[-(SWING_LOOKBACK_BARS + 1):]
    if side == "long":
        sl_prices = sub_low.values[np.where(sub_sl.values)[0]]
        zones = sr_zones(sl_prices, SR_CLUSTER_TOLERANCE_PCT)
        d = nearest_sr_zone_distance_pct(price, zones, "below")
        return d is not None and d <= SR_MAX_DIST_PCT
    sh_prices = sub_high.values[np.where(sub_sh.values)[0]]
    zones = sr_zones(sh_prices, SR_CLUSTER_TOLERANCE_PCT)
    d = nearest_sr_zone_distance_pct(price, zones, "above")
    return d is not None and d <= SR_MAX_DIST_PCT


def _fib_confirm(side: str, high: pd.Series, low: pd.Series,
                 sh: pd.Series, sl: pd.Series, price: float) -> bool:
    # Match strategy class slice exactly: iloc[i-200 : i+1] = 201 bars (inclusive)
    sub_high = high.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_low = low.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_sh = sh.iloc[-(SWING_LOOKBACK_BARS + 1):]
    sub_sl = sl.iloc[-(SWING_LOOKBACK_BARS + 1):]
    pair = recent_swing_pair(sub_sh, sub_sl, sub_high, sub_low, SWING_LOOKBACK_BARS)
    if pair is None:
        return False
    sh_p, sl_p = pair
    if side == "long":
        fib = fib_retracement_distance_pct(price, sh_p, sl_p)
    else:
        fib = fib_retracement_distance_pct(price, sl_p, sh_p)
    return fib is not None and fib[1] <= FIB_MAX_DIST_PCT


def _ta_confirmations_count(side: str, high: pd.Series, low: pd.Series,
                            sh: pd.Series, sl: pd.Series, price: float) -> int:
    c = 0
    if _trendline_confirm(side, high, low, sh, sl, price):
        c += 1
    if _sr_confirm(side, high, low, sh, sl, price):
        c += 1
    if _fib_confirm(side, high, low, sh, sl, price):
        c += 1
    return c


def evaluate_signal_v3all_wider4(
    bars_15m: pd.DataFrame,
    funding_rate: float,
    params: dict,
) -> tuple[str | None, float, float, dict]:
    """Evaluate v3-all-wider-4 signal on the last closed bar.

    bars_15m must have columns: Open, High, Low, Close, Volume — and have at
    least SWING_LOOKBACK_BARS + 50 rows. Funding is the latest 8h rate.

    Returns (side, sl_distance, tp_distance, debug):
      side ∈ {'long', 'short', None}
      sl_distance, tp_distance — in PRICE units (NaN if side is None)
      debug — dict for logging
    """
    s = params["strategy"]
    warmup = max(s["mf_trend_ema_period"], s["volume_ma_period"], s["rsi_period"]) + 5
    if len(bars_15m) < max(warmup, SWING_LOOKBACK_BARS + 10):
        return None, float("nan"), float("nan"), {"reason": "warmup"}

    close, high, low, volume = (
        bars_15m["Close"], bars_15m["High"], bars_15m["Low"], bars_15m["Volume"]
    )

    rsi_v = rsi(close, s["rsi_period"]).iloc[-1]
    vol_sma_v = sma(volume, s["volume_ma_period"]).iloc[-1]
    trend_ema_v = ema(close, s["mf_trend_ema_period"]).iloc[-1]
    cur_vol = volume.iloc[-1]
    cur_close = close.iloc[-1]
    atr_v = atr(high, low, close, ATR_PERIOD).iloc[-1]

    if not all(np.isfinite([rsi_v, vol_sma_v, trend_ema_v, cur_vol, cur_close, atr_v])):
        return None, float("nan"), float("nan"), {"reason": "nan_indicators"}

    # --- Base v1 filters ---
    vol_ok = cur_vol > s["volume_multiple"] * vol_sma_v
    trend_up = cur_close > trend_ema_v
    funding_long_blocked = (s["require_funding_not_extreme"]
                            and funding_rate > s["funding_extreme_threshold"])
    funding_short_blocked = (s["require_funding_not_extreme"]
                             and funding_rate < -s["funding_extreme_threshold"])

    debug = {
        "rsi": float(rsi_v), "vol_ok": bool(vol_ok), "trend_up": bool(trend_up),
        "funding": float(funding_rate), "atr": float(atr_v),
        "ema": float(trend_ema_v), "cur_close": float(cur_close),
    }

    # --- v3 vol-regime gate (applies to entries only) ---
    daily_pctile = _daily_atr_percentile(bars_15m, ATR_PERIOD, VOL_REGIME_LOOKBACK_DAYS)
    debug["vol_regime_pctile"] = float(daily_pctile) if np.isfinite(daily_pctile) else None
    if np.isfinite(daily_pctile) and daily_pctile > VOL_REGIME_MAX_PCTILE:
        return None, float("nan"), float("nan"), {**debug, "reason": "vol_regime_blocked"}

    # --- Swing detection for TA confirmations ---
    sh_mask, sl_mask = swing_high_low(high, low, SWING_K)

    # --- Stop / target distances (ATR-based, k=4.0 / 8.0) ---
    sl_dist = ATR_SL_K * float(atr_v)
    tp_dist = ATR_TP_K * float(atr_v)

    # --- LONG candidate ---
    long_base = (
        rsi_v < s["rsi_long_threshold"] and vol_ok and trend_up
        and not funding_long_blocked
        and _dist_ema_ok("long", cur_close, trend_ema_v)
    )
    if long_base:
        n_conf = _ta_confirmations_count("long", high, low, sh_mask, sl_mask, cur_close)
        debug["ta_conf_long"] = n_conf
        if n_conf >= CONFIRMATIONS_REQUIRED:
            return "long", sl_dist, tp_dist, debug

    # --- SHORT candidate ---
    short_base = (
        rsi_v > s["rsi_short_threshold"] and vol_ok and not trend_up
        and not funding_short_blocked
        and _dist_ema_ok("short", cur_close, trend_ema_v)
    )
    if short_base:
        n_conf = _ta_confirmations_count("short", high, low, sh_mask, sl_mask, cur_close)
        debug["ta_conf_short"] = n_conf
        if n_conf >= CONFIRMATIONS_REQUIRED:
            return "short", sl_dist, tp_dist, debug

    return None, float("nan"), float("nan"), debug
