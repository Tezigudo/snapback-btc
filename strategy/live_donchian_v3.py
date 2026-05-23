"""Live signal evaluator for Donchian-v3 cons.

Pure-function port of `DonchianBreakoutBTCv3` (strategy/signals_donchian.py)
for use in `bot.py`. Reads recent OHLCV bars, evaluates the LAST CLOSED
bar, returns (side, sl_distance, tp_distance, debug).

LIVE DIVERGES FROM BACKTEST in one place. The backtest exits on Donchian
exit-channel (20-bar opposite extreme). The live bot uses a fixed-multiple
TP because the existing bot loop is fire-and-forget on SL/TP brackets —
managing channel exits would require a separate live-position monitor loop.

To approximate "let winners run" without a per-tick exit monitor:
  - SL distance = atr_sl_multiple × ATR(20, 4h)   = 1.5 × ATR (cons)
  - TP distance = 5 × ATR(20, 4h)                  ≈ 3.3:1 R:R
  - Time stop at `time_stop_bars` (48 bars × 4h = 8 days)

This is acceptable because the realistic backtest sim showed Donchian-cons
wins last ~3-5×ATR on average. The 14-day dry-run on the droplet is
designed to surface any meaningful divergence before live trading.

Returns:
  side ∈ {'long', 'short', None}
  sl_distance, tp_distance — price units (NaN if side is None)
  debug — dict for logging
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import atr, ema

# Cons params (locked 2026-05-23). See config/params_donchian.yaml.
PERIOD_ENTRY = 80
PERIOD_EXIT = 20             # for debug logging only — live exits on SL/TP
ATR_PERIOD = 20
ATR_SL_K = 1.5
ATR_TP_K = 5.0               # simplified — see module docstring
REGIME_EMA_PERIOD = 120
REGIME_SLOPE_WINDOW = 30
SLOPE_TREND_THRESHOLD_PCT = 0.03


def _ema_slope_signed(close: pd.Series, ema_period: int, slope_window: int) -> float:
    """Mirror of strategy.regime_classifier.ema_slope_signed at the last bar.

    Returns the linear-regression slope of EMA(close, ema_period) over the
    last `slope_window` points, as a fraction of the EMA mid-value
    (positive = uptrend, negative = downtrend).
    """
    e = ema(close, ema_period)
    if len(e) < slope_window + 1:
        return float("nan")
    window = e.iloc[-slope_window:].values
    if not np.all(np.isfinite(window)):
        return float("nan")
    x = np.arange(slope_window, dtype=float)
    slope = np.polyfit(x, window, 1)[0]   # absolute slope per bar
    mid = float(np.mean(window))
    if mid <= 0:
        return float("nan")
    # Normalised to % of mid level, multiplied by the window length so
    # threshold has the same meaning as the backtest's directional gate
    # (where slope_trend_threshold_pct is % over the whole window).
    return float(slope * slope_window / mid)


def evaluate_signal_donchian_v3(
    bars_4h: pd.DataFrame,
    funding_rate: float,             # unused; Donchian doesn't gate on funding
    params: dict,
) -> tuple[str | None, float, float, dict]:
    """Evaluate Donchian-v3 cons entry on the last closed 4h bar.

    bars_4h must have columns Open, High, Low, Close, Volume and at least
    max(PERIOD_ENTRY, REGIME_EMA_PERIOD + REGIME_SLOPE_WINDOW) + 10 rows.

    Returns (side, sl_distance, tp_distance, debug).
    """
    _ = funding_rate   # signature compat with other evaluators
    s = params.get("strategy", {})
    period_entry = int(s.get("donchian_period_entry", PERIOD_ENTRY))
    period_exit = int(s.get("donchian_period_exit", PERIOD_EXIT))
    atr_period = int(s.get("atr_period", ATR_PERIOD))
    atr_sl_k = float(s.get("atr_sl_multiple", ATR_SL_K))
    regime_ema_p = int(s.get("regime_ema_period", REGIME_EMA_PERIOD))
    regime_slope_w = int(s.get("regime_slope_window", REGIME_SLOPE_WINDOW))
    slope_thr = float(s.get("slope_trend_threshold_pct", SLOPE_TREND_THRESHOLD_PCT))

    warmup = max(period_entry, regime_ema_p + regime_slope_w, atr_period) + 5
    if len(bars_4h) < warmup:
        return None, float("nan"), float("nan"), {"reason": "warmup", "have": len(bars_4h), "need": warmup}

    close, high, low = bars_4h["Close"], bars_4h["High"], bars_4h["Low"]

    # Donchian channel: rolling max/min of CLOSE over last N bars, shifted 1
    # to use bars strictly BEFORE the current bar (no lookahead).
    upper = close.rolling(period_entry, min_periods=period_entry).max().shift(1).iloc[-1]
    lower = close.rolling(period_entry, min_periods=period_entry).min().shift(1).iloc[-1]
    exit_upper = close.rolling(period_exit, min_periods=period_exit).max().shift(1).iloc[-1]
    exit_lower = close.rolling(period_exit, min_periods=period_exit).min().shift(1).iloc[-1]
    atr_v = atr(high, low, close, atr_period).iloc[-1]
    cur_close = float(close.iloc[-1])

    if not all(np.isfinite([upper, lower, atr_v, cur_close])):
        return None, float("nan"), float("nan"), {
            "reason": "nan_indicators",
            "upper": float(upper) if np.isfinite(upper) else None,
            "lower": float(lower) if np.isfinite(lower) else None,
            "atr": float(atr_v) if np.isfinite(atr_v) else None,
        }

    sl_dist = atr_sl_k * float(atr_v)
    tp_dist = ATR_TP_K * float(atr_v)

    # Regime gate (signed EMA-slope). Only active when slope_thr > 0.
    gate_on = slope_thr > 0
    slope = _ema_slope_signed(close, regime_ema_p, regime_slope_w) if gate_on else 0.0

    debug = {
        "cur_close": cur_close, "upper": float(upper), "lower": float(lower),
        "exit_upper": float(exit_upper), "exit_lower": float(exit_lower),
        "atr": float(atr_v), "sl_dist": float(sl_dist), "tp_dist": float(tp_dist),
        "slope": float(slope) if np.isfinite(slope) else None,
        "gate_on": gate_on, "slope_threshold": slope_thr,
    }

    # LONG: close > upper-channel AND (gate off OR slope >= +threshold)
    if cur_close > upper:
        if gate_on and (not np.isfinite(slope) or slope < slope_thr):
            return None, float("nan"), float("nan"), {**debug, "reason": "long_blocked_by_slope"}
        return "long", sl_dist, tp_dist, debug

    # SHORT: close < lower-channel AND (gate off OR slope <= -threshold)
    if cur_close < lower:
        if gate_on and (not np.isfinite(slope) or slope > -slope_thr):
            return None, float("nan"), float("nan"), {**debug, "reason": "short_blocked_by_slope"}
        return "short", sl_dist, tp_dist, debug

    return None, float("nan"), float("nan"), {**debug, "reason": "no_breakout"}
