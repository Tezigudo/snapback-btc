"""multifactor-v1 — live signal evaluator (pure function).

Mirror of `live_v3all_wider4.py` for the v1 strategy. Both modules export
`evaluate_signal*(bars_15m, funding_rate, params)` callable from the bot
loop and from validation harnesses.

v1 is the deployable default. Backtest evidence: 5 OOS windows 2022-2025
H1 → +55.73% compounded, 4 of 5 windows positive. Worst window 2024 H1
(-12.56%, chop). Kill switch at -18% equity drawdown.

LONG entry, ALL of:
  RSI(14, 15m) < rsi_long_threshold      (default 40)
  close > EMA(200, 15m)                  (trend filter)
  volume(15m) > volume_multiple × SMA(20, vol)  (default 2×)
  funding_rate <= funding_extreme_threshold     (default +0.05%/8h)

SHORT entry is the mirror.

Exit (sized by bot.py, not here):
  TP at +/- tp_pct × close
  SL at -/+ sl_pct × close
  Time-stop at time_stop_bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import ema, rsi, sma


def evaluate_signal(
    bars_15m: pd.DataFrame,
    funding_rate: float,
    params: dict,
) -> tuple[str | None, dict]:
    """Return ('long'/'short'/None, debug_dict) for the last CLOSED 15m bar.

    Pure-function port of `DayTradeMultiFactorBTC._long_signal/_short_signal`
    (see strategy/signals_multifactor.py for the backtest version).
    """
    s = params["strategy"]
    warm = max(s["mf_trend_ema_period"], s["volume_ma_period"], s["rsi_period"]) + 5
    if len(bars_15m) < warm:
        return None, {"reason": "warmup"}

    close = bars_15m["Close"]
    vol = bars_15m["Volume"]

    rsi_v = rsi(close, s["rsi_period"]).iloc[-1]
    vol_sma_v = sma(vol, s["volume_ma_period"]).iloc[-1]
    trend_ema_v = ema(close, s["mf_trend_ema_period"]).iloc[-1]
    cur_vol = vol.iloc[-1]
    cur_close = close.iloc[-1]

    if not all(np.isfinite([rsi_v, vol_sma_v, trend_ema_v, cur_vol, cur_close])):
        return None, {"reason": "nan_indicators"}

    vol_ok = cur_vol > s["volume_multiple"] * vol_sma_v
    trend_up = cur_close > trend_ema_v
    funding_long_blocked = (s["require_funding_not_extreme"]
                            and funding_rate > s["funding_extreme_threshold"])
    funding_short_blocked = (s["require_funding_not_extreme"]
                             and funding_rate < -s["funding_extreme_threshold"])

    debug = {
        "ts": bars_15m.index[-1].isoformat(),
        "rsi": float(rsi_v), "vol_sma": float(vol_sma_v), "trend_ema": float(trend_ema_v),
        "cur_vol": float(cur_vol), "cur_close": float(cur_close),
        "vol_ok": bool(vol_ok), "trend_up": bool(trend_up),
        "funding_rate": funding_rate,
    }

    if (rsi_v < s["rsi_long_threshold"] and vol_ok and trend_up
            and not funding_long_blocked):
        return "long", debug
    if (rsi_v > s["rsi_short_threshold"] and vol_ok and not trend_up
            and not funding_short_blocked):
        return "short", debug
    return None, debug
