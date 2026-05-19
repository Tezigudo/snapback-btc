"""Multi-timeframe confluence analyzer.

Given a timestamp, compute per-TF state and a confluence score.
Result is a dict that can be used by future strategy logic or rendered
to a report.

Confluence score (signed, range roughly [-15, +15]):
  Per TF: +1 for "up (weak)" / "up (partial)", +2 for "UP (strong)",
          -1 for "down (weak)" / "down (partial)", -2 for "DOWN (strong)",
          0 for MIXED / unknown
  Weighted by TF: 30m=1, 1h=1, 4h=2, 1d=2, 1w=3 (HTF dominates).
  Plus: RSI extreme bonus
        SAR confluence bonus

Recommendation:
  score >= +6  -> STRONG LONG bias
  score in [+3,+5] -> LONG bias
  score in [-2,+2] -> NEUTRAL (no trade)
  score in [-5,-3] -> SHORT bias
  score <= -6  -> STRONG SHORT bias
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

from strategy.indicators import ema, parabolic_sar, rsi
from tools.chart_mtf import TF_RULES, _classify_trend, _read_15m, resample_ohlcv

TF_WEIGHTS = {"30m": 1, "1h": 1, "4h": 2, "1d": 2, "1w": 3}

TREND_SCORE = {
    "UP (strong)": +2,
    "up (weak)": +1,
    "up (partial)": +1,
    "DOWN (strong)": -2,
    "down (weak)": -1,
    "down (partial)": -1,
    "MIXED": 0,
    "unknown": 0,
    "no_data": 0,
}


def analyze_tf(win: pd.DataFrame, tf_label: str) -> dict:
    if win.empty:
        return {"tf": tf_label, "trend": "no_data"}
    close = win["Close"].iloc[-1]
    ema_vals = {p: float(ema(win["Close"], p).iloc[-1])
                for p in (7, 24, 50, 100, 200) if len(win) >= p}
    sar_series = parabolic_sar(win["High"], win["Low"])
    sar = float(sar_series.iloc[-1]) if pd.notna(sar_series.iloc[-1]) else close
    rsi_v = float(rsi(win["Close"], 14).iloc[-1])
    trend = _classify_trend(ema_vals, close, sar)
    sar_side = "up" if close > sar else "down"
    return {
        "tf": tf_label,
        "close": float(close),
        "ema": ema_vals,
        "sar": sar,
        "sar_side": sar_side,
        "rsi": rsi_v,
        "trend": trend,
        "trend_score": TREND_SCORE[trend],
        "weight": TF_WEIGHTS[tf_label],
    }


def confluence(timestamp: pd.Timestamp, tfs: list[str],
               base_15m: pd.DataFrame | None = None) -> dict:
    """Compute MTF confluence at `timestamp`.

    `base_15m` can be passed by callers (e.g. backtest loops) that want to
    avoid re-reading the parquet file on every call. If None, the file is
    read fresh.
    """
    if base_15m is None:
        base_15m = _read_15m()
    base = base_15m.loc[:timestamp]
    per_tf = []
    for tf in tfs:
        rule, nbars = TF_RULES[tf]
        win = resample_ohlcv(base, rule).tail(nbars)
        per_tf.append(analyze_tf(win, tf))

    weighted = sum(p.get("trend_score", 0) * p.get("weight", 0) for p in per_tf)

    # RSI extreme bonus: oversold on entry TF (30m) in UP context = strong LONG
    rsi_30m = next((p["rsi"] for p in per_tf if p["tf"] == "30m"), None)
    rsi_bonus = 0
    if rsi_30m is not None:
        if rsi_30m < 30 and weighted > 0:
            rsi_bonus = +1  # oversold within uptrend = pullback buy
        elif rsi_30m > 70 and weighted < 0:
            rsi_bonus = -1  # overbought within downtrend = rally sell

    # SAR confluence bonus: HTF SAR agreeing with overall direction
    sar_bonus = 0
    htf_sars = [p["sar_side"] for p in per_tf if p["tf"] in ("4h", "1d", "1w") and "sar_side" in p]
    if htf_sars:
        ups = sum(1 for s in htf_sars if s == "up")
        downs = sum(1 for s in htf_sars if s == "down")
        if ups == len(htf_sars) and weighted > 0:
            sar_bonus = +1
        elif downs == len(htf_sars) and weighted < 0:
            sar_bonus = -1

    score = weighted + rsi_bonus + sar_bonus

    if score >= 6:
        rec = "STRONG LONG"
    elif score >= 3:
        rec = "LONG"
    elif score <= -6:
        rec = "STRONG SHORT"
    elif score <= -3:
        rec = "SHORT"
    else:
        rec = "NEUTRAL"

    return {
        "timestamp": str(timestamp),
        "per_tf": per_tf,
        "weighted_trend_score": weighted,
        "rsi_bonus": rsi_bonus,
        "sar_bonus": sar_bonus,
        "score": score,
        "recommendation": rec,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timestamp", required=True)
    p.add_argument("--tfs", default="30m,1h,4h,1d,1w")
    args = p.parse_args()

    ts = pd.Timestamp(datetime.fromisoformat(args.timestamp))
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    tfs = [t.strip() for t in args.tfs.split(",")]
    r = confluence(ts, tfs)
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
