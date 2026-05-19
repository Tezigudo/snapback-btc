"""
At a specific timestamp where backtest fires but live port doesn't,
dump every gate decision side-by-side so we can spot which check disagrees.

Usage:
  uv run python -m tools.diff_one_signal 2025-10-05T10:15:00
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from exchange.data import load_funding, load_klines  # noqa: E402
from strategy.indicators import (  # noqa: E402
    atr, ema, fib_retracement_distance_pct, nearest_sr_zone_distance_pct,
    recent_swing_pair, rsi, sma, sr_zones, swing_high_low,
    trendline_from_swings, trendline_proximity_pct,
)
from strategy.live_v3all_wider4 import (  # noqa: E402
    CONFIRMATIONS_REQUIRED, MAX_DISTANCE_ABOVE_EMA_PCT, MAX_DISTANCE_BELOW_EMA_PCT,
    SR_CLUSTER_TOLERANCE_PCT, SR_MAX_DIST_PCT, SWING_K, SWING_LOOKBACK_BARS,
    TRENDLINE_MAX_DIST_PCT, FIB_MAX_DIST_PCT, VOL_REGIME_LOOKBACK_DAYS,
    VOL_REGIME_MAX_PCTILE, ATR_PERIOD, evaluate_signal_v3all_wider4,
)


def diff(ts_str: str) -> None:
    """Evaluate the timestamp two ways and dump gate state."""
    ts = pd.to_datetime(ts_str)
    print(f"\n=== Diff at {ts} ===\n")

    # Load enough data (240 days warmup)
    end = ts + timedelta(days=1)
    days_back = 300
    bars = load_klines("BTC/USDT:USDT", "15m", days_back=days_back, end=end.to_pydatetime().replace(tzinfo=UTC))
    bars.columns = [c.capitalize() for c in bars.columns]
    if bars.index.tz is not None:
        bars.index = bars.index.tz_convert("UTC").tz_localize(None)
    funding = load_funding("BTC/USDT:USDT", days_back=days_back, end=end.to_pydatetime().replace(tzinfo=UTC))
    if funding.index.tz is not None:
        funding.index = funding.index.tz_convert("UTC").tz_localize(None)

    # Latest bar ≤ ts (the bar the live port would have on hand at ts close)
    # For backtest convention: signal fires when bar i CLOSES, fill is at bar i+1 OPEN.
    # Backtest EntryTime = bar (i+1).Open ts. Live port called at end of bar i sees bars up to i.
    # So to diff at backtest EntryTime ts, we look at bar (ts - 15m).
    candidates = [ts, ts - pd.Timedelta(minutes=15), ts + pd.Timedelta(minutes=15)]
    for bar_ts in candidates:
        if bar_ts not in bars.index:
            continue
        print(f"--- evaluating with bar_ts = {bar_ts} ---")
        slice_ = bars.loc[:bar_ts]
        if len(slice_) < 250:
            print("  insufficient warmup")
            continue
        f_slice = funding.loc[:bar_ts]
        funding_rate = float(f_slice["funding_rate"].iloc[-1]) if not f_slice.empty else 0.0

        with open(REPO_ROOT / "config" / "params.yaml") as f:
            params = yaml.safe_load(f)
        s = params["strategy"]
        close, high, low, volume = (slice_["Close"], slice_["High"], slice_["Low"], slice_["Volume"])

        rsi_v = rsi(close, s["rsi_period"]).iloc[-1]
        vol_sma_v = sma(volume, s["volume_ma_period"]).iloc[-1]
        trend_ema_v = ema(close, s["mf_trend_ema_period"]).iloc[-1]
        cur_vol = volume.iloc[-1]
        cur_close = close.iloc[-1]
        atr_v = atr(high, low, close, ATR_PERIOD).iloc[-1]

        # Gate-by-gate
        print(f"  cur_close: {cur_close:.2f}")
        print(f"  rsi:       {rsi_v:.2f}  (long<{s['rsi_long_threshold']}: {'PASS' if rsi_v < s['rsi_long_threshold'] else 'FAIL'} · short>{s['rsi_short_threshold']}: {'PASS' if rsi_v > s['rsi_short_threshold'] else 'FAIL'})")
        print(f"  vol:       cur={cur_vol:.0f}  sma={vol_sma_v:.0f}  ratio={cur_vol/vol_sma_v:.2f}  (>{s['volume_multiple']}x: {'PASS' if cur_vol > s['volume_multiple']*vol_sma_v else 'FAIL'})")
        print(f"  ema200:    {trend_ema_v:.2f}  (close>ema for long: {'PASS' if cur_close > trend_ema_v else 'FAIL'})")
        print(f"  funding:   {funding_rate:.6f}  (long-blocked if >{s['funding_extreme_threshold']}: {'BLOCK' if funding_rate > s['funding_extreme_threshold'] else 'PASS'})")
        print(f"  ATR(14):   {atr_v:.2f}  (SL = 4× = {4*atr_v:.2f} = {4*atr_v/cur_close*100:.2f}%)")
        # dist-ema
        dist_ratio = (cur_close / trend_ema_v) - 1.0
        print(f"  dist-ema:  {dist_ratio*100:+.2f}%  (long-blocked if >+{MAX_DISTANCE_ABOVE_EMA_PCT*100:.0f}%: {'BLOCK' if dist_ratio > MAX_DISTANCE_ABOVE_EMA_PCT else 'PASS'})")
        # vol-regime
        from strategy.live_v3all_wider4 import _daily_atr_percentile
        pctile = _daily_atr_percentile(slice_, ATR_PERIOD, VOL_REGIME_LOOKBACK_DAYS)
        print(f"  vol-regime daily-ATR pctile: {pctile:.3f}  (blocked if >{VOL_REGIME_MAX_PCTILE}: {'BLOCK' if np.isfinite(pctile) and pctile > VOL_REGIME_MAX_PCTILE else 'PASS'})")
        # TA confirmations
        sh, sl = swing_high_low(high, low, SWING_K)
        sub_high = high.iloc[-SWING_LOOKBACK_BARS:]
        sub_low = low.iloc[-SWING_LOOKBACK_BARS:]
        sub_sh = sh.iloc[-SWING_LOOKBACK_BARS:]
        sub_sl = sl.iloc[-SWING_LOOKBACK_BARS:]
        # long
        long_tline = False
        line = trendline_from_swings(sub_sl, sub_low, n_recent=3)
        if line is not None:
            d = trendline_proximity_pct(cur_close, *line, len(sub_high)-1)
            long_tline = d is not None and 0 <= d <= TRENDLINE_MAX_DIST_PCT
            print(f"  TA trendline (long): d={d} max={TRENDLINE_MAX_DIST_PCT}  {'PASS' if long_tline else 'FAIL'}")
        else:
            print("  TA trendline (long): no line  FAIL")

        sl_prices = sub_low.values[np.where(sub_sl.values)[0]]
        zones = sr_zones(sl_prices, SR_CLUSTER_TOLERANCE_PCT)
        d = nearest_sr_zone_distance_pct(cur_close, zones, "below")
        long_sr = d is not None and d <= SR_MAX_DIST_PCT
        print(f"  TA SR (long):       d={d}  {'PASS' if long_sr else 'FAIL'}")

        pair = recent_swing_pair(sub_sh, sub_sl, sub_high, sub_low, SWING_LOOKBACK_BARS)
        long_fib = False
        if pair is not None:
            sh_p, sl_p = pair
            fib = fib_retracement_distance_pct(cur_close, sh_p, sl_p)
            long_fib = fib is not None and fib[1] <= FIB_MAX_DIST_PCT
            print(f"  TA Fib (long):      fib={fib}  {'PASS' if long_fib else 'FAIL'}")
        else:
            print("  TA Fib (long): no swing pair  FAIL")

        long_conf = int(long_tline) + int(long_sr) + int(long_fib)
        print(f"  TA confirmations (long): {long_conf}/3  (need {CONFIRMATIONS_REQUIRED})")

        # Also call evaluate_signal_v3all_wider4 to confirm
        side, sld, tpd, dbg = evaluate_signal_v3all_wider4(slice_, funding_rate, params)
        print(f"  evaluate_signal_v3all_wider4: side={side}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default — the first missed signal from our last validation run
        diff("2025-10-05T10:15:00")
    else:
        diff(sys.argv[1])
