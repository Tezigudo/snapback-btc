"""Characterize each of my REAL futures trades against the pump-fade pattern +
classify pump-fade vs 'inverted strategy'. Pulls 1h/1d klines (data.binance.vision)
around each real entry and computes the setup the user actually faced.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_data as pfd  # noqa: E402

WEIRD = {"PORTALUSDT", "KATUSDT"}  # small/obscure; BTC/WLD = "ordinary"


def fetch_live_1h(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Live fapi 1h klines (serves the CURRENT month, which data.binance.vision lacks)."""
    rows, cur = [], start_ms
    while cur < end_ms:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": symbol, "interval": "1h", "startTime": cur,
                                 "endTime": end_ms, "limit": 1500}, timeout=30)
        r.raise_for_status()
        b = r.json()
        if not b:
            break
        rows += b
        cur = b[-1][0] + 3_600_000
        if len(b) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "ct",
                                     "qv", "n", "tb", "tq", "ig"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    return df[["open", "high", "low", "close", "volume"]]


def main() -> int:
    trades = json.loads(Path("/tmp/my_futures_history.json").read_text())["trades"]
    trades.sort(key=lambda t: t["open_ms"])
    for t in trades:
        sym = t["symbol"]
        o = pd.Timestamp(t["open_ms"], unit="ms", tz="UTC")
        c = pd.Timestamp(t["close_ms"], unit="ms", tz="UTC")
        try:
            h1 = fetch_live_1h(sym, int(t["open_ms"]) - 11 * 86_400_000, int(t["close_ms"]) + 2 * 86_400_000)
            d1 = pfd.load_daily(sym)
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: data err {e}"); continue
        h1 = h1[h1.volume > 0].sort_index()
        if h1.empty:
            print(f"{sym}: no 1h data"); continue
        # entry bar
        pre = h1.loc[:o]
        if len(pre) < 26:
            print(f"{sym} {o:%Y-%m-%d}: thin pre-entry data"); continue
        entry_close = float(pre["close"].iloc[-1])
        ret24 = entry_close / float(pre["close"].iloc[-25]) - 1 if len(pre) >= 25 else float("nan")
        ret72 = entry_close / float(pre["close"].iloc[-73]) - 1 if len(pre) >= 73 else float("nan")
        win3d = h1.loc[o - timedelta(days=3):o]
        peak = float(win3d["high"].max()) if len(win3d) else float("nan")
        rolled = 1 - t["entry_px"] / peak if peak else float("nan")
        peak_bar_vol = float(win3d.loc[win3d["high"].idxmax(), "volume"]) if len(win3d) else float("nan")
        entry_vol = float(pre["volume"].iloc[-3:].mean())
        volR = entry_vol / peak_bar_vol if peak_bar_vol else float("nan")
        # during-trade adverse + outcome
        dur = h1.loc[o:c]
        mae = (float(dur["high"].max()) / t["entry_px"] - 1) if len(dur) else float("nan")  # short: + = against
        # max 24h-high move INTO entry (captures an intraday spike even if pulled back by entry)
        pre24 = pre.iloc[-25:]
        dret = float(pre24["high"].max()) / float(pre["close"].iloc[-25]) - 1 if len(pre) >= 25 else float("nan")
        cls = "weird" if sym in WEIRD else "ordinary"
        print(f"\n=== {sym}  {t['side']}  {o:%Y-%m-%d %H:%M}  ({cls} coin)  PnL ${t['realizedPnl']:+.2f} ===")
        print(f"  entry {t['entry_px']:.6g} -> exit {t['exit_px']:.6g}  held {(t['close_ms']-t['open_ms'])/3.6e6:.1f}h")
        print(f"  pump INTO entry: 24h {ret24*100:+.0f}%  72h {ret72*100:+.0f}%  | 24h-HIGH spike into entry: {dret*100:+.0f}%")
        print(f"  3d-peak {peak:.6g}  entry was {rolled*100:+.0f}% below peak (rolled-over?)")
        print(f"  VOLUME at entry vs peak-bar: {volR:.2f}  ({'RISING' if volR>=0.8 else 'drying up' if volR<=0.5 else 'mid'})")
        print(f"  max move AGAINST after entry: {mae*100:+.0f}%   (short {'survived' if mae<0.15 else 'got run over' if mae>0.4 else 'pressured'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
