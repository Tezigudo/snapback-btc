"""Test the volume-exhaustion rule on the user's REAL short trades (n up to 148).
For each SHORT, pull 1h klines around entry (live fapi; vision fallback) and compute
the setup: pump into entry, volume rising vs drying, entry vs 3d-peak. Then ask:
do volume-drying shorts beat volume-rising ones? Would filtering rising-vol flip the book?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_data as pfd  # noqa: E402

MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "LTCUSDT", "AVAXUSDT",
          "SUIUSDT", "APTUSDT", "DOGEUSDT", "LINKUSDT", "BNBUSDT", "TIAUSDT", "ONDOUSDT"}
TF = 3_600_000


def live_1h(sym: str, a: int, b: int) -> pd.DataFrame:
    rows, cur = [], a
    while cur < b:
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                             params={"symbol": sym, "interval": "1h", "startTime": cur, "endTime": b, "limit": 1500},
                             timeout=30)
            if r.status_code != 200:
                break
            d = r.json()
        except Exception:  # noqa: BLE001
            break
        if not d:
            break
        rows += d
        cur = d[-1][0] + TF
        if len(d) < 1500:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume",
                                     "ct", "qv", "n", "tb", "tq", "ig"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    for k in ("open", "high", "low", "close", "volume"):
        df[k] = pd.to_numeric(df[k])
    return df[["open", "high", "low", "close", "volume"]]


def klines_for(sym: str, a_ms: int, b_ms: int) -> pd.DataFrame:
    df = live_1h(sym, a_ms - 11 * 86_400_000, b_ms + 2 * 86_400_000)
    if not df.empty and df["volume"].sum() > 0:
        return df[df["volume"] > 0].sort_index()
    try:  # delisted -> vision dump
        months = pfd.months_spanning(pd.Timestamp(a_ms - 11 * 86_400_000, unit="ms", tz="UTC").to_pydatetime(),
                                     pd.Timestamp(b_ms + 2 * 86_400_000, unit="ms", tz="UTC").to_pydatetime())
        df = pfd.load_intraday(sym, "1h", months)
        return df[df["volume"] > 0].sort_index() if not df.empty else df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def main() -> int:
    trades = json.loads(Path("/tmp/full_positions.json").read_text())
    shorts = [t for t in trades if t["side"] == "SHORT" and t["exit_px"] is not None]
    rows = []
    bycache: dict[str, pd.DataFrame] = {}
    for i, t in enumerate(shorts):
        s = t["symbol"]
        if s not in bycache:
            bycache[s] = klines_for(s, t["open_ms"], t["close_ms"])
        k = bycache[s]
        if k.empty:
            continue
        o = pd.Timestamp(t["open_ms"], unit="ms", tz="UTC")
        pre = k.loc[:o]
        if len(pre) < 26:
            continue
        ret72 = float(pre["close"].iloc[-1] / pre["close"].iloc[-73] - 1) if len(pre) >= 73 else float("nan")
        dret = float(pre["high"].iloc[-25:].max() / pre["close"].iloc[-25] - 1)
        w3 = k.loc[o - pd.Timedelta(days=3):o]
        peak = float(w3["high"].max()) if len(w3) else float("nan")
        rolled = (1 - t["entry_px"] / peak) if peak else float("nan")
        pkvol = float(w3.loc[w3["high"].idxmax(), "volume"]) if len(w3) else float("nan")
        evol = float(pre["volume"].iloc[-3:].mean())
        volR = evol / pkvol if pkvol else float("nan")
        rows.append({**t, "ret72": ret72, "dret": dret, "rolled": rolled, "volR": volR})
    df = pd.DataFrame(rows)
    df.to_parquet("/tmp/real_shorts_setup.parquet")
    print(f"shorts with setup features: {len(df)} / {len(shorts)}")

    def agg(name, d):
        if len(d) == 0:
            print(f"  {name:<34} n=0"); return
        net = d["net"].sum()
        print(f"  {name:<34} n={len(d):<4} win {100*(d.net>0).mean():>4.0f}%  netPnL ${net:>+8.1f}  avg ${net/len(d):>+6.2f}")

    alt = df[df["symbol"].apply(lambda s: s not in MAJORS)]
    print("\n=== VOLUME RULE on real SHORT trades ===")
    print("ALL shorts:")
    agg("volume DRYING (volR<0.8)", df[df.volR < 0.8])
    agg("volume RISING (volR>=0.8) — skip", df[df.volR >= 0.8])
    print("\nPUMP-FADE book (alt shorts only):")
    agg("alt shorts ALL", alt)
    agg("  volume DRYING (volR<0.8)", alt[alt.volR < 0.8])
    agg("  volume RISING (volR>=0.8) — skip", alt[alt.volR >= 0.8])
    print("\nfaded an UP-move (dret>30%) vs shorted weakness:")
    agg("alt short, UP-move dret>30%", alt[alt.dret > 0.30])
    agg("alt short, NOT up (dret<=30%)", alt[alt.dret <= 0.30])
    print("\nbest combo — fade up-move AND volume drying:")
    agg("alt short, dret>30% & volR<0.8", alt[(alt.dret > 0.30) & (alt.volR < 0.8)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
