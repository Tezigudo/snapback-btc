"""Mine the trade-level data: what separates WINNING fades from LOSING ones?

Goal: a data-driven entry checklist (and any exit signal) — tested IS->OOS so we
don't hand over an overfit rule. The steelman already killed day_ret/liquidity/
top-N slices OOS; here we test the UNtested, economically-motivated levers:
  - BTC regime that day (don't fade an alt while the whole tape is ripping)
  - entry DEPTH (entry/peak: shallow near-peak vs chasing a deep rollover)
  - TP distance (is the pre-pump support an achievable target or a moonshot?)
  - entry hour / time-since-peak
Winner = net_ret > 0. IS = entries <2025, OOS = 2025+.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_data as pfd  # noqa: E402

CACHE = ROOT / "data" / "pumpfade"
OOS = pd.Timestamp("2025-01-01", tz="UTC")


def load() -> pd.DataFrame:
    df = pd.read_parquet(CACHE / "trades_intraday.parquet")
    t = df[df.reason.isin(["TP", "STOP", "TIME", "SETTLE"])].copy()
    t["et"] = pd.to_datetime(t["entry_time"], utc=True)
    t["day_n"] = pd.to_datetime(t["day"], utc=True).dt.normalize()
    # features
    t["entry_peak"] = t["entry"] / t["peak"]                 # 1=at peak, <1 entered below
    t["tp_dist"] = 1.0 - t["tp"] / t["entry"]                # fraction down to support target
    t["hour"] = t["et"].dt.hour
    t["win"] = (t["net_ret"] > 0).astype(int)
    # BTC regime that day
    btc = pfd.load_daily("BTCUSDT")
    btc = btc[~btc.index.duplicated()].sort_index()
    btc_ret = (btc["close"] / btc["close"].shift(1) - 1.0)
    btc_ret.index = btc_ret.index.normalize()
    t["btc_ret"] = t["day_n"].map(btc_ret).astype(float)
    return t


def winner_loser(t: pd.DataFrame) -> None:
    print("=== what separates WINNERS from LOSERS? (means; all data, descriptive) ===")
    feats = ["day_ret", "entry_peak", "tp_dist", "hour", "btc_ret", "qvol", "hold_h"]
    w, l = t[t.win == 1], t[t.win == 0]
    print(f"{'feature':<12}{'winners':>12}{'losers':>12}{'spread':>12}")
    for f in feats:
        wv, lv = w[f].median(), l[f].median()
        print(f"{f:<12}{wv:>12.3f}{lv:>12.3f}{wv-lv:>12.3f}")
    print(f"\n  n winners {len(w)}  losers {len(l)}  base win-rate {100*t.win.mean():.0f}%")


def ev(d: pd.DataFrame) -> str:
    if len(d) == 0:
        return "n=0"
    return (f"n={len(d):<4} EV {100*d.net_ret.mean():+6.2f}%  win {100*(d.net_ret>0).mean():>4.0f}%  "
            f"med {100*d.net_ret.median():+5.1f}%  worst {100*d.net_ret.min():>6.0f}%")


def test_filter(t: pd.DataFrame, name: str, mask: pd.Series) -> None:
    sub = t[mask]
    isd, oos = sub[sub.et < OOS], sub[sub.et >= OOS]
    print(f"\n{name}  (keeps {len(sub)}/{len(t)} = {100*len(sub)/len(t):.0f}%)")
    print(f"   ALL  {ev(sub)}")
    print(f"   IS   {ev(isd)}")
    print(f"   OOS  {ev(oos)}   <- must be positive to be a real edge")


def main() -> int:
    t = load()
    winner_loser(t)
    print("\n=== ENTRY-FILTER TESTS (economically motivated, IS->OOS) ===")
    test_filter(t, "F1 calm tape: BTC day ret <= 0", t.btc_ret <= 0)
    test_filter(t, "F2 calm tape strict: BTC day ret <= -1%", t.btc_ret <= -0.01)
    test_filter(t, "F2b risk-on (control): BTC day ret > +2%", t.btc_ret > 0.02)
    test_filter(t, "F3 shallow entry (near peak): entry/peak >= 0.80", t.entry_peak >= 0.80)
    test_filter(t, "F3b deep chase (control): entry/peak < 0.6", t.entry_peak < 0.60)
    test_filter(t, "F4 achievable TP: support within 25%", t.tp_dist <= 0.25)
    test_filter(t, "F5 calm tape + shallow entry", (t.btc_ret <= 0) & (t.entry_peak >= 0.80))
    test_filter(t, "F6 calm + shallow + achievable TP",
                (t.btc_ret <= 0) & (t.entry_peak >= 0.80) & (t.tp_dist <= 0.30))
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
