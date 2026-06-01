"""Can the pump-fade be tuned via a SHORT time-stop (1.5-2 days) + SL choice?

User's hypothesis: a pump-fade is a fast trade — close by 36-48h. A time-stop is
a stop that CAN'T be whipsawed at a price level (the thing that sank every price
stop). Does a short hold also dodge the continuation tail?

Step 1 DIAGNOSTIC: from the existing 168h faithful run, when do wins / stops /
blow-ups actually happen relative to entry? (Tells us if a 48h cap helps.)
Step 2 MATRIX: intraday selection + dedup, vary max_hold x SL policy, IS/OOS,
worst trade, AND the risk-sized equity DD (a +EV config with -90% DD is still
uninvestable). Transparent grid — no cherry-pick.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_backtest as pf  # noqa: E402

CACHE = ROOT / "data" / "pumpfade"
OOS = pd.Timestamp("2025-01-01", tz="UTC")


def diagnostic() -> None:
    df = pd.read_parquet(CACHE / "trades_intraday.parquet")
    t = df[df.reason.isin(["TP", "STOP", "TIME", "SETTLE"])].copy()
    print("=== DIAGNOSTIC: timing (existing faithful run, 168h max_hold) ===")
    print(f"{'reason':<8}{'n':>5}{'med_hold_h':>12}{'<=24h%':>8}{'<=48h%':>8}{'>48h%':>8}")
    for r in ["TP", "STOP", "TIME", "SETTLE"]:
        g = t[t.reason == r]
        if len(g) == 0:
            continue
        h = g.hold_h
        print(f"{r:<8}{len(g):>5}{h.median():>12.0f}{100*(h<=24).mean():>8.0f}{100*(h<=48).mean():>8.0f}{100*(h>48).mean():>8.0f}")
    # where do the big losers exit?
    big = t[t.net_ret < -0.5]
    print(f"\nbig losers (net< -50%): n={len(big)}  median hold {big.hold_h.median():.0f}h  "
          f"<=48h {100*(big.hold_h<=48).mean():.0f}%  >48h {100*(big.hold_h>48).mean():.0f}%")
    print("  => if big losers cluster >48h, a 48h time-cap dodges them; if <=48h, it can't.")
    # how much TP profit is captured early?
    tp = t[t.reason == "TP"]
    print(f"TP trades: {100*(tp.hold_h<=48).mean():.0f}% reach support within 48h "
          f"(the rest need longer than a 1.5-2d hold would allow).")


def equity_dd(taken: pd.DataFrame, p: pf.Params) -> tuple[float, float]:
    ep = pf.equity_path(taken.sort_values("entry_time"), p)
    if ep.empty:
        return p.start_equity, 0.0
    e = ep["equity"].values
    peak = np.maximum.accumulate(e)
    return float(e[-1]), float(((e - peak) / peak).min() * 100)


def line(tag: str, d: pd.DataFrame, p: pf.Params) -> str:
    t = d[d.reason.isin(["TP", "STOP", "TIME", "SETTLE"])].copy()
    if len(t) == 0:
        return f"  {tag:<22} n=0"
    t["et"] = pd.to_datetime(t.entry_time, utc=True)
    isd, oos = t[t.et < OOS], t[t.et >= OOS]
    feq, dd = equity_dd(t, p)

    def ev(x):
        return f"{100*x.net_ret.mean():+6.2f}%" if len(x) else "   -  "
    return (f"  {tag:<22} n={len(t):<4} EVall {ev(t)}  IS {ev(isd)}  OOS {ev(oos)}  "
            f"win {100*(t.net_ret>0).mean():>4.0f}%  worst {100*t.net_ret.min():>6.0f}%  "
            f"finEq ${feq:>5.0f}  DD {dd:>5.0f}%")


def matrix() -> None:
    print("\n=== MATRIX: intraday select + dedup7 + thresh40 ; hold x SL policy ===")
    holds = [36, 48, 72, 168]
    stops = [
        ("peak (1.06x)", dict(sl_buf=0.06, max_stop_pct=0.0)),
        ("no price-stop", dict(sl_buf=100.0, max_stop_pct=0.0)),
        ("cap 30%", dict(sl_buf=0.06, max_stop_pct=0.30)),
    ]
    for sname, skw in stops:
        print(f"\n-- SL = {sname} --")
        for h in holds:
            p = pf.Params(select_mode="intraday_high", thresh=0.40, cooldown_days=7,
                          max_hold_h=h, **skw)
            df = pf.run_study(p, workers=16)
            print(line(f"hold {h}h", df, p), flush=True)


if __name__ == "__main__":
    diagnostic()
    matrix()
    print("\nDONE")
