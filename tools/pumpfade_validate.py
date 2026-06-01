"""Validate the 'exhaustion + extension' shape edge — is it real or multiple-
comparison luck? Three honest tests:
  1. ROBUSTNESS surface: sweep (ext_peak x vol_ratio) thresholds — a real edge is
     a smooth PLATEAU (neighbors also positive), not a lone knife-edge cell.
  2. BOOTSTRAP CI on OOS EV — is the mean distinguishable from zero?
  3. CONFOUND check: is 'extension' just re-discovering 'big pump'? (corr + within-band)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "pumpfade"
OOS = pd.Timestamp("2025-01-01", tz="UTC")


def main() -> int:
    df = pd.read_parquet(CACHE / "shape_features.parquet")
    df = df[df.peak_wick.notna() & df.vol_ratio.notna()].copy()
    df["et"] = pd.to_datetime(df["et"], utc=True)
    oos = df[df.et >= OOS]
    is_ = df[df.et < OOS]
    print(f"n total {len(df)}  IS {len(is_)}  OOS {len(oos)}\n")

    print("=== 1. ROBUSTNESS SURFACE: OOS EV%% (n) by ext_peak >= row, vol_ratio <= col ===")
    ext_thrs = [0.0, 0.5, 0.75, 1.0, 1.5]
    vol_thrs = [1.0, 0.6, 0.5, 0.4, 0.3]
    hdr = "ext\\vol " + "".join(f"{v:>12}" for v in vol_thrs)
    print(hdr)
    for e in ext_thrs:
        cells = []
        for v in vol_thrs:
            s = oos[(oos.ext_peak >= e) & (oos.vol_ratio <= v)]
            cells.append(f"{100*s.net_ret.mean():+5.1f}({len(s):>3})" if len(s) >= 15 else f"   -({len(s):>3})")
        print(f"{e:>6}  " + "".join(f"{c:>12}" for c in cells))
    print("  (a real edge = a contiguous positive PLATEAU, not one lonely cell)")

    # economically-motivated rule (not the max cell): exhausted + clearly stretched
    rule = (df.ext_peak >= 0.75) & (df.vol_ratio <= 0.5)
    r_is, r_oos = df[rule & (df.et < OOS)], df[rule & (df.et >= OOS)]
    print(f"\n=== 2. RULE = ext>=0.75 & vol_ratio<=0.5  (exhausted + stretched) ===")
    for tag, s in [("IS", r_is), ("OOS", r_oos)]:
        print(f"   {tag}: n={len(s)} EV {100*s.net_ret.mean():+.2f}% win {100*(s.net_ret>0).mean():.0f}% "
              f"med {100*s.net_ret.median():+.1f}% worst {100*s.net_ret.min():.0f}%")
    # bootstrap OOS mean
    x = r_oos.net_ret.values
    rng = np.random.default_rng(7)
    boot = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
    print(f"   OOS EV 95% bootstrap CI: [{lo:+.2f}%, {hi:+.2f}%]   "
          f"P(EV>0)={100*(boot>0).mean():.0f}%  -> {'distinguishable from 0' if lo>0 else 'CI straddles 0 (NOT proven)'}")

    print("\n=== 3. CONFOUND: is extension just 'big pump'? ===")
    print(f"   corr(ext_peak, day_ret) = {df[['ext_peak','day_ret']].corr().iloc[0,1]:.2f}")
    # within a fixed pump-magnitude band, does extension still separate?
    band = df[(df.day_ret >= 0.6) & (df.day_ret <= 1.2)]
    bo = band[band.et >= OOS]
    hi_ext = bo[bo.ext_peak >= 1.0]; lo_ext = bo[bo.ext_peak < 1.0]
    print(f"   within pump 60-120% (OOS): ext>=1.0 EV {100*hi_ext.net_ret.mean():+.2f}% (n={len(hi_ext)})  "
          f"vs ext<1.0 EV {100*lo_ext.net_ret.mean():+.2f}% (n={len(lo_ext)})")
    print(f"   => if ext still separates within a fixed pump band, it's NOT just magnitude")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
