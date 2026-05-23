"""Verify the switcher's per-day decisions OOS, and per-5-OOS-window breakdown.

Two questions:
  (c) On days switcher picked v1, did v1 actually beat Donchian? If yes,
      signal is sound. If no, signal is broken.
  (a) Per 5 OOS windows (2022H1..2025H1), how does switcher compare to 50/50?
      Does it save 2024H1 (the chop year that broke v1)?
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def latest_ts() -> str:
    cands = sorted(REPORTS.glob("regime_switcher_*.json"))
    return cands[-1].stem.replace("regime_switcher_", "")


def load_aligned(ts: str) -> pd.DataFrame:
    v1 = pd.read_csv(REPORTS / f"full_history_{ts}_v1_equity.csv", index_col=0, parse_dates=True)
    d3 = pd.read_csv(REPORTS / f"full_history_{ts}_d3cons_equity.csv", index_col=0, parse_dates=True)
    sw = pd.read_csv(REPORTS / f"regime_switcher_{ts}_eq.csv", index_col=0, parse_dates=True)
    for df in (v1, d3, sw):
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
    v1_r = v1["equity_norm"].resample("1D").last().ffill().pct_change()
    d3_r = d3["equity_norm"].resample("1D").last().ffill().pct_change()
    out = pd.concat({"v1": v1_r, "d3": d3_r, "choice_v1": sw["choice_v1"]}, axis=1).dropna()
    return out


def sharpe(r: pd.Series) -> float:
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(365))


def max_dd_pct(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    return float((eq / peak - 1).min() * 100)


def main() -> int:
    ts = latest_ts()
    print(f"Verifying switcher from run {ts}")
    df = load_aligned(ts)
    print(f"Aligned {len(df):,} days ({df.index.min().date()} → {df.index.max().date()})")

    # ====== (c) Per-decision verification ======
    test = df.loc["2023-01-01":]
    v1_days = test[test["choice_v1"] == 1]
    d3_days = test[test["choice_v1"] == 0]
    print(f"\nTEST period: {len(test):,} days. Switcher picked v1 on {len(v1_days):,} ({len(v1_days)/len(test)*100:.1f}%), d3 on {len(d3_days):,}.")

    # On v1-picked days: did v1 actually beat d3?
    print("\n=== (c) Did the switcher pick correctly? ===")
    if len(v1_days) > 0:
        v1_pick_v1_ret = v1_days["v1"].mean() * 100
        v1_pick_d3_ret = v1_days["d3"].mean() * 100
        v1_pick_winrate = (v1_days["v1"] > v1_days["d3"]).mean() * 100
        v1_pick_tstat = (v1_days["v1"] - v1_days["d3"]).mean() / (v1_days["v1"] - v1_days["d3"]).std() * np.sqrt(len(v1_days))
        print(f"  On {len(v1_days)} days switcher picked v1:")
        print(f"    v1 mean daily ret = {v1_pick_v1_ret:+.4f}%")
        print(f"    d3 mean daily ret = {v1_pick_d3_ret:+.4f}%")
        print(f"    v1 > d3 on {v1_pick_winrate:.1f}% of these days")
        print(f"    t-stat (v1−d3) = {v1_pick_tstat:+.2f}")
        verdict_c1 = "SIGNAL WORKS" if v1_pick_v1_ret > v1_pick_d3_ret else "SIGNAL BROKEN on v1 pick"
        print(f"    => {verdict_c1}")
    if len(d3_days) > 0:
        d3_pick_v1_ret = d3_days["v1"].mean() * 100
        d3_pick_d3_ret = d3_days["d3"].mean() * 100
        d3_pick_winrate = (d3_days["d3"] > d3_days["v1"]).mean() * 100
        d3_pick_tstat = (d3_days["d3"] - d3_days["v1"]).mean() / (d3_days["d3"] - d3_days["v1"]).std() * np.sqrt(len(d3_days))
        print(f"  On {len(d3_days)} days switcher picked d3:")
        print(f"    v1 mean daily ret = {d3_pick_v1_ret:+.4f}%")
        print(f"    d3 mean daily ret = {d3_pick_d3_ret:+.4f}%")
        print(f"    d3 > v1 on {d3_pick_winrate:.1f}% of these days")
        print(f"    t-stat (d3−v1) = {d3_pick_tstat:+.2f}")
        verdict_c2 = "SIGNAL WORKS" if d3_pick_d3_ret > d3_pick_v1_ret else "SIGNAL BROKEN on d3 pick"
        print(f"    => {verdict_c2}")

    # ====== (a) Per-5-OOS-window breakdown ======
    print("\n=== (a) Per-OOS-window breakdown ===")
    windows = [
        ("2022H1", "2022-01-01", "2022-06-30"),
        ("2023H1", "2023-01-01", "2023-06-30"),
        ("2024H1", "2024-01-01", "2024-06-30"),
        ("2024H2", "2024-07-01", "2024-12-31"),
        ("2025H1", "2025-01-01", "2025-05-31"),
    ]
    print(f"{'window':>8} {'days':>5} "
          f"{'v1 %':>9} {'d3 %':>9} "
          f"{'50/50 %':>9} {'50/50 Sh':>9} {'50/50 DD':>9} "
          f"{'sw %':>9} {'sw Sh':>9} {'sw DD':>9} "
          f"{'pct_v1':>7}")
    rows_out = []
    for label, start, end in windows:
        w = df.loc[start:end]
        if w.empty:
            continue
        sw_ret = np.where(w["choice_v1"] == 1, w["v1"], w["d3"])
        sw_ret = pd.Series(sw_ret, index=w.index)
        c5050 = 0.5 * w["v1"] + 0.5 * w["d3"]
        v1_ret = (1 + w["v1"]).prod() - 1
        d3_ret = (1 + w["d3"]).prod() - 1
        c_ret = (1 + c5050).prod() - 1
        s_ret = (1 + sw_ret).prod() - 1
        print(
            f"{label:>8} {len(w):>5d} "
            f"{v1_ret*100:>+8.2f}% {d3_ret*100:>+8.2f}% "
            f"{c_ret*100:>+8.2f}% {sharpe(c5050):>+9.2f} {max_dd_pct(c5050):>+9.2f}% "
            f"{s_ret*100:>+8.2f}% {sharpe(sw_ret):>+9.2f} {max_dd_pct(sw_ret):>+9.2f}% "
            f"{(w['choice_v1']==1).mean()*100:>6.1f}%"
        )
        rows_out.append({
            "window": label,
            "v1_ret_pct": v1_ret * 100,
            "d3_ret_pct": d3_ret * 100,
            "combo_50_50": {"ret_pct": c_ret * 100, "sharpe": sharpe(c5050), "max_dd_pct": max_dd_pct(c5050)},
            "switcher": {"ret_pct": s_ret * 100, "sharpe": sharpe(sw_ret), "max_dd_pct": max_dd_pct(sw_ret),
                         "pct_days_v1": float((w["choice_v1"]==1).mean()*100)},
        })

    out_path = REPORTS / f"regime_switcher_verify_{ts}.json"
    out_path.write_text(json.dumps(rows_out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
