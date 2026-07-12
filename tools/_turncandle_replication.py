"""Turn-of-15m-candle replication — LOAD-BEARING first test (gate 1).

Pinned construction per todo_leg_turn_of_candle_15m.md (NO tuning):
  - trigger: start of each 15m candle, bucketed by minute-of-hour {0,15,30,45}
  - direction: sign of the PRECEDING 15m candle's close-to-open (momentum
    continuation, paper Table 6)
  - hold: exactly one 15m candle (entry at bar open, exit at bar close)
  - trade return: direction * (close/open - 1)

Windows:
  paper_era   2020-01-01 .. 2022-08-31  (construction sanity: paper sample ends
                                          2022-08 with Sharpe 4.96 claimed)
  post_pub    2023-01-01 .. 2026-06-02  (the load-bearing OOS: crowding decay?)

Reports per turn-minute: n, mean bps/trade (gross), t-stat, win rate,
annualized Sharpe, and net mean bps at 5/10/15/20 bps round-trip cost.
Plus per-year decay table for the minute-0 headline. Shelf rule: if post-pub
gross edge is ~0 or negative, do NOT parameter-rescue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQ = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

WINDOWS = {
    "paper_era": ("2020-01-01", "2022-08-31"),
    "post_pub": ("2023-01-01", "2026-06-02"),
}
COSTS_BPS = [0, 5, 10, 15, 20]


def stats_block(r: pd.Series, trades_per_year: float) -> dict:
    n = len(r)
    mean_bps = r.mean() * 1e4
    std_bps = r.std(ddof=1) * 1e4
    t = mean_bps / (std_bps / np.sqrt(n)) if n > 1 and std_bps > 0 else 0.0
    sharpe_ann = (r.mean() / r.std(ddof=1)) * np.sqrt(trades_per_year) if r.std(ddof=1) > 0 else 0.0
    out = {
        "n": int(n),
        "gross_mean_bps": round(float(mean_bps), 3),
        "t_stat": round(float(t), 2),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2),
        "sharpe_ann_gross": round(float(sharpe_ann), 3),
    }
    for c in COSTS_BPS[1:]:
        net = mean_bps - c
        out[f"net_mean_bps_at_{c}bps"] = round(float(net), 3)
    return out


def main() -> int:
    df = pd.read_parquet(PARQ)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    bar_ret = df["close"] / df["open"] - 1.0
    prev_sign = np.sign((df["close"] - df["open"]).shift(1))
    trade_ret = prev_sign * bar_ret          # momentum continuation, pinned
    minute = df.index.minute

    out: dict = {"construction": "dir=sign(prev 15m c-o), hold=1 bar", "windows": {}}
    for wname, (s, e) in WINDOWS.items():
        mask = (df.index >= s) & (df.index <= e)
        w: dict = {}
        for m in (0, 15, 30, 45):
            r = trade_ret[mask & (minute == m)].dropna()
            w[f"min_{m}"] = stats_block(r, trades_per_year=24 * 365)
        # all four turns pooled (96/day variant)
        r_all = trade_ret[mask].dropna()
        w["all_turns"] = stats_block(r_all, trades_per_year=96 * 365)
        out["windows"][wname] = w

    # decay curve: minute-0 headline per calendar year
    decay = {}
    for yr in range(2020, 2027):
        r = trade_ret[(df.index.year == yr) & (minute == 0)].dropna()
        if len(r) > 100:
            decay[str(yr)] = {
                "n": int(len(r)),
                "gross_mean_bps": round(float(r.mean() * 1e4), 3),
                "t_stat": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2),
            }
    out["min0_decay_by_year"] = decay

    path = ROOT / "reports" / "turncandle_replication.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[turncandle] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
