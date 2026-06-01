"""Generate a capitulation-LONG trade stream from cached parquets.

Reuses the VALIDATED signal + sim from capitulation_watch.py / the report
backtest (SL=2xATR, TP=3xATR, time-stop 24h, cooldown 48h, FEE_RT 0.22%).
Multi-coin (the watchlist). Emits one merged, time-sorted trade CSV for the
DCA sim. Sizing uses 2xATR stop -> sl_frac = 2*atr/entry; alt-like exchange
minimums ($5 notional) since most capitulation names are alts.

Out: reports/capitulation_long_trades.csv
Run: uv run python tools/gen_capitulation_trades.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.capitulation_watch as cw  # noqa: E402
from tools.build_capitulation_report import _sim, FEE_RT  # noqa: E402

DATA = ROOT / "data" / "historical"


def main() -> int:
    rows = []
    used, missing = [], []
    for coin in cw.WATCHLIST:
        f = DATA / f"{coin}_USDT_USDT_1h.parquet"
        if not f.exists():
            missing.append(coin)
            continue
        df = pd.read_parquet(f)
        df.columns = [c.lower() for c in df.columns]
        df = cw._enrich(df.iloc[:-1])
        idx = df.index
        mask = cw._signal_mask(df).values
        cd = -1
        for i in np.where(mask)[0]:
            if i < cd:
                continue
            r = _sim(df, int(i))
            if r is None:
                continue
            ex, gross = r
            entry = float(df["close"].iloc[i])
            a = float(df["atr"].iloc[i])
            rows.append({
                "EntryTime": idx[i], "ExitTime": idx[ex],
                "EntryPrice": entry,
                "ret": (gross - FEE_RT) / 100.0,            # net fraction
                "atr_pct": (a / entry) if entry > 0 else np.nan,
                "coin": coin,
            })
            cd = ex + cw.COOLDOWN_BARS
        used.append(coin)
    out = pd.DataFrame(rows).sort_values("EntryTime").reset_index(drop=True)
    path = ROOT / "reports" / "capitulation_long_trades.csv"
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} capitulation trades -> {path}")
    print(f"  coins used: {len(used)}  missing parquet: {missing}")
    if len(out):
        print(f"  span {str(out['EntryTime'].iloc[0])[:10]} .. {str(out['EntryTime'].iloc[-1])[:10]}")
        print(f"  mean net/trade {out['ret'].mean()*100:+.2f}%  WR {(out['ret']>0).mean()*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
