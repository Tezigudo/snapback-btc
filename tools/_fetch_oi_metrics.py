"""Fetch Binance Vision futures metrics dumps (5m open interest) for BTCUSDT.

Free official archive: data.binance.vision futures/um/daily/metrics — one ZIP
per day with 5m rows incl. sum_open_interest. Coverage starts ~2021-12.
Builds data/historical/BTC_USDT_USDT_oi_5m.parquet for the OI-velocity
TODO_LEG (todo_leg_oi_velocity_divergence.md). 404 days are skipped.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "historical" / "BTC_USDT_USDT_oi_5m.parquet"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT"
START, END = "2021-12-01", "2026-06-02"


def main() -> int:
    frames = []
    missing = 0
    days = pd.date_range(START, END, freq="D")
    for i, d in enumerate(days):
        url = f"{BASE}/BTCUSDT-metrics-{d.date()}.zip"
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            print(f"  {d.date()} RETRY-SKIP ({exc})", file=sys.stderr)
            missing += 1
            continue
        if r.status_code != 200:
            missing += 1
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, usecols=["create_time", "sum_open_interest",
                                             "sum_open_interest_value"])
        frames.append(df)
        if i % 100 == 0:
            print(f"  {d.date()} ({i}/{len(days)}, missing={missing})",
                  file=sys.stderr)

    out = pd.concat(frames, ignore_index=True)
    out["create_time"] = pd.to_datetime(out["create_time"])
    out = out.drop_duplicates("create_time").set_index("create_time").sort_index()
    out.to_parquet(OUT)
    print(f"[oi] wrote {OUT}: {len(out)} rows "
          f"{out.index.min()} -> {out.index.max()} (missing days={missing})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
