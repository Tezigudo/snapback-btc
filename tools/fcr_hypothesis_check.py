"""Cheap sanity check before writing the FCR strategy.

Hypothesis: when 8h funding is extreme (|rate| > 0.0005 per 8h), price
reverts against the crowded side within 24h.

Method: for each funding settlement where |rate| > threshold, compute the
forward 24h return on 1h closes, sign-flip so a "successful mean reversion"
is positive. Report mean / median / win-rate / hit-by-magnitude.

If mean > ~0.5% (covering ~10bps round-trip friction), FCR has a basis.
If ~0 or negative, kill the idea and pivot to Donchian-v3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "historical"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    funding = pd.read_parquet(DATA / "BTC_USDT_USDT_funding.parquet")
    klines = pd.read_parquet(DATA / "BTC_USDT_USDT_1h.parquet")
    # Normalize timestamp columns -> UTC index
    for df in (funding, klines):
        ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "ts", "time", "datetime")), None)
        if ts_col is not None:
            df["_ts"] = pd.to_datetime(df[ts_col], utc=True)
            df.set_index("_ts", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
    funding = funding.sort_index()
    klines = klines.sort_index()
    return funding, klines


def forward_return(klines: pd.DataFrame, t0: pd.Timestamp, hours: int) -> float | None:
    t1 = t0 + pd.Timedelta(hours=hours)
    # nearest <= t0 and <= t1
    try:
        p0 = klines["close"].asof(t0)
        p1 = klines["close"].asof(t1)
    except KeyError:
        return None
    if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None
    return (p1 - p0) / p0


def main() -> int:
    funding, klines = load()
    print(f"funding rows: {len(funding):,}  range: {funding.index.min()} -> {funding.index.max()}")
    print(f"kline rows:   {len(klines):,}  range: {klines.index.min()} -> {klines.index.max()}")

    # Find the funding-rate column (be defensive about naming)
    fcol = next((c for c in funding.columns if "fund" in c.lower() and "rate" in c.lower()), None)
    if fcol is None:
        fcol = next((c for c in funding.columns if c.lower() in ("rate", "fundingrate", "funding")), None)
    if fcol is None:
        print(f"could not identify funding-rate column from {list(funding.columns)}")
        return 1
    print(f"using funding column: {fcol!r}")
    print(f"funding stats: mean={funding[fcol].mean():.6f}, std={funding[fcol].std():.6f}, "
          f"min={funding[fcol].min():.6f}, max={funding[fcol].max():.6f}")

    # Distribution of |funding|
    abs_f = funding[fcol].abs()
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  |funding| quantile {q:.2f}: {abs_f.quantile(q):.6f}")

    # Test multiple thresholds and forward windows
    print("\n=== Reversion test (sign-flipped forward return, positive = reverted) ===")
    print(f"{'threshold':>10} {'horizon':>8} {'n':>6} {'mean':>10} {'median':>10} "
          f"{'win%':>7} {'>+0.5%':>8} {'<-0.5%':>8}")

    fees = 0.001  # 10 bps round-trip = generous proxy for taker + slippage

    for thresh in (0.0003, 0.0005, 0.0008, 0.001):
        events = funding[abs_f >= thresh]
        if len(events) == 0:
            continue
        for horizon in (8, 24, 48):
            rs: list[float] = []
            for ts, row in events.iterrows():
                r = forward_return(klines, ts, horizon)
                if r is None:
                    continue
                # sign-flip so positive r means price reverted vs crowded side
                signed = -np.sign(row[fcol]) * r
                rs.append(signed - fees)  # net of friction
            if not rs:
                continue
            arr = np.array(rs)
            print(
                f"{thresh:>10.4f} {horizon:>5d}h "
                f"{len(arr):>6d} {arr.mean():>+10.4f} {np.median(arr):>+10.4f} "
                f"{(arr > 0).mean()*100:>6.1f}% "
                f"{(arr > 0.005).mean()*100:>7.1f}% "
                f"{(arr < -0.005).mean()*100:>7.1f}%"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
