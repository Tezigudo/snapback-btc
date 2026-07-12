"""OI-velocity/price-divergence cheap-kill event study (pre-harness gate).

Pinned construction per todo_leg_oi_velocity_divergence.md (no tuning):
  - 5m sum_open_interest (Binance Vision metrics) -> 15m last-value series
  - OI_z = dOI / rolling-30d std(dOI); Px_z = dClose / rolling-30d std(dClose)
  - fade_long : OI_z > +1.5 AND Px_z < -0.5 AND funding > 0          -> LONG
  - fade_short: OI_z > +1.5 AND Px_z > +0.5 AND funding > 0
                AND RSI(14,4h) > 70                                   -> SHORT
  - follow    : OI_z > +1.0 AND |Px_z| >= 0.5 AND |funding| < 0.00005
                -> direction = sign(Px_z)
  - HTF gate  : direction == sign(4H EMA200 slope) at last CLOSED 4h bar
  - outcome   : forward return at +4h / +8h / +12h (12h = memo max hold),
                gross and net of 15bps; 12h blackout between events per mode

Each mode is evaluated SEPARATELY (memo risk profile). Data-quality gate 6:
report OI missing-bar rate; abort if > 5%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQ_15M = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
PARQ_4H = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
PARQ_OI = ROOT / "data" / "historical" / "BTC_USDT_USDT_oi_5m.parquet"
PARQ_FUND = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"

STUDY_START, STUDY_END = "2022-01-01", "2026-06-02"
Z_WIN = 96 * 30            # 30 days of 15m bars
COST_BPS = 15.0
BLACKOUT = pd.Timedelta(hours=12)
HORIZONS = (4, 8, 12)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn
    return 100 - 100 / (1 + rs)


def align_backward(target_idx: pd.DatetimeIndex, s: pd.Series) -> np.ndarray:
    # force matching datetime64 precision (pandas refuses cross-precision merges)
    right_idx = pd.DatetimeIndex(s.index.astype("datetime64[us]"))
    left_idx = pd.DatetimeIndex(target_idx.astype("datetime64[us]"))
    right = pd.DataFrame({"v": s.values}, index=right_idx).sort_index()
    left = pd.DataFrame(index=left_idx)
    return pd.merge_asof(left, right, left_index=True, right_index=True,
                         direction="backward")["v"].values


def main() -> int:
    px = pd.read_parquet(PARQ_15M)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)

    oi = pd.read_parquet(PARQ_OI)
    if oi.index.tz is not None:
        oi.index = oi.index.tz_localize(None)
    # gate 6: missing-bar rate on the 5m grid inside the study window
    grid = pd.date_range(STUDY_START, STUDY_END, freq="5min")
    missing_rate = 1.0 - oi.index.intersection(grid).size / grid.size
    print(f"[oi] 5m missing-bar rate {missing_rate:.3%}", file=sys.stderr)
    if missing_rate > 0.05:
        print("[oi] DATA GATE FAIL (>5%) — abort per memo gate 6", file=sys.stderr)
        return 1

    oi15 = oi["sum_open_interest"].resample("15min").last()
    df = px.loc["2021-11-01":STUDY_END].copy()
    df["oi"] = oi15.reindex(df.index)
    df["oi"] = df["oi"].ffill(limit=8)          # bridge short gaps only

    d_oi = df["oi"].diff()
    d_px = df["close"].diff()
    df["oi_z"] = d_oi / d_oi.rolling(Z_WIN).std()
    df["px_z"] = d_px / d_px.rolling(Z_WIN).std()

    fund = pd.read_parquet(PARQ_FUND)
    if fund.index.tz is not None:
        fund.index = fund.index.tz_localize(None)
    df["funding"] = align_backward(df.index, fund["funding_rate"])

    df4 = pd.read_parquet(PARQ_4H)
    if df4.index.tz is not None:
        df4.index = df4.index.tz_localize(None)
    e4 = ema(df4["close"], 200)
    slope4 = pd.Series(np.sign(e4.diff()).values, index=df4.index + pd.Timedelta(hours=4))
    rsi4 = pd.Series(rsi(df4["close"]).values, index=df4.index + pd.Timedelta(hours=4))
    df["slope4h"] = align_backward(df.index, slope4)
    df["rsi4h"] = align_backward(df.index, rsi4)

    df = df.loc[STUDY_START:].dropna(subset=["oi_z", "px_z", "funding"])
    close = px["close"]

    def fwd(t0, hours):
        s = close.loc[t0:t0 + pd.Timedelta(hours=hours)]
        return s.iloc[-1] / s.iloc[0] - 1.0 if len(s) > 1 else np.nan

    modes = {
        "fade_long": (df["oi_z"] > 1.5) & (df["px_z"] < -0.5) & (df["funding"] > 0),
        "fade_short": (df["oi_z"] > 1.5) & (df["px_z"] > 0.5) & (df["funding"] > 0)
                      & (df["rsi4h"] > 70),
        "follow": (df["oi_z"] > 1.0) & (df["px_z"].abs() >= 0.5)
                  & (df["funding"].abs() < 0.00005),
    }
    dir_of = {
        "fade_long": lambda row: 1.0,
        "fade_short": lambda row: -1.0,
        "follow": lambda row: float(np.sign(row["px_z"])),
    }

    out = {"study_window": f"{STUDY_START} .. {STUDY_END}",
           "oi_missing_rate_pct": round(missing_rate * 100, 3), "modes": {}}
    for mode, mask in modes.items():
        trig = df[mask]
        events = []
        last_t = pd.Timestamp.min
        for t0, row in trig.iterrows():
            if t0 < last_t + BLACKOUT:
                continue
            last_t = t0
            d = dir_of[mode](row)
            gate_ok = np.isfinite(row["slope4h"]) and row["slope4h"] == d
            events.append((t0, d, gate_ok, *(fwd(t0, h) for h in HORIZONS)))
        cols = ["ts", "dir", "gate"] + [f"r{h}" for h in HORIZONS]
        ev = pd.DataFrame(events, columns=cols).dropna()
        block = {"n_events": int(len(ev))}
        years = max((df.index.max() - df.index.min()).days / 365.25, 1e-9)
        for gated, tag in [(False, "ungated"), (True, "htf_gated")]:
            sub = ev if not gated else ev[ev["gate"]]
            for h in HORIZONS:
                r = sub["dir"] * sub[f"r{h}"]
                n = len(r)
                if n < 15:
                    block[f"{tag}_r{h}"] = {"n": n, "note": "insufficient"}
                    continue
                mean_bps = r.mean() * 1e4
                tstat = mean_bps / (r.std(ddof=1) * 1e4 / np.sqrt(n))
                block[f"{tag}_r{h}"] = {
                    "n": int(n),
                    "gross_mean_bps": round(float(mean_bps), 2),
                    "t_stat": round(float(tstat), 2),
                    "win_rate_pct": round(float((r > 0).mean() * 100), 1),
                    "net_mean_bps_at_15bps": round(float(mean_bps - COST_BPS), 2),
                    "events_per_year": round(n / years, 1),
                }
        out["modes"][mode] = block

    path = ROOT / "reports" / "oi_velocity_eventstudy.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[oi] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
