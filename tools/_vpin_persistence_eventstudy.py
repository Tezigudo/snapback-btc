"""VPIN-persistence cheap-kill event study (pre-harness gate).

Per todo_leg_vpin_persistence.md, before building the full backtest harness we
check whether the pinned signal construction has ANY directional edge at
tradeable magnitude:

  - volume buckets: equal-volume V*, targeting ~50 buckets/day; V* refit
    monthly from the TRAILING 30-day mean daily volume (no lookahead)
  - VPIN = rolling mean over N=50 buckets of |V_buy - V_sell| / V*
  - event: VPIN > rolling p90 (trailing 1500 buckets ~ 30 days) for >= K
    consecutive buckets (K=3 pinned; K=1 refutation arm)
  - direction: sign of net signed imbalance over the K trigger buckets
  - HTF gate: sign agreement with 4H EMA200 slope at last CLOSED 4h bar
  - entry: close of the 15m bar in which the K-th bucket completed
  - outcome: forward return at +4h and +8h (event-study proxy for the
    dynamic exit), gross and net of 15bps round-trip

Shelf rule (memo gates 1+6): if the K=3 arm has ~no gross edge, or the K=1 arm
performs equivalently (persistence not load-bearing), SHELF without harness.
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

START, END = "2021-11-01", "2026-06-02"   # 2 months warmup before 2022-01
STUDY_START = pd.Timestamp("2022-01-01")
N_VPIN = 50                 # buckets in VPIN rolling mean
PCT_WIN = 1500              # trailing buckets for percentile (~30d at 50/day)
BUCKETS_PER_DAY = 50
COST_BPS = 15.0


def build_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Partition the tape into equal-volume buckets; V* refit monthly."""
    vol = df["volume"].values
    buy = df["taker_buy_base"].values
    sell = vol - buy
    ts = df.index

    # monthly V* from trailing 30d mean daily volume (shifted: no lookahead)
    daily_vol = df["volume"].resample("1D").sum()
    vstar_by_month = (daily_vol.rolling(30).mean() / BUCKETS_PER_DAY) \
        .resample("MS").first().shift(1)

    rows = []
    cur_v = cur_buy = cur_sell = 0.0
    cur_month = None
    vstar = None
    for i in range(len(df)):
        t = ts[i]
        m = pd.Timestamp(t.year, t.month, 1)
        if m != cur_month:
            cur_month = m
            v = vstar_by_month.get(m, np.nan)
            if np.isfinite(v) and v > 0:
                vstar = float(v)
        if vstar is None:
            continue
        v_rem, b_rem, s_rem = vol[i], buy[i], sell[i]
        while cur_v + v_rem >= vstar:
            take = vstar - cur_v
            frac = take / v_rem if v_rem > 0 else 0.0
            cur_buy += b_rem * frac
            cur_sell += s_rem * frac
            b_rem -= b_rem * frac
            s_rem -= s_rem * frac
            v_rem -= take
            rows.append((t, (cur_buy - cur_sell) / vstar))
            cur_v = cur_buy = cur_sell = 0.0
        cur_v += v_rem
        cur_buy += b_rem
        cur_sell += s_rem
    out = pd.DataFrame(rows, columns=["ts", "signed_imb"])
    out["abs_imb"] = out["signed_imb"].abs()
    return out


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def main() -> int:
    df = pd.read_parquet(PARQ_15M)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.loc[START:END]

    print("[vpin] building buckets ...", file=sys.stderr)
    bk = build_buckets(df)
    bk["vpin"] = bk["abs_imb"].rolling(N_VPIN).mean()
    bk["p90"] = bk["vpin"].shift(1).rolling(PCT_WIN).quantile(0.90)
    bk["p50"] = bk["vpin"].shift(1).rolling(PCT_WIN).quantile(0.50)
    bk = bk.dropna().reset_index(drop=True)
    bk = bk[bk["ts"] >= STUDY_START].reset_index(drop=True)
    print(f"[vpin] buckets={len(bk)} ({bk['ts'].min()} -> {bk['ts'].max()})",
          file=sys.stderr)

    # 4H EMA200 slope at last CLOSED 4h bar, aligned lookahead-safe
    df4 = pd.read_parquet(PARQ_4H)
    if df4.index.tz is not None:
        df4.index = df4.index.tz_localize(None)
    e4 = ema(df4["close"], 200)
    slope = np.sign(e4.diff())
    slope.index = df4.index + pd.Timedelta(hours=4)      # close timestamps
    slope = slope.sort_index()

    close = df["close"]

    def fwd_ret(t0: pd.Timestamp, hours: int) -> float:
        t1 = t0 + pd.Timedelta(hours=hours)
        s = close.loc[t0:t1]
        if len(s) < 2:
            return np.nan
        return s.iloc[-1] / s.iloc[0] - 1.0

    above = (bk["vpin"] > bk["p90"]).values
    results = {}
    for K, label in [(3, "K3_persistence"), (1, "K1_refutation")]:
        events = []
        run = 0
        in_event = False
        for i in range(len(bk)):
            if above[i]:
                run += 1
            else:
                run = 0
                in_event = False
            if run >= K and not in_event:
                in_event = True
                t0 = bk["ts"].iloc[i]
                direction = float(np.sign(bk["signed_imb"].iloc[max(0, i - K + 1):i + 1].sum()))
                if direction == 0:
                    continue
                # HTF gate
                sl = slope.loc[:t0]
                gate_ok = len(sl) > 0 and np.isfinite(sl.iloc[-1]) and sl.iloc[-1] == direction
                r4 = fwd_ret(t0, 4)
                r8 = fwd_ret(t0, 8)
                events.append((t0, direction, gate_ok, r4, r8))
        ev = pd.DataFrame(events, columns=["ts", "dir", "gate", "r4", "r8"]).dropna()
        arm = {"n_events_total": int(len(ev))}
        for gated, tag in [(False, "ungated"), (True, "htf_gated")]:
            sub = ev if not gated else ev[ev["gate"]]
            for h in ("r4", "r8"):
                r = sub["dir"] * sub[h]
                n = len(r)
                if n < 20:
                    arm[f"{tag}_{h}"] = {"n": n, "note": "insufficient"}
                    continue
                mean_bps = r.mean() * 1e4
                tstat = mean_bps / (r.std(ddof=1) * 1e4 / np.sqrt(n))
                arm[f"{tag}_{h}"] = {
                    "n": int(n),
                    "gross_mean_bps": round(float(mean_bps), 2),
                    "t_stat": round(float(tstat), 2),
                    "win_rate_pct": round(float((r > 0).mean() * 100), 1),
                    "net_mean_bps_at_15bps": round(float(mean_bps - COST_BPS), 2),
                    "events_per_year": round(n / ((ev["ts"].max() - ev["ts"].min()).days / 365.25), 1),
                }
        results[label] = arm

    out = {"construction": "V* monthly trailing-30d/50, VPIN N=50, p90 trailing 1500 buckets",
           "study_window": f"{STUDY_START.date()} .. {END}",
           "arms": results}
    path = ROOT / "reports" / "vpin_persistence_eventstudy.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[vpin] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
