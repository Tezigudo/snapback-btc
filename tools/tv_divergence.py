#!/usr/bin/env python3
"""TradingView's built-in Divergence Indicator, ported faithfully.

WHY THIS EXISTS ALONGSIDE strategy/indicators.py:find_divergence()
The two detect different things, and conflating them produced a misleading
result once already:

    | aspect        | find_divergence()          | this module            |
    |---------------|----------------------------|------------------------|
    | pivots on     | PRICE (swing_high_low)     | THE OSCILLATOR         |
    | lookback      | swing_k = 3                | lbL = lbR = 5          |
    | separation    | min 5 / max 60 bars        | rangeLower/Upper 5/60  |
    | fires at      | b2 + k                     | pivot + lbR            |

Pivoting on the oscillator yields a different signal set, so "divergence was
tested" is ambiguous unless you say which detector. Keep both; name which one
any verdict used.

Provenance: identified 2026-08-06 from the Cutter Trade page, whose charts carry
an indicator pane titled "อินดิเคเตอร์ RSI Divergence" emitting Bull/Bear labels
on 4h, 1D and 1W simultaneously. Port validated against his own weekly chart:
he shows 1 Bull / 5 Bear over 2021-2026, this produces 2 Bull / 6 Bear over
2019-2026 -- same counts and same bear skew on the overlapping window.

MEASURED VERDICT (BTC, 2019-09..2026-08, forward return vs same-horizon base
rate): NO EDGE at any timeframe. Best z = +1.80 (1W bear, n=6) against a 1.96
threshold; every timeframe shows hit-rate marginally above base while AVERAGE
RETURN IS NEGATIVE -- right slightly more often, more expensive when wrong.
Signal rate: 4h ~28/yr, 1D ~6/yr, 1W ~1.2/yr.
Not wired to any strategy. Use for telemetry/context, not entries.

Usage:
    uv run python -m tools.tv_divergence --tf 4h
    uv run python -m tools.tv_divergence --tf all --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # Repo root FIRST: the venv's editable-install .pth points at another
    # worktree, and importing that one silently tests the wrong code.
    sys.path.insert(0, str(ROOT))

LB_L, LB_R = 5, 5
RANGE_LOWER, RANGE_UPPER = 5, 60
DATA = ROOT / "data" / "historical"


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = (up.ewm(alpha=1 / n, adjust=False).mean()
          / dn.ewm(alpha=1 / n, adjust=False).mean())
    return 100 - 100 / (1 + rs)


def _pivots(series: np.ndarray, lbL: int, lbR: int, low: bool) -> np.ndarray:
    """True at bar i when i is a pivot. Needs lbR FUTURE bars to confirm."""
    n = len(series)
    out = np.zeros(n, dtype=bool)
    for i in range(lbL, n - lbR):
        w = series[i - lbL:i + lbR + 1]
        v = series[i]
        if np.isnan(v) or np.isnan(w).any():
            continue
        out[i] = (v == w.min()) if low else (v == w.max())
    return out


def divergences(
    df: pd.DataFrame,
    osc: pd.Series,
    lb_l: int = LB_L,
    lb_r: int = LB_R,
    range_lower: int = RANGE_LOWER,
    range_upper: int = RANGE_UPPER,
) -> tuple[pd.Series, pd.Series]:
    """(bull, bear) booleans indexed like df, fired at pivot + lb_r.

    The +lb_r shift is the confirmation lag and is NOT optional: the pivot is
    unknowable until lb_r bars later, so firing at the pivot bar is lookahead.

    Parameters default to TradingView's built-in values and should normally stay
    there -- the point of this module is fidelity to the detector Cutter Trade
    actually runs. They are exposed for SENSITIVITY ANALYSIS only. Treat any
    improvement found by tuning them as a multiple-comparisons artifact until it
    survives the usual walk-forward/OOS gates: the headline verdict already
    searched 3 variants x 3 horizons and topped out at z=+1.80 on n=6, so a
    parameter sweep will find "edge" that is not there.

    KNOWN EDGE CASE (measured, deliberately not "fixed"): _pivots() marks a bar
    when it ties the window min/max, so a perfectly flat oscillator run would
    mark several adjacent bars. Measured on BTC 4h/1D/1W: ZERO adjacent pivots,
    and only 2/1/3 exact consecutive RSI ties across thousands of bars. Adding
    tie-breaking would change a port that is validated against his real chart
    output, for no measurable gain.
    """
    o = osc.to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    n = len(df)

    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    for is_low, arr, out, cmp_osc in (
        (True, lo, bull, lambda a, b: a > b),      # bullish: osc higher low
        (False, hi, bear, lambda a, b: a < b),     # bearish: osc lower high
    ):
        piv = _pivots(o, lb_l, lb_r, low=is_low)
        prev = None
        for i in range(n):
            if not piv[i]:
                continue
            if prev is not None:
                gap = i - prev
                price_div = arr[i] < arr[prev] if is_low else arr[i] > arr[prev]
                if range_lower <= gap <= range_upper and cmp_osc(o[i], o[prev]) and price_div:
                    fire = i + lb_r
                    if fire < n:
                        out[fire] = True
            prev = i

    return pd.Series(bull, index=df.index), pd.Series(bear, index=df.index)


def load(tf: str, symbol: str = "BTC") -> pd.DataFrame:
    """4h/1h from parquet; 1D/1W resampled from 4h."""
    if tf in ("1h", "4h", "15m"):
        d = pd.read_parquet(DATA / f"{symbol}_USDT_USDT_{tf}.parquet")
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        return d
    base = load("4h", symbol)
    return base.resample({"1D": "1D", "1W": "1W"}[tf]).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()


def evaluate(tf: str, symbol: str = "BTC", horizons=None) -> dict:
    df = load(tf, symbol)
    bull, bear = divergences(df, rsi(df["close"], 14))
    horizons = horizons or {"4h": [("1d", 6), ("3d", 18), ("7d", 42)],
                            "1D": [("3d", 3), ("7d", 7), ("30d", 30)],
                            "1W": [("4w", 4), ("12w", 12)]}[tf]
    close = df["close"]
    years = (df.index[-1] - df.index[0]).days / 365.25
    res = {"tf": tf, "symbol": symbol, "bars": len(df), "years": round(years, 1),
           "bull": int(bull.sum()), "bear": int(bear.sum()), "scores": {}}
    for name, sig, sign in (("bull", bull, 1), ("bear", bear, -1)):
        res["scores"][name] = {}
        for label, h in horizons:
            fwd = (close.shift(-h) / close - 1) * 100
            base_up = float((fwd > 0).mean())
            v = (fwd[sig].dropna()) * sign
            if not len(v):
                continue
            exp = base_up if sign > 0 else 1 - base_up
            n = len(v)
            hits = int((v > 0).sum())
            se = float(np.sqrt(max(n * exp * (1 - exp), 1e-9)))
            res["scores"][name][label] = dict(
                n=n, hit_pct=round(hits / n * 100, 1), base_pct=round(exp * 100, 1),
                avg_pct=round(float(v.mean()), 3), z=round((hits - n * exp) / se, 2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tf", default="4h", choices=["4h", "1D", "1W", "all"])
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tfs = ["4h", "1D", "1W"] if a.tf == "all" else [a.tf]
    out = []
    for tf in tfs:
        r = evaluate(tf, a.symbol)
        out.append(r)
        print(f"\n### {r['symbol']} {tf}  ({r['bars']} bars, {r['years']}y)")
        print(f"  bull {r['bull']} ({r['bull']/r['years']:.1f}/yr)   "
              f"bear {r['bear']} ({r['bear']/r['years']:.1f}/yr)")
        for side in ("bull", "bear"):
            parts = [f"{k}: n={v['n']} hit={v['hit_pct']}%(base {v['base_pct']}%) "
                     f"avg={v['avg_pct']:+}% z={v['z']:+}"
                     for k, v in r["scores"][side].items()]
            print(f"   {side.upper():5s} " + " | ".join(parts))

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwritten: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
