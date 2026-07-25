"""
Round 4 — is the roller-coaster fixable by spreading it across coins?

God's read on round 3: a single alt leg is still a roller coaster (bear-biased,
-5% in SOL's bull year, +195% in one bear year). The question asked was whether
there is more on SOL, on other coins, or another BTC leg.

A single-asset trend leg on a high-beta alt cannot be made smooth — its return
IS the lumpy regime capture. The lever that actually reduces the ride is running
the SAME geometry across many uncorrelated coins and equal-weighting them: the
per-coin lumps land in different months, so the basket's drawdown collapses even
though each leg is unchanged. A smaller drawdown then buys size back inside the
-35.5% kill-switch budget, which is where the return comes back.

This script measures that, plus the thing God said yes to: how correlated these
candidates are with the BTC legs already deployed.

Method
------
* 13 coins (SOL + 12 liquid perps all listed before the 2022-04 span start).
* Both round-3 geometries, per coin, native 4h, same span, same risk.
* **No coin selection.** The headline basket is equal-weight across ALL 13, so
  the asset dimension is out-of-sample too. A "best-6" variant is reported
  separately and flagged as in-sample.
* Basket = equal-weight, per-bar rebalanced average of the legs' bar returns
  (each leg is its own sub-account holding 1/N of capital).
* Basket and single-leg are both sized by bisection to maxDD ≈ -30%, so the
  return comparison is not a risk-appetite artifact.
* Smoothness reported as max drawdown, longest time underwater, and share of
  positive months — the metrics that actually describe the ride.

Run: .venv/bin/python tools/sol_leg_basket.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TF = "4h"
OOS_START = datetime(2022, 4, 1, tzinfo=UTC)
OOS_END = datetime(2026, 7, 25, tzinfo=UTC)
YEARS = 4.32
DD_TARGET = -30.0
LEVERAGE = 3

COINS = ["SOL", "ETH", "BNB", "XRP", "ADA", "DOGE", "AVAX",
         "LINK", "LTC", "DOT", "ATOM", "BCH", "NEAR"]

GEOMETRIES: dict[str, tuple[str, dict]] = {
    "supertrend": ("supertrend", {
        "st_period": 14, "st_multiplier": 3.5, "st_sl_atr": 2.0,
        "st_tp_atr": 10.0, "allow_shorts": True}),
    "st-dual": ("st-dual", {
        "st_period": 7, "st_multiplier": 2.0, "st_slow_period": 30,
        "st_sl_atr": 1.0, "st_tp_atr": 6.0, "allow_shorts": True}),
}

_RISK_ATTRS = ("st_risk_per_trade_pct", "rider_risk_per_trade_pct",
               "sol_risk_per_trade_pct")


def _run(key: str, attrs: dict, symbol: str, risk: float) -> dict | None:
    cls = STRATEGIES[key]
    for k, v in attrs.items():
        setattr(cls, k, v)
    for ra in _RISK_ATTRS:
        if hasattr(cls, ra):
            setattr(cls, ra, risk)
    try:
        return run_backtest(
            key, symbol, TF, OOS_START, OOS_END, leverage=LEVERAGE, quiet=True,
            params_override=dataclasses.replace(
                StrategyParams.from_yaml(), risk_per_trade_pct=risk, leverage=LEVERAGE),
            return_equity=True, return_trades=True)
    except Exception as exc:
        print(f"    [warn] {symbol} {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def bar_returns(r: dict) -> pd.Series:
    eq = r["returns_series"].astype(float)
    return eq.pct_change().fillna(0.0)


def curve_stats(ret: pd.Series, label: str = "") -> dict:
    """Compound a per-bar return series and describe the ride."""
    eq = (1.0 + ret).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    under = eq < peak * 0.999
    gaps, cur = [], None
    for ts, u in under.items():
        if u and cur is None:
            cur = ts
        elif not u and cur is not None:
            gaps.append((ts - cur).days)
            cur = None
    if cur is not None:
        gaps.append((under.index[-1] - cur).days)
    monthly = eq.resample("ME").last().pct_change().dropna()
    total = (eq.iloc[-1] - 1.0) * 100.0
    return {
        "label": label,
        "ret_pct": total,
        "cagr_pct": ((eq.iloc[-1]) ** (1 / YEARS) - 1) * 100.0,
        "max_dd_pct": dd.min(),
        "max_days_underwater": max(gaps) if gaps else 0,
        "pos_months_pct": 100.0 * (monthly > 0).mean() if len(monthly) else float("nan"),
        "n_months": len(monthly),
        "monthly_vol_pct": monthly.std() * 100.0 if len(monthly) else float("nan"),
    }


def size_curve_to_dd(build, dd_target=DD_TARGET, lo=0.25, hi=20.0, iters=11):
    """Bisect risk% until the built curve's max-DD hits dd_target.

    `build(risk)` returns a per-bar return series. DD is monotone in risk for
    fractional sizing, so bisection is valid.
    """
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        ret = build(mid)
        st = curve_stats(ret)
        if st["max_dd_pct"] < dd_target:
            hi = mid
        else:
            lo = mid
            best = (mid, ret, st)
    if best is None:
        ret = build(lo)
        best = (lo, ret, curve_stats(ret))
    return best


def main() -> int:
    out: dict = {}
    base_risk = 2.0

    for geo_name, (key, attrs) in GEOMETRIES.items():
        print("=" * 112)
        print(f"GEOMETRY `{geo_name}`  —  per-coin at risk {base_risk}%, "
              f"4h, {OOS_START.date()} → {OOS_END.date()}")
        print("=" * 112)
        legs: dict[str, pd.Series] = {}
        print(f"{'coin':<6} {'ret%':>9} {'CAGR%':>7} {'maxDD%':>8} {'WR%':>6} "
              f"{'PF':>5} {'n':>4} {'posMo%':>7} {'daysUW':>7}")
        per_coin = {}
        for c in COINS:
            r = _run(key, attrs, f"{c}/USDT:USDT", base_risk)
            if r is None:
                continue
            br = bar_returns(r)
            legs[c] = br
            st = curve_stats(br, c)
            per_coin[c] = {**st, "wr": r["win_rate_pct"], "pf": r["profit_factor"],
                           "trades": r["trades"]}
            print(f"{c:<6} {st['ret_pct']:>9.1f} {st['cagr_pct']:>7.1f} "
                  f"{st['max_dd_pct']:>8.1f} {r['win_rate_pct']:>6.1f} "
                  f"{r['profit_factor']:>5.2f} {r['trades']:>4} "
                  f"{st['pos_months_pct']:>7.1f} {st['max_days_underwater']:>7}")
        out[f"{geo_name}_per_coin"] = per_coin

        # --- correlation between coin legs ---
        idx = sorted(set().union(*[set(s.index) for s in legs.values()]))
        frame = pd.DataFrame({c: s.reindex(idx).fillna(0.0) for c, s in legs.items()})
        monthly = (1 + frame).resample("ME").prod() - 1
        corr = monthly.corr()
        iu = np.triu_indices_from(corr.values, k=1)
        pair = corr.values[iu]
        print(f"\n  Monthly-return correlation between the {len(legs)} coin legs: "
              f"mean {pair.mean():.2f}, median {np.median(pair):.2f}, "
              f"min {pair.min():.2f}, max {pair.max():.2f}")
        out[f"{geo_name}_leg_corr"] = {
            "mean": float(pair.mean()), "median": float(np.median(pair)),
            "min": float(pair.min()), "max": float(pair.max()),
            "matrix": corr.round(3).to_dict()}

        # --- baskets, each sized to the same DD as the single leg ---
        def build_basket(coins):
            def _b(risk):
                rs = {}
                for c in coins:
                    r = _run(key, attrs, f"{c}/USDT:USDT", risk)
                    if r is not None:
                        rs[c] = bar_returns(r)
                ix = sorted(set().union(*[set(s.index) for s in rs.values()]))
                f = pd.DataFrame({c: s.reindex(ix).fillna(0.0) for c, s in rs.items()})
                return f.mean(axis=1)
            return _b

        print()
        print(f"  {'portfolio':<34} {'risk%':>6} {'ret%':>9} {'CAGR%':>7} "
              f"{'maxDD%':>8} {'posMo%':>7} {'daysUW':>7} {'moVol%':>7}")
        rows = []
        # single SOL leg, sized to target
        risk, _, st = size_curve_to_dd(lambda rk: bar_returns(
            _run(key, attrs, "SOL/USDT:USDT", rk)))
        rows.append(("SOL alone", risk, st))
        # BTC alone — answers "or another leg that is BTC"
        risk, _, st = size_curve_to_dd(lambda rk: bar_returns(
            _run(key, attrs, "BTC/USDT:USDT", rk)))
        rows.append(("BTC alone", risk, st))
        # all 13 coins, no selection
        risk, _, st = size_curve_to_dd(build_basket(list(legs)))
        rows.append((f"basket of {len(legs)} (no selection)", risk, st))
        # best 6 by standalone return — IN-SAMPLE, flagged
        best6 = sorted(per_coin, key=lambda c: per_coin[c]["ret_pct"], reverse=True)[:6]
        risk, _, st = size_curve_to_dd(build_basket(best6))
        rows.append((f"best-6 IN-SAMPLE {'+'.join(best6)}", risk, st))

        for label, risk, st in rows:
            print(f"  {label:<34} {risk:>6.2f} {st['ret_pct']:>9.1f} "
                  f"{st['cagr_pct']:>7.1f} {st['max_dd_pct']:>8.1f} "
                  f"{st['pos_months_pct']:>7.1f} {st['max_days_underwater']:>7} "
                  f"{st['monthly_vol_pct']:>7.1f}")
        out[f"{geo_name}_portfolios"] = [
            {"label": lab, "risk_pct": rk, **stt} for lab, rk, stt in rows]
        print()

    path = REPO / "reports" / "sol_leg_basket.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
