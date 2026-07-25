"""
Round 4b — does coin selection survive out-of-sample, and how correlated is the
SOL leg with the BTC legs already deployed?

Round 4a found two things that pull against each other:
  * Only 3 of 12 coins are strongly profitable under the round-3 geometry
    (SOL, ADA, AVAX). ETH / XRP / LTC / BCH lose money. So equal-weighting ALL
    coins dilutes into the losers: the un-selected basket smooths the ride but
    gives up most of the return (CAGR 18.5% vs SOL alone 60.7%).
  * The in-sample "best 6" is far better — and the best-6 sets for the two
    different geometries overlap on 5 of 6 coins (SOL, ADA, AVAX, ATOM, LINK).
    Cross-geometry agreement is evidence that those coins genuinely trend,
    not just that they were picked on the reported span.

Cheap in-sample top-N is not admissible evidence though, so this walk-forwards
the *coin choice* itself: in each fold pick the top-K coins by TRAIN return
only, then equal-weight exactly those in the untouched TEST window. If selected
baskets beat the un-selected basket out-of-sample, coin selection is a real
lever; if not, it was hindsight.

Part 2 answers God's "yep" to the correlation question: monthly-return
correlation of the SOL candidate against the two BTC legs actually running
(multifactor-v1 on 15m with the 4H gate, and donchian-v3 on 4h), which decides
whether this leg diversifies the book or doubles down on it.

Run: .venv/bin/python tools/sol_leg_basket_wf.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402
from tools.sol_leg_basket import (  # noqa: E402
    COINS, GEOMETRIES, LEVERAGE, OOS_END, OOS_START, TF, YEARS,
    _RISK_ATTRS, bar_returns, curve_stats,
)
from tools.sol_leg_return_search import build_folds, _to_dt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BASE_RISK = 2.0


def _run(key: str, attrs: dict, symbol: str, risk: float,
         start: datetime, end: datetime, tf: str = TF) -> dict | None:
    cls = STRATEGIES[key]
    for k, v in attrs.items():
        setattr(cls, k, v)
    for ra in _RISK_ATTRS:
        if hasattr(cls, ra):
            setattr(cls, ra, risk)
    try:
        return run_backtest(
            key, symbol, tf, start, end, leverage=LEVERAGE, quiet=True,
            params_override=dataclasses.replace(
                StrategyParams.from_yaml(), risk_per_trade_pct=risk, leverage=LEVERAGE),
            return_equity=True)
    except Exception:
        return None


def main() -> int:
    out: dict = {}
    folds = build_folds()

    # ------------------------------------------------------------------
    # Part 1 — walk-forward the coin choice
    # ------------------------------------------------------------------
    for geo_name, (key, attrs) in GEOMETRIES.items():
        print("=" * 104)
        print(f"WALK-FORWARD COIN SELECTION — geometry `{geo_name}`, risk {BASE_RISK}%")
        print("  (coins ranked by TRAIN return each fold; equal-weighted in TEST)")
        print("=" * 104)
        rows = []
        chosen_log = []
        for fi, (trs, tre, tes, tee, partial) in enumerate(folds):
            train_ret = {}
            for c in COINS:
                r = _run(key, attrs, f"{c}/USDT:USDT", BASE_RISK,
                         _to_dt(trs), _to_dt(tre + timedelta(days=1)))
                if r is not None:
                    train_ret[c] = r["after_funding_pct"]
            ranked = sorted(train_ret, key=lambda c: train_ret[c], reverse=True)

            test_ret = {}
            for c in COINS:
                r = _run(key, attrs, f"{c}/USDT:USDT", BASE_RISK,
                         _to_dt(tes), _to_dt(tee + timedelta(days=1)))
                if r is not None:
                    test_ret[c] = bar_returns(r)

            def basket(coins):
                use = {c: test_ret[c] for c in coins if c in test_ret}
                if not use:
                    return float("nan")
                ix = sorted(set().union(*[set(s.index) for s in use.values()]))
                f = pd.DataFrame({c: s.reindex(ix).fillna(0.0) for c, s in use.items()})
                m = f.mean(axis=1)
                return ((1 + m).prod() - 1) * 100.0

            row = {
                "fold": fi, "test": f"{tes}→{tee}",
                "top4": basket(ranked[:4]), "top6": basket(ranked[:6]),
                "all": basket(list(test_ret)), "sol": basket(["SOL"]),
            }
            rows.append(row)
            chosen_log.append({"fold": fi, "top6_train_pick": ranked[:6]})
            print(f"  f{fi} {row['test']}  train-pick top6={'+'.join(ranked[:6]):<34} "
                  f"OOS top4 {row['top4']:+7.1f}%  top6 {row['top6']:+7.1f}%  "
                  f"all {row['all']:+7.1f}%  SOL {row['sol']:+7.1f}%")

        print(f"\n  {'variant':<12} {'chained%':>10} {'+folds':>8} {'median%':>9} "
              f"{'worst%':>9} {'ex-best%':>9}")
        summary = {}
        for v in ("top4", "top6", "all", "sol"):
            vals = [r[v] for r in rows if r[v] == r[v]]
            comp = 1.0
            for x in vals:
                comp *= (1 + x / 100)
            bi = max(range(len(vals)), key=lambda i: vals[i])
            ex = 1.0
            for i, x in enumerate(vals):
                if i != bi:
                    ex *= (1 + x / 100)
            summary[v] = {"chained": (comp - 1) * 100, "pos": sum(1 for x in vals if x > 0),
                          "n": len(vals), "median": float(np.median(vals)),
                          "worst": min(vals), "ex_best": (ex - 1) * 100}
            s = summary[v]
            print(f"  {v:<12} {s['chained']:>10.1f} {s['pos']}/{s['n']:<6} "
                  f"{s['median']:>9.1f} {s['worst']:>9.1f} {s['ex_best']:>9.1f}")
        out[f"{geo_name}_wf_coin_selection"] = {"folds": rows, "summary": summary,
                                               "picks": chosen_log}
        print()

    # ------------------------------------------------------------------
    # Part 2 — correlation vs the deployed BTC legs
    # ------------------------------------------------------------------
    print("=" * 104)
    print("CORRELATION vs THE DEPLOYED BTC LEGS — monthly returns, "
          f"{OOS_START.date()} → {OOS_END.date()}")
    print("=" * 104)
    series: dict[str, pd.Series] = {}

    for geo_name, (key, attrs) in GEOMETRIES.items():
        r = _run(key, attrs, "SOL/USDT:USDT", BASE_RISK, OOS_START, OOS_END)
        if r is not None:
            series[f"SOL {geo_name}"] = bar_returns(r)

    print("  running BTC multifactor-v1 (15m + 4H gate, the deployed leg)...", flush=True)
    mf = run_backtest("multifactor-v1", "BTC/USDT:USDT", "15m", OOS_START, OOS_END,
                      quiet=True, return_equity=True)
    series["BTC multifactor-v1"] = bar_returns(mf)
    print(f"    ret {mf['after_funding_pct']:+.1f}%  DD {mf['max_drawdown_pct']:+.1f}%  "
          f"WR {mf['win_rate_pct']:.1f}%  n={mf['trades']}")

    # donchian-v3 needs config/params_donchian.yaml AND a manual patch for the
    # two keys StrategyParams.from_yaml silently drops (donchian_period_entry,
    # slope_trend_threshold_pct). Loading it the naive way measures a 20-bar
    # ungated breakout — 454 trades, maxDD -63.6% — instead of the deployed
    # 80-bar gated one (135 trades, -32.9%). See leg_comparison for the proof.
    print("  running BTC donchian-v3 (4h, deployed params, correctly loaded)...",
          flush=True)
    from tools.leg_comparison import deployed_donchian_params  # noqa: PLC0415
    dparams = deployed_donchian_params()
    dv3 = run_backtest("donchian-v3", "BTC/USDT:USDT", "4h", OOS_START, OOS_END,
                       leverage=dparams.leverage, quiet=True, return_equity=True,
                       params_override=dparams)
    series["BTC donchian-v3"] = bar_returns(dv3)
    print(f"    ret {dv3['after_funding_pct']:+.1f}%  DD {dv3['max_drawdown_pct']:+.1f}%  "
          f"WR {dv3['win_rate_pct']:.1f}%  n={dv3['trades']}")

    ix = sorted(set().union(*[set(s.index) for s in series.values()]))
    frame = pd.DataFrame({k: s.reindex(ix).fillna(0.0) for k, s in series.items()})
    monthly = (1 + frame).resample("ME").prod() - 1
    corr = monthly.corr()
    print("\n  Monthly-return correlation matrix:")
    print(corr.round(2).to_string())
    out["corr_vs_deployed"] = corr.round(3).to_dict()
    out["deployed_legs"] = {
        "multifactor-v1": {k: mf[k] for k in
                           ("after_funding_pct", "max_drawdown_pct", "win_rate_pct", "trades")},
        "donchian-v3": {k: dv3[k] for k in
                        ("after_funding_pct", "max_drawdown_pct", "win_rate_pct", "trades")},
    }

    # What does adding the SOL leg do to the existing two-leg BTC book?
    print("\n  Effect of adding the SOL leg to the existing BTC book "
          "(equal weight, each sized to its own risk):")
    combos = {
        "BTC book only (mf-v1 + donch-v3)": ["BTC multifactor-v1", "BTC donchian-v3"],
        "+ SOL supertrend": ["BTC multifactor-v1", "BTC donchian-v3", "SOL supertrend"],
        "+ SOL st-dual": ["BTC multifactor-v1", "BTC donchian-v3", "SOL st-dual"],
    }
    print(f"  {'book':<36} {'ret%':>9} {'CAGR%':>7} {'maxDD%':>8} "
          f"{'posMo%':>7} {'daysUW':>7} {'moVol%':>7}")
    out["book_effect"] = {}
    for label, cols in combos.items():
        m = frame[cols].mean(axis=1)
        st = curve_stats(m, label)
        out["book_effect"][label] = st
        print(f"  {label:<36} {st['ret_pct']:>9.1f} {st['cagr_pct']:>7.1f} "
              f"{st['max_dd_pct']:>8.1f} {st['pos_months_pct']:>7.1f} "
              f"{st['max_days_underwater']:>7} {st['monthly_vol_pct']:>7.1f}")

    path = REPO / "reports" / "sol_leg_basket_wf.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
