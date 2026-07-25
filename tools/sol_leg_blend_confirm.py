"""
SOL leg round 3 — matched-drawdown comparison of the win-rate-blended winners.

Round 2 ranked purely on return and produced a 14%-win-rate leg. God's verdict:
unwatchable. Round 3 re-ran the walk-forward with the selection objective
constrained to configs clearing a train win-rate floor (`blend30/40/50` in
tools/sol_leg_return_search.py) — return is still what gets maximised, but only
among configs with a tolerable hit rate.

The comparison here is at **matched drawdown**, which is the only fair way to
put a 14%-win-rate leg next to a 35%-win-rate one. A wide stop lowers win-rate
pain AND lowers max drawdown; a lower drawdown means the leg can be sized up
inside the same -35.5% kill-switch budget. So each candidate is sized by binary
search until its continuous max-DD hits the target, and only then are returns
compared. Comparing at equal `risk_per_trade_pct` instead would flatter the
narrow-stop configs for no reason other than that they take more risk.

Run: .venv/bin/python tools/sol_leg_blend_confirm.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SOL = "SOL/USDT:USDT"
BTC = "BTC/USDT:USDT"
TF = "4h"
OOS_START = datetime(2022, 4, 1, tzinfo=UTC)
OOS_END = datetime(2026, 7, 25, tzinfo=UTC)
YEARS = 4.32

DD_TARGET = -30.0        # leaves ~5.5pp of cushion under the -35.5% kill switch
LEVERAGE = 3
DEFAULT_COMMISSION = 0.0005

# (candidate label, STRATEGIES key, class attrs, provenance)
#
# Provenance matters more than the numbers. "WF blendNN" = the modal parameter
# set the walk-forward chose across its 9 folds under that objective, i.e. never
# fitted to the span reported below. "SCAN" = found by sweeping the reported
# span itself, so its numbers are in-sample and it is a hypothesis, not a
# validated candidate. Keep the distinction visible in every table.
FINALISTS: list[tuple[str, str, dict, str]] = [
    ("supertrend L+S", "supertrend",
     {"st_period": 14, "st_multiplier": 3.5, "st_sl_atr": 2.0, "st_tp_atr": 10.0,
      "allow_shorts": True}, "WF blend40 9/9"),
    ("rider-v1 wideSL", "rider-v1",
     {"rider_donchian_n": 34, "rider_ema_period": 100, "rider_sl_atr": 2.0,
      "rider_tp_atr": 12.0, "rider_trail_atr": 0.0}, "WF blend30 6/9"),
    ("rider-v1 tightTP", "rider-v1",
     {"rider_donchian_n": 34, "rider_ema_period": 100, "rider_sl_atr": 2.0,
      "rider_tp_atr": 4.0, "rider_trail_atr": 0.0}, "WF blend40 5/9"),
    ("st-dual fast", "st-dual",
     {"st_period": 7, "st_multiplier": 2.0, "st_slow_period": 30, "st_sl_atr": 1.0,
      "st_tp_atr": 6.0, "allow_shorts": True}, "WF blend30 7/9"),
    ("st-dual slow", "st-dual",
     {"st_period": 7, "st_multiplier": 3.0, "st_sl_atr": 2.0, "st_slow_period": 20,
      "st_tp_atr": 4.0, "allow_shorts": True}, "WF blend50 7/9"),
    ("sol-trend-rider", "sol-trend-rider",
     {"st_period": 7, "st_multiplier": 3.5, "st_sl_atr": 2.0,
      "sol_trail_atr": 5.0, "sol_donch_exit_period": 0, "allow_shorts": False},
     "WF blend40 7/9"),
    ("rider-v1 midWR", "rider-v1",
     {"rider_donchian_n": 34, "rider_ema_period": 200, "rider_sl_atr": 2.0,
      "rider_tp_atr": 6.0, "rider_trail_atr": 0.0}, "SCAN in-sample"),
    ("rider-v1 round-2", "rider-v1",
     {"rider_donchian_n": 34, "rider_ema_period": 200, "rider_sl_atr": 1.0,
      "rider_tp_atr": 12.0, "rider_trail_atr": 0.0}, "WF ret round-2"),
]

_RISK_ATTRS = ("st_risk_per_trade_pct", "rider_risk_per_trade_pct",
               "sol_risk_per_trade_pct")


def _run(key: str, attrs: dict, risk: float, symbol: str = SOL,
         commission: float = DEFAULT_COMMISSION,
         start: datetime = OOS_START, end: datetime = OOS_END) -> dict:
    cls = STRATEGIES[key]
    for k, v in attrs.items():
        setattr(cls, k, v)
    for ra in _RISK_ATTRS:
        if hasattr(cls, ra):
            setattr(cls, ra, risk)
    override = dataclasses.replace(
        StrategyParams.from_yaml(), risk_per_trade_pct=risk, leverage=LEVERAGE)
    return run_backtest(key, symbol, TF, start, end, leverage=LEVERAGE,
                        quiet=True, params_override=override, commission=commission)


def size_to_dd(key: str, attrs: dict, dd_target: float = DD_TARGET,
               lo: float = 0.25, hi: float = 15.0, iters: int = 14) -> tuple[float, dict]:
    """Binary-search risk_per_trade_pct until continuous max-DD ≈ dd_target.

    Max-DD is monotone in risk for these fractional-sizing strategies (every
    trade scales linearly with equity), so bisection is valid.
    """
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = _run(key, attrs, mid)
        dd = r["max_drawdown_pct"]
        if dd < dd_target:      # deeper than target -> risk down
            hi = mid
        else:
            lo = mid
            best = (mid, r)
    if best is None:
        r = _run(key, attrs, lo)
        best = (lo, r)
    return best


def main() -> int:
    out: dict = {"dd_target": DD_TARGET, "span": [str(OOS_START.date()), str(OOS_END.date())]}

    print("=" * 108)
    print(f"MATCHED-DRAWDOWN COMPARISON — SOL 4h, {OOS_START.date()} → {OOS_END.date()} "
          f"({YEARS} yr), each sized to maxDD ≈ {DD_TARGET}%")
    print("=" * 108)
    print(f"{'candidate':<20} {'provenance':<15} {'risk%':>6} {'ret%':>9} {'CAGR%':>7} "
          f"{'maxDD%':>8} {'WR%':>6} {'PF':>5} {'n':>5} {'ret/DD':>7}")
    rows = []
    for label, key, attrs, obj in FINALISTS:
        risk, r = size_to_dd(key, attrs)
        cagr = ((1 + r["after_funding_pct"] / 100) ** (1 / YEARS) - 1) * 100
        rows.append({
            "label": label, "key": key, "attrs": attrs, "objective": obj,
            "risk_pct": risk, "ret": r["after_funding_pct"], "cagr": cagr,
            "dd": r["max_drawdown_pct"], "wr": r["win_rate_pct"],
            "pf": r["profit_factor"], "trades": r["trades"],
        })
        print(f"{label:<20} {obj:<15} {risk:>6.2f} {r['after_funding_pct']:>9.1f} "
              f"{cagr:>7.1f} {r['max_drawdown_pct']:>8.1f} {r['win_rate_pct']:>6.1f} "
              f"{r['profit_factor']:>5.2f} {r['trades']:>5} "
              f"{r['after_funding_pct'] / abs(r['max_drawdown_pct']):>7.2f}")
    out["matched_dd"] = rows

    bh = run_backtest("buy-and-hold", SOL, TF, OOS_START, OOS_END, leverage=1, quiet=True)
    print(f"{'SOL buy-and-hold 1x':<20} {'benchmark':<14} {'-':>6} "
          f"{bh['after_funding_pct']:>9.1f} "
          f"{((1 + bh['after_funding_pct'] / 100) ** (1 / YEARS) - 1) * 100:>7.1f} "
          f"{bh['max_drawdown_pct']:>8.1f}")

    print()
    print("=" * 108)
    print("COST STRESS at the matched-DD size (per-side bps)")
    print("=" * 108)
    out["cost_stress"] = {}
    for row in rows:
        line, rec = [], {}
        for bps in (5, 10, 15, 20):
            r = _run(row["key"], row["attrs"], row["risk_pct"], commission=bps / 1e4)
            line.append(f"{bps:>2}: {r['after_funding_pct']:+7.1f}%")
            rec[bps] = r["after_funding_pct"]
        out["cost_stress"][row["label"]] = rec
        print(f"  {row['label']:<20} " + "  ".join(line))

    print()
    print("=" * 108)
    print("BTC CONTROL at the matched-DD size — is the edge SOL-specific?")
    print("=" * 108)
    out["btc_control"] = {}
    for row in rows:
        rb = _run(row["key"], row["attrs"], row["risk_pct"], symbol=BTC)
        out["btc_control"][row["label"]] = {
            "btc_ret": rb["after_funding_pct"], "btc_dd": rb["max_drawdown_pct"],
            "sol_ret": row["ret"]}
        print(f"  {row['label']:<20} SOL {row['ret']:+8.1f}%   BTC {rb['after_funding_pct']:+8.1f}% "
              f"(DD {rb['max_drawdown_pct']:+6.1f}%)   gap {row['ret'] - rb['after_funding_pct']:+8.1f}pp")

    print()
    print("=" * 108)
    print("ANNUAL SLICES at the matched-DD size")
    print("=" * 108)
    yrs = [
        (datetime(2022, 4, 1, tzinfo=UTC), datetime(2023, 4, 1, tzinfo=UTC), "22/23"),
        (datetime(2023, 4, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC), "23/24"),
        (datetime(2024, 4, 1, tzinfo=UTC), datetime(2025, 4, 1, tzinfo=UTC), "24/25"),
        (datetime(2025, 4, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC), "25/26"),
        (datetime(2026, 4, 1, tzinfo=UTC), OOS_END, "26 ytd"),
    ]
    print(f"  {'candidate':<20} " + "  ".join(f"{lab:>9}" for _, _, lab in yrs)
          + f"  {'ex-best-yr':>10}  {'neg yrs':>7}")
    out["annual"] = {}
    for row in rows:
        rets = [_run(row["key"], row["attrs"], row["risk_pct"], start=a, end=b)["after_funding_pct"]
                for a, b, _ in yrs]
        bi = max(range(len(rets)), key=lambda i: rets[i])
        ex = 1.0
        for i, v in enumerate(rets):
            if i != bi:
                ex *= (1 + v / 100)
        out["annual"][row["label"]] = {"rets": rets, "ex_best_year": (ex - 1) * 100}
        print(f"  {row['label']:<20} " + "  ".join(f"{v:>+9.1f}" for v in rets)
              + f"  {(ex - 1) * 100:>+10.1f}  {sum(1 for v in rets if v < 0):>7}")

    path = REPO / "reports" / "sol_leg_blend_confirm.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
