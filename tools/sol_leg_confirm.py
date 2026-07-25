"""
SOL leg — confirmation pass on the walk-forward winners.

`tools/sol_leg_return_search.py` ranks candidates by compounded walk-forward
OOS return. This script pressure-tests the survivors the way they would
actually deploy:

  1. FROZEN-PARAM CONTINUOUS RUN — one param set, one uninterrupted backtest
     over the whole OOS span, instead of nine stitched fold windows. Stitching
     resets equity and drawdown at every fold boundary and hides the real
     peak-to-trough.
  2. PLATEAU CHECK — re-run the winner across its whole grid neighbourhood on
     the continuous span. A return that only exists at one grid point is a
     spike (overfit); a return that survives its neighbours is a plateau.
  3. COST STRESS — 5 bps (deployed default: 4 bps fee + 1 bp slip), 10 bps,
     15 bps per side. The 15 bps gate is the one BTC_SOL_PORTFOLIO_VERDICT.md
     flagged as never-measured.
  4. CROSS-ASSET CONTROL — same frozen params on BTC. Tells us whether this is
     a SOL edge or just a strategy that works everywhere (in which case it is
     not a reason to add a SOL leg).
  5. LEVERAGE / RISK LADDER — return and max-DD at the risk levels the live
     legs actually use, against the -35.5% kill-switch floor.

Run: .venv/bin/python tools/sol_leg_confirm.py
"""

from __future__ import annotations

import dataclasses
import itertools
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

# Frozen sets = the modal walk-forward pick per candidate.
FROZEN: dict[str, dict] = {
    "rider-v1": {
        "rider_donchian_n": 34, "rider_ema_period": 100,
        "rider_sl_atr": 1.0, "rider_tp_atr": 12.0, "rider_trail_atr": 0.0,
    },
    "rider-v1-alt": {
        "rider_donchian_n": 34, "rider_ema_period": 200,
        "rider_sl_atr": 1.0, "rider_tp_atr": 12.0, "rider_trail_atr": 0.0,
    },
    "st-donchexit::long": {
        "_st_long_only": True, "st_period": 10, "st_multiplier": 3.0,
        "st_donch_period": 20, "st_sl_atr": 1.0, "st_tp_atr": 6.0,
    },
    "st-adx-donchexit": {
        "st_period": 10, "st_multiplier": 3.0, "st_adx_min": 20,
        "st_donch_period": 20, "st_sl_atr": 1.0, "st_tp_atr": 6.0,
    },
    "donchian-v3": {
        "_p_donchian_period_entry": 20, "_p_donchian_period_exit": 10,
        "_p_slope_trend_threshold_pct": 0.0, "_p_regime_ema_period": 120,
    },
}
_BASE = {"rider-v1": "rider-v1", "rider-v1-alt": "rider-v1",
         "st-donchexit::long": "st-donchexit", "st-adx-donchexit": "st-adx-donchexit",
         "donchian-v3": "donchian-v3"}

RISK_PCT = 2.0
LEVERAGE = 3
DEFAULT_COMMISSION = 0.0005


def _apply(name: str, combo: dict, risk: float = RISK_PCT,
           leverage: int = LEVERAGE) -> StrategyParams | None:
    cls = STRATEGIES[_BASE[name]]
    long_only = bool(combo.get("_st_long_only", False))
    if hasattr(cls, "allow_shorts"):
        cls.allow_shorts = not long_only
    param_kwargs = {}
    for k, v in combo.items():
        if k == "_st_long_only":
            continue
        if k.startswith("_p_"):
            param_kwargs[k[3:]] = v
            continue
        setattr(cls, k, v)
    if hasattr(cls, "st_risk_per_trade_pct"):
        cls.st_risk_per_trade_pct = risk
    if hasattr(cls, "rider_risk_per_trade_pct"):
        cls.rider_risk_per_trade_pct = risk
    if param_kwargs or hasattr(cls, "risk_per_trade_pct"):
        return dataclasses.replace(
            StrategyParams.from_yaml(), risk_per_trade_pct=risk,
            leverage=leverage, **param_kwargs)
    return None


def _run(name: str, combo: dict, symbol: str = SOL,
         commission: float = DEFAULT_COMMISSION, risk: float = RISK_PCT,
         leverage: int = LEVERAGE, start: datetime = OOS_START,
         end: datetime = OOS_END) -> dict:
    override = _apply(name, combo, risk=risk, leverage=leverage)
    return run_backtest(
        _BASE[name], symbol, TF, start, end, leverage=leverage, quiet=True,
        params_override=override, commission=commission,
    )


def _fmt(r: dict) -> str:
    return (f"ret {r['after_funding_pct']:+8.1f}%  maxDD {r['max_drawdown_pct']:+7.1f}%  "
            f"PF {r['profit_factor']:4.2f}  WR {r['win_rate_pct']:4.1f}%  "
            f"n={r['trades']:>3}  avg {r['avg_trade_pct']:+.2f}%  "
            f"Sharpe {r['sharpe']:5.2f}")


def main() -> int:
    out: dict = {"oos_span": [str(OOS_START.date()), str(OOS_END.date())]}

    print("=" * 100)
    print(f"1. FROZEN-PARAM CONTINUOUS RUN — SOL 4h, {OOS_START.date()} → {OOS_END.date()}, "
          f"{LEVERAGE}x, risk {RISK_PCT}%, {DEFAULT_COMMISSION*1e4:.0f} bps/side")
    print("=" * 100)
    baseline = {}
    for name, combo in FROZEN.items():
        r = _run(name, combo)
        baseline[name] = r
        print(f"  {name:<22} {_fmt(r)}")
    bh = run_backtest("buy-and-hold", SOL, TF, OOS_START, OOS_END, leverage=1, quiet=True)
    print(f"  {'SOL buy-and-hold 1x':<22} ret {bh['after_funding_pct']:+8.1f}%  "
          f"maxDD {bh['max_drawdown_pct']:+7.1f}%")
    out["frozen_continuous"] = {
        k: {m: v[m] for m in ("after_funding_pct", "max_drawdown_pct", "profit_factor",
                              "win_rate_pct", "trades", "avg_trade_pct", "sharpe")}
        for k, v in baseline.items()}
    out["buy_and_hold"] = {"after_funding_pct": bh["after_funding_pct"],
                           "max_drawdown_pct": bh["max_drawdown_pct"]}

    print()
    print("=" * 100)
    print("2. PLATEAU CHECK — rider-v1 full grid on the continuous OOS span")
    print("=" * 100)
    grid = {
        "rider_donchian_n": [20, 34, 55],
        "rider_ema_period": [100, 200],
        "rider_sl_atr": [1.0, 1.5],
        "rider_tp_atr": [6.0, 8.0, 12.0],
        "rider_trail_atr": [0.0, 5.0],
    }
    keys = list(grid)
    plateau = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        try:
            r = _run("rider-v1", combo)
        except Exception as exc:
            print(f"  [warn] {combo}: {exc}")
            continue
        plateau.append({"combo": combo, "ret": r["after_funding_pct"],
                        "dd": r["max_drawdown_pct"], "pf": r["profit_factor"],
                        "trades": r["trades"], "wr": r["win_rate_pct"]})
    plateau.sort(key=lambda x: x["ret"], reverse=True)
    n_pos = sum(1 for x in plateau if x["ret"] > 0)
    print(f"  {len(plateau)} grid points, {n_pos} positive "
          f"({100 * n_pos / max(len(plateau), 1):.0f}%)")
    print(f"  median return {sorted(x['ret'] for x in plateau)[len(plateau)//2]:+.1f}%")
    print("\n  top 8:")
    for x in plateau[:8]:
        print(f"    ret {x['ret']:+8.1f}%  dd {x['dd']:+6.1f}%  PF {x['pf']:4.2f}  "
              f"n={x['trades']:>3}  {x['combo']}")
    print("\n  bottom 4:")
    for x in plateau[-4:]:
        print(f"    ret {x['ret']:+8.1f}%  dd {x['dd']:+6.1f}%  PF {x['pf']:4.2f}  "
              f"n={x['trades']:>3}  {x['combo']}")
    out["plateau_rider_v1"] = plateau

    print()
    print("=" * 100)
    print("3. COST STRESS — per-side commission (fee + slippage)")
    print("=" * 100)
    out["cost_stress"] = {}
    for name in ("rider-v1", "rider-v1-alt", "st-donchexit::long", "donchian-v3"):
        line, rec = [], {}
        for bps in (5, 10, 15, 20):
            r = _run(name, FROZEN[name], commission=bps / 1e4)
            line.append(f"{bps:>2}bps: {r['after_funding_pct']:+8.1f}% (PF {r['profit_factor']:4.2f})")
            rec[bps] = {"ret": r["after_funding_pct"], "pf": r["profit_factor"],
                        "dd": r["max_drawdown_pct"]}
        out["cost_stress"][name] = rec
        print(f"  {name:<22} " + "   ".join(line))

    print()
    print("=" * 100)
    print("4. CROSS-ASSET CONTROL — identical frozen params on BTC 4h")
    print("=" * 100)
    out["btc_control"] = {}
    for name in ("rider-v1", "rider-v1-alt", "st-donchexit::long", "donchian-v3"):
        r = _run(name, FROZEN[name], symbol=BTC)
        out["btc_control"][name] = {"after_funding_pct": r["after_funding_pct"],
                                    "max_drawdown_pct": r["max_drawdown_pct"],
                                    "profit_factor": r["profit_factor"],
                                    "trades": r["trades"]}
        print(f"  {name:<22} BTC  {_fmt(r)}")
        s = baseline[name]
        print(f"  {'':<22} SOL  ret {s['after_funding_pct']:+8.1f}%  "
              f"(SOL − BTC = {s['after_funding_pct'] - r['after_funding_pct']:+.1f}pp)")

    print()
    print("=" * 100)
    print("5. RISK LADDER — rider-v1 frozen, SOL 4h continuous "
          "(kill switch fires at -35.5%)")
    print("=" * 100)
    out["risk_ladder"] = {}
    for risk in (1.0, 2.0, 2.75, 3.5, 5.0):
        r = _run("rider-v1", FROZEN["rider-v1"], risk=risk)
        flag = "  <-- BREACHES KILL SWITCH" if r["max_drawdown_pct"] <= -35.5 else ""
        out["risk_ladder"][risk] = {"ret": r["after_funding_pct"],
                                    "dd": r["max_drawdown_pct"], "trades": r["trades"]}
        print(f"  risk {risk:>4.2f}%/trade   ret {r['after_funding_pct']:+9.1f}%   "
              f"maxDD {r['max_drawdown_pct']:+7.1f}%   n={r['trades']}{flag}")

    print()
    print("=" * 100)
    print("6. ANNUAL SLICES — rider-v1 frozen, SOL 4h (each year run independently)")
    print("=" * 100)
    out["annual"] = {}
    for y0, y1, label in [
        (datetime(2022, 4, 1, tzinfo=UTC), datetime(2023, 4, 1, tzinfo=UTC), "2022-04→2023-04"),
        (datetime(2023, 4, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC), "2023-04→2024-04"),
        (datetime(2024, 4, 1, tzinfo=UTC), datetime(2025, 4, 1, tzinfo=UTC), "2024-04→2025-04"),
        (datetime(2025, 4, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC), "2025-04→2026-04"),
        (datetime(2026, 4, 1, tzinfo=UTC), OOS_END, "2026-04→2026-07 (partial)"),
    ]:
        r = _run("rider-v1", FROZEN["rider-v1"], start=y0, end=y1)
        b = run_backtest("buy-and-hold", SOL, TF, y0, y1, leverage=1, quiet=True)
        out["annual"][label] = {"ret": r["after_funding_pct"],
                                "dd": r["max_drawdown_pct"], "trades": r["trades"],
                                "bh": b["after_funding_pct"]}
        print(f"  {label:<28} ret {r['after_funding_pct']:+8.1f}%  "
              f"maxDD {r['max_drawdown_pct']:+7.1f}%  n={r['trades']:>3}  "
              f"| SOL B&H {b['after_funding_pct']:+8.1f}%")

    path = REPO / "reports" / "sol_leg_confirm.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
