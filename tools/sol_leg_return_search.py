"""
SOL leg strategy search — walk-forward, ranked by RETURN (not win rate).

Motivation
----------
Every prior SOL attempt in this repo was killed on a *quality* gate:
`BTC_SOL_PORTFOLIO_VERDICT.md` shelved SOL because SOL-solo PSR (0.866) sat
below the 0.90 bar, and the cnh-hybrid-short revisit compared BTC-vs-SOL on
win rate. God's ask for this round is explicit: **rank on return**. So the
objective function here is compounded return after fees + slippage + funding,
and win rate is reported but never selected on.

Method
------
Rolling 18-month TRAIN / 6-month TEST, stepping 6 months, native-4h
SOL/USDT:USDT, leverage 3x, risk 2.0%/trade for every candidate (normalised
so the return ranking is apples-to-apples and not a sizing artifact).

For each (candidate, fold):
  1. Grid-search the candidate's params on TRAIN, picking the combo that
     maximises `after_funding_pct` subject to a trade floor.
  2. Freeze those params, run once on the untouched TEST window.

Two selection objectives share the same TRAIN runs:
  * `ret`     — max train return, trades >= MIN_TRAIN_TRADES
  * `ret_dd`  — same, but reject combos whose train max-DD is worse than
                TRAIN_DD_FLOOR (a survivability guardrail, not a quality gate)

Ranking metric = compounded OOS return across folds. Max-DD, PF and win rate
are carried along for the deploy decision but do not affect the ordering.

Writes:
  reports/sol_search_train/<candidate>.json  (train-run cache, one file per candidate)
  reports/sol_leg_return_search_oos.csv      (one row per candidate/fold/objective)

The train sweep is ~8.6k backtests (~25 min), so it is chunked and cached per
candidate — run `--train` in batches, then `--analyze` once.

Run:
    .venv/bin/python tools/sol_leg_return_search.py --train supertrend,rider-v1
    .venv/bin/python tools/sol_leg_return_search.py --train all
    .venv/bin/python tools/sol_leg_return_search.py --analyze
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

SYMBOL = "SOL/USDT:USDT"
TF = "4h"
LEVERAGE = 3
RISK_PCT = 2.0

MIN_TRAIN_TRADES = 8
TRAIN_DD_FLOOR = -45.0  # for the `ret_dd` objective only

# --- Fold schedule -----------------------------------------------------------
# SOL futures data starts 2020-09-14, so the first train window opens 2020-10-01.
TRAIN_MONTHS = 18
TEST_MONTHS = 6
STEP_MONTHS = 6
FIRST_TRAIN_START = date(2020, 10, 1)
DATA_END = date(2026, 7, 25)


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, d.day)


def build_folds() -> list[tuple[date, date, date, date, bool]]:
    """(train_start, train_end, test_start, test_end, is_partial)."""
    folds = []
    train_start = FIRST_TRAIN_START
    while True:
        train_end_excl = _add_months(train_start, TRAIN_MONTHS)
        test_end_excl = _add_months(train_end_excl, TEST_MONTHS)
        test_start = train_end_excl
        test_end = test_end_excl - timedelta(days=1)
        if test_start >= DATA_END:
            break
        partial = test_end > DATA_END
        if partial:
            test_end = DATA_END
            # Only keep a partial tail fold if it covers >= 2 months.
            if (test_end - test_start).days < 60:
                break
        folds.append(
            (train_start, train_end_excl - timedelta(days=1), test_start, test_end, partial)
        )
        train_start = _add_months(train_start, STEP_MONTHS)
    return folds


def _to_dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


# --- Candidates ---------------------------------------------------------------
# Grids deliberately include the SL/TP bracket (st_sl_atr / st_tp_atr,
# rider_sl_atr / rider_tp_atr). The earlier BTC bake-off gridded only the
# trend-flip geometry, which tunes *when* you're in but not *how much of the
# move you keep* — and on a high-beta alt like SOL the bracket is the dominant
# return lever.
CANDIDATES: dict[str, dict] = {
    "supertrend": {
        "st_period": [7, 10, 14, 20],
        "st_multiplier": [2.0, 2.5, 3.0, 3.5],
        "st_sl_atr": [1.0, 1.5, 2.0],
        "st_tp_atr": [4.0, 6.0, 10.0],
    },
    "supertrend::long": {
        "_st_long_only": [True],
        "st_period": [7, 10, 14, 20],
        "st_multiplier": [2.0, 2.5, 3.0, 3.5],
        "st_sl_atr": [1.0, 1.5, 2.0],
        "st_tp_atr": [4.0, 6.0, 10.0],
    },
    "st-adx": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_adx_min": [15, 20, 25],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-adx::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_adx_min": [15, 20, 25],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-ema": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_ema_period": [100, 200],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-dual": {
        "st_period": [7, 10],
        "st_multiplier": [2.0, 2.5, 3.0],
        "st_slow_period": [20, 30],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-donchexit": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_donch_period": [20, 30],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-donchexit::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_donch_period": [20, 30],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-voladapt": {
        "_st_mult_pair": [(2.0, 3.5), (2.0, 4.0), (2.5, 4.5)],
        "st_vol_lookback": [100],
        "st_period": [10, 14],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-adx-donchexit": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0],
        "st_adx_min": [15, 20],
        "st_donch_period": [20, 30],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-adx-donchexit::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0],
        "st_adx_min": [15, 20],
        "st_donch_period": [20, 30],
        "st_sl_atr": [1.0, 1.5],
        "st_tp_atr": [6.0, 10.0],
    },
    "st-trail": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_trail_atr": [2.0, 3.0, 4.0],
        "st_adx_min": [0, 20],
    },
    "st-trail::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_trail_atr": [2.0, 3.0, 4.0],
        "st_adx_min": [0, 20],
    },
    # Long-only asymmetric trend rider — small stop, very wide fixed target.
    # This is the shape that fits "return, not win rate": ~19% WR, PF 1.68 on
    # a 2023-24 spot check.
    "rider-v1": {
        "rider_donchian_n": [20, 34, 55],
        "rider_ema_period": [100, 200],
        "rider_sl_atr": [1.0, 1.5],
        "rider_tp_atr": [6.0, 8.0, 12.0],
        "rider_trail_atr": [0.0, 5.0],
    },
    # 2026-07-25 round 2 — the explicit restatement of what `st-donchexit::long`
    # actually was (Supertrend flip entry, ATR stop, NO take-profit, exit on
    # flip), plus the two exits that variant *meant* to test, correctly shifted.
    # See strategy/signals_sol_trend_rider.py.
    "sol-trend-rider": {
        "st_period": [7, 10, 14, 20],
        "st_multiplier": [2.0, 2.5, 3.0, 3.5],
        "st_sl_atr": [1.0, 1.5, 2.0],
        "sol_donch_exit_period": [0, 30],
        "sol_trail_atr": [0.0, 5.0],
    },
    # Donchian breakout + signed-EMA-slope regime gate. Params live on
    # StrategyParams (not class attrs) because _apply_params_to_class would
    # clobber class-level values from params.yaml.
    "donchian-v3": {
        "_p_donchian_period_entry": [20, 34, 55],
        "_p_donchian_period_exit": [10, 20],
        "_p_slope_trend_threshold_pct": [0.0, 0.05, 0.1],
        "_p_regime_ema_period": [120, 200],
    },
}

_LONG_ONLY_BASE: dict[str, str] = {
    "supertrend::long": "supertrend",
    "st-adx::long": "st-adx",
    "st-donchexit::long": "st-donchexit",
    "st-adx-donchexit::long": "st-adx-donchexit",
    "st-trail::long": "st-trail",
}


def _resolve(candidate: str) -> str:
    return _LONG_ONLY_BASE.get(candidate, candidate)


def _grid_combos(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combo: dict = {}
        for k, v in zip(keys, values):
            if k == "_st_mult_pair":
                combo["st_mult_low"], combo["st_mult_high"] = v
            else:
                combo[k] = v
        combos.append(combo)
    return combos


def _apply_combo(candidate: str, combo: dict) -> StrategyParams | None:
    """Set class attrs for this combo; return a params_override if needed.

    `allow_shorts` is set EXPLICITLY on every task (not only for ::long
    candidates) because worker processes are reused — a ::long task would
    otherwise leave allow_shorts=False stuck on the shared base class and
    silently turn the next long+short task into a long-only one.
    """
    base = _resolve(candidate)
    cls = STRATEGIES[base]
    long_only = bool(combo.get("_st_long_only", False))
    if hasattr(cls, "allow_shorts"):
        cls.allow_shorts = not long_only

    param_kwargs: dict = {}
    for attr, value in combo.items():
        if attr == "_st_long_only":
            continue
        if attr.startswith("_p_"):
            param_kwargs[attr[3:]] = value
            continue
        setattr(cls, attr, value)

    # Normalise risk sizing so the return ranking is not a sizing artifact.
    for risk_attr in ("st_risk_per_trade_pct", "rider_risk_per_trade_pct",
                      "sol_risk_per_trade_pct"):
        if hasattr(cls, risk_attr):
            setattr(cls, risk_attr, RISK_PCT)

    if param_kwargs or hasattr(cls, "risk_per_trade_pct"):
        base_params = StrategyParams.from_yaml()
        return dataclasses.replace(
            base_params, risk_per_trade_pct=RISK_PCT, leverage=LEVERAGE, **param_kwargs
        )
    return None


_METRIC_KEYS = (
    "trades", "after_funding_pct", "backtest_return_pct", "max_drawdown_pct",
    "profit_factor", "win_rate_pct", "avg_trade_pct", "sharpe",
)


def _run_one(task: tuple) -> dict:
    """Worker entry point. task = (candidate, combo, start_iso, end_iso, tag)."""
    candidate, combo, start_iso, end_iso, tag = task
    params_override = _apply_combo(candidate, combo)
    out = {"candidate": candidate, "combo": combo, "tag": tag,
           "start": start_iso, "end": end_iso}
    try:
        r = run_backtest(
            _resolve(candidate), SYMBOL, TF,
            datetime.fromisoformat(start_iso), datetime.fromisoformat(end_iso),
            leverage=LEVERAGE, quiet=True, params_override=params_override,
        )
    except Exception as exc:  # data gaps, degenerate params
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    for k in _METRIC_KEYS:
        v = r.get(k)
        out[k] = None if v is None else float(v)
    return out


def _select(train_runs: list[dict], dd_floor: float | None) -> dict | None:
    """Pick the max-return train combo subject to trade floor (+ optional DD floor)."""
    eligible = [
        r for r in train_runs
        if "error" not in r
        and r.get("after_funding_pct") is not None
        and (r["trades"] or 0) >= MIN_TRAIN_TRADES
        and (dd_floor is None or (r["max_drawdown_pct"] or 0.0) >= dd_floor)
    ]
    if eligible:
        return max(eligible, key=lambda r: r["after_funding_pct"])
    # Fallback: whatever traded the most, so a fold is never silently dropped.
    traded = [r for r in train_runs if "error" not in r and (r.get("trades") or 0) > 0]
    if traded:
        return max(traded, key=lambda r: r["trades"])
    return None


REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
TRAIN_CACHE = REPORTS / "sol_search_train"


def _workers() -> int:
    return max(1, min(6, (os.cpu_count() or 4) - 2))


def _print_folds(folds) -> None:
    print("=== Fold schedule (SOL/USDT:USDT 4h, "
          f"{TRAIN_MONTHS}mo train / {TEST_MONTHS}mo test / {STEP_MONTHS}mo step) ===")
    for i, (trs, tre, tes, tee, partial) in enumerate(folds):
        flag = "  [PARTIAL]" if partial else ""
        print(f"  f{i}  train {trs} → {tre}   test {tes} → {tee}{flag}")
    print(f"\n{len(folds)} folds. OOS span {folds[0][2]} → {folds[-1][3]}.\n")


def _cache_path(candidate: str) -> Path:
    return TRAIN_CACHE / f"{candidate.replace('::', '__')}.json"


def run_train(candidates: list[str], force: bool = False) -> int:
    """Phase 1 — sweep every (combo, fold) on TRAIN, cache per candidate."""
    TRAIN_CACHE.mkdir(parents=True, exist_ok=True)
    folds = build_folds()
    _print_folds(folds)

    for candidate in candidates:
        out = _cache_path(candidate)
        if out.exists() and not force:
            print(f"{candidate}: cached ({out.name}), skipping")
            continue
        combos = _grid_combos(CANDIDATES[candidate])
        tasks = [
            (candidate, combo, _to_dt(trs).isoformat(),
             _to_dt(tre + timedelta(days=1)).isoformat(), f"train:{fold_idx}")
            for fold_idx, (trs, tre, _, _, _) in enumerate(folds)
            for combo in combos
        ]
        t0 = time.time()
        print(f"{candidate}: {len(combos)} combos x {len(folds)} folds = "
              f"{len(tasks)} runs on {_workers()} workers...", flush=True)
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=_workers()) as pool:
            for i, res in enumerate(pool.map(_run_one, tasks, chunksize=8), 1):
                results.append(res)
                if i % 400 == 0:
                    print(f"    {i}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)
        n_err = sum(1 for r in results if "error" in r)
        with open(out, "w") as f:
            json.dump(results, f, default=str)
        print(f"  -> {out.name}  {len(results)} runs, {n_err} errored, "
              f"{time.time() - t0:.0f}s\n", flush=True)
    return 0


def run_analyze() -> int:
    """Phase 2 — freeze the max-return train pick per fold, run OOS, rank."""
    folds = build_folds()
    _print_folds(folds)

    train_results: list[dict] = []
    missing = []
    for candidate in CANDIDATES:
        p = _cache_path(candidate)
        if not p.exists():
            missing.append(candidate)
            continue
        with open(p) as f:
            train_results.extend(json.load(f))
    if missing:
        print(f"[warn] no train cache for: {', '.join(missing)} "
              f"— run --train for them first.\n", file=sys.stderr)
    if not train_results:
        print("No train cache at all. Run --train first.", file=sys.stderr)
        return 1

    reports = REPORTS
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in train_results:
        by_key.setdefault((r["candidate"], r["tag"]), []).append(r)

    objectives = {"ret": None, "ret_dd": TRAIN_DD_FLOOR}
    oos_tasks, oos_meta = [], []
    for candidate in CANDIDATES:
        for fold_idx, (_, _, tes, tee, partial) in enumerate(folds):
            runs = by_key.get((candidate, f"train:{fold_idx}"), [])
            for obj_name, dd_floor in objectives.items():
                chosen = _select(runs, dd_floor)
                if chosen is None:
                    continue
                oos_tasks.append(
                    (candidate, chosen["combo"], _to_dt(tes).isoformat(),
                     _to_dt(tee + timedelta(days=1)).isoformat(),
                     f"oos:{obj_name}:{fold_idx}")
                )
                oos_meta.append({
                    "candidate": candidate, "objective": obj_name, "fold_idx": fold_idx,
                    "partial": partial,
                    "train_start": str(folds[fold_idx][0]), "train_end": str(folds[fold_idx][1]),
                    "test_start": str(tes), "test_end": str(tee),
                    "chosen_params": json.dumps(chosen["combo"]),
                    "train_return_pct": chosen.get("after_funding_pct"),
                    "train_trades": chosen.get("trades"),
                    "train_maxdd_pct": chosen.get("max_drawdown_pct"),
                })

    print(f"Phase 2: {len(oos_tasks)} OOS runs...")
    oos_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        oos_results = list(pool.map(_run_one, oos_tasks, chunksize=4))
    print("  done\n")

    # --- Benchmark: SOL buy-and-hold per test window -----------------------
    bh: dict[int, float] = {}
    for fold_idx, (_, _, tes, tee, _) in enumerate(folds):
        try:
            r = run_backtest(
                "buy-and-hold", SYMBOL, TF, _to_dt(tes),
                _to_dt(tee + timedelta(days=1)), leverage=1, quiet=True,
            )
            bh[fold_idx] = float(r["after_funding_pct"])
        except Exception as exc:
            print(f"  [warn] B&H fold {fold_idx} failed: {exc}", file=sys.stderr)
            bh[fold_idx] = float("nan")

    rows = []
    for meta, res in zip(oos_meta, oos_results):
        row = dict(meta)
        row["bh_return_pct"] = bh.get(meta["fold_idx"])
        if "error" in res:
            row["error"] = res["error"]
        for k in _METRIC_KEYS:
            row[f"oos_{k}"] = res.get(k)
        rows.append(row)

    csv_path = reports / "sol_leg_return_search_oos.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} OOS rows to {csv_path}\n")

    # --- Ranking: compounded OOS return ------------------------------------
    for obj_name in objectives:
        print(f"=== Ranked by compounded OOS return — objective `{obj_name}` ===")
        print(f"{'candidate':<24} {'folds':>5} {'compOOS%':>10} {'+folds':>7} "
              f"{'median%':>9} {'worst%':>9} {'worstDD%':>9} {'trades':>7} "
              f"{'medWR%':>7} {'medPF':>6}")
        summary = []
        for candidate in CANDIDATES:
            crs = [r for r in rows
                   if r["candidate"] == candidate and r["objective"] == obj_name
                   and r.get("oos_after_funding_pct") is not None]
            if not crs:
                continue
            rets = [r["oos_after_funding_pct"] for r in crs]
            comp = 1.0
            for x in rets:
                comp *= (1.0 + x / 100.0)
            comp_pct = (comp - 1.0) * 100.0
            summary.append({
                "candidate": candidate, "n": len(crs), "comp": comp_pct,
                "pos": sum(1 for x in rets if x > 0),
                "median": statistics.median(rets), "worst": min(rets),
                "worst_dd": min(r["oos_max_drawdown_pct"] or 0.0 for r in crs),
                "trades": sum(int(r["oos_trades"] or 0) for r in crs),
                "wr": statistics.median([r["oos_win_rate_pct"] or 0.0 for r in crs]),
                "pf": statistics.median([r["oos_profit_factor"] or 0.0 for r in crs]),
            })
        for s in sorted(summary, key=lambda s: s["comp"], reverse=True):
            print(f"{s['candidate']:<24} {s['n']:>5} {s['comp']:>10.2f} "
                  f"{s['pos']}/{s['n']:<5} {s['median']:>9.2f} {s['worst']:>9.2f} "
                  f"{s['worst_dd']:>9.2f} {s['trades']:>7} {s['wr']:>7.1f} {s['pf']:>6.2f}")
        print()

    comp_bh = 1.0
    for fold_idx in range(len(folds)):
        v = bh.get(fold_idx)
        if v is not None and v == v:
            comp_bh *= (1.0 + v / 100.0)
    print(f"Benchmark — SOL buy-and-hold 1x compounded over the same OOS folds: "
          f"{(comp_bh - 1.0) * 100.0:+.2f}%")
    print("Per-fold B&H: " + "  ".join(
        f"f{i}={bh.get(i, float('nan')):+.1f}%" for i in range(len(folds))))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", help="comma-separated candidate keys, or 'all'")
    p.add_argument("--analyze", action="store_true",
                   help="freeze train picks, run OOS, print the return ranking")
    p.add_argument("--force", action="store_true", help="ignore the train cache")
    p.add_argument("--list", action="store_true", help="list candidates + combo counts")
    args = p.parse_args()

    if args.list:
        total = 0
        for k, g in CANDIDATES.items():
            n = len(_grid_combos(g))
            total += n
            print(f"  {k:<24} {n:>4} combos")
        print(f"  {'TOTAL':<24} {total:>4} combos x {len(build_folds())} folds")
        return 0

    if args.train:
        keys = list(CANDIDATES) if args.train == "all" else args.train.split(",")
        unknown = [k for k in keys if k not in CANDIDATES]
        if unknown:
            print(f"unknown candidates: {unknown}", file=sys.stderr)
            return 2
        run_train(keys, force=args.force)

    if args.analyze:
        return run_analyze()
    if not args.train:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
