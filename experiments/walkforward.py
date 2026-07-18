"""
Walk-forward bake-off for the Supertrend family (base + 5 variants).

Rolling 24-month train / 6-month test, stepping 6 months, on native-4h
BTC/USDT:USDT data, leverage 3x. For each candidate and fold:
  1. Grid-search the candidate's param grid on the TRAIN window, selecting
     the combo that maximizes profit_factor subject to trades >= 10 (or, if
     no combo clears the trade floor, the combo with the most trades).
  2. Freeze the chosen params and run once on the TEST (OOS) window.

Writes experiments/walkforward_results.csv (one row per candidate/fold) and
prints a fold table + per-candidate aggregate summary to stdout.

Run: .venv/bin/python experiments/walkforward.py
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402

SYMBOL = "BTC/USDT:USDT"
TF = "4h"
LEVERAGE = 3

# --- Fold schedule -----------------------------------------------------------
# First train: 2020-01-01 -> 2021-12-31 (24mo). First test: 2022-01-01 -> 2022-06-30 (6mo).
# Step the whole window +6mo while test_end <= 2026-06-01.
TRAIN_MONTHS = 24
TEST_MONTHS = 6
STEP_MONTHS = 6
WALK_FORWARD_LIMIT = date(2026, 6, 1)


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, d.day)


def build_folds() -> list[tuple[date, date, date, date]]:
    folds = []
    train_start = date(2020, 1, 1)
    while True:
        train_end_exclusive = _add_months(train_start, TRAIN_MONTHS)
        train_end = train_end_exclusive - timedelta(days=1)
        test_start = train_end_exclusive
        test_end_exclusive = _add_months(test_start, TEST_MONTHS)
        test_end = test_end_exclusive - timedelta(days=1)
        if test_end > WALK_FORWARD_LIMIT:
            break
        folds.append((train_start, train_end, test_start, test_end))
        train_start = _add_months(train_start, STEP_MONTHS)
    return folds


def _to_dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


# --- Candidates ----------------------------------------------------------------
# Each candidate: strategy key + grid of {class_attr: [values]} to sweep.
# Grids per spec, with the trims noted below.
CANDIDATES: dict[str, dict] = {
    "supertrend": {
        "st_period": [7, 10, 14, 20],
        "st_multiplier": [2.0, 2.5, 3.0, 3.5, 4.0],
    },
    "st-adx": {
        # Trimmed period/mult grid (per spec's "or a reduced" option) to keep
        # total runs reasonable: 2x3x3 = 18 combos/fold vs 4x5x3 = 60.
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_adx_min": [15, 20, 25],
    },
    "st-ema": {
        "st_period": [7, 10, 14, 20],
        "st_multiplier": [2.0, 2.5, 3.0, 3.5, 4.0],
        "st_ema_period": [100, 200],
    },
    "st-dual": {
        "st_period": [7, 10],
        "st_multiplier": [2.0, 2.5, 3.0],
        "st_slow_period": [20, 30],
    },
    "st-donchexit": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_donch_period": [20, 30],
    },
    "st-voladapt": {
        "st_vol_lookback": [100],
        "_st_mult_pair": [(2.0, 3.5), (2.0, 4.0), (2.5, 4.5)],
        "st_period": [10, 14],
    },
    # --- Round 2 ---------------------------------------------------------
    # New combo: ADX entry gate + Donchian-channel exit (both-directions).
    "st-adx-donchexit": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_adx_min": [15, 20, 25],
        "st_donch_period": [20, 30],
    },
    # New: chandelier trailing stop, optional ADX gate (both-directions).
    "st-trail": {
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_trail_atr": [2.0, 3.0, 4.0],
    },
    # --- Round 2 long-only hypothesis: force allow_shorts=False (fixed, not
    # swept), grid only the other params. BTC structurally rose 2020-2026;
    # flip-shorting likely bleeds.
    "supertrend::long": {
        "_st_long_only": [True],
        "st_period": [7, 10, 14, 20],
        "st_multiplier": [2.0, 2.5, 3.0, 3.5, 4.0],
    },
    "st-adx::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_adx_min": [15, 20, 25],
    },
    "st-donchexit::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_donch_period": [20, 30],
    },
    "st-adx-donchexit::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_adx_min": [15, 20, 25],
        "st_donch_period": [20, 30],
    },
    "st-trail::long": {
        "_st_long_only": [True],
        "st_period": [10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "st_trail_atr": [2.0, 3.0, 4.0],
        "st_adx_min": [0, 20],
    },
}

# Maps a "::long"-suffixed candidate key back to its underlying STRATEGIES key.
_LONG_ONLY_BASE_STRATEGY: dict[str, str] = {
    "supertrend::long": "supertrend",
    "st-adx::long": "st-adx",
    "st-donchexit::long": "st-donchexit",
    "st-adx-donchexit::long": "st-adx-donchexit",
    "st-trail::long": "st-trail",
}


def _grid_combos(grid: dict) -> list[dict]:
    """Expand a param grid dict into a list of {attr: value} combos.

    `_st_mult_pair` is special-cased: expands to st_mult_low/st_mult_high.
    `_st_long_only` is special-cased: consumed by _apply_combo/_run to force
    allow_shorts=False on the underlying strategy class (NOT swept — every
    combo for a `::long` candidate carries the same fixed True value).
    """
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combo = {}
        for k, v in zip(keys, values):
            if k == "_st_mult_pair":
                combo["st_mult_low"], combo["st_mult_high"] = v
            else:
                combo[k] = v
        combos.append(combo)
    return combos


def _resolve_strategy_key(strategy_key: str) -> str:
    """Map a `::long`-suffixed candidate key to its underlying STRATEGIES key."""
    return _LONG_ONLY_BASE_STRATEGY.get(strategy_key, strategy_key)


def _apply_combo(strategy_key: str, combo: dict) -> None:
    base_key = _resolve_strategy_key(strategy_key)
    cls = STRATEGIES[base_key]
    for attr, value in combo.items():
        if attr == "_st_long_only":
            cls.allow_shorts = not value
            continue
        setattr(cls, attr, value)


def _run(strategy_key: str, start: date, end: date) -> dict:
    base_key = _resolve_strategy_key(strategy_key)
    return run_backtest(
        base_key, SYMBOL, TF, _to_dt(start), _to_dt(end),
        leverage=LEVERAGE, quiet=True,
    )


def _select_best(strategy_key: str, combos: list[dict], train_start: date, train_end: date) -> tuple[dict, dict]:
    """Grid-search on the train window. Returns (chosen_combo, chosen_train_result)."""
    best_pf_combo, best_pf_result = None, None
    best_trades_combo, best_trades_result = None, None

    for combo in combos:
        _apply_combo(strategy_key, combo)
        try:
            result = _run(strategy_key, train_start, train_end)
        except Exception as exc:  # pragma: no cover - data gaps etc.
            print(f"  [warn] {strategy_key} {combo} train run failed: {exc}", file=sys.stderr)
            continue

        if best_trades_result is None or result["trades"] > best_trades_result["trades"]:
            best_trades_combo, best_trades_result = combo, result

        if result["trades"] >= 10:
            if best_pf_result is None or result["profit_factor"] > best_pf_result["profit_factor"]:
                best_pf_combo, best_pf_result = combo, result

    if best_pf_combo is not None:
        return best_pf_combo, best_pf_result
    return best_trades_combo, best_trades_result


def main() -> int:
    folds = build_folds()

    print("=== Fold schedule ===")
    print(f"{'fold':>4}  {'train_start':>11}  {'train_end':>11}  {'test_start':>11}  {'test_end':>11}")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"{i:>4}  {tr_s.isoformat():>11}  {tr_e.isoformat():>11}  {te_s.isoformat():>11}  {te_e.isoformat():>11}")
    print(f"\n{len(folds)} folds total.\n")

    rows = []  # CSV rows
    per_candidate_oos: dict[str, list[dict]] = {k: [] for k in CANDIDATES}

    for strategy_key, grid in CANDIDATES.items():
        combos = _grid_combos(grid)
        print(f"=== {strategy_key} ({len(combos)} combos/fold x {len(folds)} folds = "
              f"{len(combos) * len(folds)} train runs) ===")

        for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
            chosen_combo, train_result = _select_best(strategy_key, combos, tr_s, tr_e)
            if chosen_combo is None:
                print(f"  fold {fold_idx}: no valid train result, skipping")
                continue

            _apply_combo(strategy_key, chosen_combo)
            try:
                oos = _run(strategy_key, te_s, te_e)
            except Exception as exc:  # pragma: no cover
                print(f"  fold {fold_idx}: OOS run failed: {exc}", file=sys.stderr)
                continue

            row = {
                "candidate": strategy_key,
                "fold_idx": fold_idx,
                "train_start": tr_s.isoformat(),
                "train_end": tr_e.isoformat(),
                "test_start": te_s.isoformat(),
                "test_end": te_e.isoformat(),
                "chosen_params": json.dumps(chosen_combo),
                "oos_pf": oos["profit_factor"],
                "oos_avg_trade_pct": oos["avg_trade_pct"],
                "oos_trades": oos["trades"],
                "oos_return_pct": oos["after_funding_pct"],
                "oos_maxdd_pct": oos["max_drawdown_pct"],
            }
            rows.append(row)
            per_candidate_oos[strategy_key].append(row)
            print(f"  fold {fold_idx}: chosen={chosen_combo} "
                  f"(train PF={train_result['profit_factor']:.2f}, trades={train_result['trades']}) "
                  f"-> OOS PF={oos['profit_factor']:.2f} avg_trade={oos['avg_trade_pct']:+.3f}% "
                  f"trades={oos['trades']} return={oos['after_funding_pct']:+.2f}% "
                  f"maxdd={oos['max_drawdown_pct']:.2f}%")
        print()

    # --- Write CSV ---
    out_path = Path(__file__).resolve().parent / "walkforward_results.csv"
    fieldnames = [
        "candidate", "fold_idx", "train_start", "train_end", "test_start", "test_end",
        "chosen_params", "oos_pf", "oos_avg_trade_pct", "oos_trades", "oos_return_pct", "oos_maxdd_pct",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}\n")

    # --- Aggregate summary ---
    print("=== Per-candidate aggregate summary ===")
    header = (f"{'candidate':<14} {'n_folds':>7} {'median_OOS_PF':>13} {'mean_avg_trade%':>15} "
              f"{'#PF>1.1':>8} {'#avg>0':>8} {'#both':>6} {'compounded_OOS_ret%':>20}")
    print(header)
    for strategy_key, oos_rows in per_candidate_oos.items():
        if not oos_rows:
            print(f"{strategy_key:<14} {'no folds':>7}")
            continue
        n = len(oos_rows)
        pfs = [r["oos_pf"] for r in oos_rows]
        avg_trades = [r["oos_avg_trade_pct"] for r in oos_rows]
        returns = [r["oos_return_pct"] for r in oos_rows]
        median_pf = statistics.median(pfs)
        mean_avg_trade = statistics.mean(avg_trades)
        n_pf_ok = sum(1 for pf in pfs if pf > 1.1)
        n_avg_ok = sum(1 for a in avg_trades if a > 0)
        n_both = sum(1 for r in oos_rows if r["oos_pf"] > 1.1 and r["oos_avg_trade_pct"] > 0)
        compounded = 1.0
        for r in returns:
            compounded *= (1.0 + r / 100.0)
        compounded_pct = (compounded - 1.0) * 100.0
        print(f"{strategy_key:<14} {n:>7} {median_pf:>13.2f} {mean_avg_trade:>15.3f} "
              f"{n_pf_ok:>8} {n_avg_ok:>8} {n_both:>6} {compounded_pct:>20.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
