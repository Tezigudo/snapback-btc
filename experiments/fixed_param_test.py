"""
Round 3: fixed-param walk-forward evaluation for the Supertrend family.

Prior rounds picked the best params PER FOLD via grid search on the train
window -- an optimistic upper bound a deployed bot can't realize (a bot ships
with ONE fixed param set, not a fresh re-tune every 6 months).

This script:
  1. For each candidate, picks ONE fixed param set = the MODAL (most
     frequent) `chosen_params` across the 8 folds in walkforward_results.csv
     (ties broken by better median OOS PF).
  2. Re-runs that fixed set on each of the 8 OOS test folds (no tuning).
  3. Runs a single continuous backtest over 2022-01-01 -> 2026-06-01 with the
     same fixed params.
  4. Reports aggregate stats and compares fixed-param compounded return vs
     the earlier per-fold-tuned compounded return.

Candidates: supertrend, st-adx, st-donchexit, st-adx-donchexit,
st-donchexit::long.

Run: .venv/bin/python experiments/fixed_param_test.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from experiments.walkforward import (  # noqa: E402
    _LONG_ONLY_BASE_STRATEGY,
    LEVERAGE,
    SYMBOL,
    TF,
    build_folds,
)

CONTINUOUS_START = date(2022, 1, 1)
CONTINUOUS_END = date(2026, 6, 1)

# Earlier per-fold-tuned compounded OOS returns (from walkforward.py runs),
# for the optimism-gap comparison.
TUNED_COMPOUNDED = {
    "st-donchexit": 28.31,
    "st-adx-donchexit": 10.88,
    "st-adx-donchexit::long": 75.93,  # not re-run here; provided for context
}


def _to_dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _resolve_strategy_key(strategy_key: str) -> str:
    return _LONG_ONLY_BASE_STRATEGY.get(strategy_key, strategy_key)


def _apply_combo(strategy_key: str, combo: dict) -> None:
    base_key = _resolve_strategy_key(strategy_key)
    cls = STRATEGIES[base_key]
    # `allow_shorts` is a sticky class attribute shared between a candidate
    # and its `::long` variant (same underlying class). Reset to the default
    # (both directions) unless this combo explicitly requests long-only --
    # otherwise a prior `::long` run leaves shorts disabled for everyone else.
    cls.allow_shorts = True
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


def _pick_modal_params(rows: list[dict]) -> dict:
    """Pick the modal chosen_params dict; ties broken by better median OOS PF."""
    import json

    cnt = Counter(r["chosen_params"] for r in rows)
    max_count = max(cnt.values())
    tied = [params for params, c in cnt.items() if c == max_count]

    if len(tied) == 1:
        winner = tied[0]
    else:
        best_med, winner = None, None
        for params in tied:
            pfs = [float(r["oos_pf"]) for r in rows if r["chosen_params"] == params]
            med = statistics.median(pfs)
            if best_med is None or med > best_med:
                best_med, winner = med, params

    return json.loads(winner), max_count, tied


def main() -> int:
    # --- Load prior walkforward results ---
    results_path = Path(__file__).resolve().parent / "walkforward_results.csv"
    rows = list(csv.DictReader(open(results_path)))
    by_cand: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cand[r["candidate"]].append(r)

    folds = build_folds()
    fold_test_ranges = [(te_s, te_e) for (_, _, te_s, te_e) in folds]

    candidates = ["supertrend", "st-adx", "st-donchexit", "st-adx-donchexit", "st-donchexit::long"]

    # --- Table A: chosen fixed param set per candidate ---
    chosen_params: dict[str, dict] = {}
    print("=== Table A: chosen FIXED param set per candidate ===")
    for cand in candidates:
        source_cand = "st-donchexit" if cand == "st-donchexit::long" else cand
        src_rows = by_cand[source_cand]
        params, count, tied = _pick_modal_params(src_rows)

        if cand == "st-donchexit::long":
            # Per spec: same modal set as st-donchexit, but force long-only.
            params = dict(params)
            params["_st_long_only"] = True

        chosen_params[cand] = params

        tie_note = ""
        if len(tied) > 1:
            tie_note = f" (tie among {len(tied)} sets at count={count}, broken by higher median OOS PF)"
        print(f"  {cand:<22} -> {params}  [modal count={count}/8]{tie_note}")
    print()

    # --- Per-fold fixed-param OOS runs (Table B inputs) ---
    print("=== Table B inputs: per-fold FIXED-param OOS runs ===")
    fixed_oos: dict[str, list[dict]] = {cand: [] for cand in candidates}
    new_csv_rows = []

    for cand in candidates:
        print(f"--- {cand}  fixed_params={chosen_params[cand]} ---")
        _apply_combo(cand, chosen_params[cand])
        for fold_idx, (te_s, te_e) in enumerate(fold_test_ranges):
            try:
                oos = _run(cand, te_s, te_e)
            except Exception as exc:  # pragma: no cover - data gaps etc.
                print(f"  fold {fold_idx}: OOS run failed: {exc}", file=sys.stderr)
                continue
            row = {
                "candidate": f"{cand}::fixed",
                "fold_idx": fold_idx,
                "train_start": "",
                "train_end": "",
                "test_start": te_s.isoformat(),
                "test_end": te_e.isoformat(),
                "chosen_params": __import__("json").dumps(chosen_params[cand]),
                "oos_pf": oos["profit_factor"],
                "oos_avg_trade_pct": oos["avg_trade_pct"],
                "oos_trades": oos["trades"],
                "oos_return_pct": oos["after_funding_pct"],
                "oos_maxdd_pct": oos["max_drawdown_pct"],
            }
            fixed_oos[cand].append(row)
            new_csv_rows.append(row)
            print(f"  fold {fold_idx} ({te_s}..{te_e}): "
                  f"PF={oos['profit_factor']:.2f} avg_trade={oos['avg_trade_pct']:+.3f}% "
                  f"trades={oos['trades']} return={oos['after_funding_pct']:+.2f}% "
                  f"maxdd={oos['max_drawdown_pct']:.2f}%")
        print()

    # --- Table B: aggregate per-candidate FIXED-param stats ---
    print("=== Table B: per-candidate FIXED-param aggregate over 8 folds ===")
    header = (f"{'candidate':<22} {'n_folds':>7} {'median_PF':>10} {'mean_avg_trade%':>16} "
              f"{'#PF>1.1':>8} {'#avg>0':>8} {'#both':>6} {'compounded_ret%':>16}")
    print(header)
    fixed_compounded: dict[str, float] = {}
    for cand in candidates:
        oos_rows = fixed_oos[cand]
        if not oos_rows:
            print(f"{cand:<22} {'no folds':>7}")
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
        fixed_compounded[cand] = compounded_pct
        print(f"{cand:<22} {n:>7} {median_pf:>10.2f} {mean_avg_trade:>16.3f} "
              f"{n_pf_ok:>8} {n_avg_ok:>8} {n_both:>6} {compounded_pct:>16.2f}")
    print()

    # --- Table C: continuous-run results ---
    print(f"=== Table C: single continuous run {CONTINUOUS_START} -> {CONTINUOUS_END}, fixed params ===")
    header_c = (f"{'candidate':<22} {'PF':>8} {'avg_trade%':>11} {'trades':>7} "
                f"{'total_return%':>14} {'maxDD%':>8} {'sharpe':>8}")
    print(header_c)
    continuous_results: dict[str, dict] = {}
    for cand in candidates:
        _apply_combo(cand, chosen_params[cand])
        try:
            res = _run(cand, CONTINUOUS_START, CONTINUOUS_END)
        except Exception as exc:  # pragma: no cover
            print(f"{cand:<22} run failed: {exc}", file=sys.stderr)
            continue
        continuous_results[cand] = res
        print(f"{cand:<22} {res['profit_factor']:>8.2f} {res['avg_trade_pct']:>11.3f} "
              f"{res['trades']:>7} {res['after_funding_pct']:>14.2f} "
              f"{res['max_drawdown_pct']:>8.2f} {res['sharpe']:>8.2f}")
    print()

    # --- Key comparison: fixed vs tuned compounded return ---
    print("=== Key comparison: FIXED-param compounded OOS return vs PER-FOLD-TUNED ===")
    for cand in ["st-donchexit", "st-adx-donchexit"]:
        tuned = TUNED_COMPOUNDED.get(cand)
        fixed = fixed_compounded.get(cand)
        if tuned is not None and fixed is not None:
            gap = tuned - fixed
            pct_retained = (fixed / tuned * 100.0) if tuned != 0 else float("nan")
            print(f"  {cand}: tuned={tuned:+.2f}%  fixed={fixed:+.2f}%  "
                  f"gap={gap:+.2f}pp  retained={pct_retained:.1f}% of tuned edge")
    print()

    # --- Honest verdict ---
    print("=== Honest verdict ===")
    any_pass = False
    for cand in candidates:
        oos_rows = fixed_oos[cand]
        if not oos_rows:
            continue
        n_both = sum(1 for r in oos_rows if r["oos_pf"] > 1.1 and r["oos_avg_trade_pct"] > 0)
        cont = continuous_results.get(cand)
        cont_positive = cont is not None and cont["after_funding_pct"] > 0
        passes = n_both >= 6 and cont_positive
        if passes:
            any_pass = True
        print(f"  {cand:<22} folds(PF>1.1 & avg>0)={n_both}/8  "
              f"continuous_return={'+' if cont and cont['after_funding_pct'] >= 0 else ''}"
              f"{cont['after_funding_pct']:.2f}%  -> {'PASS' if passes else 'FAIL'}")

    if any_pass:
        print("\n  >> At least one fixed-param candidate clears PF>1.1 AND positive "
              "avg-trade in >=6/8 folds AND a positive continuous-run return.")
    else:
        print("\n  >> NO fixed-param candidate clears PF>1.1 AND positive avg-trade in "
              ">=6/8 folds AND a positive continuous-run return. The per-fold-tuned "
              "numbers were an optimistic upper bound that does not survive freezing "
              "params for deployment.")
    print()

    # --- Append fixed-param rows to a new CSV (additive, never overwrite walkforward_results.csv) ---
    out_path = Path(__file__).resolve().parent / "fixed_param_results.csv"
    fieldnames = [
        "candidate", "fold_idx", "train_start", "train_end", "test_start", "test_end",
        "chosen_params", "oos_pf", "oos_avg_trade_pct", "oos_trades", "oos_return_pct", "oos_maxdd_pct",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_csv_rows)
    print(f"Wrote {len(new_csv_rows)} rows to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
