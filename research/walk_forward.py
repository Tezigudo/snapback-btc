"""
Walk-forward + out-of-sample engine for snapback-btc.

Slides a (train, test) window across the requested period. On each train
window: sweep the param grid, pick the highest deflated-Sharpe combo that
meets the min-trades floor. On the matching test window: re-run that combo
and record OOS metrics. Aggregate over folds and write three artifacts:

  reports/walk_forward_<UTC ISO>.json   — canonical machine-readable record
  reports/walk_forward_<UTC ISO>.md     — human summary + researcher commentary
  reports/walk_forward_<UTC ISO>.html   — Plotly per-fold dashboards

The Researcher is pluggable (see research/agents/AGENT_ROLES.md). Default is
DeterministicResearcher — pure stats, no LLM, no API cost.

CLI:
    python -m research.walk_forward \
        --sweep config/sweep.yaml \
        --start 2025-01-01 --end 2025-12-31 \
        --train-days 90 --test-days 30 --step-days 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from backtest import run_backtest
from exchange.env import REPO_ROOT
from research.agents.base import FoldResult, Researcher
from research.agents.deterministic import DeterministicResearcher
from research.scoring import deflated_sharpe, fold_stability_score
from strategy.signals import StrategyParams

log = logging.getLogger(__name__)


# --- windowing ---------------------------------------------------------------
def split_windows(
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
) -> Iterator[tuple[datetime, datetime, datetime, datetime]]:
    """Yield (train_start, train_end, test_start, test_end) tuples.

    Non-overlapping train/test pairs. Train length is fixed (sliding window).
    Stops when the next window would extend past `end`.
    """
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, step_days must all be > 0")
    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    step_delta = timedelta(days=step_days)
    cur = start
    while cur + train_delta + test_delta <= end:
        train_start = cur
        train_end = cur + train_delta
        test_start = train_end
        test_end = train_end + test_delta
        yield train_start, train_end, test_start, test_end
        cur += step_delta


# --- sweep -------------------------------------------------------------------
def sweep_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Cartesian product of the param grid. Sorted keys = deterministic order."""
    if not grid:
        return iter([])
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    for combo_vals in itertools.product(*values):
        yield dict(zip(keys, combo_vals))


def make_params(combo: dict[str, Any], base: StrategyParams) -> StrategyParams:
    """Overlay sweep combo onto a baseline StrategyParams.

    If rsi_long_threshold is in the combo but rsi_short_threshold is not,
    mirror it to 100 - long so we always test symmetric thresholds.
    """
    overrides = dict(combo)
    if "rsi_long_threshold" in overrides and "rsi_short_threshold" not in overrides:
        overrides["rsi_short_threshold"] = 100.0 - float(overrides["rsi_long_threshold"])
    # Drop any keys not present on StrategyParams to avoid TypeError.
    valid = set(StrategyParams.__dataclass_fields__.keys())
    overrides = {k: v for k, v in overrides.items() if k in valid}
    return replace(base, **overrides)


def _evaluate_combo(
    combo: dict[str, Any],
    base: StrategyParams,
    start: datetime,
    end: datetime,
    symbol: str,
) -> dict | None:
    """Run a single backtest. Returns the result dict, or None on failure."""
    try:
        params = make_params(combo, base)
        return run_backtest(
            "snapback-v1", symbol, "15m", start, end,
            quiet=True, params_override=params,
        )
    except Exception as e:
        log.warning("combo %s failed: %s", combo, e)
        return None


def best_combo_for_train(
    combos: list[dict[str, Any]],
    base: StrategyParams,
    train_start: datetime,
    train_end: datetime,
    min_trades: int,
    symbol: str,
) -> tuple[dict[str, Any], dict] | None:
    """Train-side sweep. Return (combo, result) for the deflated-Sharpe winner."""
    num_combos = len(combos)
    best: tuple[dict[str, Any], dict] | None = None
    best_score = -1e18
    for combo in combos:
        r = _evaluate_combo(combo, base, train_start, train_end, symbol)
        if r is None or r["trades"] < min_trades:
            continue
        score = deflated_sharpe(r["sharpe"], num_combos)
        if score > best_score:
            best_score = score
            best = (combo, r)
    return best


# --- driver ------------------------------------------------------------------
def run_walk_forward(
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    grid: dict[str, list[Any]],
    *,
    min_trades_train: int = 10,
    symbol: str = "BTC/USDT:USDT",
    base: StrategyParams | None = None,
) -> list[FoldResult]:
    """Run all folds; return the FoldResult list (may be empty)."""
    base = base or StrategyParams.from_yaml()
    combos = list(sweep_grid(grid))
    if not combos:
        log.warning("empty grid; no combos to evaluate")
        return []
    folds_iter = list(split_windows(start, end, train_days, test_days, step_days))
    log.info(
        "walk-forward: %d combos × %d folds = %d train backtests + %d OOS evals",
        len(combos), len(folds_iter), len(combos) * len(folds_iter), len(folds_iter),
    )

    results: list[FoldResult] = []
    for i, (ts, te, vs, ve) in enumerate(folds_iter):
        log.info(
            "fold %d: train [%s → %s], test [%s → %s]",
            i, ts.date(), te.date(), vs.date(), ve.date(),
        )
        winner = best_combo_for_train(combos, base, ts, te, min_trades_train, symbol)
        if winner is None:
            log.warning(
                "fold %d: no combo met min_trades_train=%d on train window; skipping",
                i, min_trades_train,
            )
            continue
        combo, train_r = winner
        test_r = _evaluate_combo(combo, base, vs, ve, symbol)
        if test_r is None:
            log.warning("fold %d: OOS evaluation failed; skipping", i)
            continue
        test_after = test_r.get("after_funding_pct")
        if test_after is None:
            test_after = test_r["backtest_return_pct"]
        results.append(
            FoldResult(
                fold_index=i,
                train_start=ts.isoformat(),
                train_end=te.isoformat(),
                test_start=vs.isoformat(),
                test_end=ve.isoformat(),
                chosen_params=combo,
                train_sharpe=float(train_r.get("sharpe") or 0.0),
                test_sharpe=float(test_r.get("sharpe") or 0.0),
                test_return_pct=float(test_r.get("backtest_return_pct") or 0.0),
                test_after_funding_pct=float(test_after),
                trades=int(test_r.get("trades") or 0),
                max_drawdown_pct=float(test_r.get("max_drawdown_pct") or 0.0),
            )
        )
    return results


# --- reports -----------------------------------------------------------------
def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_reports(
    folds: list[FoldResult],
    grid: dict[str, list[Any]],
    sweep_cfg: dict,
    researcher: Researcher | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path | None]:
    out_dir = out_dir or (REPO_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    base_name = f"walk_forward_{stamp}"
    researcher = researcher or DeterministicResearcher()
    commentary = researcher.commentary(folds)
    next_ranges = researcher.next_sweep_ranges(folds)

    json_path = out_dir / f"{base_name}.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sweep_config": sweep_cfg,
                "grid": grid,
                "folds": [asdict(f) for f in folds],
                "researcher_commentary": commentary,
                "suggested_next_ranges": next_ranges,
                "promotion": _evaluate_promotion(folds, sweep_cfg.get("promotion", {})),
            },
            indent=2,
            default=str,
        )
    )

    md_path = out_dir / f"{base_name}.md"
    md_path.write_text(_render_markdown(folds, stamp, commentary, next_ranges, sweep_cfg))

    html_path: Path | None = None
    try:
        html_path = _write_html(folds, out_dir / f"{base_name}.html")
    except Exception as e:
        log.warning("HTML report failed: %s", e)

    return {"json": json_path, "md": md_path, "html": html_path}


def _evaluate_promotion(folds: list[FoldResult], cfg: dict) -> dict:
    if not folds:
        return {"passed": False, "reason": "no folds"}
    import statistics
    test_sharpes = [f.test_sharpe for f in folds]
    train_sharpes = [f.train_sharpe for f in folds]
    median_test = statistics.median(test_sharpes)
    stability = fold_stability_score(test_sharpes)
    drift_values = []
    for tr, te in zip(train_sharpes, test_sharpes):
        if abs(tr) > 1e-6:
            drift_values.append((tr - te) / abs(tr) * 100.0)
    median_drift = statistics.median(drift_values) if drift_values else 0.0

    thresholds = {
        "min_median_test_sharpe": cfg.get("min_median_test_sharpe", 0.5),
        "min_fold_stability": cfg.get("min_fold_stability", 0.5),
        "max_train_test_drift_pct": cfg.get("max_train_test_drift_pct", 50.0),
    }
    checks = {
        "median_test_sharpe": median_test >= thresholds["min_median_test_sharpe"],
        "fold_stability": stability >= thresholds["min_fold_stability"],
        "train_test_drift": median_drift <= thresholds["max_train_test_drift_pct"],
    }
    return {
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "measured": {
            "median_test_sharpe": median_test,
            "fold_stability": stability,
            "median_train_test_drift_pct": median_drift,
        },
        "checks": checks,
    }


def _render_markdown(
    folds: list[FoldResult],
    stamp: str,
    commentary: str,
    next_ranges: dict[str, list[Any]],
    sweep_cfg: dict,
) -> str:
    out = [f"# Walk-forward report — {stamp}", ""]
    if not folds:
        out.append("No folds produced. Check that the requested window covers enough days for at least one (train + test) split.")
        return "\n".join(out)

    out.extend(
        [
            f"Folds: **{len(folds)}**",
            "",
            "## Per-fold table",
            "",
            "| # | train | test | params | trainSh | testSh | testRet% | trades | DD% |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for f in folds:
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(f.chosen_params.items()))
        out.append(
            f"| {f.fold_index} | {f.train_start[:10]}→{f.train_end[:10]} | "
            f"{f.test_start[:10]}→{f.test_end[:10]} | "
            f"`{params_str}` | {f.train_sharpe:+.2f} | {f.test_sharpe:+.2f} | "
            f"{f.test_after_funding_pct:+.2f} | {f.trades} | {f.max_drawdown_pct:.2f} |"
        )

    out.extend(["", "## Researcher commentary", "", "```", commentary, "```", ""])

    promo = _evaluate_promotion(folds, sweep_cfg.get("promotion", {}))
    out.extend([
        "## Promotion gate",
        "",
        f"**Decision:** {'✅ PASS' if promo['passed'] else '❌ FAIL'} — these defaults are *not* applied to `config/params.yaml` automatically.",
        "",
        "| check | measured | threshold | pass |",
        "|---|---:|---:|:---:|",
    ])
    for key, ok in promo["checks"].items():
        thr_key = {
            "median_test_sharpe": "min_median_test_sharpe",
            "fold_stability": "min_fold_stability",
            "train_test_drift": "max_train_test_drift_pct",
        }[key]
        meas_key = {
            "median_test_sharpe": "median_test_sharpe",
            "fold_stability": "fold_stability",
            "train_test_drift": "median_train_test_drift_pct",
        }[key]
        out.append(
            f"| {key} | {promo['measured'][meas_key]:+.2f} | "
            f"{promo['thresholds'][thr_key]} | {'✅' if ok else '❌'} |"
        )

    if next_ranges:
        out.extend(["", "## Suggested next sweep (top-2 winners per param)", "", "```yaml"])
        for k in sorted(next_ranges):
            out.append(f"{k}: {next_ranges[k]}")
        out.append("```")

    return "\n".join(out) + "\n"


def _write_html(folds: list[FoldResult], path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x = [f.fold_index for f in folds]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "OOS Sharpe per fold",
            "OOS return % per fold (after funding)",
            "Trade count per fold",
            "Max drawdown % per fold",
        ),
    )
    fig.add_trace(go.Bar(x=x, y=[f.test_sharpe for f in folds]), 1, 1)
    fig.add_trace(go.Bar(x=x, y=[f.test_after_funding_pct for f in folds]), 1, 2)
    fig.add_trace(go.Bar(x=x, y=[f.trades for f in folds]), 2, 1)
    fig.add_trace(go.Bar(x=x, y=[f.max_drawdown_pct for f in folds]), 2, 2)
    fig.update_layout(
        title=f"snapback-btc walk-forward: {len(folds)} folds",
        showlegend=False,
        height=720,
    )
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


# --- CLI ---------------------------------------------------------------------
def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Walk-forward + OOS evaluation for snapback-btc",
    )
    p.add_argument("--sweep", default=str(REPO_ROOT / "config" / "sweep.yaml"))
    p.add_argument("--start", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--train-days", type=int)
    p.add_argument("--test-days", type=int)
    p.add_argument("--step-days", type=int)
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    args = p.parse_args()

    sweep_cfg = yaml.safe_load(Path(args.sweep).read_text())
    train_days = args.train_days or sweep_cfg.get("train_days", 90)
    test_days = args.test_days or sweep_cfg.get("test_days", 30)
    step_days = args.step_days or sweep_cfg.get("step_days", 30)
    min_trades = sweep_cfg.get("min_trades_train", 10)
    grid = sweep_cfg["grid"]

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    folds = run_walk_forward(
        start, end, train_days, test_days, step_days, grid,
        min_trades_train=min_trades, symbol=args.symbol,
    )
    paths = write_reports(folds, grid, sweep_cfg)
    print()
    print(f"JSON: {paths['json']}")
    print(f"MD:   {paths['md']}")
    print(f"HTML: {paths['html']}")
    print()
    print(DeterministicResearcher().commentary(folds))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
