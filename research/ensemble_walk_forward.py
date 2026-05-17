"""
Ensemble walk-forward — pick best params per strategy per fold's train
window, then run each on the SAME OOS test window with split capital and
combine the equity curves.

Output structure mirrors `research.walk_forward` so the same researcher /
report tooling works on it. Combined per-fold metrics replace per-fold
single-strategy metrics; the per-member chosen params are stored under
`chosen_params` as a nested dict keyed by strategy name.

CLI:
    python -m research.ensemble_walk_forward \
        --member donchian-v2:config/sweep_donchian_v2.yaml:0.5 \
        --member carry-v2:config/sweep_carry_v2.yaml:0.5 \
        --start 2022-06-01 --end 2024-12-31

Each --member arg is "strategy_name:sweep_yaml_path:weight".
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from exchange.env import REPO_ROOT
from research.ensemble import run_ensemble
from research.scoring import fold_stability_score
from research.walk_forward import (
    best_combo_for_train,
    split_windows,
    sweep_grid,
)
from strategy.signals import StrategyParams

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsembleFoldResult:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_params: dict[str, dict[str, Any]]   # strategy -> combo
    member_test_sharpe: dict[str, float]
    member_test_return_pct: dict[str, float]
    member_trades: dict[str, int]
    combined_test_sharpe: float
    combined_test_return_pct: float
    combined_max_drawdown_pct: float


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _train_members(
    members_cfg: list[tuple[str, dict, float]],
    base: StrategyParams,
    train_start: datetime,
    train_end: datetime,
    symbol: str,
) -> list[tuple[str, StrategyParams, float, dict]] | None:
    """For each member, find its best train-window combo. Returns None if any
    member fails to find a valid combo (we can't ensemble a missing member).
    """
    out: list[tuple[str, StrategyParams, float, dict]] = []
    for name, sweep_cfg, weight in members_cfg:
        combos = list(sweep_grid(sweep_cfg["grid"]))
        min_trades = sweep_cfg.get("min_trades_train", 5)
        winner = best_combo_for_train(
            combos, base, train_start, train_end, min_trades, symbol, name,
        )
        if winner is None:
            log.warning("member %s found no valid combo on train window", name)
            return None
        combo, _train_r = winner
        from research.walk_forward import make_params
        params = make_params(combo, base)
        out.append((name, params, weight, combo))
    return out


def run_ensemble_walk_forward(
    members_cfg: list[tuple[str, dict, float]],
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    *,
    symbol: str = "BTC/USDT:USDT",
    total_cash: float = 1_000_000.0,
    base: StrategyParams | None = None,
) -> list[EnsembleFoldResult]:
    base = base or StrategyParams()
    folds = list(split_windows(start, end, train_days, test_days, step_days))
    log.info("ensemble walk-forward: %d folds × %d members",
             len(folds), len(members_cfg))

    results: list[EnsembleFoldResult] = []
    for i, (ts, te, vs, ve) in enumerate(folds):
        log.info("fold %d: train [%s → %s], test [%s → %s]",
                 i, ts.date(), te.date(), vs.date(), ve.date())

        trained = _train_members(members_cfg, base, ts, te, symbol)
        if trained is None:
            log.warning("fold %d: skipping — at least one member lacked a valid combo", i)
            continue

        # Run ensemble on the OOS window.
        ensemble_members = [(name, params, weight) for (name, params, weight, _combo) in trained]
        try:
            ens = run_ensemble(
                members=ensemble_members,
                symbol=symbol,
                start=vs,
                end=ve,
                total_cash=total_cash,
            )
        except Exception as e:
            log.warning("fold %d: ensemble OOS run failed: %s", i, e)
            continue

        chosen = {name: combo for (name, _p, _w, combo) in trained}
        per_sharpe: dict[str, float] = {}
        per_return: dict[str, float] = {}
        per_trades: dict[str, int] = {}
        for m in ens["members"]:
            r = m["result"]
            per_sharpe[m["name"]] = float(r.get("sharpe") or 0.0)
            per_return[m["name"]] = float(r.get("backtest_return_pct") or 0.0)
            per_trades[m["name"]] = int(r.get("trades") or 0)

        c = ens["combined"]
        results.append(EnsembleFoldResult(
            fold_index=i,
            train_start=ts.isoformat(),
            train_end=te.isoformat(),
            test_start=vs.isoformat(),
            test_end=ve.isoformat(),
            chosen_params=chosen,
            member_test_sharpe=per_sharpe,
            member_test_return_pct=per_return,
            member_trades=per_trades,
            combined_test_sharpe=float(c["sharpe"]),
            combined_test_return_pct=float(c["return_pct"]),
            combined_max_drawdown_pct=float(c["max_drawdown_pct"]),
        ))
    return results


def _summarise(folds: list[EnsembleFoldResult]) -> dict:
    if not folds:
        return {"passed": False, "reason": "no folds"}
    sharpes = [f.combined_test_sharpe for f in folds]
    returns = [f.combined_test_return_pct for f in folds]
    dds = [f.combined_max_drawdown_pct for f in folds]
    return {
        "folds": len(folds),
        "median_combined_sharpe": statistics.median(sharpes),
        "median_combined_return_pct": statistics.median(returns),
        "median_combined_max_dd_pct": statistics.median(dds),
        "fold_stability": fold_stability_score(sharpes),
        "min_combined_sharpe": min(sharpes),
        "max_combined_sharpe": max(sharpes),
    }


def write_reports(
    folds: list[EnsembleFoldResult],
    members_cfg: list[tuple[str, dict, float]],
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or (REPO_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    base = f"ensemble_walk_forward_{stamp}"

    summary = _summarise(folds)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "members": [
            {"strategy": n, "sweep": c, "weight": w}
            for (n, c, w) in members_cfg
        ],
        "summary": summary,
        "folds": [asdict(f) for f in folds],
    }
    json_path = out_dir / f"{base}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md_path = out_dir / f"{base}.md"
    md_path.write_text(_render_md(folds, summary, members_cfg, stamp))
    return {"json": json_path, "md": md_path}


def _render_md(
    folds: list[EnsembleFoldResult],
    summary: dict,
    members_cfg: list[tuple[str, dict, float]],
    stamp: str,
) -> str:
    out = [f"# Ensemble walk-forward — {stamp}", ""]
    out.append("**Members:**")
    for (n, _c, w) in members_cfg:
        out.append(f"- `{n}` @ weight {w:.0%}")
    out.append("")

    if not folds:
        out.append("No folds produced (likely all train windows had a member with no valid combo).")
        return "\n".join(out)

    out.extend([
        "## Summary",
        "",
        f"Folds: **{summary['folds']}**  ",
        f"Median combined Sharpe: **{summary['median_combined_sharpe']:+.2f}**  ",
        f"Median combined return: **{summary['median_combined_return_pct']:+.2f}%**  ",
        f"Median combined max DD: **{summary['median_combined_max_dd_pct']:.2f}%**  ",
        f"Fold stability: **{summary['fold_stability']:.0%}**  ",
        f"Sharpe range: [{summary['min_combined_sharpe']:+.2f}, {summary['max_combined_sharpe']:+.2f}]",
        "",
        "## Per-fold",
        "",
        "| # | test | combSharpe | combRet% | combDD% | members |",
        "|---|---|---:|---:|---:|---|",
    ])
    for f in folds:
        per = "  ".join(
            f"{n}: Sh {f.member_test_sharpe[n]:+.2f}/Ret {f.member_test_return_pct[n]:+.2f}%/T {f.member_trades[n]}"
            for n in f.member_test_sharpe
        )
        out.append(
            f"| {f.fold_index} | {f.test_start[:10]}→{f.test_end[:10]} | "
            f"{f.combined_test_sharpe:+.2f} | {f.combined_test_return_pct:+.2f} | "
            f"{f.combined_max_drawdown_pct:.2f} | {per} |"
        )
    return "\n".join(out) + "\n"


def _parse_member(spec: str) -> tuple[str, dict, float]:
    """Parse 'strategy:sweep_yaml:weight' into (name, sweep_cfg_dict, weight)."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--member expects 'strategy:sweep_yaml:weight', got: {spec}")
    name, sweep_path, weight_str = parts
    cfg = yaml.safe_load(Path(sweep_path).read_text())
    return (name.strip(), cfg, float(weight_str))


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Ensemble walk-forward harness.")
    p.add_argument("--member", action="append", required=True,
                   help="Repeatable: strategy:sweep_yaml:weight (e.g. donchian-v2:config/sweep_donchian_v2.yaml:0.5)")
    p.add_argument("--start", required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--end", required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=20)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--cash", type=float, default=1_000_000.0)
    args = p.parse_args()

    members_cfg = [_parse_member(s) for s in args.member]
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    folds = run_ensemble_walk_forward(
        members_cfg, start, end,
        args.train_days, args.test_days, args.step_days,
        symbol=args.symbol, total_cash=args.cash,
    )
    paths = write_reports(folds, members_cfg)
    print()
    print(f"JSON: {paths['json']}")
    print(f"MD:   {paths['md']}")
    summary = _summarise(folds)
    print()
    print(f"Folds: {summary.get('folds', 0)}")
    if folds:
        print(f"Median combined Sharpe: {summary['median_combined_sharpe']:+.2f}")
        print(f"Median combined return: {summary['median_combined_return_pct']:+.2f}%")
        print(f"Fold stability:         {summary['fold_stability']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
