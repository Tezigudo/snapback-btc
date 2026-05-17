"""
Out-of-sample validation harness.

Given a strategy + sweep grid + (in-sample window, OOS window):
  1. Run the full sweep on the in-sample window.
  2. Pick the best combo by tail_aware_score (same metric the
     walk-forward uses for per-fold selection).
  3. Apply that ONE combo to the OOS window.
  4. Report OOS metrics, compare to in-sample.

This is the cleanest "did we curve-fit?" check: the OOS window is
untouched during selection. If OOS metrics collapse vs in-sample,
the strategy was a curve-fit. If OOS metrics survive, real edge.

CLI:
    python -m research.oos_validate \
        --strategy carry-v4 --entry-tf 15m \
        --sweep config/sweep_carry_v4.yaml \
        --is-start 2022-06-01 --is-end 2024-12-31 \
        --oos-start 2025-01-01 --oos-end 2025-05-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from backtest import run_backtest
from exchange.env import REPO_ROOT
from research.scoring import tail_aware_score
from research.walk_forward import sweep_grid, make_params
from strategy.signals import StrategyParams

log = logging.getLogger(__name__)


def _best_combo_on_window(
    combos, base, start, end, symbol, strategy, entry_tf, min_trades,
):
    n = len(combos)
    best_combo = None
    best_result = None
    best_score = -1e18
    for combo in combos:
        params = make_params(combo, base)
        try:
            r = run_backtest(strategy, symbol, entry_tf, start, end,
                             quiet=True, params_override=params)
        except Exception as e:
            log.warning("combo %s failed: %s", combo, e)
            continue
        if r["trades"] < min_trades:
            continue
        score = tail_aware_score(
            after_funding_pct=r.get("after_funding_pct") or r["backtest_return_pct"],
            max_drawdown_pct=r["max_drawdown_pct"],
            sharpe=r["sharpe"],
            num_trials=n,
        )
        if score > best_score:
            best_score = score
            best_combo = combo
            best_result = r
    return best_combo, best_result, best_score


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--entry-tf", default="15m")
    p.add_argument("--sweep", required=True)
    p.add_argument("--is-start", required=True)
    p.add_argument("--is-end", required=True)
    p.add_argument("--oos-start", required=True)
    p.add_argument("--oos-end", required=True)
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    args = p.parse_args()

    sweep_cfg = yaml.safe_load(Path(args.sweep).read_text())
    grid = sweep_cfg["grid"]
    min_trades = sweep_cfg.get("min_trades_train", 5)
    combos = list(sweep_grid(grid))
    base = StrategyParams()

    is_start = datetime.fromisoformat(args.is_start).replace(tzinfo=timezone.utc)
    is_end = datetime.fromisoformat(args.is_end).replace(tzinfo=timezone.utc)
    oos_start = datetime.fromisoformat(args.oos_start).replace(tzinfo=timezone.utc)
    oos_end = datetime.fromisoformat(args.oos_end).replace(tzinfo=timezone.utc)

    log.info("In-sample window: %s → %s (%d combos)",
             is_start.date(), is_end.date(), len(combos))
    best_combo, is_result, is_score = _best_combo_on_window(
        combos, base, is_start, is_end, args.symbol, args.strategy,
        args.entry_tf, min_trades,
    )
    if best_combo is None:
        log.error("No combo passed min_trades=%d on in-sample window.", min_trades)
        return 1

    log.info("Best IS combo (score %.3f): %s", is_score, best_combo)
    log.info("In-sample result: return=%+.2f%% after_funding=%+.2f%% Sharpe=%.2f DD=%.2f%% trades=%d",
             is_result["backtest_return_pct"], is_result.get("after_funding_pct") or 0.0,
             is_result["sharpe"], is_result["max_drawdown_pct"], is_result["trades"])

    log.info("OOS window: %s → %s", oos_start.date(), oos_end.date())
    oos_params = make_params(best_combo, base)
    oos_result = run_backtest(
        args.strategy, args.symbol, args.entry_tf, oos_start, oos_end,
        quiet=True, params_override=oos_params,
    )
    log.info("OOS result: return=%+.2f%% after_funding=%+.2f%% Sharpe=%.2f DD=%.2f%% trades=%d",
             oos_result["backtest_return_pct"], oos_result.get("after_funding_pct") or 0.0,
             oos_result["sharpe"], oos_result["max_drawdown_pct"], oos_result["trades"])

    # Verdict
    is_ret = is_result.get("after_funding_pct") or is_result["backtest_return_pct"]
    oos_ret = oos_result.get("after_funding_pct") or oos_result["backtest_return_pct"]
    print()
    print(f"=== OOS verdict for {args.strategy} ===")
    print(f"  In-sample window  : {is_start.date()} → {is_end.date()}")
    print(f"  OOS window        : {oos_start.date()} → {oos_end.date()}")
    print(f"  Best combo        : {best_combo}")
    print()
    print(f"  IS  return after fund : {is_ret:+8.2f}%   Sharpe {is_result['sharpe']:+6.2f}   DD {is_result['max_drawdown_pct']:.2f}%   trades {is_result['trades']}")
    print(f"  OOS return after fund : {oos_ret:+8.2f}%   Sharpe {oos_result['sharpe']:+6.2f}   DD {oos_result['max_drawdown_pct']:.2f}%   trades {oos_result['trades']}")
    print()
    if oos_ret > 0:
        print("  ✅ OOS POSITIVE — edge survives unseen data; not pure curve-fit")
    else:
        print("  ❌ OOS NEGATIVE — strategy was a curve-fit; do NOT deploy")

    # Persist
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "reports" / f"oos_{args.strategy}_{stamp}.json"
    out.write_text(json.dumps({
        "strategy": args.strategy,
        "entry_tf": args.entry_tf,
        "is_window": [args.is_start, args.is_end],
        "oos_window": [args.oos_start, args.oos_end],
        "best_combo": best_combo,
        "in_sample_result": {k: v for k, v in is_result.items() if k != "equity_series" and k != "returns_series"},
        "oos_result": {k: v for k, v in oos_result.items() if k != "equity_series" and k != "returns_series"},
    }, indent=2, default=str))
    print(f"\n  JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
