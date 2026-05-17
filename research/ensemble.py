"""
Ensemble harness — split capital across N strategies, merge equity curves.

Run two (or more) strategies on the SAME date window, each with its own
slice of total capital, then sum the per-bar equity series to get a
combined equity curve. Compute combined metrics (Sharpe, max DD, return)
from that merged curve.

Why post-hoc merging rather than running both inside one Backtest?
backtesting.py's Backtest can only host one Strategy at a time, and
shared-equity races between two co-resident strategies would be a
nightmare to model correctly (margin contention, order priority). Running
them independently on a copy of the data with split capital is the same
math when neither runs out of margin, and the v1/v2 risk per trade is
~2% per side so collisions don't materially over-leverage.

Caveat: this assumes the two strategies are uncorrelated enough that
combining them produces a smoother equity curve. If they always trade in
the same direction, the diversification benefit collapses to zero.
Donchian is trend-following; carry is funding-direction-driven and price-
neutral. The hypothesis is they're close to uncorrelated.

Usage from CLI / scripts:

    from research.ensemble import run_ensemble
    result = run_ensemble(
        members=[
            ("donchian-v2", params_donchian, 0.5),  # 50% weight
            ("carry-v2",    params_carry,    0.5),  # 50% weight
        ],
        symbol="BTC/USDT:USDT",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        total_cash=1_000_000.0,
    )
    print(result["combined"])
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from backtest import run_backtest
from strategy.signals import StrategyParams


@dataclass(frozen=True)
class EnsembleMember:
    name: str
    params: StrategyParams
    weight: float  # 0..1; weights across an ensemble should sum to ~1.0


def _normalise_weights(members: list[tuple[str, StrategyParams, float]]) -> list[EnsembleMember]:
    total = sum(m[2] for m in members)
    if total <= 0:
        raise ValueError("ensemble weights sum to zero")
    return [EnsembleMember(name=n, params=p, weight=w / total) for (n, p, w) in members]


def _bar_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def _combined_metrics(combined_equity: pd.Series, bars_per_year: int = 365 * 24 * 4) -> dict:
    """Sharpe and max DD on the merged equity curve."""
    rets = _bar_returns(combined_equity)
    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mean / std) * np.sqrt(bars_per_year) if std > 0 else 0.0

    running_max = combined_equity.cummax()
    dd = (combined_equity - running_max) / running_max
    max_dd_pct = float(dd.min() * 100.0) if not dd.empty else 0.0

    ret_pct = float(combined_equity.iloc[-1] / combined_equity.iloc[0] - 1.0) * 100.0
    return {
        "return_pct": ret_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
    }


def run_ensemble(
    members: list[tuple[str, StrategyParams, float]],
    symbol: str,
    start: datetime,
    end: datetime,
    total_cash: float = 1_000_000.0,
    timeframe: str = "15m",
    leverage: int | None = None,
    quiet: bool = True,
) -> dict:
    """Run each member on its split of total_cash, merge equity curves.

    Returns a dict with per-member backtest results and a `combined` section
    summarising the merged equity curve.
    """
    norm = _normalise_weights(members)
    per_member: list[dict] = []
    eq_curves: list[pd.Series] = []

    for m in norm:
        member_cash = total_cash * m.weight
        result = run_backtest(
            strategy_name=m.name,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            cash=member_cash,
            leverage=leverage,
            quiet=quiet,
            params_override=m.params,
            return_equity=True,
        )
        if "equity_series" not in result:
            raise RuntimeError(f"{m.name} did not return an equity_series")
        # Dollar equity scaled to the member's actual cash (run_backtest may
        # bump small cash to SNAPBACK_DEFAULT_CASH internally; rescale here).
        eq = result["returns_series"] * member_cash
        per_member.append({"name": m.name, "weight": m.weight, "result": result})
        eq_curves.append(eq)

    # Align all curves on a common 15m index (they should already match).
    combined_idx = eq_curves[0].index
    for s in eq_curves[1:]:
        combined_idx = combined_idx.intersection(s.index)
    combined = sum(s.reindex(combined_idx).ffill() for s in eq_curves)
    if not isinstance(combined, pd.Series):
        combined = pd.Series(combined, index=combined_idx)

    metrics = _combined_metrics(combined)
    return {
        "members": per_member,
        "combined": {
            **metrics,
            "total_cash": total_cash,
            "bars": len(combined),
            "start": combined.index[0],
            "end": combined.index[-1],
        },
        "combined_equity": combined,
    }


def _main() -> int:
    """CLI: run a default Donchian+Carry ensemble on a window."""
    import argparse
    from datetime import datetime, timezone

    p = argparse.ArgumentParser(description="Run an ensemble backtest.")
    p.add_argument("--start", required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--end", required=True, help="YYYY-MM-DD UTC")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--cash", type=float, default=1_000_000.0)
    args = p.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    members = [
        ("donchian-v2", StrategyParams(), 0.5),
        ("carry-v2",    StrategyParams(), 0.5),
    ]
    res = run_ensemble(members, args.symbol, start, end, total_cash=args.cash)
    c = res["combined"]
    print()
    print(f"=== ENSEMBLE donchian-v2 + carry-v2 | {c['start']} → {c['end']} ===")
    print(f"  total cash      : ${c['total_cash']:,.0f}")
    print(f"  combined return : {c['return_pct']:+.2f}%")
    print(f"  combined Sharpe : {c['sharpe']:.2f}")
    print(f"  combined max DD : {c['max_drawdown_pct']:.2f}%")
    print()
    for m in res["members"]:
        r = m["result"]
        print(f"  - {m['name']:14s} weight {m['weight']:.0%}  "
              f"return {r['backtest_return_pct']:+.2f}%  "
              f"Sharpe {r['sharpe']:.2f}  "
              f"trades {r['trades']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
