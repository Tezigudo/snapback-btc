"""
Classify trade exits to understand low win rates.

User question: "why is win rate so low — is the algorithm wrong?"
Answer (per math): no — at 2:1 R:R, break-even win rate is 33%, not 50%.
But the gap between expected -EV (at 25% win rate) and actual ~+0% net for
multifactor-v1 means something OTHER than TP/SL is closing trades. Measure it.

For each trade we bucket the exit as TP / SL / time-stop / other (trend-break),
then tabulate count, win rate, mean PnL%, total PnL per bucket.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2026, 5, 18, tzinfo=UTC)
SYMBOL = "BTC/USDT:USDT"
CASH = 10_000.0
LEVERAGE = 20

# Exit classification tolerance (so a 1.49% move classifies as the same as
# 1.5% TP — backtesting.py uses bar close, not exact tick).
TOLERANCE = 0.0015  # 15 bps slack


def reset_classes() -> None:
    import importlib

    import strategy.signals_multifactor_v2 as v2
    import strategy.signals_multifactor_v3 as v3
    importlib.reload(v2)
    importlib.reload(v3)
    STRATEGIES["multifactor-v2-loose"] = v2.DayTradeMultiFactorBTCv2Loose
    STRATEGIES["multifactor-v2-strict"] = v2.DayTradeMultiFactorBTCv2Strict
    STRATEGIES["multifactor-v3"] = v3.DayTradeMultiFactorBTCv3
    STRATEGIES["v3-dist-ema-only"] = v3.V3DistEmaOnly
    STRATEGIES["v3-vol-regime-only"] = v3.V3VolRegimeOnly
    STRATEGIES["v3-atr-stops-only"] = v3.V3AtrStopsOnly
    STRATEGIES["v3-all"] = v3.V3All


def classify_exit(t, sl_pct: float, tp_pct: float, max_hold_bars: int,
                  bar_minutes: int = 15) -> str:
    entry = float(t["EntryPrice"])
    exit_ = float(t["ExitPrice"])
    size = float(t["Size"])
    side = "long" if size > 0 else "short"

    move = (exit_ / entry) - 1.0  # signed

    # Hold duration in bars
    held_bars = 0
    try:
        held = pd.to_datetime(t["ExitTime"]) - pd.to_datetime(t["EntryTime"])
        held_bars = int(held.total_seconds() / 60 / bar_minutes)
    except Exception:
        pass

    if side == "long":
        if move >= tp_pct - TOLERANCE:
            return "TP"
        if move <= -sl_pct + TOLERANCE:
            return "SL"
    else:
        if move <= -tp_pct + TOLERANCE:
            return "TP"
        if move >= sl_pct - TOLERANCE:
            return "SL"

    if held_bars >= max_hold_bars - 1:
        return "time-stop"
    return "trend-break"


def analyze(strategy: str, risk_pct: float = 1.0) -> None:
    print(f"\n{'='*70}")
    print(f"  {strategy} @ risk_pct={risk_pct}%  ({START.date()} → {END.date()})")
    print('='*70)

    reset_classes()
    base = StrategyParams.from_yaml()
    params = dataclasses.replace(base, risk_per_trade_pct=risk_pct, leverage=LEVERAGE)

    result = run_backtest(
        strategy_name=strategy, symbol=SYMBOL, timeframe="15m",
        start=START, end=END, cash=CASH, leverage=LEVERAGE,
        quiet=True, params_override=params, return_trades=True,
    )
    trades: pd.DataFrame = result["trades_df"]
    if trades is None or trades.empty:
        print("  (no trades)")
        return

    sl_pct = params.sl_pct
    tp_pct = params.tp_pct
    max_hold_bars = getattr(STRATEGIES[strategy], "max_hold_bars",
                            params.max_hold_bars if hasattr(params, "max_hold_bars") else 96)

    print(f"  Configured: sl_pct={sl_pct*100:.2f}%, tp_pct={tp_pct*100:.2f}%, "
          f"max_hold={max_hold_bars} bars ({max_hold_bars*15/60:.1f}h)")
    print(f"  R:R ratio: {tp_pct/sl_pct:.2f}:1   "
          f"break-even win rate: {sl_pct/(sl_pct+tp_pct)*100:.1f}%")
    print()

    trades = trades.copy()
    trades["exit_reason"] = trades.apply(
        lambda t: classify_exit(t, sl_pct, tp_pct, max_hold_bars), axis=1)
    trades["return_pct"] = (trades["ReturnPct"].astype(float)) * 100

    print(f"  Total trades: {len(trades)}")
    print(f"  Total PnL:    ${float(trades['PnL'].sum()):+,.2f}")
    print(f"  Net return:   {result['after_funding_pct']:+.2f}% (post-funding)")
    print(f"  Win rate:     {(trades['PnL'] > 0).mean() * 100:.1f}%")
    print()

    by = trades.groupby("exit_reason").agg(
        count=("PnL", "size"),
        wins=("PnL", lambda s: (s > 0).sum()),
        win_rate=("PnL", lambda s: (s > 0).mean() * 100),
        mean_ret_pct=("return_pct", "mean"),
        total_pnl=("PnL", "sum"),
    ).sort_values("count", ascending=False)

    print(f"  {'Exit':<14} {'Count':>6} {'% trades':>9} {'Wins':>6} "
          f"{'Win rate':>9} {'Mean ret':>11} {'Total PnL':>15}")
    print(f"  {'-'*14} {'-'*6} {'-'*9} {'-'*6} {'-'*9} {'-'*11} {'-'*15}")
    n = len(trades)
    for reason, row in by.iterrows():
        print(f"  {reason:<14} {int(row['count']):>6} "
              f"{row['count']/n*100:>8.1f}% "
              f"{int(row['wins']):>6} "
              f"{row['win_rate']:>8.1f}% "
              f"{row['mean_ret_pct']:>+10.3f}% "
              f"${row['total_pnl']:>+14,.2f}")


def main() -> None:
    for s in ["multifactor-v1", "multifactor-v2-strict", "v3-all", "v3-vol-regime-only"]:
        analyze(s, risk_pct=1.0)


if __name__ == "__main__":
    main()
