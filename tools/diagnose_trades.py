"""Diagnose a trade list: bucket by exit reason, compute fees, prove EV math.

Usage: python -m tools.diagnose_trades reports/trades_multifactor_2025.json

Exit-reason classification (heuristic, derived from PnL and hold time):
  - SL_HIT:        |pnl_pct| within 0.05 of -sl_pct (within tolerance — accounts for fill slippage)
  - TP_HIT:        |pnl_pct| within 0.05 of +tp_pct
  - TIME_STOP:     hold_hours >= max_hold_hours
  - TREND_FLIP:    otherwise (a discretionary close from the strategy logic)

Output: JSON + plaintext summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


def classify(pnl_pct: float, hold_hours: float, sl_pct: float, tp_pct: float,
             max_hold_hours: float, tol: float = 0.10) -> str:
    """Return SL_HIT / TP_HIT / TIME_STOP / TREND_FLIP."""
    if abs(pnl_pct - (-sl_pct * 100)) <= tol:
        return "SL_HIT"
    if abs(pnl_pct - (tp_pct * 100)) <= tol:
        return "TP_HIT"
    if hold_hours >= max_hold_hours - 1:
        return "TIME_STOP"
    return "TREND_FLIP"


def analyze(path: Path, sl_pct: float = 0.015, tp_pct: float = 0.03,
            max_hold_hours: float = 14 * 24, taker_fee_bps: float = 5.0,
            leverage: int = 20) -> dict:
    data = json.loads(path.read_text())
    trades = data["trades"]
    report_return = data.get("return", float("nan"))

    buckets: dict[str, list[dict]] = {"SL_HIT": [], "TP_HIT": [], "TIME_STOP": [], "TREND_FLIP": []}
    for t in trades:
        b = classify(t["pnl_pct"], t["hold_hours"], sl_pct, tp_pct, max_hold_hours)
        t["_exit_reason"] = b
        buckets[b].append(t)

    # NOTE: pnl_pct from backtesting.py is ALREADY net of commission.
    # Verified empirically: trade 1 raw price move 1.318%, reported pnl_pct
    # 1.217%, delta = 0.10% = 2 × 0.05% per-side commission.
    # We expose `commission_already_in_pnl_pct` as informational only.
    fee_pct_per_side_on_notional = (taker_fee_bps / 10000.0) * 100.0  # 0.05%
    fee_pct_round_trip_on_notional = 2.0 * fee_pct_per_side_on_notional  # 0.10%
    # Risk-based sizing: position notional = (risk_pct / sl_pct) × equity
    # Default 2% risk / 1.5% sl = 1.33× equity. So fees on equity ≈ 1.33 × 0.10% = 0.13%
    # But this is moot — backtesting.py already deducted it from pnl_pct.
    fee_pct_round_trip = fee_pct_round_trip_on_notional  # for display only

    # Per-bucket stats
    bucket_stats = {}
    for name, ts in buckets.items():
        if not ts:
            bucket_stats[name] = {"count": 0, "mean_pnl_pct": 0.0, "sum_pnl_pct": 0.0}
            continue
        pnls = [t["pnl_pct"] for t in ts]
        bucket_stats[name] = {
            "count": len(ts),
            "mean_pnl_pct": mean(pnls),
            "sum_pnl_pct": sum(pnls),
            "mean_hold_hours": mean(t["hold_hours"] for t in ts),
        }

    n = len(trades)
    all_pnls = [t["pnl_pct"] for t in trades]  # ALREADY net of fees
    net_return_sum = sum(all_pnls)
    net_mean_per_trade = mean(all_pnls)

    # Win rate vs R:R math (all on NET pnl_pct)
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    win_rate = len(wins) / n if n else 0
    avg_win = mean(wins) if wins else 0
    avg_loss = mean(losses) if losses else 0
    rr = abs(avg_win / avg_loss) if avg_loss else float("inf")
    # Break-even win rate for this R:R: p* = -avg_loss / (avg_win - avg_loss)
    breakeven_wr = (-avg_loss / (avg_win - avg_loss)) if (avg_win > avg_loss) else 1.0
    # Expected value per trade (NET, since pnl_pct is net)
    ev_per_trade = win_rate * avg_win + (1 - win_rate) * avg_loss
    # Margin over break-even
    margin_wr = win_rate - breakeven_wr

    return {
        "trades_count": n,
        "report_return_pct_compounded": report_return,
        "net_sum_pnl_pct": net_return_sum,
        "net_mean_per_trade_pct": net_mean_per_trade,
        "fee_assumptions": {
            "taker_fee_bps_per_side": taker_fee_bps,
            "fee_pct_round_trip_on_notional": fee_pct_round_trip,
            "note": "Fees are already deducted from pnl_pct by backtesting.py.",
        },
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "reward_risk_ratio": rr,
        "breakeven_win_rate": breakeven_wr,
        "margin_over_breakeven_wr": margin_wr,
        "ev_per_trade_net_pct": ev_per_trade,
        "ev_over_n_trades_net_pct": ev_per_trade * n,
        "buckets": bucket_stats,
        "trades": trades,  # with _exit_reason attached
    }


def print_summary(r: dict) -> None:
    print(f"=== Trade EV diagnosis ({r['trades_count']} trades) ===\n")
    print(f"Reported (compounded equity): {r['report_return_pct_compounded']:+.2f}%")
    print(f"Sum of pnl_pct (NET of fees): {r['net_sum_pnl_pct']:+.2f}%")
    print(f"Mean pnl per trade (NET):     {r['net_mean_per_trade_pct']:+.3f}%")
    print()
    print(f"Win rate:                     {r['win_rate']*100:.1f}%")
    print(f"Avg win:                      {r['avg_win_pct']:+.2f}%")
    print(f"Avg loss:                     {r['avg_loss_pct']:+.2f}%")
    print(f"Reward/Risk (|win|/|loss|):   {r['reward_risk_ratio']:.2f}")
    print(f"Break-even win rate (this RR):{r['breakeven_win_rate']*100:5.1f}%")
    print(f"Margin over break-even:       {r['margin_over_breakeven_wr']*100:+.1f} pp")
    print()
    print(f"EV/trade (NET):               {r['ev_per_trade_net_pct']:+.3f}%")
    print(f"EV over {r['trades_count']} trades (NET):     {r['ev_over_n_trades_net_pct']:+.2f}%")
    print()
    print("=== Per-bucket breakdown ===")
    print(f"{'Bucket':<14} {'Count':>6} {'MeanPnL%':>10} {'SumPnL%':>10}")
    for name, s in r["buckets"].items():
        if s["count"]:
            print(f"{name:<14} {s['count']:>6d} {s['mean_pnl_pct']:>+10.3f} {s['sum_pnl_pct']:>+10.2f}")
        else:
            print(f"{name:<14} {'0':>6}")
    print()
    print("KEY INSIGHT:")
    n = r["trades_count"]
    # 95% CI on win-rate: ±1.96 × sqrt(p(1-p)/n)
    import math
    p = r["win_rate"]
    se = math.sqrt(p * (1 - p) / n) if n else 0
    ci = 1.96 * se
    print(f"  Observed win rate: {p*100:.1f}% ± {ci*100:.1f}pp (95% CI, n={n})")
    print(f"  Break-even win rate: {r['breakeven_win_rate']*100:.1f}%")
    if r['breakeven_win_rate'] > p - ci and r['breakeven_win_rate'] < p + ci:
        print("  -> Break-even rate is INSIDE confidence interval = edge is statistically indistinguishable from zero.")
    elif p > r['breakeven_win_rate']:
        print("  -> Observed win rate exceeds break-even rate even at CI lower bound = real edge.")
    else:
        print("  -> Observed win rate below break-even rate even at CI upper bound = strategy loses.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--sl-pct", type=float, default=0.015)
    p.add_argument("--tp-pct", type=float, default=0.03)
    p.add_argument("--max-hold-hours", type=float, default=14 * 24)
    p.add_argument("--taker-fee-bps", type=float, default=5.0)
    p.add_argument("--leverage", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    r = analyze(args.path, sl_pct=args.sl_pct, tp_pct=args.tp_pct,
                max_hold_hours=args.max_hold_hours,
                taker_fee_bps=args.taker_fee_bps,
                leverage=args.leverage)
    print_summary(r)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2, default=str))
        print(f"\nFull report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
