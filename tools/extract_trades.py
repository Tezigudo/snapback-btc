"""Run a backtest with given params, dump trade list to JSON in the same
shape as reports/trades_multifactor_2025.json so the diagnose/chart tools
can consume it.

Usage:
  python -m tools.extract_trades --strategy multifactor-v2 \
      --start 2025-01-01 --end 2025-06-01 --timeframe 15m \
      --rsi-long 30 --rsi-short 70 --vol-mult 2.0 --sl-pct 0.015 \
      --max-hold-bars 1344 --trend-ema 200 \
      --out reports/trades_multifactor_v2_2025.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from strategy.signals import StrategyParams


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--cash", type=float, default=1_000_000.0)  # matches SNAPBACK_DEFAULT_CASH
    p.add_argument("--rsi-long", type=float, default=30.0)
    p.add_argument("--rsi-short", type=float, default=70.0)
    p.add_argument("--vol-mult", type=float, default=2.0)
    p.add_argument("--sl-pct", type=float, default=0.015)
    p.add_argument("--tp-pct", type=float, default=0.03)
    p.add_argument("--trend-ema", type=int, default=200)
    p.add_argument("--max-hold-bars", type=int, default=1344)
    p.add_argument("--trail-act-atr", type=float, default=1.0)
    p.add_argument("--trail-mult-atr", type=float, default=2.0)
    p.add_argument("--require-trend", action="store_true", default=True)
    p.add_argument("--require-funding-not-extreme", action="store_true", default=True)
    p.add_argument("--require-candlestick", action="store_true", default=False)
    p.add_argument("--require-macd", action="store_true", default=False)
    p.add_argument("--confirmations-required", type=int, default=None,
                   help="v2 only: 0/1/2/3 TA confirmations required")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    base = StrategyParams.from_yaml()
    overrides = {
        "rsi_long_threshold": args.rsi_long,
        "rsi_short_threshold": args.rsi_short,
        "volume_multiple": args.vol_mult,
        "sl_pct": args.sl_pct,
        "tp_pct": args.tp_pct,
        "mf_trend_ema_period": args.trend_ema,
        "max_hold_bars": args.max_hold_bars,
        "require_trend": args.require_trend,
        "require_funding_not_extreme": args.require_funding_not_extreme,
        "require_candlestick": args.require_candlestick,
        "require_macd": args.require_macd,
        "trail_activate_atr": args.trail_act_atr,
        "trail_atr_multiple": args.trail_mult_atr,
    }
    if args.confirmations_required is not None:
        overrides["confirmations_required"] = args.confirmations_required
    params = dataclasses.replace(base, **overrides)

    # Hack: run_backtest doesn't directly return trades_df. We need to call it
    # and re-extract from STRATEGIES dict. Easier path: monkey-patch by running
    # a Backtest directly. We'll add a tiny shim: import Backtest plumbing.
    from backtesting import Backtest

    from backtest import (
        COMMISSION_PER_SIDE,
        STRATEGIES,
        _apply_params_to_class,
        _prepare_snapback_data,
    )

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    cls = STRATEGIES[args.strategy]
    _apply_params_to_class(cls, params)
    cls.leverage = 20  # type: ignore[attr-defined]

    donchian_entry = getattr(params, "donchian_period_entry",
                             getattr(cls, "donchian_period_entry", 20))
    donchian_exit = getattr(params, "donchian_period_exit",
                            getattr(cls, "donchian_period_exit", 10))

    data, _ = _prepare_snapback_data(
        "BTC/USDT:USDT", start, end, params,
        donchian_entry=donchian_entry, donchian_exit=donchian_exit,
    )
    margin = 1.0 / 20.0  # match run_backtest's leverage handling
    bt = Backtest(data, cls, cash=args.cash, commission=COMMISSION_PER_SIDE,
                  margin=margin, trade_on_close=False, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run()
    trades_df = stats._trades

    trades = []
    starting_cash = args.cash
    equity = starting_cash
    for n, row in enumerate(trades_df.itertuples(index=False), start=1):
        side = "LONG" if row.Size > 0 else "SHORT"
        pnl_usd = float(row.PnL)
        equity_before = equity
        equity_after = equity + pnl_usd
        equity = equity_after
        # equity impact in % is the leveraged version of the trade
        equity_impact_pct = (pnl_usd / equity_before) * 100 if equity_before > 0 else 0.0
        trades.append({
            "n": n,
            "entry": row.EntryTime.strftime("%Y-%m-%d %H:%M"),
            "exit": row.ExitTime.strftime("%Y-%m-%d %H:%M"),
            "side": side,
            "entry_price": float(row.EntryPrice),
            "exit_price": float(row.ExitPrice),
            "size_btc": float(row.Size),
            "pnl_pct": float(row.ReturnPct * 100),       # PRICE MOVE %, NOT leveraged
            "pnl_usd": pnl_usd,                            # actual $ P&L
            "equity_impact_pct": equity_impact_pct,        # leveraged % on equity at time of trade
            "equity_after_usd": equity_after,
            "hold_hours": (row.ExitTime - row.EntryTime).total_seconds() / 3600.0,
        })
    out = {
        "strategy": args.strategy,
        "params": dataclasses.asdict(params),
        "starting_cash": starting_cash,
        "ending_equity": equity,
        "return": float(stats["Return [%]"]),
        "win_rate_pct": float(stats.get("Win Rate [%]", 0.0)),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]", 0.0)),
        "trades": trades,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {len(trades)} trades, return={out['return']:+.2f}%, "
          f"wr={out['win_rate_pct']:.1f}%, dd={out['max_drawdown_pct']:+.2f}%")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
