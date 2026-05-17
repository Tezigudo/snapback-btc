"""
Backtest harness with realistic friction modeling.

This module is the *research* backtester (Phase 1). It uses `backtesting.py`
with a commission proxy that bundles taker fee + slippage, and computes
funding-rate cost post-hoc against the equity curve so the headline number
reflects everything a live perp position would actually pay.

For P1 the only strategy registered is `buy-and-hold` — the sanity benchmark.
P2 adds the real strategy (RSI(2) + EMA(200) + volume + funding).

CLI:
    python backtest.py --strategy buy-and-hold --tf 1h --start 2024-01-01 --end 2024-04-01
    python backtest.py --strategy buy-and-hold --tf 15m --days 90

Honest reporting:
    - "Naive B&H"   : raw price change, zero friction
    - "Backtest"    : after fees + slippage
    - "After funding": also after funding payments
    - Friction drag : the difference, in percentage points

If "After funding" ever beats "Naive B&H" you have a bug.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from backtesting import Strategy
from backtesting.lib import FractionalBacktest

from exchange.data import load_funding, load_klines

# BTC trades at $40k+; with $10k cash we couldn't even afford a whole coin, so
# `Backtest` rejects orders. `FractionalBacktest` lets the broker hold
# fractional units (μBTC etc.), matching real futures sizing.

log = logging.getLogger(__name__)

# --- Friction model ----------------------------------------------------------
# Binance Futures taker fee = 0.04% per side. We bundle a 1bp slippage proxy
# into the same `commission` parameter since backtesting.py has no native
# slippage knob. Total per-side: 0.05%.
TAKER_FEE = 0.0004
SLIPPAGE_PROXY = 0.0001
COMMISSION_PER_SIDE = TAKER_FEE + SLIPPAGE_PROXY


# --- Strategies --------------------------------------------------------------
class BuyAndHold(Strategy):
    """Sanity benchmark — buy on first bar, hold forever.

    backtesting.py 0.6.x defaults to 100% equity sizing which cannot satisfy
    commission. Use 0.95 to leave a buffer. Latch on _opened so the order is
    submitted exactly once, even if the first fill is delayed by `trade_on_close=False`.
    """

    def init(self) -> None:
        self._opened = False

    def next(self) -> None:
        if not self._opened:
            # size=1.0 fails the broker margin check after commission; 0.999
            # leaves a 0.1% cash buffer that's effectively zero capital drag.
            self.buy(size=0.999)
            self._opened = True


STRATEGIES: dict[str, type[Strategy]] = {
    "buy-and-hold": BuyAndHold,
}


# --- Funding accounting ------------------------------------------------------
def funding_cost_for_long_btc(
    data: pd.DataFrame,
    funding: pd.DataFrame,
    initial_cash: float,
    commission: float,
) -> tuple[float, int]:
    """
    Sum funding payments for a long buy-and-hold position over `data`'s span.

    For a perp long, you PAY when funding_rate > 0 and RECEIVE when < 0.
    Total cost = sum over each funding event of (btc_position × price_at_event × rate).

    Returns (total_usdt_paid, n_funding_events). Positive = net cost.
    """
    if funding.empty or data.empty:
        return 0.0, 0

    first_open = float(data["Open"].iloc[0])
    btc_position = initial_cash * (1.0 - commission) / first_open

    span = funding.loc[
        (funding.index >= data.index[0]) & (funding.index <= data.index[-1])
    ]
    if span.empty:
        return 0.0, 0

    prices = data["Close"].reindex(span.index, method="ffill")
    notional = btc_position * prices
    paid = (notional * span["funding_rate"]).sum()
    return float(paid), len(span)


# --- Runner ------------------------------------------------------------------
def _prepare_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    days_back = max((end - start).days + 2, 2)
    raw = load_klines(symbol=symbol, timeframe=timeframe, days_back=days_back, end=end)
    df = raw.loc[start:end].copy()
    if df.empty:
        raise RuntimeError(
            f"No klines for {symbol} {timeframe} in {start.date()} → {end.date()}"
        )
    df.columns = [c.capitalize() for c in df.columns]
    # backtesting.py rejects tz-aware indexes in some versions; normalise to naive UTC.
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def run_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    cash: float = 10_000.0,
    quiet: bool = False,
) -> dict:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy_name}")

    data = _prepare_data(symbol, timeframe, start, end)
    funding = load_funding(symbol=symbol, days_back=max((end - start).days + 2, 2), end=end)
    funding_in_span = funding.loc[start:end]
    if funding_in_span.index.tz is not None:
        funding_in_span = funding_in_span.copy()
        funding_in_span.index = funding_in_span.index.tz_convert("UTC").tz_localize(None)

    bt = FractionalBacktest(
        data,
        STRATEGIES[strategy_name],
        cash=cash,
        commission=COMMISSION_PER_SIDE,
        trade_on_close=False,
        exclusive_orders=True,
        fractional_unit=1e-6,  # 1 satoshi-equivalent precision
        finalize_trades=True,  # mark-to-market any still-open positions at the last bar
    )
    stats = bt.run()

    naive_return_pct = (float(data["Close"].iloc[-1]) / float(data["Open"].iloc[0]) - 1.0) * 100.0
    bt_return_pct = float(stats["Return [%]"])

    # Funding only modeled accurately for buy-and-hold long here. P2 will track
    # position direction over time for the real strategy.
    funding_cost_usdt: float | None = None
    funding_events = 0
    after_funding_pct: float | None = None
    if strategy_name == "buy-and-hold":
        funding_cost_usdt, funding_events = funding_cost_for_long_btc(
            data, funding_in_span, initial_cash=cash, commission=COMMISSION_PER_SIDE
        )
        final_equity = cash * (1.0 + bt_return_pct / 100.0) - funding_cost_usdt
        after_funding_pct = (final_equity / cash - 1.0) * 100.0

    result = {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "start": data.index[0],
        "end": data.index[-1],
        "bars": len(data),
        "trades": int(stats["# Trades"]),
        "naive_return_pct": naive_return_pct,
        "backtest_return_pct": bt_return_pct,
        "after_funding_pct": after_funding_pct,
        "funding_cost_usdt": funding_cost_usdt,
        "funding_events": funding_events,
        "sharpe": float(stats.get("Sharpe Ratio") or 0.0),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]") or 0.0),
        "profit_factor": _safe_pf(stats),
        "win_rate_pct": float(stats.get("Win Rate [%]") or 0.0),
        "commission_per_side": COMMISSION_PER_SIDE,
    }

    if not quiet:
        _print_result(result)
    return result


def _safe_pf(stats) -> float:
    pf = stats.get("Profit Factor")
    if pf is None or pd.isna(pf):
        return 0.0
    return float(pf)


def _print_result(r: dict) -> None:
    print()
    print(f"=== {r['strategy']} | {r['symbol']} {r['timeframe']} ===")
    print(f"  period          : {r['start']} → {r['end']}  ({r['bars']} bars)")
    print(f"  commission/side : {r['commission_per_side']*100:.4f}%  "
          f"({TAKER_FEE*100:.4f}% fee + {SLIPPAGE_PROXY*100:.4f}% slip)")
    print(f"  trades          : {r['trades']}")
    print()
    print(f"  naive B&H       : {r['naive_return_pct']:+.2f}%   (price change, zero friction)")
    print(f"  after fees+slip : {r['backtest_return_pct']:+.2f}%   (backtesting.py)")
    if r["after_funding_pct"] is not None:
        print(f"  after funding   : {r['after_funding_pct']:+.2f}%   "
              f"(funding cost {r['funding_cost_usdt']:+.2f} USDT over {r['funding_events']} events)")
    print()
    print(f"  Sharpe          : {r['sharpe']:.2f}")
    print(f"  Max DD          : {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor   : {r['profit_factor']:.2f}")
    print(f"  Win rate        : {r['win_rate_pct']:.1f}%")

    drag = r["naive_return_pct"] - r["backtest_return_pct"]
    print()
    print(f"  friction drag   : {drag:.2f} pp  (should be ~{r['commission_per_side']*2*100:.2f}% for a single round-trip)")


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Run a backtest on cached Binance Futures data.")
    p.add_argument("--strategy", default="buy-and-hold", choices=list(STRATEGIES.keys()))
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--tf", default="1h", help="timeframe: 15m, 1h, ...")
    p.add_argument("--start", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="YYYY-MM-DD (UTC)")
    p.add_argument("--days", type=int, help="lookback days (overrides --start)")
    p.add_argument("--cash", type=float, default=10_000.0)
    args = p.parse_args()

    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )
    if args.days:
        start = end - timedelta(days=args.days)
    elif args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=365 * 3)

    run_backtest(args.strategy, args.symbol, args.tf, start, end, cash=args.cash)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
