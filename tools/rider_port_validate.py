"""Fidelity port + cross-validation of the 4h Donchian trend-rider onto the
PRODUCTION backtesting.py engine (the same library backtest.py drives).

Goal (2026-05-30): the +47%/yr "trend-rider" edge was found on the bespoke
event engine in tools/hiwr_harness.py. Before it can carry deploy-weight it
must reproduce on an INDEPENDENT engine. This module defines a faithful
backtesting.py Strategy (`RiderLong`) that mirrors the hiwr rider, runs it with
production friction/wiring, and diffs it trade-by-trade against hiwr on the
SAME data slice.

What "faithful" means here:
  - Entry: 4h close > Donchian-55 channel of prior HIGHs (shift 1) AND close >
    EMA200. Long-only. (hiwr build_breakout, allow_short=False.)
  - Stop: SL = sl_atr * ATR(14) below the SIGNAL-BAR CLOSE (production anchors
    the stop to close_v, not the next-bar open — that small gap is exactly the
    engine-convention divergence we are measuring).
  - Target: fixed TP = tp_atr * ATR(14) bracket (the defining feature the
    production donchian-v3 LACKS — it exits on the opposite channel).
  - Time-stop: max_hold bars. Optional chandelier trail (>=5 ATR) by ratcheting
    the live trade's sl.
  - ATR(14) + EMA(200) computed on NATIVE 4h bars (not 1h-resampled), via the
    same strategy.indicators functions hiwr uses.

Engine wiring mirrors backtest.py:run_backtest exactly: plain Backtest (NOT
FractionalBacktest — it desyncs custom indicator columns), cash=$1M,
commission=0.0005/side, margin=1/leverage, trade_on_close=False (next-bar-open
fill), exclusive_orders=True (one-position), finalize_trades=True.

Usage:
    .venv/bin/python -m tools.rider_port_validate --symbol BTC --tf 4h \
        --sl 1.0 --tp 8.0 --trail 0 --leverage 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr, ema  # noqa: E402
from tools.hiwr_harness import (  # noqa: E402
    OOS_END,
    OOS_START,
    TR_BASE,
    _slice,
    _tr_run_config,
    _tr_variant,
    load,
    span_days_of,
)

# Production constants (backtest.py).
SNAPBACK_DEFAULT_CASH = 1_000_000.0
COMMISSION_PER_SIDE = 0.0005


# ---------------------------------------------------------------------------
# Faithful backtesting.py port
# ---------------------------------------------------------------------------
def attach_rider_columns(df: pd.DataFrame, *, donchian_n: int, trend_ema: int,
                         atr_period: int) -> pd.DataFrame:
    """Attach native-4h indicator columns the way hiwr build_breakout computes
    them: DonHi = rolling max of HIGH over donchian_n bars, shifted 1 (channel
    excludes the current bar); Ema = EMA(close); Atr = Wilder ATR. Capitalised
    OHLCV + datetime index for backtesting.py."""
    out = df.copy()
    out.columns = [c.capitalize() for c in out.columns]
    out["DonHi"] = out["High"].rolling(donchian_n).max().shift(1)
    out["Ema"] = ema(out["Close"], trend_ema)
    out["Atr"] = atr(out["High"], out["Low"], out["Close"], atr_period)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


class RiderLong(Strategy):
    """Long-only Donchian trend-rider: small ATR stop, big ATR take-profit,
    optional chandelier trail, time-stop. Mirrors hiwr run_engine geometry."""

    sl_atr = 1.0
    tp_atr = 8.0
    trail_atr = 0.0          # 0 = no trail; >=5 = chandelier
    time_stop_bars = 200
    risk_per_trade_pct = 2.0
    leverage = 3

    def init(self) -> None:
        self._entry_bar: int | None = None
        self._high_water: float = 0.0

    def _position_units(self, sl_distance: float, price: float) -> int:
        if sl_distance <= 0 or not np.isfinite(sl_distance) or price <= 0:
            return 0
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        target = risk_amount / sl_distance
        cap = (self.equity * self.leverage * 0.95) / price
        return max(int(min(target, cap)), 0)

    def next(self) -> None:
        close_v = self.data.Close[-1]
        high_v = self.data.High[-1]
        don_hi = self.data.DonHi[-1]
        ema_v = self.data.Ema[-1]
        atr_v = self.data.Atr[-1]

        if any(v is None or not np.isfinite(v) for v in (don_hi, ema_v, atr_v)):
            return

        if self.position:
            # time-stop (max hold)
            if self.time_stop_bars > 0 and self._entry_bar is not None:
                if (len(self.data) - self._entry_bar) >= self.time_stop_bars:
                    self.position.close()
                    self._entry_bar = None
                    return
            # chandelier trail: ratchet sl up to high_water - K*ATR
            if self.trail_atr > 0 and self.trades:
                self._high_water = max(self._high_water, high_v)
                new_sl = self._high_water - self.trail_atr * atr_v
                tr = self.trades[-1]
                if tr.sl is None or new_sl > tr.sl:
                    tr.sl = new_sl
            return

        # entry: breakout above Donchian-high channel AND above EMA trend
        if close_v > don_hi and close_v > ema_v:
            sl_dist = self.sl_atr * atr_v
            sl = close_v - sl_dist
            tp = close_v + self.tp_atr * atr_v
            units = self._position_units(sl_dist, close_v)
            if units > 0 and sl < close_v:
                self.buy(size=units, sl=sl, tp=tp)
                self._entry_bar = len(self.data)
                self._high_water = high_v


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _prod_trades_summary(trades: pd.DataFrame, commission_side: float) -> dict:
    """Per-trade GROSS price return from fill prices, NET = gross - round-trip
    commission, to match hiwr ret_net (which is price-return minus 2*comm)."""
    if trades is None or len(trades) == 0:
        return {"n": 0}
    g = (trades["ExitPrice"] / trades["EntryPrice"] - 1.0) * np.sign(trades["Size"])
    net = g - 2 * commission_side
    wins = net[net > 0]
    losses = net[net < 0]
    return {
        "n": int(len(net)),
        "win_rate": round(float((net > 0).mean()) * 100, 1),
        "expectancy_pct": round(float(net.mean()) * 100, 3),
        "avg_win_pct": round(float(wins.mean()) * 100, 3) if len(wins) else 0.0,
        "avg_loss_pct": round(float(losses.mean()) * 100, 3) if len(losses) else 0.0,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() != 0 else float("inf"),
        "sum_net_pct": round(float(net.sum()) * 100, 1),
        "_net": net,
        "_entry_time": trades["EntryTime"],
        "_gross": g,
    }


def validate(symbol: str, tf: str, *, sl: float, tp: float, trail: float,
             leverage: float, start: str | None, end: str | None) -> None:
    start = start or OOS_START
    end = end or OOS_END
    df_raw = _slice(load(symbol, tf), start, end)
    span = span_days_of(df_raw)

    # --- hiwr reference (bespoke engine) on the same slice ---
    cfg = _tr_variant(sl_atr=sl, tp_atr=tp,
                      trail_atr=(trail if trail and trail > 0 else None))
    hi = _tr_run_config(df_raw, cfg)
    hiwr = {
        "n": hi.n,
        "win_rate": round(hi.win_rate * 100, 1) if hi.n else None,
        "expectancy_pct": round(hi.expectancy * 100, 3) if hi.n else None,
        "avg_win_pct": round(hi.avg_win * 100, 3),
        "avg_loss_pct": round(hi.avg_loss * 100, 3),
        "profit_factor": round(hi.profit_factor, 2) if hi.n else None,
        "sum_net_pct": round(sum(t.ret_net for t in hi.trades) * 100, 1),
    }
    hiwr_winners = sorted(hi.trades, key=lambda t: t.ret_net, reverse=True)[:6]

    # --- production engine (backtesting.py) on the same slice ---
    data = attach_rider_columns(df_raw, donchian_n=cfg["donchian_n"],
                                trend_ema=cfg["trend_ema"], atr_period=14)
    RiderLong.sl_atr = sl
    RiderLong.tp_atr = tp
    RiderLong.trail_atr = trail if trail else 0.0
    RiderLong.time_stop_bars = cfg["max_hold"]
    RiderLong.leverage = leverage
    bt = Backtest(
        data, RiderLong,
        cash=SNAPBACK_DEFAULT_CASH,
        commission=COMMISSION_PER_SIDE,
        margin=1.0 / max(leverage, 1),
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()
    trades = getattr(stats, "_trades", None)
    prod = _prod_trades_summary(trades, COMMISSION_PER_SIDE)

    # --- report ---
    print(f"\n=== RIDER PORT VALIDATION: {symbol} {tf}  {start}..{end} ({span:.0f}d) ===")
    print(f"    config: don{cfg['donchian_n']} sl{sl}ATR tp{tp}ATR "
          f"trail{trail or 0}ATR hold{cfg['max_hold']} EMA{cfg['trend_ema']} long-only "
          f"lev{leverage}x comm{COMMISSION_PER_SIDE*1e4:.0f}bps/side")
    cols = ["n", "win_rate", "expectancy_pct", "avg_win_pct", "avg_loss_pct",
            "profit_factor", "sum_net_pct"]
    print(f"\n    {'metric':<16}{'hiwr(bespoke)':>16}{'prod(backtesting.py)':>22}")
    for k in cols:
        hv = hiwr.get(k)
        pv = prod.get(k)
        print(f"    {k:<16}{str(hv):>16}{str(pv):>22}")

    # production headline (account return, includes sizing/leverage)
    print(f"\n    prod headline (lev{leverage}x, $1M cash, risk{RiderLong.risk_per_trade_pct}%): "
          f"Return {float(stats['Return [%]']):+.1f}%  "
          f"maxDD {float(stats['Max. Drawdown [%]']):.1f}%  "
          f"Sharpe {float(stats.get('Sharpe Ratio') or 0):.2f}  "
          f"#Trades {int(stats['# Trades'])}")

    # big-winner overlap: do the SAME trades make the money on both engines?
    print(f"\n    -- hiwr top winners (entry date, net%) --")
    for t in hiwr_winners:
        print(f"       {pd.Timestamp(t.entry_t).date()}  {t.ret_net*100:+.1f}%  "
              f"({t.reason}, {t.bars_held}b)")
    if prod.get("n"):
        order = np.argsort(-prod["_net"].to_numpy())
        et = prod["_entry_time"].to_numpy()
        nv = prod["_net"].to_numpy()
        print(f"    -- prod top winners (entry date, net%) --")
        for i in order[:6]:
            print(f"       {pd.Timestamp(et[i]).date()}  {nv[i]*100:+.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--sl", type=float, default=TR_BASE["sl_atr"])
    ap.add_argument("--tp", type=float, default=TR_BASE["tp_atr"])
    ap.add_argument("--trail", type=float, default=0.0)
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    validate(args.symbol, args.tf, sl=args.sl, tp=args.tp, trail=args.trail,
             leverage=args.leverage, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
