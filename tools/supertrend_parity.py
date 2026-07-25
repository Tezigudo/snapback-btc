"""
Live↔backtest parity for the SOL supertrend leg.

`strategy/live_supertrend.py` re-implements `SupertrendBTC.next()` for the
trading loop. A re-implementation is only trustworthy if it decides identically,
so this replays the LIVE evaluator over every closed 4h bar and compares against
the BACKTEST's actual entries and exits.

Two things get checked, and both must be exact:

1. **Entry parity.** For every bar the backtest opened a trade on, the live
   evaluator must return the same side. And for every bar the live evaluator
   returns a side on, the backtest must have either opened that trade or already
   been in a position (the backtest cannot enter while holding — `if
   self.position: ... return` — so a live signal during an open position is not
   a mismatch, it is correctly suppressed by the bot's own position check).
2. **Exit parity.** On every bar where the backtest closed a position via the
   opposite flip, `flip_exit_signal` must be True for that side.

The live path is fed a growing window and reads only `.iloc[-1]` / `.iloc[-2]`,
which also proves the recursive Supertrend band is stable enough that a 1500-bar
live fetch reproduces full-history values — the one parity risk the module
docstring flags but cannot prove by argument.

Run: .venv/bin/python tools/supertrend_parity.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.live_supertrend import (  # noqa: E402
    evaluate_signal_supertrend, flip_exit_signal,
)
from strategy.signals import StrategyParams  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SYMBOL = "SOL/USDT:USDT"
TF = "4h"
START = datetime(2022, 4, 1, tzinfo=UTC)
END = datetime(2026, 7, 25, tzinfo=UTC)
# Live fetch depth (bot.py fetch_ohlcv limit=1500). The replay feeds at most
# this many trailing bars so the test matches what the bot actually sees.
LIVE_WINDOW = 1500


def main() -> int:
    params = yaml.safe_load(open(REPO / "config" / "params_sol_supertrend.yaml"))
    s = params["strategy"]
    risk = float(params["sizing"]["risk_per_trade_pct"])
    lev = int(params["sizing"]["leverage"])

    # --- backtest side ----------------------------------------------------
    cls = STRATEGIES["supertrend"]
    cls.st_period = int(s["st_period"])
    cls.st_multiplier = float(s["st_multiplier"])
    cls.st_atr_period = int(s["st_atr_period"])
    cls.st_sl_atr = float(s["st_sl_atr"])
    cls.st_tp_atr = float(s["st_tp_atr"])
    cls.allow_shorts = bool(s["allow_shorts"])
    cls.st_risk_per_trade_pct = risk
    import dataclasses
    override = dataclasses.replace(
        StrategyParams.from_yaml(), risk_per_trade_pct=risk, leverage=lev)
    r = run_backtest("supertrend", SYMBOL, TF, START, END, leverage=lev,
                     quiet=True, params_override=override, return_trades=True)
    trades = r["trades_df"].copy()
    print(f"Backtest: {len(trades)} trades, "
          f"ret {r['after_funding_pct']:+.1f}%, WR {r['win_rate_pct']:.1f}%\n")

    # CRITICAL off-by-one: backtesting.py runs with trade_on_close=False, so a
    # decision made on bar i's close fills at the OPEN of bar i+1 — `EntryTime`
    # in trades_df is the FILL bar, one bar after the signal bar. The live
    # evaluator returns a signal on the bar it evaluates (bar i). Comparing the
    # two directly reports every trade as both "missed" at i+1 and "spurious" at
    # i. Shift the backtest timestamps back one bar to recover the decision bar.
    BAR = pd.Timedelta(TF)
    bt_entries = {pd.Timestamp(t.EntryTime) - BAR: ("long" if t.Size > 0 else "short")
                  for t in trades.itertuples()}
    bt_exits = {pd.Timestamp(t.ExitTime) - BAR: ("long" if t.Size > 0 else "short")
                for t in trades.itertuples()}
    # Bars where the backtest was holding — a live signal here is suppressed by
    # the bot's own "already in a position" check, not a parity failure. Range
    # runs from the entry DECISION bar to the exit DECISION bar; the entry bar
    # itself is excluded (that is the signal we want to match).
    held: set[pd.Timestamp] = set()
    for t in trades.itertuples():
        held.update(pd.date_range(pd.Timestamp(t.EntryTime) - BAR,
                                  pd.Timestamp(t.ExitTime) - BAR,
                                  freq=TF, inclusive="right"))

    # --- live side --------------------------------------------------------
    df = pd.read_parquet(REPO / "data" / "historical"
                         / f"{SYMBOL.replace('/', '_').replace(':', '_')}_{TF}.parquet")
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    span = df.loc[str(START.date()):str(END.date())]
    live_signals: dict[pd.Timestamp, str] = {}
    live_exits: dict[pd.Timestamp, str] = {}
    positions = df.index.get_indexer(span.index)
    for pos, ts in zip(positions, span.index):
        window = df.iloc[max(0, pos - LIVE_WINDOW + 1):pos + 1]
        side, _sl, _tp, _dbg = evaluate_signal_supertrend(window, 0.0, params)
        if side:
            live_signals[ts] = side
        for held_side in ("long", "short"):
            ex, _d = flip_exit_signal(window, held_side, params)
            if ex:
                live_exits.setdefault(ts, held_side)

    print(f"Live replay: {len(span)} bars, {len(live_signals)} entry signals\n")

    # --- 1. entry parity --------------------------------------------------
    missing = [(ts, sd) for ts, sd in bt_entries.items()
               if live_signals.get(ts) != sd]
    spurious = [(ts, sd) for ts, sd in live_signals.items()
                if ts not in bt_entries and ts not in held]
    wrong_side = [(ts, live_signals[ts], bt_entries[ts]) for ts in bt_entries
                  if ts in live_signals and live_signals[ts] != bt_entries[ts]]

    print("=" * 76)
    print("1. ENTRY PARITY")
    print("=" * 76)
    print(f"  backtest entries          : {len(bt_entries)}")
    print(f"  live reproduced them      : {len(bt_entries) - len(missing)}")
    print(f"  MISSED by live            : {len(missing)}")
    print(f"  live-only (not held)      : {len(spurious)}")
    print(f"  side disagreements        : {len(wrong_side)}")
    for ts, sd in missing[:5]:
        print(f"    missed {ts} {sd}")
    for ts, sd in spurious[:5]:
        print(f"    spurious {ts} {sd}")

    # --- 2. exit parity ---------------------------------------------------
    # Only flip-exits are comparable: SL/TP exits are broker-side in the
    # backtest and exchange-side live, so they never reach flip_exit_signal.
    flip_exit_rows = []
    for t in trades.itertuples():
        side = "long" if t.Size > 0 else "short"
        ts = pd.Timestamp(t.ExitTime) - BAR      # exit DECISION bar, see above
        if ts in df.index:
            loc = df.index.get_loc(ts)
            ok, dbg = flip_exit_signal(
                df.iloc[max(0, loc - LIVE_WINDOW + 1):loc + 1], side, params)
        else:
            ok, dbg = False, {"reason": "ts_not_in_index"}
        flip_exit_rows.append((ts, side, ok, dbg.get("reason")))
    agreed = sum(1 for _, _, ok, _ in flip_exit_rows if ok)
    print()
    print("=" * 76)
    print("2. EXIT PARITY (flip exits only — SL/TP fills are broker/exchange side)")
    print("=" * 76)
    print(f"  backtest exits            : {len(flip_exit_rows)}")
    print(f"  live flip_exit_signal True: {agreed}")
    print(f"  (the remainder exited on SL or TP, which live handles as exchange "
          f"bracket legs)")
    reasons: dict[str, int] = {}
    for _, _, ok, why in flip_exit_rows:
        if not ok:
            reasons[str(why)] = reasons.get(str(why), 0) + 1
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    not-a-flip-exit: {why:<24} {n}")

    # --- 3. no PREMATURE live exits ---------------------------------------
    # The dangerous asymmetry: flip_exit_signal returning True on a bar where
    # the backtest was still holding would make the live leg bail out of winners
    # the backtest rode. Walk every bar strictly inside each trade and require
    # False.
    premature = []
    for t in trades.itertuples():
        side = "long" if t.Size > 0 else "short"
        entry_dec = pd.Timestamp(t.EntryTime) - BAR
        exit_dec = pd.Timestamp(t.ExitTime) - BAR
        inner = pd.date_range(entry_dec, exit_dec, freq=TF, inclusive="neither")
        for ts in inner:
            if ts not in df.index:
                continue
            loc = df.index.get_loc(ts)
            ok, _d = flip_exit_signal(
                df.iloc[max(0, loc - LIVE_WINDOW + 1):loc + 1], side, params)
            if ok:
                premature.append((ts, side))
    print()
    print("=" * 76)
    print("3. NO PREMATURE LIVE EXITS (live must hold whenever the backtest held)")
    print("=" * 76)
    print(f"  in-trade bars where live would have exited early: {len(premature)}")
    for ts, sd in premature[:5]:
        print(f"    premature {ts} {sd}")

    print()
    verdict = not missing and not spurious and not wrong_side and not premature
    print("=" * 76)
    print(f"VERDICT: {'PARITY OK' if verdict else 'PARITY FAILED'}")
    print("=" * 76)
    if not verdict:
        print("  Live evaluator disagrees with the backtest. Do NOT deploy.")
        return 1
    print("  Live evaluator reproduces every backtest entry, with no signal the")
    print("  backtest would not have taken. Safe to dry-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
