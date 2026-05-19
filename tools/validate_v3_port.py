"""
Paranoid validation: does the live evaluate_signal_v3all_wider4() port match
the V3AllWider4 backtest strategy's entry signals exactly?

Loads a backtest window, runs both, compares entry timestamps. Mismatches = bug.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from exchange.data import load_funding, load_klines  # noqa: E402
from strategy.live_v3all_wider4 import evaluate_signal_v3all_wider4  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402


def main() -> None:
    end = datetime(2025, 12, 31, tzinfo=UTC)
    start = datetime(2025, 10, 1, tzinfo=UTC)  # 3 months, manageable
    symbol = "BTC/USDT:USDT"

    print(f"Validating port on {start.date()} → {end.date()}")

    # ---- Backtest ----
    import importlib
    import strategy.signals_multifactor_v3 as v3
    importlib.reload(v3)
    STRATEGIES["v3-all-wider-4"] = v3.V3AllWider4

    params = dataclasses.replace(
        StrategyParams.from_yaml(), risk_per_trade_pct=2.25, leverage=20)
    print("Running V3AllWider4 backtest...")
    t0 = time.time()
    result = run_backtest(
        strategy_name="v3-all-wider-4", symbol=symbol, timeframe="15m",
        start=start, end=end, cash=10_000.0, leverage=20,
        quiet=True, params_override=params, return_trades=True,
    )
    backtest_trades = result["trades_df"]
    print(f"  {len(backtest_trades) if backtest_trades is not None else 0} trades · "
          f"{result['after_funding_pct']:+.2f}% return ({time.time()-t0:.1f}s)")

    bt_entries = set()
    if backtest_trades is not None and not backtest_trades.empty:
        for _, t in backtest_trades.iterrows():
            ts = pd.to_datetime(t["EntryTime"]).replace(microsecond=0)
            bt_entries.add(ts)

    # ---- Live port: iterate bar-by-bar ----
    print("Running live port bar-by-bar...")
    t0 = time.time()
    # Pull data with warmup
    days_back = (end - start).days + 240  # extra warmup
    bars = load_klines(symbol, "15m", days_back=days_back, end=end)
    bars.columns = [c.capitalize() for c in bars.columns]
    if bars.index.tz is not None:
        bars.index = bars.index.tz_convert("UTC").tz_localize(None)
    funding = load_funding(symbol, days_back=days_back, end=end)
    if funding.index.tz is not None:
        funding.index = funding.index.tz_convert("UTC").tz_localize(None)

    # Build params dict (mimic params.yaml structure)
    with open(REPO_ROOT / "config" / "params.yaml") as f:
        import yaml
        params_dict = yaml.safe_load(f)

    naive_start = start.replace(tzinfo=None)
    naive_end = end.replace(tzinfo=None)
    visible_idx = bars.loc[naive_start:naive_end].index

    live_entries = set()
    n_evaluated = 0
    for i, ts in enumerate(visible_idx):
        # Use all bars up to and including ts
        slice_bars = bars.loc[:ts]
        if len(slice_bars) < 250:
            continue
        # Funding closest to ts (forward-fill)
        f_slice = funding.loc[:ts]
        funding_rate = float(f_slice["funding_rate"].iloc[-1]) if not f_slice.empty else 0.0

        side, sl_dist, tp_dist, dbg = evaluate_signal_v3all_wider4(
            slice_bars, funding_rate, params_dict
        )
        n_evaluated += 1
        if side is not None:
            live_entries.add(ts.to_pydatetime().replace(microsecond=0))
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(visible_idx)} bars evaluated...")

    print(f"  {len(live_entries)} signal fires from live port "
          f"({time.time()-t0:.1f}s, {n_evaluated} bars evaluated)")

    # ---- Compare with ±1 bar slack (backtest reports EntryTime at fill = next
    # bar open with trade_on_close=False; live port reports at signal close) ----
    from datetime import timedelta
    BAR = timedelta(minutes=15)

    matched_bt: set = set()
    matched_live: set = set()
    for bt_ts in bt_entries:
        # accept any live ts within ±1 bar
        candidates = [bt_ts - BAR, bt_ts, bt_ts + BAR]
        for cand in candidates:
            if cand in live_entries and cand not in matched_live:
                matched_bt.add(bt_ts)
                matched_live.add(cand)
                break

    only_bt = bt_entries - matched_bt
    only_live = live_entries - matched_live

    print()
    print(f"Backtest entries: {len(bt_entries)}")
    print(f"Live port entries: {len(live_entries)}")
    print(f"  Matched (±1 bar): {len(matched_bt)}")
    print(f"  Only in backtest: {len(only_bt)}")
    print(f"  Only in live port: {len(only_live)}")

    if only_bt:
        print("\nMissed by live port (backtest entry but live didn't fire):")
        for ts in sorted(only_bt)[:10]:
            print(f"  {ts}")
    if only_live:
        print("\nExtra in live port (live fired but backtest didn't enter):")
        for ts in sorted(only_live)[:10]:
            print(f"  {ts}")

    total = len(matched_bt) + len(only_bt) + len(only_live)
    match_rate = len(matched_bt) / max(total, 1) * 100
    print(f"\nMatch rate (Jaccard-style): {match_rate:.1f}%  ({len(matched_bt)}/{total})")
    if match_rate >= 95:
        print("✓ Port is faithful (>= 95% match)")
    elif match_rate >= 80:
        print("⚠️ Port has minor drift — review missing/extra entries.")
    else:
        print("✗ Port has serious drift — investigate.")

    return bt_entries, live_entries, matched_bt


if __name__ == "__main__":
    main()
