"""Fractional-BTC sizing via harness-level price scaling.

WHY THIS EXISTS
---------------
backtesting.py 0.6.5 only accepts integer "units" via Backtest.run(). The
strategy `_position_units` returns int(target_btc), which truncates to 0
for any equity level where target_btc < 1 BTC. At BTC ~$70k with 1% risk,
this happens for all equity below ~$1M cash — silently dropping signals.

Binance USDT-M perps support 0.001 BTC step in live trading (see
exchange/constraints.py: qty_step=0.001). To match this in the backtest
without forking backtesting.py, we scale the OHLC feed by `PRICE_SCALE`
at Backtest() construction time. Under PRICE_SCALE=0.001:
    1 broker-int-unit  ==  0.001 BTC
    int(target_btc)    ==  floor(real_btc * 1000)  == milli-BTC count
    units * price_scaled == real notional in USD (cash stays unscaled)

The math is fully scale-invariant for: equity, Return%, MaxDD%, commission
rate, and trade pnl_pct. Only the per-trade absolute pnl-in-dollars goes
through the broker's `units * price_diff` formula, which is also invariant
(price_diff scales the same as price_open and units = milli-units count).

USAGE
-----
    .venv/bin/python tools/_fractional_run.py

Sanity-checks 8 strategies on 2024 H1 at $1M cash, 20x leverage cap.
Reports trade count + return; warns on regressions vs pre-fix baseline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_adaptive_trend import AdaptiveTrendV1  # noqa: E402
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2  # noqa: E402
from strategy.signals_adaptive_trend_v2_vol_scaled_sizing import (  # noqa: E402
    AdaptiveTrendV2_vol_scaled_sizing,
)
from strategy.signals_adx_dual_regime import ADXDualRegimeV1  # noqa: E402
from strategy.signals_divergence import DivergenceV1  # noqa: E402
from strategy.signals_divergence_v2 import DivergenceV2, DivergenceV2Loose  # noqa: E402
from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from strategy.signals_volume_profile import VolumeProfilePOC  # noqa: E402


PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20  # 20x leverage cap
PRICE_SCALE = 0.001  # 1 broker-unit == 0.001 BTC


def _load_2024h1_scaled(scale: float) -> pd.DataFrame:
    """Load BTC 15m parquet, slice 2024 H1, scale OHLC by `scale`.

    Volume is NOT scaled (only used for SMA gates / OBV / MFI — all
    scale-invariant when used as ratios).
    """
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-07-01")
    sl = df.loc[(df.index >= start) & (df.index < end)].copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * scale
    return sl


def _load_2024h1_unscaled() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-07-01")
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def _run(name: str, cls, df: pd.DataFrame, **kw) -> tuple[int, float, str, float]:
    try:
        bt = Backtest(
            df, cls,
            cash=CASH,
            commission=COMMISSION,
            margin=MARGIN,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
        # Silence the warnings about scaled prices / 4H parquet mismatch — they
        # are expected (multifactor-v1 and divergence-v2 load 4H parquets which
        # are NOT scaled; the 4H EMA200 gate will be wrong in scaled runs).
        stats = bt.run(**kw)
        n = int(stats["# Trades"])
        ret = float(stats["Return [%]"])
        dd = float(stats.get("Max. Drawdown [%]") or 0.0)
        return n, ret, "ok", dd
    except Exception as e:  # noqa: BLE001
        return 0, 0.0, f"CRASH: {type(e).__name__}: {e}", 0.0


def main() -> int:
    print(f"Loading {PARQUET.name} ...", flush=True)
    df_scaled = _load_2024h1_scaled(PRICE_SCALE)
    df_unscaled = _load_2024h1_unscaled()
    print(f"  bars: {len(df_scaled)}  span: {df_scaled.index[0]} .. {df_scaled.index[-1]}", flush=True)
    print(f"  PRICE_SCALE = {PRICE_SCALE}  (1 unit = {PRICE_SCALE} BTC)", flush=True)
    print(f"  unscaled close range: ${df_unscaled['Close'].min():.0f} .. ${df_unscaled['Close'].max():.0f}", flush=True)
    print(f"    scaled close range: {df_scaled['Close'].min():.2f} .. {df_scaled['Close'].max():.2f}", flush=True)
    print()

    # Disable 4H regime gate where present — the 4H parquet is loaded UNSCALED
    # inside the strategy init(), so combining it with scaled 15m prices is
    # apples-to-oranges. The 4H gate is not part of the sizing bug fix.
    cases = [
        ("multifactor-v1",            DayTradeMultiFactorBTC,           {"use_mtf_4h_gate": False}),
        ("adaptive-trend-v1",         AdaptiveTrendV1,                  {}),
        ("adaptive-trend-v2",         AdaptiveTrendV2,                  {}),
        ("adaptive-trend-v2-vscaled", AdaptiveTrendV2_vol_scaled_sizing, {}),
        ("divergence-v1",             DivergenceV1,                     {}),
        ("divergence-v2-loose",       DivergenceV2Loose,                {"use_4h_regime_gate": False}),
        ("adx-dual-regime",           ADXDualRegimeV1,                  {}),
        ("volume-profile-poc",        VolumeProfilePOC,                 {}),
    ]

    print("=== PRE-FIX (unscaled prices, int(BTC) truncation active) ===", flush=True)
    pre_rows = []
    for label, cls, kw in cases:
        t0 = time.time()
        n, ret, status, dd = _run(label, cls, df_unscaled, **kw)
        pre_rows.append((label, n, ret, status, dd))
        print(f"  {label:32s}  trades={n:5d}  ret={ret:+8.2f}%  dd={dd:+7.2f}%  [{status}]  ({time.time()-t0:.1f}s)", flush=True)

    print()
    print("=== POST-FIX (scaled prices, 0.001 BTC step) ===", flush=True)
    post_rows = []
    for label, cls, kw in cases:
        t0 = time.time()
        n, ret, status, dd = _run(label, cls, df_scaled, **kw)
        post_rows.append((label, n, ret, status, dd))
        print(f"  {label:32s}  trades={n:5d}  ret={ret:+8.2f}%  dd={dd:+7.2f}%  [{status}]  ({time.time()-t0:.1f}s)", flush=True)

    print()
    print("=== SANITY TABLE (2024 H1, $1M, 20x cap) ===", flush=True)
    print(f"{'Strategy':<32s} {'Pre trades / return':>24s}  {'Post trades / return':>24s}  Notes", flush=True)
    print("-" * 110, flush=True)
    crash_count = 0
    for (lab, n0, r0, s0, _), (_, n1, r1, s1, _) in zip(pre_rows, post_rows):
        note = ""
        if s1 != "ok":
            note = "POST CRASH"
            crash_count += 1
        elif n1 < n0:
            note = f"trades decreased ({n0} -> {n1})"
        print(f"{lab:<32s} {n0:>5d} / {r0:+8.2f}%    {n1:>5d} / {r1:+8.2f}%      {note}", flush=True)

    if crash_count:
        print(f"\n*** {crash_count} POST-FIX CRASH(ES) — investigate before declaring done ***", flush=True)
        return 1
    print("\nAll runs completed without errors.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
