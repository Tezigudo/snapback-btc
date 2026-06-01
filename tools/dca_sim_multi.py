"""Multi-leg DCA sim — v1 + donchian + SOL-hybrid-short + capitulation, +$100/mo.

Generalizes tools/dca_sim.py to N legs. Each leg: own wallet, risk-based sizing
on current equity (notional = risk/sl_frac * equity, capped at 20x), that leg's
real exchange minimums (BTC legs $50/0.001, alt legs $5/0.01), per-trade net ret.

Legs & stop conventions:
  v1            BTC  sl_frac = 1.5% fixed
  donchian-cons BTC  sl_frac = 1.5 x ATR(20,4h)/entry   (computed from parquet)
  sol-hybrid    SOL  sl_frac = 1.5 x ATR/entry          (from CSV)
  capitulation  alts sl_frac = 2.0 x ATR/entry          (from CSV)

Reports full-history AND OOS-only (forward proxy) + BTC-spot-DCA benchmark.
Run: uv run python tools/dca_sim_multi.py [--monthly 100] [--risk 0.0275]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "historical"

from tools.dca_sim import money_weighted_annual, btc_spot_daily, btc_dca  # noqa: E402

LEVERAGE = 20
OOS_START = pd.Timestamp("2024-07-01")
BTC_MIN = (50.0, 0.001)
ALT_MIN = (5.0, 0.01)


def _btc_4h_atr_pct() -> pd.Series:
    k = pd.read_parquet(DATA / "BTC_USDT_USDT_4h.parquet")
    k.columns = [c.lower() for c in k.columns]
    if k.index.tz is not None:
        k.index = k.index.tz_localize(None)
    hi, lo, cl = k["high"], k["low"], k["close"]
    pc = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20, min_periods=20).mean()
    return (atr / cl)


def load_v1():
    p = sorted(REPORTS.glob("full_history_*_v1_trades.csv"))[-1]
    df = pd.read_csv(p)
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True).dt.tz_localize(None)
    df["ExitTime"] = pd.to_datetime(df["ExitTime"], utc=True).dt.tz_localize(None)
    df["ret"] = df["ReturnPct"] / 100.0
    df["sl_frac"] = 0.015
    df["leg"], df["min_n"], df["min_q"] = "v1", BTC_MIN[0], BTC_MIN[1]
    return df[["EntryTime", "ExitTime", "EntryPrice", "ret", "sl_frac", "leg", "min_n", "min_q"]]


def load_donchian():
    p = sorted(REPORTS.glob("full_history_*_d3cons_trades.csv"))[-1]
    df = pd.read_csv(p)
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True).dt.tz_localize(None)
    df["ExitTime"] = pd.to_datetime(df["ExitTime"], utc=True).dt.tz_localize(None)
    df["ret"] = df["ReturnPct"] / 100.0
    atr_pct = _btc_4h_atr_pct()
    df["sl_frac"] = [1.5 * float(atr_pct.asof(t - pd.Timedelta(minutes=1)))
                     for t in df["EntryTime"]]
    df["leg"], df["min_n"], df["min_q"] = "donchian", BTC_MIN[0], BTC_MIN[1]
    return df[["EntryTime", "ExitTime", "EntryPrice", "ret", "sl_frac", "leg", "min_n", "min_q"]]


def load_csv_leg(fname, leg, sl_k, mn):
    df = pd.read_csv(REPORTS / fname)
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True).dt.tz_localize(None)
    df["ExitTime"] = pd.to_datetime(df["ExitTime"], utc=True).dt.tz_localize(None)
    if "net_pct" in df.columns:
        df["ret"] = df["net_pct"]
    df["sl_frac"] = sl_k * df["atr_pct"]
    df["leg"], df["min_n"], df["min_q"] = leg, mn[0], mn[1]
    return df[["EntryTime", "ExitTime", "EntryPrice", "ret", "sl_frac", "leg", "min_n", "min_q"]]


def run(trades, legs_weights, monthly, risk, start_ts, end_ts):
    legs = {lg: 0.0 for lg in legs_weights}
    deposits, eq_curve = [], []
    fires = {lg: 0 for lg in legs_weights}
    skips = {lg: 0 for lg in legs_weights}
    months = pd.date_range(start_ts.normalize().replace(day=1), end_ts, freq="MS")
    tr = trades[(trades["EntryTime"] >= start_ts) & (trades["EntryTime"] <= end_ts)
                & (trades["leg"].isin(legs_weights))].sort_values("EntryTime")

    def deposit(ts):
        for lg, wt in legs_weights.items():
            legs[lg] += monthly * wt
        deposits.append((ts, monthly))

    mi = 0
    for _, t in tr.iterrows():
        while mi < len(months) and months[mi] <= t["EntryTime"]:
            deposit(months[mi]); eq_curve.append((months[mi], sum(legs.values()))); mi += 1
        lg = t["leg"]; eq = legs[lg]
        sl = t["sl_frac"]
        if eq <= 0 or not np.isfinite(sl) or sl <= 0:
            continue
        notional = min(risk / sl * eq, eq * LEVERAGE * 0.95)
        price = float(t["EntryPrice"]); qty = notional / price
        if qty < t["min_q"] or notional < t["min_n"]:
            skips[lg] += 1; continue
        legs[lg] += notional * float(t["ret"]); fires[lg] += 1
        eq_curve.append((t["ExitTime"], sum(legs.values())))
    while mi < len(months):
        deposit(months[mi]); eq_curve.append((months[mi], sum(legs.values()))); mi += 1

    curve = pd.Series({ts: v for ts, v in eq_curve}).sort_index()
    deposited = sum(a for _, a in deposits); terminal = float(sum(legs.values()))
    peak = curve.cummax(); max_dd = float((curve / peak - 1).min() * 100) if len(curve) else 0
    irr = money_weighted_annual(deposits, curve.index[-1], terminal) if deposits else float("nan")
    return {"deposited": deposited, "terminal": terminal, "profit": terminal - deposited,
            "roi": (terminal - deposited) / deposited * 100 if deposited else 0,
            "irr": irr, "max_dd": max_dd, "fires": fires, "skips": skips,
            "per_leg": dict(legs), "n_months": len(deposits)}


def report(tag, r, bench=None):
    print(f"\n=== {tag} ===")
    print(f"  deposited ${r['deposited']:,.0f}  ->  terminal ${r['terminal']:,.0f}  "
          f"(profit ${r['profit']:,.0f})")
    print(f"  ROI {r['roi']:+.1f}%   money-weighted {r['irr']:+.1f}%/yr   maxDD {r['max_dd']:+.1f}%")
    print(f"  fires: " + "  ".join(f"{k}={v}" for k, v in r['fires'].items()))
    if bench:
        print(f"  benchmark BTC-spot DCA: ${bench['terminal']:,.0f}  "
              f"{bench['roi_on_deposits_pct']:+.1f}%  {bench['irr_pct']:+.1f}%/yr")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly", type=float, default=100.0)
    ap.add_argument("--risk", type=float, default=0.0275)
    args = ap.parse_args()

    trades = pd.concat([
        load_v1(), load_donchian(),
        load_csv_leg("sol_hybrid_short_trades.csv", "sol", 1.5, ALT_MIN),
        load_csv_leg("capitulation_long_trades.csv", "capit", 2.0, ALT_MIN),
    ], ignore_index=True).sort_values("EntryTime")
    for lg in ("v1", "donchian", "sol", "capit"):
        print(f"  {lg}: {(trades['leg']==lg).sum()} trades")

    prices = btc_spot_daily()
    first = max(trades["EntryTime"].min(), prices.index.min())
    last = min(trades["EntryTime"].max(), prices.index.max())

    # equal-weight 4 legs
    w = {"v1": 0.25, "donchian": 0.25, "sol": 0.25, "capit": 0.25}
    print(f"\n+${args.monthly:.0f}/mo, risk {args.risk*100:.2f}%, equal-weight 4 legs, {LEVERAGE}x")
    report(f"FULL {str(first)[:7]}..{str(last)[:7]}", run(trades, w, args.monthly, args.risk, first, last),
           btc_dca(prices, args.monthly, first, last))
    report(f"OOS {str(OOS_START)[:7]}..{str(last)[:7]} (forward proxy)",
           run(trades, w, args.monthly, args.risk, OOS_START, last),
           btc_dca(prices, args.monthly, OOS_START, last))

    # 2-leg (v1+sol) vs 4-leg DD comparison, OOS
    print("\n--- OOS drawdown comparison (does diversification smooth the -25%?) ---")
    for name, ww in [
        ("v1+sol (50/50)", {"v1": .5, "sol": .5}),
        ("4-leg equal", {"v1": .25, "donchian": .25, "sol": .25, "capit": .25}),
        ("v1+don+sol+capit (sol/capit heavy)", {"v1": .15, "donchian": .15, "sol": .35, "capit": .35}),
    ]:
        r = run(trades, ww, args.monthly, args.risk, OOS_START, last)
        print(f"  {name:<38} {r['irr']:+6.1f}%/yr  maxDD {r['max_dd']:+6.1f}%")
    print("\nNOTE: only v1 is live. donchian/sol/capit gated (multi-coin infra +")
    print("ALLOWED_SYMBOLS). alt min-qty approximated. Backtest analogue, not a guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
