"""DCA sim — v1 (BTC) + SOL cnh-hybrid-short, with +$100/month contributions.

Event-driven. Each leg has its own wallet; the monthly deposit is split between
them. Risk-based sizing on each leg's CURRENT equity, with that leg's REAL
exchange minimums (v1/BTC: $50 min-notional, 0.001 min-qty; SOL: $5, 0.01).
A trade below the minimum is SKIPPED (the capital-efficiency story).

Per-trade return: v1 uses CSV ReturnPct (prod backtest, commission-included);
SOL uses net_pct (friction-included). Both already directional.

Benchmark: the SAME $100/mo dumped into spot BTC (no leverage), to answer
"does the bot stack beat just stacking BTC?".

Honest split: reports full-history AND OOS-only (2024-07+) — the OOS money-
weighted return is the only unbiased forward proxy (pre-2024 is in-sample).

Run: uv run python tools/dca_sim.py [--risk 0.0275] [--monthly 100] [--weight-v1 0.5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "historical"

LEVERAGE = 20
OOS_START = pd.Timestamp("2024-07-01")

# Exchange minimums (live 2026-05-30)
MIN = {"v1": (50.0, 0.001), "sol": (5.0, 0.01)}  # (min_notional, min_qty)
V1_SL_FRAC = 0.015      # v1 fixed 1.5% stop
SOL_SL_K = 1.5          # SOL stop = 1.5 x ATR


def latest_v1_csv() -> Path:
    cands = sorted(REPORTS.glob("full_history_*_v1_trades.csv"))
    if not cands:
        raise RuntimeError("no v1 trade CSV")
    return cands[-1]


def load_v1() -> pd.DataFrame:
    df = pd.read_csv(latest_v1_csv())
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True).dt.tz_localize(None)
    df["ExitTime"] = pd.to_datetime(df["ExitTime"], utc=True).dt.tz_localize(None)
    df["leg"] = "v1"
    df["ret"] = df["ReturnPct"] / 100.0
    df["sl_frac"] = V1_SL_FRAC
    return df[["EntryTime", "ExitTime", "EntryPrice", "ret", "sl_frac", "leg"]]


def load_sol() -> pd.DataFrame:
    df = pd.read_csv(REPORTS / "sol_hybrid_short_trades.csv")
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True).dt.tz_localize(None)
    df["ExitTime"] = pd.to_datetime(df["ExitTime"], utc=True).dt.tz_localize(None)
    df["leg"] = "sol"
    df["ret"] = df["net_pct"]
    df["sl_frac"] = SOL_SL_K * df["atr_pct"]
    return df[["EntryTime", "ExitTime", "EntryPrice", "ret", "sl_frac", "leg"]]


def btc_spot_daily() -> pd.Series:
    df = pd.read_parquet(DATA / "BTC_USDT_USDT_1h.parquet")
    df.columns = [c.lower() for c in df.columns]
    s = df["close"].resample("1D").last().dropna()
    s.index = s.index.tz_localize(None) if s.index.tz else s.index
    return s


def money_weighted_annual(cashflows: list[tuple[pd.Timestamp, float]],
                          terminal_ts: pd.Timestamp, terminal_val: float) -> float:
    """Annualized IRR from dated cashflows (deposits negative) + terminal value."""
    flows = [(t, -amt) for t, amt in cashflows] + [(terminal_ts, terminal_val)]
    t0 = flows[0][0]
    yrs = np.array([(t - t0).days / 365.25 for t, _ in flows])
    amts = np.array([a for _, a in flows])

    def npv(r):
        return np.sum(amts / (1.0 + r) ** yrs)

    lo, hi = -0.99, 10.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2 * 100.0


def run(trades: pd.DataFrame, monthly: float, risk: float, weight_v1: float,
        start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """Event loop: deposits monthly, trades sized on current per-leg equity."""
    legs = {"v1": 0.0, "sol": 0.0}
    wts = {"v1": weight_v1, "sol": 1.0 - weight_v1}
    deposits: list[tuple[pd.Timestamp, float]] = []
    eq_curve: list[tuple[pd.Timestamp, float]] = []
    fires = {"v1": 0, "sol": 0}
    skips = {"v1": 0, "sol": 0}
    realized = {"v1": 0, "sol": 0}

    months = pd.date_range(start_ts.normalize().replace(day=1), end_ts, freq="MS")
    tr = trades[(trades["EntryTime"] >= start_ts) & (trades["EntryTime"] <= end_ts)]
    tr = tr.sort_values("EntryTime")

    def deposit(ts):
        for lg in legs:
            legs[lg] += monthly * wts[lg]
        deposits.append((ts, monthly))

    mi = 0
    for _, t in tr.iterrows():
        # process any monthly deposits up to this trade's entry
        while mi < len(months) and months[mi] <= t["EntryTime"]:
            deposit(months[mi])
            eq_curve.append((months[mi], sum(legs.values())))
            mi += 1
        lg = t["leg"]
        eq = legs[lg]
        if eq <= 0 or t["sl_frac"] <= 0 or not np.isfinite(t["sl_frac"]):
            continue
        nf = risk / t["sl_frac"]                  # notional / equity
        notional = min(nf * eq, eq * LEVERAGE * 0.95)
        price = float(t["EntryPrice"])
        qty = notional / price
        min_notional, min_qty = MIN[lg]
        if qty < min_qty or notional < min_notional:
            skips[lg] += 1
            continue
        pnl = notional * float(t["ret"])
        legs[lg] += pnl
        fires[lg] += 1
        realized[lg] += 1
        eq_curve.append((t["ExitTime"], sum(legs.values())))
    # remaining deposits
    while mi < len(months):
        deposit(months[mi])
        eq_curve.append((months[mi], sum(legs.values())))
        mi += 1

    curve = pd.Series({ts: v for ts, v in eq_curve}).sort_index()
    deposited = sum(a for _, a in deposits)
    terminal = float(sum(legs.values()))
    peak = curve.cummax()
    max_dd = float((curve / peak - 1.0).min() * 100.0) if len(curve) else 0.0
    irr = money_weighted_annual(deposits, curve.index[-1], terminal) if deposits else float("nan")
    return {
        "deposited": deposited, "terminal": terminal,
        "profit": terminal - deposited,
        "roi_on_deposits_pct": (terminal - deposited) / deposited * 100 if deposited else 0,
        "irr_pct": irr, "max_dd_pct": max_dd,
        "fires": fires, "skips": skips, "curve": curve,
        "n_months": len(deposits),
    }


def btc_dca(prices: pd.Series, monthly: float, start_ts, end_ts):
    months = pd.date_range(start_ts.normalize().replace(day=1), end_ts, freq="MS")
    btc, deposited = 0.0, 0.0
    flows = []
    for m in months:
        px = prices.asof(m)
        if not np.isfinite(px) or px <= 0:
            continue
        btc += monthly / px
        deposited += monthly
        flows.append((m, monthly))
    terminal = btc * float(prices.asof(end_ts))
    irr = money_weighted_annual(flows, end_ts, terminal)
    return {"deposited": deposited, "terminal": terminal,
            "profit": terminal - deposited,
            "roi_on_deposits_pct": (terminal - deposited) / deposited * 100,
            "irr_pct": irr}


def report(tag, r, bench=None):
    print(f"\n=== {tag} ===")
    print(f"  months: {r['n_months']}   deposited: ${r['deposited']:,.0f}")
    print(f"  terminal wallet: ${r['terminal']:,.0f}   profit: ${r['profit']:,.0f}")
    print(f"  ROI on deposits: {r['roi_on_deposits_pct']:+.1f}%   "
          f"money-weighted annual: {r['irr_pct']:+.1f}%/yr")
    print(f"  max drawdown: {r['max_dd_pct']:+.1f}%")
    print(f"  fires: v1={r['fires']['v1']} sol={r['fires']['sol']}   "
          f"skips: v1={r['skips']['v1']} sol={r['skips']['sol']}")
    if bench:
        print(f"  -- benchmark: same $/mo into spot BTC --")
        print(f"     terminal ${bench['terminal']:,.0f}  profit ${bench['profit']:,.0f}  "
              f"ROI {bench['roi_on_deposits_pct']:+.1f}%  "
              f"annual {bench['irr_pct']:+.1f}%/yr")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, default=0.0275)
    ap.add_argument("--monthly", type=float, default=100.0)
    ap.add_argument("--weight-v1", type=float, default=0.5)
    args = ap.parse_args()

    trades = pd.concat([load_v1(), load_sol()], ignore_index=True).sort_values("EntryTime")
    prices = btc_spot_daily()
    first = max(trades["EntryTime"].min(), prices.index.min())
    last = min(trades["EntryTime"].max(), prices.index.max())

    print(f"DCA sim: +${args.monthly:.0f}/mo, risk {args.risk*100:.2f}%, "
          f"weights v1={args.weight_v1:.0%}/sol={1-args.weight_v1:.0%}, leverage {LEVERAGE}x")
    print(f"v1 trades: {(trades['leg']=='v1').sum()}   sol trades: {(trades['leg']=='sol').sum()}")

    # Full history
    r_full = run(trades, args.monthly, args.risk, args.weight_v1, first, last)
    b_full = btc_dca(prices, args.monthly, first, last)
    report(f"FULL HISTORY {str(first)[:7]}..{str(last)[:7]} (incl. in-sample yrs)",
           r_full, b_full)

    # OOS-only (honest forward proxy)
    r_oos = run(trades, args.monthly, args.risk, args.weight_v1, OOS_START, last)
    b_oos = btc_dca(prices, args.monthly, OOS_START, last)
    report(f"OOS ONLY {str(OOS_START)[:7]}..{str(last)[:7]} (held-out = forward proxy)",
           r_oos, b_oos)

    print("\nNOTE: assumes BOTH legs run from the start; in reality the SOL leg is")
    print("gated behind ALLOWED_SYMBOLS + multi-coin infra. Past returns are a")
    print("backtest analogue, NOT a forward guarantee. No kill-switch modeled here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
