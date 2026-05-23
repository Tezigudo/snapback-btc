"""Walk-forward realistic-environment simulator for combined 50/50 deploy
at $50.50 per leg ($101 total) on one Binance account in hedge mode.

What this models (that the original backtest did NOT):
  - min_qty 0.001 BTC and min_notional $50 SKIP behaviour at the real
    capital (most signals get skipped at $50.50)
  - Dynamic position sizing — target_btc recomputed from CURRENT logical
    equity at every entry, so compounding gradually opens up more signals
  - Two-bot logical-equity tracking — each strategy independently tracks
    its $50.50 base and grows/shrinks with its own PnL
  - Combined wallet equity = sum of two legs

What this does NOT model (still):
  - Race conditions on the actual exchange API
  - Margin contention at the wallet level (assumes isolated margin per
    position so each leg's losses are bounded by its own allocation)
  - The exchange rejecting orders for any reason other than min-qty/notional

Input: existing trade ledgers from full_history_*_trades.csv (the original
backtest at $1M cash). We replay each signal at $50.50, check min-qty,
fire or skip.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "historical"

# --- Constants -----------------------------------------------------------
MIN_QTY_BTC = 0.001
MIN_NOTIONAL_USDT = 50.0
START_PER_LEG = 50.50
START_COMBINED = 101.00
BACKTEST_CASH_BASE = 1_000_000.0   # original backtest cash

# v1 risk params (mirror config/params.yaml). RISK_PCT overridable via CLI.
V1_RISK_PCT = 0.02         # 2% per trade — default
V1_SL_PCT = 0.015          # 1.5% stop

# Donchian-v3 cons risk params
DON_RISK_PCT = 0.02
DON_ATR_SL_K = 1.5         # SL distance = 1.5 × ATR
DON_ATR_PERIOD = 20        # ATR(20) on 4h


def latest_ts() -> str:
    cands = sorted(glob.glob(str(REPORTS / "full_history_*_v1_trades.csv")))
    if not cands:
        raise RuntimeError("no v1 trade CSV found")
    return Path(cands[-1]).stem.replace("full_history_", "").replace("_v1_trades", "")


def load_trades(strategy_slug: str, ts: str) -> pd.DataFrame:
    p = REPORTS / f"full_history_{ts}_{strategy_slug}_trades.csv"
    df = pd.read_csv(p)
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df["ExitTime"]  = pd.to_datetime(df["ExitTime"],  utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df["side"] = np.where(df["Size"] > 0, +1, -1)
    return df.sort_values("EntryTime").reset_index(drop=True)


# --- ATR(20) on 4h for Donchian SL distance --------------------------------
_ATR_CACHE: pd.Series | None = None


def atr_4h_at(t: pd.Timestamp) -> float:
    """Return the 4h ATR(20) value at the latest 4h close strictly before t.

    Cached on first call.
    """
    global _ATR_CACHE
    if _ATR_CACHE is None:
        k4 = pd.read_parquet(DATA / "BTC_USDT_USDT_4h.parquet")
        ts_col = next((c for c in k4.columns if c.lower() in ("timestamp", "ts", "time", "datetime")), None)
        if ts_col:
            k4["_ts"] = pd.to_datetime(k4[ts_col], utc=True)
            k4 = k4.set_index("_ts")
        if k4.index.tz is not None:
            k4.index = k4.index.tz_convert("UTC").tz_localize(None)
        k4 = k4.sort_index()
        hi, lo, cl = k4["high"], k4["low"], k4["close"]
        prev_c = cl.shift(1)
        tr = pd.concat([hi - lo, (hi - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
        _ATR_CACHE = tr.rolling(DON_ATR_PERIOD, min_periods=DON_ATR_PERIOD).mean()
    return float(_ATR_CACHE.asof(t - pd.Timedelta(minutes=1)))


# --- Sizing decisions at $50.50 --------------------------------------------
def v1_target_btc(equity: float, entry_price: float) -> tuple[float, str | None]:
    """Return (target_btc, skip_reason). skip_reason is None if fires."""
    if entry_price <= 0:
        return 0.0, "bad_price"
    sl_dist = V1_SL_PCT * entry_price
    target = (V1_RISK_PCT * equity) / sl_dist
    if target < MIN_QTY_BTC:
        return target, "min_qty"
    notional = target * entry_price
    if notional < MIN_NOTIONAL_USDT:
        return target, "min_notional"
    return target, None


def don_target_btc(equity: float, entry_price: float, entry_time: pd.Timestamp) -> tuple[float, str | None]:
    a = atr_4h_at(entry_time)
    if not (a > 0 and np.isfinite(a)):
        return 0.0, "no_atr"
    sl_dist = DON_ATR_SL_K * a
    target = (DON_RISK_PCT * equity) / sl_dist
    if target < MIN_QTY_BTC:
        return target, "min_qty"
    notional = target * entry_price
    if notional < MIN_NOTIONAL_USDT:
        return target, "min_notional"
    return target, None


# --- Walk-forward sim of one leg -------------------------------------------
def simulate_leg(trades: pd.DataFrame, start_equity: float,
                 sizer: callable, label: str,
                 kill_switch_frac: float = 0.82) -> dict:
    """Simulate one strategy's leg with min-qty skip + kill-switch halt.

    kill_switch_frac: if equity falls to start_equity * this_frac, halt.
    Set to 0 to disable kill-switch.
    """
    kill_floor = start_equity * kill_switch_frac if kill_switch_frac > 0 else 0.0
    killed_at: pd.Timestamp | None = None
    killed_equity: float | None = None

    equity = start_equity
    history = [{"ts": trades["EntryTime"].iloc[0] - pd.Timedelta(days=1), "equity": equity}]
    fires = 0
    skips_min_qty = 0
    skips_min_notional = 0
    skips_other = 0
    realized = []

    for _, t in trades.iterrows():
        if killed_at is not None:
            break
        entry_price = float(t["EntryPrice"])
        if "exit_time" in t and pd.notna(t["ExitTime"]):
            pass
        # Compute target_btc at CURRENT equity
        if sizer.__name__ == "v1_target_btc":
            target, reason = sizer(equity, entry_price)
        else:
            target, reason = sizer(equity, entry_price, t["EntryTime"])

        if reason is not None:
            if reason == "min_qty":
                skips_min_qty += 1
            elif reason == "min_notional":
                skips_min_notional += 1
            else:
                skips_other += 1
            continue

        # Fire — use target_btc, compute PnL from realized exit price
        exit_price = float(t["ExitPrice"])
        side = int(t["side"])
        # Round target_btc DOWN to 0.001 step (Binance precision — floor, not round).
        # The live bot uses round_qty_down() in exchange/constraints.py; matching here.
        qty = math.floor(target / 0.001) * 0.001
        if qty < MIN_QTY_BTC - 1e-9:  # tolerate float noise
            skips_min_qty += 1
            continue
        # After rounding-down, re-check notional in case rounding pushed it below the min
        if qty * entry_price < MIN_NOTIONAL_USDT:
            skips_min_notional += 1
            continue
        # Commission: 5 bps per side = 10 bps round-trip
        notional_entry = qty * entry_price
        notional_exit = qty * exit_price
        commission = 0.0005 * (notional_entry + notional_exit)
        # Crude funding proxy: 0.01% per 8h, average over hold time
        hold_hours = (pd.Timestamp(t["ExitTime"]) - pd.Timestamp(t["EntryTime"])).total_seconds() / 3600
        funding_periods = max(int(hold_hours / 8), 0)
        avg_notional = (notional_entry + notional_exit) / 2
        funding = 0.0001 * avg_notional * funding_periods * side  # long pays positive funding

        gross_pnl = side * (exit_price - entry_price) * qty
        net_pnl = gross_pnl - commission - funding

        equity += net_pnl
        fires += 1
        realized.append({
            "entry_time": t["EntryTime"].isoformat(),
            "exit_time": t["ExitTime"].isoformat(),
            "qty": qty,
            "entry": entry_price,
            "exit": exit_price,
            "side": side,
            "pnl": net_pnl,
            "equity_after": equity,
        })
        history.append({"ts": t["ExitTime"], "equity": equity})

        # Kill-switch check AFTER each closed trade. The trade is already
        # closed at exit price (we don't simulate intrabar lows separately —
        # the SL was already hit if it would have fired mid-trade, and the
        # exit_price in the trade record reflects that).
        if kill_floor > 0 and equity <= kill_floor:
            killed_at = pd.Timestamp(t["ExitTime"])
            killed_equity = equity
            break

    # Compute stats
    eq_series = pd.DataFrame(history).set_index("ts")["equity"]
    eq_series = eq_series[~eq_series.index.duplicated(keep="last")].sort_index()
    daily_eq = eq_series.resample("1D").last().ffill()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(365)) if daily_ret.std() > 0 else 0.0
    peak = daily_eq.cummax()
    max_dd_pct = float((daily_eq / peak - 1).min() * 100)
    ret_pct = (equity - start_equity) / start_equity * 100

    print(f"\n=== {label} @ ${start_equity:.2f} ===")
    print(f"  signals available: {len(trades)}")
    print(f"  fires: {fires}  ({fires/len(trades)*100:.1f}% of signals)")
    print(f"  skips: min_qty={skips_min_qty}  min_notional={skips_min_notional}  other={skips_other}")
    if killed_at is not None:
        print(f"  KILL-SWITCH TRIPPED on {killed_at.date()} at equity ${killed_equity:.2f}")
        print(f"  (skipped remaining {len(trades) - fires - skips_min_qty - skips_min_notional - skips_other} signals after trip)")
    print(f"  final equity:  ${equity:+.2f}  ({ret_pct:+.2f}%)")
    print(f"  Sharpe:        {sharpe:+.2f}")
    print(f"  max DD:        {max_dd_pct:+.2f}%")

    return {
        "label": label,
        "start_equity": start_equity,
        "final_equity": equity,
        "ret_pct": ret_pct,
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "n_signals": len(trades),
        "fires": fires,
        "skips_min_qty": skips_min_qty,
        "skips_min_notional": skips_min_notional,
        "skips_other": skips_other,
        "fire_rate_pct": fires / len(trades) * 100,
        "kill_switch_tripped": killed_at is not None,
        "kill_switch_date": str(killed_at.date()) if killed_at is not None else None,
        "kill_switch_equity": killed_equity,
        "realized_trades": realized,
        "_daily_eq": daily_eq,
    }


def main() -> int:
    ts = latest_ts()
    print(f"Using backtest trades from run {ts}")

    v1_trades = load_trades("v1", ts)
    d3_trades = load_trades("d3cons", ts)
    print(f"v1 trade signals: {len(v1_trades):,}")
    print(f"Donchian-cons signals: {len(d3_trades):,}")

    # CLI args
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--kill-frac", type=float, default=0.82,
                   help="Kill-switch equity fraction (0.82 = -18%, 0.645 = -35.5%, 0 = disabled)")
    p.add_argument("--risk-pct", type=float, default=0.02,
                   help="Risk per trade as fraction (0.02 = 2%, 0.03 = 3%, etc.)")
    p.add_argument("--weight-v1", type=float, default=0.5,
                   help="Weight allocated to v1 leg (0.5 = 50/50, 0.0 = Donchian only, 1.0 = v1 only)")
    args = p.parse_args()
    # Apply CLI overrides to global risk_pct (used by sizers)
    global V1_RISK_PCT, DON_RISK_PCT
    V1_RISK_PCT = args.risk_pct
    DON_RISK_PCT = args.risk_pct
    v1_start = START_COMBINED * args.weight_v1
    d3_start = START_COMBINED * (1.0 - args.weight_v1)
    print(f"\nKill-switch: {args.kill_frac} ({(1 - args.kill_frac)*100:.1f}% drawdown floor)")
    print(f"Risk per trade: {args.risk_pct*100:.1f}%")
    print(f"Weights: v1=${v1_start:.2f} ({args.weight_v1*100:.0f}%), Donchian=${d3_start:.2f} ({(1-args.weight_v1)*100:.0f}%)")

    # Skip legs with zero allocation
    if v1_start > 0:
        v1_res = simulate_leg(v1_trades, v1_start, v1_target_btc,
                              f"multifactor-v1 (15m, ${v1_start:.2f}, risk={args.risk_pct*100:.1f}%)",
                              kill_switch_frac=args.kill_frac)
    else:
        v1_res = {"label": "v1 (disabled)", "start_equity": 0, "final_equity": 0, "ret_pct": 0,
                  "sharpe": 0, "max_dd_pct": 0, "n_signals": 0, "fires": 0,
                  "skips_min_qty": 0, "skips_min_notional": 0, "skips_other": 0,
                  "fire_rate_pct": 0, "kill_switch_tripped": False, "kill_switch_date": None,
                  "kill_switch_equity": None, "realized_trades": [],
                  "_daily_eq": pd.Series([0.0], index=[pd.Timestamp("2019-09-10")])}
    if d3_start > 0:
        d3_res = simulate_leg(d3_trades, d3_start, don_target_btc,
                              f"Donchian-v3 cons (4h, ${d3_start:.2f}, risk={args.risk_pct*100:.1f}%)",
                              kill_switch_frac=args.kill_frac)
    else:
        d3_res = {"label": "d3 (disabled)", "start_equity": 0, "final_equity": 0, "ret_pct": 0,
                  "sharpe": 0, "max_dd_pct": 0, "n_signals": 0, "fires": 0,
                  "skips_min_qty": 0, "skips_min_notional": 0, "skips_other": 0,
                  "fire_rate_pct": 0, "kill_switch_tripped": False, "kill_switch_date": None,
                  "kill_switch_equity": None, "realized_trades": [],
                  "_daily_eq": pd.Series([0.0], index=[pd.Timestamp("2019-09-10")])}

    # Combined wallet — sum of legs on a common daily index
    combined_eq = (
        v1_res["_daily_eq"].reindex(
            v1_res["_daily_eq"].index.union(d3_res["_daily_eq"].index)
        ).ffill().fillna(v1_start)
        + d3_res["_daily_eq"].reindex(
            v1_res["_daily_eq"].index.union(d3_res["_daily_eq"].index)
        ).ffill().fillna(d3_start)
    )
    combined_eq = combined_eq.sort_index()
    combined_daily_ret = combined_eq.pct_change().dropna()
    combined_sharpe = (
        float(combined_daily_ret.mean() / combined_daily_ret.std() * math.sqrt(365))
        if combined_daily_ret.std() > 0 else 0.0
    )
    combined_peak = combined_eq.cummax()
    combined_dd = float((combined_eq / combined_peak - 1).min() * 100)
    combined_ret = (float(combined_eq.iloc[-1]) - START_COMBINED) / START_COMBINED * 100

    print(f"\n=== COMBINED wallet @ ${START_COMBINED:.2f} ($50.50 each leg) ===")
    print(f"  final wallet equity: ${combined_eq.iloc[-1]:+.2f} ({combined_ret:+.2f}%)")
    print(f"  Sharpe:        {combined_sharpe:+.2f}")
    print(f"  max DD:        {combined_dd:+.2f}%")
    print(f"  total fires:   v1={v1_res['fires']} + Donchian={d3_res['fires']} = {v1_res['fires'] + d3_res['fires']}")
    print(f"  total signals: {v1_res['n_signals'] + d3_res['n_signals']}")
    print(f"  overall fire rate: {(v1_res['fires'] + d3_res['fires']) / (v1_res['n_signals'] + d3_res['n_signals']) * 100:.1f}%")

    # Compare to upper-bound (proportional sizing)
    full_h_json = REPORTS / f"full_history_{ts}.json"
    if full_h_json.exists():
        ub = json.loads(full_h_json.read_text())
        v1_ub_end = START_PER_LEG * (1 + ub["v1"]["ret_pct"] / 100)
        d3_ub_end = START_PER_LEG * (1 + ub["d3cons"]["ret_pct"] / 100)
        co_ub_end = START_COMBINED * (1 + ub["combined_cons"]["ret_pct"] / 100)
        print("\n=== Upper-bound (proportional sizing, ignores min-qty) for comparison ===")
        print(f"  v1:              ${v1_ub_end:.2f} ({ub['v1']['ret_pct']:+.2f}%)")
        print(f"  Donchian-cons:   ${d3_ub_end:.2f} ({ub['d3cons']['ret_pct']:+.2f}%)")
        print(f"  combined:        ${co_ub_end:.2f} ({ub['combined_cons']['ret_pct']:+.2f}%)")

    # Persist
    out = {
        "ts": ts,
        "constants": {
            "min_qty_btc": MIN_QTY_BTC,
            "min_notional_usdt": MIN_NOTIONAL_USDT,
            "start_per_leg": START_PER_LEG,
            "start_combined": START_COMBINED,
            "kill_switch_frac": args.kill_frac,
        },
        "v1": {k: v for k, v in v1_res.items() if k != "_daily_eq"},
        "donchian_cons": {k: v for k, v in d3_res.items() if k != "_daily_eq"},
        "combined": {
            "start": START_COMBINED,
            "final": float(combined_eq.iloc[-1]),
            "ret_pct": combined_ret,
            "sharpe": combined_sharpe,
            "max_dd_pct": combined_dd,
            "total_fires": v1_res["fires"] + d3_res["fires"],
            "total_signals": v1_res["n_signals"] + d3_res["n_signals"],
        },
    }
    suffix = f"_k{int(args.kill_frac * 100)}" if args.kill_frac > 0 else "_nokill"
    out_path = REPORTS / f"realistic_50_50_{ts}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # Save the combined daily equity for HTML rendering
    combined_eq.to_frame("equity_usd").to_csv(REPORTS / f"realistic_50_50_{ts}{suffix}_combined.csv")
    v1_res["_daily_eq"].to_frame("equity_usd").to_csv(REPORTS / f"realistic_50_50_{ts}{suffix}_v1.csv")
    d3_res["_daily_eq"].to_frame("equity_usd").to_csv(REPORTS / f"realistic_50_50_{ts}{suffix}_d3cons.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
