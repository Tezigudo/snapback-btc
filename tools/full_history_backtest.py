"""Run a single continuous backtest from data start (~2020-04) → today for
multifactor-v1, Donchian-v3 (cons), Donchian-v3 (agg). Save full trade lists
+ equity curves so we can render TRADING_HISTORY-style HTML.

Output:
  reports/full_history_<UTC>.json   — per-strategy stats + trades + monthly returns
  reports/full_history_<UTC>_v1_equity.csv
  reports/full_history_<UTC>_d3cons_equity.csv
  reports/full_history_<UTC>_d3agg_equity.csv
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

SYMBOL = "BTC/USDT:USDT"
START = datetime(2019, 9, 10, tzinfo=UTC)   # Binance Futures BTC/USDT launch + 2d funding
END = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def donchian_params(combo: str) -> StrategyParams:
    p = StrategyParams.from_yaml()
    if combo == "agg":
        return dataclasses.replace(
            p,
            donchian_period_entry=40, donchian_period_exit=10,
            atr_sl_multiple=1.5, atr_tp_multiple=1.5, atr_trail_multiple=0.0,
            leverage=20, regime_ema_period=120, regime_slope_window=30,
            slope_trend_threshold_pct=0.0, time_stop_bars=48,
            volume_multiple=1.0,
        )
    if combo == "cons":
        return dataclasses.replace(
            p,
            donchian_period_entry=80, donchian_period_exit=20,
            atr_sl_multiple=1.5, atr_tp_multiple=1.5, atr_trail_multiple=0.0,
            leverage=20, regime_ema_period=120, regime_slope_window=30,
            slope_trend_threshold_pct=0.03, time_stop_bars=48,
            volume_multiple=1.0,
        )
    raise ValueError(combo)


def trades_to_records(trades_df: pd.DataFrame) -> list[dict]:
    if trades_df is None or trades_df.empty:
        return []
    cols = [c for c in trades_df.columns if c in (
        "EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size",
        "PnL", "ReturnPct", "Duration",
    )]
    out = []
    for _, t in trades_df[cols].iterrows():
        rec = {}
        for c in cols:
            v = t[c]
            if isinstance(v, pd.Timestamp):
                rec[c] = v.isoformat()
            elif isinstance(v, pd.Timedelta):
                rec[c] = str(v)
            elif isinstance(v, (np.integer, np.floating)):
                rec[c] = float(v)
            else:
                rec[c] = v
        out.append(rec)
    return out


def monthly_returns(equity: pd.Series) -> dict:
    """Per-calendar-month percentage change of normalised equity."""
    norm = equity / float(equity.iloc[0])
    last_per_month = norm.resample("ME").last().ffill()
    monthly = last_per_month.pct_change()
    monthly.iloc[0] = last_per_month.iloc[0] - 1.0   # first month vs start
    return {ts.strftime("%Y-%m"): float(v) for ts, v in monthly.items() if pd.notna(v)}


def run_one(label: str, strategy: str, tf: str, params: StrategyParams | None) -> dict:
    print(f"\n=== {label} ({strategy}, tf={tf}, {START.date()} → {END.date()}) ===")
    r = run_backtest(
        strategy_name=strategy,
        symbol=SYMBOL,
        timeframe=tf,
        start=START,
        end=END,
        params_override=params,
        quiet=True,
        return_equity=True,
        return_trades=True,
    )
    eq = r["equity_series"]
    norm = eq / float(eq.iloc[0])
    print(
        f"  trades={r['trades']} "
        f"ret={r.get('after_funding_pct', r['backtest_return_pct']):+.2f}% "
        f"sharpe={r.get('sharpe', float('nan')):+.2f} "
        f"max_dd={r.get('max_drawdown_pct', float('nan')):.2f}% "
        f"win_rate={r.get('win_rate_pct', float('nan')):.1f}%"
    )

    trades = trades_to_records(r.get("trades_df"))
    monthly = monthly_returns(eq)
    return {
        "label": label,
        "strategy": strategy,
        "timeframe": tf,
        "start": START.date().isoformat(),
        "end": END.date().isoformat(),
        "trades": r["trades"],
        "ret_pct": r.get("after_funding_pct", r["backtest_return_pct"]),
        "naive_return_pct": r.get("naive_return_pct"),
        "sharpe": r.get("sharpe"),
        "max_dd_pct": r.get("max_drawdown_pct"),
        "win_rate_pct": r.get("win_rate_pct"),
        "profit_factor": r.get("profit_factor"),
        "avg_trade_pct": r.get("avg_trade_pct"),
        "funding_cost_usdt": r.get("funding_cost_usdt"),
        "trade_records": trades,
        "monthly_returns": monthly,
        "_norm": norm,
    }


def main() -> int:
    out = {}

    v1 = run_one("multifactor-v1", "multifactor-v1", "15m", None)
    cons = run_one("Donchian-v3 cons", "donchian-v3", "4h", donchian_params("cons"))
    agg = run_one("Donchian-v3 agg", "donchian-v3", "4h", donchian_params("agg"))

    # Combined 50/50 on common daily index
    def daily(n):
        return n.resample("1D").last().ffill()

    v1_d = daily(v1["_norm"])
    cc_d = daily(cons["_norm"])
    cg_d = daily(agg["_norm"])
    common = v1_d.index.intersection(cc_d.index).intersection(cg_d.index)
    v1_d = v1_d.loc[common]
    cc_d = cc_d.loc[common]
    cg_d = cg_d.loc[common]

    combo_cons = 0.5 * v1_d + 0.5 * cc_d
    combo_agg = 0.5 * v1_d + 0.5 * cg_d

    def stats_from_eq(eq: pd.Series) -> dict:
        dr = eq.pct_change().dropna()
        peak = eq.cummax()
        dd_min = float((eq / peak - 1.0).min() * 100.0)
        sharpe = float(dr.mean() / dr.std() * math.sqrt(365)) if dr.std() > 0 else 0.0
        return {
            "ret_pct": float((eq.iloc[-1] - 1.0) * 100.0),
            "sharpe": sharpe,
            "max_dd_pct": dd_min,
        }

    daily_corr = float(v1_d.pct_change().dropna().corr(cc_d.pct_change().dropna()))
    print(f"\n=== combined daily corr v1↔Donchian-cons = {daily_corr:+.3f} ===")
    print("combined-cons:", stats_from_eq(combo_cons))
    print("combined-agg: ", stats_from_eq(combo_agg))

    # Persist
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    # Save normalized equity series for HTML render
    for name, series in (("v1", v1["_norm"]),
                        ("d3cons", cons["_norm"]),
                        ("d3agg", agg["_norm"]),
                        ("combo_cons", combo_cons),
                        ("combo_agg", combo_agg)):
        df_eq = pd.DataFrame({"equity_norm": series})
        df_eq.to_csv(ROOT / "reports" / f"full_history_{ts}_{name}_equity.csv")

    # Drop _norm from JSON dump (it's a series)
    for d in (v1, cons, agg):
        d.pop("_norm", None)

    out = {
        "ts": ts,
        "symbol": SYMBOL,
        "window": {"start": START.date().isoformat(), "end": END.date().isoformat()},
        "v1": v1,
        "d3cons": cons,
        "d3agg": agg,
        "combined_cons": {**stats_from_eq(combo_cons), "daily_corr_v1_d3": daily_corr},
        "combined_agg":  {**stats_from_eq(combo_agg),  "daily_corr_v1_d3": float(v1_d.pct_change().dropna().corr(cg_d.pct_change().dropna()))},
    }
    json_path = ROOT / "reports" / f"full_history_{ts}.json"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {json_path} ({json_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
