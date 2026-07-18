"""
AdaptiveTrendV2_funding_skip backtest wrapper — prefix-buffered like v2 base.

Mirrors tools/_adaptrend_v2_run.py but:
  1. Imports the +funding_skip subclass.
  2. Loads the funding series for [prefix_start, end] and attaches it to the
     strategy CLASS (`AdaptiveTrendV2_funding_skip.funding_series = ...`)
     before constructing Backtest.  The init() of the strategy then builds
     a per-15m-bar skip mask once.

Same OOS-trim, same funding cost post-process — so the result is directly
comparable to v2's net_return_pct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from backtesting import Backtest  # noqa: E402

from backtest import funding_cost_for_trades  # noqa: E402
from strategy.signals_adaptive_trend_v2_funding_skip import (  # noqa: E402
    AdaptiveTrendV2_funding_skip,
)

_DEFAULT_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
_DEFAULT_FUNDING = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
_COMMISSION = 0.0005
_MARGIN = 1.0 / 20


def _load_15m_with_prefix(
    parquet: Path, start: str, end: str, prefix_months: int
) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    prefix_start = start_ts - pd.DateOffset(months=prefix_months)
    sliced = df.loc[(df.index >= prefix_start) & (df.index <= end_ts)].copy()
    if sliced.empty:
        raise ValueError(f"No 15m bars in [{prefix_start}, {end}].")
    return sliced, start_ts


def _load_funding_slice(
    parquet: Path, start: str, end: str, prefix_months: int
) -> pd.DataFrame:
    """Funding slice for FILTER input — must include the prefix so the strategy
    can look up 'most recent funding < bar ts' for early bars.  For the
    POST-PROCESS funding cost (different call site below), we still trim to
    OOS only — that math is per-trade and v1/v2 already use the trimmed slice.
    """
    f = pd.read_parquet(parquet)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    start_ts = pd.Timestamp(start) - pd.DateOffset(months=prefix_months)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return f.loc[(f.index >= start_ts) & (f.index <= end_ts)].copy()


def _load_funding_slice_oos(parquet: Path, start: str, end: str) -> pd.DataFrame:
    f = pd.read_parquet(parquet)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return f.loc[(f.index >= start_ts) & (f.index <= end_ts)].copy()


def run(
    start: str,
    end: str,
    cash: float,
    config: dict | None = None,
    save_trades: Path | None = None,
    parquet: Path = _DEFAULT_PARQUET,
    funding_parquet: Path = _DEFAULT_FUNDING,
    label: str | None = None,
    return_monthly_choices: bool = False,
) -> dict:
    config = config or {}
    prefix_months = int(config.get("fit_window_months", 6))

    data_15m_full, oos_start_ts = _load_15m_with_prefix(parquet, start, end, prefix_months)
    funding_full = _load_funding_slice(funding_parquet, start, end, prefix_months)
    funding_oos = _load_funding_slice_oos(funding_parquet, start, end)

    # Inject funding series into strategy class BEFORE constructing Backtest.
    # backtesting.py snapshots class attrs at Backtest() __init__, so this
    # must precede the Backtest() call.
    funding_series = (
        funding_full["funding_rate"]
        if "funding_rate" in funding_full.columns
        else funding_full.iloc[:, 0]
    )
    AdaptiveTrendV2_funding_skip.funding_series = funding_series

    bt = Backtest(
        data_15m_full,
        AdaptiveTrendV2_funding_skip,
        cash=cash,
        commission=_COMMISSION,
        margin=_MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    run_kwargs = dict(config)
    run_kwargs.setdefault("trade_start_ns", oos_start_ts.value)
    print(
        f"[adaptrend_v2_fs] {start}..{end} prefix={prefix_months}mo "
        f"bars={len(data_15m_full)} funding={len(funding_series)} cash={cash}",
        file=sys.stderr,
    )
    stats = bt.run(**run_kwargs)

    trades_df = getattr(stats, "_trades", None)

    if trades_df is not None and len(trades_df) > 0:
        if "EntryTime" in trades_df.columns:
            trades_oos = trades_df[trades_df["EntryTime"] >= oos_start_ts].copy()
        else:
            trades_oos = trades_df.copy()
    else:
        trades_oos = trades_df

    n_trades = int(len(trades_oos)) if trades_oos is not None else 0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        gross_pnl_usdt = float(trades_oos["PnL"].sum())
    else:
        gross_pnl_usdt = 0.0
    gross_return_pct = gross_pnl_usdt / cash * 100.0

    win_rate = 0.0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        win_rate = float((trades_oos["PnL"] > 0).mean() * 100.0)

    funding_cost_usdt = 0.0
    funding_events = 0
    if trades_oos is not None and len(trades_oos) > 0 and not funding_oos.empty:
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_oos, data_15m_full, funding_oos
        )

    gross_final_equity = cash * (1.0 + gross_return_pct / 100.0)
    net_final_equity = gross_final_equity - funding_cost_usdt
    net_return_pct = (net_final_equity / cash - 1.0) * 100.0

    max_dd_pct = 0.0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        ordered = trades_oos.sort_values("ExitTime") if "ExitTime" in trades_oos.columns else trades_oos
        equity = cash + ordered["PnL"].cumsum()
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max * 100.0
        max_dd_pct = float(dd.min()) if len(dd) > 0 else 0.0

    # Filter diagnostic.
    strategy_inst = getattr(stats, "_strategy", None)
    n_skipped = int(getattr(strategy_inst, "_n_skipped_by_funding", 0)) if strategy_inst else 0

    monthly_choices = []
    if strategy_inst is not None and hasattr(strategy_inst, "monthly_choices"):
        for c in strategy_inst.monthly_choices:
            monthly_choices.append({
                "month_start": str(c.month_start),
                "L": c.L,
                "theta": c.theta,
                "n_fit_trades": c.n_fit_trades,
                "fit_sharpe": c.fit_sharpe,
                "reason": c.reason,
            })

    if save_trades is not None and trades_oos is not None and len(trades_oos) > 0:
        out_cols = []
        for col in ("ReturnPct", "PnL", "EntryTime", "ExitTime", "Size"):
            if col in trades_oos.columns:
                out_cols.append(col)
        out = trades_oos[out_cols].copy()
        if "ReturnPct" in out.columns:
            out = out.rename(columns={"ReturnPct": "pnl_pct"})
            out["pnl_pct"] = out["pnl_pct"] * 100.0
        out["window_start"] = start
        out["window_end"] = end
        if label is not None:
            out["label"] = label
        header = not save_trades.exists()
        out.to_csv(save_trades, mode="a", index=False, header=header)
        print(f"[adaptrend_v2_fs] saved {len(out)} trades to {save_trades}", file=sys.stderr)

    result = {
        "label": label,
        "start": start,
        "end": end,
        "cash": cash,
        "trades": n_trades,
        "gross_return_pct": round(gross_return_pct, 4),
        "net_return_pct": round(net_return_pct, 4),
        "funding_cost_usdt": round(funding_cost_usdt, 2),
        "funding_events": funding_events,
        "win_rate_pct": round(win_rate, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "final_equity_gross": round(gross_final_equity, 2),
        "final_equity_net": round(net_final_equity, 2),
        "config_applied": config,
        "prefix_months_used": prefix_months,
        "n_skipped_by_funding": n_skipped,
    }
    if return_monthly_choices:
        result["monthly_choices"] = monthly_choices
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AdaptiveTrendV2 + funding_skip backtest.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--config-json", default="{}")
    p.add_argument("--label", default=None)
    p.add_argument("--save-trades", default=None)
    p.add_argument("--parquet", default=str(_DEFAULT_PARQUET))
    p.add_argument("--funding-parquet", default=str(_DEFAULT_FUNDING))
    args = p.parse_args(argv)

    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --config-json not valid JSON: {exc}", file=sys.stderr)
        return 1

    result = run(
        start=args.start,
        end=args.end,
        cash=args.cash,
        config=config,
        save_trades=Path(args.save_trades) if args.save_trades else None,
        parquet=Path(args.parquet),
        funding_parquet=Path(args.funding_parquet),
        label=args.label,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
