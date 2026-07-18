"""
AdaptiveTrendV1 backtest wrapper — runs the strategy AND applies realised
Binance perpetual funding to the equity curve as a post-process.

Why a separate runner: `tools/run_strategy_experiment.py` doesn't apply
funding, and AdaptiveTrend holds positions across 8h boundaries by design
(MOM-driven, trailing-stop exit).  Without funding, the verdict would be
biased upward on longs in the typical positive-funding regime.

Reuses the proven `backtest.funding_cost_for_trades()` for the signed
funding math — DO NOT re-implement here.

CLI:
    python tools/_adaptrend_run.py \\
        --start 2024-01-01 --end 2024-06-30 \\
        --cash 1000000 --label 2024_H1

Prints ONE JSON object to stdout containing both the gross (no-funding) and
net (funding-applied) numbers plus the funding event count.
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

from backtest import funding_cost_for_trades  # noqa: E402 — REUSE, don't reimplement
from strategy.signals_adaptive_trend import AdaptiveTrendV1  # noqa: E402

_DEFAULT_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
_DEFAULT_FUNDING = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
_COMMISSION = 0.0005   # 5 bps/side (matches divergence/adx smoke convention)
_MARGIN = 1.0 / 20     # 20x leverage ceiling


def _load_15m_slice(parquet: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if sliced.empty:
        raise ValueError(f"No 15m bars in [{start}, {end}].")
    return sliced


def _load_funding_slice(parquet: Path, start: str, end: str) -> pd.DataFrame:
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
) -> dict:
    config = config or {}
    data_15m = _load_15m_slice(parquet, start, end)
    funding = _load_funding_slice(funding_parquet, start, end)

    bt = Backtest(
        data_15m,
        AdaptiveTrendV1,
        cash=cash,
        commission=_COMMISSION,
        margin=_MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    print(f"[adaptrend] running {start}..{end} cash={cash} config={config}", file=sys.stderr)
    stats = bt.run(**config)

    trades_df = getattr(stats, "_trades", None)
    gross_return_pct = float(stats.get("Return [%]", 0.0))
    n_trades = int(stats.get("# Trades", 0))
    win_rate = float(stats.get("Win Rate [%]") or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]") or 0.0)
    sharpe = float(stats.get("Sharpe Ratio") or 0.0)

    funding_cost_usdt = 0.0
    funding_events = 0
    if trades_df is not None and len(trades_df) > 0 and not funding.empty:
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_df, data_15m, funding
        )

    gross_final_equity = cash * (1.0 + gross_return_pct / 100.0)
    net_final_equity = gross_final_equity - funding_cost_usdt
    net_return_pct = (net_final_equity / cash - 1.0) * 100.0

    if save_trades is not None and trades_df is not None and len(trades_df) > 0:
        out_cols = []
        for col in ("ReturnPct", "PnL", "EntryTime", "ExitTime", "Size"):
            if col in trades_df.columns:
                out_cols.append(col)
        out = trades_df[out_cols].copy()
        if "ReturnPct" in out.columns:
            out = out.rename(columns={"ReturnPct": "pnl_pct"})
            out["pnl_pct"] = out["pnl_pct"] * 100.0  # percent
        out["window_start"] = start
        out["window_end"] = end
        if label is not None:
            out["label"] = label
        header = not save_trades.exists()
        out.to_csv(save_trades, mode="a", index=False, header=header)
        print(f"[adaptrend] saved {len(out)} trades to {save_trades}", file=sys.stderr)

    return {
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
        "max_dd_pct": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
        "final_equity_gross": round(gross_final_equity, 2),
        "final_equity_net": round(net_final_equity, 2),
        "config_applied": config,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AdaptiveTrendV1 backtest with funding cost.")
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
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
