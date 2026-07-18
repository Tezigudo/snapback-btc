"""
Reusable single-window backtest runner for any backtesting.py Strategy class.

Usage:
    python tools/run_strategy_experiment.py \\
        --strategy-class strategy.signals_adx_dual_regime:ADXDualRegimeV1 \\
        --config-json '{"adx_chop_threshold": 20}' \\
        --start 2024-01-01 --end 2024-06-30

The default strategy class (for back-compat) is DivergenceV1, so callers that
previously called run_divergence_experiment.py can switch without changing args.

Outputs ONE JSON object to stdout. All progress / warnings go to stderr.
Exit 0 on success (including 0-trade windows).

Output shape:
    {
        "trades":           int,
        "total_return_pct": float,
        "win_rate_pct":     float,   # 0..100
        "max_dd_pct":       float,   # negative
        "sharpe":           float,   # 0.0 if undefined
        "equity_final":     float,
        "config_applied":   {…}
    }
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from backtesting import Backtest  # noqa: E402

_DEFAULT_STRATEGY = "strategy.signals_divergence:DivergenceV1"
_DEFAULT_PARQUET   = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
# NOTE: previous default was $10_000.0 — at BTC ~$70k with 1% risk and ATR-based sizing,
# int(risk_amount / sl_distance) truncates to 0 for most bars → signals silently dropped.
# Raised to $100_000.0 (realistic mid-size retail account, avoids integer truncation floor).
_CASH_DEFAULT = 100_000.0
_COMMISSION = 0.0005   # 5 bps (matches smoke check in signals_divergence.py)
_MARGIN     = 1.0 / 20  # 20x leverage ceiling (mirrors smoke check margin kwarg)


def _load_strategy_class(spec: str):
    """Load a strategy class from a 'module:ClassName' spec string.

    Examples
    --------
    _load_strategy_class("strategy.signals_divergence:DivergenceV1")
    _load_strategy_class("strategy.signals_adx_dual_regime:ADXDualRegimeV1")
    """
    if ":" not in spec:
        raise ValueError(
            f"--strategy-class must be in 'module:ClassName' format, got: {spec!r}"
        )
    module_path, class_name = spec.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Cannot import module '{module_path}': {exc}") from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ImportError(
            f"Module '{module_path}' has no attribute '{class_name}'"
        )
    return cls


def _load_slice(parquet: Path, start: str, end: str) -> pd.DataFrame:
    """Load parquet and slice [start, end] inclusive (full end-day included)."""
    print(f"[run_strategy] loading {parquet.name}", file=sys.stderr)
    df = pd.read_parquet(parquet)

    # backtesting.py wants capitalised column names
    df = df.rename(columns={c: c.capitalize() for c in df.columns})

    # Normalise index timezone: strip to naive UTC so backtesting.py doesn't
    # complain (it doesn't accept tz-aware indices on some versions).
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Include the full end day: "2024-06-30" → up to (but not including) "2024-07-01"
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    n = len(sliced)
    print(
        f"[run_strategy] slice {start} → {end}: {n} bars "
        f"({sliced.index[0] if n else 'empty'} … {sliced.index[-1] if n else 'empty'})",
        file=sys.stderr,
    )
    if n == 0:
        raise ValueError(f"No bars found in [{start}, {end}] — check date range and parquet coverage.")
    return sliced


def run(
    config: dict,
    start: str,
    end: str,
    strategy_class=None,
    parquet: Path = _DEFAULT_PARQUET,
    cash: float = _CASH_DEFAULT,
) -> dict:
    """Run one backtest window and return a result dict.

    Parameters
    ----------
    config:
        Dict of strategy class-attr overrides (same names as the attrs).
        Empty dict → all defaults.
    start, end:
        ISO date strings, inclusive.
    strategy_class:
        A backtesting.py Strategy subclass. If None, defaults to DivergenceV1.
    parquet:
        Path to the 15m OHLCV parquet file.
    """
    if strategy_class is None:
        from strategy.signals_divergence import DivergenceV1
        strategy_class = DivergenceV1

    df = _load_slice(parquet, start, end)

    bt = Backtest(
        df,
        strategy_class,
        cash=cash,
        commission=_COMMISSION,
        margin=_MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )

    print(
        f"[run_strategy] running {strategy_class.__name__} with config={config}",
        file=sys.stderr,
    )
    stats = bt.run(**config)

    def _safe_float(val, default: float = 0.0) -> float:
        """Return float(val) if finite, else default."""
        import math
        try:
            v = float(val)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    n_trades      = int(stats.get("# Trades", 0))
    total_ret_pct = _safe_float(stats.get("Return [%]", 0.0))
    win_rate_pct  = _safe_float(stats.get("Win Rate [%]"))
    max_dd_pct    = _safe_float(stats.get("Max. Drawdown [%]"))
    sharpe        = _safe_float(stats.get("Sharpe Ratio"))
    equity_final  = _safe_float(stats.get("Equity Final [$]", cash), cash)

    # Optionally extract per-trade pnl_pct from bt._results._trades
    trades_df = None
    try:
        raw_trades = getattr(stats, "_trades", None)
        if raw_trades is not None and len(raw_trades) > 0:
            trades_df = raw_trades.copy()
    except Exception:
        pass

    return {
        "trades":           n_trades,
        "total_return_pct": round(total_ret_pct, 4),
        "win_rate_pct":     round(win_rate_pct, 4),
        "max_dd_pct":       round(max_dd_pct, 4),
        "sharpe":           round(sharpe, 4),
        "equity_final":     round(equity_final, 4),
        "config_applied":   config,
        "_trades_df":       trades_df,  # internal use — not serialized by default
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single backtest window for any backtesting.py Strategy class "
            "and print JSON to stdout."
        )
    )
    parser.add_argument(
        "--strategy-class",
        default=_DEFAULT_STRATEGY,
        help=(
            "Strategy class in 'module:ClassName' format. "
            f"Default: {_DEFAULT_STRATEGY!r} (back-compat with run_divergence_experiment.py)"
        ),
    )
    parser.add_argument(
        "--config-json",
        default="{}",
        help="JSON string of strategy class-attr overrides, e.g. '{\"swing_k\": 5}'",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Window start date (inclusive), ISO format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        default="2022-06-30",
        help="Window end date (inclusive), ISO format YYYY-MM-DD",
    )
    parser.add_argument(
        "--parquet",
        default=str(_DEFAULT_PARQUET),
        help="Path to 15m OHLCV parquet file",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=_CASH_DEFAULT,
        help=(
            "Starting cash for the backtest. Default: 100_000. "
            "DEPRECATED: previous default was 10_000, which caused integer truncation "
            "in _position_units() at BTC ~$70k, silently dropping most trade signals."
        ),
    )
    parser.add_argument(
        "--save-trades",
        default=None,
        metavar="PATH",
        help=(
            "If given, save per-trade P&L to this CSV path (appending if the file exists). "
            "Columns: window_start, window_end, pnl_pct (one row per closed trade). "
            "Suitable for piping into tools/psr_eval.py."
        ),
    )
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --config-json is not valid JSON: {exc}", file=sys.stderr)
        return 1

    try:
        strategy_cls = _load_strategy_class(args.strategy_class)
    except (ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    result = run(
        config=config,
        start=args.start,
        end=args.end,
        strategy_class=strategy_cls,
        parquet=Path(args.parquet),
        cash=args.cash,
    )

    # Optionally save per-trade P&L
    if args.save_trades is not None:
        trades_df = result.pop("_trades_df", None)
        if trades_df is not None and len(trades_df) > 0:
            import os
            save_path = Path(args.save_trades)
            # Build a minimal per-trade CSV with pnl_pct
            try:
                # backtesting.py trades df has 'ReturnPct' column (decimal, e.g. 0.03 = 3%)
                pnl_col = None
                for col in ["ReturnPct", "PnL", "return_pct", "pnl_pct"]:
                    if col in trades_df.columns:
                        pnl_col = col
                        break
                if pnl_col is not None:
                    out = trades_df[[pnl_col]].copy()
                    if pnl_col == "ReturnPct":
                        out = out.rename(columns={"ReturnPct": "pnl_pct"})
                        out["pnl_pct"] = out["pnl_pct"] * 100.0  # convert to percent
                    out["window_start"] = args.start
                    out["window_end"] = args.end
                    header = not os.path.exists(save_path)
                    out.to_csv(save_path, mode="a", index=False, header=header)
                    print(
                        f"[run_strategy] saved {len(out)} trades to {save_path}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"[run_strategy] WARNING: could not save trades: {exc}", file=sys.stderr)
    else:
        result.pop("_trades_df", None)

    # Single JSON object on stdout, nothing else.
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
