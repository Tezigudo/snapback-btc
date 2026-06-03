"""Validate TODO_LEG: Taker-flow imbalance (candle-level CVD proxy).

5 OOS windows on BTC 15m at $1M cash, PRICE_SCALE=0.001 fractional sizing.

Requires `taker_buy_base` column in the cached 15m parquet. If missing, this
runner aborts loudly with DATA_BLOCKED — the parquet must be re-fetched after
patching exchange/data.py::load_klines (line 120) to preserve the column.

Outputs:
  - reports/_postfrac_taker_flow_<window>.csv  (per-window trades)
  - reports/postfrac_taker_flow.json           (aggregated result)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_taker_flow import TakerFlowImbalance  # noqa: E402
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
    equity_impact_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

CONFIG = {
    "tibs_period":         4,
    "tibs_threshold":      0.25,
    "ema_1h_period":       20,
    "atr_period":          14,
    "atr_sl_mult":         1.0,
    "atr_tp_mult":         2.0,
    "max_hold_bars":       16,
    "risk_per_trade_pct":  0.5,
    "allow_shorts":        True,
    "leverage":            20,
}

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]


def _check_data_or_block() -> tuple[bool, str]:
    """Return (ok, reason). If not ok, caller should write DATA_BLOCKED JSON."""
    if not PARQUET.exists():
        return False, f"parquet missing: {PARQUET}"
    df_head = pd.read_parquet(PARQUET).head(5)
    cols = [c.lower() for c in df_head.columns]
    if "taker_buy_base" not in cols and "takerbuybase" not in cols:
        return False, (
            f"parquet at {PARQUET.name} lacks taker_buy_base column "
            f"(has: {list(df_head.columns)}). "
            "Patch exchange/data.py line 120 to preserve taker_buy_base, "
            "then re-fetch the parquet."
        )
    return True, "ok"


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    # Normalize column case
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "volume":
            rename_map[c] = "Volume"
        elif cl in ("taker_buy_base", "takerbuybase"):
            rename_map[c] = "TakerBuyBase"
    df = df.rename(columns=rename_map)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sl) == 0:
        raise ValueError(f"Empty slice {start}..{end}")

    # Scale OHLC. Volume and TakerBuyBase stay raw — they are absolute base-asset units
    # and only their RATIO is used by the strategy.
    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * PRICE_SCALE
    return sl


def run_window(label: str, start: str, end: str) -> dict:
    df = _load_slice_scaled(start, end)
    bt = Backtest(
        df, TakerFlowImbalance,
        cash=CASH, commission=COMMISSION, margin=MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run(**CONFIG)

    n_trades = int(stats.get("# Trades", 0) or 0)
    ret_pct = float(stats.get("Return [%]", 0.0) or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)
    win_rate = float(stats.get("Win Rate [%]") or 0.0)
    equity_final = float(stats.get("Equity Final [$]", CASH) or CASH)

    trades_df = getattr(stats, "_trades", None)
    pnl_pct_list = []
    eq_impact_pnl_pct: list[float] = []
    if trades_df is not None and len(trades_df) > 0 and "ReturnPct" in trades_df.columns:
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        # CANONICAL (v2): sizing-aware equity-impact returns for per-window PSR.
        # Slice IS the OOS window (no warm-prefix) -> all trades count.
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
        out_csv = ROOT / "reports" / f"_postfrac_taker_flow_{label}.csv"
        out = pd.DataFrame({
            "pnl_pct": pnl_pct_list,
            "window_start": start,
            "window_end": end,
        })
        out.to_csv(out_csv, index=False)
        print(f"  [{label}] saved {len(out)} trades -> {out_csv.name}", file=sys.stderr)

    return {
        "label":             label,
        "start":             start,
        "end":               end,
        "trades":            n_trades,
        "return_pct":        round(ret_pct, 4),
        "max_dd_pct":        round(max_dd, 4),
        "win_rate_pct":      round(win_rate, 4),
        "equity_final":      round(equity_final, 4),
        "pnl_pct":           pnl_pct_list,
        "eq_impact_pnl_pct": eq_impact_pnl_pct,
    }


def aggregate(per_window: list[dict]) -> dict:
    all_pnl: list[float] = []
    n_trades = 0
    n_pos = 0
    compounded = 1.0
    per_win_ret = []
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        if r["return_pct"] > 0:
            n_pos += 1
    return {
        "n_trades":              n_trades,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{len(per_window)}",
        "per_window_return_pct": per_win_ret,
        "all_pnl_pct":           all_pnl,
    }


def _write_data_blocked(reason: str) -> None:
    payload = {
        "strategy_id":    "taker_flow",
        "strategy_class": "strategy.signals_taker_flow:TakerFlowImbalance",
        "verdict":        "DATA_BLOCKED",
        "reason":         reason,
        "patch_proposal": (
            "exchange/data.py line 120: change "
            "`return df.set_index('open_time')[['open','high','low','close','volume']]` "
            "to also include 'taker_buy_base' (and ideally 'taker_buy_quote'); "
            "then delete data/historical/BTC_USDT_USDT_15m.parquet and re-run "
            "`python -m exchange.data --symbol BTC/USDT:USDT --tf 15m --days 2400`."
        ),
        "next_step":      "Re-run tools/_postfrac_taker_flow.py once parquet has taker_buy_base column.",
    }
    out_path = ROOT / "reports" / "postfrac_taker_flow.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[postfrac_taker_flow] DATA_BLOCKED -> {out_path}", file=sys.stderr)


def main() -> int:
    t0 = time.time()
    ok, reason = _check_data_or_block()
    if not ok:
        print(f"[postfrac_taker_flow] DATA_BLOCKED: {reason}", file=sys.stderr)
        _write_data_blocked(reason)
        return 0  # exit clean — verdict is in the JSON

    per_window = []
    print("[postfrac_taker_flow] running 5 OOS windows ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        tw = time.time()
        r = run_window(label, start, end)
        print(
            f"  {label}  trades={r['trades']:4d}  ret={r['return_pct']:+8.2f}%  "
            f"dd={r['max_dd_pct']:+7.2f}%  win={r['win_rate_pct']:5.2f}%  "
            f"({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )
        per_window.append(r)

    agg = aggregate(per_window)
    all_pnl = agg.pop("all_pnl_pct")
    pnl_arr = np.asarray(all_pnl, dtype=float)
    # LEGACY stitched-per-trade PSR (N-inflated; observability only).
    legacy_psr_stitched = compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95) if len(pnl_arr) >= 2 else {
        "n_trades": int(len(pnl_arr)),
        "psr_vs_hurdle": 0.0,
        "interpretation": "insufficient_evidence",
    }

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    # PSR axis migrated from stitched per-trade ReturnPct to the equity-curve
    # window-level aggregation (psr_walkforward).
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)
    psr = canon["psr_walkforward"]  # canonical headline PSR

    # Gate decision — now reads the CANONICAL psr (was stitched pre-migration).
    compounded = agg["compounded_pct"]
    psr_val = float(psr.get("psr_vs_hurdle", 0.0) or 0.0)
    if psr_val > 0.95 and compounded > 0.0:
        verdict = "CANDIDATE_PROMOTE"
    elif psr_val > 0.80:
        verdict = "ITERATE"
    else:
        verdict = "SHELF"

    result = {
        "strategy_id":     "taker_flow",
        "strategy_class":  "strategy.signals_taker_flow:TakerFlowImbalance",
        "verdict":         verdict,
        "cash":            CASH,
        "commission":      COMMISSION,
        "margin":          MARGIN,
        "price_scale":     PRICE_SCALE,
        "config":          CONFIG,
        "windows":         [w[0] for w in WINDOWS],
        "per_window":      [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":              agg,
        "psr":                  psr,                  # canonical psr_walkforward
        "legacy_psr_stitched":  legacy_psr_stitched,  # observability only
        "canonical":            canon,                # v2 dual-emit block
        "aggregation_method":   canon["aggregation_method"],
        "elapsed_sec":          round(time.time() - t0, 2),
    }

    # --- bit-for-bit round-trip check (migration verification) --------------
    persisted = np.asarray(canon["per_window_return_pct"], dtype=float)
    recomputed = (
        compute_psr(persisted, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(persisted) >= 2
        else {"psr_vs_hurdle": 0.0}
    )
    assert recomputed.get("psr_vs_hurdle") == psr.get("psr_vs_hurdle"), (
        f"canonical PSR round-trip MISMATCH: recomputed="
        f"{recomputed.get('psr_vs_hurdle')} headline={psr.get('psr_vs_hurdle')}"
    )
    print(
        f"[postfrac_taker_flow] canonical PSR round-trip OK: "
        f"{psr.get('psr_vs_hurdle')}  verdict={verdict}",
        file=sys.stderr,
    )

    out_path = ROOT / "reports" / "postfrac_taker_flow.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[postfrac_taker_flow] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
