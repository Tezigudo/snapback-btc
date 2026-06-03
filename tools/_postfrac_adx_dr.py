"""Re-validate ADXDualRegimeV1 (shelved sanity) after fractional-sizing refactor.

Runs 5 OOS windows at $1M cash with PRICE_SCALE=0.001 fractional sizing,
empty CONFIG (pure class defaults: leverage=5, risk_per_trade_pct=1.0,
adx_chop_threshold=25, use_donchian_retest=True), aggregates per-trade
pnl_pct, computes PSR, writes:
  - reports/_postfrac_adx_dr_<window>.csv (per-window trades)
  - reports/_postfrac_adx_dr_aggregated.csv (all trades)
  - reports/postfrac_adx_dr.json (full result)

Mirrors tools/_postfrac_adaptrend_v1_volsize.py.

Pre-fix reference (int-truncation $100k): compounded -99.23%, trades 3627.
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

from strategy.signals_adx_dual_regime import ADXDualRegimeV1  # noqa: E402
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

# Task spec: empty config + the overrides in {}.  No overrides → empty.
CONFIG: dict = {}

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame | None:
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sl) == 0:
        return None

    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * PRICE_SCALE
    return sl


def run_window(label: str, start: str, end: str) -> dict | None:
    df = _load_slice_scaled(start, end)
    if df is None:
        print(f"  [{label}] SKIP — no data in window", file=sys.stderr)
        return None
    bt = Backtest(
        df,
        ADXDualRegimeV1,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
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
    if (
        trades_df is not None
        and len(trades_df) > 0
        and "ReturnPct" in trades_df.columns
    ):
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
        out_csv = ROOT / "reports" / f"_postfrac_adx_dr_{label}.csv"
        out = pd.DataFrame(
            {
                "pnl_pct": pnl_pct_list,
                "window_start": start,
                "window_end": end,
            }
        )
        out.to_csv(out_csv, index=False)

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


def main() -> int:
    t0 = time.time()
    per_window: list[dict] = []
    print("[postfrac_adx_dr] running 5 OOS windows ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        tw = time.time()
        r = run_window(label, start, end)
        if r is None:
            continue
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
    psr = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(pnl_arr) >= 2
        else {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    agg_csv = ROOT / "reports" / "_postfrac_adx_dr_aggregated.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
    print(f"[postfrac_adx_dr] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    # Canonical v2 dual-emit (methodology debt #1): headline PSR comes from the
    # equity-curve aggregation (per-window return-series -> psr_walkforward),
    # NOT the N-inflated stitched per-trade ReturnPct union. Legacy stitched PSR
    # is kept as `psr` + inside canon["legacy_psr_stitched"] for observability.
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)

    result = {
        "strategy_id":    "adx_dr",
        "strategy_class": "strategy.signals_adx_dual_regime:ADXDualRegimeV1",
        "cash":           CASH,
        "commission":     COMMISSION,
        "margin":         MARGIN,
        "price_scale":    PRICE_SCALE,
        "config":         CONFIG,
        "windows":        [w[0] for w in WINDOWS],
        "per_window":     [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":        agg,
        "psr":            psr,            # legacy stitched (observability only)
        "canonical":      canon,          # v2 dual-emit (headline PSR = psr_walkforward)
        "aggregation_method": canon["aggregation_method"],
        "pre_fix": {
            "compounded_pct": -99.23,
            "trades":         3627,
            "note": "pre-fix PSR not recorded; -99.23% blowup implies effectively no edge "
                    "(treat as ~0 for comparison)",
        },
        "elapsed_sec":    round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_adx_dr.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[postfrac_adx_dr] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
