"""Re-validate AdaptiveTrendV1 (bare signal) after fractional-sizing refactor.

Runs 5 OOS windows + 11-window extended set at $1M cash with PRICE_SCALE=0.001
fractional sizing, aggregates per-trade pnl_pct, computes PSR, writes:
  - reports/_postfrac_adaptrend_v1_<window>.csv  (per-window trades)
  - reports/_postfrac_adaptrend_v1_aggregated.csv (all trades)
  - reports/postfrac_adaptrend_v1.json           (full result)

Mirrors tools/_postfrac_adaptrend_v1_volsize.py.

Config note:
The task asked for empty config. The pre-fix REFERENCE (67.09% / 563 trades,
PSR 0.894) was produced by `adaptive_trend_extended_psr.json` with
winner_config = {momentum_lookback_h6: 4, theta_entry: 0.02, alpha: 2.0}.
Code defaults are momentum_lookback_h6=4 / theta_entry=0.02 / alpha=2.5.
To compare apples-to-apples against the reference we must hold alpha=2.0 —
otherwise the "post-fix delta" mixes two effects (refactor + alpha change).
We pass {"alpha": 2.0} so the only thing changing vs the reference is the
fractional-sizing harness change.
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

from strategy.signals_adaptive_trend import AdaptiveTrendV1  # noqa: E402
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

# Pre-fix reference used alpha=2.0 (winner_config); defaults are alpha=2.5.
# Hold alpha=2.0 to match the reference baseline; otherwise the delta would
# conflate the fractional refactor with an alpha change.
CONFIG = {"alpha": 2.0}

WINDOWS_5 = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]

WINDOWS_11 = [
    ("2020_H2", "2020-07-01", "2020-12-31"),
    ("2021_H1", "2021-01-01", "2021-06-30"),
    ("2021_H2", "2021-07-01", "2021-12-31"),
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2022_H2", "2022-07-01", "2022-12-31"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2023_H2", "2023-07-01", "2023-12-31"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
    ("2025_H2", "2025-07-01", "2025-12-31"),
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


def run_window(label: str, start: str, end: str, csv_prefix: str) -> dict | None:
    df = _load_slice_scaled(start, end)
    if df is None:
        print(f"  [{label}] SKIP — no data in window", file=sys.stderr)
        return None
    bt = Backtest(
        df,
        AdaptiveTrendV1,
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
    pnl_pct_list: list[float] = []
    eq_impact_pnl_pct: list[float] = []
    if (
        trades_df is not None
        and len(trades_df) > 0
        and "ReturnPct" in trades_df.columns
    ):
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
        out_csv = ROOT / "reports" / f"{csv_prefix}_{label}.csv"
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


def run_set(set_label: str, windows: list[tuple[str, str, str]], csv_prefix: str) -> dict:
    per_window: list[dict] = []
    print(f"[postfrac_adaptrend_v1] running {set_label} ({len(windows)} windows) ...",
          file=sys.stderr)
    for label, start, end in windows:
        tw = time.time()
        r = run_window(label, start, end, csv_prefix)
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
    # LEGACY stitched-per-trade PSR (N-inflated; observability only).
    legacy_psr_stitched = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(pnl_arr) >= 2
        else {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    agg_csv = ROOT / "reports" / f"{csv_prefix}_aggregated.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
    print(f"  aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    # Headline PSR migrated from the N-inflated stitched per-trade ReturnPct
    # union to the equity-curve window-level aggregation (psr_walkforward).
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)
    psr = canon["psr_walkforward"]  # canonical headline PSR

    # --- bit-for-bit round-trip check (migration verification) --------------
    persisted = np.asarray(canon["per_window_return_pct"], dtype=float)
    recomputed = (
        compute_psr(persisted, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(persisted) >= 2
        else {"psr_vs_hurdle": 0.0}
    )
    assert recomputed.get("psr_vs_hurdle") == psr.get("psr_vs_hurdle"), (
        f"[{set_label}] canonical PSR round-trip MISMATCH: recomputed="
        f"{recomputed.get('psr_vs_hurdle')} headline={psr.get('psr_vs_hurdle')}"
    )
    print(
        f"  [{set_label}] canonical PSR round-trip OK: "
        f"{psr.get('psr_vs_hurdle')}",
        file=sys.stderr,
    )

    return {
        "set":        set_label,
        "per_window": [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":    agg,
        "psr":        psr,                    # canonical psr_walkforward
        "legacy_psr_stitched": legacy_psr_stitched,  # observability only
        "canonical":  canon,                  # v2 dual-emit block
        "aggregation_method": canon["aggregation_method"],
    }


def main() -> int:
    t0 = time.time()

    res_5 = run_set("5_OOS", WINDOWS_5, "_postfrac_adaptrend_v1")
    res_11 = run_set("11_OOS_extended", WINDOWS_11, "_postfrac_adaptrend_v1_ext")

    result = {
        "strategy_id":    "adaptrend_v1",
        "strategy_class": "strategy.signals_adaptive_trend:AdaptiveTrendV1",
        "cash":           CASH,
        "commission":     COMMISSION,
        "margin":         MARGIN,
        "price_scale":    PRICE_SCALE,
        "config":         CONFIG,
        "set_5_OOS":      res_5,
        "set_11_OOS_extended": res_11,
        "pre_fix": {
            "compounded_pct_11_oos": 67.09,
            "trades_11_oos":         563,
            "psr_11_oos":            0.894,
            "min_trl_11_oos":        976,
            "source": "reports/adaptive_trend_extended_psr.json + SESSION_SUMMARY_2026-06-01_to_06-02.html",
            "note": "Reference is GROSS of funding (run_strategy_experiment.py does not "
                    "apply funding; matches our post-fix basis). Reference winner_config "
                    "was {momentum_lookback_h6:4, theta_entry:0.02, alpha:2.0}; we pass "
                    "{alpha:2.0} so the only delta vs reference is the fractional refactor.",
        },
        "elapsed_sec":    round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_adaptrend_v1.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[postfrac_adaptrend_v1] wrote {out_path}  "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
