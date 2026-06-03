"""Re-test funding_skip on AdaptiveTrendV1 (POST fractional-sizing refactor).

Question:
  Does the funding_skip filter help V1 once fractional sizing is enabled?
  V2 + funding_skip (with the int() truncation bug) ran -4.42pp delta. With
  fractional sizing (PRICE_SCALE=0.001 milli-BTC units), does V1 see the
  same penalty, neutral, or an improvement?

Design (per advisor):
  - BOTH arms (base + fs) run inside one harness on identical basis.
  - BOTH arms apply funding_cost_for_trades post-process. Delta is net-vs-net,
    matching V2's prior comparison framing.
  - BOTH arms use PRICE_SCALE=0.001 (milli-BTC fractional sizing). This is the
    *post-fractional-fix* basis — V1 reports the same number it would on the
    pre-fix harness because V1 already runs at PRICE_SCALE=0.001 in the
    rebaseline; the fix changed V2/MF, not V1.
  - 11 OOS windows from the V2 funding_skip benchmark.
  - $1,000,000 cash per window.
  - alpha = 2.0 (matches V1 baseline post-fix config).
  - funding_skip params kept at module defaults (threshold 0.0005, window 30 min).

Funding handling:
  - The funding parquet is loaded with 6 months of prefix history for the
    FILTER input (so `searchsorted` finds a valid prior funding tick for the
    very first OOS bar). The strategy is wired with this prefix-included
    series.
  - For the POST-PROCESS cost computation, the OOS-only funding slice is used
    (mirrors _adaptrend_v2_funding_skip_run.py).
  - Trades use the OOS slice (trade entry/exit times are inside the window).

Output:
  reports/postfrac_adaptrendV1_imp_funding_skip.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from backtesting import Backtest  # noqa: E402

from backtest import funding_cost_for_trades  # noqa: E402
from strategy.signals_adaptive_trend import AdaptiveTrendV1  # noqa: E402
from strategy.signals_adaptive_trend_v1_funding_skip import (  # noqa: E402
    AdaptiveTrendV1_funding_skip,
)
from tools.aggregate import equity_impact_returns  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
FUNDING_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

CONFIG_V1 = {"alpha": 2.0}

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

FUNDING_PREFIX_MONTHS = 6  # filter lookback for first-bar safety


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------


def _load_window_scaled(start: str, end: str) -> pd.DataFrame | None:
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


def _load_funding_full(start: str, end: str) -> pd.DataFrame:
    f = pd.read_parquet(FUNDING_PARQUET)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    start_ts = pd.Timestamp(start) - pd.DateOffset(months=FUNDING_PREFIX_MONTHS)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return f.loc[(f.index >= start_ts) & (f.index <= end_ts)].copy()


def _load_funding_oos(start: str, end: str) -> pd.DataFrame:
    f = pd.read_parquet(FUNDING_PARQUET)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return f.loc[(f.index >= start_ts) & (f.index <= end_ts)].copy()


# ----------------------------------------------------------------------------
# Backtest helpers
# ----------------------------------------------------------------------------


def _run_one(
    strategy_cls,
    df_scaled: pd.DataFrame,
    cash: float,
    config: dict,
) -> dict:
    bt = Backtest(
        df_scaled,
        strategy_cls,
        cash=cash,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(**config)
    trades_df = getattr(stats, "_trades", None)
    strategy_inst = getattr(stats, "_strategy", None)
    return {
        "stats": stats,
        "trades": trades_df,
        "strategy": strategy_inst,
    }


def _summarize_run(
    label: str,
    arm: str,
    start: str,
    end: str,
    cash: float,
    df_scaled: pd.DataFrame,
    trades_df: pd.DataFrame | None,
    funding_oos: pd.DataFrame,
) -> dict:
    """Compute net-of-funding return + per-trade pnl_pct list."""
    n_trades = int(len(trades_df)) if trades_df is not None else 0

    if trades_df is None or len(trades_df) == 0:
        return {
            "label": label,
            "arm": arm,
            "start": start,
            "end": end,
            "trades": 0,
            "gross_return_pct": 0.0,
            "net_return_pct": 0.0,
            "funding_cost_usdt": 0.0,
            "funding_events": 0,
            "win_rate_pct": 0.0,
            "max_dd_pct": 0.0,
            "pnl_pct": [],
        }

    gross_pnl_usdt = float(trades_df["PnL"].sum()) if "PnL" in trades_df.columns else 0.0
    gross_return_pct = gross_pnl_usdt / cash * 100.0

    win_rate = float((trades_df["PnL"] > 0).mean() * 100.0) if "PnL" in trades_df.columns else 0.0

    funding_cost_usdt = 0.0
    funding_events = 0
    if not funding_oos.empty:
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_df, df_scaled, funding_oos
        )

    gross_final_equity = cash * (1.0 + gross_return_pct / 100.0)
    net_final_equity = gross_final_equity - funding_cost_usdt
    net_return_pct = (net_final_equity / cash - 1.0) * 100.0

    max_dd_pct = 0.0
    if "PnL" in trades_df.columns:
        ordered = (
            trades_df.sort_values("ExitTime")
            if "ExitTime" in trades_df.columns
            else trades_df
        )
        equity = cash + ordered["PnL"].cumsum()
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max * 100.0
        max_dd_pct = float(dd.min()) if len(dd) > 0 else 0.0

    pnl_list: list[float] = []
    if "ReturnPct" in trades_df.columns:
        pnl_list = (trades_df["ReturnPct"].values * 100.0).tolist()

    return {
        "label": label,
        "arm": arm,
        "start": start,
        "end": end,
        "trades": n_trades,
        "gross_return_pct": round(gross_return_pct, 4),
        "net_return_pct": round(net_return_pct, 4),
        "funding_cost_usdt": round(funding_cost_usdt, 2),
        "funding_events": funding_events,
        "win_rate_pct": round(win_rate, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "pnl_pct": pnl_list,
    }


def _compounded(net_pcts: list[float]) -> float:
    x = 1.0
    for r in net_pcts:
        x *= 1.0 + r / 100.0
    return (x - 1.0) * 100.0


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()
    rows: list[dict] = []
    base_all_pnl: list[float] = []
    fs_all_pnl: list[float] = []
    base_summary_csv = ROOT / "reports" / "_postfrac_adaptrend_v1_fs_base_trades.csv"
    fs_summary_csv = ROOT / "reports" / "_postfrac_adaptrend_v1_fs_fs_trades.csv"
    for p in (base_summary_csv, fs_summary_csv):
        if p.exists():
            p.unlink()

    print(
        "[postfrac_adaptrendV1_imp_funding_skip] "
        f"11 windows, cash={CASH}, PRICE_SCALE={PRICE_SCALE}, config={CONFIG_V1}",
        file=sys.stderr,
    )

    for label, start, end in WINDOWS_11:
        tw = time.time()
        df_scaled = _load_window_scaled(start, end)
        if df_scaled is None:
            print(f"  [{label}] SKIP — no data", file=sys.stderr)
            continue
        funding_full = _load_funding_full(start, end)
        funding_oos = _load_funding_oos(start, end)
        funding_series_full = (
            funding_full["funding_rate"]
            if "funding_rate" in funding_full.columns
            else funding_full.iloc[:, 0]
        )

        # --- BASE arm: V1, no skip filter. ---
        base_run = _run_one(AdaptiveTrendV1, df_scaled, CASH, CONFIG_V1)
        base_res = _summarize_run(
            label, "base", start, end, CASH, df_scaled, base_run["trades"], funding_oos
        )

        # --- FS arm: V1 + funding_skip. Inject series BEFORE Backtest. ---
        AdaptiveTrendV1_funding_skip.funding_series = funding_series_full
        fs_run = _run_one(
            AdaptiveTrendV1_funding_skip, df_scaled, CASH, CONFIG_V1
        )
        fs_res = _summarize_run(
            label, "fs", start, end, CASH, df_scaled, fs_run["trades"], funding_oos
        )
        # Skipped-entry diagnostic.
        fs_strategy = fs_run["strategy"]
        fs_res["n_skipped_by_funding"] = (
            int(getattr(fs_strategy, "_n_skipped_by_funding", 0))
            if fs_strategy is not None
            else 0
        )

        # Per-trade CSVs for PSR aggregation.
        if base_res["pnl_pct"]:
            pd.DataFrame(
                {"pnl_pct": base_res["pnl_pct"], "window": label}
            ).to_csv(
                base_summary_csv,
                mode="a",
                index=False,
                header=not base_summary_csv.exists(),
            )
            base_all_pnl.extend(base_res["pnl_pct"])
        if fs_res["pnl_pct"]:
            pd.DataFrame(
                {"pnl_pct": fs_res["pnl_pct"], "window": label}
            ).to_csv(
                fs_summary_csv,
                mode="a",
                index=False,
                header=not fs_summary_csv.exists(),
            )
            fs_all_pnl.extend(fs_res["pnl_pct"])

        # Canonical-block inputs for the FS arm (the deployed-candidate arm).
        # eq_impact_pnl_pct = sizing-aware PnL/equity-at-entry series; this is
        # what feeds psr_per_window. No window_start: each window is already
        # its own OOS slice in this runner. Computed here while fs_run["stats"]
        # is in scope.
        fs_res["eq_impact_pnl_pct"] = equity_impact_returns(
            fs_run["stats"], cash=CASH
        ).tolist()

        # Pop heavy arrays from the BASE per-window dict before storing (base
        # is diagnostic only — the canonical block reads the FS arm). Keep
        # pnl_pct AND eq_impact_pnl_pct on the FS arm so the canonical block
        # is not emitted hollow.
        base_res_light = {k: v for k, v in base_res.items() if k != "pnl_pct"}
        rows.append({"label": label, "base": base_res_light, "fs": fs_res})

        print(
            f"  {label}  base: tr={base_res['trades']:3d} "
            f"gross={base_res['gross_return_pct']:+7.2f}% "
            f"net={base_res['net_return_pct']:+7.2f}% "
            f"fund=${base_res['funding_cost_usdt']:>10,.0f}  |  "
            f"fs: tr={fs_res['trades']:3d} "
            f"net={fs_res['net_return_pct']:+7.2f}% "
            f"skipped={fs_res['n_skipped_by_funding']:3d}  "
            f"({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )

    # --- Aggregation ---
    base_nets = [r["base"]["net_return_pct"] for r in rows]
    fs_nets = [r["fs"]["net_return_pct"] for r in rows]
    base_gross = [r["base"]["gross_return_pct"] for r in rows]
    fs_gross = [r["fs"]["gross_return_pct"] for r in rows]
    base_trades_total = sum(r["base"]["trades"] for r in rows)
    fs_trades_total = sum(r["fs"]["trades"] for r in rows)
    base_wins = sum(1 for n in base_nets if n > 0)
    fs_wins = sum(1 for n in fs_nets if n > 0)
    base_comp_net = _compounded(base_nets)
    fs_comp_net = _compounded(fs_nets)
    base_comp_gross = _compounded(base_gross)
    fs_comp_gross = _compounded(fs_gross)

    base_funding_total = sum(r["base"]["funding_cost_usdt"] for r in rows)
    fs_funding_total = sum(r["fs"]["funding_cost_usdt"] for r in rows)

    # PSR (uses per-trade pnl_pct, NOT compounded per-window).
    base_pnl_arr = np.asarray(base_all_pnl, dtype=float)
    fs_pnl_arr = np.asarray(fs_all_pnl, dtype=float)
    base_psr = (
        compute_psr(base_pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(base_pnl_arr) >= 2
        else {"n_trades": int(len(base_pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )
    fs_psr = (
        compute_psr(fs_pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(fs_pnl_arr) >= 2
        else {"n_trades": int(len(fs_pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    print("\n=== SUMMARY ===", file=sys.stderr)
    print(
        f"{'Window':<10} {'base_net':>10} {'fs_net':>10}  delta",
        file=sys.stderr,
    )
    for r, bn, fn in zip(rows, base_nets, fs_nets):
        print(
            f"{r['label']:<10} {bn:>+9.2f}% {fn:>+9.2f}%  {fn-bn:+7.2f}pp",
            file=sys.stderr,
        )
    print(
        f"{'COMP_NET':<10} {base_comp_net:>+9.2f}% {fs_comp_net:>+9.2f}%  "
        f"{fs_comp_net-base_comp_net:+7.2f}pp",
        file=sys.stderr,
    )
    print(
        f"{'COMP_GROSS':<10} {base_comp_gross:>+9.2f}% {fs_comp_gross:>+9.2f}%",
        file=sys.stderr,
    )
    print(f"{'WINS':<10} {base_wins:>9d}  {fs_wins:>9d}", file=sys.stderr)
    print(
        f"{'TRADES':<10} {base_trades_total:>9d}  {fs_trades_total:>9d}",
        file=sys.stderr,
    )
    print(
        f"base PSR: {base_psr['psr_vs_hurdle']:.4f} "
        f"(sharpe={base_psr['point_sharpe']:.4f} n={base_psr['n_trades']} "
        f"MinTRL={base_psr['min_trl']})",
        file=sys.stderr,
    )
    print(
        f"+fs  PSR: {fs_psr['psr_vs_hurdle']:.4f} "
        f"(sharpe={fs_psr['point_sharpe']:.4f} n={fs_psr['n_trades']} "
        f"MinTRL={fs_psr['min_trl']})",
        file=sys.stderr,
    )

    # Canonical v2 dual-emit — soft-drift runner (funding-net is semantically
    # required; we keep the funding-adjusted equity AS the v2 metric but tag
    # it so cross-comparison logic can tell it apart from the strict v2_equity_curve.
    from tools.aggregate import build_canonical_block
    per_window_canon = [
        {
            "label": r["label"],
            # v2 headline = funding-adjusted net (sizing-aware via PnL/cash)
            "return_pct": r["fs"]["net_return_pct"],
            "trades": r["fs"]["trades"],
            # gross per-trade ReturnPct% — drives the synthetic stitched ref.
            # Stays GROSS by definition (funding applied only at window level
            # via net_return_pct); do not reconcile to the funding-net headline.
            "pnl_pct": r["fs"]["pnl_pct"],
            # sizing-aware equity-impact series — LOAD-BEARING: feeds psr_per_window.
            "eq_impact_pnl_pct": r["fs"]["eq_impact_pnl_pct"],
        }
        for r in rows
    ]
    canon = build_canonical_block(
        per_window_canon,
        aggregation_method="v2_equity_curve_funding_adjusted",
    )

    out = {
        "improvement_id":  "funding_skip",
        "applied_to":      "AdaptiveTrendV1",
        "fractional_sizing": True,
        "price_scale":     PRICE_SCALE,
        "oos_windows":     [w[0] for w in WINDOWS_11],
        "cash_per_window": CASH,
        "config":          CONFIG_V1,
        "rows":            rows,
        "aggregation_method": "v2_equity_curve_funding_adjusted",
        "funding_adjusted":   True,
        "canonical":          canon,
        "summary": {
            "base_compounded_net_pct":  round(base_comp_net, 4),
            "fs_compounded_net_pct":    round(fs_comp_net, 4),
            "delta_compounded_net_pp":  round(fs_comp_net - base_comp_net, 4),
            "base_compounded_gross_pct": round(base_comp_gross, 4),
            "fs_compounded_gross_pct":   round(fs_comp_gross, 4),
            "base_wins":                base_wins,
            "fs_wins":                  fs_wins,
            "base_total_trades":        base_trades_total,
            "fs_total_trades":          fs_trades_total,
            "base_total_funding_usdt":  round(base_funding_total, 2),
            "fs_total_funding_usdt":    round(fs_funding_total, 2),
            "delta_psr":                round(fs_psr["psr_vs_hurdle"]
                                              - base_psr["psr_vs_hurdle"], 4),
        },
        "base_psr": base_psr,
        "fs_psr":   fs_psr,
        "v2_prior_finding": {
            "delta_compounded_net_pp": -4.42,
            "note": "V2 + funding_skip with int() truncation in V2 sizing",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path = ROOT / "reports" / "postfrac_adaptrendV1_imp_funding_skip.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
