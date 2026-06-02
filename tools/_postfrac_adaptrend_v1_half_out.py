"""AdaptiveTrendV1 + half_out_at_1R ablation runner (post-fractional-refactor).

Compares AdaptiveTrendV1 (base, alpha=2.0) vs AdaptiveTrendV1_half_out_1r
(same alpha, plus 50% scale-out at +1R) on the 5 OOS windows at $1M cash
with PRICE_SCALE=0.001 fractional sizing.

Harness mirrors tools/_postfrac_adaptrend_v1_adx.py — direct OHLC slice,
PRICE_SCALE applied, NO prefix buffer, NO funding net.  Apples-to-apples
with the locked +45.52% / 255 trades / PSR 0.905 reference.

Key methodological detail (different from ADX runner):
The half-out splits each TOUCHED entry into TWO rows in backtesting.py's
trades_df (the +1R partial + the runner exit).  Pooling per-ROW
ReturnPct inflates the test arm's Sharpe/PSR because the +1R partials
are guaranteed-positive small returns.  We compute PSR on TWO bases:
  - psr_row : raw per-row (diagnostic; biased for half-out arm)
  - psr_entry : per-ENTRY (groupby EntryTime, sum PnL) — APPLES-TO-APPLES
                with the base arm which has 1 row per entry.
The headline PSR delta uses psr_entry.

Adversarial check:
- Entry count (unique EntryTime) MUST match between arms — entries are
  unchanged, only exits differ.  We halt and report if they diverge.
- Base arm MUST reproduce +45.52% / 255 trades / PSR ~0.905.  We log the
  delta vs reference but do not halt (small ~floating-point drift is OK).

Writes:
  - reports/postfrac_adaptrend_v1_half_out_1r.json
  - reports/_postfrac_adaptrend_v1_half_out_<window>.csv per OOS window
  - reports/_postfrac_adaptrend_v1_half_out_aggregated.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_adaptive_trend import AdaptiveTrendV1  # noqa: E402
from strategy.signals_adaptive_trend_v1_half_out_1r import (  # noqa: E402
    AdaptiveTrendV1_half_out_1r,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

# Hold alpha=2.0 to match the postfrac base run (winner_config from
# reports/adaptive_trend_extended_psr.json).  Half-out is the ONLY variable.
BASE_CONFIG = {"alpha": 2.0}
TEST_CONFIG = {"alpha": 2.0, "half_out_at_1r": True}

WINDOWS_5 = [
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


def run_window(
    strategy_cls: Type[Strategy],
    config: dict,
    label: str,
    start: str,
    end: str,
    csv_prefix: str,
) -> dict | None:
    df = _load_slice_scaled(start, end)
    if df is None:
        print(f"  [{label}] SKIP - no data", file=sys.stderr)
        return None
    bt = Backtest(
        df,
        strategy_cls,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(**config)

    n_rows = int(stats.get("# Trades", 0) or 0)
    ret_pct = float(stats.get("Return [%]", 0.0) or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)
    win_rate = float(stats.get("Win Rate [%]") or 0.0)
    equity_final = float(stats.get("Equity Final [$]", CASH) or CASH)

    trades_df = getattr(stats, "_trades", None)
    pnl_pct_list: list[float] = []
    n_entries = 0
    per_entry_returns_pct: list[float] = []  # in PERCENT

    if trades_df is not None and len(trades_df) > 0:
        if "ReturnPct" in trades_df.columns:
            pnl_pct_list = (trades_df["ReturnPct"].astype(float).values * 100.0).tolist()
        if "EntryTime" in trades_df.columns and "PnL" in trades_df.columns:
            grouped = trades_df.groupby("EntryTime", as_index=False)["PnL"].sum()
            n_entries = int(len(grouped))
            # Per-entry return = PnL / CASH * 100 — % return basis for PSR.
            per_entry_returns_pct = (
                grouped["PnL"].astype(float).values / CASH * 100.0
            ).tolist()

        if csv_prefix:
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
        "label":              label,
        "start":              start,
        "end":                end,
        "rows":               n_rows,
        "entries":            n_entries,
        "return_pct":         round(ret_pct, 4),
        "max_dd_pct":         round(max_dd, 4),
        "win_rate_pct":       round(win_rate, 4),
        "equity_final":       round(equity_final, 4),
        "pnl_pct_rows":       pnl_pct_list,
        "pnl_pct_per_entry":  per_entry_returns_pct,
    }


def aggregate(per_window: list[dict]) -> dict:
    all_row_pnl: list[float] = []
    all_entry_pnl: list[float] = []
    n_rows = 0
    n_entries = 0
    n_pos = 0
    compounded = 1.0
    per_win_ret = []
    for r in per_window:
        n_rows += r["rows"]
        n_entries += r["entries"]
        all_row_pnl.extend(r["pnl_pct_rows"])
        all_entry_pnl.extend(r["pnl_pct_per_entry"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        if r["return_pct"] > 0:
            n_pos += 1
    return {
        "n_rows":                n_rows,
        "n_entries":             n_entries,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{len(per_window)}",
        "per_window_return_pct": per_win_ret,
        "all_pnl_pct_rows":      all_row_pnl,
        "all_pnl_pct_entries":   all_entry_pnl,
    }


def run_arm(
    arm_label: str,
    strategy_cls: Type[Strategy],
    config: dict,
    csv_prefix: str,
) -> dict:
    per_window: list[dict] = []
    print(
        f"[half_out_ablation] running arm={arm_label} ({len(WINDOWS_5)} windows) ...",
        file=sys.stderr,
    )
    for label, start, end in WINDOWS_5:
        tw = time.time()
        r = run_window(strategy_cls, config, label, start, end, csv_prefix)
        if r is None:
            continue
        print(
            f"  [{arm_label}] {label}  entries={r['entries']:4d}  "
            f"rows={r['rows']:4d}  ret={r['return_pct']:+8.2f}%  "
            f"dd={r['max_dd_pct']:+7.2f}%  win_rate={r['win_rate_pct']:5.2f}%  "
            f"({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )
        per_window.append(r)

    agg = aggregate(per_window)
    all_row = agg.pop("all_pnl_pct_rows")
    all_entry = agg.pop("all_pnl_pct_entries")

    def _psr(arr: list[float]) -> dict:
        a = np.asarray(arr, dtype=float)
        if len(a) >= 2:
            return compute_psr(a, sr_hurdle=0.0, confidence=0.95)
        return {
            "n_trades": int(len(a)),
            "psr_vs_hurdle": 0.0,
            "interpretation": "insufficient_evidence",
        }

    psr_row = _psr(all_row)
    psr_entry = _psr(all_entry)

    if csv_prefix:
        agg_csv = ROOT / "reports" / f"{csv_prefix}_aggregated.csv"
        pd.DataFrame({"pnl_pct": all_row}).to_csv(agg_csv, index=False)
        print(f"  [{arm_label}] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    return {
        "arm":        arm_label,
        "config":     config,
        "per_window": [
            {k: v for k, v in r.items() if k not in ("pnl_pct_rows", "pnl_pct_per_entry")}
            for r in per_window
        ],
        "summary":    agg,
        "psr_row":    psr_row,    # diagnostic (biased for half-out arm)
        "psr_entry":  psr_entry,  # headline (apples-to-apples)
        "psr":        psr_entry,  # alias for primary verdict
    }


def verdict(base: dict, test: dict) -> dict:
    base_comp = base["summary"]["compounded_pct"]
    test_comp = test["summary"]["compounded_pct"]
    base_psr = base["psr"]["psr_vs_hurdle"]
    test_psr = test["psr"]["psr_vs_hurdle"]
    base_sharpe = base["psr"].get("point_sharpe")
    test_sharpe = test["psr"].get("point_sharpe")
    base_entries = base["summary"]["n_entries"]
    test_entries = test["summary"]["n_entries"]
    base_rows = base["summary"]["n_rows"]
    test_rows = test["summary"]["n_rows"]

    delta_comp = test_comp - base_comp
    delta_psr = test_psr - base_psr
    entries_match = (base_entries == test_entries)

    cleared_compounded = delta_comp >= 5.0
    psr_not_worse = test_psr >= base_psr - 1e-6

    if not entries_match:
        decision = (
            "INVESTIGATE - entry counts diverge (base={base} vs test={test}); "
            "the half-out should not change entries. Possible size=1 collapse "
            "or exclusive_orders re-entry artifact."
        ).format(base=base_entries, test=test_entries)
    elif cleared_compounded and psr_not_worse:
        decision = "PROMOTE_CANDIDATE - extend to walk-forward 2020-2026"
    elif cleared_compounded and not psr_not_worse:
        decision = "ITERATE - compounded improves but PSR drops; risk-adjusted edge unclear"
    elif not cleared_compounded and psr_not_worse:
        decision = "ITERATE - PSR holds but compounded fails +5pp bar"
    else:
        decision = "SHELF - fails both bars"

    return {
        "base_compounded_pct":   base_comp,
        "test_compounded_pct":   test_comp,
        "delta_compounded_pp":   round(delta_comp, 4),
        "base_psr_entry":        base_psr,
        "test_psr_entry":        test_psr,
        "delta_psr_entry":       round(delta_psr, 4),
        "base_point_sharpe":     base_sharpe,
        "test_point_sharpe":     test_sharpe,
        "base_entries":          base_entries,
        "test_entries":          test_entries,
        "entries_match":         entries_match,
        "base_rows":             base_rows,
        "test_rows":             test_rows,
        "row_count_split_ratio": round(test_rows / max(base_rows, 1), 3),
        "cleared_+5pp_bar":      cleared_compounded,
        "psr_not_worse":         psr_not_worse,
        "decision":              decision,
    }


def main() -> int:
    t0 = time.time()

    res_base = run_arm(
        "base",
        AdaptiveTrendV1,
        BASE_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_half_out_baseRECHECK",
    )
    res_test = run_arm(
        "half_out_1r",
        AdaptiveTrendV1_half_out_1r,
        TEST_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_half_out",
    )

    v = verdict(res_base, res_test)

    # Per-window variance comparison (does scaling out shrink dispersion?).
    base_per_win = res_base["summary"]["per_window_return_pct"]
    test_per_win = res_test["summary"]["per_window_return_pct"]
    var_comp = {
        "base_per_window_pct":     base_per_win,
        "test_per_window_pct":     test_per_win,
        "base_per_window_std_pp":  round(float(np.std(base_per_win)), 4),
        "test_per_window_std_pp":  round(float(np.std(test_per_win)), 4),
        "base_per_window_min":     round(float(np.min(base_per_win)), 4),
        "test_per_window_min":     round(float(np.min(test_per_win)), 4),
        "base_per_window_max":     round(float(np.max(base_per_win)), 4),
        "test_per_window_max":     round(float(np.max(test_per_win)), 4),
    }

    result = {
        "experiment":     "adaptrend_v1_half_out_at_1R",
        "base_strategy":  "strategy.signals_adaptive_trend:AdaptiveTrendV1",
        "test_strategy":  "strategy.signals_adaptive_trend_v1_half_out_1r:AdaptiveTrendV1_half_out_1r",
        "cash":           CASH,
        "commission":     COMMISSION,
        "margin":         MARGIN,
        "price_scale":    PRICE_SCALE,
        "windows":        [w[0] for w in WINDOWS_5],
        "base":           res_base,
        "test":           res_test,
        "verdict":        v,
        "variance":       var_comp,
        "reference_postfrac_base": {
            "source": "reports/postfrac_adaptrend_v1.json (set_5_OOS)",
            "compounded_pct": 45.5222,
            "n_trades":       255,
            "psr_vs_hurdle":  0.905331,
            "point_sharpe":   0.076677,
            "note": "Base arm should reproduce these exactly (within fp drift).",
        },
        "elapsed_sec":    round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_adaptrend_v1_half_out_1r.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[half_out_ablation] verdict={v['decision']}  "
        f"base={v['base_compounded_pct']:+.2f}% -> test={v['test_compounded_pct']:+.2f}% "
        f"(delta {v['delta_compounded_pp']:+.2f}pp)  "
        f"base_PSR={v['base_psr_entry']:.3f} -> test_PSR={v['test_psr_entry']:.3f}  "
        f"entries_match={v['entries_match']}",
        file=sys.stderr,
    )
    print(f"[half_out_ablation] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
