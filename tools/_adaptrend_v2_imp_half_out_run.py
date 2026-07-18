"""
Orchestration runner for the AdaptiveTrendV2 + half_out_at_1R experiment.

For each of the 11 OOS windows used in ADAPTIVE_TREND_EXTENDED_VERDICT.md:
  - Run AdaptiveTrendV2 (baseline)
  - Run AdaptiveTrendV2_half_out_at_1R (improvement)
Compute per-window and aggregate:
  - compounded return (chained net%)
  - wins/11
  - total trades
  - pooled per-trade Sharpe + PSR + MinTRL (psr_eval)

Writes reports/adaptrend_v2_imp_half_out_at_1R.json.

Reuses the prefix-buffered wrapper in tools/_adaptrend_v2_run.py for the
funding-aware run; we just swap the strategy class via a small ad-hoc
wrapper because that runner hard-codes AdaptiveTrendV2.

Not a CLI tool — invoked directly as: python tools/_adaptrend_v2_imp_half_out_run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import funding_cost_for_trades  # noqa: E402
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2  # noqa: E402
from strategy.signals_adaptive_trend_v2_half_out_at_1R import (  # noqa: E402
    AdaptiveTrendV2_half_out_at_1R,
)
from tools.aggregate import build_canonical_block, equity_impact_returns  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
_FUNDING = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
_COMMISSION = 0.0005
_MARGIN = 1.0 / 20
_CASH = 1_000_000.0
_PREFIX_MONTHS = 6

OOS_WINDOWS = [
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


def _load_15m_with_prefix(start: str, end: str, prefix_months: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = pd.read_parquet(_PARQUET)
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


def _load_funding(start: str, end: str) -> pd.DataFrame:
    f = pd.read_parquet(_FUNDING)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return f.loc[(f.index >= start_ts) & (f.index <= end_ts)].copy()


def run_window(
    strategy_cls: Type[Strategy],
    label: str,
    start: str,
    end: str,
) -> dict:
    """Run one window with prefix buffer.  Returns per-window stats + trades_df."""
    data_15m_full, oos_start_ts = _load_15m_with_prefix(start, end, _PREFIX_MONTHS)
    funding = _load_funding(start, end)

    bt = Backtest(
        data_15m_full,
        strategy_cls,
        cash=_CASH,
        commission=_COMMISSION,
        margin=_MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(trade_start_ns=oos_start_ts.value)
    trades_df = getattr(stats, "_trades", None)

    if trades_df is not None and len(trades_df) > 0 and "EntryTime" in trades_df.columns:
        trades_oos = trades_df[trades_df["EntryTime"] >= oos_start_ts].copy()
    else:
        trades_oos = trades_df.copy() if trades_df is not None else pd.DataFrame()

    n_trades = int(len(trades_oos))
    if n_trades > 0 and "PnL" in trades_oos.columns:
        gross_pnl_usdt = float(trades_oos["PnL"].sum())
    else:
        gross_pnl_usdt = 0.0
    gross_return_pct = gross_pnl_usdt / _CASH * 100.0

    funding_cost_usdt = 0.0
    funding_events = 0
    if n_trades > 0 and not funding.empty:
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_oos, data_15m_full, funding
        )

    net_final_equity = _CASH * (1.0 + gross_return_pct / 100.0) - funding_cost_usdt
    net_return_pct = (net_final_equity / _CASH - 1.0) * 100.0

    win_rate = 0.0
    if n_trades > 0 and "PnL" in trades_oos.columns:
        win_rate = float((trades_oos["PnL"] > 0).mean() * 100.0)

    max_dd_pct = 0.0
    if n_trades > 0 and "PnL" in trades_oos.columns:
        ordered = trades_oos.sort_values("ExitTime") if "ExitTime" in trades_oos.columns else trades_oos
        equity = _CASH + ordered["PnL"].cumsum()
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max * 100.0
        max_dd_pct = float(dd.min()) if len(dd) > 0 else 0.0

    return {
        "label": label,
        "start": start,
        "end": end,
        "trades": n_trades,
        "gross_return_pct": round(gross_return_pct, 4),
        "net_return_pct": round(net_return_pct, 4),
        "funding_cost_usdt": round(funding_cost_usdt, 2),
        "funding_events": funding_events,
        "win_rate_pct": round(win_rate, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "trades_df": trades_oos,
    }


def aggregate(per_window: list[dict]) -> dict:
    """Aggregate per-window stats and compute PSR on TWO bases:

    - psr_row: one observation per row in trades_df (raw backtesting.py output).
      For the improvement, half_out splits each touched entry into TWO rows
      (a guaranteed-positive +1R partial + a runner exit), which biases the
      pooled distribution.  Kept for diagnostic completeness.
    - psr_entry: one observation per ENTRY (groupby EntryTime, sum PnL, divide
      by cash to get per-entry return).  This is the apples-to-apples basis
      vs the v2 baseline (which has 1 row per entry).  USE THIS FOR THE
      HEADLINE COMPARISON.
    """
    eq = 1.0
    wins = 0
    total_trades = 0
    pooled_row_returns: list[float] = []
    pooled_entry_returns: list[float] = []
    rows_for_print = []
    entries_total = 0
    # canonical per-window list (methodology debt #1 — built from the SAME
    # in-memory trades_df, no CSV round-trip needed for this runner).
    per_window_canon: list[dict] = []
    for w in per_window:
        net = w["net_return_pct"] / 100.0
        eq *= (1.0 + net)
        if w["net_return_pct"] > 0:
            wins += 1
        total_trades += w["trades"]
        tdf = w["trades_df"]
        if tdf is not None and len(tdf) > 0 and "ReturnPct" in tdf.columns:
            pooled_row_returns.extend(tdf["ReturnPct"].astype(float).tolist())
        # Per-entry regrouping.
        if tdf is not None and len(tdf) > 0 and "EntryTime" in tdf.columns and "PnL" in tdf.columns:
            grouped_pnl = tdf.groupby("EntryTime", as_index=False)["PnL"].sum()
            # Per-entry return = PnL / cash (consistent unit basis with row-level ReturnPct
            # which is PnL / notional_at_entry; pooled SR is scale-invariant either way
            # for comparing two strategies on the same cash basis).
            entry_returns = (grouped_pnl["PnL"].astype(float) / _CASH).tolist()
            pooled_entry_returns.extend(entry_returns)
            entries_total += len(grouped_pnl)
        rows_for_print.append({
            "label": w["label"],
            "trades": w["trades"],
            "net_return_pct": w["net_return_pct"],
            "win_rate_pct": w["win_rate_pct"],
            "max_dd_pct": w["max_dd_pct"],
        })
        # Canonical per-window: sizing-aware eq-impact series (per-row PnL /
        # equity-at-entry) via the {"_trades": df} shim; ReturnPct% as the
        # legacy stitched ref.  return_pct = funding-net window return.
        if tdf is not None and len(tdf) > 0:
            eq_impact = equity_impact_returns({"_trades": tdf}, cash=_CASH).tolist()
            pnl_pct = (
                (tdf["ReturnPct"].astype(float) * 100.0).tolist()
                if "ReturnPct" in tdf.columns
                else []
            )
        else:
            eq_impact = []
            pnl_pct = []
        per_window_canon.append({
            "label": w["label"],
            "return_pct": w["net_return_pct"],
            "trades": w["trades"],
            "pnl_pct": pnl_pct,
            "eq_impact_pnl_pct": eq_impact,
        })

    compounded_pct = round((eq - 1.0) * 100.0, 4)

    # Canonical equity-curve dual-emit block (additive — does NOT replace the
    # headline psr_entry below, which stays the apples-to-apples verdict basis).
    canon = build_canonical_block(
        per_window_canon,
        aggregation_method="v2_equity_curve_funding_adjusted",
    )

    def _psr(arr_pct: np.ndarray) -> dict:
        if len(arr_pct) >= 2:
            return compute_psr(arr_pct, sr_hurdle=0.0, confidence=0.95)
        return {
            "n_trades": len(arr_pct),
            "point_sharpe": 0.0,
            "sr_se_lo": 0.0,
            "psr_vs_hurdle": 0.0,
            "min_trl": int(1e9),
            "skew": 0.0,
            "kurt": 3.0,
            "interpretation": "insufficient_evidence",
        }

    psr_row = _psr(np.asarray(pooled_row_returns, dtype=float) * 100.0)
    psr_entry = _psr(np.asarray(pooled_entry_returns, dtype=float) * 100.0)

    return {
        "compounded_pct": compounded_pct,
        "wins": wins,
        "total_trades": total_trades,
        "entries_total": entries_total,
        "per_window": rows_for_print,
        "psr_row": psr_row,
        "psr_entry": psr_entry,
        "psr": psr_entry,  # headline = per-entry (apples-to-apples)
        # canonical v2 equity-curve aggregation (primary metric — additive).
        "canonical": canon,
    }


def main() -> int:
    print(f"[adaptrend_v2_imp] running 11 OOS windows x 2 strategies", file=sys.stderr)

    base_windows = []
    imp_windows = []
    for label, start, end in OOS_WINDOWS:
        print(f"  [base] {label}", file=sys.stderr)
        base_windows.append(run_window(AdaptiveTrendV2, label, start, end))
        print(f"  [imp ] {label}", file=sys.stderr)
        imp_windows.append(run_window(AdaptiveTrendV2_half_out_at_1R, label, start, end))

    base_agg = aggregate(base_windows)
    imp_agg = aggregate(imp_windows)

    base_canon = base_agg["canonical"]
    imp_canon = imp_agg["canonical"]

    # --- Canonical self-check (bit-for-bit): recompute headline PSR from the
    # PERSISTED canonical per-window return series; assert == psr_walkforward. ---
    canonical_selfcheck = {}
    for arm_label, canon in (("base", base_canon), ("imp", imp_canon)):
        persisted_returns = np.asarray(canon["per_window_return_pct"], dtype=float)
        recomputed = compute_psr(persisted_returns, contiguous=False)
        headline = canon["psr_walkforward"]
        match = (
            recomputed["psr_vs_hurdle"] == headline["psr_vs_hurdle"]
            and recomputed["n_trades"] == headline["n_trades"]
        )
        canonical_selfcheck[arm_label] = {
            "canonical_psr": headline["psr_vs_hurdle"],
            "recomputed_psr": recomputed["psr_vs_hurdle"],
            "matches_headline": bool(match),
        }
        assert match, (
            f"CANONICAL SELF-CHECK FAILED [{arm_label}]: "
            f"recomputed={recomputed['psr_vs_hurdle']} (n={recomputed['n_trades']}) "
            f"!= headline={headline['psr_vs_hurdle']} (n={headline['n_trades']})"
        )

    print(
        f"canon PSR: base psr_wf={base_canon['psr_walkforward']['psr_vs_hurdle']:.4f} "
        f"imp psr_wf={imp_canon['psr_walkforward']['psr_vs_hurdle']:.4f} "
        f"(legacy per-entry base={base_agg['psr']['psr_vs_hurdle']:.3f} "
        f"imp={imp_agg['psr']['psr_vs_hurdle']:.3f})",
        file=sys.stderr,
    )

    result = {
        "id": "half_out_at_1R",
        "config": {
            "cash": _CASH,
            "commission": _COMMISSION,
            "margin": _MARGIN,
            "prefix_months": _PREFIX_MONTHS,
            "feature": "Scale 50% of position at +1R (alpha*ATR_entry) take-profit; runner trails the rest.",
        },
        "windows": [w[0] for w in OOS_WINDOWS],
        "base": {
            "strategy": "AdaptiveTrendV2",
            "compounded_pct": base_agg["compounded_pct"],
            "wins": base_agg["wins"],
            "total_trades": base_agg["total_trades"],
            "entries_total": base_agg["entries_total"],
            "per_window": base_agg["per_window"],
            "psr": base_agg["psr"],            # per-entry (apples-to-apples)
            "psr_per_row": base_agg["psr_row"],
            "canonical": base_canon,           # canonical equity-curve (primary)
        },
        "improvement": {
            "strategy": "AdaptiveTrendV2_half_out_at_1R",
            "compounded_pct": imp_agg["compounded_pct"],
            "wins": imp_agg["wins"],
            "total_trades": imp_agg["total_trades"],
            "entries_total": imp_agg["entries_total"],
            "per_window": imp_agg["per_window"],
            "psr": imp_agg["psr"],            # per-entry (apples-to-apples)
            "psr_per_row": imp_agg["psr_row"],
            "canonical": imp_canon,           # canonical equity-curve (primary)
        },
        "delta": {
            "compounded_pp": round(imp_agg["compounded_pct"] - base_agg["compounded_pct"], 4),
            "wins": imp_agg["wins"] - base_agg["wins"],
            "trades": imp_agg["total_trades"] - base_agg["total_trades"],
            "entries": imp_agg["entries_total"] - base_agg["entries_total"],
            "per_trade_sharpe": round(
                imp_agg["psr"]["point_sharpe"] - base_agg["psr"]["point_sharpe"], 6
            ),
            "psr": round(imp_agg["psr"]["psr_vs_hurdle"] - base_agg["psr"]["psr_vs_hurdle"], 6),
            "min_trl": imp_agg["psr"]["min_trl"] - base_agg["psr"]["min_trl"],
            "note": "psr/sharpe/min_trl deltas computed on per-ENTRY basis (apples-to-apples).",
        },
        # legacy stitched/per-entry PSR (observability/diff only — N-inflated).
        "legacy_psr_stitched": {
            "base_per_row": base_agg["psr_row"],
            "improvement_per_row": imp_agg["psr_row"],
            "base_per_entry": base_agg["psr_entry"],
            "improvement_per_entry": imp_agg["psr_entry"],
        },
        # canonical v2 equity-curve aggregation (primary metric).
        "aggregation_method": "v2_equity_curve_funding_adjusted",
        "funding_adjusted": True,
        "canonical": {"base": base_canon, "improvement": imp_canon},
        "canonical_psr_selfcheck": canonical_selfcheck,
    }

    out_path = ROOT / "reports" / "adaptrend_v2_imp_half_out_at_1R.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[adaptrend_v2_imp] wrote {out_path}", file=sys.stderr)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
