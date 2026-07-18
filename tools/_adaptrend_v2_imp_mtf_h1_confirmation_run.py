"""
AdaptiveTrendV2 + mtf_h1_confirmation — improvement experiment runner.

Runs BOTH AdaptiveTrendV2 (base) AND AdaptiveTrendV2_mtf_h1_confirmation
across the 11 OOS windows from ADAPTIVE_TREND_EXTENDED_VERDICT, at
$1M cash with real Binance funding, and reports the delta + PSR.

Architecture: clone of tools/_adaptrend_v2_imp_regime_gate_adx_run.py,
parameterised on the strategy class so we can run base and improvement
back-to-back under the IDENTICAL prefix-buffered harness.

Saves:
  reports/adaptrend_v2_imp_mtf_h1_confirmation.json   (full result)
  reports/_adaptrend_v2_base_mtf_trades.csv           (base trades)
  reports/_adaptrend_v2_imp_mtf_trades.csv            (improvement trades)

Authority: research-only. No bot wiring, no risk.py edits, no commits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Type

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from backtesting import Backtest  # noqa: E402

from backtest import funding_cost_for_trades  # noqa: E402
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2  # noqa: E402
from strategy.signals_adaptive_trend_v2_mtf_h1_confirmation import (  # noqa: E402
    AdaptiveTrendV2_mtf_h1_confirmation,
)
from tools.aggregate import build_canonical_block, equity_impact_returns  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

_DEFAULT_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
_DEFAULT_FUNDING = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
_COMMISSION = 0.0005
_MARGIN = 1.0 / 20
_CASH = 1_000_000.0

# Same 11 OOS windows as the extended verdict.
OOS_WINDOWS = [
    ("2020-07-01", "2020-12-31", "2020_H2"),
    ("2021-01-01", "2021-06-30", "2021_H1"),
    ("2021-07-01", "2021-12-31", "2021_H2"),
    ("2022-01-01", "2022-06-30", "2022_H1"),
    ("2022-07-01", "2022-12-31", "2022_H2"),
    ("2023-01-01", "2023-06-30", "2023_H1"),
    ("2023-07-01", "2023-12-31", "2023_H2"),
    ("2024-01-01", "2024-06-30", "2024_H1"),
    ("2024-07-01", "2024-12-31", "2024_H2"),
    ("2025-01-01", "2025-06-30", "2025_H1"),
    ("2025-07-01", "2025-12-31", "2025_H2"),
]


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


def _load_funding_slice(parquet: Path, start: str, end: str) -> pd.DataFrame:
    f = pd.read_parquet(parquet)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return f.loc[(f.index >= start_ts) & (f.index <= end_ts)].copy()


def run_one(
    strategy_cls: Type,
    start: str,
    end: str,
    label: str,
    save_trades: Path | None,
    prefix_months: int = 6,
) -> dict:
    data_15m_full, oos_start_ts = _load_15m_with_prefix(
        _DEFAULT_PARQUET, start, end, prefix_months
    )
    funding = _load_funding_slice(_DEFAULT_FUNDING, start, end)

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
        trades_oos = trades_df

    n_trades = int(len(trades_oos)) if trades_oos is not None else 0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        gross_pnl_usdt = float(trades_oos["PnL"].sum())
    else:
        gross_pnl_usdt = 0.0
    gross_return_pct = gross_pnl_usdt / _CASH * 100.0

    win_rate = 0.0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        win_rate = float((trades_oos["PnL"] > 0).mean() * 100.0)

    funding_cost_usdt = 0.0
    funding_events = 0
    if trades_oos is not None and len(trades_oos) > 0 and not funding.empty:
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_oos, data_15m_full, funding
        )

    gross_final_equity = _CASH * (1.0 + gross_return_pct / 100.0)
    net_final_equity = gross_final_equity - funding_cost_usdt
    net_return_pct = (net_final_equity / _CASH - 1.0) * 100.0

    max_dd_pct = 0.0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        ordered = (
            trades_oos.sort_values("ExitTime")
            if "ExitTime" in trades_oos.columns
            else trades_oos
        )
        equity = _CASH + ordered["PnL"].cumsum()
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max * 100.0
        max_dd_pct = float(dd.min()) if len(dd) > 0 else 0.0

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
        out["label"] = label
        header = not save_trades.exists()
        out.to_csv(save_trades, mode="a", index=False, header=header)

    return {
        "label": label,
        "start": start,
        "end": end,
        "trades": n_trades,
        "net_return_pct": round(net_return_pct, 4),
        "gross_return_pct": round(gross_return_pct, 4),
        "funding_cost_usdt": round(funding_cost_usdt, 2),
        "funding_events": funding_events,
        "win_rate_pct": round(win_rate, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "final_equity_net": round(net_final_equity, 2),
    }


def compounded(nets: list[float]) -> float:
    x = 1.0
    for r in nets:
        x *= 1.0 + r / 100.0
    return (x - 1.0) * 100.0


def main() -> None:
    base_csv = ROOT / "reports" / "_adaptrend_v2_base_mtf_trades.csv"
    imp_csv = ROOT / "reports" / "_adaptrend_v2_imp_mtf_trades.csv"
    for p in (base_csv, imp_csv):
        if p.exists():
            p.unlink()

    base_rows: list[dict] = []
    imp_rows: list[dict] = []

    print("=== AdaptiveTrendV2 base vs +mtf_h1_confirmation — 11 OOS @ $1M ===", flush=True)
    print(
        f"{'Window':<10} {'base trades':>12} {'base net%':>10} {'imp trades':>11} {'imp net%':>9}",
        flush=True,
    )
    print("-" * 60, flush=True)

    for start, end, label in OOS_WINDOWS:
        b = run_one(AdaptiveTrendV2, start, end, label, base_csv)
        i = run_one(
            AdaptiveTrendV2_mtf_h1_confirmation, start, end, label, imp_csv
        )
        base_rows.append(b)
        imp_rows.append(i)
        print(
            f"{label:<10} {b['trades']:>12d} {b['net_return_pct']:>+10.2f} "
            f"{i['trades']:>11d} {i['net_return_pct']:>+9.2f}",
            flush=True,
        )

    base_nets = [r["net_return_pct"] for r in base_rows]
    imp_nets = [r["net_return_pct"] for r in imp_rows]
    base_comp = compounded(base_nets)
    imp_comp = compounded(imp_nets)
    base_wins = sum(1 for n in base_nets if n > 0)
    imp_wins = sum(1 for n in imp_nets if n > 0)
    base_trades = sum(r["trades"] for r in base_rows)
    imp_trades = sum(r["trades"] for r in imp_rows)

    # PSR — per-trade pnl_pct from CSVs.
    def _psr_from(csv: Path) -> dict:
        if not csv.exists():
            return compute_psr(np.array([], dtype=float))
        df = pd.read_csv(csv)
        if "pnl_pct" not in df.columns:
            return compute_psr(np.array([], dtype=float))
        pnl = df["pnl_pct"].dropna().values.astype(float)
        return compute_psr(pnl, sr_hurdle=0.0, confidence=0.95)

    base_psr = _psr_from(base_csv)
    imp_psr = _psr_from(imp_csv)

    print("-" * 60, flush=True)
    print(
        f"{'TOTAL':<10} {base_trades:>12d} {base_comp:>+10.2f} "
        f"{imp_trades:>11d} {imp_comp:>+9.2f}",
        flush=True,
    )
    print(f"Wins: base={base_wins}/11  improvement={imp_wins}/11", flush=True)
    print(
        f"Base PSR: {base_psr['psr_vs_hurdle']:.3f}  "
        f"point Sharpe={base_psr['point_sharpe']:.4f}  "
        f"MinTRL={base_psr['min_trl']}  n={base_psr['n_trades']}",
        flush=True,
    )
    print(
        f"Improvement PSR: {imp_psr['psr_vs_hurdle']:.3f}  "
        f"point Sharpe={imp_psr['point_sharpe']:.4f}  "
        f"MinTRL={imp_psr['min_trl']}  n={imp_psr['n_trades']}",
        flush=True,
    )

    # NOTE: verdict wiring kept BYTE-IDENTICAL to pre-migration — it gates on
    # delta_pp first then the STITCHED PSR comparison. Migration is additive;
    # it must NOT repoint this at canonical psr_walkforward (would manufacture
    # a flip). Canonical block below is observability/headline only.
    delta_pp = imp_comp - base_comp
    if delta_pp > 5.0 and imp_psr["psr_vs_hurdle"] > base_psr["psr_vs_hurdle"]:
        verdict = "ACCRETIVE"
    elif delta_pp < -5.0:
        verdict = "HURTING"
    else:
        verdict = "NEUTRAL"

    # --- Canonical equity-curve dual-emit (methodology debt #1) ---
    # Headline canonical PSR = psr_walkforward (compute_psr on the n-window
    # net-return series, contiguous=False). The legacy stitched per-trade PSR
    # (base_psr/imp_psr above) is N-inflated and kept verbatim for diff only.
    # Per-window eq-impact + per-trade ReturnPct% are reconstructed from the
    # saved trades CSVs via the {"_trades": df} mapping shim (zero drift vs
    # the canonical helper).
    def _build_canon(rows_arm: list[dict], csv_path: Path) -> dict:
        eq_by_label: dict[str, list[float]] = {}
        pnl_by_label: dict[str, list[float]] = {}
        if csv_path.exists():
            tdf = pd.read_csv(csv_path)
            if "label" in tdf.columns:
                for win_label, wdf in tdf.groupby("label"):
                    eq_by_label[str(win_label)] = equity_impact_returns(
                        {"_trades": wdf}, cash=_CASH
                    ).tolist()
                    if "pnl_pct" in wdf.columns:
                        pnl_by_label[str(win_label)] = (
                            wdf["pnl_pct"].dropna().astype(float).tolist()
                        )
        per_window_canon = []
        for r in rows_arm:
            win_label = r["label"]
            per_window_canon.append(
                {
                    "label": win_label,
                    # canonical v2 headline = funding-adjusted net (sizing-aware).
                    "return_pct": r["net_return_pct"],
                    "trades": r["trades"],
                    "pnl_pct": pnl_by_label.get(win_label, []),
                    "eq_impact_pnl_pct": eq_by_label.get(win_label, []),
                }
            )
        return build_canonical_block(
            per_window_canon,
            aggregation_method="v2_equity_curve_funding_adjusted",
        )

    base_canon = _build_canon(base_rows, base_csv)
    imp_canon = _build_canon(imp_rows, imp_csv)

    # --- Self-check (bit-for-bit): recompute headline PSR from the PERSISTED
    # canonical per-window return series; assert == psr_walkforward. ---
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
        f"(legacy stitched base={base_psr['psr_vs_hurdle']:.3f} "
        f"imp={imp_psr['psr_vs_hurdle']:.3f})",
        flush=True,
    )

    out = {
        "improvement_id": "mtf_h1_confirmation",
        "feature": "Entries gated on H1 RSI(14) in (30, 70) — neutral RSI band.",
        "windows": OOS_WINDOWS,
        "cash": _CASH,
        "base_rows": base_rows,
        "improvement_rows": imp_rows,
        "summary": {
            "base_compounded_pct": round(base_comp, 4),
            "improvement_compounded_pct": round(imp_comp, 4),
            "delta_pp": round(delta_pp, 4),
            "base_wins": base_wins,
            "improvement_wins": imp_wins,
            "base_trades": base_trades,
            "improvement_trades": imp_trades,
            "base_psr": base_psr,
            "improvement_psr": imp_psr,
            "verdict": verdict,
        },
        # legacy stitched PSR (observability/diff only — N-inflated).
        "legacy_psr_stitched": {"base": base_psr, "improvement": imp_psr},
        # canonical v2 equity-curve aggregation (primary metric).
        "aggregation_method": "v2_equity_curve_funding_adjusted",
        "funding_adjusted": True,
        "canonical": {"base": base_canon, "improvement": imp_canon},
        "canonical_psr_selfcheck": canonical_selfcheck,
    }

    out_path = ROOT / "reports" / "adaptrend_v2_imp_mtf_h1_confirmation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
