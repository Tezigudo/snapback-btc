"""
AdaptiveTrendV2 improvement experiment — regime_gate_vol.

Runs BOTH v2-base and v2+gate over the same 11 OOS windows in the SAME
prefix-buffered harness (so the only delta is the gate).  Aggregates trades,
computes per-trade PSR for each.  Writes reports/adaptrend_v2_imp_regime_gate_vol.json.

CLI:
    .venv/bin/python tools/adaptrend_v2_imp_regime_gate_vol.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from backtesting import Backtest  # noqa: E402

from backtest import funding_cost_for_trades  # noqa: E402
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2  # noqa: E402
from strategy.signals_adaptive_trend_v2_regime_gate_vol import (  # noqa: E402
    AdaptiveTrendV2_regime_gate_vol,
)
from tools.aggregate import build_canonical_block, equity_impact_returns  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

_BTC_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
_BTC_FUNDING = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
_COMMISSION = 0.0005
_MARGIN = 1.0 / 20
CASH = 1_000_000

WINDOWS = [
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


def _load_15m_with_prefix(parquet, start, end, prefix_months):
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    prefix_start = start_ts - pd.DateOffset(months=prefix_months)
    sliced = df.loc[(df.index >= prefix_start) & (df.index <= end_ts)].copy()
    return sliced, start_ts


def _load_funding_slice(parquet, start, end):
    f = pd.read_parquet(parquet)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return f.loc[(f.index >= start_ts) & (f.index <= end_ts)].copy()


def run_window(strategy_cls, start, end, label, cash=CASH, prefix_months=6,
               save_trades_csv=None):
    """Run one window for the given strategy class, return result dict.

    The dict carries `eq_impact_pnl_pct` (sizing-aware equity-impact returns,
    OOS-attributed via window_start) for the canonical per-window PSR.
    """
    data, oos_start_ts = _load_15m_with_prefix(_BTC_PARQUET, start, end, prefix_months)
    funding = _load_funding_slice(_BTC_FUNDING, start, end)

    bt = Backtest(
        data, strategy_cls, cash=cash, commission=_COMMISSION, margin=_MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    run_kwargs = {"trade_start_ns": oos_start_ts.value}
    stats = bt.run(**run_kwargs)

    trades_df = getattr(stats, "_trades", None)
    if trades_df is not None and len(trades_df) > 0 and "EntryTime" in trades_df.columns:
        trades_oos = trades_df[trades_df["EntryTime"] >= oos_start_ts].copy()
    else:
        trades_oos = trades_df if trades_df is not None else pd.DataFrame()

    n_trades = int(len(trades_oos))
    gross_pnl = float(trades_oos["PnL"].sum()) if n_trades > 0 and "PnL" in trades_oos.columns else 0.0
    gross_return_pct = gross_pnl / cash * 100.0
    win_rate = float((trades_oos["PnL"] > 0).mean() * 100.0) if n_trades > 0 else 0.0

    funding_cost_usdt = 0.0
    funding_events = 0
    if n_trades > 0 and not funding.empty:
        funding_cost_usdt, funding_events = funding_cost_for_trades(trades_oos, data, funding)

    gross_eq = cash * (1 + gross_return_pct / 100.0)
    net_eq = gross_eq - funding_cost_usdt
    net_return_pct = (net_eq / cash - 1.0) * 100.0

    max_dd_pct = 0.0
    if n_trades > 0 and "PnL" in trades_oos.columns:
        ordered = trades_oos.sort_values("ExitTime") if "ExitTime" in trades_oos.columns else trades_oos
        eq = cash + ordered["PnL"].cumsum()
        rm = eq.cummax()
        dd = (eq - rm) / rm * 100.0
        max_dd_pct = float(dd.min()) if len(dd) > 0 else 0.0

    # Per-trade ReturnPct% (stitched legacy ref) + sizing-aware equity-impact.
    win_pnl_pct: list[float] = []
    if n_trades > 0 and "ReturnPct" in trades_oos.columns:
        win_pnl_pct = (trades_oos["ReturnPct"].astype(float) * 100.0).tolist()
    eq_impact_pnl_pct = equity_impact_returns(
        stats, cash=cash, window_start=oos_start_ts
    ).tolist()

    # Save trades to CSV for pooled (legacy stitched) PSR
    if save_trades_csv is not None and n_trades > 0:
        out_cols = [c for c in ("ReturnPct", "PnL", "EntryTime", "ExitTime", "Size")
                    if c in trades_oos.columns]
        out = trades_oos[out_cols].copy()
        if "ReturnPct" in out.columns:
            out = out.rename(columns={"ReturnPct": "pnl_pct"})
            out["pnl_pct"] = out["pnl_pct"] * 100.0
        out["window"] = label
        header = not save_trades_csv.exists()
        out.to_csv(save_trades_csv, mode="a", index=False, header=header)

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
        "final_equity_net": round(net_eq, 2),
        "pnl_pct": win_pnl_pct,
        "eq_impact_pnl_pct": eq_impact_pnl_pct,
    }


def compounded(net_pcts):
    x = 1.0
    for r in net_pcts:
        x *= (1.0 + r / 100.0)
    return (x - 1.0) * 100.0


def run_variant(strategy_cls, label_prefix, trades_csv):
    if trades_csv.exists():
        trades_csv.unlink()
    rows = []
    for start, end, label in WINDOWS:
        print(f"[{label_prefix}] {label} {start}..{end} ...", flush=True)
        r = run_window(strategy_cls, start, end, label, save_trades_csv=trades_csv)
        rows.append(r)
        print(f"  -> trades={r['trades']:3d} net={r['net_return_pct']:+.2f}% "
              f"dd={r['max_dd_pct']:+.2f}%", flush=True)
    net_pcts = [r["net_return_pct"] for r in rows]
    comp = compounded(net_pcts)
    wins = sum(1 for p in net_pcts if p > 0)
    total_trades = sum(r["trades"] for r in rows)

    # Pool per-trade returns (LEGACY stitched PSR — N-inflated, sizing-blind;
    # observability/diff only).
    if trades_csv.exists():
        df = pd.read_csv(trades_csv)
        pnl = df.get("pnl_pct", pd.Series(dtype=float)).dropna().values.astype(float)
        psr = compute_psr(pnl, sr_hurdle=0.0, confidence=0.95)
    else:
        psr = compute_psr(np.array([], dtype=float))

    # Canonical equity-curve dual-emit (methodology debt #1). Funding-net
    # headline -> tag v2_equity_curve_funding_adjusted.
    per_window_canon = [
        {
            "label": r["label"],
            "return_pct": r["net_return_pct"],
            "trades": r["trades"],
            "pnl_pct": r.get("pnl_pct", []),
            "eq_impact_pnl_pct": r.get("eq_impact_pnl_pct", []),
        }
        for r in rows
    ]
    canon = build_canonical_block(
        per_window_canon,
        aggregation_method="v2_equity_curve_funding_adjusted",
    )

    # Strip heavy per-window arrays before persisting.
    per_window_light = [
        {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
        for r in rows
    ]

    return {
        "label": label_prefix,
        "per_window": per_window_light,
        "compounded_pct": round(comp, 4),
        "wins": wins,
        "total_windows": len(WINDOWS),
        "total_trades": total_trades,
        # legacy stitched PSR (observability/diff only).
        "psr": psr,
        "legacy_psr_stitched": psr,
        # canonical v2 equity-curve aggregation (primary metric).
        "canonical": canon,
        "aggregation_method": canon["aggregation_method"],
    }


def main():
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    base_trades_csv = reports_dir / "_adaptrend_v2_imp_regime_gate_vol_BASE_trades.csv"
    imp_trades_csv = reports_dir / "_adaptrend_v2_imp_regime_gate_vol_IMP_trades.csv"

    print("=== AdaptiveTrendV2 BASE — 11 OOS ===", flush=True)
    base = run_variant(AdaptiveTrendV2, "v2_base", base_trades_csv)

    print("\n=== AdaptiveTrendV2 + regime_gate_vol — 11 OOS ===", flush=True)
    imp = run_variant(AdaptiveTrendV2_regime_gate_vol, "v2_gate", imp_trades_csv)

    # Verdict
    delta_pp = imp["compounded_pct"] - base["compounded_pct"]
    psr_improved = imp["psr"]["psr_vs_hurdle"] > base["psr"]["psr_vs_hurdle"]
    if delta_pp > 5.0 and psr_improved:
        verdict = "ACCRETIVE"
    elif delta_pp < -5.0:
        verdict = "HURTING"
    else:
        verdict = "NEUTRAL"

    # --- Canonical self-check (bit-for-bit): re-derive headline PSR from the
    # PERSISTED canonical per-window return series; assert == psr_walkforward. ---
    canonical_selfcheck = {}
    for arm_label, blk in (("base", base), ("imp", imp)):
        canon = blk["canonical"]
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

    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"  base comp={base['compounded_pct']:.2f}%  imp comp={imp['compounded_pct']:.2f}%  "
          f"delta={delta_pp:+.2f}pp", flush=True)
    print(f"  base wins={base['wins']}/11  imp wins={imp['wins']}/11", flush=True)
    print(f"  base PSR(stitched)={base['psr']['psr_vs_hurdle']:.3f}  imp PSR(stitched)={imp['psr']['psr_vs_hurdle']:.3f}", flush=True)
    print(f"  base psr_wf={base['canonical']['psr_walkforward']['psr_vs_hurdle']:.4f}  "
          f"imp psr_wf={imp['canonical']['psr_walkforward']['psr_vs_hurdle']:.4f}", flush=True)

    out = {
        "improvement_id": "regime_gate_vol",
        "windows": [{"start": s, "end": e, "label": l} for s, e, l in WINDOWS],
        "cash": CASH,
        "v2_base": base,
        "v2_with_improvement": imp,
        "delta": {
            "compounded_pp": round(delta_pp, 4),
            "wins_delta": imp["wins"] - base["wins"],
            "trades_delta": imp["total_trades"] - base["total_trades"],
            "sharpe_delta": round(imp["psr"]["point_sharpe"] - base["psr"]["point_sharpe"], 6),
            "psr_delta": round(imp["psr"]["psr_vs_hurdle"] - base["psr"]["psr_vs_hurdle"], 6),
            "min_trl_delta": imp["psr"]["min_trl"] - base["psr"]["min_trl"],
        },
        "verdict": verdict,
        "aggregation_method": "v2_equity_curve_funding_adjusted",
        "funding_adjusted": True,
        "canonical_psr_selfcheck": canonical_selfcheck,
        "implementation_notes": {
            "approach": "subclass AdaptiveTrendV2 (no v1/v2-base files modified)",
            "vol_estimator": "rolling std of H6 log returns, window=14 H6 bars (~3.5 days)",
            "gate_threshold": "50th pct (median) of trailing 60-day vol distribution (240 H6 bars)",
            "gate_scope": "live entries only; fit simulator and refit untouched",
            "causality": "vol and median computed on H6 frame then shift(1); ffill onto 15m index",
            "prefix_warm": "6mo prefix > 60d gate lookback, so first OOS H6 close has a warm gate",
        },
    }

    out_path = reports_dir / "adaptrend_v2_imp_regime_gate_vol.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
