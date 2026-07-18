"""
AdaptiveTrend-v1 EXTENDED sweep — Tasks A-E per spec.

Extends the prior OOS sweep in two dimensions:
  1. Lower alpha grid: L∈{3,4,5} × theta∈{0.015,0.02,0.025} × alpha∈{1.0,1.25,1.5,1.75,2.0}
     (2.0 retained as prior winner boundary for continuity check)
  2. More OOS windows: all windows 2020_H2 → 2025_H2 that have both 15m + funding data.

DO NOT import from or call adaptrend_oos_sweep.main() — different output files.
Prior outputs (adaptive_trend_grid_sweep.json, etc.) are preserved untouched.

Run from repo root:
    .venv/bin/python tools/adaptrend_extended_sweep.py

All outputs to reports/:
  - adaptive_trend_lower_alpha_grid.json
  - adaptive_trend_walk_forward_extended.json
  - adaptive_trend_extended_psr.json
  ADAPTIVE_TREND_EXTENDED_VERDICT.md (repo root)
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools._adaptrend_run import run as _run_window  # noqa: E402
from tools.aggregate import aggregate_windows  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402


def _canonical_window_psr(
    per_window: list,
    *,
    aggregation_method: str = "v2_equity_curve",
) -> dict:
    """Canonical window-level PSR dual-emit for the extended AdaptiveTrend sweep.

    Maps each ``_adaptrend_run.run`` result's ``net_return_pct`` to the canonical
    per-window ``return_pct`` and aggregates via ``aggregate_windows``
    (contiguous=False -> Lo no-op across disjoint windows). Per-trade series are
    unavailable from this runner so ``pnl_pct`` / ``eq_impact_pnl_pct`` are
    empty. ``block["psr_walkforward"]`` is the canonical headline PSR and equals
    ``compute_psr(block["per_window_return_pct"], contiguous=False)`` bit-for-bit.
    """
    pw = [
        {
            "label":             w.get("label"),
            "return_pct":        float(w.get("net_return_pct", 0.0) or 0.0),
            "trades":            int(w.get("trades", 0) or 0),
            "pnl_pct":           [],
            "eq_impact_pnl_pct": [],
        }
        for w in per_window
    ]
    return aggregate_windows(pw, aggregation_method=aggregation_method)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASH = 1_000_000
_BTC_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
_BTC_FUNDING = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"

# Extended window set — all windows with confirmed 15m + funding coverage.
# Prior 5: 2022_H1, 2023_H1, 2024_H1, 2024_H2, 2025_H1
# Newly added: 2020_H2, 2021_H1, 2021_H2, 2022_H2, 2023_H2, 2025_H2
WINDOWS_FULL = [
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

# Lower-alpha grid (Task B per spec)
L_VALUES = [3, 4, 5]
THETA_VALUES = [0.015, 0.02, 0.025]
ALPHA_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0]

MIN_TRADES_FLOOR = 100   # relaxed from 150 — more windows but some may be thinner

# Prior winner for reference comparison
PRIOR_WINNER = {"L": 4, "theta": 0.02, "alpha": 2.0}

# Promotion win gate — ≥70% of extended windows positive
WIN_GATE_PCT = 0.70


# ---------------------------------------------------------------------------
# Helpers (copied from sweep to avoid importing the old file's module-level
# OOS_WINDOWS constant which would contaminate results)
# ---------------------------------------------------------------------------

def compounded(net_pcts: list[float]) -> float:
    """Compounded return across independent windows (each starts at $1M)."""
    result = 1.0
    for r in net_pcts:
        result *= (1.0 + r / 100.0)
    return (result - 1.0) * 100.0


def _run_variant(
    L: int,
    theta: float,
    alpha: float,
    windows: list = WINDOWS_FULL,
    parquet: Path = _BTC_PARQUET,
    funding_parquet: Path = _BTC_FUNDING,
) -> dict:
    """Run one (L, theta, alpha) variant across all extended windows."""
    config = {
        "momentum_lookback_h6": L,
        "theta_entry": theta,
        "alpha": alpha,
    }
    per_window = []

    for start, end, label in windows:
        r = _run_window(
            start=start,
            end=end,
            cash=CASH,
            config=config,
            parquet=parquet,
            funding_parquet=funding_parquet,
            label=label,
        )
        per_window.append(r)

    net_pcts = [w["net_return_pct"] for w in per_window]
    win_windows = sum(1 for p in net_pcts if p > 0)
    comp = compounded(net_pcts)
    worst_dd = min(w["max_dd_pct"] for w in per_window)
    total_trades = sum(w["trades"] for w in per_window)
    mean_net = sum(net_pcts) / len(net_pcts)

    # LEGACY (v1): window-level PSR (n=11, contiguous=True — tiebreaker only;
    # per-trade PSR done in Task D). Kept verbatim as the gate/score input so
    # this shelved variant's verdict is unchanged by the metric migration.
    psr_result = compute_psr(
        np.array(net_pcts, dtype=float),
        sr_hurdle=0.0,
        confidence=0.95,
    )

    # CANONICAL (v2): same window-level series via aggregate_windows
    # (contiguous=False -> Lo no-op across disjoint OOS windows). Additive.
    canon = _canonical_window_psr(per_window, aggregation_method="v2_equity_curve")

    eligible = total_trades >= MIN_TRADES_FLOOR
    passes = (
        comp > 0
        and win_windows >= math.floor(WIN_GATE_PCT * len(windows))
        and worst_dd > -10.0
        and psr_result["psr_vs_hurdle"] > 0.5
    )
    score = comp * min(1.0, psr_result["psr_vs_hurdle"]) if eligible else -1e9

    return {
        "L": L,
        "theta": theta,
        "alpha": alpha,
        "config": config,
        "per_window": per_window,
        "summary": {
            "compounded_net_pct": round(comp, 4),
            "win_windows": win_windows,
            "total_windows": len(windows),
            "total_trades": total_trades,
            "mean_net_pct": round(mean_net, 4),
            "worst_dd_pct": round(worst_dd, 4),
            "psr": psr_result,
            "legacy_psr_stitched": psr_result,
            "canonical": canon,
            "aggregation_method": canon["aggregation_method"],
            "eligible": eligible,
            "passes_gates": passes,
            "score": round(score, 6),
        },
    }


# ---------------------------------------------------------------------------
# Task A — Check history availability (report only)
# ---------------------------------------------------------------------------

def check_history() -> dict:
    df = pd.read_parquet(_BTC_PARQUET)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    f = pd.read_parquet(_BTC_FUNDING)
    if f.index.tz is not None:
        f.index = f.index.tz_localize(None)

    window_checks = []
    for start, end, label in WINDOWS_FULL:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        bars = len(df.loc[s:e])
        frows = len(f.loc[s:e])
        window_checks.append({
            "label": label,
            "start": start,
            "end": end,
            "bars_15m": bars,
            "funding_rows": frows,
            "usable": bars > 100 and frows > 0,
        })
        print(f"  {label}: 15m_bars={bars} funding_rows={frows}", flush=True)

    return {
        "btc_15m_min": str(df.index.min()),
        "btc_15m_max": str(df.index.max()),
        "btc_15m_len": len(df),
        "funding_min": str(f.index.min()),
        "funding_max": str(f.index.max()),
        "funding_len": len(f),
        "windows_checked": window_checks,
    }


# ---------------------------------------------------------------------------
# Task B — Lower alpha grid sweep (45 cells × 11 windows)
# ---------------------------------------------------------------------------

def run_lower_alpha_grid() -> dict:
    grid = list(product(L_VALUES, THETA_VALUES, ALPHA_VALUES))
    total = len(grid)
    print(f"[grid] {total} variants × {len(WINDOWS_FULL)} windows = {total*len(WINDOWS_FULL)} backtests", flush=True)

    results = []
    for idx, (L, theta, alpha) in enumerate(grid, 1):
        print(f"[grid] ({idx}/{total}) L={L} theta={theta} alpha={alpha} ...", flush=True)
        variant = _run_variant(L, theta, alpha)
        results.append(variant)
        print(
            f"  -> compounded={variant['summary']['compounded_net_pct']:.2f}% "
            f"wins={variant['summary']['win_windows']}/{len(WINDOWS_FULL)} "
            f"trades={variant['summary']['total_trades']} "
            f"passes={variant['summary']['passes_gates']}",
            flush=True,
        )

    results_sorted = sorted(results, key=lambda x: x["summary"]["score"], reverse=True)

    # Winner selection: most central in any 5pp plateau among top eligible
    eligible = [v for v in results_sorted if v["summary"]["eligible"]]
    winner = _pick_central_winner(eligible) if eligible else results_sorted[0]

    # Is the winner at the new floor?
    at_new_floor = winner["alpha"] == min(ALPHA_VALUES)

    print(f"\n[grid] Winner: L={winner['L']} theta={winner['theta']} alpha={winner['alpha']}", flush=True)
    print(f"  compounded={winner['summary']['compounded_net_pct']:.2f}%  at_new_floor={at_new_floor}", flush=True)

    out = {
        "task": "B_lower_alpha_grid",
        "parameters": {
            "L_values": L_VALUES,
            "theta_values": THETA_VALUES,
            "alpha_values": ALPHA_VALUES,
            "cash": CASH,
            "windows": [{"start": s, "end": e, "label": l} for s, e, l in WINDOWS_FULL],
            "min_trades_floor": MIN_TRADES_FLOOR,
            "win_gate_pct": WIN_GATE_PCT,
        },
        "variants": results_sorted,
        "winner": {
            "L": winner["L"],
            "theta": winner["theta"],
            "alpha": winner["alpha"],
            "config": winner["config"],
            "summary": winner["summary"],
            "per_window": winner["per_window"],
            "at_alpha_floor": at_new_floor,
        },
        "prior_winner": PRIOR_WINNER,
    }

    out_path = ROOT / "reports" / "adaptive_trend_lower_alpha_grid.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[grid] Saved to {out_path}", flush=True)
    return out


def _pick_central_winner(ranked: list[dict]) -> dict:
    """Pick the most central cell within any 5pp compounded-return plateau."""
    if not ranked:
        return ranked[0]
    top_score = ranked[0]["summary"]["compounded_net_pct"]
    plateau = [v for v in ranked if top_score - v["summary"]["compounded_net_pct"] <= 5.0]
    if len(plateau) <= 1:
        return ranked[0]
    # Prefer cells not at alpha floor or ceiling
    floor_alpha = min(ALPHA_VALUES)
    ceil_alpha = max(ALPHA_VALUES)
    interior = [v for v in plateau if v["alpha"] != floor_alpha and v["alpha"] != ceil_alpha]
    if interior:
        # Among interior, pick highest score
        return sorted(interior, key=lambda v: v["summary"]["score"], reverse=True)[0]
    return plateau[0]  # fallback to highest score


# ---------------------------------------------------------------------------
# Task C — Extended walk-forward on winner config
# ---------------------------------------------------------------------------

def run_walk_forward_extended(winner_config: dict) -> dict:
    df_btc = pd.read_parquet(_BTC_PARQUET)
    if df_btc.index.tz is not None:
        df_btc.index = df_btc.index.tz_localize(None)

    data_start = df_btc.index[0]
    data_end = df_btc.index[-1]

    # Earliest possible test start = data_start + 6mo train
    actual_start = data_start + pd.Timedelta(days=182)
    actual_end = data_end.normalize()  # through all available data

    print(f"[wf] Data: {data_start.date()} → {data_end.date()}", flush=True)
    print(f"[wf] Test windows: {actual_start.date()} → {actual_end.date()}", flush=True)

    # Generate quarterly test windows
    wf_windows = []
    test_start = actual_start.normalize()

    while test_start + pd.Timedelta(days=89) <= actual_end:
        train_start = test_start - pd.Timedelta(days=182)
        train_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        if test_end > actual_end:
            test_end = actual_end
        if (test_end - test_start).days < 60:
            break
        wf_windows.append({
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        })
        test_start = test_start + pd.DateOffset(months=3)

    print(f"[wf] {len(wf_windows)} walk-forward windows", flush=True)

    test_results = []
    all_net_pcts = []

    for i, w in enumerate(wf_windows, 1):
        print(f"[wf] ({i}/{len(wf_windows)}) {w['test_start']} .. {w['test_end']}", flush=True)
        r = _run_window(
            start=w["test_start"],
            end=w["test_end"],
            cash=CASH,
            config=winner_config,
            label=f"wf_{i:02d}",
        )
        r["train_start"] = w["train_start"]
        r["train_end"] = w["train_end"]
        r["test_start"] = w["test_start"]
        r["test_end"] = w["test_end"]
        test_results.append(r)
        all_net_pcts.append(r["net_return_pct"])
        print(f"  -> net={r['net_return_pct']:.2f}% trades={r['trades']}", flush=True)

    pos = sum(1 for p in all_net_pcts if p > 0)
    total = len(all_net_pcts)
    pct_pos = pos / total if total else 0.0
    agg_comp = compounded(all_net_pcts)
    # LEGACY (v1): WF PSR on the per-quarter net return series (contiguous=True).
    # Kept verbatim as the verdict input.
    psr_wf = compute_psr(np.array(all_net_pcts, dtype=float), sr_hurdle=0.0, confidence=0.95)
    # CANONICAL (v2_walkforward): same per-quarter series via aggregate_windows
    # (contiguous=False -> Lo no-op across disjoint quarters). Additive.
    canon_wf = _canonical_window_psr(test_results, aggregation_method="v2_walkforward")

    # Relaxed gate: ≥70% positive (was 75% in prior sweep)
    # (UNCHANGED — reads legacy psr_wf for verdict stability).
    verdict = (
        "pass"
        if pct_pos >= WIN_GATE_PCT and agg_comp > 0 and psr_wf["psr_vs_hurdle"] > 0.5
        else "fail"
    )

    print(f"[wf] {pos}/{total} positive ({pct_pos:.1%}), agg={agg_comp:.2f}%, verdict={verdict}", flush=True)

    out = {
        "task": "C_walk_forward_extended",
        "winner_config": winner_config,
        "windows": test_results,
        "aggregate": {
            "compounded_net_pct": round(agg_comp, 4),
            "positive_windows": pos,
            "total_windows": total,
            "pct_positive": round(pct_pos, 4),
            "psr": psr_wf,
            "legacy_psr_stitched": psr_wf,
            "canonical": canon_wf,
            "aggregation_method": canon_wf["aggregation_method"],
        },
        "verdict": verdict,
        "win_gate_pct": WIN_GATE_PCT,
    }

    out_path = ROOT / "reports" / "adaptive_trend_walk_forward_extended.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[wf] Saved to {out_path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Task D — Aggregate per-trade PSR on winner config (all extended windows)
# ---------------------------------------------------------------------------

def run_aggregate_psr(winner_config: dict, grid_out: dict) -> dict:
    """Collect pooled per-trade returns from winner across all extended windows."""
    import tempfile, os

    trades_path = ROOT / "reports" / "_extended_winner_trades.csv"
    # Delete if exists so we don't accumulate stale rows
    if trades_path.exists():
        trades_path.unlink()

    total_trades = 0
    per_window_d: list = []
    for start, end, label in WINDOWS_FULL:
        r = _run_window(
            start=start,
            end=end,
            cash=CASH,
            config=winner_config,
            save_trades=trades_path,
            label=label,
        )
        total_trades += r["trades"]
        per_window_d.append(r)

    # LEGACY (v1): pooled per-trade PSR — this is the genuine STITCHED
    # per-trade ReturnPct union across DISJOINT windows. N-inflated and
    # sizing-blind. Kept verbatim as `legacy_psr_stitched` / the headline
    # `psr_result` for verdict stability on this already-shelved strategy.
    if not trades_path.exists() or total_trades == 0:
        psr_result = compute_psr(np.array([], dtype=float))
        interpretation = "no_trades"
    else:
        df_trades = pd.read_csv(trades_path)
        if "pnl_pct" not in df_trades.columns:
            # trades file uses pnl_pct already (see _adaptrend_run save_trades code)
            pnl_pct = df_trades.get("ReturnPct", pd.Series(dtype=float)) * 100.0
        else:
            pnl_pct = df_trades["pnl_pct"].dropna()
        returns = pnl_pct.values.astype(float)
        psr_result = compute_psr(returns, sr_hurdle=0.0, confidence=0.95)
        interpretation = psr_result["interpretation"]

    # CANONICAL (v2): window-level PSR from the winner's per-window net returns
    # across the 11 extended windows (contiguous=False -> Lo no-op). This is the
    # canonical fix for the stitched per-trade PSR above — defeats N-inflation.
    canon = _canonical_window_psr(per_window_d, aggregation_method="v2_equity_curve")

    prior_min_trl = 400
    n_new = psr_result["n_trades"]
    min_trl_new = psr_result["min_trl"]
    mintrl_gap = max(0, min_trl_new - n_new)

    print(f"[psr] Pooled trades: {n_new}, MinTRL: {min_trl_new}, gap: {mintrl_gap}", flush=True)

    out = {
        "task": "D_aggregate_psr",
        "winner_config": winner_config,
        "windows_used": len(WINDOWS_FULL),
        "psr_result": psr_result,                 # legacy stitched per-trade (verdict input)
        "legacy_psr_stitched": psr_result,        # explicit alias for observability
        "canonical": canon,                       # v2 window-level dual-emit
        "aggregation_method": canon["aggregation_method"],
        "n_trades": n_new,
        "min_trl": min_trl_new,
        "mintrl_gap": mintrl_gap,
        "mintrl_satisfied": n_new >= min_trl_new,
        "prior_min_trl": prior_min_trl,
        "trades_csv": str(trades_path),
    }

    out_path = ROOT / "reports" / "adaptive_trend_extended_psr.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[psr] Saved to {out_path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Task E — Final extended verdict
# ---------------------------------------------------------------------------

def _promotion_recommendation(
    winner: dict,
    wf: dict,
    psr_out: dict,
) -> str:
    g = winner["summary"]
    comp = g["compounded_net_pct"]
    wins = g["win_windows"]
    n_win = len(WINDOWS_FULL)
    worst_dd = g["worst_dd_pct"]
    psr_val = psr_out["psr_result"]["psr_vs_hurdle"]
    n_trades = psr_out["n_trades"]
    min_trl = psr_out["min_trl"]
    at_floor = winner.get("at_alpha_floor", False)

    # Grid-level gates
    grid_passes = (
        comp > 0
        and wins >= math.floor(WIN_GATE_PCT * n_win)
        and worst_dd > -10.0
        and psr_val > 0.5
    )
    if not grid_passes:
        return "shelf"

    wf_verdict = wf["verdict"]
    if wf_verdict == "fail":
        return "iterate_more"

    # Promotion gates per spec
    if (
        comp > 20.0
        and wins / n_win >= WIN_GATE_PCT
        and psr_val > 0.95
        and n_trades >= min_trl
        and not at_floor
    ):
        return "promote_to_dry_run_leg"
    else:
        return "iterate_more"


def write_extended_verdict(
    history_info: dict,
    grid_out: dict,
    wf_out: dict,
    psr_out: dict,
) -> None:
    winner = grid_out["winner"]
    wg = winner["summary"]
    wf_agg = wf_out["aggregate"]
    wf_windows = wf_out["windows"]
    wf_verdict = wf_out["verdict"]
    psr_result = psr_out["psr_result"]
    n_trades = psr_out["n_trades"]
    min_trl = psr_out["min_trl"]
    mintrl_gap = psr_out["mintrl_gap"]
    mintrl_ok = psr_out["mintrl_satisfied"]

    btc_comp = wg["compounded_net_pct"]
    promotion = _promotion_recommendation(winner, wf_out, psr_out)
    at_floor_msg = " (**AT NEW ALPHA FLOOR**)" if winner.get("at_alpha_floor") else ""

    variants = grid_out["variants"]
    top5 = variants[:5]
    bottom3 = variants[-3:]

    n_win_total = len(WINDOWS_FULL)
    win_gate_n = math.floor(WIN_GATE_PCT * n_win_total)

    lines = []
    lines.append("# ADAPTIVE_TREND_EXTENDED_VERDICT")
    lines.append("")
    lines.append("Research only. Not wired to live bot.")
    lines.append("")

    lines.append("## TL;DR")
    lines.append(
        f"- **Lower-alpha winner**: L={winner['L']}, theta={winner['theta']}, "
        f"alpha={winner['alpha']}{at_floor_msg}, "
        f"compounded={btc_comp:.2f}%, {wg['win_windows']}/{n_win_total} wins, "
        f"worst DD={wg['worst_dd_pct']:.2f}%"
    )
    mintrl_status = (
        f"SATISFIED (n={n_trades} >= MinTRL={min_trl})"
        if mintrl_ok
        else f"NOT MET (n={n_trades}, MinTRL={min_trl}, gap={mintrl_gap})"
    )
    lines.append(f"- **n_trades vs MinTRL**: {mintrl_status}, per-trade PSR={psr_result['psr_vs_hurdle']:.3f}")
    lines.append(
        f"- **Walk-forward extended**: {wf_agg['positive_windows']}/{wf_agg['total_windows']} positive "
        f"({wf_agg['pct_positive']:.1%}), agg_comp={wf_agg['compounded_net_pct']:.2f}%, verdict={wf_verdict}"
    )
    lines.append(f"- **Promotion recommendation**: `{promotion}`")
    lines.append("")

    lines.append("## History Availability")
    lines.append("")
    lines.append(f"- BTC 15m: {history_info['btc_15m_min']} → {history_info['btc_15m_max']} ({history_info['btc_15m_len']:,} bars)")
    lines.append(f"- Funding: {history_info['funding_min']} → {history_info['funding_max']} ({history_info['funding_len']:,} rows)")
    lines.append("")
    lines.append("### Extended window set (11 windows used)")
    lines.append("")
    lines.append("| Window | Start | End | 15m bars | Funding rows | Used |")
    lines.append("|--------|-------|-----|----------|--------------|------|")
    for w in history_info["windows_checked"]:
        lines.append(
            f"| {w['label']} | {w['start']} | {w['end']} | "
            f"{w['bars_15m']:,} | {w['funding_rows']} | {'yes' if w['usable'] else 'NO'} |"
        )
    lines.append("")

    lines.append("## Lower-Alpha Grid Sweep Summary")
    lines.append("")
    lines.append(
        f"Grid: L∈{L_VALUES} × theta∈{THETA_VALUES} × alpha∈{ALPHA_VALUES} = "
        f"{len(grid_out['variants'])} variants × {n_win_total} OOS windows."
    )
    lines.append(f"Eligibility gate: total_trades >= {MIN_TRADES_FLOOR}. Win gate: >= {win_gate_n}/{n_win_total} ({WIN_GATE_PCT:.0%}).")
    lines.append("Rank metric: `compounded × min(1, PSR)` (window-level PSR, n=11 — tiebreaker only).")
    lines.append(f"Prior winner for delta: L=4, theta=0.02, alpha=2.0, comp=43.48% (5 windows).")
    lines.append("")

    lines.append("### Top 5 variants")
    lines.append("")
    lines.append(f"| Rank | L | theta | alpha | Comp% | Wins/{n_win_total} | Trades | WorstDD% | Score |")
    lines.append(f"|------|---|-------|-------|-------|-------|--------|----------|-------|")
    for i, v in enumerate(top5, 1):
        s = v["summary"]
        lines.append(
            f"| {i} | {v['L']} | {v['theta']} | {v['alpha']} | "
            f"{s['compounded_net_pct']:.2f} | {s['win_windows']}/{n_win_total} | "
            f"{s['total_trades']} | {s['worst_dd_pct']:.2f} | {s['score']:.4f} |"
        )
    lines.append("")
    lines.append("### Bottom 3 variants (honesty)")
    lines.append("")
    lines.append(f"| Rank | L | theta | alpha | Comp% | Wins/{n_win_total} | Trades | WorstDD% | Score |")
    lines.append(f"|------|---|-------|-------|-------|-------|--------|----------|-------|")
    for i, v in enumerate(bottom3, 1):
        rank = len(variants) - 3 + i
        s = v["summary"]
        lines.append(
            f"| {rank} | {v['L']} | {v['theta']} | {v['alpha']} | "
            f"{s['compounded_net_pct']:.2f} | {s['win_windows']}/{n_win_total} | "
            f"{s['total_trades']} | {s['worst_dd_pct']:.2f} | {s['score']:.4f} |"
        )
    lines.append("")

    lines.append("## Extended OOS Table (Winner Config)")
    lines.append("")
    lines.append(f"Winner config: L={winner['L']}, theta={winner['theta']}, alpha={winner['alpha']}")
    lines.append("")
    lines.append("| Window | Start | End | Net% | Trades | WinRate% | MaxDD% |")
    lines.append("|--------|-------|-----|------|--------|----------|--------|")
    for pw in winner["per_window"]:
        lines.append(
            f"| {pw.get('label','?')} | {pw['start']} | {pw['end']} | "
            f"{pw['net_return_pct']:.2f} | {pw['trades']} | "
            f"{pw.get('win_rate_pct', 0.0):.1f} | {pw['max_dd_pct']:.2f} |"
        )
    # Summary row
    net_pcts_winner = [pw["net_return_pct"] for pw in winner["per_window"]]
    comp_winner = compounded(net_pcts_winner)
    total_tr = sum(pw["trades"] for pw in winner["per_window"])
    lines.append(f"| **TOTAL** | | | **{comp_winner:.2f}** | **{total_tr}** | | **{wg['worst_dd_pct']:.2f}** |")
    lines.append("")

    lines.append("## Walk-Forward Extended")
    lines.append("")
    lines.append(f"Sliding 6-month train / 3-month test, quarterly advance. Data through {history_info['btc_15m_max'][:10]}.")
    lines.append(f"Win gate relaxed to ≥{WIN_GATE_PCT:.0%} (was 75% in prior sweep).")
    lines.append(
        f"**Verdict: {wf_verdict}** ({wf_agg['positive_windows']}/{wf_agg['total_windows']} positive, "
        f"agg={wf_agg['compounded_net_pct']:.2f}%, PSR={wf_agg['psr']['psr_vs_hurdle']:.3f})"
    )
    lines.append("")
    lines.append("| # | Test Period | Net% | Trades |")
    lines.append("|---|-------------|------|--------|")
    for i, w in enumerate(wf_windows, 1):
        lines.append(
            f"| {i} | {w.get('test_start','?')} .. {w.get('test_end','?')} | "
            f"{w['net_return_pct']:.2f} | {w['trades']} |"
        )
    lines.append("")

    lines.append("## Aggregate PSR / MinTRL")
    lines.append("")
    lines.append(f"Pooled per-trade gross returns from winner across all {n_win_total} extended windows.")
    lines.append(f"Note: per-trade funding is modeled at window-aggregate level, not per-trade — consistent with prior 0.905 PSR basis.")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| n_trades | {n_trades} |")
    lines.append(f"| Point Sharpe (per-trade) | {psr_result['point_sharpe']:.4f} |")
    lines.append(f"| PSR vs SR=0 | {psr_result['psr_vs_hurdle']:.3f} |")
    lines.append(f"| MinTRL | {min_trl} |")
    lines.append(f"| MinTRL satisfied | {'YES' if mintrl_ok else f'NO (need {mintrl_gap} more trades)'} |")
    lines.append(f"| Skew | {psr_result['skew']:.3f} |")
    lines.append(f"| Kurt (raw) | {psr_result['kurt']:.3f} |")
    lines.append(f"| Interpretation | {psr_result['interpretation']} |")
    lines.append(f"| Prior MinTRL (n=255) | 400 |")
    lines.append("")

    lines.append("## Promotion Recommendation")
    lines.append("")
    lines.append(f"**`{promotion}`**")
    lines.append("")

    at_floor = winner.get("at_alpha_floor", False)

    if promotion == "promote_to_dry_run_leg":
        lines.append("All promotion gates cleared:")
        lines.append(f"- compounded > 20%: {btc_comp:.2f}%")
        lines.append(f"- wins ≥ {WIN_GATE_PCT:.0%}: {wg['win_windows']}/{n_win_total}")
        lines.append(f"- walk-forward ≥ {WIN_GATE_PCT:.0%}: {wf_verdict}")
        lines.append(f"- PSR > 0.95: {psr_result['psr_vs_hurdle']:.3f}")
        lines.append(f"- n_trades ≥ MinTRL: {n_trades} ≥ {min_trl}")
        lines.append(f"- alpha NOT at floor: {'YES (interior)' if not at_floor else 'NO'}")
    elif promotion == "iterate_more":
        lines.append("Grid winner passes core OOS gates, but not all promotion gates:")
        if wf_verdict == "fail":
            lines.append(
                f"- Walk-forward failed: {wf_agg['positive_windows']}/{wf_agg['total_windows']} "
                f"({wf_agg['pct_positive']:.1%}) < {WIN_GATE_PCT:.0%} gate"
            )
        if btc_comp <= 20.0:
            lines.append(f"- compounded ({btc_comp:.2f}%) does not exceed 20% gate")
        if psr_result["psr_vs_hurdle"] <= 0.95:
            lines.append(f"- PSR ({psr_result['psr_vs_hurdle']:.3f}) < 0.95 gate")
        if not mintrl_ok:
            lines.append(f"- n_trades ({n_trades}) < MinTRL ({min_trl}) — gap of {mintrl_gap}")
        if at_floor:
            lines.append(f"- alpha={winner['alpha']} is the new grid floor — optimum still not bracketed; extend down further")
    else:
        lines.append("Grid winner failed one or more required core gates. Strategy shelved.")
        if btc_comp <= 0:
            lines.append(f"- compounded negative: {btc_comp:.2f}%")
        if wg["win_windows"] < math.floor(WIN_GATE_PCT * n_win_total):
            lines.append(f"- wins ({wg['win_windows']}/{n_win_total}) below floor")
        if wg["worst_dd_pct"] <= -10.0:
            lines.append(f"- worst DD ({wg['worst_dd_pct']:.2f}%) breaches -10% gate")

    lines.append("")
    lines.append("## Delta vs Prior Sweep")
    lines.append("")
    lines.append("| Metric | Prior (5 windows, α≥2.0) | Extended (11 windows) | Change |")
    lines.append("|--------|--------------------------|----------------------|--------|")
    lines.append(f"| Winner alpha | 2.0 | {winner['alpha']} | {'+' if winner['alpha'] >= 2.0 else ''}{winner['alpha']-2.0:.2f} |")
    lines.append(f"| Compounded net% | 43.48% | {btc_comp:.2f}% | {btc_comp-43.48:+.2f}pp |")
    lines.append(f"| Win windows | 5/5 | {wg['win_windows']}/{n_win_total} | — |")
    lines.append(f"| Total trades | 255 | {n_trades} | +{n_trades-255} |")
    prior_mintrl = 400
    lines.append(f"| MinTRL | 400 | {min_trl} | {min_trl-prior_mintrl:+d} |")
    lines.append(f"| MinTRL satisfied | NO | {'YES' if mintrl_ok else 'NO'} | — |")
    lines.append(f"| Walk-forward % positive | 55% (11/20) | {wf_agg['pct_positive']:.1%} ({wf_agg['positive_windows']}/{wf_agg['total_windows']}) | — |")
    lines.append("")

    lines.append("## Open Questions")
    lines.append("")
    if at_floor:
        lines.append(
            "1. **Alpha still at floor**: winner's alpha is the new grid floor — true optimum still not bracketed. "
            "Run alpha ∈ {0.5, 0.75, 1.0} extension before promotion."
        )
    lines.append(
        "2. **Walk-forward temporal pattern**: inspect per-window table for secular degradation "
        "(later windows systematically weaker = regime change in BTC MOM structure)."
    )
    lines.append(
        "3. **Monthly re-optimisation (Algorithm 2)**: paper's adaptive layer would lift trade count "
        "and likely improve walk-forward consistency; not implemented."
    )
    lines.append(
        "4. **Live wiring cost**: AdaptiveTrend requires H6 resampling inside live bot — "
        "different architecture from multifactor's bar-by-bar approach."
    )
    lines.append(
        "5. **2025_H2 coverage**: data runs through 2026-05-29; 2025_H2 window used in full."
    )
    lines.append("")
    lines.append("---")
    lines.append("*Research only. Not wired to live bot. No risk.py or config/params.yaml touched.*")

    out_path = ROOT / "ADAPTIVE_TREND_EXTENDED_VERDICT.md"
    out_path.write_text("\n".join(lines))
    print(f"[verdict] Written to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== AdaptiveTrend-v1 EXTENDED Sweep (lower alpha + more windows) ===", flush=True)

    print("\n--- Task A: History check ---", flush=True)
    history_info = check_history()
    print(f"  15m: {history_info['btc_15m_min']} → {history_info['btc_15m_max']}", flush=True)
    print(f"  funding: {history_info['funding_min']} → {history_info['funding_max']}", flush=True)

    print("\n--- Task B: Lower-alpha grid (45 cells × 11 windows) ---", flush=True)
    grid_out = run_lower_alpha_grid()
    winner_config = grid_out["winner"]["config"]

    print("\n--- Task C: Extended walk-forward ---", flush=True)
    wf_out = run_walk_forward_extended(winner_config)

    print("\n--- Task D: Aggregate per-trade PSR ---", flush=True)
    psr_out = run_aggregate_psr(winner_config, grid_out)

    print("\n--- Task E: Write extended verdict ---", flush=True)
    write_extended_verdict(history_info, grid_out, wf_out, psr_out)

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
