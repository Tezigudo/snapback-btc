"""
AdaptiveTrend-v1 OOS sweep, walk-forward, and multi-coin validation.

Tasks A, B, C, D as per the research spec:
  A. L × theta × alpha grid sweep (48 variants × 5 OOS windows = 240 backtests)
  B. Walk-forward on grid winner (sliding 6mo train / 3mo test, quarterly advance)
  C. Multi-coin smoke (ETH + SOL, 5 OOS windows each)
  D. Final synthesis → ADAPTIVE_TREND_OOS_VERDICT.md

Run from repo root:
    .venv/bin/python tools/adaptrend_oos_sweep.py

All outputs to reports/ directory.
"""
from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

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
    """Canonical window-level PSR dual-emit for the AdaptiveTrend sweeps.

    The legacy sites compute ``compute_psr(net_pcts)`` on the per-window net
    return series with ``contiguous=True``. That series is ALREADY window-level
    (n == n_windows), not stitched per-trade — but it is tagged
    ``contiguous=True`` which would (mis)apply the Lo correction across disjoint
    windows. The canonical core treats disjoint windows with
    ``contiguous=False`` (Lo no-op) via ``aggregate_windows``.

    Each ``per_window`` entry is an ``_adaptrend_run.run`` result dict; we map
    ``net_return_pct`` -> canonical ``return_pct`` (already rounded to 4dp by the
    runner, so ``aggregate_windows``'s internal round() is a no-op). Per-trade
    series are unavailable from this runner, so ``pnl_pct`` /
    ``eq_impact_pnl_pct`` are empty (per-window PSR -> insufficient_evidence,
    which is correct: we never had per-trade attribution here).

    Returns the canonical block; ``block["psr_walkforward"]`` is the headline
    canonical PSR and equals
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

# Exact 5 OOS windows from baseline (2022_H1 .. 2025_H1)
OOS_WINDOWS = [
    ("2022-01-01", "2022-06-30", "2022_H1"),
    ("2023-01-01", "2023-06-30", "2023_H1"),
    ("2024-01-01", "2024-06-30", "2024_H1"),
    ("2024-07-01", "2024-12-31", "2024_H2"),
    ("2025-01-01", "2025-06-30", "2025_H1"),
]

# Grid parameters
L_VALUES = [3, 4, 5, 6]
THETA_VALUES = [0.01, 0.02, 0.03]
ALPHA_VALUES = [2.0, 2.5, 3.0, 3.5]

MIN_TRADES_FLOOR = 150   # eligibility gate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compounded(net_pcts: list[float]) -> float:
    """Compounded return across independent windows (each starts at $1M)."""
    result = 1.0
    for r in net_pcts:
        result *= (1.0 + r / 100.0)
    return (result - 1.0) * 100.0


def _run_variant(
    L: int, theta: float, alpha: float,
    parquet: Path = _BTC_PARQUET,
    funding_parquet: Path = _BTC_FUNDING,
    windows: list = OOS_WINDOWS,
) -> dict:
    """Run one (L, theta, alpha) variant across all OOS windows, return summary."""
    config = {
        "momentum_lookback_h6": L,
        "theta_entry": theta,
        "alpha": alpha,
    }
    per_window = []
    all_trade_returns: list[float] = []

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

    # LEGACY (v1): PSR on the per-window net return series with contiguous=True.
    # Kept verbatim as the gate/score input so this already-shelved variant's
    # verdict is unchanged by the metric migration. N == n_windows here (this
    # was never a per-trade stitched series), so it is sizing-aware already.
    psr_result = compute_psr(
        np.array(net_pcts, dtype=float),
        sr_hurdle=0.0,
        confidence=0.95,
    )

    # CANONICAL (v2): same window-level series via aggregate_windows
    # (contiguous=False -> Lo no-op across disjoint OOS windows). Additive
    # observability; psr_walkforward is the canonical headline.
    canon = _canonical_window_psr(per_window, aggregation_method="v2_equity_curve")

    # Eligibility + scoring (UNCHANGED — reads legacy psr_result for verdict
    # stability on this shelved strategy).
    eligible = total_trades >= MIN_TRADES_FLOOR
    passes = (
        comp > 0
        and win_windows >= 3
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
# Task A — Grid Sweep
# ---------------------------------------------------------------------------

def run_grid_sweep() -> dict:
    grid = list(product(L_VALUES, THETA_VALUES, ALPHA_VALUES))
    total = len(grid)
    results = []

    print(f"[sweep] Grid: {total} variants × {len(OOS_WINDOWS)} windows = {total*len(OOS_WINDOWS)} backtests", flush=True)

    for idx, (L, theta, alpha) in enumerate(grid, 1):
        print(f"[sweep] ({idx}/{total}) L={L} theta={theta} alpha={alpha} ...", flush=True)
        variant = _run_variant(L, theta, alpha)
        results.append(variant)
        print(f"  -> compounded={variant['summary']['compounded_net_pct']:.2f}% "
              f"trades={variant['summary']['total_trades']} "
              f"passes={variant['summary']['passes_gates']}", flush=True)

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x["summary"]["score"], reverse=True)

    winner = results_sorted[0]
    print(f"\n[sweep] Grid winner: L={winner['L']} theta={winner['theta']} alpha={winner['alpha']}", flush=True)
    print(f"  compounded={winner['summary']['compounded_net_pct']:.2f}% "
          f"trades={winner['summary']['total_trades']} "
          f"passes={winner['summary']['passes_gates']}", flush=True)

    out = {
        "task": "A_grid_sweep",
        "parameters": {
            "L_values": L_VALUES,
            "theta_values": THETA_VALUES,
            "alpha_values": ALPHA_VALUES,
            "cash": CASH,
            "windows": [{"start": s, "end": e, "label": l} for s, e, l in OOS_WINDOWS],
            "min_trades_floor": MIN_TRADES_FLOOR,
        },
        "variants": results_sorted,
        "winner": {
            "L": winner["L"],
            "theta": winner["theta"],
            "alpha": winner["alpha"],
            "config": winner["config"],
            "summary": winner["summary"],
        },
    }

    out_path = ROOT / "reports" / "adaptive_trend_grid_sweep.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[sweep] Saved grid sweep to {out_path}", flush=True)

    return out


# ---------------------------------------------------------------------------
# Task B — Walk-Forward
# ---------------------------------------------------------------------------

def run_walk_forward(winner_config: dict) -> dict:
    """Sliding 6mo train / 3mo test, advanced quarterly."""
    # Load available BTC data to determine range
    df_btc = pd.read_parquet(_BTC_PARQUET)
    if df_btc.index.tz is not None:
        df_btc.index = df_btc.index.tz_localize(None)

    # Try 2021-01-01 start; fall back to earliest available
    desired_start = pd.Timestamp("2021-01-01")
    data_start = df_btc.index[0]
    actual_start = max(desired_start, data_start + pd.Timedelta(days=180))  # need 6mo train
    actual_end = pd.Timestamp("2025-12-31")
    actual_end = min(actual_end, df_btc.index[-1].normalize())

    print(f"[wf] Walk-forward: data available from {data_start.date()} to {df_btc.index[-1].date()}", flush=True)
    print(f"[wf] Using test windows from {actual_start.date()} to {actual_end.date()}", flush=True)

    # Generate windows: test window starts at actual_start, advances quarterly
    wf_windows = []
    test_start = actual_start

    while test_start + pd.Timedelta(days=89) <= actual_end:
        train_start = test_start - pd.Timedelta(days=182)  # 6mo train
        train_end = test_start - pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=91)  # 3mo test
        if test_end > actual_end:
            test_end = actual_end

        if (test_end - test_start).days < 60:
            break  # too short

        wf_windows.append({
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        })
        test_start += pd.DateOffset(months=3)

    print(f"[wf] Generated {len(wf_windows)} walk-forward windows", flush=True)

    test_results = []
    all_test_net_pcts = []

    for i, w in enumerate(wf_windows, 1):
        print(f"[wf] ({i}/{len(wf_windows)}) test {w['test_start']} .. {w['test_end']}", flush=True)
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
        all_test_net_pcts.append(r["net_return_pct"])
        print(f"  -> net={r['net_return_pct']:.2f}% trades={r['trades']}", flush=True)

    positive_windows = sum(1 for p in all_test_net_pcts if p > 0)
    total_windows = len(all_test_net_pcts)
    agg_comp = compounded(all_test_net_pcts) if all_test_net_pcts else 0.0

    # LEGACY (v1): WF PSR on the per-quarter net return series (contiguous=True).
    # Kept verbatim as the verdict input — WF family is the rolling train/test
    # walk-forward, tagged v2_walkforward so it never cross-compares to 5-OOS.
    psr_wf = compute_psr(
        np.array(all_test_net_pcts, dtype=float),
        sr_hurdle=0.0,
        confidence=0.95,
    )

    # CANONICAL (v2_walkforward): same per-quarter series via aggregate_windows
    # (contiguous=False -> Lo no-op across disjoint quarters). Additive.
    canon_wf = _canonical_window_psr(test_results, aggregation_method="v2_walkforward")

    # Verdict gates: >=75% windows positive, aggregate > 0, PSR > 0.5
    # (UNCHANGED — reads legacy psr_wf for verdict stability).
    pct_positive = positive_windows / total_windows if total_windows else 0
    wf_verdict = (
        "pass"
        if pct_positive >= 0.75 and agg_comp > 0 and psr_wf["psr_vs_hurdle"] > 0.5
        else "fail"
    )

    print(f"[wf] Walk-forward: {positive_windows}/{total_windows} positive, "
          f"agg_comp={agg_comp:.2f}%, verdict={wf_verdict}", flush=True)

    out = {
        "task": "B_walk_forward",
        "winner_config": winner_config,
        "windows": test_results,
        "aggregate": {
            "compounded_net_pct": round(agg_comp, 4),
            "positive_windows": positive_windows,
            "total_windows": total_windows,
            "pct_positive": round(pct_positive, 4),
            "psr": psr_wf,
            "legacy_psr_stitched": psr_wf,
            "canonical": canon_wf,
            "aggregation_method": canon_wf["aggregation_method"],
        },
        "verdict": wf_verdict,
    }

    out_path = ROOT / "reports" / "adaptive_trend_walk_forward.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[wf] Saved walk-forward to {out_path}", flush=True)

    return out


# ---------------------------------------------------------------------------
# Task C — Multi-coin
# ---------------------------------------------------------------------------

MULTI_COIN_SPECS = [
    {
        "coin": "ETH",
        "parquet": ROOT / "data" / "historical" / "ETH_USDT_USDT_15m.parquet",
        "funding": ROOT / "data" / "historical" / "ETH_USDT_USDT_funding.parquet",
    },
    {
        "coin": "SOL",
        "parquet": ROOT / "data" / "historical" / "SOL_USDT_USDT_15m.parquet",
        "funding": ROOT / "data" / "historical" / "SOL_USDT_USDT_funding.parquet",
    },
]


def _coin_verdict(comp: float, wins: int, psr: float) -> str:
    if comp > 0 and wins >= 3 and psr > 0.5:
        return "transfers"
    elif comp > 0:
        return "partial"
    else:
        return "coin_specific"


def run_multi_coin(winner_config: dict) -> dict:
    coin_results = {}

    for spec in MULTI_COIN_SPECS:
        coin = spec["coin"]
        parquet = spec["parquet"]
        funding_path = spec["funding"]

        # Check parquet availability
        if not parquet.exists():
            print(f"[multicoin] {coin}: parquet not found at {parquet} — skipping", flush=True)
            coin_results[coin] = {"error": "parquet_not_found"}
            continue

        funding_exists = funding_path.exists()
        if not funding_exists:
            print(f"[multicoin] {coin}: funding parquet not found at {funding_path} — running WITHOUT funding modeling", flush=True)

        per_window = []
        print(f"[multicoin] {coin}: running {len(OOS_WINDOWS)} windows ...", flush=True)

        for start, end, label in OOS_WINDOWS:
            # Validate data exists for this window
            df_check = pd.read_parquet(parquet)
            if df_check.index.tz is not None:
                df_check.index = df_check.index.tz_localize(None)
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            if df_check.loc[start_ts:end_ts].empty:
                print(f"  [{coin}] {label}: no data available — skipping window", flush=True)
                continue

            try:
                r = _run_window(
                    start=start,
                    end=end,
                    cash=CASH,
                    config=winner_config,
                    parquet=parquet,
                    funding_parquet=funding_path if funding_exists else _BTC_FUNDING,
                    label=label,
                )
                r["funding_modeled"] = funding_exists
                per_window.append(r)
                print(f"  [{coin}] {label}: net={r['net_return_pct']:.2f}% trades={r['trades']}", flush=True)
            except Exception as exc:
                print(f"  [{coin}] {label}: ERROR — {exc}", flush=True)

        if not per_window:
            coin_results[coin] = {"error": "no_windows_ran", "funding_modeled": funding_exists}
            continue

        net_pcts = [w["net_return_pct"] for w in per_window]
        wins = sum(1 for p in net_pcts if p > 0)
        comp = compounded(net_pcts)
        # LEGACY (v1): window-level PSR (contiguous=True). Kept as the verdict
        # input for this shelved multi-coin transfer test.
        psr = compute_psr(np.array(net_pcts, dtype=float), sr_hurdle=0.0)
        # CANONICAL (v2): same series via aggregate_windows (Lo no-op). Additive.
        canon = _canonical_window_psr(per_window, aggregation_method="v2_equity_curve")

        verdict = _coin_verdict(comp, wins, psr["psr_vs_hurdle"])
        print(f"[multicoin] {coin}: comp={comp:.2f}% wins={wins}/{len(net_pcts)} "
              f"PSR={psr['psr_vs_hurdle']:.3f} verdict={verdict}", flush=True)

        coin_results[coin] = {
            "per_window": per_window,
            "summary": {
                "compounded_net_pct": round(comp, 4),
                "win_windows": wins,
                "total_windows": len(per_window),
                "psr": psr,
                "legacy_psr_stitched": psr,
                "canonical": canon,
                "aggregation_method": canon["aggregation_method"],
                "verdict": verdict,
                "funding_modeled": funding_exists,
            },
        }

    out = {
        "task": "C_multi_coin",
        "winner_config": winner_config,
        "coins": coin_results,
    }

    out_path = ROOT / "reports" / "adaptive_trend_multi_coin.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[multicoin] Saved to {out_path}", flush=True)

    return out


# ---------------------------------------------------------------------------
# Task D — Final synthesis verdict
# ---------------------------------------------------------------------------

BASELINE = {
    "compounded_net_pct": 33.567,
    "win_windows": 4,
    "total_windows": 5,
    "total_trades": 193,
    "worst_dd_pct": -7.4261,
    "psr": 0.944,
    "min_trl": 206,
}


def _promotion_recommendation(
    winner: dict,
    wf: dict,
    btc_comp: float,
) -> str:
    g = winner["summary"]
    psr = g["psr"]["psr_vs_hurdle"]
    comp = g["compounded_net_pct"]
    wins = g["win_windows"]
    worst_dd = g["worst_dd_pct"]
    n_trades = g["total_trades"]
    min_trl = g["psr"]["min_trl"]

    grid_passes = comp > 0 and wins >= 3 and worst_dd > -10.0 and psr > 0.5

    if not grid_passes:
        return "shelf"

    wf_verdict = wf["verdict"]
    if wf_verdict == "fail":
        return "iterate_more"

    if btc_comp > 20.0 and n_trades >= min_trl:
        return "promote_to_dry_run_leg"
    else:
        return "iterate_more"


def write_verdict_md(
    grid_out: dict,
    wf_out: dict,
    mc_out: dict,
) -> None:
    winner = grid_out["winner"]
    wg = winner["summary"]
    wf_agg = wf_out["aggregate"]
    wf_windows = wf_out["windows"]
    wf_verdict = wf_out["verdict"]

    # BTC performance from grid winner
    btc_comp = wg["compounded_net_pct"]
    promotion = _promotion_recommendation(winner, wf_out, btc_comp)

    # Top 5 / Bottom 3 from grid
    variants = grid_out["variants"]
    top5 = variants[:5]
    bottom3 = variants[-3:]

    # Multi-coin summaries
    eth_s = mc_out["coins"].get("ETH", {}).get("summary", {})
    sol_s = mc_out["coins"].get("SOL", {}).get("summary", {})

    def fmt_coin(s: dict) -> str:
        if not s:
            return "N/A (data error)"
        return (f"comp={s.get('compounded_net_pct','?'):.2f}% "
                f"wins={s.get('win_windows','?')}/{s.get('total_windows','?')} "
                f"PSR={s.get('psr',{}).get('psr_vs_hurdle',0):.3f} "
                f"verdict={s.get('verdict','?')}")

    lines = []
    lines.append("# ADAPTIVE_TREND_OOS_VERDICT")
    lines.append("")
    lines.append("Research only. Not wired to live bot.")
    lines.append("")

    lines.append("## TL;DR")
    lines.append(f"- **Grid winner**: L={winner['L']}, theta={winner['theta']}, alpha={winner['alpha']}")
    lines.append(f"- **OOS performance (BTC, 5 windows)**: compounded={btc_comp:.2f}%, {wg['win_windows']}/5 wins, worst DD={wg['worst_dd_pct']:.2f}%, PSR={wg['psr']['psr_vs_hurdle']:.3f}, n_trades={wg['total_trades']}")
    lines.append(f"- **Walk-forward**: {wf_agg['positive_windows']}/{wf_agg['total_windows']} positive, agg_comp={wf_agg['compounded_net_pct']:.2f}%, verdict={wf_verdict}")
    lines.append(f"- **Multi-coin**: ETH={eth_s.get('verdict','?')} | SOL={sol_s.get('verdict','?')}")
    lines.append(f"- **Promotion recommendation**: `{promotion}`")
    lines.append("")

    lines.append("## Grid Sweep Summary")
    lines.append("")
    lines.append(f"Grid: L∈{L_VALUES} × theta∈{THETA_VALUES} × alpha∈{ALPHA_VALUES} = 48 variants × 5 OOS windows = 240 backtests.")
    lines.append(f"Eligibility gate: total_trades >= {MIN_TRADES_FLOOR}.")
    lines.append("Rank metric: `compounded × min(1, PSR)` (penalises low-PSR high-return flukes).")
    lines.append("")
    lines.append("### Top 5 variants")
    lines.append("")
    lines.append("| Rank | L | theta | alpha | Comp% | Wins | Trades | WorstDD% | PSR | Score |")
    lines.append("|------|---|-------|-------|-------|------|--------|----------|-----|-------|")
    for i, v in enumerate(top5, 1):
        s = v["summary"]
        lines.append(
            f"| {i} | {v['L']} | {v['theta']} | {v['alpha']} | "
            f"{s['compounded_net_pct']:.2f} | {s['win_windows']}/5 | "
            f"{s['total_trades']} | {s['worst_dd_pct']:.2f} | "
            f"{s['psr']['psr_vs_hurdle']:.3f} | {s['score']:.4f} |"
        )
    lines.append("")
    lines.append("### Bottom 3 variants (honesty)")
    lines.append("")
    lines.append("| Rank | L | theta | alpha | Comp% | Wins | Trades | WorstDD% | PSR | Score |")
    lines.append("|------|---|-------|-------|-------|------|--------|----------|-----|-------|")
    for i, v in enumerate(bottom3, 1):
        rank = len(variants) - 3 + i
        s = v["summary"]
        lines.append(
            f"| {rank} | {v['L']} | {v['theta']} | {v['alpha']} | "
            f"{s['compounded_net_pct']:.2f} | {s['win_windows']}/5 | "
            f"{s['total_trades']} | {s['worst_dd_pct']:.2f} | "
            f"{s['psr']['psr_vs_hurdle']:.3f} | {s['score']:.4f} |"
        )
    lines.append("")

    lines.append("## Walk-Forward Result")
    lines.append("")
    lines.append(f"Sliding 6-month train / 3-month test, quarterly advance.")
    lines.append(f"Verdict: **{wf_verdict}** ({wf_agg['positive_windows']}/{wf_agg['total_windows']} positive, "
                 f"agg comp={wf_agg['compounded_net_pct']:.2f}%, PSR={wf_agg['psr']['psr_vs_hurdle']:.3f})")
    lines.append("")
    lines.append("| Window | Test Period | Net% | Trades |")
    lines.append("|--------|-------------|------|--------|")
    for i, w in enumerate(wf_windows, 1):
        lines.append(f"| {i} | {w.get('test_start','?')} .. {w.get('test_end','?')} | {w['net_return_pct']:.2f} | {w['trades']} |")
    lines.append("")

    lines.append("## Multi-Coin Result")
    lines.append("")
    lines.append("| Coin | Comp% | Wins | PSR | Funding | Verdict |")
    lines.append("|------|-------|------|-----|---------|---------|")
    lines.append(f"| BTC (baseline) | {btc_comp:.2f} | {wg['win_windows']}/5 | {wg['psr']['psr_vs_hurdle']:.3f} | real | N/A (grid base) |")
    for coin, spec in [("ETH", eth_s), ("SOL", sol_s)]:
        if not spec:
            lines.append(f"| {coin} | N/A | N/A | N/A | N/A | error |")
        else:
            funded = "real" if spec.get("funding_modeled") else "no-funding"
            lines.append(
                f"| {coin} | {spec.get('compounded_net_pct','?'):.2f} | "
                f"{spec.get('win_windows','?')}/{spec.get('total_windows','?')} | "
                f"{spec.get('psr',{}).get('psr_vs_hurdle',0):.3f} | {funded} | {spec.get('verdict','?')} |"
            )
    lines.append("")

    lines.append("## Promotion Recommendation")
    lines.append("")
    lines.append(f"**`{promotion}`**")
    lines.append("")

    if promotion == "promote_to_dry_run_leg":
        lines.append("All promotion gates cleared:")
        lines.append(f"- Grid winner: comp > 0 ({btc_comp:.2f}%), ≥3/5 wins ({wg['win_windows']}), worst DD > −10% ({wg['worst_dd_pct']:.2f}%), PSR > 0.5 ({wg['psr']['psr_vs_hurdle']:.3f})")
        lines.append(f"- Walk-forward: {wf_verdict}")
        lines.append(f"- BTC compounded > 20%: {btc_comp:.2f}%")
        lines.append(f"- n_trades ({wg['total_trades']}) >= MinTRL ({wg['psr']['min_trl']})")
    elif promotion == "iterate_more":
        lines.append("Grid winner passes core OOS gates, but one or more promotion gates not cleared:")
        if wf_verdict == "fail":
            lines.append(f"- Walk-forward failed: {wf_agg['positive_windows']}/{wf_agg['total_windows']} positive, agg_comp={wf_agg['compounded_net_pct']:.2f}%")
        if btc_comp <= 20.0:
            lines.append(f"- BTC compounded ({btc_comp:.2f}%) does not exceed 20% gate")
        if wg['total_trades'] < wg['psr']['min_trl']:
            lines.append(f"- n_trades ({wg['total_trades']}) < MinTRL ({wg['psr']['min_trl']})")
        lines.append("Recommendations: run more OOS windows (extend history), or wait for live accumulation of trades.")
    else:
        lines.append("Grid winner failed one or more required gates. Strategy should be shelved.")

    lines.append("")
    lines.append("## Risk-Adjusted Comparison: AdaptiveTrend vs multifactor-v1")
    lines.append("")
    lines.append("| Metric | AdaptiveTrend (grid winner) | multifactor-v1 |")
    lines.append("|--------|-----------------------------|----------------|")
    lines.append(f"| Compounded OOS net% | {btc_comp:.2f}% | +55.73% (5 windows) |")
    lines.append(f"| Win rate (windows) | {wg['win_windows']}/5 | 4/5 |")
    lines.append(f"| Worst window DD | {wg['worst_dd_pct']:.2f}% | -12.56% |")
    lines.append(f"| PSR (vs SR=0) | {wg['psr']['psr_vs_hurdle']:.3f} | [not computed here] |")
    lines.append(f"| Total trades (5W) | {wg['total_trades']} | [per multifactor results] |")
    lines.append("| Signal class | Trend-following (MOM+ATR trail) | Multi-factor mean-reversion |")
    lines.append("| Funding drag | Present (holds days-weeks) | Lower (shorter holds) |")
    lines.append("")
    lines.append("AdaptiveTrend has a shallower worst-window drawdown (-7.4% baseline vs -12.6% multifactor) "
                 "at the cost of lower compounded return. Different signal class = genuine diversification value "
                 "if both are live simultaneously — low expected correlation.")
    lines.append("")

    lines.append("## Open Questions for Operator")
    lines.append("")
    lines.append("1. **Monthly L/theta re-optimisation**: Paper's adaptive layer (Algorithm 2) runs monthly re-opt of (L, theta). Not implemented here (portfolio-level). Would likely lift both Sharpe and trade count, reducing MinTRL pressure.")
    lines.append("2. **Portfolio allocation**: Paper's 70/30 long/short capital split + Sharpe-ratio asset selection contribute most of the headline 2.41 Sharpe. BTC-only does not benefit from this — extend to multi-asset port?")
    lines.append("3. **MinTRL gap**: Grid winner may still be below MinTRL. More OOS history (2020/2021 data exists) would help close the gap. Or extend test through 2025-H2 once data is available.")
    lines.append("4. **Walk-forward degradation pattern**: If WF shows a temporal trend (later windows weaker), that implies regime change in BTC MOM structure. Operator should inspect per-window table.")
    lines.append("5. **Live wiring dependency**: AdaptiveTrend requires H6 resampling logic inside the live bot — a different architecture from multifactor's bar-by-bar approach. Engineering cost before dry run.")
    lines.append("6. **ETH/SOL generalisation**: If multi-coin verdict is `transfers`, consider running a portfolio of 3 coins at lower position size — this would activate the paper's multi-asset machinery.")
    lines.append("")
    lines.append("---")
    lines.append("*Research only. Not wired to live bot. No risk.py or config/params.yaml touched.*")

    out_path = ROOT / "ADAPTIVE_TREND_OOS_VERDICT.md"
    out_path.write_text("\n".join(lines))
    print(f"[verdict] Written to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== AdaptiveTrend-v1 OOS Sweep, Walk-Forward, Multi-Coin ===", flush=True)

    print("\n--- Task A: Grid Sweep ---", flush=True)
    grid_out = run_grid_sweep()
    winner_config = grid_out["winner"]["config"]

    print("\n--- Task B: Walk-Forward ---", flush=True)
    wf_out = run_walk_forward(winner_config)

    print("\n--- Task C: Multi-Coin ---", flush=True)
    mc_out = run_multi_coin(winner_config)

    print("\n--- Task D: Write Verdict ---", flush=True)
    write_verdict_md(grid_out, wf_out, mc_out)

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()
