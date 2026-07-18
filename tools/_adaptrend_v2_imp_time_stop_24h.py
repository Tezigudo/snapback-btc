"""
AdaptiveTrendV2 — improvement experiment: time_stop_24h.

Hypothesis: closing a position after 96 H6 bars (= 24 days, vs base default
of 120 H6 bars = 30 days) reduces loser-hold time and lifts expectancy.

Implementation: pure config toggle on AdaptiveTrendV2 — the base strategy
already exposes `max_hold_h6_bars` (used by check `(i - entry_bar) >=
max_hold_h6_bars * 24` where i is the 15m bar index — 24 fifteen-min bars
per H6 bar). Setting it to 96 tightens the existing time stop from 30 days
to 24 days.

No new strategy class, no edits to v1 or v2 base files.

Run from repo root:
    .venv/bin/python tools/_adaptrend_v2_imp_time_stop_24h.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools._adaptrend_v2_run import run as _run_v2  # noqa: E402
from tools.aggregate import (  # noqa: E402
    build_canonical_block,
    legacy_stitched_psr,
)

# 11 OOS windows (matches ADAPTIVE_TREND_EXTENDED_VERDICT.md).
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
CASH = 1_000_000


def _compounded(nets_pct: list[float]) -> float:
    x = 1.0
    for r in nets_pct:
        x *= 1.0 + r / 100.0
    return (x - 1.0) * 100.0


def _eq_impact_from_window_csv(g: pd.DataFrame) -> list[float]:
    """Per-window equity-impact returns (PnL / equity-at-entry * 100).

    Mirrors tools.aggregate.equity_impact_returns but operates on the
    persisted per-window trade rows (the stats object is not retained by the
    _run_v2 wrapper). Trades are sorted by ExitTime, equity is CASH compounded
    by all prior PnL in the window — sizing-aware, single contiguous window.
    """
    if "PnL" not in g.columns or len(g) == 0:
        return []
    gg = g.copy()
    if "ExitTime" in gg.columns:
        gg = gg.sort_values("ExitTime")
    pnls = gg["PnL"].astype(float).values
    cum_pnl_prev = np.concatenate(([0.0], np.cumsum(pnls)[:-1]))
    equity_at_entry = float(CASH) + cum_pnl_prev
    equity_at_entry = np.where(equity_at_entry == 0.0, float(CASH), equity_at_entry)
    return ((pnls / equity_at_entry) * 100.0).tolist()


def run_arm(label: str, config: dict, trades_csv: Path) -> dict:
    """Run all 11 windows for one configuration; return aggregated stats."""
    if trades_csv.exists():
        trades_csv.unlink()

    rows = []
    nets = []
    total_trades = 0
    wins = 0

    for start, end, wlabel in OOS_WINDOWS:
        res = _run_v2(
            start=start, end=end, cash=CASH, label=wlabel,
            config=config, save_trades=trades_csv,
        )
        net = float(res["net_return_pct"])
        nets.append(net)
        total_trades += int(res["trades"])
        if net > 0:
            wins += 1
        rows.append({
            "label": wlabel,
            "trades": int(res["trades"]),
            "net_return_pct": net,
            "max_dd_pct": float(res["max_dd_pct"]),
        })
        print(
            f"  [{label}] {wlabel}: trades={res['trades']:3d} "
            f"net={net:+6.2f}% dd={res['max_dd_pct']:+6.2f}%",
            flush=True,
        )

    comp = _compounded(nets)

    # --- Per-window per-trade pnl_pct + equity-impact, grouped from the CSV. ---
    # The CSV is the per-window union of all 11 windows' trades, carrying a
    # `label` column (from _run_v2's save_trades). Group on it so the canonical
    # block gets one entry per window with sizing-aware equity-impact returns.
    per_window_canon: list[dict] = []
    if trades_csv.exists():
        df = pd.read_csv(trades_csv)
    else:
        df = pd.DataFrame()
    by_label = (
        {str(k): v for k, v in df.groupby("label")}
        if (not df.empty and "label" in df.columns)
        else {}
    )
    for r in rows:
        g = by_label.get(r["label"], pd.DataFrame())
        if not g.empty and "pnl_pct" in g.columns:
            pnl_list = g["pnl_pct"].dropna().astype(float).tolist()
        else:
            pnl_list = []
        per_window_canon.append({
            "label":             r["label"],
            # v2 headline = funding-adjusted net (sizing-aware: PnL net of
            # funding over starting cash). Tag the family accordingly below.
            "return_pct":        r["net_return_pct"],
            "trades":            r["trades"],
            # gross per-trade ReturnPct% — drives the synthetic stitched ref.
            "pnl_pct":           pnl_list,
            # sizing-aware equity-impact series — feeds psr_per_window.
            "eq_impact_pnl_pct": _eq_impact_from_window_csv(g),
        })

    # CANONICAL (v2) dual-emit. Funding-net headline -> funding_adjusted tag.
    canon = build_canonical_block(
        per_window_canon,
        aggregation_method="v2_equity_curve_funding_adjusted",
    )
    # Reported headline PSR = canonical window-level psr_walkforward.
    psr = canon["psr_walkforward"]
    # LEGACY stitched-per-trade PSR — observability sidecar only (N-inflated).
    legacy_psr = legacy_stitched_psr(per_window_canon)

    return {
        "label": label,
        "windows": rows,
        "compounded_pct": comp,
        "wins": wins,
        "n_windows": len(OOS_WINDOWS),
        "total_trades": total_trades,
        "psr": psr,                       # CANONICAL headline (psr_walkforward)
        "legacy_psr_stitched": legacy_psr,  # observability only
        "canonical": canon,
        "aggregation_method": canon["aggregation_method"],
    }


def main() -> None:
    out_csv_base = ROOT / "reports" / "_adaptrend_v2_imp_time_stop_24h_base_trades.csv"
    out_csv_imp = ROOT / "reports" / "_adaptrend_v2_imp_time_stop_24h_treat_trades.csv"

    print("=== AdaptiveTrendV2 baseline (max_hold_h6_bars=120 = 30 days) ===", flush=True)
    base = run_arm("base", config={}, trades_csv=out_csv_base)

    print("\n=== AdaptiveTrendV2 + time_stop_24h (max_hold_h6_bars=96) ===", flush=True)
    treat = run_arm(
        "time_stop_24h",
        config={"max_hold_h6_bars": 96},
        trades_csv=out_csv_imp,
    )

    # Engagement diagnostic: count trades whose hold-time (15m bars from
    # EntryTime to ExitTime) is between 24 days and 30 days. These are the
    # only trades the lever could possibly touch.
    engagement = {}
    if out_csv_base.exists():
        df = pd.read_csv(out_csv_base)
        if "EntryTime" in df.columns and "ExitTime" in df.columns:
            df["EntryTime"] = pd.to_datetime(df["EntryTime"], errors="coerce")
            df["ExitTime"] = pd.to_datetime(df["ExitTime"], errors="coerce")
            df["hold_days"] = (df["ExitTime"] - df["EntryTime"]).dt.total_seconds() / 86400.0
            engagement["base_trades_24_to_30_days"] = int(((df["hold_days"] >= 24) & (df["hold_days"] <= 30)).sum())
            engagement["base_trades_at_30d_cap"] = int((df["hold_days"] >= 29.9).sum())
            engagement["base_n_trades"] = int(len(df))
    if out_csv_imp.exists():
        df2 = pd.read_csv(out_csv_imp)
        if "EntryTime" in df2.columns and "ExitTime" in df2.columns:
            df2["EntryTime"] = pd.to_datetime(df2["EntryTime"], errors="coerce")
            df2["ExitTime"] = pd.to_datetime(df2["ExitTime"], errors="coerce")
            df2["hold_days"] = (df2["ExitTime"] - df2["EntryTime"]).dt.total_seconds() / 86400.0
            engagement["treat_trades_at_24d_cap"] = int((df2["hold_days"] >= 23.9).sum())
            engagement["treat_n_trades"] = int(len(df2))

    # Verdict — Δcompounded is funding-net and UNCHANGED by the PSR migration.
    # psr_improved now reads the CANONICAL window-level psr_walkforward (was the
    # stitched per-trade PSR). HURTING/NEUTRAL depend only on delta_comp; only
    # the ACCRETIVE branch reads PSR, so a non-accretive arm is verdict-stable.
    delta_comp = treat["compounded_pct"] - base["compounded_pct"]
    psr_improved = treat["psr"]["psr_vs_hurdle"] > base["psr"]["psr_vs_hurdle"]
    if delta_comp > 5.0 and psr_improved:
        verdict = "ACCRETIVE"
    elif delta_comp < -5.0:
        verdict = "HURTING"
    else:
        verdict = "NEUTRAL"

    print("\n=== SUMMARY ===", flush=True)
    print(f"Base   comp={base['compounded_pct']:+.2f}%  wins={base['wins']}/11  trades={base['total_trades']}  "
          f"Sharpe={base['psr'].get('point_sharpe', 0.0):.4f}  PSR={base['psr']['psr_vs_hurdle']:.3f}  MinTRL={base['psr'].get('min_trl')}  "
          f"[canonical psr_walkforward]",
          flush=True)
    print(f"Treat  comp={treat['compounded_pct']:+.2f}%  wins={treat['wins']}/11  trades={treat['total_trades']}  "
          f"Sharpe={treat['psr'].get('point_sharpe', 0.0):.4f}  PSR={treat['psr']['psr_vs_hurdle']:.3f}  MinTRL={treat['psr'].get('min_trl')}  "
          f"[canonical psr_walkforward]",
          flush=True)
    print(f"Delta  Δcomp={delta_comp:+.2f}pp  PSR_improved={psr_improved}  -> {verdict}", flush=True)
    print(f"  legacy_stitched PSR  base={base['legacy_psr_stitched']['psr_vs_hurdle']:.3f} "
          f"treat={treat['legacy_psr_stitched']['psr_vs_hurdle']:.3f} (observability only, N-inflated)", flush=True)
    print(f"Engagement: {engagement}", flush=True)

    result = {
        "improvement_id": "time_stop_24h",
        "description": "Hard close after 96 H6 bars (= 24 days), tightens base's 120-bar (30d) cap.",
        "implementation": "config toggle: max_hold_h6_bars=96 on AdaptiveTrendV2 (no new class).",
        "n_windows": 11,
        "cash": CASH,
        "windows": [w[2] for w in OOS_WINDOWS],
        "aggregation_method": base["aggregation_method"],
        "base": {
            "compounded_pct": round(base["compounded_pct"], 4),
            "wins": base["wins"],
            "total_trades": base["total_trades"],
            "psr": base["psr"],                              # canonical psr_walkforward
            "legacy_psr_stitched": base["legacy_psr_stitched"],
            "canonical": base["canonical"],
            "windows": base["windows"],
        },
        "treatment": {
            "compounded_pct": round(treat["compounded_pct"], 4),
            "wins": treat["wins"],
            "total_trades": treat["total_trades"],
            "psr": treat["psr"],                            # canonical psr_walkforward
            "legacy_psr_stitched": treat["legacy_psr_stitched"],
            "canonical": treat["canonical"],
            "windows": treat["windows"],
        },
        "delta": {
            "compounded_pp": round(delta_comp, 4),
            "psr_improved": psr_improved,
            "verdict": verdict,
        },
        "engagement": engagement,
    }
    out_path = ROOT / "reports" / "adaptrend_v2_imp_time_stop_24h.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
