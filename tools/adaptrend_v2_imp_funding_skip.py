"""
Run AdaptiveTrendV2 base vs AdaptiveTrendV2+funding_skip on 11 OOS windows
(matches ADAPTIVE_TREND_EXTENDED_VERDICT.md window set).  Aggregates per-trade
returns, runs psr_eval.compute_psr, writes JSON.

Output: reports/adaptrend_v2_imp_funding_skip.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools._adaptrend_v2_run import run as run_v2_base  # noqa: E402
from tools._adaptrend_v2_funding_skip_run import run as run_v2_fs  # noqa: E402
from tools.aggregate import build_canonical_block, equity_impact_returns  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

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


def compounded(net_pcts):
    x = 1.0
    for r in net_pcts:
        x *= 1.0 + r / 100.0
    return (x - 1.0) * 100.0


def main():
    base_csv = ROOT / "reports" / "_adaptrend_v2_base_imp_trades.csv"
    fs_csv = ROOT / "reports" / "_adaptrend_v2_fs_imp_trades.csv"
    for p in (base_csv, fs_csv):
        if p.exists():
            p.unlink()

    rows = []
    for start, end, label in OOS_WINDOWS:
        print(f"\n=== {label} {start}..{end} ===", flush=True)

        base = run_v2_base(
            start=start, end=end, cash=CASH, label=label,
            save_trades=base_csv,
        )
        fs = run_v2_fs(
            start=start, end=end, cash=CASH, label=label,
            save_trades=fs_csv,
        )

        print(
            f"  base: trades={base['trades']:4d} net={base['net_return_pct']:+7.2f}% "
            f"dd={base['max_dd_pct']:+7.2f}%",
            flush=True,
        )
        print(
            f"  +fs : trades={fs['trades']:4d} net={fs['net_return_pct']:+7.2f}% "
            f"dd={fs['max_dd_pct']:+7.2f}% skipped={fs.get('n_skipped_by_funding', 0)}",
            flush=True,
        )
        rows.append({"label": label, "base": base, "fs": fs})

    # Aggregate
    base_nets = [r["base"]["net_return_pct"] for r in rows]
    fs_nets = [r["fs"]["net_return_pct"] for r in rows]
    base_trades_total = sum(r["base"]["trades"] for r in rows)
    fs_trades_total = sum(r["fs"]["trades"] for r in rows)
    base_wins = sum(1 for n in base_nets if n > 0)
    fs_wins = sum(1 for n in fs_nets if n > 0)
    base_comp = compounded(base_nets)
    fs_comp = compounded(fs_nets)

    # PSR
    def load_psr(csv):
        if not csv.exists():
            return compute_psr(np.array([], dtype=float))
        df = pd.read_csv(csv)
        pnl = df.get("pnl_pct", pd.Series(dtype=float)).dropna().values.astype(float)
        return compute_psr(pnl, sr_hurdle=0.0, confidence=0.95)

    base_psr = load_psr(base_csv)
    fs_psr = load_psr(fs_csv)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'Window':<10} {'base':>10} {'+fs':>10}", flush=True)
    for r in rows:
        print(
            f"{r['label']:<10} {r['base']['net_return_pct']:>+9.2f}% "
            f"{r['fs']['net_return_pct']:>+9.2f}%",
            flush=True,
        )
    print(f"{'COMP':<10} {base_comp:>+9.2f}% {fs_comp:>+9.2f}%", flush=True)
    print(f"{'WINS':<10} {base_wins:>9d}  {fs_wins:>9d}", flush=True)
    print(f"{'TRADES':<10} {base_trades_total:>9d}  {fs_trades_total:>9d}", flush=True)
    print(f"base PSR: {base_psr['psr_vs_hurdle']:.3f} "
          f"(sharpe={base_psr['point_sharpe']:.4f} n={base_psr['n_trades']} MinTRL={base_psr['min_trl']})",
          flush=True)
    print(f"+fs  PSR: {fs_psr['psr_vs_hurdle']:.3f} "
          f"(sharpe={fs_psr['point_sharpe']:.4f} n={fs_psr['n_trades']} MinTRL={fs_psr['min_trl']})",
          flush=True)

    # --- Canonical equity-curve dual-emit (methodology debt #1) ---
    # Headline canonical PSR = psr_walkforward (compute_psr on the n-window
    # net-return series), NOT the stitched per-trade union above. The legacy
    # stitched base_psr/fs_psr are kept verbatim for observability/diff.
    #
    # per-window eq_impact reconstructed from the saved trades CSV (we must NOT
    # modify the shared helper to surface `stats`); the CSV holds OOS-only
    # trades with PnL + ExitTime, exactly what equity_impact_returns needs via
    # the {"_trades": df} mapping shim (reuses canonical code, zero drift).
    def _build_canon(arm_key: str, csv_path: Path) -> dict:
        # Group saved trades by window label for per-window equity-impact +
        # per-trade ReturnPct% (the helper returns dicts WITHOUT pnl_pct, so we
        # read both series from the same CSV).
        eq_by_label: dict[str, list[float]] = {}
        pnl_by_label: dict[str, list[float]] = {}
        if csv_path.exists():
            tdf = pd.read_csv(csv_path)
            if "label" in tdf.columns:
                for win_label, wdf in tdf.groupby("label"):
                    eq_by_label[str(win_label)] = equity_impact_returns(
                        {"_trades": wdf}, cash=CASH
                    ).tolist()
                    if "pnl_pct" in wdf.columns:
                        pnl_by_label[str(win_label)] = (
                            wdf["pnl_pct"].dropna().astype(float).tolist()
                        )
        per_window_canon = []
        for r in rows:
            arm = r[arm_key]
            win_label = r["label"]
            pnl_pct = pnl_by_label.get(win_label, [])
            per_window_canon.append(
                {
                    "label": win_label,
                    # canonical v2 headline = funding-adjusted net (sizing-aware).
                    "return_pct": arm["net_return_pct"],
                    "trades": arm["trades"],
                    # gross per-trade ReturnPct% -> synthetic stitched legacy ref.
                    "pnl_pct": pnl_pct,
                    # sizing-aware equity-impact -> feeds psr_per_window.
                    "eq_impact_pnl_pct": eq_by_label.get(win_label, []),
                }
            )
        return build_canonical_block(
            per_window_canon,
            aggregation_method="v2_equity_curve_funding_adjusted",
        )

    base_canon = _build_canon("base", base_csv)
    fs_canon = _build_canon("fs", fs_csv)

    # --- Self-check (bit-for-bit): recompute headline PSR from the PERSISTED
    # canonical per-window return series and assert it equals psr_walkforward. ---
    canonical_selfcheck = {}
    for arm_label, canon in (("base", base_canon), ("fs", fs_canon)):
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
        f"fs psr_wf={fs_canon['psr_walkforward']['psr_vs_hurdle']:.4f} "
        f"(legacy stitched base={base_psr['psr_vs_hurdle']:.3f} "
        f"fs={fs_psr['psr_vs_hurdle']:.3f})",
        flush=True,
    )

    out = {
        "improvement_id": "funding_skip",
        "oos_windows": [w[2] for w in OOS_WINDOWS],
        "cash_per_window": CASH,
        "rows": rows,
        "summary": {
            "base_compounded_pct": round(base_comp, 4),
            "fs_compounded_pct": round(fs_comp, 4),
            "delta_compounded_pp": round(fs_comp - base_comp, 4),
            "base_wins": base_wins,
            "fs_wins": fs_wins,
            "base_total_trades": base_trades_total,
            "fs_total_trades": fs_trades_total,
        },
        # legacy stitched PSR (observability/diff only — N-inflated).
        "base_psr": base_psr,
        "fs_psr": fs_psr,
        "legacy_psr_stitched": {"base": base_psr, "fs": fs_psr},
        # canonical v2 equity-curve aggregation (primary metric).
        "aggregation_method": "v2_equity_curve_funding_adjusted",
        "funding_adjusted": True,
        "canonical": {"base": base_canon, "fs": fs_canon},
        "canonical_psr_selfcheck": canonical_selfcheck,
    }
    out_path = ROOT / "reports" / "adaptrend_v2_imp_funding_skip.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
