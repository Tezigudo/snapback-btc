"""Cost-stress a strategy's PSR by re-evaluating trade pnl_pct under elevated
transaction costs.

The current harness uses commission=0.0005 (5bps round-trip proxy) and no
slippage model. Realistic round-trip on BTC perp at $1M cash is closer to
10-15 bps including taker slippage. This tool answers: would the strategy's
PSR survive at the realistic cost level?

Usage:
    .venv/bin/python tools/cost_stress_psr.py \\
        --aggregate-csv reports/postfrac_mf_baseline_aggregated.csv \\
        --base-commission-bps 5 \\
        --stress-bps 15

    # Or stress multiple levels at once:
    .venv/bin/python tools/cost_stress_psr.py \\
        --aggregate-csv reports/_postfrac_mf_4h_btc_AGGREGATE.csv \\
        --stress-bps 5,10,15,20

Output: a table showing how PSR and compounded-equity-proxy change at each
cost level. NOT a replacement for running a re-backtest with a real
slippage model — this is a quick first-pass sensitivity check.

Mechanism: each trade's pnl_pct is degraded by (stress - base) bps. Because
this is a stat-level adjustment, it does NOT account for cost-aware position
sizing or signals that would have been filtered at higher cost. Read it as
an upper bound on survival.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.psr_eval import compute_psr  # noqa: E402


def _stress_pnl(pnl_pct: np.ndarray, base_bps: float, stress_bps: float) -> np.ndarray:
    """Degrade per-trade pnl_pct by (stress - base) bps in percent units.

    pnl_pct is in percent (e.g. 1.5 = +1.5%). bps stress translates to
    bps_diff / 100 (=% units). One degradation per trade entry+exit pair.
    """
    delta_pct = (stress_bps - base_bps) / 100.0
    return pnl_pct - delta_pct


def _compounded_proxy(pnl_pct: np.ndarray) -> float:
    """Compounded return assuming each trade returns its pnl_pct on its own
    capital. Real strategy compounding depends on sizing; this is a
    rough proxy. NaN if any pnl_pct <= -100%.
    """
    factors = 1 + pnl_pct / 100.0
    if (factors <= 0).any():
        return float("nan")
    return float((np.prod(factors) - 1) * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregate-csv", required=True, help="path to per-trade pnl_pct CSV")
    ap.add_argument("--base-commission-bps", type=float, default=5.0,
                    help="commission used in the original backtest (default 5)")
    ap.add_argument("--stress-bps", required=True,
                    help="single bps value (e.g. '15') or comma list (e.g. '5,10,15,20')")
    ap.add_argument("--label", default="strategy", help="label for output table")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of table")
    args = ap.parse_args()

    csv_path = Path(args.aggregate_csv)
    if not csv_path.exists():
        print(f"error: aggregate CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    col = "pnl_pct" if "pnl_pct" in df.columns else df.columns[0]
    pnl = df[col].dropna().values.astype(float)
    if len(pnl) < 2:
        print(f"error: need at least 2 trades, found {len(pnl)}", file=sys.stderr)
        return 1

    base_psr_result = compute_psr(pnl, sr_hurdle=0.0, confidence=0.95)

    stress_levels = [float(x.strip()) for x in str(args.stress_bps).split(",")]
    rows = []
    for stress in stress_levels:
        stressed = _stress_pnl(pnl, args.base_commission_bps, stress)
        psr_result = compute_psr(stressed, sr_hurdle=0.0, confidence=0.95) \
            if len(stressed) >= 2 else {"psr_vs_hurdle": 0.0, "interpretation": "insufficient"}
        rows.append({
            "stress_bps":       stress,
            "compounded_pct":   round(_compounded_proxy(stressed), 4),
            "psr_vs_hurdle":    round(float(psr_result.get("psr_vs_hurdle", 0.0)), 4),
            "interpretation":   psr_result.get("interpretation", "?"),
            "mean_pnl_pct":     round(float(stressed.mean()), 4),
            "sharpe_proxy":     round(float(stressed.mean() / (stressed.std() + 1e-9)), 4),
        })

    if args.json:
        out = {
            "label": args.label,
            "n_trades": int(len(pnl)),
            "base_commission_bps": args.base_commission_bps,
            "base_compounded_pct": round(_compounded_proxy(pnl), 4),
            "base_psr_vs_hurdle": round(float(base_psr_result.get("psr_vs_hurdle", 0.0)), 4),
            "stress_rows": rows,
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"\n{'='*72}")
    print(f"Cost-stress: {args.label}  (n_trades={len(pnl)}, base={args.base_commission_bps}bps)")
    print(f"{'='*72}")
    print(f"  {'bps':>5}  {'compounded':>13}  {'PSR':>8}  {'interpretation':>22}  "
          f"{'mean':>8}  {'sharpe':>8}")
    for r in rows:
        print(f"  {r['stress_bps']:>5.1f}  {r['compounded_pct']:>11.2f}%  "
              f"{r['psr_vs_hurdle']:>8.4f}  {r['interpretation']:>22}  "
              f"{r['mean_pnl_pct']:>7.3f}%  {r['sharpe_proxy']:>8.4f}")
    print(f"{'='*72}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
