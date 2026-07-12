"""SOL 4H-gated multifactor-v1 SOLO leg — deep validation (promote decider).

Prior evidence (reports/multifactor_v1_4h_multi_coin.json): SOL at the LOCKED
BTC geometry + 4H gate = +41.93%, 5/5 OOS windows, 92 trades, canonical
psr_walkforward 0.9992 (evidence_of_edge). What was never measured for the
SOLO leg (the BTC+SOL PORTFOLIO todo shelved on portfolio-arm artifacts):

  1. Quarterly walk-forward 2021Q1..2026Q1 — >=70% positive among
     sufficient-trade (>=3) quarters, the same gate every promote decision uses.
  2. Cost stress — the 5 OOS windows at 5 / 10 / 15 bps per side.

Protocol matches the prior multi-coin run: locked config + use_mtf_4h_gate,
gate pointed at SOL's OWN 4h parquet (per the mtf_4h_parquet_path gotcha),
funding column not attached (gate no-ops, same as prior run), $1M cash,
margin 1/20.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_mf_deepening import (  # noqa: E402
    LOCKED,
    PARQ,
    WINDOWS,
    _load_slice,
    run_one,
    summarize,
)

SOL_4H = str(ROOT / "data" / "historical" / "SOL_USDT_USDT_4h.parquet")
OV = {**LOCKED, "mtf_4h_parquet_path": SOL_4H}
SUFFICIENT_TRADES = 3


def quarters() -> list[tuple[str, str, str]]:
    out = []
    for y in range(2021, 2027):
        for q, (s, e) in enumerate(
            [("01-01", "03-31"), ("04-01", "06-30"),
             ("07-01", "09-30"), ("10-01", "12-31")], start=1):
            if (y, q) > (2026, 1):
                break
            out.append((f"{y}Q{q}", f"{y}-{s}", f"{y}-{e}"))
    return out


def main() -> int:
    out: dict = {"protocol": "locked cfg + 4H gate (SOL parquet), no funding col",
                 "commission_note": "per-side"}

    # --- 1. quarterly walk-forward at 5 bps ---
    wf = []
    for label, s, e in quarters():
        df = _load_slice(PARQ["SOL"], s, e, attach_funding=False)
        r = run_one(df, OV)
        wf.append({"q": label, "trades": r["trades"],
                   "return_pct": round(r["return_pct"], 2),
                   "max_dd_pct": round(r["max_dd_pct"], 2)})
        print(f"  WF {label}: {r['trades']}tr {r['return_pct']:.1f}%",
              file=sys.stderr)
    suff = [q for q in wf if q["trades"] >= SUFFICIENT_TRADES]
    pos = [q for q in suff if q["return_pct"] > 0]
    out["walk_forward"] = {
        "quarters": wf,
        "n_quarters": len(wf),
        "n_sufficient": len(suff),
        "n_positive_sufficient": len(pos),
        "pct_positive_sufficient": round(100 * len(pos) / len(suff), 1) if suff else None,
        "gate_70pct": bool(suff and len(pos) / len(suff) >= 0.70),
        "worst_quarter": min(wf, key=lambda q: q["return_pct"]),
    }

    # --- 2. cost stress on the standard 5 OOS windows ---
    import tools.run_mf_deepening as mfd
    out["cost_stress"] = {}
    slices = {label: _load_slice(PARQ["SOL"], s, e, attach_funding=False)
              for label, s, e in WINDOWS}
    for bps in (5, 10, 15):
        mfd.COMMISSION = bps / 1e4
        per_window = {}
        for label, _s, _e in WINDOWS:
            per_window[label] = run_one(slices[label], OV)
        summ = summarize(f"sol_{bps}bps", per_window)
        s = summ["summary"]
        psr = (summ["canonical"].get("psr_walkforward") or {})
        out["cost_stress"][f"{bps}bps"] = {
            "compounded_pct": s["compounded_pct"],
            "windows_positive": s["windows_positive"],
            "n_trades": s["n_trades"],
            "per_window": s["per_window_return_pct"],
            "psr_wf": round(psr.get("psr_vs_hurdle", 0), 4),
            "interpretation": psr.get("interpretation"),
        }
        print(f"  stress {bps}bps: comp {s['compounded_pct']}% wins "
              f"{s['windows_positive']} psr {psr.get('psr_vs_hurdle')}",
              file=sys.stderr)

    wfres = out["walk_forward"]
    c15 = out["cost_stress"]["15bps"]
    out["verdict"] = (
        "PROMOTE_CANDIDATE"
        if wfres["gate_70pct"] and c15["psr_wf"] >= 0.80 and c15["compounded_pct"] > 0
        else "SHELF"
    )

    path = ROOT / "reports" / "sol_leg_deep_validation.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "walk_forward"}
                     | {"walk_forward_summary": {k: v for k, v in wfres.items() if k != "quarters"}},
                     indent=2, default=str))
    print(f"[sol] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
