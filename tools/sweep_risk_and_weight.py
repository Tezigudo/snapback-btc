"""Sweep risk_per_trade_pct and v1/Donchian weight to find configs
hitting $4/mo at $101 budget with −35.5% kill-switch.

Runs realistic_50_50_sim.py with various overrides, collects results,
prints a ranked table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(kill_frac: float, risk_pct: float, weight_v1: float) -> dict:
    cmd = [
        "uv", "run", "python", str(ROOT / "tools" / "realistic_50_50_sim.py"),
        f"--kill-frac={kill_frac}",
        f"--risk-pct={risk_pct}",
        f"--weight-v1={weight_v1}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    if r.returncode != 0:
        return {"error": r.stderr[-500:] if r.stderr else "no stderr"}
    # Parse the output JSON path from the trailing "Wrote ..." line
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith("Wrote "):
            path = line.split(" ", 1)[1].strip()
            data = json.loads(Path(path).read_text())
            data["_path"] = path
            return data
    return {"error": "could not find Wrote line"}


def monthly_dollars(start: float, final: float, months: float) -> float:
    return (final - start) / months


def main() -> int:
    months_in_run = 6.67 * 12     # ≈ 80 months
    print(f"Months in continuous run: {months_in_run:.1f}")
    print(f"\nTarget: $4/mo avg on $101 → end equity ≥ ${101 + 4 * months_in_run:.0f}")

    configs = []
    # Sweep: kill at −35.5% (user choice), risk_pct {2.0, 2.5, 3.0, 3.5, 4.0, 5.0}%, weights {50/50, 30/70, 0/100}
    for risk_pct in (0.020, 0.025, 0.030, 0.035, 0.040, 0.050):
        for weight_v1, weight_label in ((0.5, "50/50"), (0.3, "30/70"), (0.0, "all-Donchian")):
            configs.append({"kill_frac": 0.645, "risk_pct": risk_pct, "weight_v1": weight_v1,
                            "label": weight_label})

    results = []
    for cfg in configs:
        print(f"\nRunning: kill=−{(1-cfg['kill_frac'])*100:.1f}% risk={cfg['risk_pct']*100:.1f}% weights={cfg['label']}")
        r = run_one(cfg["kill_frac"], cfg["risk_pct"], cfg["weight_v1"])
        if "error" in r:
            print(f"  ERROR: {r['error'][:200]}")
            continue
        co = r["combined"]
        mo_dollars = monthly_dollars(co["start"], co["final"], months_in_run)
        # Kill-switch trip info per leg
        v1_tripped = r["v1"].get("kill_switch_tripped", False)
        d3_tripped = r["donchian_cons"].get("kill_switch_tripped", False)
        results.append({
            **cfg,
            "final_combined": co["final"],
            "ret_pct": co["ret_pct"],
            "sharpe": co["sharpe"],
            "max_dd_pct": co["max_dd_pct"],
            "monthly_dollars": mo_dollars,
            "hits_4_per_mo": mo_dollars >= 4.0,
            "v1_kill_tripped": v1_tripped,
            "d3_kill_tripped": d3_tripped,
            "any_kill_tripped": v1_tripped or d3_tripped,
        })

    # Sort by monthly $/avg descending
    results.sort(key=lambda r: r["monthly_dollars"], reverse=True)

    print("\n\n=== SWEEP RESULTS (sorted by monthly $/avg) ===")
    print(f"{'risk':>6} {'weights':>13} {'kill':>7} "
          f"{'final $':>10} {'ret %':>9} {'Sharpe':>7} "
          f"{'peakDD':>8} {'$/mo':>8} {'≥$4':>5} {'kill?':>6}")
    for r in results:
        kill_str = f"-{(1-r['kill_frac'])*100:.1f}%"
        marker = "✓" if r["hits_4_per_mo"] else " "
        kill_mark = "TRIP" if r["any_kill_tripped"] else "ok"
        print(
            f"{r['risk_pct']*100:>5.1f}% "
            f"{r['label']:>13} "
            f"{kill_str:>7} "
            f"{r['final_combined']:>+9.2f} "
            f"{r['ret_pct']:>+8.2f}% "
            f"{r['sharpe']:>+6.2f} "
            f"{r['max_dd_pct']:>+7.2f}% "
            f"{r['monthly_dollars']:>+7.3f} "
            f"{marker:>5} "
            f"{kill_mark:>6}"
        )

    # Find cheapest config that hits $4/mo with no kill trips
    hits = [r for r in results if r["hits_4_per_mo"] and not r["any_kill_tripped"]]
    if hits:
        # Cheapest = lowest risk_pct (least aggressive)
        hits.sort(key=lambda r: (r["risk_pct"], -r["sharpe"]))
        winner = hits[0]
        print("\n\n=== CHEAPEST CONFIG THAT HITS $4/mo WITHOUT TRIPPING KILL ===")
        print(f"  risk_per_trade_pct = {winner['risk_pct']*100:.1f}%")
        print(f"  weights = {winner['label']}")
        print(f"  kill_switch = {(1-winner['kill_frac'])*100:.1f}%")
        print(f"  expected: ${winner['final_combined']:.2f} final, {winner['ret_pct']:+.1f}% return,")
        print(f"            Sharpe {winner['sharpe']:+.2f}, peak-DD {winner['max_dd_pct']:+.1f}%,")
        print(f"            monthly avg ${winner['monthly_dollars']:.2f}")
    else:
        print("\n\nNO config hits $4/mo without tripping kill switch")

    # Save
    out_path = ROOT / "reports" / "sweep_risk_weight.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
