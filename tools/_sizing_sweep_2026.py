"""Sizing sweep for the DEPLOYED multifactor-v1 config — risk 2.75->4.0%.

Question (God, 2026-07-12): how much more profit does sizing up the validated
edge buy, and does it stay inside the safety rails?

Arms: risk_per_trade_pct in {2.75 (deployed anchor), 3.0, 3.5, 4.0}
Config: locked params + 4H gate (MultiFactorMTF4H) + funding threshold 0.0015
(as deployed 2026-07-12). BTC funding attached. $1M cash, 5bps, margin 1/20.

Surfaces per arm:
  - 5-OOS windows: compounded, wins, worst-window peak max-DD
  - quarterly walk-forward 2020Q1..2026Q1: % positive, worst-quarter return
    and worst-quarter peak max-DD (the DD tail that matters for the kill switch)
Safety framing: kill switch fires at -35.5% from the principal anchor;
backtesting.py max-DD is PEAK-anchored >= start-anchored, so
"worst peak-DD < 35.5%" is a conservative sufficient condition.
Daily-loss breaker: one full SL day = -risk%, so breaker must exceed risk
(deployed breaker = 3.5% -> risk 3.5/4.0 would need breaker ~4.5/5.0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_mf_deepening import (  # noqa: E402
    PARQ,
    WINDOWS,
    MultiFactorMTF4H,
    _load_4h_ema200_aligned,
    _load_slice,
    run_one,
)

RISKS = (2.75, 3.0, 3.5, 4.0)
BASE_OV = {"funding_extreme_threshold": 0.0015}   # deployed 2026-07-12
BTC_4H = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
SUFFICIENT_TRADES = 3


def prep(start: str, end: str):
    df = _load_slice(PARQ["BTC"], start, end, attach_funding=True)
    df["Ema4h"] = _load_4h_ema200_aligned(df.index, BTC_4H)
    return df


def quarters() -> list[tuple[str, str, str]]:
    out = []
    for y in range(2020, 2027):
        for q, (s, e) in enumerate(
            [("01-01", "03-31"), ("04-01", "06-30"),
             ("07-01", "09-30"), ("10-01", "12-31")], start=1):
            if (y, q) > (2026, 1):
                break
            out.append((f"{y}Q{q}", f"{y}-{s}", f"{y}-{e}"))
    return out


def main() -> int:
    print("[sizing] loading slices ...", file=sys.stderr)
    oos = {label: prep(s, e) for label, s, e in WINDOWS}
    wf = {label: prep(s, e) for label, s, e in quarters()}

    out = {"config": "locked + 4H gate + funding 0.0015", "arms": {}}
    for r in RISKS:
        ov = {**BASE_OV, "risk_per_trade_pct": r}
        # 5-OOS
        comp = 1.0
        wins = 0
        worst_dd = 0.0
        per_w = []
        for label, _s, _e in WINDOWS:
            res = run_one(oos[label], ov, strategy_class=MultiFactorMTF4H)
            comp *= 1 + res["return_pct"] / 100
            wins += res["return_pct"] > 0
            worst_dd = min(worst_dd, res["max_dd_pct"])
            per_w.append(round(res["return_pct"], 2))
        # quarterly WF
        q_res = []
        for label, _s, _e in quarters():
            res = run_one(wf[label], ov, strategy_class=MultiFactorMTF4H)
            q_res.append({"q": label, "trades": res["trades"],
                          "ret": round(res["return_pct"], 2),
                          "dd": round(res["max_dd_pct"], 2)})
        suff = [q for q in q_res if q["trades"] >= SUFFICIENT_TRADES]
        pos = [q for q in suff if q["ret"] > 0]
        worst_q = min(q_res, key=lambda q: q["ret"])
        worst_q_dd = min(q_res, key=lambda q: q["dd"])
        arm = {
            "oos_compounded_pct": round((comp - 1) * 100, 2),
            "oos_wins": f"{wins}/5",
            "oos_per_window": per_w,
            "oos_worst_peak_dd_pct": round(worst_dd, 2),
            "wf_pct_positive_sufficient": round(100 * len(pos) / len(suff), 1) if suff else None,
            "wf_worst_quarter": worst_q,
            "wf_worst_quarter_dd": worst_q_dd,
            "kill_margin_pp": round(35.5 + worst_q_dd["dd"], 2),
            "breaker_needed_pct": round(r + 1.0, 2),
        }
        out["arms"][f"risk_{r}"] = arm
        print(f"[sizing] risk={r}: oos {arm['oos_compounded_pct']}% "
              f"({arm['oos_wins']}), worst q-DD {worst_q_dd['dd']}%, "
              f"kill margin {arm['kill_margin_pp']}pp", file=sys.stderr)

    path = ROOT / "reports" / "sizing_sweep_2026.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[sizing] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
