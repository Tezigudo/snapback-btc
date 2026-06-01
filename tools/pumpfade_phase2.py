"""Phase 2: do the two defensible structural fixes (cooldown dedup + stop-distance
cap) rescue the pump-fade edge — and does any improvement survive OUT OF SAMPLE?

IS  = entries before 2025-01-01   (2020-2024)
OOS = entries 2025-01-01 onward    (2025-2026)

If a variant is only positive IS, it's overfit. The bar is positive (and
friction-survivable) on BOTH, plus consistent across cohorts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_backtest as pf  # noqa: E402

OOS_START = pd.Timestamp("2025-01-01", tz="UTC")


def split_report(df: pd.DataFrame, p: pf.Params, name: str) -> None:
    taken = df[df.reason.isin(["TP", "STOP", "TIME", "SETTLE"])].copy()
    taken["et"] = pd.to_datetime(taken["entry_time"], utc=True)
    is_df = df[df.index.isin(taken[taken.et < OOS_START].index)]
    oos_df = df[df.index.isin(taken[taken.et >= OOS_START].index)]

    def line(d: dict, tag: str) -> str:
        if not d or d.get("n_taken", 0) == 0:
            return f"  {tag:<5} n=0"
        return (f"  {tag:<5} n={d['n_taken']:<4} win {d['win_pct']:>5}%  EV {d['ev_net_pct']:>+7}%  "
                f"med {d['median_net_pct']:>+7}%  sum {d['sum_net_pct']:>+8}%  worst {d['worst_trade_pct']:>+7}%  "
                f"stop {d['stop_rate_pct']:>4}%  tp {d['tp_rate_pct']:>4}%  finEq {d['final_equity']}  DD {d['max_dd_pct']}%")

    print(f"\n### {name}")
    print(line(pf.summarize(df, p, "ALL"), "ALL"))
    print(line(pf.summarize(is_df, p, "IS"), "IS"))
    print(line(pf.summarize(oos_df, p, "OOS"), "OOS"))


VARIANTS = [
    ("baseline (peak stop, no dedup)", dict()),
    ("dedup7", dict(cooldown_days=7)),
    ("dedup7 + stopcap12%", dict(cooldown_days=7, max_stop_pct=0.12)),
    ("dedup7 + stopcap20%", dict(cooldown_days=7, max_stop_pct=0.20)),
    ("dedup7 + stopcap30%", dict(cooldown_days=7, max_stop_pct=0.30)),
]


def main() -> int:
    for name, kw in VARIANTS:
        p = pf.Params(**kw)
        df = pf.run_study(p, workers=16)
        split_report(df, p, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
