"""Funding-gate relax validation: 0.0005 -> 0.0015 on the DEPLOYED config.

The MULTIFACTOR_V1_DEEPENING.md `no_funding` result (+3.23pp) was measured on
the pre-4H-gate baseline. The deployed config includes the 4H EMA200 regime
gate, so this run re-measures the funding lever on the deployed geometry:

  arm A  deployed   : MTF4H gate + funding_extreme_threshold=0.0005 (anchor)
  arm B  relax_0015 : MTF4H gate + funding_extreme_threshold=0.0015 (candidate)
  arm C  no_funding : MTF4H gate + require_funding_not_extreme=False (reference)

Same 5 OOS windows / $1M / 5bps / 1/20 margin as tools/run_mf_deepening.py.
Writes reports/fundrelax_0015_validation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_mf_deepening import (  # noqa: E402
    LOCKED,
    PARQ,
    WINDOWS,
    MultiFactorMTF4H,
    _load_4h_ema200_aligned,
    _load_slice,
    run_one,
    summarize,
)

ARMS = {
    "deployed_0005": {},
    "relax_0015": {"funding_extreme_threshold": 0.0015},
    "no_funding": {"require_funding_not_extreme": False},
}


def main() -> int:
    slices = {}
    print("[fundrelax] loading BTC slices + 4H EMA ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        df = _load_slice(PARQ["BTC"], start, end, attach_funding=True)
        df["Ema4h"] = _load_4h_ema200_aligned(
            df.index, ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
        )
        slices[label] = df
        print(f"  {label} bars={len(df)}", file=sys.stderr)

    out = {"locked_config": LOCKED, "windows": [w[0] for w in WINDOWS], "arms": {}}
    for arm, ov in ARMS.items():
        print(f"[fundrelax] arm={arm} ov={ov}", file=sys.stderr)
        per_window = {}
        for label, _s, _e in WINDOWS:
            per_window[label] = run_one(slices[label], ov, strategy_class=MultiFactorMTF4H)
        out["arms"][arm] = summarize(arm, per_window)
        s = out["arms"][arm]["summary"]
        print(
            f"  -> trades={s['n_trades']} compounded={s['compounded_pct']}% "
            f"wins={s['windows_positive']} per_window={s['per_window_return_pct']}",
            file=sys.stderr,
        )

    base = out["arms"]["deployed_0005"]["summary"]["compounded_pct"]
    for arm, v in out["arms"].items():
        v["delta_compounded_pp"] = round(v["summary"]["compounded_pct"] - base, 4)

    out_path = ROOT / "reports" / "fundrelax_0015_validation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[fundrelax] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
