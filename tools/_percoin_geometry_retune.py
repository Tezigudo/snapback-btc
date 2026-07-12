"""Per-coin geometry retune for ETH/SOL — Track C validation.

MULTIFACTOR_V1_DEEPENING.md showed the BTC-locked geometry does not transfer
(ETH -31.6%, SOL +13.9% partial) and said: if ETH/SOL exposure is wanted,
retune SL/TP geometry and RSI thresholds per coin. This runner does that with
a lookahead-clean protocol:

  1. TUNE window 2020-10-01 .. 2021-12-31 (strictly BEFORE the standard five
     OOS windows) — grid sweep sl_pct x tp_pct x rsi_long per coin on the
     4H-gated multifactor-v1 (MultiFactorMTF4H), rank by tune-window return
     with a >=15-trade floor.
  2. LOCK the per-coin winner, then validate on the SAME 5 OOS windows
     (2022H1..2025H1) used for every deployed-strategy decision, with the
     canonical PSR block.

Promote bar (repo convention): >=4/5 windows positive AND canonical
psr_walkforward >= 0.95 AND positive compounded. Anything less = SHELF —
the deepening verdict already predicts failure; this is the disciplined test.

Funding gate disabled for ETH/SOL (matches the deepening multi-coin protocol;
BTC's `no_funding` ablation showed the gate is barely active anyway).
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
    _load_slice,
    run_one,
    summarize,
)

TUNE = ("2020-10-01", "2021-12-31")
GRID_SL = (0.010, 0.015, 0.020)
GRID_TP = (0.020, 0.030, 0.045)
GRID_RSI = (30.0, 35.0, 40.0)
MIN_TUNE_TRADES = 15
COINS = ("ETH", "SOL")
BASE_OV = {"require_funding_not_extreme": False}


def prep_slice(coin: str, start: str, end: str):
    # The BASE class carries its own lookahead-safe 4H EMA200 gate; we point
    # it at the coin's OWN 4h parquet via mtf_4h_parquet_path (the class
    # default is BTC's file, which on ETH/SOL makes longs structurally
    # impossible — ETH close can never exceed BTC's 4H EMA).
    return _load_slice(PARQ[coin], start, end, attach_funding=False)


def coin_ov(coin: str) -> dict:
    return {**BASE_OV,
            "mtf_4h_parquet_path": str(ROOT / "data" / "historical" / f"{coin}_USDT_USDT_4h.parquet")}


def main() -> int:
    out = {"tune_window": f"{TUNE[0]} .. {TUNE[1]}",
           "grid": {"sl": GRID_SL, "tp": GRID_TP, "rsi_long": GRID_RSI},
           "coins": {}}
    for coin in COINS:
        print(f"[retune] {coin}: loading tune slice ...", file=sys.stderr)
        tune_df = prep_slice(coin, *TUNE)
        rows = []
        for sl in GRID_SL:
            for tp in GRID_TP:
                for rsi in GRID_RSI:
                    ov = {**coin_ov(coin), "sl_pct": sl, "tp_pct": tp,
                          "rsi_long_threshold": rsi}
                    r = run_one(tune_df, ov)
                    rows.append({"sl": sl, "tp": tp, "rsi": rsi,
                                 "trades": r["trades"],
                                 "return_pct": round(r["return_pct"], 2),
                                 "max_dd_pct": round(r["max_dd_pct"], 2)})
                    print(f"  {coin} sl={sl} tp={tp} rsi={rsi} -> "
                          f"{r['trades']}tr {r['return_pct']:.1f}%",
                          file=sys.stderr)
        eligible = [r for r in rows if r["trades"] >= MIN_TUNE_TRADES]
        coin_block = {"tune_grid": rows}
        if not eligible:
            coin_block["verdict"] = "SHELF_no_eligible_tune_cell"
            out["coins"][coin] = coin_block
            continue
        best = max(eligible, key=lambda r: r["return_pct"])
        coin_block["locked_pick"] = best
        print(f"[retune] {coin} locked: {best}", file=sys.stderr)

        # OOS validation of the locked pick on the standard 5 windows
        ov = {**coin_ov(coin), "sl_pct": best["sl"], "tp_pct": best["tp"],
              "rsi_long_threshold": best["rsi"]}
        per_window = {}
        for label, s, e in WINDOWS:
            df = prep_slice(coin, s, e)
            per_window[label] = run_one(df, ov)
            print(f"  {coin} OOS {label}: {per_window[label]['trades']}tr "
                  f"{per_window[label]['return_pct']:.1f}%", file=sys.stderr)
        summ = summarize(f"{coin}_retuned", per_window)
        coin_block["oos"] = {
            "summary": summ["summary"],
            "psr_walkforward": summ["canonical"].get("psr_walkforward"),
        }
        s = summ["summary"]
        wins = int(s["windows_positive"].split("/")[0])
        psr = (summ["canonical"].get("psr_walkforward") or {}).get("psr_vs_hurdle", 0)
        promote = wins >= 4 and psr >= 0.95 and s["compounded_pct"] > 0
        coin_block["verdict"] = "PROMOTE_CANDIDATE" if promote else "SHELF"
        out["coins"][coin] = coin_block

    path = ROOT / "reports" / "percoin_geometry_retune.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({c: {k: v for k, v in b.items() if k != "tune_grid"}
                      for c, b in out["coins"].items()}, indent=2, default=str))
    print(f"[retune] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
