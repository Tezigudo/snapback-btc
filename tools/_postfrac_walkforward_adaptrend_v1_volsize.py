"""Walk-forward validation: AdaptiveTrendV1 + vol_scaled_sizing.

Rolling 6mo-train / 3mo-test windows from 2020-01-01 to 2026-02-28 at $1M
cash with fractional sizing (PRICE_SCALE=0.001).

The strategy is deterministic (no fit/train step) — the 6mo "train" prefix
serves purely as warmup so indicators (H6 EMAs, 80-bar realised vol, etc.)
are spun up before the OOS test slice. Trades that close inside the test
slice (entry_time within test window) are counted toward that window's
return.

Output:
  reports/postfrac_walkforward_adaptrend_v1_volsize.json
  reports/_postfrac_wf_adaptrend_v1_volsize_<test_label>.csv (per-window trades)
  reports/_postfrac_wf_adaptrend_v1_volsize_AGG.csv          (all trades)

Verdict: PASSES_WALKFORWARD if >=70% test windows positive.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_adaptive_trend_v1_vol_scaled_sizing import (  # noqa: E402
    AdaptiveTrendV1_vol_scaled_sizing,
)
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

CONFIG = {"use_vol_scaled_sizing": True}

WF_START = pd.Timestamp("2020-01-01")
WF_END = pd.Timestamp("2026-02-28")


def build_walkforward_windows() -> list[dict]:
    out = []
    ts = WF_START
    while ts < WF_END:
        test_start = ts
        test_end = min(
            ts + pd.DateOffset(months=3) - pd.Timedelta(days=1), WF_END
        )
        train_start = ts - pd.DateOffset(months=6)
        label = f"{test_start.year}_{((test_start.month-1)//3)+1}Q"
        # disambiguate years that wrap
        label = f"{test_start.strftime('%Y%m')}_3mo"
        out.append(
            {
                "label":       label,
                "train_start": train_start,
                "test_start":  test_start,
                "test_end":    test_end,
            }
        )
        ts = ts + pd.DateOffset(months=3)
    return out


def _load_full_scaled() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] * PRICE_SCALE
    return df


def run_window(
    full_df: pd.DataFrame,
    train_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    label: str,
) -> dict | None:
    # Slice: [train_start, test_end]; we filter trades to those that ENTERED
    # within [test_start, test_end] for the test stat (warmup discards).
    sl = full_df.loc[
        (full_df.index >= train_start) & (full_df.index <= test_end)
    ].copy()
    if len(sl) < 200:
        print(f"  [{label}] SKIP — too few bars ({len(sl)})", file=sys.stderr)
        return None

    bt = Backtest(
        sl,
        AdaptiveTrendV1_vol_scaled_sizing,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(**CONFIG)

    trades_df = getattr(stats, "_trades", None)
    pnl_pct_list: list[float] = []
    n_trades_oos = 0

    # Index trades by EntryTime (which is the bar timestamp in the slice).
    if (
        trades_df is not None
        and len(trades_df) > 0
        and "EntryTime" in trades_df.columns
        and "ReturnPct" in trades_df.columns
    ):
        et = pd.to_datetime(trades_df["EntryTime"])
        mask = (et >= test_start) & (et <= test_end)
        oos = trades_df.loc[mask]
        n_trades_oos = int(len(oos))
        if n_trades_oos > 0:
            pnl_pct_list = (oos["ReturnPct"].values * 100.0).tolist()

            out_csv = ROOT / "reports" / f"_postfrac_wf_adaptrend_v1_volsize_{label}.csv"
            pd.DataFrame(
                {
                    "pnl_pct":     pnl_pct_list,
                    "test_start":  test_start.date().isoformat(),
                    "test_end":    test_end.date().isoformat(),
                    "entry_time":  oos["EntryTime"].astype(str).tolist(),
                    "exit_time":   oos["ExitTime"].astype(str).tolist()
                                   if "ExitTime" in oos.columns
                                   else [""] * n_trades_oos,
                }
            ).to_csv(out_csv, index=False)

    # Compounded return for this OOS slice = product(1 + pnl_pct/100) - 1
    # because trades are sequential.
    compounded_pct = 0.0
    if pnl_pct_list:
        c = 1.0
        for p in pnl_pct_list:
            c *= (1.0 + p / 100.0)
        compounded_pct = (c - 1.0) * 100.0

    return {
        "label":          label,
        "train_start":    train_start.date().isoformat(),
        "test_start":     test_start.date().isoformat(),
        "test_end":       test_end.date().isoformat(),
        "trades":         n_trades_oos,
        "return_pct":     round(compounded_pct, 4),
        "pnl_pct":        pnl_pct_list,
    }


def main() -> int:
    t0 = time.time()
    print("[wf adaptrend_v1_volsize] loading 15m parquet...", file=sys.stderr)
    full_df = _load_full_scaled()
    print(f"  data range: {full_df.index.min()} -> {full_df.index.max()}",
          file=sys.stderr)

    windows = build_walkforward_windows()
    print(f"  {len(windows)} walk-forward windows planned", file=sys.stderr)

    per_window: list[dict] = []
    for w in windows:
        tw = time.time()
        r = run_window(
            full_df,
            w["train_start"], w["test_start"], w["test_end"], w["label"],
        )
        if r is None:
            continue
        print(
            f"  {w['label']}  test={w['test_start'].date()}->{w['test_end'].date()}  "
            f"trades={r['trades']:4d}  ret={r['return_pct']:+8.2f}%  "
            f"({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )
        per_window.append(r)

    # Aggregate
    all_pnl: list[float] = []
    n_trades = 0
    n_pos = 0
    n_test = 0
    compounded = 1.0
    per_win_ret = []
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        n_test += 1
        if r["return_pct"] > 0:
            n_pos += 1

    pct_positive = (n_pos / n_test * 100.0) if n_test else 0.0
    verdict = (
        "PASSES_WALKFORWARD"
        if pct_positive >= 70.0
        else "FAILS_WALKFORWARD"
    )

    pnl_arr = np.asarray(all_pnl, dtype=float)
    # LEGACY stitched-per-trade PSR (N-inflated; observability only).
    legacy_psr_stitched = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(pnl_arr) >= 2
        else {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    agg_csv = ROOT / "reports" / "_postfrac_wf_adaptrend_v1_volsize_AGG.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)

    # Canonical v2 dual-emit — WF FAMILY: tag distinctly so it is NEVER
    # cross-compared with 5-OOS family numbers.  Per-quarter return_pct is
    # legitimately prod(1+ReturnPct) here because the underlying engine's
    # headline spans train+OOS.
    canon_windows = [
        {
            "label": r["label"],
            "return_pct": r["return_pct"],
            "trades": r["trades"],
            "pnl_pct": r.get("pnl_pct", []),
            "eq_impact_pnl_pct": [],
        }
        for r in per_window
    ]
    canon = build_canonical_block(
        canon_windows, aggregation_method="v2_walkforward"
    )
    # Headline PSR migrated to the canonical window-level aggregation
    # (psr_walkforward, n==n_quarters). Defeats the N-inflation of the
    # stitched per-trade ReturnPct union. Stitched value kept as
    # legacy_psr_stitched (observability only).
    psr = canon["psr_walkforward"]

    result = {
        "strategy_id":        "adaptrend_v1_volsize",
        "strategy_class":     "strategy.signals_adaptive_trend_v1_vol_scaled_sizing:"
                              "AdaptiveTrendV1_vol_scaled_sizing",
        "config":             CONFIG,
        "cash":               CASH,
        "commission":         COMMISSION,
        "margin":             MARGIN,
        "price_scale":        PRICE_SCALE,
        "walkforward": {
            "scheme":         "rolling 6mo-train / 3mo-test",
            "start":          str(WF_START.date()),
            "end":            str(WF_END.date()),
            "n_test_windows": n_test,
            "n_pos_windows":  n_pos,
            "pct_positive":   round(pct_positive, 2),
            "aggregate_compounded_pct": round((compounded - 1.0) * 100.0, 4),
            "n_trades_total": n_trades,
            "per_window":     [
                {k: v for k, v in r.items() if k != "pnl_pct"}
                for r in per_window
            ],
        },
        "psr":                psr,           # canonical psr_walkforward (headline)
        "legacy_psr_stitched": legacy_psr_stitched,  # observability only
        "canonical":          canon,         # v2 dual-emit (WF-tagged)
        "aggregation_method": canon["aggregation_method"],
        "verdict":            verdict,
        "verdict_threshold":  "PASSES if >=70% test windows positive",
        "elapsed_sec":        round(time.time() - t0, 2),
    }

    # --- bit-for-bit round-trip check (migration verification) --------------
    # contiguous=False matches aggregate_windows' psr_walkforward computation
    # on the ROUNDED per_window_return_pct array.
    persisted = np.asarray(canon["per_window_return_pct"], dtype=float)
    recomputed = (
        compute_psr(persisted, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(persisted) >= 2
        else {"psr_vs_hurdle": 0.0}
    )
    assert recomputed.get("psr_vs_hurdle") == psr.get("psr_vs_hurdle"), (
        f"canonical PSR round-trip MISMATCH: recomputed="
        f"{recomputed.get('psr_vs_hurdle')} headline={psr.get('psr_vs_hurdle')}"
    )
    print(
        f"[wf adaptrend_v1_volsize] canonical PSR round-trip OK: "
        f"{psr.get('psr_vs_hurdle')}",
        file=sys.stderr,
    )

    out_path = ROOT / "reports" / "postfrac_walkforward_adaptrend_v1_volsize.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[wf adaptrend_v1_volsize] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    print(f"  pct_positive={pct_positive:.2f}% verdict={verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
