"""Quarterly walk-forward: AdaptiveTrendV1 + RV-band gate.

Rolling 3mo OOS windows from 2021-01-01 to 2026-06-30 at $1M cash with
fractional sizing (PRICE_SCALE=0.001).  Each window's data slice begins
WARM_PREFIX_DAYS (=395d) before window_start so the RV-band gate
(rv_window_hours=720 + rv_rank_lookback_days=365) is fully warmed up
when the OOS window begins.  Trades are attributed by EntryTime to the
OOS slice only; warmup discards.

Pinned config (from postfrac_adaptrend_v1_rv_band default):
    alpha=2.0, rv_band_lo=0.25, rv_band_hi=0.75,
    rv_window_hours=720, rv_rank_lookback_days=365.

Commission: 7.5 bps per side = 15 bps round-trip (matches base ablation
runner so the WF cost basis is consistent).

Output:
  reports/postfrac_walkforward_adaptrend_v1_rv_band.json
  reports/_postfrac_wf_adaptrend_v1_rv_band_<label>.csv (per-window trades)
  reports/_postfrac_wf_adaptrend_v1_rv_band_AGG.csv     (all trades)

Verdict: PASSES_WALKFORWARD if >=70% test windows positive.  Compared
against the volsize baseline 56% (14/25, 2020-2026 sweep).
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

from strategy.signals_adaptive_trend_v1_rv_band import (  # noqa: E402
    AdaptiveTrendV1_rv_band,
)
from tools.aggregate import (  # noqa: E402
    build_canonical_block,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.00075   # 7.5 bps per side = 15 bps round-trip
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

# Pinned RV-band config (matches default tag in the ablation runner).
RV_WINDOW_HOURS = 720
RV_LOOKBACK_DAYS = 365
RV_BAND_LO = 0.25
RV_BAND_HI = 0.75
CONFIG = {
    "alpha": 2.0,
    "rv_window_hours": RV_WINDOW_HOURS,
    "rv_rank_lookback_days": RV_LOOKBACK_DAYS,
    "rv_band_lo": RV_BAND_LO,
    "rv_band_hi": RV_BAND_HI,
}

# 30d (rv_window_hours=720) + 365d rank lookback = 395d warm prefix.
WARM_PREFIX_DAYS = 395

WF_START = pd.Timestamp("2021-01-01")
WF_END = pd.Timestamp("2026-06-30")

SUFFICIENT_TRADES_THRESHOLD = 5


def build_walkforward_windows() -> list[dict]:
    out = []
    ts = WF_START
    while ts < WF_END:
        test_start = ts
        test_end = min(
            ts + pd.DateOffset(months=3) - pd.Timedelta(days=1), WF_END
        )
        label = f"{test_start.strftime('%Y%m')}_3mo"
        out.append(
            {
                "label":       label,
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
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    label: str,
) -> dict | None:
    """Run one OOS quarter with warm-prefix slice [test_start-395d, test_end]."""
    warm_start = test_start - pd.Timedelta(days=WARM_PREFIX_DAYS)
    end_inclusive = test_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = full_df.loc[
        (full_df.index >= warm_start) & (full_df.index <= end_inclusive)
    ].copy()
    if len(sl) < 200:
        print(f"  [{label}] SKIP - too few bars ({len(sl)})", file=sys.stderr)
        return None

    bt = Backtest(
        sl,
        AdaptiveTrendV1_rv_band,
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

            out_csv = ROOT / "reports" / f"_postfrac_wf_adaptrend_v1_rv_band_{label}.csv"
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

    compounded_pct = 0.0
    if pnl_pct_list:
        c = 1.0
        for p in pnl_pct_list:
            c *= (1.0 + p / 100.0)
        compounded_pct = (c - 1.0) * 100.0

    return {
        "label":          label,
        "warm_start":     warm_start.date().isoformat(),
        "test_start":     test_start.date().isoformat(),
        "test_end":       test_end.date().isoformat(),
        "trades":         n_trades_oos,
        "return_pct":     round(compounded_pct, 4),
        "pnl_pct":        pnl_pct_list,
    }


def main() -> int:
    t0 = time.time()
    print("[wf adaptrend_v1_rv_band] loading 15m parquet...", file=sys.stderr)
    full_df = _load_full_scaled()
    print(f"  data range: {full_df.index.min()} -> {full_df.index.max()}",
          file=sys.stderr)

    windows = build_walkforward_windows()
    print(f"  {len(windows)} walk-forward quarters planned", file=sys.stderr)

    per_window: list[dict] = []
    for w in windows:
        tw = time.time()
        r = run_window(full_df, w["test_start"], w["test_end"], w["label"])
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
    quarters_with_sufficient_trades = 0
    pos_among_sufficient = 0
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        n_test += 1
        if r["return_pct"] > 0:
            n_pos += 1
        if r["trades"] >= SUFFICIENT_TRADES_THRESHOLD:
            quarters_with_sufficient_trades += 1
            if r["return_pct"] > 0:
                pos_among_sufficient += 1

    pct_positive = (n_pos / n_test * 100.0) if n_test else 0.0
    pct_positive_sufficient = (
        (pos_among_sufficient / quarters_with_sufficient_trades * 100.0)
        if quarters_with_sufficient_trades else 0.0
    )
    verdict = (
        "PASSES_WALKFORWARD"
        if pct_positive_sufficient >= 70.0
        else "FAILS_WALKFORWARD"
    )

    # LEGACY (kept for observability/diff only): stitched per-trade ReturnPct
    # PSR across DISJOINT walk-forward quarters. N-inflated + sizing-blind —
    # never the headline. contiguous=False so the Lo serial-corr correction is
    # a no-op on the spurious cross-window autocorrelation.
    pnl_arr = np.asarray(all_pnl, dtype=float)
    legacy_psr_stitched = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(pnl_arr) >= 2
        else {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )
    legacy_psr_stitched["deprecation"] = (
        "stitched_per_trade_pl_pct_psr_is_N_inflated"
    )

    # CANONICAL (methodology debt #1): equity-curve aggregation, WF-tagged so
    # it is NEVER cross-compared with 5-OOS family numbers. Per-quarter
    # return_pct is legitimately prod(1+ReturnPct) here (warm-prefix harness;
    # OOS-attributed). Headline PSR = psr_walkforward, computed on the n-quarter
    # return series (n == n_windows) — defeats the stitched N-inflation.
    canon_windows = [
        {
            "label":            r["label"],
            "return_pct":       r["return_pct"],
            "trades":           r["trades"],
            "pnl_pct":          r.get("pnl_pct", []),
            "eq_impact_pnl_pct": [],
        }
        for r in per_window
    ]
    canon = build_canonical_block(
        canon_windows, aggregation_method="v2_walkforward"
    )
    # Headline canonical PSR (window-level walk-forward return series).
    psr = dict(canon["psr_walkforward"])

    agg_csv = ROOT / "reports" / "_postfrac_wf_adaptrend_v1_rv_band_AGG.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)

    result = {
        "strategy_id":        "adaptrend_v1_rv_band",
        "strategy_class":     "strategy.signals_adaptive_trend_v1_rv_band:"
                              "AdaptiveTrendV1_rv_band",
        "config":             CONFIG,
        "cash":               CASH,
        "commission":         COMMISSION,
        "commission_note":    "7.5 bps per side = 15 bps round-trip",
        "margin":             MARGIN,
        "price_scale":        PRICE_SCALE,
        "warm_prefix_days":   WARM_PREFIX_DAYS,
        "walkforward": {
            "scheme":            "quarterly 3mo OOS, warm-prefix 395d",
            "start":             str(WF_START.date()),
            "end":               str(WF_END.date()),
            "quarters_total":    n_test,
            "quarters_positive": n_pos,
            "pct_positive":      round(pct_positive, 2),
            "sufficient_trades_threshold": SUFFICIENT_TRADES_THRESHOLD,
            "quarters_with_sufficient_trades": quarters_with_sufficient_trades,
            "pos_among_sufficient":           pos_among_sufficient,
            "pct_positive_sufficient":        round(pct_positive_sufficient, 2),
            "aggregate_compounded_pct":       round((compounded - 1.0) * 100.0, 4),
            "n_trades_total":                 n_trades,
            "per_window":                     [
                {k: v for k, v in r.items() if k != "pnl_pct"}
                for r in per_window
            ],
        },
        "psr":                psr,                  # canonical headline (WF)
        "legacy_psr_stitched": legacy_psr_stitched,  # observability only
        "canonical":          canon,                # full dual-emit block
        "aggregation_method": canon["aggregation_method"],
        "verdict":            verdict,
        "verdict_threshold":  "PASSES if >=70% sufficient-trade quarters positive",
        "baseline_reference": {
            "source": "reports/postfrac_walkforward_adaptrend_v1_volsize.json",
            "scheme": "2020-2026 quarterly 3mo",
            "pct_positive": 56.0,
            "n_test_windows": 25,
            "n_pos_windows":  14,
        },
        "elapsed_sec":        round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_walkforward_adaptrend_v1_rv_band.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[wf adaptrend_v1_rv_band] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    print(
        f"  quarters_total={n_test}  pct_positive={pct_positive:.2f}%  "
        f"sufficient={quarters_with_sufficient_trades}  "
        f"pct_positive_sufficient={pct_positive_sufficient:.2f}%  "
        f"verdict={verdict}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
