"""AdaptiveTrendV1 + RV-band gate ablation runner (post-fractional-refactor).

Compares AdaptiveTrendV1 (base, alpha=2.0) vs AdaptiveTrendV1_rv_band
(same alpha, plus two-sided RV-percentile entry gate) on the 5 OOS
windows at $1M cash with PRICE_SCALE=0.001 fractional sizing.

Hypothesis (pinned)
-------------------
Trend-following pays best in the MIDDLE of the realised-vol distribution.
Block bottom 25% (dead vol, noise breakouts) AND top 25% (vol-crush,
regime breaks).  Default band = [0.25, 0.75], rank window = 365d.

Promotion bar (mirrors sibling vol_gate runner):
  - Compounded equity must improve by >= +5pp over base (target >= +50.52%).
  - PSR must NOT decrease vs base (0.905).

Parameterisation via env vars:
  RV_BAND_LO         (default "0.25")  — lower band edge
  RV_BAND_HI         (default "0.75")  — upper band edge
  RV_LOOKBACK_DAYS   (default "365")   — rank window (days)
  RV_TAG             (default "default") — output filename suffix

Writes:
  - reports/postfrac_adaptrend_v1_rv_band_<RV_TAG>.json
  - reports/_postfrac_adaptrend_v1_rv_band_<RV_TAG>_<window>.csv
  - reports/_postfrac_adaptrend_v1_rv_band_<RV_TAG>_baseRECHECK_<window>.csv
  - reports/_postfrac_adaptrend_v1_rv_band_<RV_TAG>_aggregated.csv
  - reports/_postfrac_adaptrend_v1_rv_band_<RV_TAG>_baseRECHECK_aggregated.csv

PRICE_SCALE pattern mirrors tools/_postfrac_adaptrend_v1_vol_gate.py —
apples to apples.  NO funding net applied (matches the base's
gross-of-funding basis), so the delta vs base is purely the gate effect.

Warm-prefix harness
-------------------
The RV-band gate needs ~(30d + RV_LOOKBACK_DAYS) of hourly history
warmed up before it can emit a non-NaN percentile rank.  At the spec
default (RV_LOOKBACK_DAYS=365) that is ~395d, which exceeds each 6-month
OOS window — so a strict cold-start would leave the gate NaN-blocked
for the entire window (0 trades, observed in workflow w5mi9k5uu).

The fix lives in the RUNNER: each OOS window's data slice is widened to
start at (window_start - WARM_PREFIX_DAYS), so the strategy sees enough
prior history to compute the gate by the time the OOS window begins.

BOTH ARMS (base and rv_band) get the same wide slice so they are
directly comparable.  The base arm trades the same way it always did
inside the OOS portion — it just sees extra leading bars.

Trade attribution: after the backtest runs we filter the trades_df by
EntryTime >= window_start to keep only OOS-portion trades.  The
re-compounded return is built ONLY from those OOS trades' ReturnPct, and
max-drawdown is computed on the equity-curve slice from window_start
onward.  The per-window CSVs likewise contain only OOS trades.

Commission: 7.5 bps per side = 15 bps round-trip (matches TODO_LEG
gate #1 — realistic Binance USDT-M perp costs for our size band).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_adaptive_trend import AdaptiveTrendV1  # noqa: E402
from strategy.signals_adaptive_trend_v1_rv_band import (  # noqa: E402
    AdaptiveTrendV1_rv_band,
)
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
    equity_impact_returns,
    window_return_pct,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.00075   # 7.5 bps per side  →  15 bps round-trip (TODO_LEG gate #1)
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

# Warm prefix: 30d (rv_window_hours=720h) + RV_LOOKBACK_DAYS, with a 30d
# safety pad so the very first OOS bar has a non-NaN gate already.
# 720h=30d + 365d + 0 = 395d.  We use 395 directly (no extra pad) to
# match the spec; if the gate needs more head-room set this higher.
WARM_PREFIX_DAYS = 395

# --- Env-var contract (defaults match pinned spec) -------------------------
RV_BAND_LO = float(os.environ.get("RV_BAND_LO", "0.25"))
RV_BAND_HI = float(os.environ.get("RV_BAND_HI", "0.75"))
RV_LOOKBACK_DAYS = int(os.environ.get("RV_LOOKBACK_DAYS", "365"))
RV_TAG = os.environ.get("RV_TAG", "default")
# rv_window_hours is FIXED at 720 (30d annualised) per spec — not tunable here.
RV_WINDOW_HOURS = 720

# Hold alpha=2.0 to match the postfrac base run (see _postfrac_adaptrend_v1.py
# header for the reasoning).  The RV-band gate is the ONLY variable changing.
BASE_CONFIG = {"alpha": 2.0}
RV_CONFIG = {
    "alpha": 2.0,
    "rv_window_hours": RV_WINDOW_HOURS,
    "rv_rank_lookback_days": RV_LOOKBACK_DAYS,
    "rv_band_lo": RV_BAND_LO,
    "rv_band_hi": RV_BAND_HI,
}

WINDOWS_5 = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]


def _load_slice_scaled(start: str, end: str, warm_days: int = 0) -> pd.DataFrame | None:
    """Load BTC 15m bars between [start - warm_days, end], price-scaled.

    The OOS window remains [start, end] — the caller filters trades by
    EntryTime >= start to attribute results.  Warm-prefix bars give the
    strategy enough history to compute the RV-band gate.
    """
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start_ts = pd.Timestamp(start) - pd.Timedelta(days=warm_days)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sl) == 0:
        return None

    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * PRICE_SCALE
    return sl


def run_window(
    strategy_cls: Type[Strategy],
    config: dict,
    label: str,
    start: str,
    end: str,
    csv_prefix: str,
) -> dict | None:
    df = _load_slice_scaled(start, end, warm_days=WARM_PREFIX_DAYS)
    if df is None:
        print(f"  [{label}] SKIP - no data", file=sys.stderr)
        return None

    window_start_ts = pd.Timestamp(start)

    bt = Backtest(
        df,
        strategy_cls,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(**config)

    # --- OOS trade attribution -------------------------------------------
    trades_df = getattr(stats, "_trades", None)
    if trades_df is None or len(trades_df) == 0:
        oos_trades = pd.DataFrame(columns=["EntryTime", "ExitTime", "ReturnPct"])
    else:
        # backtesting.py uses 'EntryTime' (Timestamp) in _trades
        if "EntryTime" in trades_df.columns:
            oos_trades = trades_df[trades_df["EntryTime"] >= window_start_ts].copy()
        else:
            # fallback: keep all (shouldn't happen with backtesting>=0.3)
            oos_trades = trades_df.copy()

    pnl_pct_list: list[float] = []
    eq_impact_pnl_pct: list[float] = []
    n_trades = int(len(oos_trades))
    win_rate = 0.0
    legacy_compounded_oos = 0.0
    equity_final = CASH
    if n_trades > 0 and "ReturnPct" in oos_trades.columns:
        ret_pct_series = oos_trades["ReturnPct"].astype(float).values
        pnl_pct_list = (ret_pct_series * 100.0).tolist()
        # LEGACY (v1): sizing-blind prod(1+ReturnPct) — kept for diff only
        compounded_factor = float(np.prod(1.0 + ret_pct_series))
        legacy_compounded_oos = compounded_factor - 1.0
        equity_final = CASH * compounded_factor
        n_wins = int((ret_pct_series > 0).sum())
        win_rate = 100.0 * n_wins / n_trades

        # CANONICAL (v2): equity-impact returns, OOS-filtered
        # window_start kwarg ensures attribution matches OOS portion.
        stub_stats = type("S", (), {"_trades": oos_trades})()
        eq_impact_pnl_pct = (
            equity_impact_returns(stub_stats, cash=CASH).tolist()
        )

    # Canonical headline = backtesting.py Return [%] when available, else
    # fall back to the OOS-recompound (still sizing-blind under warm-prefix
    # because stats['Return [%]'] would cover the WIDE slice).  RV-band is
    # a warm-prefix runner — Return[%] spans the prefix too — so we keep
    # the OOS recompound for the v2 headline, but compute it via
    # equity_impact_returns so it's sizing-aware.
    if eq_impact_pnl_pct:
        c = 1.0
        for r in eq_impact_pnl_pct:
            c *= 1.0 + r / 100.0
        ret_pct = (c - 1.0) * 100.0
    else:
        ret_pct = 0.0
    legacy_ret_pct = legacy_compounded_oos * 100.0

    # --- OOS max drawdown -------------------------------------------------
    eq_curve = getattr(stats, "_equity_curve", None)
    max_dd = 0.0
    if eq_curve is not None and len(eq_curve) > 0 and "Equity" in eq_curve.columns:
        eq_slice = eq_curve.loc[eq_curve.index >= window_start_ts, "Equity"]
        if len(eq_slice) > 1:
            running_max = eq_slice.cummax()
            dd_series = (eq_slice / running_max - 1.0) * 100.0
            max_dd = float(dd_series.min())

    if csv_prefix and n_trades > 0:
        out_csv = ROOT / "reports" / f"{csv_prefix}_{label}.csv"
        out = pd.DataFrame(
            {
                "pnl_pct":      pnl_pct_list,
                "window_start": start,
                "window_end":   end,
            }
        )
        out.to_csv(out_csv, index=False)

    return {
        "label":              label,
        "start":              start,
        "end":                end,
        "trades":             n_trades,
        "return_pct":         round(ret_pct, 4),
        "legacy_return_pct":  round(legacy_ret_pct, 4),
        "max_dd_pct":         round(max_dd, 4),
        "win_rate_pct":       round(win_rate, 4),
        "equity_final":       round(equity_final, 4),
        "pnl_pct":            pnl_pct_list,
        "eq_impact_pnl_pct":  eq_impact_pnl_pct,
    }


def aggregate(per_window: list[dict]) -> dict:
    all_pnl: list[float] = []
    n_trades = 0
    n_pos = 0
    compounded = 1.0
    per_win_ret = []
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        if r["return_pct"] > 0:
            n_pos += 1
    return {
        "n_trades":              n_trades,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{len(per_window)}",
        "per_window_return_pct": per_win_ret,
        "all_pnl_pct":           all_pnl,
    }


def run_arm(
    arm_label: str,
    strategy_cls: Type[Strategy],
    config: dict,
    csv_prefix: str,
) -> dict:
    per_window: list[dict] = []
    print(f"[rv_band_ablation] running arm={arm_label} ({len(WINDOWS_5)} windows) ...",
          file=sys.stderr)
    for label, start, end in WINDOWS_5:
        tw = time.time()
        r = run_window(strategy_cls, config, label, start, end, csv_prefix)
        if r is None:
            continue
        print(
            f"  [{arm_label}] {label}  trades={r['trades']:4d}  "
            f"ret={r['return_pct']:+8.2f}%  dd={r['max_dd_pct']:+7.2f}%  "
            f"win={r['win_rate_pct']:5.2f}%  ({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )
        per_window.append(r)

    agg = aggregate(per_window)
    all_pnl = agg.pop("all_pnl_pct")

    pnl_arr = np.asarray(all_pnl, dtype=float)
    # LEGACY stitched-per-trade PSR (N-inflated; observability only).
    legacy_psr_stitched = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(pnl_arr) >= 2
        else {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    # Headline / verdict PSR migrated from the N-inflated stitched per-trade
    # ReturnPct union to the equity-curve window-level aggregation
    # (psr_walkforward); see verdict() below. Stitched kept as observability.
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)

    # --- bit-for-bit round-trip check (migration verification) --------------
    canon_psr = canon["psr_walkforward"]
    persisted = np.asarray(canon["per_window_return_pct"], dtype=float)
    recomputed = (
        compute_psr(persisted, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(persisted) >= 2
        else {"psr_vs_hurdle": 0.0}
    )
    assert recomputed.get("psr_vs_hurdle") == canon_psr.get("psr_vs_hurdle"), (
        f"[{arm_label}] canonical PSR round-trip MISMATCH: recomputed="
        f"{recomputed.get('psr_vs_hurdle')} headline={canon_psr.get('psr_vs_hurdle')}"
    )
    print(
        f"  [{arm_label}] canonical PSR round-trip OK: "
        f"{canon_psr.get('psr_vs_hurdle')}",
        file=sys.stderr,
    )

    if csv_prefix:
        agg_csv = ROOT / "reports" / f"{csv_prefix}_aggregated.csv"
        pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
        print(f"  [{arm_label}] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    return {
        "arm":        arm_label,
        "config":     config,
        "per_window": [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":    agg,
        "psr":        legacy_psr_stitched,  # legacy stitched (observability only)
        "canonical":  canon,                # v2 dual-emit (headline PSR = psr_walkforward)
        "aggregation_method": canon["aggregation_method"],
    }


def verdict(base: dict, rv: dict) -> dict:
    base_comp = base["summary"]["compounded_pct"]
    rv_comp = rv["summary"]["compounded_pct"]
    # Canonical headline PSR = equity-curve aggregation's psr_walkforward
    # (compute_psr on the n-window return series). The stitched per-trade PSR
    # is N-inflated and MUST NOT be the primary verdict input
    # (tools/aggregate.py docstring lines 33-35). Stitched values are still
    # emitted below as *_psr_stitched_legacy for observability/diff.
    base_psr = base["canonical"]["psr_walkforward"]["psr_vs_hurdle"]
    rv_psr = rv["canonical"]["psr_walkforward"]["psr_vs_hurdle"]
    base_psr_stitched = base["psr"]["psr_vs_hurdle"]
    rv_psr_stitched = rv["psr"]["psr_vs_hurdle"]
    base_sharpe = base["canonical"]["psr_walkforward"].get("point_sharpe")
    rv_sharpe = rv["canonical"]["psr_walkforward"].get("point_sharpe")
    base_trades = base["summary"]["n_trades"]
    rv_trades = rv["summary"]["n_trades"]

    delta_comp = rv_comp - base_comp
    delta_psr = rv_psr - base_psr
    delta_trades = rv_trades - base_trades

    cleared_compounded = delta_comp >= 5.0
    psr_not_worse = rv_psr >= base_psr - 1e-6

    if cleared_compounded and psr_not_worse:
        decision = "PROMOTE_CANDIDATE — extend to walk-forward 2020-2026"
    elif cleared_compounded and not psr_not_worse:
        decision = "ITERATE — compounded improves but PSR drops; risk-adjusted edge unclear"
    elif not cleared_compounded and psr_not_worse:
        decision = "ITERATE — PSR holds but compounded fails +5pp bar"
    else:
        decision = "SHELF — fails both bars"

    return {
        "base_compounded_pct":    base_comp,
        "rv_compounded_pct":      rv_comp,
        "delta_compounded_pp":    round(delta_comp, 4),
        "psr_basis":              "canonical_psr_walkforward",
        "base_psr":               base_psr,
        "rv_psr":                 rv_psr,
        "delta_psr":              round(delta_psr, 4),
        "base_psr_stitched_legacy": base_psr_stitched,
        "rv_psr_stitched_legacy":   rv_psr_stitched,
        "base_point_sharpe":      base_sharpe,
        "rv_point_sharpe":        rv_sharpe,
        "base_trades":            base_trades,
        "rv_trades":              rv_trades,
        "delta_trades":           delta_trades,
        "cleared_+5pp_bar":       cleared_compounded,
        "psr_not_worse":          psr_not_worse,
        "decision":               decision,
    }


def main() -> int:
    t0 = time.time()

    base_prefix = f"_postfrac_adaptrend_v1_rv_band_{RV_TAG}_baseRECHECK"
    rv_prefix   = f"_postfrac_adaptrend_v1_rv_band_{RV_TAG}"

    res_base = run_arm(
        "base",
        AdaptiveTrendV1,
        BASE_CONFIG,
        csv_prefix=base_prefix,
    )
    res_rv = run_arm(
        "rv_band",
        AdaptiveTrendV1_rv_band,
        RV_CONFIG,
        csv_prefix=rv_prefix,
    )

    v = verdict(res_base, res_rv)

    result = {
        "experiment":     "adaptrend_v1_rv_band",
        "rv_tag":         RV_TAG,
        "base_strategy":  "strategy.signals_adaptive_trend:AdaptiveTrendV1",
        "rv_strategy":    "strategy.signals_adaptive_trend_v1_rv_band:AdaptiveTrendV1_rv_band",
        "cash":           CASH,
        "commission":     COMMISSION,
        "commission_note": "7.5 bps per side = 15 bps round-trip (TODO_LEG gate #1)",
        "margin":         MARGIN,
        "price_scale":    PRICE_SCALE,
        "warm_prefix_days": WARM_PREFIX_DAYS,
        "aggregation_method": AGGREGATION_VERSION,
        "env_contract": {
            "RV_BAND_LO":       RV_BAND_LO,
            "RV_BAND_HI":       RV_BAND_HI,
            "RV_LOOKBACK_DAYS": RV_LOOKBACK_DAYS,
            "RV_TAG":           RV_TAG,
            "RV_WINDOW_HOURS_FIXED": RV_WINDOW_HOURS,
        },
        "windows":        [w[0] for w in WINDOWS_5],
        "base":           res_base,
        "rv":             res_rv,
        "verdict":        v,
        "reference_postfrac_base": {
            "source": "reports/postfrac_adaptrend_v1.json (set_5_OOS)",
            "compounded_pct": 45.5222,
            "n_trades": 255,
            "psr_vs_hurdle": 0.905331,
            "point_sharpe": 0.076677,
            "note": (
                "Reference is at 5bps RT.  Current run uses 15bps RT + "
                "warm-prefix harness, so base numbers WILL differ from the "
                "reference — both arms share the new cost basis so the "
                "delta is the gate effect under realistic costs."
            ),
        },
        "notes": (
            f"Warm-prefix harness: each window's data slice starts "
            f"{WARM_PREFIX_DAYS}d before window_start so the RV-band gate "
            f"(rv_window_hours=720 + rv_rank_lookback_days={RV_LOOKBACK_DAYS}) "
            f"is warmed up by the time the OOS window begins.  Trades, "
            f"win-rate, compounded return, max-DD, and per-window CSVs are "
            f"computed strictly on trades whose EntryTime >= window_start. "
            f"Commission = 15 bps round-trip (7.5 bps/side).  Base arm sees "
            f"identical warm prefix so OOS-portion stats are directly "
            f"comparable; both arms are apples-to-apples."
        ),
        "elapsed_sec":    round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / f"postfrac_adaptrend_v1_rv_band_{RV_TAG}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[rv_band_ablation] verdict={v['decision']}  "
        f"base={v['base_compounded_pct']:+.2f}% -> rv={v['rv_compounded_pct']:+.2f}% "
        f"(delta {v['delta_compounded_pp']:+.2f}pp)  "
        f"base_PSR={v['base_psr']:.3f} -> rv_PSR={v['rv_psr']:.3f}",
        file=sys.stderr,
    )
    print(f"[rv_band_ablation] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
