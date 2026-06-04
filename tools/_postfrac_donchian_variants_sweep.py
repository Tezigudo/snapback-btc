"""debt #3 -- donchian-v3 slope-gate variants sweep (5 variants + locked baseline).

Runs A/B/C/D/E + baseline across:
  * 5 OOS windows (2022_H1 .. 2025_H1, same as multifactor-v1 path-2)
  * Quarterly walk-forward (2021-01-01 -> 2026-06-30, 3mo OOS slices)

Output:
  reports/donchian_variants_sweep.json
  reports/_donchian_variants_sweep_<variant>_<window>.csv (per OOS-window trades)
  reports/_donchian_variants_sweep_<variant>_wf_<qlabel>.csv (per WF-quarter trades)

Design notes (per advisor):
  * Each variant YAML is loaded directly with yaml.safe_load, NOT via
    StrategyParams.from_yaml (which silently drops donchian/regime/gate
    fields and would collapse all variants to identical defaults).
  * Class attrs are passed as bt.run(**class_attrs); backtesting.py
    requires every key to be a declared class attribute -- so we rely on
    DonchianBreakoutBTCv3 declaring use_ema_direction_filter / ema_direction_period
    / atr_breakout_buffer_mult (added in this debt #3 patch).
  * Direct Backtest() at $1M cash + PRICE_SCALE=0.001 -- matches postfrac
    convention so absolute numbers are comparable to prior shelved sweeps.
  * COMMISSION = 7.5 bps per side -> 15 bps round-trip; matches recent
    AdaptiveTrend / multifactor ablation runners.
  * NO 1h reindex -- 4H entry TF runs directly on 4H parquet; the
    attach_donchian helper would re-pull 1h data via load_klines (which
    can hit the network) so we skip it and build Donchian/ATR inline.
  * Warmup: 60d prefix for 4H sweeps (= 360 bars >= 1.5x EMA(200)).
    Trades attributed by EntryTime to the OOS slice.

Formula reconciled: live_donchian_v3.py now mirrors
strategy.regime_classifier.ema_slope_signed (endpoint-diff / slope_window
/ close[-1] × 100). Kill-condition #1 resolved — sweep is safe to run.
Any PSR from this harness is backtest-only.

This module is import-safe -- main() is only invoked under
`if __name__ == "__main__"`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr as _atr_fn  # noqa: E402
from strategy.signals_donchian import DonchianBreakoutBTCv3  # noqa: E402
from tools.aggregate import build_canonical_block  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

# --- harness constants (apples-to-apples across all 6 arms) -----------------

PARQ_4H = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"

CASH = 1_000_000.0
COMMISSION = 0.00075      # 7.5 bps / side = 15 bps round-trip
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001
WARM_PREFIX_DAYS = 60     # ~360 4H bars; >= 1.5x EMA(200) warmup

# 5 OOS windows -- exactly the multifactor-v1 path-2 windows.
OOS_WINDOWS: list[tuple[str, str, str]] = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]

# Walk-forward: quarterly OOS 2021..2026, matches the rv_band WF scheme
# (just on 4H instead of 15m).
WF_START = pd.Timestamp("2021-01-01")
WF_END = pd.Timestamp("2026-06-30")
SUFFICIENT_TRADES_THRESHOLD = 8   # plan kill-condition: <8 trades = INCONCLUSIVE

# --- variant resolution -----------------------------------------------------

# Baseline = the locked donchian-v3 cons params. Hard-coded so the runner
# stays valid even if config/params_donchian.yaml is ever bumped, and so
# the "baseline" arm is recorded in the run JSON without relying on the
# live YAML.
LOCKED_BASELINE = {
    "variant_id":          "baseline_locked_signed_slope_0.03",
    "strategy_name":       "donchian-v3",
    "symbol":              "BTC/USDT:USDT",
    "timeframe":           "4h",
    "class_attrs": {
        "donchian_period_entry":      80,
        "donchian_period_exit":       20,
        "atr_sl_multiple":            1.5,
        # atr_tp_multiple intentionally omitted -- Donchian has no TP.
        # Adding it would AttributeError on bt.run because the class doesn't
        # declare it (only StrategyParams has it for v1/multifactor compat).
        "atr_trail_multiple":         0.0,
        "time_stop_bars":             48,
        "risk_per_trade_pct":         2.75,
        "leverage":                   20,
        "allow_shorts":               True,
        "regime_ema_period":          120,
        "regime_slope_window":        30,
        "slope_trend_threshold_pct":  0.03,
        "use_ema_direction_filter":   False,
        "ema_direction_period":       200,
        "atr_breakout_buffer_mult":   0.0,
    },
    "notes": "locked donchian-v3 cons params -- live config/params_donchian.yaml mirror",
}

VARIANT_YAML_NAMES = [
    "params_donchian_v3_var_A.yaml",
    "params_donchian_v3_var_B.yaml",
    "params_donchian_v3_var_C.yaml",
    "params_donchian_v3_var_D.yaml",
    "params_donchian_v3_var_E.yaml",
]

# atr_period for the inline Donchian/ATR attach (matches strategy default).
ATR_PERIOD = 20


def load_variants() -> list[dict]:
    """Load baseline + 5 YAML variants. Baseline is run first so failures
    in a variant YAML don't kill the comparator arm."""
    variants: list[dict] = [LOCKED_BASELINE]
    cfg_dir = ROOT / "config"
    for name in VARIANT_YAML_NAMES:
        p = cfg_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing variant YAML: {p}")
        with open(p) as f:
            cfg = yaml.safe_load(f)
        if "class_attrs" not in cfg:
            raise ValueError(f"{name}: missing required `class_attrs` block")
        variants.append(cfg)
    return variants


# --- data prep -------------------------------------------------------------

def _load_full_scaled_4h() -> pd.DataFrame:
    df = pd.read_parquet(PARQ_4H)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] * PRICE_SCALE
    return df


def _attach_donchian_inline(
    df: pd.DataFrame,
    period_entry: int,
    period_exit: int,
    atr_period: int,
) -> pd.DataFrame:
    """Compute Donchian channels + ATR on the SAME 4H bars (single-TF setup).

    Mirrors strategy.signals_donchian.attach_donchian but reuses the entry-TF
    OHLC directly instead of going through a separate 1h frame. The
    `shift(1)` on each rolling column ensures the bar at T sees only data
    strictly before T (no lookahead).
    """
    out = df.copy()
    out["DonchianUpper"] = (
        out["Close"].rolling(period_entry, min_periods=period_entry).max().shift(1)
    )
    out["DonchianLower"] = (
        out["Close"].rolling(period_entry, min_periods=period_entry).min().shift(1)
    )
    out["DonchianExitUpper"] = (
        out["Close"].rolling(period_exit, min_periods=period_exit).max().shift(1)
    )
    out["DonchianExitLower"] = (
        out["Close"].rolling(period_exit, min_periods=period_exit).min().shift(1)
    )
    out["ATR_1h"] = _atr_fn(out["High"], out["Low"], out["Close"], atr_period).shift(1)
    return out


def _prep_slice(
    full_df: pd.DataFrame,
    warm_start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
    period_entry: int,
    period_exit: int,
) -> pd.DataFrame:
    """Slice the 4H parquet, then attach Donchian/ATR columns."""
    sl = full_df.loc[(full_df.index >= warm_start) & (full_df.index <= end_inclusive)].copy()
    if sl.empty:
        return sl
    return _attach_donchian_inline(sl, period_entry, period_exit, ATR_PERIOD)


# --- backtest runner --------------------------------------------------------

def _run_bt(
    df: pd.DataFrame,
    class_attrs: dict,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict:
    """Run a backtest on `df`, attribute trades by EntryTime to [test_start, test_end].

    Returns: {trades, return_pct (compounded over OOS trades only),
              pnl_pct list, max_dd_pct, win_rate_pct, equity_final, sharpe,
              stats_return_pct (full slice return for reference)}.
    """
    bt = Backtest(
        df,
        DonchianBreakoutBTCv3,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(**class_attrs)
    trades_df = getattr(stats, "_trades", None)

    pnl_pct_list: list[float] = []
    eq_impact_pnl_pct: list[float] = []
    n_oos = 0
    if (
        trades_df is not None
        and len(trades_df) > 0
        and "EntryTime" in trades_df.columns
        and "ReturnPct" in trades_df.columns
    ):
        et = pd.to_datetime(trades_df["EntryTime"])
        mask = (et >= test_start) & (et <= test_end)
        oos = trades_df.loc[mask]
        n_oos = int(len(oos))
        if n_oos > 0:
            pnl_pct_list = (oos["ReturnPct"].values * 100.0).tolist()
            # CANONICAL (v2): sizing-aware equity-impact returns over the OOS
            # trade subset (cash baseline; matches the OOS-only compounding).
            from tools.aggregate import equity_impact_returns  # local import
            stub = type("S", (), {"_trades": oos})()
            eq_impact_pnl_pct = equity_impact_returns(stub, cash=CASH).tolist()

    compounded_pct = 0.0
    if pnl_pct_list:
        c = 1.0
        for p in pnl_pct_list:
            c *= (1.0 + p / 100.0)
        compounded_pct = (c - 1.0) * 100.0

    return {
        "trades":           n_oos,
        "return_pct":       round(compounded_pct, 4),
        "pnl_pct":          pnl_pct_list,
        "eq_impact_pnl_pct": eq_impact_pnl_pct,
        "max_dd_pct":       float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
        "win_rate_pct":     float(stats.get("Win Rate [%]") or 0.0),
        "equity_final":     float(stats.get("Equity Final [$]", CASH) or CASH),
        "sharpe":           float(stats.get("Sharpe Ratio") or 0.0),
        "stats_return_pct": float(stats.get("Return [%]", 0.0) or 0.0),
        "entry_times":      [str(t) for t in oos["EntryTime"].tolist()]
                            if n_oos > 0 else [],
    }


def run_oos_window(
    full_df: pd.DataFrame,
    variant: dict,
    label: str,
    start: str,
    end: str,
    save_csv_path: Path,
) -> dict:
    test_start = pd.Timestamp(start)
    test_end = (
        pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    )
    warm_start = test_start - pd.Timedelta(days=WARM_PREFIX_DAYS)

    attrs = variant["class_attrs"]
    df = _prep_slice(
        full_df, warm_start, test_end,
        period_entry=int(attrs["donchian_period_entry"]),
        period_exit=int(attrs["donchian_period_exit"]),
    )
    if df.empty:
        return {"label": label, "start": start, "end": end,
                "error": "empty_slice", "trades": 0,
                "return_pct": 0.0, "pnl_pct": []}

    r = _run_bt(df, attrs, test_start, test_end)

    if r["trades"] > 0:
        pd.DataFrame({"pnl_pct": r["pnl_pct"], "entry_time": r["entry_times"]}) \
          .to_csv(save_csv_path, index=False)

    return {
        "label":            label,
        "start":            start,
        "end":              end,
        "trades":           r["trades"],
        "return_pct":       r["return_pct"],
        "max_dd_pct":       r["max_dd_pct"],
        "win_rate_pct":     r["win_rate_pct"],
        "equity_final":     r["equity_final"],
        "sharpe":           r["sharpe"],
        "stats_return_pct": r["stats_return_pct"],
        "pnl_pct":          r["pnl_pct"],
        "eq_impact_pnl_pct": r["eq_impact_pnl_pct"],
        "csv_path":         str(save_csv_path) if r["trades"] > 0 else None,
    }


# --- walk-forward -----------------------------------------------------------

def build_wf_windows() -> list[dict]:
    out = []
    ts = WF_START
    while ts < WF_END:
        test_start = ts
        test_end = min(ts + pd.DateOffset(months=3) - pd.Timedelta(days=1), WF_END)
        label = f"{test_start.strftime('%Y%m')}_3mo"
        out.append({"label": label, "test_start": test_start, "test_end": test_end})
        ts = ts + pd.DateOffset(months=3)
    return out


def run_wf_window(
    full_df: pd.DataFrame,
    variant: dict,
    w: dict,
    save_csv_path: Path,
) -> dict | None:
    test_start = w["test_start"]
    test_end = w["test_end"]
    end_inclusive = test_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    warm_start = test_start - pd.Timedelta(days=WARM_PREFIX_DAYS)

    attrs = variant["class_attrs"]
    df = _prep_slice(
        full_df, warm_start, end_inclusive,
        period_entry=int(attrs["donchian_period_entry"]),
        period_exit=int(attrs["donchian_period_exit"]),
    )
    if len(df) < 200:
        return None

    r = _run_bt(df, attrs, test_start, test_end)
    if r["trades"] > 0:
        pd.DataFrame({"pnl_pct": r["pnl_pct"], "entry_time": r["entry_times"]}) \
          .to_csv(save_csv_path, index=False)

    return {
        "label":          w["label"],
        "test_start":     test_start.date().isoformat(),
        "test_end":       test_end.date().isoformat(),
        "trades":         r["trades"],
        "return_pct":     r["return_pct"],
        "max_dd_pct":     r["max_dd_pct"],
        "pnl_pct":        r["pnl_pct"],
        "eq_impact_pnl_pct": r["eq_impact_pnl_pct"],
    }


# --- aggregation ------------------------------------------------------------

def aggregate_oos(per_window: list[dict]) -> dict:
    all_pnl: list[float] = []
    n_trades = 0
    n_pos = 0
    compounded = 1.0
    per_w_ret: list[float] = []
    worst_dd = 0.0
    for r in per_window:
        n_trades += r.get("trades", 0)
        all_pnl.extend(r.get("pnl_pct", []))
        rp = r.get("return_pct", 0.0) / 100.0
        compounded *= (1.0 + rp)
        per_w_ret.append(r.get("return_pct", 0.0))
        if r.get("return_pct", 0.0) > 0:
            n_pos += 1
        dd = r.get("max_dd_pct", 0.0) or 0.0
        if dd < worst_dd:
            worst_dd = dd
    # LEGACY stitched-per-trade PSR (N-inflated; observability only).
    legacy_psr_stitched = (
        compute_psr(np.asarray(all_pnl, dtype=float), sr_hurdle=0.0, confidence=0.95)
        if len(all_pnl) >= 2
        else {"n_trades": len(all_pnl), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )
    # CANONICAL (v2) dual-emit — 5-OOS family -> aggregation_method=v2_equity_curve.
    canon = build_canonical_block(per_window, aggregation_method="v2_equity_curve")
    psr = canon["psr_walkforward"]  # canonical headline PSR (gate reads this)
    return {
        "n_trades_total":           n_trades,
        "windows_positive":         f"{n_pos}/{len(per_window)}",
        "compounded_pct":           round((compounded - 1.0) * 100.0, 4),
        "per_window_return_pct":    [round(v, 4) for v in per_w_ret],
        "worst_window_max_dd_pct":  round(worst_dd, 4),
        "psr":                      psr,                  # canonical psr_walkforward
        "legacy_psr_stitched":      legacy_psr_stitched,  # observability only
        "canonical":                canon,                # v2 dual-emit block
        "aggregation_method":       canon["aggregation_method"],
    }


def aggregate_wf(per_window: list[dict]) -> dict:
    n_trades = 0
    n_pos = 0
    n_test = 0
    suff = 0
    pos_suff = 0
    compounded = 1.0
    all_pnl: list[float] = []
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        compounded *= 1.0 + r["return_pct"] / 100.0
        n_test += 1
        if r["return_pct"] > 0:
            n_pos += 1
        if r["trades"] >= SUFFICIENT_TRADES_THRESHOLD:
            suff += 1
            if r["return_pct"] > 0:
                pos_suff += 1
    pct_pos = (n_pos / n_test * 100.0) if n_test else 0.0
    pct_pos_suff = (pos_suff / suff * 100.0) if suff else 0.0
    # LEGACY stitched-per-trade PSR (N-inflated; observability only).
    legacy_psr_stitched = (
        compute_psr(np.asarray(all_pnl, dtype=float), sr_hurdle=0.0, confidence=0.95)
        if len(all_pnl) >= 2
        else {"n_trades": len(all_pnl), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )
    # CANONICAL (v2) dual-emit — WF family -> aggregation_method=v2_walkforward.
    canon = build_canonical_block(per_window, aggregation_method="v2_walkforward")
    psr = canon["psr_walkforward"]  # canonical headline PSR (window-level series)
    return {
        "quarters_total":               n_test,
        "quarters_positive":            n_pos,
        "pct_positive":                 round(pct_pos, 2),
        "sufficient_trades_threshold":  SUFFICIENT_TRADES_THRESHOLD,
        "quarters_with_sufficient_trades": suff,
        "pos_among_sufficient":         pos_suff,
        "pct_positive_sufficient":      round(pct_pos_suff, 2),
        "aggregate_compounded_pct":     round((compounded - 1.0) * 100.0, 4),
        "n_trades_total":               n_trades,
        "psr":                          psr,                  # canonical psr_walkforward
        "legacy_psr_stitched":          legacy_psr_stitched,  # observability only
        "canonical":                    canon,                # v2 dual-emit block
        "aggregation_method":           canon["aggregation_method"],
    }


# --- main -------------------------------------------------------------------

def run_sweep(skip_wf: bool = False, only_variants: list[str] | None = None) -> dict:
    """Run all 6 arms across 5 OOS windows + (optionally) WF quarters.

    Returns the full result dict; caller writes JSON.
    """
    t0 = time.time()
    print("[donchian-variants] loading 4H parquet ...", file=sys.stderr)
    full_df = _load_full_scaled_4h()
    print(f"  data range: {full_df.index.min()} -> {full_df.index.max()} "
          f"({len(full_df)} bars)", file=sys.stderr)

    variants = load_variants()
    if only_variants:
        variants = [v for v in variants if v["variant_id"] in only_variants]
    print(f"  variants to run: {[v['variant_id'] for v in variants]}",
          file=sys.stderr)

    wf_windows = build_wf_windows() if not skip_wf else []
    if not skip_wf:
        print(f"  walk-forward quarters planned: {len(wf_windows)}",
              file=sys.stderr)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    per_variant: dict[str, dict] = {}
    for v in variants:
        vid = v["variant_id"]
        print(f"\n[variant {vid}]", file=sys.stderr)

        # OOS sweep
        oos_results: list[dict] = []
        for label, start, end in OOS_WINDOWS:
            csv_path = reports_dir / f"_donchian_variants_sweep_{vid}_{label}.csv"
            tw = time.time()
            try:
                r = run_oos_window(full_df, v, label, start, end, csv_path)
            except Exception as exc:
                r = {"label": label, "start": start, "end": end,
                     "error": repr(exc), "trades": 0,
                     "return_pct": 0.0, "pnl_pct": []}
            print(
                f"  OOS {label}  trades={r.get('trades', 0):4d}  "
                f"ret={r.get('return_pct', 0.0):+8.2f}%  "
                f"dd={r.get('max_dd_pct', 0.0):+7.2f}%  "
                f"({time.time()-tw:.1f}s)",
                file=sys.stderr,
            )
            oos_results.append(r)

        oos_agg = aggregate_oos(oos_results)

        # Walk-forward
        wf_results: list[dict] = []
        if not skip_wf:
            for w in wf_windows:
                csv_path = reports_dir / f"_donchian_variants_sweep_{vid}_wf_{w['label']}.csv"
                try:
                    r = run_wf_window(full_df, v, w, csv_path)
                except Exception as exc:
                    r = None
                    print(f"  WF {w['label']} FAILED: {exc}", file=sys.stderr)
                if r is not None:
                    wf_results.append(r)
            wf_agg = aggregate_wf(wf_results)
        else:
            wf_agg = {"skipped": True}

        per_variant[vid] = {
            "variant_meta": {k: v[k] for k in ("variant_id", "strategy_name",
                                               "symbol", "timeframe") if k in v},
            "class_attrs":  v["class_attrs"],
            "notes":        v.get("notes", ""),
            "oos": {
                "per_window": [
                    {k: vv for k, vv in r.items()
                     if k not in ("pnl_pct", "entry_times", "eq_impact_pnl_pct")}
                    for r in oos_results
                ],
                "aggregate": oos_agg,
            },
            "walkforward": {
                "per_quarter": [
                    {k: vv for k, vv in r.items()
                     if k not in ("pnl_pct", "eq_impact_pnl_pct")}
                    for r in wf_results
                ],
                "aggregate": wf_agg,
            },
        }

    elapsed = time.time() - t0

    # --- bit-for-bit round-trip check (migration verification) --------------
    # For every variant's OOS and WF canonical block: the headline psr must
    # equal compute_psr on the PERSISTED per-window return series (rounded as
    # aggregate_windows stores it), contiguous=False (disjoint windows; this is
    # load-bearing for the WF arm where n>=20 would otherwise fire Lo's
    # correction and change psr_lo_adjusted).
    for vid, data in per_variant.items():
        for leg in ("oos", "walkforward"):
            agg = data.get(leg, {}).get("aggregate", {})
            if not isinstance(agg, dict) or "canonical" not in agg:
                continue  # skipped WF or empty
            canon = agg["canonical"]
            headline = agg["psr"]
            persisted = np.asarray(canon["per_window_return_pct"], dtype=float)
            recomputed = (
                compute_psr(persisted, sr_hurdle=0.0, confidence=0.95,
                            contiguous=False)
                if len(persisted) >= 2
                else {"psr_vs_hurdle": 0.0}
            )
            assert recomputed.get("psr_vs_hurdle") == headline.get("psr_vs_hurdle"), (
                f"canonical PSR round-trip MISMATCH [{vid}/{leg}]: recomputed="
                f"{recomputed.get('psr_vs_hurdle')} "
                f"headline={headline.get('psr_vs_hurdle')}"
            )
            print(
                f"[donchian-variants] round-trip OK [{vid}/{leg}]: "
                f"psr={headline.get('psr_vs_hurdle')} "
                f"method={canon.get('aggregation_method')}",
                file=sys.stderr,
            )

    # Gate evaluation per variant -- echoes the plan's promotion rule.
    baseline_compounded = per_variant.get(
        LOCKED_BASELINE["variant_id"], {}
    ).get("oos", {}).get("aggregate", {}).get("compounded_pct")
    baseline_wf_pct = per_variant.get(
        LOCKED_BASELINE["variant_id"], {}
    ).get("walkforward", {}).get("aggregate", {}).get("pct_positive_sufficient")

    gates = {
        "psr_min":                    0.90,
        "wf_min_pct":                 70.0,
        "lift_vs_baseline_min_pp":    0.0,
        "min_trades_per_window":      8,
        "max_dd_worst_window_pct":    -15.0,
        "wins_min_of_5":              4,
        "baseline_compounded_locked": baseline_compounded,
        "baseline_wf_pct_locked":     baseline_wf_pct,
    }

    promotion: dict[str, dict] = {}
    for vid, data in per_variant.items():
        if vid == LOCKED_BASELINE["variant_id"]:
            continue
        agg = data["oos"]["aggregate"]
        wf_agg = data["walkforward"].get("aggregate", {})
        psr_v = agg.get("psr", {}).get("psr_vs_hurdle")
        wins_str = agg.get("windows_positive", "0/5")
        wins = int(wins_str.split("/")[0]) if "/" in wins_str else 0
        worst_dd = agg.get("worst_window_max_dd_pct", 0.0)
        compounded = agg.get("compounded_pct", 0.0)
        wf_pct = wf_agg.get("pct_positive_sufficient") if isinstance(wf_agg, dict) else None
        lift_pp = (
            compounded - baseline_compounded
            if baseline_compounded is not None else None
        )
        # min trades per window
        min_tr = min((r.get("trades", 0) for r in data["oos"]["per_window"]), default=0)

        checks = {
            "psr_ge_0.90":          psr_v is not None and psr_v >= 0.90,
            "wf_pct_ge_70":         wf_pct is not None and wf_pct >= 70.0,
            "lift_pp_ge_0":         lift_pp is not None and lift_pp >= 0.0,
            "wins_ge_4_of_5":       wins >= 4,
            "worst_dd_above_-15":   worst_dd > -15.0,
            "min_trades_ge_8":      min_tr >= 8,
        }
        promotion[vid] = {
            "psr_vs_hurdle":              psr_v,
            "compounded_pct":             compounded,
            "lift_vs_baseline_pp":        lift_pp,
            "wins":                       wins_str,
            "worst_window_max_dd_pct":    worst_dd,
            "wf_pct_positive_sufficient": wf_pct,
            "min_trades_per_window":      min_tr,
            "checks":                     checks,
            "promote":                    all(checks.values()),
        }

    return {
        "schema_version":     1,
        "task":               "debt #3 -- donchian-v3 slope-gate variants sweep",
        "harness": {
            "cash":              CASH,
            "commission":        COMMISSION,
            "commission_note":   "7.5bps/side = 15bps round-trip",
            "margin":            MARGIN,
            "price_scale":       PRICE_SCALE,
            "warm_prefix_days":  WARM_PREFIX_DAYS,
            "atr_period":        ATR_PERIOD,
            "oos_windows":       OOS_WINDOWS,
            "wf_start":          str(WF_START.date()),
            "wf_end":            str(WF_END.date()),
            "sufficient_trades_threshold": SUFFICIENT_TRADES_THRESHOLD,
        },
        "gates":              gates,
        "per_variant":        per_variant,
        "promotion":          promotion,
        "kill_condition_1_resolved": (
            "live_donchian_v3.py formula reconciled with "
            "regime_classifier.ema_slope_signed (endpoint-diff / slope_window "
            "/ close[-1] * 100). PSR from this harness is backtest-only."
        ),
        "elapsed_sec":        round(elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-wf", action="store_true",
                        help="Skip walk-forward (5 OOS only, faster smoke test)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated variant_ids to run "
                             "(e.g. 'baseline_locked_signed_slope_0.03,A_lower_threshold_0.015')")
    args = parser.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    result = run_sweep(skip_wf=args.skip_wf, only_variants=only)
    out_path = ROOT / "reports" / "donchian_variants_sweep.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[donchian-variants] wrote {out_path}", file=sys.stderr)

    # one-line summary per variant
    for vid, data in result["per_variant"].items():
        agg = data["oos"]["aggregate"]
        wf = data.get("walkforward", {}).get("aggregate", {})
        psr_v = agg.get("psr", {}).get("psr_vs_hurdle")
        wf_pct = wf.get("pct_positive_sufficient") if isinstance(wf, dict) else None
        prom = result["promotion"].get(vid, {}).get("promote") if vid in result["promotion"] else "(baseline)"
        print(
            f"  {vid:42s} "
            f"compounded={agg.get('compounded_pct', 0.0):+8.2f}%  "
            f"wins={agg.get('windows_positive', '?')}  "
            f"psr={psr_v:.3f}  " if psr_v is not None else f"  {vid:42s} compounded={agg.get('compounded_pct', 0.0):+8.2f}%  "
            f"wins={agg.get('windows_positive', '?')}  psr=N/A  ",
        )
        print(
            f"    wf%_sufficient={wf_pct}  promote={prom}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
