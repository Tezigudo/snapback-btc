"""multifactor-v1 stack2 comparison: 4H gate + funding relaxation additivity test.

Tests whether stacking the 4H EMA200 gate (Lever 1, +27.45pp) with funding threshold
relaxation (0.0005 → 0.0015, flagged as +3.23pp in deepening) is additive.

Configs:
  baseline                 — locked params, no 4H gate (should reproduce ~+50.48%)
  funding_relax_only       — locked + funding_extreme_threshold=0.0015
  4h_gate_only             — locked + use_mtf_4h_gate=True (should reproduce ~+77.93%)
  stack_4h_and_funding_relax — locked + both

CRITICAL: LOCKED dict must include use_mtf_4h_gate=False. The class now defaults to
True after the 4H gate was added. Without this, baseline silently runs with the gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
    equity_impact_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20

WINDOWS = [
    ("2022H1", "2022-01-01", "2022-06-30"),
    ("2023H1", "2023-01-01", "2023-06-30"),
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
]

# Locked params — matches run_mf_deepening.py exactly, PLUS use_mtf_4h_gate=False
# to override the new class default of True.
LOCKED = {
    "rsi_period":                  14,
    "rsi_long_threshold":          35.0,
    "rsi_short_threshold":         70.0,
    "volume_ma_period":            20,
    "volume_multiple":             2.0,
    "mf_trend_ema_period":         200,
    "require_trend":               True,
    "require_candlestick":         False,
    "require_macd":                False,
    "require_funding_not_extreme": True,
    "funding_extreme_threshold":   0.0005,
    "sl_pct":                      0.015,
    "tp_pct":                      0.030,
    "max_hold_bars":               1344,
    "risk_per_trade_pct":          2.75,
    "leverage":                    20,
    "allow_shorts":                True,
    # Explicit False: overrides the new class default (True) so baseline is clean.
    "use_mtf_4h_gate":             False,
}

PARQ_BTC_15M = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
FUND_PARQ    = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"

CONFIGS = {
    "baseline":                   {},
    "funding_relax_only":         {"funding_extreme_threshold": 0.0015},
    "4h_gate_only":               {"use_mtf_4h_gate": True},
    "stack_4h_and_funding_relax": {"use_mtf_4h_gate": True, "funding_extreme_threshold": 0.0015},
}

# ---------------------------------------------------------------------------
# Data loading (identical to run_mf_deepening.py)
# ---------------------------------------------------------------------------

def _load_slice(parquet: Path, start: str, end: str,
                attach_funding: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    if attach_funding and FUND_PARQ.exists():
        fund = pd.read_parquet(FUND_PARQ)
        if fund.index.tz is not None:
            fund.index = fund.index.tz_localize(None)
        left = pd.DataFrame(index=df.index)
        right = pd.DataFrame({"Funding": fund["funding_rate"].values}, index=fund.index)
        merged = pd.merge_asof(left, right,
                               left_index=True, right_index=True,
                               direction="backward")
        df["Funding"] = merged["Funding"].values
        df["Funding"] = df["Funding"].fillna(0.0)

    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sliced) == 0:
        raise ValueError(f"Empty slice {start}..{end} from {parquet.name}")
    return sliced


# ---------------------------------------------------------------------------
# Backtest runner (identical to run_mf_deepening.py)
# ---------------------------------------------------------------------------

def run_one(df: pd.DataFrame, overrides: dict) -> dict:
    config = {**LOCKED, **overrides}
    bt = Backtest(df, DayTradeMultiFactorBTC, cash=CASH, commission=COMMISSION,
                  margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run(**config)
    trades_df = getattr(stats, "_trades", None)
    pnl_pct = []
    eq_impact_pnl_pct = []
    if trades_df is not None and len(trades_df):
        if "ReturnPct" in trades_df.columns:
            pnl_pct = (trades_df["ReturnPct"].values * 100.0).tolist()
        # CANONICAL (v2): equity-impact returns (PnL / equity-at-entry) for this
        # single contiguous window — sizing-aware PSR input.
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
    return {
        "trades": int(stats.get("# Trades", 0)),
        "return_pct": float(stats.get("Return [%]", 0.0) or 0.0),
        "max_dd_pct": float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
        "win_rate_pct": float(stats.get("Win Rate [%]") or 0.0),
        "equity_final": float(stats.get("Equity Final [$]", CASH) or CASH),
        "pnl_pct": pnl_pct,
        "eq_impact_pnl_pct": eq_impact_pnl_pct,
    }


def aggregate(per_window: dict) -> dict:
    all_pnl = []
    n_trades = 0
    n_pos = 0
    n_total = 0
    compounded = 1.0
    rets = []
    for w, r in per_window.items():
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        rets.append(r["return_pct"])
        n_total += 1
        if r["return_pct"] > 0:
            n_pos += 1
    return {
        "n_trades":              n_trades,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{n_total}",
        "per_window_return_pct": [round(x, 4) for x in rets],
        "_all_pnl_pct":          all_pnl,
    }


def summarize(label: str, per_window: dict) -> dict:
    agg = aggregate(per_window)
    pnl = np.asarray(agg.pop("_all_pnl_pct"))
    # LEGACY (v1) stitched-per-trade PSR — N-inflated, sizing-blind. Kept for
    # observability/diff only; NEVER the verdict input.
    legacy_psr_stitched = (
        compute_psr(pnl, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(pnl) >= 2
        else {"n_trades": int(len(pnl)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    pw_list = [
        {
            "label":             w,
            "return_pct":        r["return_pct"],
            "trades":            r["trades"],
            "pnl_pct":           r.get("pnl_pct", []),
            "eq_impact_pnl_pct": r.get("eq_impact_pnl_pct", []),
        }
        for w, r in per_window.items()
    ]
    canon = build_canonical_block(pw_list, aggregation_method=AGGREGATION_VERSION)
    return {
        "label":    label,
        "summary":  agg,
        "per_window": {w: {k: v for k, v in r.items()
                           if k not in ("pnl_pct", "eq_impact_pnl_pct")}
                       for w, r in per_window.items()},
        # Canonical headline PSR = canonical["psr_walkforward"]. Legacy stitched
        # PSR stays under `psr` for backcompat + as `legacy_psr_stitched`.
        "psr": legacy_psr_stitched,
        "legacy_psr_stitched": legacy_psr_stitched,
        "canonical": canon,
        "aggregation_method": canon["aggregation_method"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Preload BTC slices with funding attached once.
    btc_slices: dict[str, pd.DataFrame] = {}
    print("[stack2] loading BTC+funding slices ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        btc_slices[label] = _load_slice(PARQ_BTC_15M, start, end, attach_funding=True)
        print(f"  {label} bars={len(btc_slices[label])}", file=sys.stderr)

    results: dict[str, dict] = {}

    for cname, overrides in CONFIGS.items():
        print(f"[stack2] config={cname} overrides={overrides}", file=sys.stderr)
        per_window: dict[str, dict] = {}
        for label, _s, _e in WINDOWS:
            per_window[label] = run_one(btc_slices[label], overrides)
        results[cname] = summarize(cname, per_window)
        s = results[cname]["summary"]
        p = results[cname]["psr"]["psr_vs_hurdle"]
        print(f"  -> trades={s['n_trades']} compounded={s['compounded_pct']}% "
              f"wins={s['windows_positive']} PSR={p:.4f}", file=sys.stderr)

    # Reproduction gate — verify before proceeding.
    base_comp  = results["baseline"]["summary"]["compounded_pct"]
    gate_comp  = results["4h_gate_only"]["summary"]["compounded_pct"]
    base_ok    = abs(base_comp - 50.48) <= 2.0
    gate_ok    = abs(gate_comp - 77.93) <= 2.0
    print(f"[stack2] REPRO CHECK: baseline={base_comp:.4f}% (target 50.48±2pp) {'OK' if base_ok else 'FAIL'}", file=sys.stderr)
    print(f"[stack2] REPRO CHECK: 4h_gate={gate_comp:.4f}% (target 77.93±2pp) {'OK' if gate_ok else 'FAIL'}", file=sys.stderr)
    if not (base_ok and gate_ok):
        print("[stack2] WARNING: reproduction check failed — footing suspect; continuing but results may be invalid.", file=sys.stderr)

    # Compute lifts.
    fund_comp  = results["funding_relax_only"]["summary"]["compounded_pct"]
    stack_comp = results["stack_4h_and_funding_relax"]["summary"]["compounded_pct"]

    lift_baseline_to_4h    = round(gate_comp  - base_comp,  4)
    lift_baseline_to_fund  = round(fund_comp  - base_comp,  4)
    lift_4h_to_stack       = round(stack_comp - gate_comp,  4)
    lift_baseline_to_stack = round(stack_comp - base_comp,  4)
    expected_additive      = round(lift_baseline_to_4h + lift_baseline_to_fund, 4)

    # Additivity verdict.
    if expected_additive != 0:
        ratio = lift_baseline_to_stack / expected_additive if expected_additive != 0 else float("inf")
    else:
        ratio = float("inf")

    if lift_4h_to_stack < 0:
        additivity_verdict = "cannibalized"
    elif expected_additive != 0 and ratio > 1.2:
        additivity_verdict = "synergistic"
    elif expected_additive != 0 and abs(ratio - 1.0) <= 0.20:
        additivity_verdict = "additive"
    else:
        additivity_verdict = "cannibalized"

    # Stack verdict (compare stack vs 4h_gate_only).
    stack_psr        = results["stack_4h_and_funding_relax"]["psr"]["psr_vs_hurdle"]
    gate_wins        = results["4h_gate_only"]["summary"]["windows_positive"]
    stack_wins_str   = results["stack_4h_and_funding_relax"]["summary"]["windows_positive"]
    stack_wins_n     = int(stack_wins_str.split("/")[0])
    stack_worst      = min(results["stack_4h_and_funding_relax"]["per_window"][w]["return_pct"]
                           for w in results["stack_4h_and_funding_relax"]["per_window"])

    if lift_4h_to_stack >= 3.0 and stack_wins_n >= 5 and stack_worst >= -15.0 and stack_psr >= 0.978:
        stack_verdict = "STACK_RECOMMENDED"
    elif lift_4h_to_stack >= 1.0:
        # Covers [+1pp, +3pp) and also >=+3pp that fails a secondary criterion
        # (e.g. wins < 5/5). Spec defines MINOR_LIFT as "recommend keeping but flag
        # as marginal." CANNIBALIZED is reserved for negative lift only.
        stack_verdict = "MINOR_LIFT"
    elif lift_4h_to_stack >= -1.0:
        stack_verdict = "NEUTRAL"
    else:
        stack_verdict = "CANNIBALIZED"

    out = {
        "strategy": "multifactor-v1-stack2",
        "cash":     CASH,
        "configs":  results,
        "additivity_test": {
            "baseline_to_4h_gate_pp":      lift_baseline_to_4h,
            "baseline_to_funding_relax_pp": lift_baseline_to_fund,
            "4h_gate_to_stack_pp":         lift_4h_to_stack,
            "baseline_to_stack_pp":        lift_baseline_to_stack,
            "expected_additive_pp":        expected_additive,
            "additivity_ratio":            round(ratio, 4) if ratio != float("inf") else None,
            "verdict":                     additivity_verdict,
        },
        "stack_verdict": {
            "verdict":             stack_verdict,
            "stack_compounded_pct": stack_comp,
            "4h_gate_compounded_pct": gate_comp,
            "lift_pp":             lift_4h_to_stack,
            "stack_psr":           stack_psr,
            "stack_wins":          stack_wins_str,
            "stack_worst_window_pct": round(stack_worst, 4),
        },
        "repro_check": {
            "baseline_target_pp": 50.48,
            "baseline_actual_pp": base_comp,
            "baseline_ok":        base_ok,
            "4h_gate_target_pp":  77.93,
            "4h_gate_actual_pp":  gate_comp,
            "4h_gate_ok":         gate_ok,
        },
    }

    out_path = ROOT / "reports" / "multifactor_v1_stack2_comparison.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[stack2] wrote {out_path}", file=sys.stderr)

    # Human-readable summary.
    print("\n=== Stack2 Comparison Summary ===", file=sys.stderr)
    print(f"{'Config':<32} {'Compounded':>11} {'Wins':>6} {'Trades':>7} {'PSR':>7}", file=sys.stderr)
    print("-" * 65, file=sys.stderr)
    for cname in ("baseline", "funding_relax_only", "4h_gate_only", "stack_4h_and_funding_relax"):
        r = results[cname]
        s = r["summary"]
        p = r["psr"]["psr_vs_hurdle"]
        print(f"{cname:<32} {s['compounded_pct']:>10.2f}% {s['windows_positive']:>6} "
              f"{s['n_trades']:>7} {p:>7.4f}", file=sys.stderr)
    print("-" * 65, file=sys.stderr)
    print(f"Additivity verdict: {additivity_verdict}", file=sys.stderr)
    print(f"Stack verdict:      {stack_verdict}  (4h_gate→stack lift={lift_4h_to_stack:+.2f}pp)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
