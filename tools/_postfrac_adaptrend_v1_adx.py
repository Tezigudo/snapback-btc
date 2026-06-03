"""AdaptiveTrendV1 + regime_gate_adx ablation runner (post-fractional-refactor).

Compares AdaptiveTrendV1 (base, alpha=2.0) vs AdaptiveTrendV1_regime_gate_adx
(same alpha, plus H6 ADX(14) > 25 entry gate) on the 5 OOS windows at
$1M cash with PRICE_SCALE=0.001 fractional sizing.

Promotion bar (from handoff):
  - Compounded equity must improve by >= +5pp over base.
  - PSR must NOT decrease vs base.

Writes:
  - reports/postfrac_adaptrend_v1_adx_gate.json
  - reports/_postfrac_adaptrend_v1_adx_<window>.csv per OOS window
  - reports/_postfrac_adaptrend_v1_adx_aggregated.csv

PRICE_SCALE pattern mirrors tools/_postfrac_adaptrend_v1.py (base) — apples
to apples. NO funding net applied (matches the base's gross-of-funding
basis), so the delta vs base is purely the gate effect.
"""
from __future__ import annotations

import json
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
from strategy.signals_adaptive_trend_v1_regime_gate_adx import (  # noqa: E402
    AdaptiveTrendV1_regime_gate_adx,
)
from tools.psr_eval import compute_psr  # noqa: E402
from tools.aggregate import build_canonical_block  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

# Hold alpha=2.0 to match the postfrac base run (see _postfrac_adaptrend_v1.py
# header for the reasoning). The ADX gate is the ONLY variable changing here.
BASE_CONFIG = {"alpha": 2.0}
ADX_CONFIG = {"alpha": 2.0, "adx_threshold": 25.0, "adx_period_h6": 14}

WINDOWS_5 = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame | None:
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start_ts = pd.Timestamp(start)
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
    df = _load_slice_scaled(start, end)
    if df is None:
        print(f"  [{label}] SKIP - no data", file=sys.stderr)
        return None
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

    n_trades = int(stats.get("# Trades", 0) or 0)
    ret_pct = float(stats.get("Return [%]", 0.0) or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)
    win_rate = float(stats.get("Win Rate [%]") or 0.0)
    equity_final = float(stats.get("Equity Final [$]", CASH) or CASH)

    trades_df = getattr(stats, "_trades", None)
    pnl_pct_list: list[float] = []
    if (
        trades_df is not None
        and len(trades_df) > 0
        and "ReturnPct" in trades_df.columns
    ):
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        if csv_prefix:
            out_csv = ROOT / "reports" / f"{csv_prefix}_{label}.csv"
            out = pd.DataFrame(
                {
                    "pnl_pct": pnl_pct_list,
                    "window_start": start,
                    "window_end": end,
                }
            )
            out.to_csv(out_csv, index=False)

    return {
        "label":        label,
        "start":        start,
        "end":          end,
        "trades":       n_trades,
        "return_pct":   round(ret_pct, 4),
        "max_dd_pct":   round(max_dd, 4),
        "win_rate_pct": round(win_rate, 4),
        "equity_final": round(equity_final, 4),
        "pnl_pct":      pnl_pct_list,
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
    print(f"[adx_ablation] running arm={arm_label} ({len(WINDOWS_5)} windows) ...",
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

    # --- Canonical equity-curve aggregation (methodology debt #1) -----------
    # Headline PSR is now psr_walkforward: compute_psr on the n-window
    # equity-curve Return[%] series (sizing-aware), NOT the stitched per-trade
    # ReturnPct union (sizing-blind, N-inflated). build_canonical_block also
    # dual-emits legacy_psr_stitched for backcompat observability.
    canon = build_canonical_block(per_window)
    psr = canon["psr_walkforward"]

    # Legacy stitched-per-trade PSR — diagnostic only (what this runner emitted
    # pre-migration). N-inflated; never the verdict input.
    pnl_arr = np.asarray(all_pnl, dtype=float)
    legacy_psr_stitched = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(pnl_arr) >= 2
        else {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    if csv_prefix:
        agg_csv = ROOT / "reports" / f"{csv_prefix}_aggregated.csv"
        pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
        print(f"  [{arm_label}] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    return {
        "arm":        arm_label,
        "config":     config,
        "per_window": [{k: v for k, v in r.items() if k != "pnl_pct"} for r in per_window],
        "summary":    agg,
        "psr":                 psr,                  # canonical (psr_walkforward)
        "canonical":           canon,                # full canonical block
        "legacy_psr_stitched": legacy_psr_stitched,  # diagnostic observability
    }


def verdict(base: dict, adx: dict) -> dict:
    base_comp = base["summary"]["compounded_pct"]
    adx_comp = adx["summary"]["compounded_pct"]
    base_psr = base["psr"]["psr_vs_hurdle"]
    adx_psr = adx["psr"]["psr_vs_hurdle"]
    base_sharpe = base["psr"].get("point_sharpe")
    adx_sharpe = adx["psr"].get("point_sharpe")
    base_trades = base["summary"]["n_trades"]
    adx_trades = adx["summary"]["n_trades"]

    delta_comp = adx_comp - base_comp
    delta_psr = adx_psr - base_psr

    cleared_compounded = delta_comp >= 5.0
    psr_not_worse = adx_psr >= base_psr - 1e-6

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
        "adx_compounded_pct":     adx_comp,
        "delta_compounded_pp":    round(delta_comp, 4),
        "base_psr":               base_psr,
        "adx_psr":                adx_psr,
        "delta_psr":              round(delta_psr, 4),
        "base_point_sharpe":      base_sharpe,
        "adx_point_sharpe":       adx_sharpe,
        "base_trades":            base_trades,
        "adx_trades":             adx_trades,
        "cleared_+5pp_bar":       cleared_compounded,
        "psr_not_worse":          psr_not_worse,
        "decision":               decision,
    }


def _arm_with_psr(arm: dict, psr_block: dict) -> dict:
    """Shallow copy of an arm result with its `psr` swapped (verdict re-eval)."""
    a = dict(arm)
    a["psr"] = psr_block
    return a


def _migration_selfcheck(base: dict, test: dict, canonical_verdict: dict, verdict_fn) -> dict:
    """Assert canonical PSR is reproducible bit-for-bit + flag verdict flips.

    (a) matches_headline: compute_psr on the persisted per-window Return[%]
        array (canonical['per_window_return_pct'], contiguous=False to match
        aggregate._safe_compute_psr) must equal each arm's headline
        psr_walkforward exactly.
    (b) verdict_changed: re-run verdict() feeding the OLD stitched PSR; if the
        decision differs from the canonical-PSR decision, the migration moved a
        verdict and MUST be surfaced loudly.
    """
    def _check_arm(arm: dict) -> bool:
        canon = arm["canonical"]
        arr = np.asarray(canon["per_window_return_pct"], dtype=float)
        recomputed = (
            compute_psr(arr, sr_hurdle=0.0, confidence=0.95, contiguous=False)
            if len(arr) >= 2
            else {"n_trades": int(len(arr)), "psr_vs_hurdle": 0.0,
                  "psr_lo_adjusted": 0.0, "interpretation": "insufficient_evidence"}
        )
        return recomputed == canon["psr_walkforward"]

    matches_headline = _check_arm(base) and _check_arm(test)

    # Legacy decision: feed the pre-migration stitched PSR back through verdict().
    legacy_v = verdict_fn(
        _arm_with_psr(base, base["legacy_psr_stitched"]),
        _arm_with_psr(test, test["legacy_psr_stitched"]),
    )
    legacy_decision = legacy_v["decision"]
    canonical_decision = canonical_verdict["decision"]

    return {
        "matches_headline":   bool(matches_headline),
        "verdict_changed":    bool(legacy_decision != canonical_decision),
        "canonical_decision": canonical_decision,
        "legacy_decision":    legacy_decision,
        "canonical_psr_base": base["psr"]["psr_vs_hurdle"],
        "canonical_psr_test": test["psr"]["psr_vs_hurdle"],
        "legacy_psr_base":    base["legacy_psr_stitched"]["psr_vs_hurdle"],
        "legacy_psr_test":    test["legacy_psr_stitched"]["psr_vs_hurdle"],
    }


def main() -> int:
    t0 = time.time()

    res_base = run_arm(
        "base",
        AdaptiveTrendV1,
        BASE_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_adx_baseRECHECK",
    )
    res_adx = run_arm(
        "adx",
        AdaptiveTrendV1_regime_gate_adx,
        ADX_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_adx",
    )

    v = verdict(res_base, res_adx)

    # --- Migration self-checks (methodology debt #1) ------------------------
    # (a) bit-for-bit: compute_psr on the persisted per-window Return[%] array
    #     must equal the headline psr_walkforward each arm reports.
    # (b) verdict-flip guard: re-run verdict() feeding the OLD stitched PSR and
    #     confirm the migration does NOT change the SHELF/headline decision.
    migration = _migration_selfcheck(res_base, res_adx, v, verdict)
    print(
        f"[adx_ablation] migration matches_headline={migration['matches_headline']} "
        f"verdict_changed={migration['verdict_changed']}",
        file=sys.stderr,
    )
    if migration["verdict_changed"]:
        print(
            "[adx_ablation] !!! VERDICT CHANGED under canonical PSR — "
            f"legacy={migration['legacy_decision']!r} -> "
            f"canonical={migration['canonical_decision']!r}. SURFACE TO USER.",
            file=sys.stderr,
        )

    result = {
        "experiment":     "adaptrend_v1_regime_gate_adx",
        "base_strategy":  "strategy.signals_adaptive_trend:AdaptiveTrendV1",
        "adx_strategy":   "strategy.signals_adaptive_trend_v1_regime_gate_adx:AdaptiveTrendV1_regime_gate_adx",
        "cash":           CASH,
        "commission":     COMMISSION,
        "margin":         MARGIN,
        "price_scale":    PRICE_SCALE,
        "windows":        [w[0] for w in WINDOWS_5],
        "base":           res_base,
        "adx":            res_adx,
        "verdict":        v,
        "migration":      migration,
        "reference_postfrac_base": {
            "source": "reports/postfrac_adaptrend_v1.json (set_5_OOS)",
            "compounded_pct": 45.5222,
            "n_trades": 255,
            "psr_vs_hurdle": 0.905331,
            "point_sharpe": 0.076677,
            "note": "Re-running base here as sanity check; numbers should match exactly.",
        },
        "elapsed_sec":    round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_adaptrend_v1_adx_gate.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[adx_ablation] verdict={v['decision']}  "
        f"base={v['base_compounded_pct']:+.2f}% -> adx={v['adx_compounded_pct']:+.2f}% "
        f"(delta {v['delta_compounded_pp']:+.2f}pp)  "
        f"base_PSR={v['base_psr']:.3f} -> adx_PSR={v['adx_psr']:.3f}",
        file=sys.stderr,
    )
    print(f"[adx_ablation] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
