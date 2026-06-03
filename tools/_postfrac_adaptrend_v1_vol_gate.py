"""AdaptiveTrendV1 + regime_gate_vol ablation runner (post-fractional-refactor).

Compares AdaptiveTrendV1 (base, alpha=2.0) vs AdaptiveTrendV1_regime_gate_vol
(same alpha, plus ATR/Close > 60th-pct of trailing 60-day distribution entry
gate) on the 5 OOS windows at $1M cash with PRICE_SCALE=0.001 fractional
sizing.

Promotion bar (from handoff):
  - Compounded equity must improve by >= +5pp over base (target >= +50.52%).
  - PSR must NOT decrease vs base (0.905).

Writes:
  - reports/postfrac_adaptrend_v1_vol_gate.json
  - reports/_postfrac_adaptrend_v1_regime_gate_vol_<window>.csv  (test arm)
  - reports/_postfrac_adaptrend_v1_regime_gate_vol_baseRECHECK_<window>.csv
  - reports/_postfrac_adaptrend_v1_regime_gate_vol_aggregated.csv
  - reports/_postfrac_adaptrend_v1_regime_gate_vol_baseRECHECK_aggregated.csv

PRICE_SCALE pattern mirrors tools/_postfrac_adaptrend_v1_adx.py — apples to
apples.  NO funding net applied (matches the base's gross-of-funding basis),
so the delta vs base is purely the gate effect.

Warmup
------
We do NOT prepend a historical prefix (matches the ADX-gate runner).  As a
consequence the vol-gate's first ~60 days of each OOS window are
NaN-blocked: the rolling-240-bar quantile cannot be computed until 60 days
of H6 history have accumulated INSIDE the window.  That's an honest
representation of a "cold start" — same behaviour the gate would have on
its first 60 days of live deploy.  The base arm sees the full window and
should exactly reproduce the +45.52% / 255 trades / PSR 0.905 reference.

If the gate clears the +5pp bar despite the cold-start handicap, that's
strong evidence; we can then re-test with a warm-prefix harness to
quantify the additional uplift the production deploy would see.
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
from strategy.signals_adaptive_trend_v1_regime_gate_vol import (  # noqa: E402
    AdaptiveTrendV1_regime_gate_vol,
)
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
    equity_impact_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

# Hold alpha=2.0 to match the postfrac base run (see _postfrac_adaptrend_v1.py
# header for the reasoning).  The vol gate is the ONLY variable changing here.
BASE_CONFIG = {"alpha": 2.0}
VOL_CONFIG = {
    "alpha": 2.0,
    "vol_lookback_days": 60,
    "vol_quantile": 0.60,
    "vol_atr_period_h6": 14,
}

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
    eq_impact_pnl_pct: list[float] = []
    if (
        trades_df is not None
        and len(trades_df) > 0
        and "ReturnPct" in trades_df.columns
    ):
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
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
        "label":             label,
        "start":             start,
        "end":               end,
        "trades":            n_trades,
        "return_pct":        round(ret_pct, 4),
        "max_dd_pct":        round(max_dd, 4),
        "win_rate_pct":      round(win_rate, 4),
        "equity_final":      round(equity_final, 4),
        "pnl_pct":           pnl_pct_list,
        "eq_impact_pnl_pct": eq_impact_pnl_pct,
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
    print(f"[vol_ablation] running arm={arm_label} ({len(WINDOWS_5)} windows) ...",
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
    psr = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(pnl_arr) >= 2
        else {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    if csv_prefix:
        agg_csv = ROOT / "reports" / f"{csv_prefix}_aggregated.csv"
        pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
        print(f"  [{arm_label}] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    # Canonical v2 dual-emit (methodology debt #1): headline PSR comes from the
    # equity-curve aggregation (per-window return-series -> psr_walkforward),
    # NOT the N-inflated stitched per-trade ReturnPct union. Legacy stitched PSR
    # is kept as `psr` + inside canon["legacy_psr_stitched"] for observability.
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)

    return {
        "arm":        arm_label,
        "config":     config,
        "per_window": [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":    agg,
        "psr":        psr,            # legacy stitched (observability only)
        "canonical":  canon,          # v2 dual-emit (headline PSR = psr_walkforward)
        "aggregation_method": canon["aggregation_method"],
    }


def verdict(base: dict, vol: dict) -> dict:
    base_comp = base["summary"]["compounded_pct"]
    vol_comp = vol["summary"]["compounded_pct"]
    # Canonical headline PSR = equity-curve aggregation's psr_walkforward
    # (compute_psr on the n-window return series). The stitched per-trade
    # PSR is N-inflated and MUST NOT be the primary verdict input
    # (tools/aggregate.py docstring lines 33-35). Stitched values are still
    # emitted below as *_psr_stitched_legacy for observability/diff.
    base_psr = base["canonical"]["psr_walkforward"]["psr_vs_hurdle"]
    vol_psr = vol["canonical"]["psr_walkforward"]["psr_vs_hurdle"]
    base_psr_stitched = base["psr"]["psr_vs_hurdle"]
    vol_psr_stitched = vol["psr"]["psr_vs_hurdle"]
    base_sharpe = base["canonical"]["psr_walkforward"].get("point_sharpe")
    vol_sharpe = vol["canonical"]["psr_walkforward"].get("point_sharpe")
    base_trades = base["summary"]["n_trades"]
    vol_trades = vol["summary"]["n_trades"]

    delta_comp = vol_comp - base_comp
    delta_psr = vol_psr - base_psr
    delta_trades = vol_trades - base_trades

    cleared_compounded = delta_comp >= 5.0
    psr_not_worse = vol_psr >= base_psr - 1e-6

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
        "vol_compounded_pct":     vol_comp,
        "delta_compounded_pp":    round(delta_comp, 4),
        "psr_basis":              "canonical_psr_walkforward",
        "base_psr":               base_psr,
        "vol_psr":                vol_psr,
        "delta_psr":              round(delta_psr, 4),
        "base_psr_stitched_legacy": base_psr_stitched,
        "vol_psr_stitched_legacy":  vol_psr_stitched,
        "base_point_sharpe":      base_sharpe,
        "vol_point_sharpe":       vol_sharpe,
        "base_trades":            base_trades,
        "vol_trades":             vol_trades,
        "delta_trades":           delta_trades,
        "cleared_+5pp_bar":       cleared_compounded,
        "psr_not_worse":          psr_not_worse,
        "decision":               decision,
    }


def main() -> int:
    t0 = time.time()

    res_base = run_arm(
        "base",
        AdaptiveTrendV1,
        BASE_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_regime_gate_vol_baseRECHECK",
    )
    res_vol = run_arm(
        "vol",
        AdaptiveTrendV1_regime_gate_vol,
        VOL_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_regime_gate_vol",
    )

    v = verdict(res_base, res_vol)

    result = {
        "experiment":     "adaptrend_v1_regime_gate_vol",
        "base_strategy":  "strategy.signals_adaptive_trend:AdaptiveTrendV1",
        "vol_strategy":   "strategy.signals_adaptive_trend_v1_regime_gate_vol:AdaptiveTrendV1_regime_gate_vol",
        "cash":           CASH,
        "commission":     COMMISSION,
        "margin":         MARGIN,
        "price_scale":    PRICE_SCALE,
        "windows":        [w[0] for w in WINDOWS_5],
        "base":           res_base,
        "vol":            res_vol,
        "verdict":        v,
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

    out_path = ROOT / "reports" / "postfrac_adaptrend_v1_vol_gate.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[vol_ablation] verdict={v['decision']}  "
        f"base={v['base_compounded_pct']:+.2f}% -> vol={v['vol_compounded_pct']:+.2f}% "
        f"(delta {v['delta_compounded_pp']:+.2f}pp)  "
        f"base_PSR={v['base_psr']:.3f} -> vol_PSR={v['vol_psr']:.3f}",
        file=sys.stderr,
    )
    print(f"[vol_ablation] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
