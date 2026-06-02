"""AdaptiveTrendV1 + h1_confirmation ablation runner (post-fractional-refactor).

Compares AdaptiveTrendV1 (base, alpha=2.0) vs AdaptiveTrendV1_h1_confirmation
(same alpha, plus H1 EMA50 MTF entry gate: slope>=0 AND price>EMA for longs;
mirror for shorts) on the 5 OOS windows at $1M cash with PRICE_SCALE=0.001
fractional sizing.

Promotion bar:
  - Compounded equity must improve by >= +5pp over base (45.52%).
  - PSR must NOT decrease vs base (0.905).
  - Trade count likely DROPS (gate filters); fine if PSR + return improve.

Mirrors tools/_postfrac_adaptrend_v1_adx.py exactly except:
  - Test strategy class swapped to AdaptiveTrendV1_h1_confirmation.
  - H1 parquet is pre-scaled by PRICE_SCALE to a temp path so the
    EMA50 lives on the same scale as the (already-scaled) 15m close.

Writes:
  - reports/postfrac_adaptrend_v1_h1_conf.json
  - reports/_postfrac_adaptrend_v1_h1_conf_<window>.csv per OOS window
  - reports/_postfrac_adaptrend_v1_h1_conf_aggregated.csv
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
from strategy.signals_adaptive_trend_v1_h1_confirmation import (  # noqa: E402
    AdaptiveTrendV1_h1_confirmation,
)
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET_15M = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
PARQUET_1H_SRC = ROOT / "data" / "historical" / "BTC_USDT_USDT_1h.parquet"
PARQUET_1H_SCALED = ROOT / "reports" / "_tmp" / "BTC_USDT_USDT_1h_scaled_0.001.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

BASE_CONFIG = {"alpha": 2.0}
H1_CONFIG = {
    "alpha":             2.0,
    "use_h1_confirmation": True,
    "h1_ema_period":     50,
    "h1_slope_lookback": 10,
    "h1_parquet_path":   str(PARQUET_1H_SCALED),
}

WINDOWS_5 = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]


def _ensure_h1_scaled() -> None:
    """One-time pre-scale of the H1 parquet to match PRICE_SCALE.

    The fractional-sizing harness scales 15m OHLC by PRICE_SCALE so that
    backtesting.py's integer unit count maps to 0.001 BTC. Any auxiliary
    parquet read INSIDE a strategy (4H EMA in multifactor, H1 EMA here)
    must be scaled by the SAME factor or the indicator lives on the wrong
    scale and the price-vs-EMA comparison is meaningless.

    Idempotent: rewrites only when source is newer than dest.
    """
    PARQUET_1H_SCALED.parent.mkdir(parents=True, exist_ok=True)
    if (
        PARQUET_1H_SCALED.exists()
        and PARQUET_1H_SCALED.stat().st_mtime >= PARQUET_1H_SRC.stat().st_mtime
    ):
        return
    df = pd.read_parquet(PARQUET_1H_SRC)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col] * PRICE_SCALE
    df.to_parquet(PARQUET_1H_SCALED)
    print(
        f"[h1_conf] pre-scaled H1 parquet -> {PARQUET_1H_SCALED.name} "
        f"({len(df):,} bars)", file=sys.stderr,
    )


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame | None:
    df = pd.read_parquet(PARQUET_15M)
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
    print(f"[h1_conf_ablation] running arm={arm_label} ({len(WINDOWS_5)} windows) ...",
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

    return {
        "arm":        arm_label,
        "config":     config,
        "per_window": [{k: v for k, v in r.items() if k != "pnl_pct"} for r in per_window],
        "summary":    agg,
        "psr":        psr,
    }


def verdict(base: dict, test: dict) -> dict:
    base_comp = base["summary"]["compounded_pct"]
    test_comp = test["summary"]["compounded_pct"]
    base_psr = base["psr"]["psr_vs_hurdle"]
    test_psr = test["psr"]["psr_vs_hurdle"]
    base_sharpe = base["psr"].get("point_sharpe")
    test_sharpe = test["psr"].get("point_sharpe")
    base_trades = base["summary"]["n_trades"]
    test_trades = test["summary"]["n_trades"]

    delta_comp = test_comp - base_comp
    delta_psr = test_psr - base_psr

    cleared_compounded = delta_comp >= 5.0
    psr_not_worse = test_psr >= base_psr - 1e-6

    # Adversarial check: if trade counts match exactly the gate did nothing.
    gate_inert = (test_trades == base_trades)

    if gate_inert:
        decision = "INVESTIGATE — trade counts match base; gate may be inert"
    elif cleared_compounded and psr_not_worse:
        decision = "PROMOTE_CANDIDATE — extend to walk-forward 2020-2026"
    elif cleared_compounded and not psr_not_worse:
        decision = "ITERATE — compounded improves but PSR drops; risk-adj edge unclear"
    elif not cleared_compounded and psr_not_worse:
        decision = "ITERATE — PSR holds but compounded fails +5pp bar"
    else:
        decision = "SHELF — fails both bars"

    return {
        "base_compounded_pct":    base_comp,
        "test_compounded_pct":    test_comp,
        "delta_compounded_pp":    round(delta_comp, 4),
        "base_psr":               base_psr,
        "test_psr":               test_psr,
        "delta_psr":              round(delta_psr, 4),
        "base_point_sharpe":      base_sharpe,
        "test_point_sharpe":      test_sharpe,
        "base_trades":            base_trades,
        "test_trades":            test_trades,
        "gate_inert":             gate_inert,
        "cleared_+5pp_bar":       cleared_compounded,
        "psr_not_worse":          psr_not_worse,
        "decision":               decision,
    }


def main() -> int:
    t0 = time.time()
    _ensure_h1_scaled()

    res_base = run_arm(
        "base",
        AdaptiveTrendV1,
        BASE_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_h1_conf_baseRECHECK",
    )
    res_test = run_arm(
        "h1_confirmation",
        AdaptiveTrendV1_h1_confirmation,
        H1_CONFIG,
        csv_prefix="_postfrac_adaptrend_v1_h1_conf",
    )

    v = verdict(res_base, res_test)

    result = {
        "experiment":       "adaptrend_v1_h1_confirmation",
        "base_strategy":    "strategy.signals_adaptive_trend:AdaptiveTrendV1",
        "test_strategy":    "strategy.signals_adaptive_trend_v1_h1_confirmation:AdaptiveTrendV1_h1_confirmation",
        "gate_logic":       (
            "long: H1 EMA50 slope(10) >= 0 AND close > EMA50; "
            "short: mirror"
        ),
        "cash":             CASH,
        "commission":       COMMISSION,
        "margin":           MARGIN,
        "price_scale":      PRICE_SCALE,
        "h1_parquet":       str(PARQUET_1H_SCALED),
        "windows":          [w[0] for w in WINDOWS_5],
        "base":             res_base,
        "test":             res_test,
        "verdict":          v,
        "reference_postfrac_base": {
            "source": "reports/postfrac_adaptrend_v1.json (set_5_OOS)",
            "compounded_pct": 45.5222,
            "n_trades": 255,
            "psr_vs_hurdle": 0.905331,
            "point_sharpe": 0.076677,
            "note": "Re-running base here as sanity check; numbers should match exactly.",
        },
        "elapsed_sec":      round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_adaptrend_v1_h1_conf.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[h1_conf_ablation] verdict={v['decision']}  "
        f"base={v['base_compounded_pct']:+.2f}% -> test={v['test_compounded_pct']:+.2f}% "
        f"(delta {v['delta_compounded_pp']:+.2f}pp)  "
        f"base_PSR={v['base_psr']:.3f} -> test_PSR={v['test_psr']:.3f}",
        file=sys.stderr,
    )
    print(f"[h1_conf_ablation] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
