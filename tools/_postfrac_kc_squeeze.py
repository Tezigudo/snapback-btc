"""KCSqueezeBreakout (1H) — 5 OOS validation under fractional sizing.

Mirrors the structure of tools/_postfrac_adaptrend_v1.py (5 OOS sliced from
the 15m parquet) and adopts the warm-prefix pattern from
tools/_postfrac_adaptrend_v1_rv_band.py so the 1H BB/KC/ATR/linreg
indicators have enough warm-up at each window's start.

Configuration (PINNED — do not deviate):
  - timeframe inside strategy: 1H (15m parquet → resampled in strategy.init)
  - cash:       $1,000,000
  - commission: 0.00075 (7.5 bps per side ≈ 15 bps round-trip)
  - margin:     1/leverage = 1/5
  - PRICE_SCALE: 0.001 applied to OHLC at slice time
  - 5 OOS:      2022_H1, 2023_H1, 2024_H1, 2024_H2, 2025_H1
  - Warm prefix: WARM_PREFIX_DAYS = 395 (same as rv_band runner; gives the
    1H linreg + BB + KC + Wilder ATR ample head-room. Conservative.)

Output: reports/kc_squeeze_5oos.json with:
  - per-window:   return_pct, sharpe, n_trades, max_dd, win_rate
  - aggregate:    compounded, mean_return, std_return, psr (Bailey/Lopez de
                  Prado on the per-trade pnl_pct stream)

Trade attribution: trades with EntryTime < window_start (i.e. trades that
opened during the warm-prefix lead-in) are EXCLUDED from per-window stats
so we count strictly OOS-portion trades. The equity-curve drawdown is also
restricted to the OOS slice.

This is the FIRST OOS validation of the pinned KC squeeze candidate.
Compare against gates baked into TODO_LEG / task spec:
  - PSR >= 0.97
  - >= 30 trades total across 5 OOS, >= 3 trades per window
  - lift vs donchian-v3 baseline >= 3pp compounded (NOT checked here; the
    follow-on script compares against donchian-v3 numbers)
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

from strategy.signals_kc_squeeze import KCSqueezeBreakout  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.00075          # 7.5 bps per side  →  15 bps round-trip
LEVERAGE = 5                  # pinned
MARGIN = 1.0 / LEVERAGE
PRICE_SCALE = 0.001

# Warm prefix days. Chosen the same as the rv_band runner (395) which is
# overkill for BB/KC/ATR/linreg(20 1H bars) but ensures any rolling
# indicator added later still warms cleanly. Cost is just a few seconds
# per window.
WARM_PREFIX_DAYS = 395

# Strategy config (pinned spec). These match defaults inside
# KCSqueezeBreakout but pass them explicitly so the JSON record captures
# exactly what was run.
CONFIG = {
    "bb_period":                 20,
    "bb_k":                      2.0,
    "kc_period":                 20,
    "kc_atr_mult":               1.5,
    "squeeze_min_bars":          10,
    "lr_slope_window":           20,
    "volume_mult":               1.5,
    "volume_ma_period":          20,
    "atr_period":                14,
    "atr_sl_mult":               1.5,
    "atr_trail_mult":            2.0,
    "min_move_to_arm_trail_atr": 1.0,
    "risk_per_trade_pct":        0.5,
    "leverage":                  LEVERAGE,
    "allow_shorts":              True,
    "exit_on_resqueeze":         True,
}

WINDOWS_5 = [
    ("2022_H1", "2022-01-01", "2022-07-01"),
    ("2023_H1", "2023-01-01", "2023-07-01"),
    ("2024_H1", "2024-01-01", "2024-07-01"),
    ("2024_H2", "2024-07-01", "2025-01-01"),
    ("2025_H1", "2025-01-01", "2025-07-01"),
]


def _load_slice_scaled(start: str, end: str, warm_days: int = 0) -> pd.DataFrame | None:
    """Load BTC 15m bars between [start - warm_days, end), price-scaled.

    `start`/`end` follow the half-open convention specified in the pinned
    windows (e.g. 2022-01-01 / 2022-07-01). The strategy resamples this
    15m frame to 1H internally.
    """
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start_ts = pd.Timestamp(start) - pd.Timedelta(days=warm_days)
    end_ts = pd.Timestamp(end) - pd.Timedelta(nanoseconds=1)
    sl = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sl) == 0:
        return None

    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * PRICE_SCALE
    return sl


def run_window(label: str, start: str, end: str) -> dict | None:
    df = _load_slice_scaled(start, end, warm_days=WARM_PREFIX_DAYS)
    if df is None:
        print(f"  [{label}] SKIP — no data in window", file=sys.stderr)
        return None

    window_start_ts = pd.Timestamp(start)

    bt = Backtest(
        df,
        KCSqueezeBreakout,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(**CONFIG)

    # --- OOS-only trade attribution --------------------------------------
    trades_df = getattr(stats, "_trades", None)
    if trades_df is None or len(trades_df) == 0:
        oos_trades = pd.DataFrame(columns=["EntryTime", "ExitTime", "ReturnPct"])
    else:
        if "EntryTime" in trades_df.columns:
            oos_trades = trades_df[trades_df["EntryTime"] >= window_start_ts].copy()
        else:
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
        # LEGACY (v1)
        compounded_factor = float(np.prod(1.0 + ret_pct_series))
        legacy_compounded_oos = compounded_factor - 1.0
        equity_final = CASH * compounded_factor
        n_wins = int((ret_pct_series > 0).sum())
        win_rate = 100.0 * n_wins / n_trades

        # CANONICAL (v2): equity-impact returns, sizing-aware
        from tools.aggregate import equity_impact_returns as _eir
        stub = type("S", (), {"_trades": oos_trades})()
        eq_impact_pnl_pct = _eir(stub, cash=CASH).tolist()

    # v2 headline = compounded equity-impact (sizing-aware)
    if eq_impact_pnl_pct:
        c = 1.0
        for r in eq_impact_pnl_pct:
            c *= 1.0 + r / 100.0
        ret_pct = (c - 1.0) * 100.0
    else:
        ret_pct = 0.0
    legacy_ret_pct = legacy_compounded_oos * 100.0

    # --- OOS max drawdown -----------------------------------------------
    eq_curve = getattr(stats, "_equity_curve", None)
    max_dd = 0.0
    if eq_curve is not None and len(eq_curve) > 0 and "Equity" in eq_curve.columns:
        eq_slice = eq_curve.loc[eq_curve.index >= window_start_ts, "Equity"]
        if len(eq_slice) > 1:
            running_max = eq_slice.cummax()
            dd_series = (eq_slice / running_max - 1.0) * 100.0
            max_dd = float(dd_series.min())

    # --- per-trade sharpe (point estimate; per-trade pnl, NOT annualised) ---
    # The aggregate PSR uses the same trade stream; this per-window number is
    # just a sanity signal alongside the headline return.
    sharpe = 0.0
    if n_trades >= 2:
        arr = np.asarray(pnl_pct_list, dtype=float) / 100.0
        s = float(arr.std(ddof=1))
        if s > 0:
            sharpe = float(arr.mean()) / s

    out_csv = ROOT / "reports" / f"_postfrac_kc_squeeze_{label}.csv"
    if n_trades > 0:
        pd.DataFrame(
            {
                "pnl_pct":      pnl_pct_list,
                "window_start": start,
                "window_end":   end,
            }
        ).to_csv(out_csv, index=False)

    return {
        "label":             label,
        "start":             start,
        "end":               end,
        "trades":            n_trades,
        "return_pct":        round(ret_pct, 4),
        "legacy_return_pct": round(legacy_ret_pct, 4),
        "sharpe":            round(sharpe, 6),
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
    per_win_ret: list[float] = []
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        if r["return_pct"] > 0:
            n_pos += 1

    # mean / std of per-window returns
    arr = np.asarray(per_win_ret, dtype=float)
    mean_return = float(arr.mean()) if len(arr) else 0.0
    std_return = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0

    return {
        "n_trades":              n_trades,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "mean_return_pct":       round(mean_return, 4),
        "std_return_pct":        round(std_return, 4),
        "windows_positive":      f"{n_pos}/{len(per_window)}",
        "per_window_return_pct": per_win_ret,
        "all_pnl_pct":           all_pnl,
    }


def main() -> int:
    t0 = time.time()
    per_window: list[dict] = []
    print(f"[postfrac_kc_squeeze] running {len(WINDOWS_5)} OOS windows "
          f"(warm_prefix={WARM_PREFIX_DAYS}d) ...", file=sys.stderr)

    for label, start, end in WINDOWS_5:
        tw = time.time()
        r = run_window(label, start, end)
        if r is None:
            continue
        print(
            f"  {label}  trades={r['trades']:4d}  ret={r['return_pct']:+8.2f}%  "
            f"sharpe={r['sharpe']:+.3f}  dd={r['max_dd_pct']:+7.2f}%  "
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
        else {
            "n_trades": int(len(pnl_arr)),
            "psr_vs_hurdle": 0.0,
            "interpretation": "insufficient_evidence",
        }
    )

    agg_csv = ROOT / "reports" / "_postfrac_kc_squeeze_aggregated.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
    print(f"[postfrac_kc_squeeze] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    # Headline / gate PSR migrated from the N-inflated stitched per-trade
    # ReturnPct union to the equity-curve window-level aggregation
    # (psr_walkforward). Stitched PSR kept as observability only.
    from tools.aggregate import build_canonical_block, AGGREGATION_VERSION
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)
    psr = canon["psr_walkforward"]  # canonical headline PSR

    # --- bit-for-bit round-trip check (migration verification) --------------
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
        f"[postfrac_kc_squeeze] canonical PSR round-trip OK: "
        f"{psr.get('psr_vs_hurdle')}",
        file=sys.stderr,
    )

    # Pinned task gates — pure reporting, not "verdict" arms (the workflow
    # decides the verdict from the JSON downstream). PSR gate now reads the
    # CANONICAL psr_walkforward (was stitched per-trade pre-migration).
    gate_checks = {
        "psr_basis":           "canonical_psr_walkforward",
        "psr_min":             0.97,
        "psr_actual":          float(psr.get("psr_vs_hurdle", 0.0)),
        "psr_cleared":         float(psr.get("psr_vs_hurdle", 0.0)) >= 0.97,
        "psr_stitched_legacy": float(legacy_psr_stitched.get("psr_vs_hurdle", 0.0)),
        "min_trades_total":    30,
        "trades_actual":       int(agg["n_trades"]),
        "trades_cleared":      int(agg["n_trades"]) >= 30,
        "min_trades_per_window": 3,
        "per_window_trade_counts": [r["trades"] for r in per_window],
        "per_window_trades_cleared": all(r["trades"] >= 3 for r in per_window),
        "worst_window_dd_max_abs_pct": 35.5,
        "worst_window_dd_actual_pct": (
            min((r["max_dd_pct"] for r in per_window), default=0.0)
            if per_window else 0.0
        ),
        "worst_window_dd_cleared": (
            min((r["max_dd_pct"] for r in per_window), default=0.0) >= -35.5
            if per_window else True
        ),
    }

    result = {
        "strategy_id":     "kc_squeeze_v0",
        "strategy_class":  "strategy.signals_kc_squeeze:KCSqueezeBreakout",
        "timeframe":       "1h (resampled in-strategy from 15m parquet)",
        "cash":            CASH,
        "commission":      COMMISSION,
        "commission_note": "7.5 bps per side  →  15 bps round-trip",
        "margin":          MARGIN,
        "leverage":        LEVERAGE,
        "price_scale":     PRICE_SCALE,
        "warm_prefix_days": WARM_PREFIX_DAYS,
        "config":          CONFIG,
        "windows":         [w[0] for w in WINDOWS_5],
        "per_window":      [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":         agg,
        "psr":             psr,                    # canonical psr_walkforward
        "legacy_psr_stitched": legacy_psr_stitched,  # observability only
        "canonical":       canon,                  # v2 dual-emit
        "aggregation_method": canon["aggregation_method"],
        "gates":           gate_checks,
        "notes": (
            "Warm-prefix harness: each window's slice starts "
            f"{WARM_PREFIX_DAYS}d before window_start so 1H BB/KC/ATR/linreg "
            "indicators are warmed up by the time the OOS window begins. "
            "Trades, win-rate, compounded return, sharpe, max-DD, and per-window "
            "CSVs are computed strictly on trades whose EntryTime >= window_start. "
            "Commission = 15 bps round-trip (7.5 bps/side). PRICE_SCALE=0.001 "
            "gives fractional 0.001 BTC sizing under the integer-units harness."
        ),
        "elapsed_sec":     round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "kc_squeeze_5oos.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[postfrac_kc_squeeze] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
