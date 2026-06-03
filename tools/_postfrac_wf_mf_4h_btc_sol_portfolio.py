"""Quarterly walk-forward — multifactor-v1 4H gate, BTC+SOL 50/50 portfolio.

Best arm under audit:
    port_5050_noveto_v2  (w_btc=0.5, w_sol=0.5, veto=off)

Rolling 6mo-train / 3mo-test windows from 2021-01-01 to 2026-06-30 at $1M
cash with fractional sizing (PRICE_SCALE=0.001) and 15bps RT commission
(matches the 5-OOS baseline).

Per window:
  * Load BTC + SOL 15m slices [train_start..test_end] (warmup + test).
  * Run DayTradeMultiFactorBTC on each (locked multifactor-v1 config + 4H
    EMA200 gate; no cross-coin veto).
  * Filter trades whose EntryTime falls inside [test_start..test_end] —
    warmup trades are discarded.
  * Reconstruct truncated equity curves from the OOS trades and synthesize
    portfolio return / drawdown via _synth_portfolio_window (PSR-weighted
    pnl per allocation).

Output:
  reports/postfrac_wf_mf_4h_btc_sol_portfolio_port_5050_noveto_v2.json

Verdict:
  PASSES if pct_positive (return_pct > 0) among quarters_with_sufficient_trades
  (n_trades >= 5) is >= 70%.
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

from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402
from tools.portfolio_psr import (  # noqa: E402
    aggregate_portfolio_psr,
    equity_to_period_returns,
)
# Reuse the synth from the H1 portfolio harness to keep semantics identical.
from tools._postfrac_mf_4h_btc_sol_portfolio import (  # noqa: E402
    _synth_portfolio_window,
    _ensure_scaled_4h_parquet,
    LOCKED,
    PARQ_15m,
    PARQ_FUND,
    CASH,
    COMMISSION,
    MARGIN,
    PRICE_SCALE,
)

# ---------------------------------------------------------------------------
# Walk-forward grid
# ---------------------------------------------------------------------------

WF_START = pd.Timestamp("2021-01-01")
WF_END = pd.Timestamp("2026-06-30")

# Portfolio config (best arm).
ARM_TAG = "port_5050_noveto_v2"
W_BTC = 0.5
W_SOL = 0.5
VETO_ON = False

# Minimum trades for a quarter to count toward pct_positive_sufficient.
MIN_TRADES_FOR_SUFFICIENT = 5


def build_quarterly_windows() -> list[dict]:
    """3-month rolling test windows, with 6-month warmup prefix each."""
    out = []
    ts = WF_START
    while ts < WF_END:
        test_start = ts
        # Quarter end = ts + 3mo - 1ns
        test_end = min(
            ts + pd.DateOffset(months=3) - pd.Timedelta(nanoseconds=1),
            WF_END,
        )
        train_start = ts - pd.DateOffset(months=6)
        label = f"{test_start.strftime('%Y%m')}_3mo"
        out.append({
            "label":       label,
            "train_start": train_start,
            "test_start":  test_start,
            "test_end":    test_end,
        })
        ts = ts + pd.DateOffset(months=3)
    return out


def _load_slice_scaled_wf(coin: str, start: pd.Timestamp,
                          end: pd.Timestamp) -> pd.DataFrame | None:
    """15m slice loader (scaled OHLC + funding merge) for arbitrary timestamps."""
    df = pd.read_parquet(PARQ_15m[coin])
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    fund_path = PARQ_FUND.get(coin)
    if fund_path and fund_path.exists():
        fund = pd.read_parquet(fund_path)
        if fund.index.tz is not None:
            fund.index = fund.index.tz_localize(None)
        left = pd.DataFrame(index=df.index)
        right = pd.DataFrame({"Funding": fund["funding_rate"].values},
                             index=fund.index)
        merged = pd.merge_asof(left, right, left_index=True, right_index=True,
                               direction="backward")
        df["Funding"] = merged["Funding"].fillna(0.0).values

    sliced = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if len(sliced) == 0:
        return None

    for col in ("Open", "High", "Low", "Close"):
        if col in sliced.columns:
            sliced[col] = sliced[col] * PRICE_SCALE
    return sliced


def _run_coin_wf_window(coin: str, label: str, train_start: pd.Timestamp,
                        test_start: pd.Timestamp,
                        test_end: pd.Timestamp) -> dict | None:
    sl = _load_slice_scaled_wf(coin, train_start, test_end)
    if sl is None or len(sl) < 500:
        print(f"  {coin} {label}: SKIP (insufficient bars "
              f"{0 if sl is None else len(sl)})", file=sys.stderr)
        return None

    primary_4h = _ensure_scaled_4h_parquet(coin)
    config = {**LOCKED,
              "mtf_4h_parquet_path": str(primary_4h),
              "cross_4h_parquet_path": ""}  # veto OFF

    bt = Backtest(
        sl, DayTradeMultiFactorBTC,
        cash=CASH, commission=COMMISSION, margin=MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run(**config)
    trades_df = getattr(stats, "_trades", None)
    eq_curve = getattr(stats, "_equity_curve", None)

    # Extract the real bar-by-bar equity curve once (mirror H1 runner lines
    # 224-228). This is the TRUE continuous curve over [train_start..test_end];
    # it is clipped to the OOS test window below so its period grid is dense
    # daily (idle/flat days included) rather than sparse exit anchors.
    eq_series_full = None
    if eq_curve is not None and "Equity" in eq_curve.columns:
        eq_series_full = eq_curve["Equity"].copy()
        if eq_series_full.index.tz is not None:
            eq_series_full.index = eq_series_full.index.tz_localize(None)

    if trades_df is None or len(trades_df) == 0:
        return {
            "label":         label,
            "trades":        0,
            "return_pct":    0.0,
            "max_dd_pct":    0.0,
            "win_rate_pct":  0.0,
            "equity_final":  CASH,
            "pnl_pct":       [],
            "equity_series": None,
        }

    # Filter to OOS-entry trades.
    et = pd.to_datetime(trades_df["EntryTime"])
    mask = (et >= test_start) & (et <= test_end)
    oos = trades_df.loc[mask].copy()
    n_oos = int(len(oos))

    if n_oos == 0:
        return {
            "label":         label,
            "trades":        0,
            "return_pct":    0.0,
            "max_dd_pct":    0.0,
            "win_rate_pct":  0.0,
            "equity_final":  CASH,
            "pnl_pct":       [],
            "equity_series": None,
        }

    pnl_pct = (oos["ReturnPct"].values * 100.0).tolist()
    wins = int((oos["ReturnPct"] > 0).sum())
    wr_pct = (wins / n_oos) * 100.0

    # Compounded OOS return from sequential trades.
    c = 1.0
    for p in pnl_pct:
        c *= (1.0 + p / 100.0)
    ret_pct = (c - 1.0) * 100.0

    # Use the REAL bar-by-bar equity curve clipped to the OOS test window
    # instead of rebuilding from exit timestamps. The exit-anchored rebuild
    # produced a sparse ~one-obs-per-trade-exit grid; after a 1D resample
    # downstream that collapsed to ~15 obs/quarter, omitting the idle/flat
    # days where capital sits between trades (methodology debt #2 grid bug).
    #
    # CLIP is required here and is the one asymmetry vs the H1 runner: the WF
    # slice spans [train_start..test_end] (warmup + test), so the raw
    # _equity_curve carries warmup bars that must be dropped. The H1 slice is
    # the test window only, so H1 needs no clip. The clipped curve will not
    # start at CASH/1.0, but downstream build_portfolio_equity_curve ->
    # _normalized_equity divides by the first value (relative path only), and
    # pct_change is intra-window only (no spurious boundary return injected).
    if eq_series_full is not None:
        equity_series = eq_series_full.loc[
            (eq_series_full.index >= test_start)
            & (eq_series_full.index <= test_end)
        ].copy()
        if len(equity_series) == 0:
            equity_series = None
    else:
        equity_series = None

    if equity_series is not None and len(equity_series) > 0:
        # MaxDD on the dense bar-by-bar curve (deeper than the exit-anchored
        # curve since it now captures intra-window drawdowns — correctness
        # improvement, not a regression).
        port_eq = equity_series.values
        running_max = np.maximum.accumulate(port_eq)
        dd_series = (port_eq / running_max) - 1.0
        max_dd_pct = float(dd_series.min() * 100.0)
    else:
        max_dd_pct = 0.0

    return {
        "label":         label,
        "trades":        n_oos,
        "return_pct":    float(ret_pct),
        "max_dd_pct":    float(max_dd_pct),
        "win_rate_pct":  float(wr_pct),
        "equity_final":  float(CASH * c),
        "pnl_pct":       pnl_pct,
        "equity_series": equity_series,
    }


def main() -> int:
    t0 = time.time()
    print(f"[wf_port] arm={ARM_TAG}  w_btc={W_BTC} w_sol={W_SOL} veto_on={VETO_ON}",
          file=sys.stderr)

    windows = build_quarterly_windows()
    print(f"[wf_port] {len(windows)} quarterly windows planned "
          f"({WF_START.date()} -> {WF_END.date()})", file=sys.stderr)

    btc_windows: dict = {}
    sol_windows: dict = {}
    port_windows: dict = {}

    for w in windows:
        label = w["label"]
        tw = time.time()
        btc_r = _run_coin_wf_window(
            "BTC", label, w["train_start"], w["test_start"], w["test_end"],
        )
        sol_r = _run_coin_wf_window(
            "SOL", label, w["train_start"], w["test_start"], w["test_end"],
        )
        if btc_r is None and sol_r is None:
            print(f"  {label}: SKIP (no data either coin)", file=sys.stderr)
            continue
        btc_windows[label] = btc_r
        sol_windows[label] = sol_r
        pw = _synth_portfolio_window(btc_r, sol_r, W_BTC, W_SOL)
        port_windows[label] = pw

        btc_t = btc_r["trades"] if btc_r else 0
        sol_t = sol_r["trades"] if sol_r else 0
        print(f"  {label} test={w['test_start'].date()}->{w['test_end'].date()}  "
              f"btc(t={btc_t}, r={btc_r['return_pct']:+.2f}%) "
              f"sol(t={sol_t}, r={sol_r['return_pct']:+.2f}%) "
              f"port(t={pw['trades']}, r={pw['return_pct']:+.2f}%) "
              f"({time.time()-tw:.1f}s)", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------------
    quarters_total = len(port_windows)
    quarters_with_sufficient_trades = 0
    quarters_positive_sufficient = 0
    quarters_positive_all = 0

    all_port_pnl: list[float] = []
    per_window_summary = []
    compounded = 1.0

    for label, pw in port_windows.items():
        n_trades = pw["trades"]
        ret = pw["return_pct"]
        sufficient = n_trades >= MIN_TRADES_FOR_SUFFICIENT
        if sufficient:
            quarters_with_sufficient_trades += 1
            if ret > 0:
                quarters_positive_sufficient += 1
        if ret > 0:
            quarters_positive_all += 1
        compounded *= (1.0 + ret / 100.0)
        # Legacy stitched-trade-pool stream (kept for diff-ability against
        # pre-fix WF JSONs); replaced as headline by equity-curve PSR below.
        all_port_pnl.extend(
            pw.get("pnl_pct_trade_pool_proxy", pw.get("pnl_pct", []))
        )

        per_window_summary.append({
            "label":          label,
            "trades":         n_trades,
            "return_pct":     round(ret, 4),
            "btc_return_pct": pw.get("btc_return_pct"),
            "sol_return_pct": pw.get("sol_return_pct"),
            "max_dd_pct":     pw.get("max_dd_pct"),
            "sufficient":     bool(sufficient),
        })

    pct_positive_sufficient = (
        (quarters_positive_sufficient / quarters_with_sufficient_trades * 100.0)
        if quarters_with_sufficient_trades else 0.0
    )
    pct_positive_all = (
        (quarters_positive_all / quarters_total * 100.0)
        if quarters_total else 0.0
    )

    pnl_arr = np.asarray(all_port_pnl, dtype=float)
    psr_trade_pool_proxy = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)
        if len(pnl_arr) >= 2 else
        {"n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
         "interpretation": "insufficient_evidence"}
    )

    # Methodology debt #2 fix: equity-curve PSR across the quarterly windows.
    port_eq_per_window = {
        label: pw.get("portfolio_equity_series")
        for label, pw in port_windows.items()
        if pw is not None and pw.get("portfolio_equity_series") is not None
    }
    port_psr_true = aggregate_portfolio_psr(
        port_eq_per_window,
        resample_period="1D",
        sr_hurdle=0.0,
        confidence=0.95,
    )

    # Persist the literal daily-return arrays that compute_psr was fed, so the
    # headline PSR is recomputable bit-for-bit from the JSON. Built by calling
    # the canonical public helper (equity_to_period_returns) on the SAME dict
    # in the SAME iteration order aggregate_portfolio_psr uses; empty windows
    # contribute [] to per-window but are excluded from `combined` exactly as
    # the aggregator does (it skips empty pieces), so combined reproduces
    # n_periods_total and the headline PSR identically.
    psr_per_window_returns = {
        label: equity_to_period_returns(eq, resample_period="1D").tolist()
        for label, eq in port_eq_per_window.items()
    }
    _combined_pieces = [
        np.asarray(v, dtype=float)
        for v in psr_per_window_returns.values()
        if len(v) > 0
    ]
    psr_combined_returns = (
        np.concatenate(_combined_pieces).tolist() if _combined_pieces else []
    )

    # Headline `psr` (used in stdout + verdict block) now points at the
    # equity-curve metric; the legacy proxy is preserved alongside.
    psr = {
        "n_trades":      int(len(pnl_arr)),
        "psr_vs_hurdle": port_psr_true.get("psr_equity_curve"),
        "interpretation": port_psr_true.get("psr_interpretation"),
    }

    # Baselines being compared.
    baselines_compared = {
        "volsize_56":             56.0,   # AdaptiveTrend+volsize WF baseline
        "multifactor_v1_4h_btc_80": 80.0, # BTC-only multifactor WF baseline
        "gate_threshold":         70.0,
    }

    gate_70_met = pct_positive_sufficient >= 70.0

    verdict = ("PASSES_WALKFORWARD"
               if gate_70_met else "FAILS_WALKFORWARD")

    result = {
        "best_arm_tag":                       ARM_TAG,
        "allocation_btc":                     W_BTC,
        "allocation_sol":                     W_SOL,
        "veto_on":                            VETO_ON,
        "cash":                               CASH,
        "commission":                         COMMISSION,
        "margin":                             MARGIN,
        "price_scale":                        PRICE_SCALE,
        "walkforward": {
            "scheme":                         "rolling 6mo-train / 3mo-test",
            "start":                          str(WF_START.date()),
            "end":                            str(WF_END.date()),
            "min_trades_for_sufficient":      MIN_TRADES_FOR_SUFFICIENT,
        },
        "quarters_total":                     quarters_total,
        "quarters_with_sufficient_trades":    quarters_with_sufficient_trades,
        "quarters_positive_sufficient":       quarters_positive_sufficient,
        "quarters_positive_all":              quarters_positive_all,
        "pct_positive_sufficient":            round(pct_positive_sufficient, 2),
        "pct_positive_all":                   round(pct_positive_all, 2),
        "aggregate_compounded_pct":           round((compounded - 1.0) * 100.0, 4),
        "n_trades_total":                     int(len(pnl_arr)),
        "aggregate_psr":                      psr.get("psr_vs_hurdle"),
        "psr_interpretation":                 psr.get("interpretation"),
        "aggregate_psr_equity_curve":         port_psr_true.get(
            "psr_equity_curve"
        ),
        "aggregate_psr_trade_pool_proxy":     psr_trade_pool_proxy.get(
            "psr_vs_hurdle"
        ),
        "psr_n_periods":                      port_psr_true.get(
            "n_periods_total"
        ),
        "psr_resample_period":                port_psr_true.get(
            "resample_period"
        ),
        "sharpe_units":                       port_psr_true.get("sharpe_units"),
        # Forensic-recompute arrays: the literal per-period (daily) return
        # series fed to compute_psr. Sized to match psr_n_periods; combined is
        # the post-differencing concatenation across windows.
        "psr_per_window_returns":             psr_per_window_returns,
        "psr_combined_returns":               psr_combined_returns,
        "baselines_compared":                 baselines_compared,
        "gate_70_met":                        bool(gate_70_met),
        "verdict":                            verdict,
        "per_window_summary":                 per_window_summary,
        "elapsed_sec":                        round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / f"postfrac_wf_mf_4h_btc_sol_portfolio_{ARM_TAG}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[wf_port] wrote {out_path}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    print(f"  quarters_total={quarters_total}  sufficient="
          f"{quarters_with_sufficient_trades}  positive(suff)="
          f"{quarters_positive_sufficient}  pct={pct_positive_sufficient:.2f}%  "
          f"psr={psr.get('psr_vs_hurdle')}  gate_70_met={gate_70_met}",
          file=sys.stderr)

    # Headline JSON to stdout
    print(json.dumps({
        "best_arm_tag":                    ARM_TAG,
        "quarters_total":                  quarters_total,
        "quarters_with_sufficient_trades": quarters_with_sufficient_trades,
        "quarters_positive_sufficient":    quarters_positive_sufficient,
        "pct_positive_sufficient":         round(pct_positive_sufficient, 2),
        "aggregate_psr":                   psr.get("psr_vs_hurdle"),
        "baselines_compared":              baselines_compared,
        "gate_70_met":                     bool(gate_70_met),
        "verdict":                         verdict,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
