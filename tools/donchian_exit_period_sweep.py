"""donchian-v3: is `donchian_period_exit: 10` an optimum, or an artifact?

MOTIVATION (God, 2026-08-22): "why must it be last 10 bar? cant it be 5? or any
number, or you have backtested for all".

Answer from the tree before running anything: 10 vs 20 is the ENTIRE search that
has ever been performed on this parameter. Every sweep grid in config/ lists
`donchian_period_exit: [10, 20]` — sweep_donchian.yaml, sweep_donchian_v2_tf.yaml,
sweep_donchian_v3.yaml. The live 10 arrived in 4ab656c ("the validated 2026-07-16
sweep geometry (exit-channel 20->10: +31pp OOS, +18% trades/yr, better DD)"), i.e.
it beat the only other value anyone tried. 5 was never run. Nor 8, 12, 15, 30, 40.

This sweep fills that gap. It varies `donchian_period_exit` ALONE across a wide
grid and holds every other class attr at the DEPLOYED config
(config/params_donchian.yaml) — so the entry channel stays 80 and what moves is
purely the exit geometry, i.e. the entry:exit ratio.

DESIGN NOTES
  * There is deliberately NO separate "baseline" arm. The deployed value is just
    the arm labelled 10; adding a baseline pinned to 10 would be a
    self-comparison and would make the table look like it had a control when it
    did not.
  * Methodology is INHERITED, not reimplemented: the OOS windows, the quarterly
    walk-forward, `_prep_slice`, `_run_bt`, and both aggregators are imported
    from tools/_postfrac_donchian_variants_sweep. That module is the provenance
    for the prior donchian verdict and is import-safe (its main() is guarded).
    Numbers here are therefore comparable to that run, arm for arm.
  * DEGENERACY GUARD: `donchian_period_exit` feeds
    `Close.rolling(N, min_periods=N).min()/.max().shift(1)` with a STRICT
    comparison. As N falls the exit channel collapses toward the previous bar's
    close, so any pullback exits immediately — high trade counts made of instant
    round-trips each paying 15bps. That is not a finding about channel geometry,
    it is a parameter the code accepts and the semantics do not support (same
    shape as the `max_hold_bars: 0` incident). So every arm reports MEDIAN BARS
    HELD, and any arm whose median hold is <= 2 bars is flagged `degenerate` and
    must not be read as a result.
  * 15bps round-trip is pessimistic vs the ~5-10bps this account actually pays.
    Kept anyway: it is the convention these sweeps use, and comparability across
    arms matters more than the absolute level. Do NOT quote these returns as
    expected P&L.
  * READ-ONLY research. This script never writes config/. The leg is live; a
    config change means a restart and boot() flattens open positions.

Usage:
  uv run python -m tools.donchian_exit_period_sweep              # full run
  uv run python -m tools.donchian_exit_period_sweep --skip-wf    # OOS only (fast)
  uv run python -m tools.donchian_exit_period_sweep --periods 8,10,12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.signals_donchian import DonchianBreakoutBTCv3  # noqa: E402
from tools._postfrac_donchian_variants_sweep import (  # noqa: E402
    CASH,
    COMMISSION,
    MARGIN,
    OOS_WINDOWS,
    WARM_PREFIX_DAYS,
    _load_full_scaled_4h,
    _prep_slice,
    aggregate_oos,
    aggregate_wf,
    build_wf_windows,
    run_oos_window,
    run_wf_window,
)

REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "donchian_exit_period_sweep.json"

# The grid. 10 and 20 are the only two ever run before today; the rest are new.
# 3 and 5 are included so the degeneracy floor is VISIBLE rather than assumed.
DEFAULT_PERIODS = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40]

# Every non-swept attr, pinned to config/params_donchian.yaml as deployed.
# Verified field-by-field against that file on 2026-08-22.
DEPLOYED_ATTRS: dict = {
    "donchian_period_entry":      80,
    # donchian_period_exit is the swept axis — injected per arm.
    "atr_sl_multiple":            1.5,
    # atr_tp_multiple intentionally absent: donchian has NO TP (channel exit),
    # and the class does not declare it, so passing it AttributeErrors in bt.run.
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
}

DEGENERATE_MEDIAN_BARS = 2.0   # median hold <= this ⇒ arm is not a real result


def arm(period_exit: int) -> dict:
    """One sweep arm: the deployed config with a single value swapped."""
    return {
        "variant_id":    f"exit_{period_exit}",
        "strategy_name": "donchian-v3",
        "symbol":        "BTC/USDT:USDT",
        "timeframe":     "4h",
        "class_attrs":   {**DEPLOYED_ATTRS, "donchian_period_exit": int(period_exit)},
        "notes":         f"deployed params_donchian.yaml with donchian_period_exit={period_exit}",
    }


def full_period_profile(full_df: pd.DataFrame, attrs: dict) -> dict:
    """Whole-history run for one arm, kept for the HOLD-DURATION distribution.

    The OOS/WF numbers below come from the shared harness; this exists because
    `_run_bt` returns aggregates only and the degeneracy check needs bars-held.
    Same Backtest construction as `_run_bt` so the two stay comparable.
    """
    df = _prep_slice(
        full_df, full_df.index[0], full_df.index[-1],
        period_entry=int(attrs["donchian_period_entry"]),
        period_exit=int(attrs["donchian_period_exit"]),
    )
    bt = Backtest(
        df, DonchianBreakoutBTCv3,
        cash=CASH, commission=COMMISSION, margin=MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run(**attrs)
    trades = getattr(stats, "_trades", None)

    bars_held: list[float] = []
    if trades is not None and len(trades) > 0:
        if {"EntryBar", "ExitBar"}.issubset(trades.columns):
            bars_held = (trades["ExitBar"] - trades["EntryBar"]).astype(float).tolist()

    n = int(len(trades)) if trades is not None else 0
    span_years = (df.index[-1] - df.index[0]).days / 365.25
    med = float(pd.Series(bars_held).median()) if bars_held else 0.0
    return {
        "trades":            n,
        "trades_per_year":   round(n / span_years, 2) if span_years else 0.0,
        "return_pct":        round(float(stats.get("Return [%]", 0.0) or 0.0), 2),
        "max_dd_pct":        round(float(stats.get("Max. Drawdown [%]", 0.0) or 0.0), 2),
        "win_rate_pct":      round(float(stats.get("Win Rate [%]") or 0.0), 2),
        "sharpe":            round(float(stats.get("Sharpe Ratio") or 0.0), 3),
        "profit_factor":     round(float(stats.get("Profit Factor") or 0.0), 3),
        "median_bars_held":  round(med, 1),
        "mean_bars_held":    round(float(pd.Series(bars_held).mean()), 1) if bars_held else 0.0,
        "min_bars_held":     round(float(pd.Series(bars_held).min()), 1) if bars_held else 0.0,
        "pct_1bar_exits":    round(100.0 * sum(1 for b in bars_held if b <= 1) / n, 1) if n else 0.0,
        "degenerate":        bool(bars_held) and med <= DEGENERATE_MEDIAN_BARS,
        "span_years":        round(span_years, 2),
    }


def run_arm(full_df: pd.DataFrame, period_exit: int, skip_wf: bool) -> dict:
    v = arm(period_exit)
    attrs = v["class_attrs"]
    t0 = time.time()

    profile = full_period_profile(full_df, attrs)

    oos_windows = [
        run_oos_window(
            full_df, v, label, start, end,
            REPORTS / f"_dexit_{period_exit}_{label}.csv",
        )
        for (label, start, end) in OOS_WINDOWS
    ]
    oos = aggregate_oos(oos_windows)

    wf = None
    if not skip_wf:
        wf_rows = []
        for w in build_wf_windows():
            r = run_wf_window(
                full_df, v, w, REPORTS / f"_dexit_{period_exit}_wf_{w['label']}.csv",
            )
            if r is not None:
                wf_rows.append(r)
        wf = aggregate_wf(wf_rows) if wf_rows else None

    return {
        "period_exit":  period_exit,
        "class_attrs":  attrs,
        "full_period":  profile,
        "oos":          oos,
        "oos_windows":  oos_windows,
        "wf":           wf,
        "elapsed_s":    round(time.time() - t0, 1),
    }


def _fmt_table(results: list[dict]) -> str:
    hdr = (f"{'exit':>5} {'trades':>7} {'/yr':>6} {'medBars':>8} {'1bar%':>6} "
           f"{'fullRet%':>9} {'DD%':>7} {'WR%':>6} {'PF':>6} "
           f"{'OOS5 cmp%':>10} {'win':>4} {'WF pos%':>8} {'WFtr':>5}  flag")
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        p, o, w = r["full_period"], r["oos"], r["wf"]
        flag = "DEGENERATE" if p["degenerate"] else ""
        if r["period_exit"] == 10:
            flag = (flag + " <-- DEPLOYED").strip()
        lines.append(
            f"{r['period_exit']:>5} {p['trades']:>7} {p['trades_per_year']:>6.1f} "
            f"{p['median_bars_held']:>8.1f} {p['pct_1bar_exits']:>6.1f} "
            f"{p['return_pct']:>9.1f} {p['max_dd_pct']:>7.1f} {p['win_rate_pct']:>6.1f} "
            f"{p['profit_factor']:>6.2f} {o['compounded_pct']:>10.1f} "
            f"{o['windows_positive']:>4} "
            f"{(w['pct_positive'] if w else float('nan')):>8.1f} "
            f"{(w['n_trades_total'] if w else 0):>5}  {flag}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-wf", action="store_true", help="OOS only (much faster)")
    ap.add_argument("--periods", type=str, default="",
                    help="comma-separated exit periods (default: the full grid)")
    args = ap.parse_args()

    periods = ([int(x) for x in args.periods.split(",") if x.strip()]
               if args.periods else DEFAULT_PERIODS)

    REPORTS.mkdir(exist_ok=True)
    full_df = _load_full_scaled_4h()
    print(f"data: {full_df.index[0]} -> {full_df.index[-1]}  ({len(full_df)} 4H bars)")
    print(f"arms: {periods}   (entry channel fixed at 80)\n")

    results = []
    for p in periods:
        print(f"  running exit={p} ...", flush=True)
        results.append(run_arm(full_df, p, args.skip_wf))

    payload = {
        "generated_at":  pd.Timestamp.now("UTC").isoformat(),
        "question":      "is donchian_period_exit=10 an optimum or an artifact of a 2-point grid?",
        "entry_period_fixed": DEPLOYED_ATTRS["donchian_period_entry"],
        "commission_roundtrip_bps": COMMISSION * 2 * 10_000,
        "deployed_period_exit": 10,
        "previously_tested":  [10, 20],
        "arms":          results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))

    print("\n" + _fmt_table(results))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
