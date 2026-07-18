"""TurnOfCandle15m — 6-OOS-window validation runner.

Implements the gate-1 sweep for the turn-of-15m-candle TODO_LEG (spec:
.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_turn_of_candle_15m.md).

Configuration (pinned to spec)
------------------------------
- 6 OOS windows: 2023_H1 / 2023_H2 / 2024_H1 / 2024_H2 / 2025_H1 / 2025_H2
- CASH         = 1_000_000
- COMMISSION   = 0.00075   (7.5 bps per side = 15 bps round-trip — REALISTIC
                            Binance USDT-M perp; matches the load-bearing
                            gate-4 PSR-at-15bps stress level)
- MARGIN       = 1.0 / 3   (low leverage cap for a tiny-edge time-of-day
                            seasonal — paper edge is a few bps/trade gross)
- PRICE_SCALE  = 0.001     (fractional-sizing harness; 1 unit == 0.001 BTC)
- WARM_PREFIX  = 395 days  (no rolling stats actually need this in v1, but
                            mirrors the rv_band / portfolio runner contract
                            so cross-experiment OOS slicing is identical)

OOS trade attribution
---------------------
The data slice begins at (window_start - WARM_PREFIX_DAYS) so any future
add-on (vol filter, ATR exit) has warm history. Trades are attributed to
the OOS window only if EntryTime >= window_start — the warm prefix is
strictly for indicator stabilisation, NOT for return accumulation.

Equity-curve max-drawdown is computed on the equity slice from
window_start forward. Per-trade pnl_pct is concatenated across windows
and fed to PSR (hurdle=0, 95% confidence) for the aggregate gate-1 read.

Output
------
- reports/_postfrac_turn_of_candle_15m_<window>.csv  (per-window trades)
- reports/_postfrac_turn_of_candle_15m_aggregated.csv (all trades)
- reports/turn_of_candle_15m_oos.json                (full result)

Authority: research-only. Does NOT promote anything; gate-1 read only.
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

from strategy.signals_turn_of_candle_15m import TurnOfCandle15m  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.00075       # 7.5 bps per side -> 15 bps RT (gate-1 realistic)
MARGIN = 1.0 / 3           # low leverage cap for tiny-edge seasonal
PRICE_SCALE = 0.001        # 1 unit == 0.001 BTC under harness scaling
WARM_PREFIX_DAYS = 395     # mirror sibling runners (rv_band, portfolio)

# v1 = paper headline pinned. No in-window tuning. The trigger_minutes
# tuple is the ONLY knob the spec calls out (start hour-only; gate-4
# decides whether to even try the 4-turn aggressive variant).
CONFIG = {
    "trigger_minutes": (0,),
    "hold_bars_15m":   1,
    "allow_shorts":    True,
    "risk_per_trade_pct": 0.25,
    "leverage":        5,
}

WINDOWS_6 = [
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2023_H2", "2023-07-01", "2023-12-31"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
    ("2025_H2", "2025-07-01", "2025-12-31"),
]


def _load_slice_scaled(start: str, end: str, warm_days: int = 0) -> pd.DataFrame | None:
    """Load BTC 15m bars between [start - warm_days, end], price-scaled.

    The OOS window remains [start, end] — the caller filters trades by
    EntryTime >= start to attribute results.
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


def run_window(label: str, start: str, end: str, csv_prefix: str) -> dict | None:
    df = _load_slice_scaled(start, end, warm_days=WARM_PREFIX_DAYS)
    if df is None:
        print(f"  [{label}] SKIP — no data in window", file=sys.stderr)
        return None

    window_start_ts = pd.Timestamp(start)

    bt = Backtest(
        df,
        TurnOfCandle15m,
        cash=CASH,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(**CONFIG)

    # --- OOS trade attribution ------------------------------------------
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
        # LEGACY (v1) sizing-blind
        compounded_factor = float(np.prod(1.0 + ret_pct_series))
        legacy_compounded_oos = compounded_factor - 1.0
        equity_final = CASH * compounded_factor
        n_wins = int((ret_pct_series > 0).sum())
        win_rate = 100.0 * n_wins / n_trades

        # CANONICAL (v2) sizing-aware
        from tools.aggregate import equity_impact_returns as _eir
        stub = type("S", (), {"_trades": oos_trades})()
        eq_impact_pnl_pct = _eir(stub, cash=CASH).tolist()

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
        "label":             label,
        "start":             start,
        "end":               end,
        "trades":            n_trades,
        "return_pct":        round(ret_pct, 4),
        "legacy_return_pct": round(legacy_ret_pct, 4),
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


def main() -> int:
    t0 = time.time()
    csv_prefix = "_postfrac_turn_of_candle_15m"

    per_window: list[dict] = []
    print(f"[turn_of_candle_15m] running 6 OOS windows ...", file=sys.stderr)
    for label, start, end in WINDOWS_6:
        tw = time.time()
        r = run_window(label, start, end, csv_prefix)
        if r is None:
            continue
        print(
            f"  {label}  trades={r['trades']:5d}  ret={r['return_pct']:+8.2f}%  "
            f"dd={r['max_dd_pct']:+7.2f}%  win={r['win_rate_pct']:5.2f}%  "
            f"({time.time()-tw:.1f}s)",
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

    agg_csv = ROOT / "reports" / f"{csv_prefix}_aggregated.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
    print(f"  aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    # Canonical v2 dual-emit (methodology debt #1): headline PSR now comes from
    # the equity-curve window-level aggregation (psr_walkforward, n==n_windows),
    # NOT the N-inflated stitched per-trade ReturnPct union. Stitched value kept
    # as legacy_psr_stitched (observability only).
    from tools.aggregate import build_canonical_block, AGGREGATION_VERSION
    canon = build_canonical_block(per_window, aggregation_method=AGGREGATION_VERSION)
    psr = canon["psr_walkforward"]  # canonical headline PSR

    result = {
        "experiment":     "turn_of_candle_15m_v1",
        "strategy_class": "strategy.signals_turn_of_candle_15m:TurnOfCandle15m",
        "cash":           CASH,
        "commission":     COMMISSION,
        "commission_note": "7.5 bps per side = 15 bps RT (gate-4 load-bearing stress level)",
        "margin":         MARGIN,
        "margin_note":    "1/3 = up to 3x leverage cap (tiny-edge seasonal)",
        "price_scale":    PRICE_SCALE,
        "warm_prefix_days": WARM_PREFIX_DAYS,
        "config":         CONFIG,
        "windows":        [w[0] for w in WINDOWS_6],
        "per_window":     [
            {k: v for k, v in r.items() if k not in ("pnl_pct", "eq_impact_pnl_pct")}
            for r in per_window
        ],
        "summary":        agg,
        "psr":            psr,                  # canonical psr_walkforward (headline)
        "legacy_psr_stitched": legacy_psr_stitched,  # observability only
        "canonical":      canon,                # v2 dual-emit
        "aggregation_method": canon["aggregation_method"],
        "spec_reference": (
            "Caporale, Plastun & Oliinyk (Heliyon 2023, 9(3) e14077) "
            "'Turn-of-the-candle effect in bitcoin returns'. Paper headline "
            "Sharpe 4.96 net of 2022 retail fees; 2026 Binance USDT-M perp "
            "RT cost ~15-20 bps. Gate-4 PSR-at-15bps test is the load-bearing "
            "kill switch — if costs eat the edge, no parameter rescue."
        ),
        "elapsed_sec":    round(time.time() - t0, 2),
    }

    # --- bit-for-bit round-trip check (migration verification) --------------
    # contiguous=False matches aggregate_windows' psr_walkforward computation
    # on the ROUNDED per_window_return_pct array.
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
        f"[turn_of_candle_15m] canonical PSR round-trip OK: "
        f"{psr.get('psr_vs_hurdle')}",
        file=sys.stderr,
    )

    out_path = ROOT / "reports" / "turn_of_candle_15m_oos.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[turn_of_candle_15m] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
