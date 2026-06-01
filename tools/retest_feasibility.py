"""Phase 1 feasibility for Retest-Resume v1.

Concept: catch trend resumptions by entering AFTER a Donchian breakout's
retest, not on the breakout itself. The retest gives a tight structural
stop (just past the retested level), which fits $50/leg sizing where the
HYBRID short's wider 1.5×ATR stop did not.

Signal (LONG; SHORT mirrors):
  - Trend gate: daily close > daily EMA(50)
  - Breakout: 4h close > Donchian-20 upper, and prior 4h close ≤ prior upper
  - Retest:  within next 5 bars, low touches the breakout level within 0.5%
  - Confirmation: a 4h close back above the breakout level (after the retest)
  - Entry at confirmation close
  - Stop: 0.8 × ATR(14, 4h) past the breakout level (below for longs)
  - TP: 2.5R fixed (no trail yet — feasibility check only)
  - Time stop: 42 bars (7 trading days at 4h)

Gate (Phase 1):
  - ≥ 40 signals/year (≈1/week with realistic variance)
  - Win rate ≥ 45% (covers 30 bps net-edge requirement with 13 bps friction)
  - Sizing skip rate ≤ 30% at $50 equity / 2.0% risk

Run:
    uv run python tools/retest_feasibility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr, ema  # noqa: E402

DATA = ROOT / "data" / "historical"
RESULTS_PATH = ROOT / "data" / "retest_feasibility_results.json"

# ---- signal params ----
DONCH_N = 20            # 4h bars for Donchian channel
RETEST_MAX_BARS = 5     # bars after breakout to allow retest
RETEST_TOL_PCT = 0.005  # 0.5% tolerance on retest touch
ATR_PERIOD = 14
SL_ATR_MULT = 0.8
TP_R_MULT = 2.5         # take profit at 2.5R
TIME_STOP_BARS = 42     # 7 days at 4h
DAILY_TREND_EMA = 50

# ---- fee / sizing params ----
FRICTION_BPS_RT = 13.0          # live taker (8) + 5 bps slippage
EQUITY = 50.0
RISK_PCT = 2.0                  # %
MIN_NOTIONAL_USDT = 50.0
MIN_QTY_BTC = 0.001

# ---- universe ----
START = pd.Timestamp("2020-07-01", tz="UTC")
END = pd.Timestamp("2026-05-23", tz="UTC")

# ---- gates ----
GATE_MIN_SIGNALS_PER_YEAR = 40.0
GATE_MIN_WIN_RATE = 0.45
GATE_MAX_SKIP_PCT = 30.0


def _load_4h() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "BTC_USDT_USDT_4h.parquet")
    df.columns = [c.lower() for c in df.columns]
    df = df.loc[START:END].copy()
    df["atr14"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["donch_upper"] = df["close"].shift(1).rolling(DONCH_N).max()
    df["donch_lower"] = df["close"].shift(1).rolling(DONCH_N).min()
    # Daily trend gate — resample close to daily, compute EMA(50), reindex onto 4h.
    daily_close = df["close"].resample("1D").last().dropna()
    daily_ema = ema(daily_close, DAILY_TREND_EMA)
    # Forward-fill onto 4h so each 4h bar sees yesterday's daily reading (no peek).
    daily_ema_shifted = daily_ema.shift(1)
    df["daily_trend_ema"] = daily_ema_shifted.reindex(df.index, method="ffill")
    df["daily_close_prev"] = daily_close.shift(1).reindex(df.index, method="ffill")
    df["daily_trend_up"] = df["daily_close_prev"] > df["daily_trend_ema"]
    return df


def _scan_signals(df: pd.DataFrame) -> list[dict]:
    """Walk the 4h bars and emit one dict per retest-confirmed entry."""
    out: list[dict] = []
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    upper = df["donch_upper"].values
    lower = df["donch_lower"].values
    atr_v = df["atr14"].values
    trend_up = df["daily_trend_up"].values
    timestamps = df.index

    start = max(DONCH_N + 1, ATR_PERIOD + 1)
    for i in range(start, len(df) - TIME_STOP_BARS - 1):
        # Fresh breakout: just crossed upper or lower this bar (close-based).
        if not (np.isfinite(upper[i]) and np.isfinite(lower[i])):
            continue
        prev_above = np.isfinite(upper[i - 1]) and close[i - 1] > upper[i - 1]
        prev_below = np.isfinite(lower[i - 1]) and close[i - 1] < lower[i - 1]
        long_breakout = close[i] > upper[i] and not prev_above
        short_breakout = close[i] < lower[i] and not prev_below
        if not (long_breakout or short_breakout):
            continue

        direction = "long" if long_breakout else "short"
        level = upper[i] if direction == "long" else lower[i]
        if not np.isfinite(level):
            continue

        # Trend gate (daily side must agree).
        if direction == "long" and not bool(trend_up[i]):
            continue
        if direction == "short" and bool(trend_up[i]):
            continue

        # Retest scan: next 1..RETEST_MAX_BARS bars must touch level within tol.
        retest_idx = None
        tol_low = level * (1.0 - RETEST_TOL_PCT) if direction == "long" else level
        tol_high = level if direction == "long" else level * (1.0 + RETEST_TOL_PCT)
        for j in range(i + 1, min(i + 1 + RETEST_MAX_BARS, len(df))):
            if direction == "long":
                if low[j] <= level and low[j] >= tol_low:
                    retest_idx = j
                    break
            else:
                if high[j] >= level and high[j] <= tol_high:
                    retest_idx = j
                    break
        if retest_idx is None:
            continue

        # Confirmation: a later 4h close back through the level.
        confirm_idx = None
        for k in range(retest_idx + 1, min(retest_idx + 1 + RETEST_MAX_BARS, len(df))):
            if direction == "long" and close[k] > level:
                confirm_idx = k
                break
            if direction == "short" and close[k] < level:
                confirm_idx = k
                break
        if confirm_idx is None:
            continue

        entry_idx = confirm_idx
        entry_price = close[entry_idx]
        a = atr_v[entry_idx]
        if not (np.isfinite(a) and a > 0):
            continue

        stop_distance = SL_ATR_MULT * a
        if direction == "long":
            stop = entry_price - stop_distance
            tp = entry_price + TP_R_MULT * stop_distance
        else:
            stop = entry_price + stop_distance
            tp = entry_price - TP_R_MULT * stop_distance

        # Forward simulation up to TIME_STOP_BARS.
        exit_idx, exit_price, exit_reason = None, None, ""
        for m in range(entry_idx + 1, min(entry_idx + 1 + TIME_STOP_BARS, len(df))):
            h_m = high[m]
            l_m = low[m]
            if direction == "long":
                if l_m <= stop:
                    exit_idx, exit_price, exit_reason = m, stop, "sl"
                    break
                if h_m >= tp:
                    exit_idx, exit_price, exit_reason = m, tp, "tp"
                    break
            else:
                if h_m >= stop:
                    exit_idx, exit_price, exit_reason = m, stop, "sl"
                    break
                if l_m <= tp:
                    exit_idx, exit_price, exit_reason = m, tp, "tp"
                    break
        if exit_idx is None:
            exit_idx = min(entry_idx + TIME_STOP_BARS, len(df) - 1)
            exit_price = close[exit_idx]
            exit_reason = "time"

        if direction == "long":
            gross_pct = (exit_price - entry_price) / entry_price
        else:
            gross_pct = (entry_price - exit_price) / entry_price
        net_pct = gross_pct - FRICTION_BPS_RT / 10_000.0
        r_multiple = (exit_price - entry_price) / stop_distance if direction == "long" else (entry_price - exit_price) / stop_distance

        out.append({
            "entry_ts": str(timestamps[entry_idx]),
            "exit_ts": str(timestamps[exit_idx]),
            "direction": direction,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "stop": float(stop),
            "tp": float(tp),
            "stop_distance": float(stop_distance),
            "stop_pct": float(stop_distance / entry_price),
            "atr_pct": float(a / entry_price),
            "exit_reason": exit_reason,
            "bars_held": int(exit_idx - entry_idx),
            "gross_pct": float(gross_pct),
            "net_pct": float(net_pct),
            "r_multiple": float(r_multiple),
        })

    return out


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+6.2f}%"


def main() -> int:
    print("=" * 78)
    print("RETEST-RESUME v1 — Phase 1 feasibility")
    print(f"Window:    {START.date()} → {END.date()}")
    print(f"Sizing:    equity ${EQUITY:.0f}, risk {RISK_PCT}%, "
          f"min-notional ${MIN_NOTIONAL_USDT:.0f}, min-qty {MIN_QTY_BTC} BTC")
    print(f"Friction:  {FRICTION_BPS_RT} bps round-trip")
    print("=" * 78)

    df = _load_4h()
    trades = _scan_signals(df)

    n = len(trades)
    years = (END - START).days / 365.25
    if n == 0:
        print("No signals — abort.")
        return 1

    # ---- frequency + outcome ----
    longs = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    nets = np.array([t["net_pct"] for t in trades])
    rs = np.array([t["r_multiple"] for t in trades])
    bars = np.array([t["bars_held"] for t in trades])
    win = nets > 0

    print(f"\nSignals: {n} over {years:.2f} years  →  {n / years:.1f}/yr  "
          f"(≈{n / years / 52:.2f}/wk)")
    print(f"  long={len(longs)}  short={len(shorts)}")
    print(f"  exit reasons: "
          f"tp={sum(1 for t in trades if t['exit_reason'] == 'tp')}  "
          f"sl={sum(1 for t in trades if t['exit_reason'] == 'sl')}  "
          f"time={sum(1 for t in trades if t['exit_reason'] == 'time')}")
    print(f"\nOutcome:")
    print(f"  win rate          {win.mean() * 100:>5.1f}%")
    print(f"  mean R            {rs.mean():>+6.2f}")
    print(f"  median R          {np.median(rs):>+6.2f}")
    print(f"  cum (net, all)    {_fmt_pct(float(np.prod(1.0 + nets) - 1.0))}")
    print(f"  mean net / trade  {nets.mean() * 10_000:+.1f} bps")
    print(f"  bars held (avg)   {bars.mean():.1f} (≈{bars.mean() * 4 / 24:.1f} days)")
    print(f"  bars held (med)   {np.median(bars):.0f}")

    # Stop-pct distribution (for sizing math).
    stop_pcts = np.array([t["stop_pct"] for t in trades])
    print(f"\nStop-pct distribution (at entry):")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"  q={q:.2f}  {np.quantile(stop_pcts, q) * 100:.2f}%")

    # ---- sizing skip rate ----
    skip_min_not = 0
    skip_min_qty = 0
    skipped = 0
    kept_nets: list[float] = []
    for t, sp in zip(trades, stop_pcts):
        if not (np.isfinite(sp) and sp > 0):
            skipped += 1
            continue
        notional = EQUITY * (RISK_PCT / 100.0) / sp
        qty = notional / t["entry_price"]
        if notional < MIN_NOTIONAL_USDT:
            skipped += 1
            skip_min_not += 1
            continue
        if qty < MIN_QTY_BTC:
            skipped += 1
            skip_min_qty += 1
            continue
        kept_nets.append(t["net_pct"])
    skip_pct = skipped / n * 100
    kept_n = len(kept_nets)
    kept_cum = (float(np.prod(1.0 + np.array(kept_nets)) - 1.0)
                if kept_nets else 0.0)

    print(f"\nSizing at ${EQUITY:.0f} / {RISK_PCT}% risk:")
    print(f"  signals      {n}")
    print(f"  skipped      {skipped} ({skip_pct:.1f}%) "
          f"[min-not {skip_min_not}, min-qty {skip_min_qty}]")
    print(f"  kept         {kept_n}")
    print(f"  kept cum     {_fmt_pct(kept_cum)}")
    print(f"  kept per yr  {kept_n / years:.1f}")

    # ---- per-year breakdown ----
    by_year: dict[int, list[dict]] = {}
    for t in trades:
        y = pd.Timestamp(t["entry_ts"]).year
        by_year.setdefault(y, []).append(t)
    print(f"\nPer year:")
    print(f"  {'year':<6}{'n':>5}{'WR':>8}{'meanR':>8}{'cum':>12}")
    for y in sorted(by_year):
        ts = by_year[y]
        n_y = len(ts)
        wr_y = sum(1 for t in ts if t["net_pct"] > 0) / n_y
        meanR_y = float(np.mean([t["r_multiple"] for t in ts]))
        cum_y = float(np.prod([1.0 + t["net_pct"] for t in ts]) - 1.0)
        print(f"  {y:<6}{n_y:>5}{wr_y * 100:>7.1f}%{meanR_y:>+8.2f}{_fmt_pct(cum_y):>12}")

    # ---- verdict ----
    sig_per_year = n / years
    win_rate = float(win.mean())
    sig_pass = sig_per_year >= GATE_MIN_SIGNALS_PER_YEAR
    wr_pass = win_rate >= GATE_MIN_WIN_RATE
    skip_pass = skip_pct <= GATE_MAX_SKIP_PCT

    print("\n" + "=" * 78)
    print("PHASE 1 VERDICT")
    print("=" * 78)
    print(f"  signals/yr  ≥ {GATE_MIN_SIGNALS_PER_YEAR:.0f}: "
          f"{sig_per_year:.1f}  → {'PASS' if sig_pass else 'FAIL'}")
    print(f"  win rate    ≥ {GATE_MIN_WIN_RATE * 100:.0f}%: "
          f"{win_rate * 100:.1f}%  → {'PASS' if wr_pass else 'FAIL'}")
    print(f"  skip rate   ≤ {GATE_MAX_SKIP_PCT:.0f}%: "
          f"{skip_pct:.1f}%  → {'PASS' if skip_pass else 'FAIL'}")
    overall = sig_pass and wr_pass and skip_pass
    print(f"  Overall: {'PASS — proceed to Phase 2' if overall else 'FAIL — abandon or re-pitch'}")
    print("=" * 78)

    RESULTS_PATH.write_text(json.dumps({
        "window": [str(START.date()), str(END.date())],
        "signals": n,
        "signals_per_year": sig_per_year,
        "win_rate": win_rate,
        "mean_r": float(rs.mean()),
        "median_r": float(np.median(rs)),
        "cum_pct": float(np.prod(1.0 + nets) - 1.0),
        "mean_net_bps": float(nets.mean() * 10_000),
        "bars_held_mean": float(bars.mean()),
        "stop_pct_quantiles": {
            f"q{int(q * 100)}": float(np.quantile(stop_pcts, q))
            for q in (0.10, 0.25, 0.50, 0.75, 0.90)
        },
        "skip_pct": skip_pct,
        "skip_min_notional": skip_min_not,
        "skip_min_qty": skip_min_qty,
        "kept_trades": kept_n,
        "kept_cum_pct": kept_cum,
        "verdict": "PASS" if overall else "FAIL",
        "trades": trades,
    }, indent=2, default=str))
    print(f"Saved → {RESULTS_PATH}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
