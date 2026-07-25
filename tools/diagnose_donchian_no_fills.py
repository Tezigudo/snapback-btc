"""
Why has donchian-v3 not fired a single order since going live on 2026-07-02?

Replays the LIVE evaluator (`strategy.live_donchian_v3.evaluate_signal_donchian_v3`)
bar-by-bar over real BTC 4h data with the deployed config, and classifies every
bar's blocking reason. This is the live code path, not the backtest one, so the
answer is what the bot itself would have decided.

Run: .venv/bin/python tools/diagnose_donchian_no_fills.py
"""

from __future__ import annotations

import collections
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.live_donchian_v3 import (  # noqa: E402
    _ema_slope_signed, evaluate_signal_donchian_v3,
)

REPO = Path(__file__).resolve().parent.parent
LIVE_DATE = datetime(2026, 7, 2, tzinfo=UTC)   # donchian flipped real-money
DONCHIAN_FUNDED = datetime(2026, 7, 2, tzinfo=UTC)


def load_btc_4h() -> pd.DataFrame:
    df = pd.read_parquet(REPO / "data" / "historical" / "BTC_USDT_USDT_4h.parquet")
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def main() -> int:
    params = yaml.safe_load(open(REPO / "config" / "params_donchian.yaml"))
    s = params["strategy"]
    pe = int(s["donchian_period_entry"])
    thr = float(s["slope_trend_threshold_pct"])
    ema_p = int(s["regime_ema_period"])
    win = int(s["regime_slope_window"])

    df = load_btc_4h()
    warmup = max(pe, ema_p + win, int(s["atr_period"])) + 5
    print(f"Deployed config: entry channel {pe} · exit {s['donchian_period_exit']} · "
          f"regime EMA {ema_p} slope-window {win} · slope threshold ±{thr} %/bar")
    print(f"BTC 4h data through {df.index.max()}  ({len(df)} bars)\n")

    # ---- Replay every bar since going live -------------------------------
    for label, since in [("SINCE GOING LIVE 2026-07-02", LIVE_DATE),
                         ("LAST 12 MONTHS", datetime(2025, 7, 25, tzinfo=UTC))]:
        idx = df.index[df.index >= since]
        reasons = collections.Counter()
        signals = []
        for ts in idx:
            window = df.loc[:ts]
            if len(window) < warmup:
                continue
            side, _sl, _tp, dbg = evaluate_signal_donchian_v3(window, 0.0, params)
            if side:
                reasons[f"SIGNAL {side}"] += 1
                signals.append((ts, side, dbg))
            else:
                reasons[dbg.get("reason", "?")] += 1
        total = sum(reasons.values())
        print("=" * 84)
        print(f"{label} — {total} closed 4h bars evaluated")
        print("=" * 84)
        for r, n in reasons.most_common():
            print(f"  {r:<26} {n:>5}  ({100 * n / total:5.1f}%)")
        if signals:
            print(f"  last signals: "
                  + ", ".join(f"{ts:%Y-%m-%d %H:%M} {sd}" for ts, sd, _ in signals[-5:]))
        print()

    # ---- How close is it right now? --------------------------------------
    last = df.index[-1]
    window = df.loc[:last]
    side, _sl, _tp, dbg = evaluate_signal_donchian_v3(window, 0.0, params)
    close = df["Close"]
    upper = close.rolling(pe, min_periods=pe).max().shift(1).iloc[-1]
    lower = close.rolling(pe, min_periods=pe).min().shift(1).iloc[-1]
    slope = _ema_slope_signed(close, ema_p, win)
    cur = float(close.iloc[-1])

    print("=" * 84)
    print(f"CURRENT STATE at last closed 4h bar {last:%Y-%m-%d %H:%M} UTC")
    print("=" * 84)
    print(f"  BTC close            ${cur:,.0f}")
    print(f"  {pe}-bar upper channel  ${upper:,.0f}   "
          f"(need close > this for LONG — {100 * (upper / cur - 1):+.2f}% away)")
    print(f"  {pe}-bar lower channel  ${lower:,.0f}   "
          f"(need close < this for SHORT — {100 * (lower / cur - 1):+.2f}% away)")
    print(f"  EMA{ema_p} slope         {slope:+.4f} %/bar   "
          f"(need ≥ +{thr} for LONG, ≤ -{thr} for SHORT)")
    print(f"  verdict              {side or dbg.get('reason')}")
    print()

    # ---- Decompose: breakout vs gate over the last 12 months -------------
    print("=" * 84)
    print("DECOMPOSITION — is it the channel or the gate that blocks?")
    print("=" * 84)
    idx = df.index[df.index >= datetime(2025, 7, 25, tzinfo=UTC)]
    up_break = gate_ok_long = both_long = 0
    dn_break = gate_ok_short = both_short = 0
    for ts in idx:
        w = df.loc[:ts]
        if len(w) < warmup:
            continue
        c = w["Close"]
        u = c.rolling(pe, min_periods=pe).max().shift(1).iloc[-1]
        lo = c.rolling(pe, min_periods=pe).min().shift(1).iloc[-1]
        sl_ = _ema_slope_signed(c, ema_p, win)
        cc = float(c.iloc[-1])
        if cc > u:
            up_break += 1
        if sl_ >= thr:
            gate_ok_long += 1
        if cc > u and sl_ >= thr:
            both_long += 1
        if cc < lo:
            dn_break += 1
        if sl_ <= -thr:
            gate_ok_short += 1
        if cc < lo and sl_ <= -thr:
            both_short += 1
    n = len(idx)
    print(f"  bars: {n}")
    print(f"  LONG : channel breakout {up_break:>4} bars | slope gate open "
          f"{gate_ok_long:>4} bars | BOTH {both_long:>4}")
    print(f"  SHORT: channel breakout {dn_break:>4} bars | slope gate open "
          f"{gate_ok_short:>4} bars | BOTH {both_short:>4}")
    print()
    print("  Reading: if a breakout column is non-zero but BOTH is zero, the gate is")
    print("  the binding constraint. If the breakout column is itself ~zero, the")
    print("  80-bar channel simply has not been pierced and the gate is irrelevant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
