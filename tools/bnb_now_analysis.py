"""Fetch fresh BNB, chart it, and scan ALL validated signals for a CURRENT trigger.

Builds reports/BNB_NOW.html (4h candles + EMA24/200 + RSI, with hybrid-short
pattern markers) and prints which validated signal (if any) is true now.

Run: uv run python tools/bnb_now_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exchange.data import load_klines  # noqa: E402
from strategy.indicators import atr, ema  # noqa: E402
from tools.icnh_mega_sweep import Config, simulate_trades  # noqa: E402
from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
import tools.capitulation_watch as cw  # noqa: E402


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def main() -> int:
    h1 = load_klines("BNB/USDT:USDT", "1h", days_back=200).copy()
    h1.columns = [c.lower() for c in h1.columns]
    r4 = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                "close": "last", "volume": "sum"}).dropna()
    for p in (7, 24, 50, 100, 200):
        r4[f"ema{p}"] = ema(r4.close, p)
    r4["atr14"] = atr(r4.high, r4.low, r4.close, 14)
    r4["rsi"] = rsi(r4.close)
    h1["rsi"] = rsi(h1.close)
    h1["ema200"] = ema(h1.close, 200)

    now = r4.iloc[-1]
    px = float(now.close)
    ret24 = (h1.close.iloc[-1] / h1.close.iloc[-25] - 1) * 100
    ret7d = (h1.close.iloc[-1] / h1.close.iloc[-169] - 1) * 100

    print("=" * 60)
    print(f"BNB NOW  ({str(h1.index[-1])[:16]} UTC)  close=${px:.1f}")
    print("=" * 60)
    print(f"  24h return : {ret24:+.1f}%   7d return: {ret7d:+.1f}%")
    print(f"  1h  RSI={h1.rsi.iloc[-1]:.0f}   close>EMA200? {h1.close.iloc[-1] > h1.ema200.iloc[-1]}")
    print(f"  4h  RSI={now.rsi:.0f}   close>EMA200? {px > now.ema200}   "
          f"close>EMA24? {px > now.ema24}")
    print(f"  4h  EMA24=${now.ema24:.1f}  EMA200=${now.ema200:.1f}  ATR14=${now.atr14:.1f} "
          f"({now.atr14/px*100:.1f}%)")

    # --- Signal scan ---
    print("\n--- VALIDATED SIGNAL SCAN (does any fire on BNB now?) ---")
    # 1) hybrid-short (DT/ICnH) on 4h
    dt = Config(name="dt", pattern_type="distribution_top", direction="short", tf="4h",
                uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
                require_chop_at_top=True, breakdown_mode="chop_low_or_ema24",
                sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
                entry_emas=("ema24",), dedup_bars=15)
    ic = Config(name="ic", pattern_type="inverse_cnh", direction="short", tf="4h",
                cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
                handle_max_depth_frac=0.70, peak_tolerance=6, entry_emas=("ema24",),
                sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",), dedup_bars=15)
    hits = find_hybrid_patterns(r4, dt, ic)
    recent_short = [(str(r4.index[h])[:16], src) for h, src in hits if h >= len(r4) - 12]
    print(f"  hybrid-short (DT/ICnH, 4h): {'FIRES -> ' + str(recent_short) if recent_short else 'no signal in last 48h'}")

    # 2) capitulation LONG on 1h
    enr = cw._enrich(h1.copy())
    mask = cw._signal_mask(enr).values
    recent_capit = [str(enr.index[i])[:16] for i in np.where(mask)[0] if i >= len(enr) - 48]
    print(f"  capitulation LONG (1h): {'FIRES -> ' + str(recent_capit) if recent_capit else 'no signal in last 48h (needs a -15% crash, not a rally)'}")
    print(f"  v1 / donchian: BTC-only + dont transfer off BTC -> N/A for BNB")

    # --- Chart ---
    win = r4.iloc[-260:]  # ~43 days of 4h
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04, subplot_titles=("BNB/USDT 4h", "RSI(14)"))
    fig.add_trace(go.Candlestick(x=win.index, open=win.open, high=win.high,
                                 low=win.low, close=win.close, name="BNB"), 1, 1)
    for p, c in [(24, "#f59e0b"), (200, "#2563eb")]:
        fig.add_trace(go.Scatter(x=win.index, y=win[f"ema{p}"], name=f"EMA{p}",
                                 line=dict(width=1.2, color=c)), 1, 1)
    sh = [(r4.index[h], float(r4.close.iloc[h]), src) for h, src in hits if h >= len(r4) - 260]
    if sh:
        fig.add_trace(go.Scatter(x=[s[0] for s in sh], y=[s[1] for s in sh], mode="markers",
                                 marker=dict(symbol="triangle-down", size=12, color="red"),
                                 name="hybrid-short signal"), 1, 1)
    fig.add_trace(go.Scatter(x=win.index, y=win.rsi, name="RSI", line=dict(color="#7c3aed")), 2, 1)
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(height=760, xaxis_rangeslider_visible=False,
                      title=f"BNB NOW ${px:.0f} | 24h {ret24:+.1f}% | 4h RSI {now.rsi:.0f} | "
                            f"hybrid-short signals last 48h: {len(recent_short)}",
                      template="plotly_white")
    out = ROOT / "reports" / "BNB_NOW.html"
    fig.write_html(out)
    print(f"\nchart -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
