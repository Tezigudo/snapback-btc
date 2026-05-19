"""Multi-timeframe BTC chart.

For a given timestamp, render 5 timeframes stacked vertically:
  30m, 1h, 4h, 1d, 1w (configurable subset).

Each timeframe panel shows:
  - Candlesticks
  - EMA ribbon: 7, 24, 50, 100, 200 (color-graded)
  - Parabolic SAR(0.02, 0.2) dots
  - Auto trendlines: resistance from last 3 swing highs, support from last 3 swing lows
  - RSI(14) sub-band (overlay at bottom of price panel as a small inset)
  - Volume bars below

Usage:
  python -m tools.chart_mtf --timestamp 2025-04-23T15:30 --out reports/mtf_charts/sample.png
  python -m tools.chart_mtf --timestamp 2025-04-23T15:30 --tfs 1h,4h,1d --out ...
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategy.indicators import (
    ema,
    parabolic_sar,
    rsi,
    sma,
    swing_high_low,
    trendline_from_swings,
)

# Map TF label to (pandas resample rule, bars-to-show)
TF_RULES = {
    # (resample_rule, bars_to_show). Windows sized so EMA200 fits.
    "30m": ("30min", 260),    # ~5.4 days
    "1h":  ("1h",    240),    # ~10 days
    "4h":  ("4h",    240),    # ~40 days
    "1d":  ("1D",    240),    # ~8 months
    "1w":  ("1W",    150),    # ~3 years (EMA200 won't fit here, fallback used)
}

EMA_PERIODS = [7, 24, 50, 100, 200]
# Color gradient: short = warm/red-ish, long = cool/blue. Matches "ribbon" convention.
EMA_COLORS = {
    7:   "#e53935",   # red
    24:  "#fb8c00",   # orange
    50:  "#fdd835",   # yellow
    100: "#43a047",   # green
    200: "#1e88e5",   # blue
}


def _read_15m() -> pd.DataFrame:
    df = pd.read_parquet("data/historical/BTC_USDT_USDT_15m.parquet")
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV using standard bar aggregation."""
    o = df.resample(rule, label="left", closed="left").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open"])
    return o


def _plot_candles(ax, win: pd.DataFrame, width_frac: float = 0.8) -> None:
    if win.empty:
        return
    for ts, row in win.iterrows():
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax.plot([ts, ts], [row["Low"], row["High"]], color=color, linewidth=0.6, alpha=0.7)
        # Body as fat segment
        ax.plot([ts, ts], [row["Open"], row["Close"]], color=color, linewidth=2.0)


def _draw_trendlines(ax, win: pd.DataFrame) -> None:
    """Resistance (red) from last 3 swing highs; support (green) from last 3 swing lows."""
    sh, sl = swing_high_low(win["High"], win["Low"], k=3)
    sh_idx = np.where(sh.values)[0]
    sl_idx = np.where(sl.values)[0]

    # Plot swing markers
    if len(sh_idx):
        ax.scatter(win.index[sh_idx], win["High"].values[sh_idx],
                   marker="v", color="#c62828", s=25, alpha=0.7, zorder=4, label="swing high")
    if len(sl_idx):
        ax.scatter(win.index[sl_idx], win["Low"].values[sl_idx],
                   marker="^", color="#2e7d32", s=25, alpha=0.7, zorder=4, label="swing low")

    res = trendline_from_swings(sh, win["High"], n_recent=3)
    sup = trendline_from_swings(sl, win["Low"], n_recent=3)
    if res is not None and len(sh_idx) >= 3:
        slope, intercept = res
        x0, x1 = sh_idx[-3], len(win) - 1
        y0 = slope * x0 + intercept
        y1 = slope * x1 + intercept
        ax.plot([win.index[x0], win.index[x1]], [y0, y1],
                color="#c62828", linewidth=1.2, linestyle="--", alpha=0.7, label="resistance")
    if sup is not None and len(sl_idx) >= 3:
        slope, intercept = sup
        x0, x1 = sl_idx[-3], len(win) - 1
        y0 = slope * x0 + intercept
        y1 = slope * x1 + intercept
        ax.plot([win.index[x0], win.index[x1]], [y0, y1],
                color="#2e7d32", linewidth=1.2, linestyle="--", alpha=0.7, label="support")


def render_tf_panel(ax_price, ax_vol, win: pd.DataFrame, tf_label: str) -> dict:
    """Render one TF panel onto provided axes. Returns a summary dict of state."""
    if win.empty:
        ax_price.set_title(f"{tf_label} — no data", fontsize=10)
        return {"tf": tf_label, "trend": "no_data"}

    # --- candles ---
    _plot_candles(ax_price, win)

    # --- EMA ribbon ---
    ema_vals = {}
    for p in EMA_PERIODS:
        if len(win) >= p:
            e = ema(win["Close"], p)
            ema_vals[p] = e.iloc[-1]
            ax_price.plot(win.index, e, color=EMA_COLORS[p],
                          linewidth=1.0, alpha=0.85, label=f"EMA{p}")

    # --- Parabolic SAR ---
    sar = parabolic_sar(win["High"], win["Low"])
    # plot SAR dots colored by side
    sar_close = win["Close"]
    long_mask = sar < sar_close
    ax_price.scatter(win.index[long_mask], sar[long_mask],
                     color="#2e7d32", s=6, alpha=0.6)
    ax_price.scatter(win.index[~long_mask & sar.notna()], sar[~long_mask & sar.notna()],
                     color="#c62828", s=6, alpha=0.6)

    # --- Trendlines + swings ---
    _draw_trendlines(ax_price, win)

    # --- Titling + legend ---
    close = win["Close"].iloc[-1]
    trend_state = _classify_trend(ema_vals, close, sar.iloc[-1] if pd.notna(sar.iloc[-1]) else close)
    ax_price.set_title(f"{tf_label}   close ${close:,.0f}   trend: {trend_state}",
                        fontsize=10, loc="left")
    ax_price.legend(loc="upper left", fontsize=6, ncol=4)
    ax_price.grid(True, alpha=0.2)
    ax_price.set_ylabel("Price")

    # --- Volume panel ---
    colors = ["#26a69a" if c >= o else "#ef5350"
              for o, c in zip(win["Open"], win["Close"])]
    if len(win) >= 2:
        w_days = (win.index[1] - win.index[0]).total_seconds() / 86400.0 * 0.7
    else:
        w_days = 0.7
    ax_vol.bar(win.index, win["Volume"], color=colors, alpha=0.7, width=w_days)
    if len(win) >= 20:
        vsma = sma(win["Volume"], 20)
        ax_vol.plot(win.index, vsma, color="#1565c0", linewidth=0.7, label="vol SMA(20)")
        ax_vol.legend(loc="upper left", fontsize=6)
    ax_vol.set_ylabel("Vol")
    ax_vol.grid(True, alpha=0.2)

    return {
        "tf": tf_label,
        "close": close,
        "trend": trend_state,
        "ema": ema_vals,
        "sar": float(sar.iloc[-1]) if pd.notna(sar.iloc[-1]) else None,
        "rsi": float(rsi(win["Close"], 14).iloc[-1]),
    }


def _classify_trend(ema_vals: dict, close: float, sar: float) -> str:
    """Trend classification using EMA stack alignment + SAR side.

    UP: EMA7 > EMA24 > EMA50 > EMA100 > EMA200 AND close > SAR
    DOWN: reverse
    MIXED: anything else
    """
    needed = [7, 24, 50, 100, 200]
    if not all(p in ema_vals for p in needed):
        # Insufficient data for stack alignment
        if 50 in ema_vals and 200 in ema_vals:
            if close > ema_vals[200] and ema_vals[50] > ema_vals[200]:
                return "up (partial)"
            if close < ema_vals[200] and ema_vals[50] < ema_vals[200]:
                return "down (partial)"
        return "unknown"
    vals = [ema_vals[p] for p in needed]
    if all(vals[i] > vals[i + 1] for i in range(4)) and close > sar:
        return "UP (strong)"
    if all(vals[i] < vals[i + 1] for i in range(4)) and close < sar:
        return "DOWN (strong)"
    # Weaker confirmations
    if close > ema_vals[200] and ema_vals[50] > ema_vals[200]:
        return "up (weak)"
    if close < ema_vals[200] and ema_vals[50] < ema_vals[200]:
        return "down (weak)"
    return "MIXED"


def build_mtf_figure(timestamp: pd.Timestamp, tfs: list[str]) -> tuple[plt.Figure, list[dict]]:
    base = _read_15m()
    # truncate to up-to-and-including timestamp
    base = base.loc[:timestamp]

    n = len(tfs)
    # Each TF: 1 price panel (3 units high) + 1 volume panel (1 unit high)
    h_ratios = []
    for _ in tfs:
        h_ratios.extend([3, 1])
    fig, axes = plt.subplots(2 * n, 1, figsize=(13, 3.0 * n),
                              gridspec_kw={"height_ratios": h_ratios},
                              sharex=False)
    if n == 1:
        axes = [axes[0], axes[1]]

    fig.suptitle(f"BTC/USDT Multi-Timeframe — anchored at {timestamp:%Y-%m-%d %H:%M} UTC",
                 fontsize=12, y=1.0)

    summaries = []
    for i, tf in enumerate(tfs):
        rule, n_bars = TF_RULES[tf]
        resampled = resample_ohlcv(base, rule).tail(n_bars)
        ax_p = axes[i * 2]
        ax_v = axes[i * 2 + 1]
        summary = render_tf_panel(ax_p, ax_v, resampled, tf)
        summaries.append(summary)
        # Date formatting on volume axis only (last for each pair)
        if tf == "1w":
            ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        elif tf in ("1d", "4h"):
            ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        else:
            ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax_v.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
        plt.setp(ax_p.xaxis.get_majorticklabels(), visible=False)

    plt.tight_layout()
    return fig, summaries


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timestamp", required=True, help="ISO timestamp, UTC")
    p.add_argument("--tfs", default="30m,1h,4h,1d,1w",
                    help="comma list. Choices: 30m,1h,4h,1d,1w")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ts = pd.Timestamp(datetime.fromisoformat(args.timestamp))
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    tfs = [t.strip() for t in args.tfs.split(",")]
    for t in tfs:
        if t not in TF_RULES:
            raise SystemExit(f"unknown TF: {t}")

    fig, summaries = build_mtf_figure(ts, tfs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=85, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.out}\n")
    print(f"{'TF':<5} {'close':>10} {'trend':<14} {'rsi':>6} {'sar':>10}")
    for s in summaries:
        sar = f"${s.get('sar'):,.0f}" if s.get("sar") else "n/a"
        print(f"{s['tf']:<5} ${s.get('close',0):>9,.0f} {s.get('trend','?'):<14} {s.get('rsi',0):>6.1f} {sar:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
