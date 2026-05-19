"""Generate per-trade PNG charts for visual diagnosis.

For each trade in a trades JSON, render a 3-panel chart:
  - Top:    price candles (50 bars pre, duration, 20 bars post),
            entry arrow, exit X, SL/TP horizontal lines, EMA(200)
  - Middle: RSI(14) with 30/70 lines
  - Bottom: volume bars with SMA(20)

Saved to: reports/trade_charts/<group>/trade_<n>_<side>_<entry>.png
  group ∈ {winners, losers}

Usage:
  python -m tools.chart_trade reports/trades_multifactor_2025.json \
      --sl-pct 0.015 --tp-pct 0.03 --out-dir reports/trade_charts_2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from strategy.indicators import ema, rsi, sma


def _read_15m() -> pd.DataFrame:
    df = pd.read_parquet("data/historical/BTC_USDT_USDT_15m.parquet")
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI"] = rsi(out["Close"], 14)
    out["EMA200"] = ema(out["Close"], 200)
    out["VolSMA"] = sma(out["Volume"], 20)
    return out


def _slice(df: pd.DataFrame, entry: pd.Timestamp, exit_: pd.Timestamp,
           pre_bars: int = 50, post_bars: int = 20) -> pd.DataFrame:
    # Find closest indices
    entry_idx = df.index.searchsorted(entry)
    exit_idx = df.index.searchsorted(exit_)
    start = max(0, entry_idx - pre_bars)
    end = min(len(df), exit_idx + post_bars)
    return df.iloc[start:end].copy()


def _outcome_label(t: dict, sl_pct: float, tp_pct: float) -> str:
    p = t["pnl_pct"]
    h = t["hold_hours"]
    tol = 0.10
    if abs(p - (-sl_pct * 100)) <= tol:
        return "SL_HIT"
    if abs(p - (tp_pct * 100)) <= tol:
        return "TP_HIT"
    if h >= 14 * 24 - 1:
        return "TIME_STOP"
    return "TREND_FLIP"


def _plot_candle(ax, win: pd.DataFrame) -> None:
    # Simple bar-style candle: thin line for range, fat marker for body color
    for ts, row in win.iterrows():
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax.plot([ts, ts], [row["Low"], row["High"]], color=color, linewidth=0.6, alpha=0.7)
        ax.plot([ts, ts], [row["Open"], row["Close"]], color=color, linewidth=2.0)


def plot_trade(t: dict, df: pd.DataFrame, sl_pct: float, tp_pct: float,
               out_dir: Path) -> Path:
    entry = pd.to_datetime(t["entry"])
    exit_ = pd.to_datetime(t["exit"])
    side = t["side"]
    win = _slice(df, entry, exit_)

    outcome = _outcome_label(t, sl_pct, tp_pct)
    sign = +1 if side == "LONG" else -1
    sl_price = t["entry_price"] * (1 - sign * sl_pct)
    tp_price = t["entry_price"] * (1 + sign * tp_pct)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        f"#{t['n']}  {side}  {outcome}   "
        f"entry {entry:%Y-%m-%d %H:%M} @ ${t['entry_price']:,.0f}   "
        f"exit {exit_:%Y-%m-%d %H:%M} @ ${t['exit_price']:,.0f}   "
        f"PnL {t['pnl_pct']:+.2f}%   held {t['hold_hours']:.1f}h",
        fontsize=10,
    )

    # --- Price panel ---
    ax_p = axes[0]
    _plot_candle(ax_p, win)
    if "EMA200" in win:
        ax_p.plot(win.index, win["EMA200"], color="#9c27b0",
                  linewidth=1.0, linestyle="--", label="EMA(200)")
    # Entry/exit markers
    ax_p.axhline(t["entry_price"], color="#1976d2", linewidth=0.8, alpha=0.5)
    ax_p.axhline(sl_price, color="#ef5350", linewidth=0.8, linestyle=":", alpha=0.7, label=f"SL ({-sl_pct*100:.1f}%)")
    ax_p.axhline(tp_price, color="#26a69a", linewidth=0.8, linestyle=":", alpha=0.7, label=f"TP (+{tp_pct*100:.1f}%)")
    ax_p.axvline(entry, color="#1976d2", linewidth=0.8, alpha=0.6)
    ax_p.axvline(exit_, color="#424242", linewidth=0.8, alpha=0.6)
    marker = "^" if side == "LONG" else "v"
    color = "#1976d2" if side == "LONG" else "#d32f2f"
    ax_p.scatter([entry], [t["entry_price"]], marker=marker, color=color, s=120, zorder=5, label=f"{side} entry")
    ax_p.scatter([exit_], [t["exit_price"]], marker="X", color="#424242", s=80, zorder=5, label="exit")
    ax_p.legend(loc="upper left", fontsize=7)
    ax_p.set_ylabel("BTC price")
    ax_p.grid(True, alpha=0.2)

    # --- RSI panel ---
    ax_r = axes[1]
    ax_r.plot(win.index, win["RSI"], color="#7b1fa2", linewidth=0.8)
    ax_r.axhline(70, color="#ef5350", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_r.axhline(30, color="#26a69a", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_r.axhline(40, color="#ffa726", linewidth=0.5, linestyle=":", alpha=0.4)
    ax_r.axhline(50, color="#9e9e9e", linewidth=0.4, alpha=0.3)
    ax_r.axvline(entry, color="#1976d2", linewidth=0.6, alpha=0.5)
    ax_r.axvline(exit_, color="#424242", linewidth=0.6, alpha=0.5)
    ax_r.set_ylim(0, 100)
    ax_r.set_ylabel("RSI(14)")
    ax_r.grid(True, alpha=0.2)

    # --- Volume panel ---
    ax_v = axes[2]
    colors = ["#26a69a" if c >= o else "#ef5350"
              for o, c in zip(win["Open"], win["Close"])]
    ax_v.bar(win.index, win["Volume"], color=colors, alpha=0.7, width=0.01)
    if "VolSMA" in win:
        ax_v.plot(win.index, win["VolSMA"], color="#1565c0", linewidth=0.8, label="SMA(20)")
        ax_v.plot(win.index, 2.0 * win["VolSMA"], color="#1565c0", linewidth=0.6, linestyle="--", alpha=0.5, label="2× SMA gate")
    ax_v.legend(loc="upper left", fontsize=7)
    ax_v.set_ylabel("Volume")
    ax_v.axvline(entry, color="#1976d2", linewidth=0.6, alpha=0.5)
    ax_v.axvline(exit_, color="#424242", linewidth=0.6, alpha=0.5)
    ax_v.grid(True, alpha=0.2)
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax_v.xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    plt.tight_layout()
    group = "winners" if t["pnl_pct"] > 0 else "losers"
    sub = out_dir / group
    sub.mkdir(parents=True, exist_ok=True)
    fname = f"trade_{t['n']:03d}_{side}_{entry:%Y%m%d_%H%M}_{outcome}.png"
    out = sub / fname
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--sl-pct", type=float, default=0.015)
    p.add_argument("--tp-pct", type=float, default=0.03)
    p.add_argument("--out-dir", type=Path, default=Path("reports/trade_charts"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    data = json.loads(args.path.read_text())
    trades = data["trades"]
    if args.limit:
        trades = trades[: args.limit]

    print("Loading 15m data...")
    df = _enrich(_read_15m())

    print(f"Rendering {len(trades)} trade charts to {args.out_dir}/...")
    for t in trades:
        out = plot_trade(t, df, args.sl_pct, args.tp_pct, args.out_dir)
        print(f"  -> {out.relative_to(args.out_dir.parent)}")
    print(f"\nDone. {len(trades)} charts written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
