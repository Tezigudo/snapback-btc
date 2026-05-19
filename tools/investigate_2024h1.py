"""Investigate the 2024 H1 underperformance.

Generates:
  - reports/2024h1_v1_overlay.png — BTC price with v1 entry/exit markers
  - reports/2024h1_v2_overlay.png — BTC price with v2 entry/exit markers
  - prints monthly breakdown + loss-cluster analysis to stdout
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_btc() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/historical/BTC_USDT_USDT_15m.parquet")
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df.loc["2024-01-01":"2024-07-01"]


def overlay(version: str, trades: list, btc: pd.DataFrame, out_path: Path) -> None:
    daily = btc.resample("1D").agg({"Open": "first", "High": "max",
                                    "Low": "min", "Close": "last"})
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 7),
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(daily.index, daily["Close"], color="#424242", linewidth=1.2, label="BTC daily close")
    ax1.fill_between(daily.index, daily["Low"], daily["High"], color="#9e9e9e", alpha=0.15)

    long_x_in, long_y_in, long_x_out, long_y_out = [], [], [], []
    short_x_in, short_y_in, short_x_out, short_y_out = [], [], [], []
    wins, losses = [], []
    for t in trades:
        ein = pd.Timestamp(t["entry"])
        eout = pd.Timestamp(t["exit"])
        if t["side"] == "LONG":
            long_x_in.append(ein)
            long_y_in.append(t["entry_price"])
            long_x_out.append(eout)
            long_y_out.append(t["exit_price"])
        else:
            short_x_in.append(ein)
            short_y_in.append(t["entry_price"])
            short_x_out.append(eout)
            short_y_out.append(t["exit_price"])
        # connector line, coloured by win/loss
        color = "#2e7d32" if t["pnl_pct"] > 0.05 else ("#c62828" if t["pnl_pct"] < -0.05 else "#9e9e9e")
        ax1.plot([ein, eout], [t["entry_price"], t["exit_price"]],
                 color=color, linewidth=0.7, alpha=0.55)
        if t["pnl_pct"] > 0.05:
            wins.append(t)
        elif t["pnl_pct"] < -0.05:
            losses.append(t)

    ax1.scatter(long_x_in, long_y_in, marker="^", color="#1565c0", s=40, zorder=5,
                label=f"LONG entry ({len(long_x_in)})")
    ax1.scatter(short_x_in, short_y_in, marker="v", color="#d32f2f", s=40, zorder=5,
                label=f"SHORT entry ({len(short_x_in)})")
    ax1.scatter(long_x_out + short_x_out, long_y_out + short_y_out,
                marker="x", color="#424242", s=30, alpha=0.7, zorder=4, label="exit")

    ax1.set_title(f"{version} on 2024 H1 — {len(trades)} trades, "
                  f"{len(wins)} wins / {len(losses)} losses, "
                  f"BTC: $42k → $63k (+50% over period)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylabel("BTC price")
    ax1.grid(True, alpha=0.2)

    # Cumulative equity (start=100) on lower panel
    eq = [100.0]
    eq_ts = [pd.Timestamp("2024-01-01")]
    for t in trades:
        eq.append(eq[-1] * (1 + t["equity_impact_pct"] / 100))
        eq_ts.append(pd.Timestamp(t["exit"]))
    eq_color = "#1b5e20" if eq[-1] > 100 else "#b71c1c"
    ax2.plot(eq_ts, eq, color=eq_color, linewidth=1.4)
    ax2.fill_between(eq_ts, 100, eq, where=[e >= 100 for e in eq], color="#43a047", alpha=0.2)
    ax2.fill_between(eq_ts, 100, eq, where=[e < 100 for e in eq], color="#e53935", alpha=0.2)
    ax2.axhline(100, color="#9e9e9e", linewidth=0.6, linestyle="--", alpha=0.6)
    ax2.set_ylabel(f"equity (end: {eq[-1]:.1f})")
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Annotate ATH and halving
    ath_ts = daily["High"].idxmax()
    ath_price = daily["High"].max()
    ax1.annotate(f"BTC ATH ${ath_price:,.0f}", xy=(ath_ts, ath_price),
                 xytext=(ath_ts, ath_price + 4000),
                 arrowprops=dict(arrowstyle="->", color="#1565c0"),
                 fontsize=9, color="#1565c0")
    halving = pd.Timestamp("2024-04-19")
    ax1.axvline(halving, color="#ff8f00", linewidth=1.2, linestyle="--", alpha=0.6)
    ax1.text(halving, daily["Close"].min(), "  halving",
             fontsize=9, color="#ff8f00", va="bottom")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)


def loss_clusters(trades: list, threshold_consec: int = 3) -> None:
    """Find runs of >= threshold_consec consecutive losses."""
    print(f"\n--- Loss clusters (≥ {threshold_consec} consecutive losers) ---")
    streak = []
    for t in trades:
        if t["pnl_pct"] < -0.05:
            streak.append(t)
        else:
            if len(streak) >= threshold_consec:
                print(f"  {streak[0]['entry']} → {streak[-1]['exit']}  "
                      f"{len(streak)} losses, sum {sum(s['pnl_pct'] for s in streak):+.2f}%, "
                      f"sides: {[s['side'][0] for s in streak]}")
            streak = []
    if len(streak) >= threshold_consec:
        print(f"  {streak[0]['entry']} → {streak[-1]['exit']}  "
              f"{len(streak)} losses, sum {sum(s['pnl_pct'] for s in streak):+.2f}%, "
              f"sides: {[s['side'][0] for s in streak]}")


def main() -> int:
    btc = load_btc()
    print(f"BTC 2024 H1: ${btc['Open'].iloc[0]:,.0f} → ${btc['Close'].iloc[-1]:,.0f}, "
          f"high ${btc['High'].max():,.0f}, low ${btc['Low'].min():,.0f}")

    for version in ("v1", "v2"):
        data = json.loads((ROOT / f"reports/trades_{version}_2024H1.json").read_text())
        trades = data["trades"]
        print(f"\n=== {version}: {len(trades)} trades, "
              f"return {data['return']:+.2f}%, "
              f"wr {data['win_rate_pct']:.1f}%, "
              f"max DD {data['max_drawdown_pct']:+.2f}% ===")

        # Side breakdown
        longs = [t for t in trades if t["side"] == "LONG"]
        shorts = [t for t in trades if t["side"] == "SHORT"]
        print(f"  LONGs:  {len(longs)} ({sum(1 for t in longs if t['pnl_pct']>0)} wins) "
              f"sum {sum(t['pnl_pct'] for t in longs):+.2f}%")
        print(f"  SHORTs: {len(shorts)} ({sum(1 for t in shorts if t['pnl_pct']>0)} wins) "
              f"sum {sum(t['pnl_pct'] for t in shorts):+.2f}%")

        loss_clusters(trades)

        out = ROOT / f"reports/2024h1_{version}_overlay.png"
        overlay(version, trades, btc, out)
        print(f"  → {out.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    main()
