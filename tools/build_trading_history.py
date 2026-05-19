"""Build TRADING_HISTORY.html — v1 vs v2 side-by-side trade ledger.

For each of 6 OOS windows: stats + equity curve + full trade table for
both strategies, so you can compare what each one did, trade-by-trade.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ["2022H1", "2023H1", "2024H1", "2024H2", "2025H1", "2026Q1"]


def equity_curve_png(trades: list, label: str, color: str = "#1565c0") -> str:
    """Render the TRUE equity curve using equity_impact_pct (leveraged)."""
    fig, ax = plt.subplots(figsize=(7, 2.4))
    equity = [100.0]
    for t in trades:
        equity.append(equity[-1] * (1 + t["equity_impact_pct"] / 100))
    xs = list(range(len(equity)))
    ax.plot(xs, equity, color=color, linewidth=1.4)
    ax.axhline(100, color="#9e9e9e", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.fill_between(xs, 100, equity, where=[e >= 100 for e in equity],
                    color="#43a047", alpha=0.15)
    ax.fill_between(xs, 100, equity, where=[e < 100 for e in equity],
                    color="#e53935", alpha=0.15)
    final = equity[-1] if equity else 100.0
    finc = "#1b5e20" if final > 100 else "#b71c1c"
    ax.set_title(f"{label}: start 100 → end {final:.1f}", fontsize=10, color=finc)
    ax.set_xlabel("trade #")
    ax.set_ylabel("equity")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "wins": 0, "losses": 0, "wr": 0, "compounded": 0,
                "sum_pnl_price": 0, "sum_pnl_equity": 0,
                "avg_win_eq": 0, "avg_loss_eq": 0,
                "best_eq": 0, "worst_eq": 0, "max_dd": 0}
    pnls = [t["pnl_pct"] for t in trades]
    eqimp = [t["equity_impact_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    eq = 100.0
    eqs = [eq]
    for t in trades:
        eq *= (1 + t["equity_impact_pct"] / 100)
        eqs.append(eq)
    peak = eqs[0]
    max_dd = 0.0
    for e in eqs:
        if e > peak:
            peak = e
        dd = (e - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    eqimp_wins = [e for e in eqimp if e > 0]
    eqimp_losses = [e for e in eqimp if e <= 0]
    return {
        "n": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(trades) if trades else 0,
        "compounded": (eq / 100 - 1) * 100,
        "sum_pnl_price": sum(pnls),
        "sum_pnl_equity": sum(eqimp),
        "avg_win_eq": sum(eqimp_wins) / len(eqimp_wins) if eqimp_wins else 0,
        "avg_loss_eq": sum(eqimp_losses) / len(eqimp_losses) if eqimp_losses else 0,
        "best_eq": max(eqimp) if eqimp else 0,
        "worst_eq": min(eqimp) if eqimp else 0,
        "max_dd": max_dd,
    }


def trade_rows(trades: list) -> str:
    """Render a <tbody> worth of trade rows (does NOT include <table>/<thead>)."""
    rows = []
    eq = 100.0
    for t in trades:
        eq *= (1 + t["equity_impact_pct"] / 100)
        cls = "win" if t["pnl_pct"] > 0.05 else ("scratch" if abs(t["pnl_pct"]) <= 0.05 else "lose")
        hold = (f"{t['hold_hours']:.1f}h" if t["hold_hours"] < 24
                else f"{t['hold_hours']/24:.1f}d")
        rows.append(
            f'<tr class="{cls}"><td>{t["n"]}</td><td>{t["side"]}</td>'
            f'<td class="l">{t["entry"]}</td><td class="l">{t["exit"]}</td>'
            f'<td>{hold}</td>'
            f'<td>${t["entry_price"]:,.1f}</td>'
            f'<td>${t["exit_price"]:,.1f}</td>'
            f'<td>{t["pnl_pct"]:+.2f}%</td>'
            f'<td>{t["equity_impact_pct"]:+.2f}%</td>'
            f'<td>${t["pnl_usd"]:+,.0f}</td>'
            f'<td>{eq:.2f}</td></tr>'
        )
    return "".join(rows)


def build() -> str:
    parts = []
    parts.append("""<!doctype html><html><head><meta charset="utf-8">
<title>Trading History — v1 vs v2 side-by-side</title>
<style>
  body { font: 13px/1.5 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1320px; margin: 24px auto; padding: 0 20px; color: #2c2c2c; background: #fafafa; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  h2 { margin-top: 32px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h3 { margin-top: 22px; color: #555; }
  h4 { margin: 14px 0 6px; }
  .sub { color: #666; font-style: italic; font-size: 12px; }

  table.trades { border-collapse: collapse; width: 100%; font-size: 11px;
                  font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  table.trades th, table.trades td { padding: 3px 6px; border: 1px solid #e0e0e0;
                                      text-align: right; white-space: nowrap; }
  table.trades th { background: #eceff1; position: sticky; top: 0; }
  table.trades td.l { text-align: left; }
  table.trades tr.win td { background: #e8f5e9; }
  table.trades tr.lose td { background: #ffebee; }
  table.trades tr.scratch td { background: #fafafa; color: #666; }

  table.summary { border-collapse: collapse; margin: 8px 0; font-size: 13px; width: 100%; }
  table.summary th, table.summary td { padding: 5px 10px; border: 1px solid #ddd; text-align: right; }
  table.summary th { background: #eee; }
  table.summary td.l { text-align: left; font-weight: 600; }

  .green { color: #1b5e20; font-weight: 600; }
  .red { color: #b71c1c; font-weight: 600; }
  .nav { background: #eceff1; padding: 10px 16px; border-radius: 6px; margin: 14px 0; }
  .nav a { margin-right: 16px; color: #1565c0; text-decoration: none; }
  .nav a:hover { text-decoration: underline; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: 12px 16px; margin: 12px 0; }
  .v1-card { border-top: 4px solid #1565c0; }
  .v2-card { border-top: 4px solid #2e7d32; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }
  details { margin: 8px 0; }
  details > summary { cursor: pointer; padding: 6px 10px; background: #f5f5f5;
                       border-radius: 4px; font-weight: 600; }
  details[open] > summary { background: #e3f2fd; }
  code { background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
</style></head><body>""")

    parts.append("<h1>Trading History — v1 vs v2 side-by-side</h1>")

    # Pre-load all data
    win_data: list[dict] = []
    total_v1 = 0
    total_v2 = 0
    for w in WINDOWS:
        v1 = json.loads((ROOT / f"reports/trades_v1_{w}.json").read_text())
        v2 = json.loads((ROOT / f"reports/trades_v2_{w}.json").read_text())
        win_data.append({"window": w, "v1": v1, "v2": v2,
                         "s1": stats(v1["trades"]), "s2": stats(v2["trades"])})
        total_v1 += len(v1["trades"])
        total_v2 += len(v2["trades"])

    parts.append(f'<p class="sub">Every trade for both strategies, across 6 OOS windows. '
                 f'v1: {total_v1} trades. v2 (conf=3): {total_v2} trades. '
                 f'Strategy details in <code>V2_RESULTS.html</code>.</p>')

    parts.append("""<div class="card" style="background:#fff8e1;border-left:4px solid #f57c00">
<b>Column meanings:</b>
<ul>
<li><b>Price PnL %</b> — raw BTC price move (entry → exit), net of commission. NOT leveraged.</li>
<li><b>Equity Δ %</b> — leveraged impact on your account at trade time (~1.33× the price PnL).</li>
<li><b>$ P&amp;L</b> — actual dollar P&amp;L on the trade (backtest used $1M starting cash).</li>
<li><b>Equity</b> — your account compounded within that window, starts at 100.</li>
</ul>
<b>Row colour:</b> green = winner, red = loser, gray = scratch (~flat).
</div>""")

    # Nav
    parts.append('<div class="nav">Jump to: ')
    for w in WINDOWS:
        parts.append(f'<a href="#{w}">{w}</a>')
    parts.append('<a href="#cumulative">Cumulative</a>')
    parts.append('</div>')

    # ---- Cumulative top summary ----
    parts.append('<a id="cumulative"></a>')
    parts.append("<h2>Cumulative across 6 windows</h2>")

    cum_v1 = 100.0
    cum_v2 = 100.0
    cum_rows: list[dict] = []
    for d in win_data:
        cum_v1 *= (1 + d["s1"]["compounded"] / 100)
        cum_v2 *= (1 + d["s2"]["compounded"] / 100)
        cum_rows.append({**d, "cum_v1": cum_v1, "cum_v2": cum_v2})

    parts.append('<table class="summary">')
    parts.append('<tr><th>Window</th>'
                 '<th colspan="3" style="background:#e3f2fd">v1 baseline</th>'
                 '<th colspan="3" style="background:#e8f5e9">v2 (conf=3)</th>'
                 '<th>Δ return</th></tr>')
    parts.append('<tr><th></th><th>trades</th><th>return</th><th>cum eq</th>'
                 '<th>trades</th><th>return</th><th>cum eq</th><th></th></tr>')
    for r in cum_rows:
        s1, s2 = r["s1"], r["s2"]
        ret1, ret2 = s1["compounded"], s2["compounded"]
        delta = ret2 - ret1
        d_cls = "green" if delta > 0 else ("red" if delta < 0 else "")
        c1 = "green" if ret1 > 0 else "red"
        c2 = "green" if ret2 > 0 else "red"
        parts.append(
            f'<tr><td class="l">{r["window"]}</td>'
            f'<td>{s1["n"]}</td><td class="{c1}">{ret1:+.2f}%</td><td>{r["cum_v1"]:.2f}</td>'
            f'<td>{s2["n"]}</td><td class="{c2}">{ret2:+.2f}%</td><td>{r["cum_v2"]:.2f}</td>'
            f'<td class="{d_cls}">{delta:+.2f}%</td></tr>'
        )
    final_v1 = (cum_v1 / 100 - 1) * 100
    final_v2 = (cum_v2 / 100 - 1) * 100
    parts.append('<tr style="background:#f5f5f5;font-weight:600">'
                 f'<td class="l">TOTAL compounded</td>'
                 f'<td>{total_v1}</td><td class="green">{final_v1:+.2f}%</td><td>{cum_v1:.2f}</td>'
                 f'<td>{total_v2}</td><td class="green">{final_v2:+.2f}%</td><td>{cum_v2:.2f}</td>'
                 f'<td class="green">{final_v2-final_v1:+.2f}pp</td></tr>')
    parts.append('</table>')

    # Cumulative equity curves side-by-side
    all_v1 = [t for d in win_data for t in d["v1"]["trades"]]
    all_v2 = [t for d in win_data for t in d["v2"]["trades"]]
    parts.append('<div class="grid2">')
    parts.append(f'<div class="card v1-card"><h4>v1 cumulative equity</h4>'
                 f'<img src="{equity_curve_png(all_v1, "v1 all 6 windows", "#1565c0")}"></div>')
    parts.append(f'<div class="card v2-card"><h4>v2 cumulative equity</h4>'
                 f'<img src="{equity_curve_png(all_v2, "v2 all 6 windows", "#2e7d32")}"></div>')
    parts.append('</div>')

    # ---- Per-window detail ----
    for d in win_data:
        w = d["window"]
        s1, s2 = d["s1"], d["s2"]
        c1 = "green" if s1["compounded"] > 0 else "red"
        c2 = "green" if s2["compounded"] > 0 else "red"
        parts.append(f'<a id="{w}"></a>')
        parts.append(
            f'<h2>{w} — '
            f'v1 <span class="{c1}">{s1["compounded"]:+.2f}%</span> · '
            f'v2 <span class="{c2}">{s2["compounded"]:+.2f}%</span></h2>'
        )

        # Side-by-side stats + equity curves
        parts.append('<div class="grid2">')
        for ver, stats_d, trades, color in [
            ("v1", s1, d["v1"]["trades"], "#1565c0"),
            ("v2", s2, d["v2"]["trades"], "#2e7d32"),
        ]:
            cls = f"{ver}-card"
            cls_color = "green" if stats_d["compounded"] > 0 else "red"
            parts.append(f'<div class="card {cls}">')
            parts.append(f'<h4>{ver}: {stats_d["n"]} trades, '
                         f'<span class="{cls_color}">{stats_d["compounded"]:+.2f}%</span></h4>')
            parts.append('<table class="summary">')
            parts.append(f'<tr><td class="l">Win rate</td><td>{stats_d["wr"]*100:.1f}%</td>'
                         f'<td class="l">Max DD</td><td>{stats_d["max_dd"]:+.2f}%</td></tr>')
            parts.append(f'<tr><td class="l">Avg win (eq)</td>'
                         f'<td class="green">{stats_d["avg_win_eq"]:+.2f}%</td>'
                         f'<td class="l">Avg loss (eq)</td>'
                         f'<td class="red">{stats_d["avg_loss_eq"]:+.2f}%</td></tr>')
            parts.append(f'<tr><td class="l">Best (eq)</td>'
                         f'<td class="green">{stats_d["best_eq"]:+.2f}%</td>'
                         f'<td class="l">Worst (eq)</td>'
                         f'<td class="red">{stats_d["worst_eq"]:+.2f}%</td></tr>')
            parts.append('</table>')
            parts.append(f'<img src="{equity_curve_png(trades, ver + " " + w, color)}">')
            parts.append('</div>')
        parts.append('</div>')

        # Trade tables in collapsible details
        for ver, trades in [("v1", d["v1"]["trades"]), ("v2", d["v2"]["trades"])]:
            n = len(trades)
            parts.append(f'<details><summary>{ver}: all {n} trades (click to expand)</summary>')
            parts.append('<table class="trades">')
            parts.append('<tr><th>#</th><th>Side</th><th>Entry (UTC)</th><th>Exit (UTC)</th>'
                         '<th>Hold</th><th>Entry $</th><th>Exit $</th>'
                         '<th>Price PnL%</th><th>Eq Δ%</th><th>$ P&amp;L</th><th>Eq (start=100)</th></tr>')
            parts.append(trade_rows(trades))
            parts.append('</table></details>')

    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    out = ROOT / "TRADING_HISTORY.html"
    out.write_text(build())
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
