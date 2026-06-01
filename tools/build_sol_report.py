"""SOL cnh-hybrid-short backtest report (HTML).

- Loads the SOL hybrid-short trade stream (gen_sol_hybrid_trades.py).
- ENFORCES the live bot's 1-position rule (MAX_OPEN_POSITIONS=1): skips any
  trade whose entry falls inside an already-open position. This is the
  "prevent order conflict" step — overlapping signals can't both be taken.
- Simulates equity with risk-based sizing (risk 2.75%, SL=1.5xATR), SOL's real
  exchange minimums ($5 notional / 0.01 qty), friction already in net_pct, and
  the -35.5% kill switch.
- Renders reports/SOL_HYBRID_SHORT_REPORT.html: summary, equity curve, IS/OOS
  table, full trade history.

Run: uv run --with plotly python tools/build_sol_report.py
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
from tools.icnh_mega_sweep import WINDOWS  # noqa: E402

START_EQUITY = 100.0
RISK_PCT = 0.0275
SL_ATR_MULT = 1.5
LEVERAGE = 20
MIN_NOTIONAL, MIN_QTY = 5.0, 0.01
KILL_FRAC = 0.645          # -35.5%
IS_END = pd.Timestamp("2024-06-30", tz="UTC")


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "reports" / "sol_hybrid_short_trades.csv")
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True)
    df["ExitTime"] = pd.to_datetime(df["ExitTime"], utc=True)
    return df.sort_values("EntryTime").reset_index(drop=True)


def enforce_one_position(df: pd.DataFrame):
    """Skip trades that would open while another is still open (1-position rule)."""
    taken, conflicts = [], []
    last_exit = None
    for _, t in df.iterrows():
        if last_exit is not None and t["EntryTime"] < last_exit:
            conflicts.append(t)
            continue
        taken.append(t)
        last_exit = t["ExitTime"]
    return pd.DataFrame(taken).reset_index(drop=True), pd.DataFrame(conflicts)


def simulate(df: pd.DataFrame):
    """Risk-based equity sim with exchange minimums + kill switch."""
    eq = START_EQUITY
    kill_floor = START_EQUITY * KILL_FRAC
    rows, curve = [], [(df["EntryTime"].iloc[0] - pd.Timedelta(days=1), eq)]
    killed = None
    skipped_size = 0
    for _, t in df.iterrows():
        if killed:
            continue
        sl_frac = SL_ATR_MULT * float(t["atr_pct"])
        if not np.isfinite(sl_frac) or sl_frac <= 0:
            continue
        notional = min(RISK_PCT / sl_frac * eq, eq * LEVERAGE * 0.95)
        price = float(t["EntryPrice"]); qty = notional / price
        if qty < MIN_QTY or notional < MIN_NOTIONAL:
            skipped_size += 1
            continue
        pnl = notional * float(t["net_pct"])
        eq += pnl
        curve.append((t["ExitTime"], eq))
        rows.append({
            "EntryTime": str(t["EntryTime"])[:16], "ExitTime": str(t["ExitTime"])[:16],
            "EntryPrice": round(price, 2), "ExitPrice": round(float(t["ExitPrice"]), 2),
            "net_pct": round(float(t["net_pct"]) * 100, 2),
            "notional": round(notional, 1), "pnl": round(pnl, 2),
            "equity": round(eq, 2), "exit_reason": t.get("exit_reason", ""),
        })
        if eq <= kill_floor:
            killed = str(t["ExitTime"])[:10]
    return pd.DataFrame(rows), pd.Series(dict(curve)).sort_index(), killed, skipped_size


def window_of(ts: pd.Timestamp) -> str:
    for label, s, e in WINDOWS:
        if pd.Timestamp(s, tz="UTC") <= ts <= pd.Timestamp(e, tz="UTC") + pd.Timedelta(days=1):
            return label
    return "?"


def main() -> int:
    raw = load_trades()
    clean, conflicts = enforce_one_position(raw)
    trades, curve, killed, skipped_size = simulate(clean)

    nets = trades["net_pct"].values / 100.0
    wr = float((nets > 0).mean() * 100)
    total = (curve.iloc[-1] / START_EQUITY - 1) * 100
    peak = curve.cummax(); max_dd = float((curve / peak - 1).min() * 100)
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / START_EQUITY) ** (1 / yrs) * 100 - 100 if yrs > 0 else 0
    sharpe = float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0

    # IS/OOS split on entry time
    et = pd.to_datetime(clean["EntryTime"])
    is_n = int((et <= IS_END).sum()); oos_n = int((et > IS_END).sum())

    # ---- chart ----
    fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], shared_xaxes=False,
                        vertical_spacing=0.1, subplot_titles=("Equity ($100 start, risk 2.75%)",
                                                              "Per-trade net %"))
    fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name="equity",
                             line=dict(color="#2563eb", width=2)), 1, 1)
    fig.add_hline(y=START_EQUITY * KILL_FRAC, line_dash="dash", line_color="red",
                  annotation_text="kill switch -35.5%", row=1, col=1)
    colors = ["#16a34a" if x > 0 else "#dc2626" for x in nets]
    fig.add_trace(go.Bar(x=pd.to_datetime(trades["ExitTime"]), y=trades["net_pct"],
                         marker_color=colors, name="net %"), 2, 1)
    fig.update_layout(height=620, template="plotly_white", showlegend=False,
                      title=f"SOL cnh-hybrid-short — ${START_EQUITY:.0f}->${curve.iloc[-1]:.0f} "
                            f"({total:+.0f}%) | {len(trades)} trades | WR {wr:.0f}% | maxDD {max_dd:.0f}%")
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    def card(label, val, sub=""):
        return (f"<div style='display:inline-block;min-width:150px;margin:6px;padding:12px 16px;"
                f"border:1px solid #e5e7eb;border-radius:10px'>"
                f"<div style='font-size:12px;color:#6b7280'>{label}</div>"
                f"<div style='font-size:22px;font-weight:700'>{val}</div>"
                f"<div style='font-size:11px;color:#9ca3af'>{sub}</div></div>")

    cards = "".join([
        card("Total return", f"{total:+.0f}%", f"${START_EQUITY:.0f}->${curve.iloc[-1]:.0f}"),
        card("CAGR", f"{cagr:+.1f}%/yr", f"{yrs:.1f} yr"),
        card("Sharpe", f"{sharpe:.2f}", "per-trade, ann."),
        card("Win rate", f"{wr:.0f}%", f"{len(trades)} trades"),
        card("Max drawdown", f"{max_dd:.0f}%", "kill at -35.5%"),
        card("Kill switch", "TRIPPED " + killed if killed else "never tripped", ""),
        card("Order conflicts removed", f"{len(conflicts)}", "1-position rule"),
        card("Sizing skips", f"{skipped_size}", "below SOL $5 min"),
        card("IS / OOS trades", f"{is_n} / {oos_n}", "split 2024-06-30"),
    ])

    th = "background:#f9fafb;text-align:left;padding:6px 10px;border-bottom:1px solid #e5e7eb;font-size:12px"
    td = "padding:5px 10px;border-bottom:1px solid #f3f4f6;font-size:12px;font-variant-numeric:tabular-nums"
    rows_html = "".join(
        f"<tr><td style='{td}'>{r.EntryTime}</td><td style='{td}'>{r.ExitTime}</td>"
        f"<td style='{td}'>{r.EntryPrice}</td><td style='{td}'>{r.ExitPrice}</td>"
        f"<td style='{td};color:{'#16a34a' if r.net_pct>0 else '#dc2626'}'>{r.net_pct:+.2f}%</td>"
        f"<td style='{td}'>${r.notional}</td>"
        f"<td style='{td};color:{'#16a34a' if r.pnl>0 else '#dc2626'}'>{r.pnl:+.2f}</td>"
        f"<td style='{td}'>${r.equity}</td><td style='{td}'>{r.exit_reason}</td></tr>"
        for r in trades.itertuples())

    conflict_note = ""
    if len(conflicts):
        c = conflicts.iloc[0]
        conflict_note = (f"<p style='color:#b45309'>⚠ {len(conflicts)} signal(s) SKIPPED because a "
                         f"position was already open (1-position rule). E.g. entry "
                         f"{str(c['EntryTime'])[:16]} fell inside a prior trade. The live bot "
                         f"(MAX_OPEN_POSITIONS=1) would not take these — so they are excluded.</p>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SOL cnh-hybrid-short backtest</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;margin:24px;color:#111;max-width:1100px}}
h1{{font-size:22px}} h2{{font-size:16px;margin-top:28px}} table{{border-collapse:collapse;width:100%}}</style>
</head><body>
<h1>SOL × cnh-hybrid-short — backtest report</h1>
<p style="color:#6b7280">Generated {str(pd.Timestamp.utcnow())[:16]} UTC · trade source
<code>reports/sol_hybrid_short_trades.csv</code> · 1-position rule enforced ·
friction in net_pct (10bps) · risk 2.75% · SOL minimums $5/0.01.</p>
<div>{cards}</div>
{conflict_note}
<h2>Equity & per-trade</h2>
{chart_html}
<h2>Gate context (validated 2026-05-30)</h2>
<p style="color:#374151">Walk-forward gate: OOS <b>+42.4%</b>, <b>4/4</b> windows positive,
worst window <b>+3.48%</b>. Survives 30bps friction. dedup=15. Corr to BTC short -0.03.
CAVEAT: only {oos_n} OOS trades; SOL's 2022 bear is in-sample, so worst-case drawdown is
NOT OOS-tested. This report's full-history curve mixes the weak in-sample years with the
strong OOS period.</p>
<h2>Full trade history ({len(trades)} taken; {len(conflicts)} conflict-skipped)</h2>
<table><thead><tr>
<th style="{th}">Entry</th><th style="{th}">Exit</th><th style="{th}">EntryPx</th>
<th style="{th}">ExitPx</th><th style="{th}">Net%</th><th style="{th}">Notional</th>
<th style="{th}">PnL$</th><th style="{th}">Equity$</th><th style="{th}">Exit</th></tr></thead>
<tbody>{rows_html}</tbody></table>
</body></html>"""
    out = ROOT / "reports" / "SOL_HYBRID_SHORT_REPORT.html"
    out.write_text(html, encoding="utf-8")
    print(f"raw trades {len(raw)} | conflicts removed {len(conflicts)} | "
          f"sizing skips {skipped_size} | taken {len(trades)}")
    print(f"${START_EQUITY:.0f} -> ${curve.iloc[-1]:.0f} ({total:+.0f}%) | CAGR {cagr:+.1f}% | "
          f"Sharpe {sharpe:.2f} | WR {wr:.0f}% | maxDD {max_dd:.0f}% | kill={killed}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
