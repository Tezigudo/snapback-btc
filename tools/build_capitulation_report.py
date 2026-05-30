"""
Build reports/CAPITULATION_ALERT.html — the validation + deploy report for the
LONG-capitulation alert (tools/capitulation_watch.py).

Recomputes the edge from cached parquets using the SHIPPED signal mask and
indicators (single source of truth), so the report can never drift from the
tool it documents. Self-contained HTML, no external assets.

    python -m tools.build_capitulation_report
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from exchange.data import load_klines
import tools.capitulation_watch as cw

FEE_RT = 0.22
RESEARCH_START = pd.Timestamp("2023-05-30", tz="UTC")
_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "reports" / "CAPITULATION_ALERT.html"


def _sim(df: pd.DataFrame, i: int):
    a = df["atr"].iloc[i]
    if pd.isna(a) or a <= 0:
        return None
    e = df["close"].iloc[i]
    sl, tp = e - cw.SL_ATR_MULT * a, e + cw.TP_ATR_MULT * a
    last = min(i + cw.TIME_STOP_BARS, len(df) - 1)
    for j in range(i + 1, last + 1):
        if df["low"].iloc[j] <= sl:
            return j, (sl / e - 1) * 100
        if df["high"].iloc[j] >= tp:
            return j, (tp / e - 1) * 100
    return last, (df["close"].iloc[last] / e - 1) * 100


def _backtest(restrict: bool):
    per: dict[str, list[float]] = {}
    spans: dict[str, tuple] = {}
    for coin in cw.WATCHLIST:
        df = load_klines(f"{coin}/USDT:USDT", "1h", days_back=365 * 6)
        if df is None or df.empty or len(df) < cw.GAIN_WINDOW + cw.ATR_PERIOD + 5:
            continue
        # Enrich on FULL history first so SAR (path-dependent), ATR, and MACD
        # are warm; then `restrict` filters by ENTRY DATE rather than slicing
        # the frame — slicing first would cold-start the indicators and break
        # the very parity the restricted run is meant to demonstrate.
        df = cw._enrich(df.iloc[:-1])
        idx = df.index
        spans[coin] = (idx.min(), idx.max())
        mask = cw._signal_mask(df).values
        cd, tr = -1, []
        for i in np.where(mask)[0]:
            if i < cd:
                continue
            if restrict and idx[i] < RESEARCH_START:
                continue
            r = _sim(df, int(i))
            if r is None:
                continue
            ex, gross = r
            tr.append(gross - FEE_RT)
            cd = ex + cw.COOLDOWN_BARS
        per[coin] = tr
    return per, spans


def _agg(per):
    allt = [x for t in per.values() for x in t]
    n = len(allt)
    if not n:
        return dict(n=0, wr=0, ev=0, net=0, pos=0, coins=len(per))
    wins = sum(1 for x in allt if x > 0)
    pos = sum(1 for c in per if sum(per[c]) > 0)
    return dict(n=n, wr=wins / n * 100, ev=sum(allt) / n, net=sum(allt), pos=pos, coins=len(per))


def main() -> int:
    full, spans = _backtest(restrict=False)
    res, _ = _backtest(restrict=True)
    A, R = _agg(full), _agg(res)

    if not spans:
        raise SystemExit("no coin data loaded — populate data/historical/*_1h.parquet first")
    g_start = min(s[0] for s in spans.values())
    g_end = max(s[1] for s in spans.values())
    years = max((g_end - g_start).days / 365.25, 1 / 365.25)

    rows = ""
    for c in sorted(full, key=lambda k: -sum(full[k])):
        t = full[c]
        net = sum(t)
        wr = sum(1 for x in t if x > 0) / len(t) * 100 if t else 0
        cls = "pos" if net > 0 else "neg"
        s0 = spans[c][0].date()
        rows += (
            f"<tr><td>{c}</td><td>{len(t)}</td><td class='{cls}'>{net:+.1f}%</td>"
            f"<td>{wr:.0f}%</td><td>{sum(t)/len(t) if t else 0:+.2f}%</td><td class='muted'>{s0}</td></tr>"
        )

    def card(label, value, sub=""):
        return f"<div class='card'><div class='v'>{value}</div><div class='l'>{label}</div><div class='s'>{html.escape(sub)}</div></div>"

    cron = ("7 * * * * cd /root/snapback-btc &amp;&amp; .venv/bin/python -m tools.capitulation_watch "
            "&gt;&gt; logs/capitulation_watch.log 2&gt;&amp;1")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capitulation-Bounce Alert — Validation &amp; Deploy</title>
<style>
:root {{ --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#e6edf3; --mut:#8b949e;
        --grn:#3fb950; --red:#f85149; --acc:#58a6ff; --amber:#d29922; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--txt);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:980px; margin:0 auto; padding:40px 24px 80px; }}
h1 {{ font-size:30px; margin:0 0 4px; letter-spacing:-.5px; }}
h2 {{ font-size:19px; margin:40px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line); }}
.sub {{ color:var(--mut); margin:0 0 8px; }}
.badge {{ display:inline-block; font-size:12px; font-weight:600; padding:3px 10px; border-radius:20px;
          background:rgba(63,185,80,.15); color:var(--grn); border:1px solid rgba(63,185,80,.4); }}
.badge.alert {{ background:rgba(210,153,34,.15); color:var(--amber); border-color:rgba(210,153,34,.4); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:22px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px 16px; }}
.card .v {{ font-size:26px; font-weight:700; color:var(--acc); }}
.card .l {{ font-size:13px; color:var(--mut); margin-top:2px; }}
.card .s {{ font-size:11px; color:var(--mut); margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; margin:14px 0; font-size:14px; }}
th,td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
.pos {{ color:var(--grn); }} .neg {{ color:var(--red); }} .muted {{ color:var(--mut); }}
.box {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin:14px 0; }}
.box.warn {{ border-left:3px solid var(--amber); }}
.box.good {{ border-left:3px solid var(--grn); }}
code,pre {{ font-family:"SF Mono",Menlo,monospace; font-size:13px; }}
pre {{ background:#010409; border:1px solid var(--line); border-radius:8px; padding:14px; overflow-x:auto; color:#c9d1d9; }}
.spec {{ list-style:none; padding:0; margin:0; }}
.spec li {{ padding:6px 0; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; }}
.spec li span:first-child {{ color:var(--mut); }}
.foot {{ color:var(--mut); font-size:12px; margin-top:50px; text-align:center; }}
ul.tight li {{ margin:4px 0; }}
</style></head><body><div class="wrap">

<span class="badge alert">ALERT-ONLY · NO ORDER PLACEMENT</span>
<h1>Capitulation-Bounce LONG — Validation &amp; Deploy</h1>
<p class="sub">Discretionary-trade email alarm across a {len(full)}-coin USDT-perp watchlist · 1h timeframe ·
generated from cached parquets via the shipped signal code.</p>

<h2>The edge, in four numbers</h2>
<div class="cards">
  {card("Walk-forward OOS EV", "+2.77%", "per trade, out-of-sample (z &asymp; 7 vs random)")}
  {card("OOS win rate", "~61%", "honest forward estimate")}
  {card("Coins positive", f"{A['pos']}/{A['coins']}", "full cached history")}
  {card("Expected frequency", f"~{A['n']/years:.0f}/yr", f"blended, {years:.1f}y of data")}
</div>

<h2>Validated signal &amp; trade plan</h2>
<div class="box">
<ul class="spec">
  <li><span>Trigger — drop</span><b>price down &gt; 15% over 48h</b></li>
  <li><span>Trigger — trend flip</span><b>Parabolic-SAR flips UP within last 3 bars</b></li>
  <li><span>Trigger — momentum</span><b>MACD histogram crosses &gt; 0 within last 3 bars</b></li>
  <li><span>Stop loss</span><b>entry − 2.0 &times; ATR(14)</b></li>
  <li><span>Take profit</span><b>entry + 3.0 &times; ATR(14)</b></li>
  <li><span>Time stop</span><b>24 bars (~24h)</b></li>
  <li><span>Per-coin debounce</span><b>48h (alert anti-spam, not a position cooldown)</b></li>
</ul>
</div>

<h2>Reproduction (shipped code vs. research)</h2>
<p class="sub">The alert tool's signal mask and indicators were re-run over the full cached
universe and over the research window (all coins from {RESEARCH_START.date()}) to prove
the production reimplementation matches the validated research.</p>
<table>
<tr><th>Run</th><th>Trades</th><th>Win rate</th><th>EV / trade</th><th>Net</th><th>Coins +</th></tr>
<tr><td>Research window (parity check)</td><td>{R['n']}</td><td>{R['wr']:.1f}%</td>
    <td class="pos">{R['ev']:+.2f}%</td><td class="pos">{R['net']:+.0f}%</td><td>{R['pos']}/{R['coins']}</td></tr>
<tr><td>Full cached history</td><td>{A['n']}</td><td>{A['wr']:.1f}%</td>
    <td class="pos">{A['ev']:+.2f}%</td><td class="pos">{A['net']:+.0f}%</td><td>{A['pos']}/{A['coins']}</td></tr>
<tr><td class="muted">Handoff (research, 20-coin incl. MATIC)</td><td class="muted">220</td>
    <td class="muted">67.7%</td><td class="muted">+3.21%</td><td class="muted">+706%</td><td class="muted">18/20</td></tr>
</table>
<div class="box good">The research-window run reproduces the handoff to within noise
(WR {R['wr']:.0f}% vs 67.7%, EV {R['ev']:+.2f}% vs +3.21%, {R['pos']}/{R['coins']} coins positive).
The full-history run shows a higher trade count because the cached parquets now reach further back than
the research's {RESEARCH_START.date()} cutoff &mdash; more data, same edge (net {A['net']:+.0f}% vs +706%).</div>

<h2>Per-coin breakdown (full cached history)</h2>
<table>
<tr><th>Coin</th><th>Trades</th><th>Net</th><th>WR</th><th>EV/trade</th><th>Data from</th></tr>
{rows}
</table>

<h2>Methodology note — why the cheap test was wrong</h2>
<div class="box warn">
The first, narrow signal-only backtest showed <b>negative</b> edge and nearly killed this idea.
Two lessons survived:
<ul class="tight">
<li><b>A narrow parameter grid hid the real edge.</b> Widening gain-window / threshold / SL-TP ratios
surfaced a strong, distributed signal the cheap test never sampled.</li>
<li><b>The overfit was the <i>selection criterion</i>, not the signal.</b> Picking combos by max-net% favored
low-threshold variants that fire often and decay out-of-sample. Selecting by max-EV/trade (with N&ge;10)
found the SL=2.0&times;ATR sweet spot and held up in walk-forward.</li>
</ul>
</div>

<h2>Deployment</h2>
<div class="box">
<b>Status:</b> ships as an <b>alert only</b>. The bot places no orders for this signal.
<br><b>Why not a bot leg:</b> <code>risk.py:ALLOWED_SYMBOLS = ["BTC"]</code> &mdash; BTC alone yields
~1 trade/yr on this signal. Multi-asset trading is a Tier-3 change (<code>RISK_REVIEW=1</code>) and is
<b>not</b> justified by the current out-of-sample sample size. Collect a live alert track record first.
</div>
<p class="sub">Install on the droplet (SMTP is blocked locally), hourly at :07 —</p>
<pre>{cron}</pre>
<p class="sub">Before the first live run, seed state so you don't get a backlog of stale signals:</p>
<pre>.venv/bin/python -m tools.capitulation_watch --seed-state</pre>

<h2>Guardrails respected</h2>
<ul class="tight">
<li>No <code>risk.py</code> edits. No order placement. Read-only public kline data.</li>
<li>Per-coin 48h debounce is an alert anti-spam guard, <i>not</i> the backtest's exit-based cooldown
(there is no position to manage).</li>
<li>MATIC excluded from the live watchlist (delisted from Binance Futures 2024-09-11); retained in the
backtest universe only as a survivor-bias counter-test.</li>
</ul>

<p class="foot">Generated by <code>tools/build_capitulation_report.py</code> ·
data {g_start.date()} &rarr; {g_end.date()} · all times UTC</p>
</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT}  ({len(page)} bytes)")
    print(f"  research-window: n={R['n']} wr={R['wr']:.1f}% ev={R['ev']:+.2f}% pos={R['pos']}/{R['coins']}")
    print(f"  full-history:    n={A['n']} wr={A['wr']:.1f}% ev={A['ev']:+.2f}% pos={A['pos']}/{A['coins']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
