"""Build PATH2_RESULTS.html — the verdict after running v1 + MTF-gate on 5 windows.

Surprise finding: v1 unfiltered (the strategy we ARCHIVED as 'noise')
actually has +55.73% compounded across 5 OOS windows — much stronger
than we knew. The MTF gate REDUCES this to +15.77%. Path 2 fails its
own hypothesis, but the test surfaced the real deployable strategy:
v1 unfiltered itself.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build() -> str:
    raw = json.loads((ROOT / "reports/path2_oos_results.json").read_text())

    v1 = raw["multifactor-v1"]
    mtf = raw["multifactor-mtf"]

    # Compounded
    v1_comp = 1.0
    mtf_comp = 1.0
    for r in v1:
        v1_comp *= (1 + r["ret_pct"] / 100)
    for r in mtf:
        mtf_comp *= (1 + r["ret_pct"] / 100)

    # EV stats per window for v1
    ev_table = []
    for w in ("2022H1", "2023H1", "2024H1", "2024H2", "2025H1"):
        d = json.loads((ROOT / f"reports/trades_v1_{w}.json").read_text())
        trades = d["trades"]
        n = len(trades)
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / n if n else 0
        aw = sum(wins) / len(wins) if wins else 0
        al = sum(losses) / len(losses) if losses else 0
        ev = wr * aw + (1 - wr) * al
        be = -al / (aw - al) if (aw > al and (aw - al) != 0) else 1.0
        margin = wr - be
        ev_table.append({"window": w, "n": n, "wr": wr, "aw": aw, "al": al,
                         "ev": ev, "be": be, "margin": margin})

    parts = []
    parts.append("""<!doctype html><html><head><meta charset="utf-8">
<title>Path 2 — OOS Results & Verdict</title>
<style>
  body { font: 14px/1.55 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1100px; margin: 32px auto; padding: 0 24px; color: #2c2c2c; background: #fafafa; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  h2 { margin-top: 36px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h3 { margin-top: 24px; color: #555; }
  .sub { color: #666; font-style: italic; }
  table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th, td { padding: 6px 12px; border: 1px solid #ddd; text-align: right; }
  th { background: #eee; }
  td.l { text-align: left; }
  .green { color: #1b5e20; font-weight: 600; }
  .red { color: #b71c1c; font-weight: 600; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: 14px 18px; margin: 14px 0; }
  .key { background: #fff8e1; border-left: 4px solid #f57c00; padding: 10px 16px; margin: 14px 0; }
  .verdict { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 12px 16px; margin: 18px 0; }
  .truth { background: #ffebee; border-left: 4px solid #c62828; padding: 12px 16px; margin: 18px 0; }
  .big { font-size: 32px; font-weight: 700; }
  code { background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  pre { background: #f3f3f3; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 12px; }
</style></head><body>""")

    parts.append("<h1>Path 2 — OOS Results & Verdict</h1>")
    parts.append('<p class="sub">Plot twist: the gate failed, but the gate-less strategy is much better than we thought.</p>')

    # ---- Headline ----
    parts.append("""<div class="truth">
<b>Headline:</b> The MTF gate did NOT improve multifactor-v1. It REDUCED 5-window
compounded return from <b>+55.73%</b> (unfiltered) to <b>+15.77%</b> (gated).
The gate killed too many real winners on the strongly-trending years (2022, 2023, 2024H2).
Path 2 as proposed fails its own hypothesis.<br><br>

<b>Bigger surprise:</b> While building the test, I discovered that v1 unfiltered
<b>is actually a strong strategy across multiple windows</b> — not the noise I
called it in ROOT_CAUSE.html. My earlier diagnosis was wrong because I only
tested 2025 H1 (a chop window where v1 IS marginal). Across 5 windows v1 has
4 positive, 1 loss, and ~+20% annualized.
</div>""")

    # ---- The big numbers ----
    parts.append("<h2>1. 5-window OOS test results</h2>")
    parts.append("""<table>
<tr><th>Window</th>
    <th colspan="3" style="background:#e3f2fd">multifactor-v1 (unfiltered)</th>
    <th colspan="3" style="background:#fce4ec">multifactor-v1 + MTF gate</th></tr>
<tr><th></th><th>trades</th><th>return</th><th>max DD</th>
    <th>trades</th><th>return</th><th>max DD</th></tr>""")
    for v, m in zip(v1, mtf):
        v_cls = "green" if v["ret_pct"] > 0 else "red"
        m_cls = "green" if m["ret_pct"] > 0 else "red"
        parts.append(f'<tr><td class="l">{v["window"]}</td>'
                     f'<td>{v["trades"]}</td><td class="{v_cls}">{v["ret_pct"]:+.2f}%</td><td>{v["max_dd_pct"]:+.2f}%</td>'
                     f'<td>{m["trades"]}</td><td class="{m_cls}">{m["ret_pct"]:+.2f}%</td><td>{m["max_dd_pct"]:+.2f}%</td></tr>')
    parts.append(f'<tr style="font-weight:600;background:#f5f5f5">'
                 f'<td class="l">5-window compounded</td>'
                 f'<td>{sum(r["trades"] for r in v1)}</td>'
                 f'<td class="green">{(v1_comp-1)*100:+.2f}%</td><td>—</td>'
                 f'<td>{sum(r["trades"] for r in mtf)}</td>'
                 f'<td>{(mtf_comp-1)*100:+.2f}%</td><td>—</td></tr>')
    parts.append("</table>")

    parts.append("""<div class="key">
<b>Read carefully:</b>
<ul>
<li>v1 unfiltered: <b>4 of 5 windows positive</b>. Lone loss: 2024H1 (-12.56%).</li>
<li>v1 + MTF gate: <b>also 4 of 5 positive</b>. Same lone loss. But all the wins are SMALLER.</li>
<li>The gate didn't prevent the only loss. It just clipped the winners.</li>
<li>v1 unfiltered compounded result: <b class="green">+55.73% over 2.5 years</b> = ~20% annualized.</li>
</ul>
</div>""")

    # ---- Why MTF gate clips winners ----
    parts.append("<h2>2. Why the MTF gate hurts</h2>")
    parts.append("""<p>The MTF analyzer scores trend direction using EMA stack alignment across timeframes.
v1's entry filter <b>already requires close > EMA(200) on the entry TF</b>. So when v1 fires LONG,
the entry TF is already up-trending — the MTF analyzer mostly confirms what v1 already knew.</p>

<p>What MTF UNIQUELY adds: information about whether the trend is also confirmed on 30m/1d/1w.
But on a strong trending year (2022 H1, 2023 H1, 2024 H2), the 30m/1h SAR often shows
short-term retracement = MTF score stays below the +6 threshold = we skip the trade.
And those trades are exactly the highest-quality pullback entries.</p>

<p>In other words: the MTF gate filters out the BEST trades (deep pullbacks in strong trends)
and only keeps the OBVIOUS ones (strong alignment everywhere). The obvious ones are smaller PnL
moves because the easy money is already priced in.</p>""")

    # ---- Per-window EV math ----
    parts.append("<h2>3. v1 per-window EV math (the proof it's not noise)</h2>")
    parts.append("""<p>For each window: win rate, average win, average loss, expected value per trade,
break-even win rate for the realized R:R, and margin over break-even. Earlier I declared
v1 was "+1.7pp margin = noise" — that was true only for 2025 H1. Across windows:</p>""")
    parts.append("""<table>
<tr><th>Window</th><th>n</th><th>WR</th><th>avg win</th><th>avg loss</th><th>EV/trade</th><th>BE WR</th><th>Margin</th><th>Verdict</th></tr>""")
    for r in ev_table:
        m_cls = "green" if r["margin"] > 0.03 else ("red" if r["margin"] < 0 else "")
        verdict = "STRONG" if r["margin"] > 0.05 else ("LOSS" if r["margin"] < 0 else "MARGINAL")
        parts.append(f'<tr><td class="l">{r["window"]}</td><td>{r["n"]}</td>'
                     f'<td>{r["wr"]*100:.1f}%</td>'
                     f'<td>{r["aw"]:+.2f}%</td><td>{r["al"]:+.2f}%</td>'
                     f'<td class="{m_cls}">{r["ev"]:+.3f}%</td><td>{r["be"]*100:.1f}%</td>'
                     f'<td class="{m_cls}">{r["margin"]*100:+.1f}pp</td>'
                     f'<td class="{m_cls}">{verdict}</td></tr>')
    parts.append("</table>")
    parts.append("""<p>3 STRONG, 1 MARGINAL, 1 LOSS. The "noise" was actually the chop window
(2025 H1) being unrepresentative. The strategy has a real but regime-dependent edge.</p>""")

    # ---- Drawdown reality ----
    parts.append("<h2>4. Drawdown reality (what could go wrong)</h2>")
    parts.append("""<table>
<tr><th>Window</th><th>Return</th><th>Max Drawdown</th><th>What it felt like</th></tr>
<tr><td class="l">2022 H1</td><td class="green">+19.29%</td><td>-9.98%</td><td class="l">You're up 30% by April, then a 10% dip, then back up. Manageable.</td></tr>
<tr><td class="l">2023 H1</td><td class="green">+21.83%</td><td>-12.54%</td><td class="l">Strong year but a 12% mid-year dip would test conviction.</td></tr>
<tr><td class="l">2024 H1</td><td class="red">-12.56%</td><td>-14.95%</td><td class="l">This is THE losing window. You'd watch -15% of equity evaporate.</td></tr>
<tr><td class="l">2024 H2</td><td class="green">+21.23%</td><td>-14.23%</td><td class="l">Big year overall but a -14% mid-year dip first.</td></tr>
<tr><td class="l">2025 H1</td><td class="green">+1.09%</td><td>-11.46%</td><td class="l">Net breakeven with a -11% rollercoaster. The chop window.</td></tr>
</table>""")
    parts.append("""<div class="key">
<b>Drawdown profile:</b> -15% is the realistic worst-case for any given 6-month window.
At 20x leverage with risk-based sizing (2% risk per trade / 1.5% SL = ~1.33x effective leverage),
these are real equity drawdowns, not leverage-amplified. Survivable.<br><br>
<b>But:</b> if 2024 H1 had been 2024 full year (could've been worse), you'd be down maybe -25% by July
before any recovery. <b>You must be psychologically prepared for a year of losses</b> before
deploying real money.
</div>""")

    # ---- Decision ----
    parts.append("<h2>5. The recommendation</h2>")
    parts.append("""<div class="verdict">
<b>Deploy multifactor-v1 unfiltered to TESTNET for 90 days of paper trading.</b><br><br>

Rationale:
<ul>
<li>4 of 5 OOS windows positive across 2.5 years (2022 H1 through 2025 H1).</li>
<li>Compounded backtest return: <b class="green">+55.73%</b> (~20% annualized).</li>
<li>Worst single window: -12.56% (2024 H1 chop). Worst rolling drawdown: -14.95%.</li>
<li>Real edge (3 of 5 windows show +5pp margin over break-even win rate).</li>
<li>MTF gate (Path 2's hypothesis) does NOT improve the strategy — skip it.</li>
</ul>

<b>Deploy gates (mandatory):</b>
<ol>
<li><code>multifactor-v1</code> with config: RSI 40 long / RSI 70 short, volume mult 2.0,
    SL 1.5%, TP 3.0%, EMA200 trend filter, funding sentiment gate, max hold 14d.</li>
<li>Testnet only first 90 days (paper trade). Track real outcomes vs backtest predictions.</li>
<li>If 90-day testnet shows return within ±5% of backtest expectation, promote to mainnet small ($1k).</li>
<li>Hard kill switch: if equity drops -15% from start, HALT (touch <code>data/HALT</code>).</li>
<li>Future improvement: build regime detector (EMA-200 1d slope) to pause in chop years. Could turn 2024 H1's -12% into ~0% by sitting out.</li>
</ol>
</div>""")

    # ---- What we built ----
    parts.append("<h2>Appendix: artifacts</h2>")
    parts.append("""<table>
<tr><th>File</th><th>Purpose</th></tr>
<tr><td class="l"><code>strategy/signals_multifactor.py</code></td><td class="l">v1 strategy (restored from archive). The deployable winner.</td></tr>
<tr><td class="l"><code>strategy/signals_multifactor_mtf.py</code></td><td class="l">MTF-gated variant. KEEP for reference but DO NOT deploy — proven to clip winners.</td></tr>
<tr><td class="l"><code>reports/path2_oos_results.json</code></td><td class="l">Full per-window backtest stats for both strategies.</td></tr>
<tr><td class="l"><code>reports/trades_v1_*.json</code></td><td class="l">Per-window v1 trade lists (5 files) for diagnose_trades.py.</td></tr>
</table>""")

    parts.append("""<div class="card">
<b>Lessons logged to memory:</b>
<ul>
<li><code>snapback_v1_actually_works.md</code> — corrects ROOT_CAUSE.html's "noise" claim.</li>
<li>Future Claude sessions will see this and not repeat the chop-only window misdiagnosis.</li>
</ul>
</div>""")

    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    out = ROOT / "PATH2_RESULTS.html"
    out.write_text(build())
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
