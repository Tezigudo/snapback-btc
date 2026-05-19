"""Build V2_RESULTS.html — v1 vs v2 (TA-confirmation) head-to-head.

Per-window comparison + honest verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ["2022H1", "2023H1", "2024H1", "2024H2", "2025H1", "2026Q1"]


def load_pair(w: str) -> tuple[dict, dict]:
    v1 = json.loads((ROOT / f"reports/trades_v1_{w}.json").read_text())
    v2 = json.loads((ROOT / f"reports/trades_v2_{w}.json").read_text())
    return v1, v2


def load_full_oos() -> dict:
    return json.loads((ROOT / "reports/v2_oos_results.json").read_text())


def build() -> str:
    full = load_full_oos()
    # Compute compounded per-config
    def cum(rows: list) -> float:
        c = 1.0
        for r in rows:
            c *= (1 + r["ret"] / 100)
        return (c - 1) * 100

    cum_v1 = cum(full["v1 baseline"])
    cum_v2_1 = cum(full["v2 conf=1"])
    cum_v2_2 = cum(full["v2 conf=2"])
    cum_v2_3 = cum(full["v2 conf=3"])

    parts = []
    parts.append("""<!doctype html><html><head><meta charset="utf-8">
<title>multifactor-v2 — TA confirmation results</title>
<style>
  body { font: 14px/1.55 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1180px; margin: 24px auto; padding: 0 20px; color: #2c2c2c; background: #fafafa; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  h2 { margin-top: 36px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h3 { margin-top: 22px; color: #555; }
  .sub { color: #666; font-style: italic; }
  table { border-collapse: collapse; margin: 12px 0; font-size: 13px; width: 100%; }
  th, td { padding: 6px 12px; border: 1px solid #ddd; text-align: right; }
  th { background: #eee; }
  td.l { text-align: left; }
  .green { color: #1b5e20; font-weight: 600; }
  .red { color: #b71c1c; font-weight: 600; }
  .neutral { color: #757575; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: 14px 18px; margin: 14px 0; }
  .key { background: #fff8e1; border-left: 4px solid #f57c00; padding: 10px 16px; margin: 14px 0; }
  .verdict { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 12px 16px; margin: 18px 0; }
  .truth { background: #ffebee; border-left: 4px solid #c62828; padding: 12px 16px; margin: 18px 0; }
  code { background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  .big { font-size: 28px; font-weight: 700; }
</style></head><body>""")

    parts.append("<h1>multifactor-v2 — TA confirmation results</h1>")
    parts.append('<p class="sub">v1 vs v2 (trendline + S/R zone + Fibonacci confirmation) across 6 OOS windows.</p>')

    # ---- Headline ----
    parts.append(f"""<div class="verdict">
<b>Headline:</b> v2 with all 3 TA confirmations required improves compounded return from
<b class="big neutral">+{cum_v1:.1f}%</b> to <b class="big green">+{cum_v2_3:.1f}%</b>
(<b>+{cum_v2_3-cum_v1:.1f}pp</b> better) across 2.5 years of OOS data.<br><br>

<b>2026 Q1 specifically — what you asked about:</b> v1 was −0.22% (essentially flat).
v2 is <b class="green">+4.94%</b>, win rate jumped from 23.1% → 36.4%, max drawdown
shrank from −14.99% to −7.92%. The strategy that was "slightly negative" in your
question is now meaningfully positive.
</div>""")

    # ---- The 4 configs ----
    parts.append("<h2>1. Four configs tested (more confirmations = stricter entries)</h2>")
    parts.append("""<p>v2 keeps v1's 4 base filters (RSI, volume, EMA200, funding) and adds 3 TA filters:</p>
<ul>
<li><b>Trendline proximity</b> — for LONG: price within 1.5% of a support trendline drawn through the last 3 swing lows.</li>
<li><b>S/R zone proximity</b> — for LONG: price within 1.0% of a clustered support zone (swing lows that landed within 0.5% of each other).</li>
<li><b>Fibonacci retracement</b> — for LONG: price within 1.0% of a 23.6 / 38.2 / 50 / 61.8 / 78.6% retracement level between the most recent swing high and swing low.</li>
</ul>
<p>SHORT entry is the mirror (resistance lines, resistance zones, downtrend retracements).
<code>confirmations_required</code> sets how many of the 3 must agree:</p>""")

    parts.append('<table>')
    parts.append('<tr><th>Config</th><th>Description</th><th>6-window compounded</th><th>vs v1</th></tr>')
    parts.append(f'<tr><td class="l"><b>v1 baseline</b></td>'
                 f'<td class="l">4 filters only (RSI, volume, EMA, funding)</td>'
                 f'<td class="neutral">{cum_v1:+.2f}%</td><td>—</td></tr>')
    parts.append(f'<tr><td class="l">v2 conf=1</td>'
                 f'<td class="l">v1 + at least 1 of 3 TA filters</td>'
                 f'<td>{cum_v2_1:+.2f}%</td>'
                 f'<td class="{"green" if cum_v2_1 > cum_v1 else "red"}">{cum_v2_1-cum_v1:+.2f}pp</td></tr>')
    parts.append(f'<tr><td class="l">v2 conf=2</td>'
                 f'<td class="l">v1 + at least 2 of 3 TA filters</td>'
                 f'<td class="green">{cum_v2_2:+.2f}%</td>'
                 f'<td class="green">{cum_v2_2-cum_v1:+.2f}pp</td></tr>')
    parts.append(f'<tr style="background:#e8f5e9"><td class="l"><b>v2 conf=3 (strict, recommended)</b></td>'
                 f'<td class="l">v1 + ALL 3 TA filters</td>'
                 f'<td class="green"><b>{cum_v2_3:+.2f}%</b></td>'
                 f'<td class="green"><b>{cum_v2_3-cum_v1:+.2f}pp</b></td></tr>')
    parts.append('</table>')

    parts.append("""<div class="key">
<b>Read this:</b> more confirmations = better return AND fewer trades. The trade-off is
"more conviction per trade, but fewer chances to print." At $60 capital this matters
because exchange minimums already skip many signals.
</div>""")

    # ---- Per-window detail ----
    parts.append("<h2>2. Per-window head-to-head: v1 vs v2 conf=3</h2>")
    parts.append('<table>')
    parts.append('<tr><th rowspan="2">Window</th><th colspan="4" style="background:#e3f2fd">v1 baseline</th>'
                 '<th colspan="4" style="background:#e8f5e9">v2 (conf=3)</th><th rowspan="2">Δreturn</th></tr>')
    parts.append('<tr><th>trades</th><th>return</th><th>WR</th><th>max DD</th>'
                 '<th>trades</th><th>return</th><th>WR</th><th>max DD</th></tr>')
    for w in WINDOWS:
        a, b = load_pair(w)
        a_ret = a["return"]
        b_ret = b["return"]
        delta = b_ret - a_ret
        a_cls = "green" if a_ret > 0 else "red"
        b_cls = "green" if b_ret > 0 else "red"
        d_cls = "green" if delta > 0 else ("red" if delta < 0 else "neutral")
        parts.append(f'<tr><td class="l"><b>{w}</b></td>'
                     f'<td>{len(a["trades"])}</td><td class="{a_cls}">{a_ret:+.2f}%</td>'
                     f'<td>{a["win_rate_pct"]:.1f}%</td><td>{a["max_drawdown_pct"]:+.2f}%</td>'
                     f'<td>{len(b["trades"])}</td><td class="{b_cls}">{b_ret:+.2f}%</td>'
                     f'<td>{b["win_rate_pct"]:.1f}%</td><td>{b["max_drawdown_pct"]:+.2f}%</td>'
                     f'<td class="{d_cls}">{delta:+.2f}%</td></tr>')
    parts.append(f'<tr style="background:#f5f5f5;font-weight:600">'
                 f'<td class="l">COMPOUND</td>'
                 f'<td colspan="4" class="neutral">{cum_v1:+.2f}%</td>'
                 f'<td colspan="4" class="green">{cum_v2_3:+.2f}%</td>'
                 f'<td class="green">{cum_v2_3-cum_v1:+.2f}pp</td></tr>')
    parts.append('</table>')

    # ---- Honest caveats ----
    parts.append("<h2>3. Where v2 LOSES (the honest part)</h2>")
    parts.append("""<div class="truth">
v2 is not strictly better on every window. Two cases where v1 beats v2:
<ul>
<li><b>2022 H1:</b> v1 +19.29% vs v2 +14.74% (−4.5pp). v2's stricter filters
rejected some of v1's good entries.</li>
<li><b>2024 H2:</b> v1 +21.23% vs v2 +4.60% (<b>−16.6pp</b>). This is the major regression.
2024 H2 was a strong, smooth post-halving uptrend. v1's "dip in uptrend" entries fired often
and printed. v2 required price to also be near a support trendline / S/R zone / Fib level —
but in a steady uptrend, dips don't go deep enough to hit those classical-TA levels.
v2 sat out most of the action.</li>
</ul>

<b>Net effect:</b> v2's huge wins on 2023 H1 (+27pp), 2025 H1 (+6pp), and 2026 Q1 (+5pp)
more than offset the 2024 H2 underperformance. But the strategy character has changed.
v2 trades smarter in chop and rangier markets, less aggressively in clean trends.
</div>""")

    # ---- Trade-frequency reality ----
    parts.append("<h2>4. Trade frequency at $60 (deploy reality)</h2>")
    parts.append("""<p>With Binance minimums (0.001 BTC qty, $50 notional), the actual deployed
trade count is lower than the backtest count. The deeper question: <b>does the bot fire often
enough at $60 to matter?</b></p>""")
    parts.append('<table>')
    parts.append('<tr><th>Window</th><th>v1 backtest trades</th><th>v2 conf=3 backtest trades</th>'
                 '<th>v2 trades/month avg</th></tr>')
    for w in WINDOWS:
        a, b = load_pair(w)
        months = 6.0 if w != "2026Q1" else 3.0
        parts.append(f'<tr><td class="l">{w}</td><td>{len(a["trades"])}</td>'
                     f'<td>{len(b["trades"])}</td>'
                     f'<td>{len(b["trades"])/months:.1f}</td></tr>')
    parts.append('</table>')
    parts.append("""<p><b>Rule of thumb:</b> at $60, expect roughly half of those v2 trades to
actually execute (the other half get skipped by exchange minimums). So 2-6 real trades per
month. Tight, but tradeable.</p>""")

    # ---- Recommendation ----
    parts.append("<h2>5. Recommendation</h2>")
    parts.append("""<div class="verdict">
<b>Deploy multifactor-v2-strict (confirmations_required=3).</b><br><br>

Evidence:
<ul>
<li>5 of 6 OOS windows positive (vs 4 of 6 for v1)</li>
<li>Compounded return +82.66% vs v1's +55.40% across 2022–2026 Q1</li>
<li>Win rates LIFTED across the board: 23–44% (v2) vs 23–32% (v1)</li>
<li>Drawdowns SMALLER: worst window v2 −12.51% vs v1 −14.99%</li>
<li>2026 Q1 specifically: from −0.22% loss → +4.94% win</li>
</ul>

<b>Caveats:</b>
<ul>
<li>v2 trades less in strong trends (2024 H2 was −16.6pp vs v1)</li>
<li>Thresholds (1.5% trendline, 1.0% S/R, 1.0% fib) were chosen by intuition, not validated
by sweep. Forward-test will tell us if they generalize.</li>
<li>The TA filters are 3 different ways of saying "near support / resistance" — partly
correlated, so requiring all 3 is closer to "really near a level" than "3 independent checks."</li>
</ul>

<b>How to deploy v2:</b><br>
Edit <code>config/params.yaml</code>:
<pre>strategy_name: multifactor-v2-strict
strategy:
  ...
  confirmations_required: 3</pre>
Then follow the existing DEPLOY.md flow (preflight → dry-run → live).
The kill switch and all safety gates apply identically to v2.
</div>""")

    parts.append("""<div class="key">
<b>What I'd actually do if it were my money:</b><br>
Run v2-strict in dry-run for the first 3-5 days alongside v1 dry-run. Compare WHICH signals
each strategy fires on. If v2 looks reasonable in live data (not just backtest), switch live
deploy to v2. If v2 looks suspiciously sparse or fires in weird places, fall back to v1.
The 27pp backtest improvement is genuine but I don't trust it ALL the way until I see it in
live data for a week.
</div>""")

    # ---- Files ----
    parts.append("<h2>Appendix: what's in the repo</h2>")
    parts.append("""<ul>
<li><code>strategy/signals_multifactor_v2.py</code> — v2 strategy class (loose + strict variants)</li>
<li><code>strategy/indicators.py</code> — added <code>trendline_proximity_pct</code>,
<code>sr_zones</code>, <code>nearest_sr_zone_distance_pct</code>,
<code>fib_retracement_distance_pct</code>, <code>recent_swing_pair</code></li>
<li><code>backtest.py</code> — registered <code>multifactor-v2-loose</code> and
<code>multifactor-v2-strict</code></li>
<li><code>reports/v2_oos_results.json</code> — full 4-config × 6-window matrix</li>
<li><code>reports/trades_v2_*.json</code> — per-window trade lists for v2 conf=3</li>
</ul>""")

    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    out = ROOT / "V2_RESULTS.html"
    out.write_text(build())
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
