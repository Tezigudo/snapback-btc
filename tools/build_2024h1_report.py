"""Build INVESTIGATE_2024H1.html — what went wrong in the lone losing window.

Embeds the BTC-overlay charts produced by tools/investigate_2024h1.py.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def b64png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build() -> str:
    v1 = json.loads((ROOT / "reports/trades_v1_2024H1.json").read_text())
    v2 = json.loads((ROOT / "reports/trades_v2_2024H1.json").read_text())

    img_v1 = b64png(ROOT / "reports/2024h1_v1_overlay.png")
    img_v2 = b64png(ROOT / "reports/2024h1_v2_overlay.png")

    out = """<!doctype html><html><head><meta charset="utf-8">
<title>2024 H1 investigation — what went wrong</title>
<style>
  body { font: 14px/1.55 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1200px; margin: 24px auto; padding: 0 20px; color: #2c2c2c; background: #fafafa; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  h2 { margin-top: 32px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h3 { margin-top: 22px; color: #555; }
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
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 6px 0; }
  code { background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
</style></head><body>
<h1>2024 H1 — what went wrong?</h1>
<p class="sub">The only window where both v1 and v2 lost money. The post-mortem.</p>
"""

    # Headline
    out += """<div class="truth">
<b>The result:</b> v1 lost <b>−12.56%</b> (47 trades, 23.4% wr).
v2 lost <b>−9.29%</b> (25 trades, 20.0% wr). Both well below break-even WR.
<br><br>
<b>The irony:</b> BTC went from <b>$42,314 → $62,884 (+50% over the period)</b>
with an ATH of $73,881. The strategies are dip-buyers in uptrends — and BTC
was in an uptrend. So why did they lose?
</div>"""

    # Macro context
    out += """<h2>1. BTC's actual price action in 2024 H1</h2>
<table>
<tr><th>Month</th><th>Open → Close</th><th>Range</th><th>What happened</th></tr>
<tr><td class="l">Jan</td><td>$42,314 → $42,560 (+0.6%)</td><td>$38.5k–$49.0k</td>
    <td class="l">ETF approval Jan 11 pumped to $48k, dump to $39k. Whipsaw.</td></tr>
<tr><td class="l">Feb</td><td>$42,560 → $61,203 (<span class="green">+43.8%</span>)</td><td>$41.9k–$64.3k</td>
    <td class="l">ETF inflows. <b>Parabolic rally</b>.</td></tr>
<tr><td class="l">Mar</td><td>$61,203 → $71,363 (<span class="green">+16.6%</span>)</td><td>$59.1k–$73.9k</td>
    <td class="l">ATH $73,881 on Mar 14. Choppy topping process all month.</td></tr>
<tr><td class="l">Apr</td><td>$71,363 → $60,651 (<span class="red">−15.0%</span>)</td><td>$59.2k–$72.9k</td>
    <td class="l"><b>Halving Apr 19-20.</b> Post-top correction. Steep down.</td></tr>
<tr><td class="l">May</td><td>$60,651 → $67,578 (<span class="green">+11.4%</span>)</td><td>$56.5k–$72.1k</td>
    <td class="l">Volatile bounce attempts, no follow-through.</td></tr>
<tr><td class="l">Jun</td><td>$67,578 → $62,766 (<span class="red">−7.1%</span>)</td><td>$58.2k–$72.1k</td>
    <td class="l">Slow grind down. Chop continues.</td></tr>
</table>"""

    # Trade overlays
    out += """<h2>2. Every trade overlaid on the BTC chart</h2>
<p>The Y-axis on top is BTC price. Each line is a trade — green if it printed,
red if it lost, gray if it scratched. The bottom panel shows compounded equity (start=100).
Orange dashed line = halving date (Apr 19, 2024).</p>"""

    out += f"""<h3>v1 baseline (47 trades, ends at 87.4)</h3>
<img src="{img_v1}">"""

    out += f"""<h3>v2 (conf=3) (25 trades, ends at 90.7)</h3>
<img src="{img_v2}">"""

    # The bleed
    out += """<h2>3. The loss clusters</h2>
<p>Both versions had several streaks of 4–8 consecutive losing LONGs. Where and when:</p>

<div class="card">
<b>v1 loss clusters (≥3 consecutive losers):</b>
<table>
<tr><th>Window</th><th>n losses</th><th>Sum</th><th>Sides</th><th>Context</th></tr>
<tr><td>Feb 13 – Mar 5</td><td>6</td><td class="red">−5.27%</td><td>LLLLLL</td>
    <td class="l">Buying every dip during the +44% Feb rally. Each dip kept going. Trend was UP but pullback timing was wrong.</td></tr>
<tr><td>Mar 12 – Mar 23</td><td>3</td><td class="red">−3.30%</td><td>LLS</td>
    <td class="l">Buying near the ATH. Tops form by repeated dips that look like RSI buys but the rally is exhausted.</td></tr>
<tr><td>Apr 9 – Apr 21</td><td>6</td><td class="red">−2.89%</td><td>LLLLLL</td>
    <td class="l">Buying into the −15% post-ATH correction. Every dip was a deeper dip until halving.</td></tr>
<tr><td>May 6 – May 16</td><td>3</td><td class="red">−3.18%</td><td>LLL</td>
    <td class="l">Buying into post-halving sell-off.</td></tr>
<tr><td>May 21 – Jun 4</td><td>5</td><td class="red">−3.39%</td><td>LLLLL</td>
    <td class="l">Buying dips during chop. No trend follow-through.</td></tr>
<tr><td>Jun 9 – Jun 25</td><td>5</td><td class="red">−3.05%</td><td>SSSLS</td>
    <td class="l">Trying both directions. Got chopped both ways.</td></tr>
</table>
</div>

<div class="card">
<b>v2 loss clusters (≥3 consecutive losers):</b>
<table>
<tr><th>Window</th><th>n losses</th><th>Sum</th><th>Sides</th><th>Context</th></tr>
<tr><td>Feb 3 – Mar 23</td><td>8</td><td class="red">−6.74%</td><td>LLLLLLLS</td>
    <td class="l">v2's TA filters DID confirm "near support" — but in a parabolic move,
    every support break-and-retest becomes the next leg up <i>after</i> hitting the SL.</td></tr>
<tr><td>Mar 26 – Apr 20</td><td>4</td><td class="red">−3.46%</td><td>LLLL</td>
    <td class="l">Same as v1: buying into the post-ATH correction. v2's S/R levels were broken by the move.</td></tr>
</table>
</div>"""

    # Side breakdown
    longs_v1 = [t for t in v1["trades"] if t["side"] == "LONG"]
    shorts_v1 = [t for t in v1["trades"] if t["side"] == "SHORT"]
    longs_v2 = [t for t in v2["trades"] if t["side"] == "LONG"]
    shorts_v2 = [t for t in v2["trades"] if t["side"] == "SHORT"]
    sum_long_v1 = sum(t["pnl_pct"] for t in longs_v1)
    sum_short_v1 = sum(t["pnl_pct"] for t in shorts_v1)
    sum_long_v2 = sum(t["pnl_pct"] for t in longs_v2)
    sum_short_v2 = sum(t["pnl_pct"] for t in shorts_v2)

    out += f"""<h2>4. LONGs vs SHORTs</h2>
<table>
<tr><th>Version</th><th>LONGs n / sum%</th><th>SHORTs n / sum%</th></tr>
<tr><td class="l">v1</td>
    <td>{len(longs_v1)} / <span class="red">{sum_long_v1:+.2f}%</span></td>
    <td>{len(shorts_v1)} / <span class="red">{sum_short_v1:+.2f}%</span></td></tr>
<tr><td class="l">v2 conf=3</td>
    <td>{len(longs_v2)} / <span class="red">{sum_long_v2:+.2f}%</span></td>
    <td>{len(shorts_v2)} / <span class="red">{sum_short_v2:+.2f}%</span></td></tr>
</table>
<p>Both versions lost on BOTH sides, but the long bleed was 6–7× the short bleed.
The strategies were biased long (BTC was in an uptrend by EMA200) but the timing was wrong.</p>"""

    # Root causes
    out += """<h2>5. Root causes — why did dip-buying fail in a +50% window?</h2>

<div class="card">
<b>Cause 1: Parabolic moves have shallow pullbacks that LOOK like normal dips.</b><br>
In Feb's +44% rally, BTC dipped 4–6% repeatedly. RSI(14) crossed below 40 on each dip.
Volume was elevated (it's a parabolic move). EMA(200) was clearly below price.
All 4 v1 filters fired. Bot bought.<br>
But 1.5% SL is tighter than typical pullback noise during high-vol rallies — price kept
dipping past the SL before bouncing. Bot got stopped, then watched the next leg up.
</div>

<div class="card">
<b>Cause 2: Tops don't ring a bell. RSI tells you "oversold relative to recent" — not
"this is the bottom".</b><br>
Mar 12–23 cluster: BTC dipped from $73k ATH to $63k over 10 days. Each dip looked
"oversold" by 15m RSI. But the broader trend was rolling over. The 1h+15m setup
can't see the macro topping process.
</div>

<div class="card">
<b>Cause 3: EMA(200) is a coarse trend filter.</b><br>
Throughout 2024 H1, BTC was 30–60% ABOVE the EMA(200). The "close > EMA(200)" check
provided no margin-of-safety information — at $73k ATH, EMA(200) was around $50k. A
20% correction from ATH still leaves you above EMA. So the filter said "uptrend OK"
during the very correction that wiped out the strategy.
</div>

<div class="card">
<b>Cause 4: Post-halving chop violated mean-reversion.</b><br>
May–June 2024 had no trend. BTC oscillated $58k–$72k. Strategy's mean-reversion entries
on either side failed to find follow-through. v2's TA filters helped (smaller loss
than v1) but couldn't generate a true edge in noise.
</div>"""

    # Fixes
    out += """<h2>6. What could fix it (NOT implemented yet — just analysis)</h2>

<table>
<tr><th>Idea</th><th>Mechanism</th><th>Trade-off</th></tr>

<tr><td class="l"><b>Distance-from-EMA filter</b></td>
<td class="l">Refuse to buy if close > X% above EMA(200). E.g. block longs when
price is >20% above EMA(200).</td>
<td class="l">Would have skipped the late-Feb/early-Mar tops. Also would have
skipped some of 2023 H1's biggest wins. Likely a moderate net positive.</td></tr>

<tr><td class="l"><b>ATR-based stops</b></td>
<td class="l">Replace fixed 1.5% SL with k × ATR(14). Stops widen in high-vol periods.</td>
<td class="l">Fewer stops triggered in parabolic moves. But also bigger losses per
stop. Net unclear without backtest.</td></tr>

<tr><td class="l"><b>Vol-regime gate</b></td>
<td class="l">Pause entries when 30d ATR percentile > 85th. "Vol too high, sit out."</td>
<td class="l">Would have skipped the Feb–Mar parabolic move entirely. Misses some
wins. Likely small improvement.</td></tr>

<tr><td class="l"><b>Halving / event blackouts</b></td>
<td class="l">Hard-code "pause N days around Bitcoin halving".</td>
<td class="l">Avoid the Apr cluster specifically, but overfit to known event.</td></tr>

<tr><td class="l"><b>Time-of-month patterns</b></td>
<td class="l">In rapid-rally months, only trade first or last week.</td>
<td class="l">Pure overfitting risk. Don't.</td></tr>
</table>"""

    # Verdict
    out += """<h2>7. The honest verdict</h2>

<div class="verdict">
<b>2024 H1 wasn't a strategy failure — it was a regime failure.</b> The market made
+50% in 6 months and our dip-buyer lost 9–13%. That's a tells-us-something result.<br><br>

The strategy is designed for "<b>steady uptrend with normal pullbacks</b>" (think 2022 H1,
2023 H1, 2024 H2). In those regimes it prints. 2024 H1 had two specific failure modes
the strategy doesn't handle:
<ul>
<li><b>Feb–Mar:</b> parabolic move where every "dip" is a brief continuation, not a real
mean-reversion opportunity.</li>
<li><b>Apr–Jun:</b> topping process + chop, where mean-reversion has no anchor.</li>
</ul>

<b>What the data says you SHOULDN'T do:</b> tweak parameters to "fix" 2024 H1
specifically. That's overfitting. The strategy got −9% to −13% in its worst regime out
of 6 windows. Across all 6 it still compounds to +55% (v1) / +82% (v2). That's the
honest profile of this strategy class.<br><br>

<b>What you COULD do:</b> add a regime detector that pauses the bot during conditions
this strategy historically fails — high-vol parabolic rallies AND topping/chop after
ATH. That's a v3 conversation, not a parameter tweak.
</div>

<div class="key">
<b>For your live $60 deploy:</b> 2024 H1 is the worst case. If you've sized the kill
switch (−18%) to survive it, you can deploy through whatever comes next. If you
can't stomach a 2024-H1-like 6-month run, this strategy isn't right for you.
</div>

</body></html>
"""
    return out


if __name__ == "__main__":
    p = ROOT / "INVESTIGATE_2024H1.html"
    p.write_text(build())
    print(f"Wrote {p} ({p.stat().st_size // 1024} KB)")
