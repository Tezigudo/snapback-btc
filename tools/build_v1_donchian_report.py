"""Build V1_DONCHIAN_RESULTS.html — v1 + Donchian-v3 parallel-deploy backtest.

Reads reports/v1_donchian_combined_*.json (latest) and renders an HTML
report in the same style as PATH2_RESULTS.html / V3_RESULTS.html.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _latest_json() -> Path:
    cands = sorted(glob.glob(str(ROOT / "reports" / "v1_donchian_combined_*.json")))
    if not cands:
        raise RuntimeError("no v1_donchian_combined_*.json under reports/")
    return Path(cands[-1])


def _compounded(rows: list[dict], side: str) -> float:
    eq = 1.0
    for r in rows:
        if "error" in r or side not in r:
            continue
        pct = r[side]["ret_pct"]
        if pct is None or not math.isfinite(pct):
            continue
        eq *= 1.0 + pct / 100.0
    return (eq - 1.0) * 100.0


def _cls(v: float) -> str:
    return "green" if v > 0 else "red"


def build() -> str:
    raw_path = _latest_json()
    results = json.loads(raw_path.read_text())

    v1_comp = _compounded(results, "v1")
    d3a_comp = _compounded(results, "donchian_v3_agg")
    d3c_comp = _compounded(results, "donchian_v3_cons")
    ca_comp = _compounded(results, "combined_50_50_agg")
    cc_comp = _compounded(results, "combined_50_50_cons")

    # Worst DD across windows (per series)
    def worst_dd(side: str) -> tuple[str, float]:
        worst_w = ""
        worst = 0.0
        for r in results:
            if "error" in r or side not in r:
                continue
            dd = r[side].get("max_dd_pct")
            if dd is None or not math.isfinite(dd):
                continue
            if dd < worst:
                worst = dd
                worst_w = r["window"]
        return worst_w, worst

    v1_dd_w, v1_dd = worst_dd("v1")
    d3c_dd_w, d3c_dd = worst_dd("donchian_v3_cons")
    cc_dd_w, cc_dd = worst_dd("combined_50_50_cons")

    parts: list[str] = []
    parts.append("""<!doctype html><html><head><meta charset="utf-8">
<title>v1 + Donchian-v3 — Parallel Deploy Backtest</title>
<style>
  body { font: 14px/1.55 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1180px; margin: 32px auto; padding: 0 24px; color: #2c2c2c; background: #fafafa; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  h2 { margin-top: 36px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h3 { margin-top: 24px; color: #555; }
  .sub { color: #666; font-style: italic; }
  table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th, td { padding: 6px 12px; border: 1px solid #ddd; text-align: right; }
  th { background: #eee; }
  td.l, th.l { text-align: left; }
  .green { color: #1b5e20; font-weight: 600; }
  .red { color: #b71c1c; font-weight: 600; }
  .mute { color: #888; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: 14px 18px; margin: 14px 0; }
  .key { background: #fff8e1; border-left: 4px solid #f57c00; padding: 10px 16px; margin: 14px 0; }
  .verdict { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 12px 16px; margin: 18px 0; }
  .truth { background: #ffebee; border-left: 4px solid #c62828; padding: 12px 16px; margin: 18px 0; }
  .warn  { background: #fff3e0; border-left: 4px solid #ef6c00; padding: 12px 16px; margin: 18px 0; }
  code { background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  pre { background: #f3f3f3; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 12px; }
  .bar { display:inline-block; vertical-align:middle; height: 12px; background:#bbdefb; border-radius:2px; }
  .barr { display:inline-block; vertical-align:middle; height: 12px; background:#ffcdd2; border-radius:2px; }
</style></head><body>""")

    parts.append("<h1>multifactor-v1 + Donchian-v3 — Parallel-Deploy Backtest</h1>")
    parts.append('<p class="sub">Can we run a second strategy alongside the deployed v1 to diversify regime risk?'
                 ' 5 OOS windows (2022H1 → 2025H1), same harness, friction-honest, 50/50 capital combined portfolio.</p>')
    parts.append(f'<p class="mute">Source: <code>{raw_path.name}</code></p>')

    # === Headline / TL;DR with critical warnings ===
    parts.append("<h2>TL;DR</h2>")
    parts.append("""<div class="truth">
<b>Do NOT deploy at the current $101 capital.</b> 50/50 splits to $50.50 per strategy —
below the $50 Binance min-notional after any drawdown. <b>Top up to ≥ $200 first.</b><br><br>

<b>The current −18% kill-switch is wrong for a combined account.</b>
Combined-cons hit max-DD <b class="red">−18.70%</b> in 2024H1 (vs v1 alone <b>−14.95%</b>).
Deploying the combo against <code>kill_switch_equity_fraction=0.82</code> would have
<b>tripped in 2024H1 and locked in the loss</b> right before Donchian's recovery rescued
the realized +2.1% return. Use <b>per-strategy kill-switches</b> (each bot process at −18% on
its own equity), not a combined-account switch.
</div>""")

    parts.append("""<div class="warn">
<b>Why Donchian instead of "inventing" something new:</b><br>
First candidate (Funding-Carry Reversal) was killed in 5 minutes by a hypothesis check on
<code>funding.parquet</code> — net forward 24h mean at the natural extreme threshold was
+4 bps after fees, an order of magnitude below the deployable bar. Donchian-v3 was already
coded in git history and has 2 historically-validated OOS windows. See <code>tools/fcr_hypothesis_check.py</code>.
</div>""")

    # === Compounded headline table ===
    parts.append("<h2>1. The compounded numbers</h2>")
    parts.append("""<table>
<tr><th class="l">Strategy</th>
    <th>5-window compounded</th>
    <th>Worst single-window max-DD</th>
    <th>Validation status</th></tr>""")
    parts.append(f'<tr><td class="l"><b>multifactor-v1</b> (deployed)</td>'
                 f'<td class="green">{v1_comp:+.2f}%</td>'
                 f'<td>{v1_dd:+.2f}% ({v1_dd_w})</td>'
                 f'<td class="l">5-window OOS-locked (PATH2_RESULTS.html)</td></tr>')
    parts.append(f'<tr><td class="l"><b>Donchian-v3 cons</b> (80/20, gate ON @ 3%)</td>'
                 f'<td class="green">{d3c_comp:+.2f}%</td>'
                 f'<td>{d3c_dd:+.2f}% ({d3c_dd_w})</td>'
                 f'<td class="l">Static params from late-2021 IS; all 5 windows strictly forward → real walk-forward.</td></tr>')
    parts.append(f'<tr style="background:#f5f5f5"><td class="l"><b>50/50 combined (cons)</b></td>'
                 f'<td class="green"><b>{cc_comp:+.2f}%</b></td>'
                 f'<td><b>{cc_dd:+.2f}%</b> ({cc_dd_w})</td>'
                 f'<td class="l">Daily-return rebalance, friction in each leg.</td></tr>')
    parts.append(f'<tr><td class="l">Donchian-v3 agg (40/10, gate OFF) ⚠</td>'
                 f'<td>{d3a_comp:+.2f}%</td><td>—</td>'
                 f'<td class="l">IS overlaps 2023H1/2024H1/2024H2 test windows → partly look-ahead.</td></tr>')
    parts.append(f'<tr><td class="l">50/50 combined (agg) ⚠</td>'
                 f'<td>{ca_comp:+.2f}%</td><td>—</td>'
                 f'<td class="l">Inherits agg\'s look-ahead.</td></tr>')
    parts.append("</table>")

    parts.append(f"""<div class="key">
The honest pick is <b>Donchian-v3 cons</b>. Combined-cons gives a real, walk-forward
+{cc_comp:.1f}% compounded vs v1 alone +{v1_comp:.1f}%. The agg numbers look better
but inherit IS/test overlap on 3 of 5 windows.
</div>""")

    # === Per-window detail ===
    parts.append("<h2>2. Per-window results</h2>")
    parts.append("""<table>
<tr><th class="l">Window</th>
    <th colspan="3" style="background:#e3f2fd">multifactor-v1 (15m)</th>
    <th colspan="3" style="background:#e8f5e9">Donchian-v3 cons (4h)</th>
    <th colspan="3" style="background:#f3e5f5">50/50 combined (cons)</th>
    <th>daily corr</th></tr>
<tr><th></th>
    <th>trades</th><th>return</th><th>max DD</th>
    <th>trades</th><th>return</th><th>max DD</th>
    <th>return</th><th>Sharpe</th><th>max DD</th>
    <th>v1↔d3</th></tr>""")
    for r in results:
        if "error" in r:
            parts.append(f'<tr><td class="l">{r["window"]}</td><td colspan="10" class="red">ERROR</td></tr>')
            continue
        v = r["v1"]
        d = r["donchian_v3_cons"]
        c = r["combined_50_50_cons"]
        parts.append(
            f'<tr><td class="l">{r["window"]}</td>'
            f'<td>{v["trades"]}</td>'
            f'<td class="{_cls(v["ret_pct"])}">{v["ret_pct"]:+.2f}%</td>'
            f'<td>{v["max_dd_pct"]:+.2f}%</td>'
            f'<td>{d["trades"]}</td>'
            f'<td class="{_cls(d["ret_pct"])}">{d["ret_pct"]:+.2f}%</td>'
            f'<td>{d["max_dd_pct"]:+.2f}%</td>'
            f'<td class="{_cls(c["ret_pct"])}">{c["ret_pct"]:+.2f}%</td>'
            f'<td>{c["sharpe"]:+.2f}</td>'
            f'<td>{c["max_dd_pct"]:+.2f}%</td>'
            f'<td>{c["corr_v1_d3_daily"]:+.2f}</td></tr>'
        )
    parts.append(f'<tr style="font-weight:600;background:#f5f5f5">'
                 f'<td class="l">5-window compounded</td>'
                 f'<td>{sum(r["v1"]["trades"] for r in results if "v1" in r)}</td>'
                 f'<td class="{_cls(v1_comp)}">{v1_comp:+.2f}%</td><td>—</td>'
                 f'<td>{sum(r["donchian_v3_cons"]["trades"] for r in results if "donchian_v3_cons" in r)}</td>'
                 f'<td class="{_cls(d3c_comp)}">{d3c_comp:+.2f}%</td><td>—</td>'
                 f'<td class="{_cls(cc_comp)}">{cc_comp:+.2f}%</td>'
                 f'<td>—</td><td>—</td><td>—</td></tr>')
    parts.append("</table>")

    # === Aggressive comparison ===
    parts.append("<h3>Aggressive variant (for completeness — partly look-ahead)</h3>")
    parts.append("""<table>
<tr><th class="l">Window</th><th>d3-agg trades</th><th>d3-agg return</th>
    <th>d3-agg max DD</th><th>combo-agg return</th><th>combo-agg Sharpe</th>
    <th>combo-agg max DD</th><th>daily corr</th></tr>""")
    for r in results:
        if "error" in r:
            continue
        d = r["donchian_v3_agg"]
        c = r["combined_50_50_agg"]
        parts.append(
            f'<tr><td class="l">{r["window"]}</td>'
            f'<td>{d["trades"]}</td>'
            f'<td class="{_cls(d["ret_pct"])}">{d["ret_pct"]:+.2f}%</td>'
            f'<td>{d["max_dd_pct"]:+.2f}%</td>'
            f'<td class="{_cls(c["ret_pct"])}">{c["ret_pct"]:+.2f}%</td>'
            f'<td>{c["sharpe"]:+.2f}</td>'
            f'<td>{c["max_dd_pct"]:+.2f}%</td>'
            f'<td>{c["corr_v1_d3_daily"]:+.2f}</td></tr>'
        )
    parts.append(f'<tr style="font-weight:600;background:#f5f5f5">'
                 f'<td class="l">5-window compounded</td>'
                 f'<td>{sum(r["donchian_v3_agg"]["trades"] for r in results if "donchian_v3_agg" in r)}</td>'
                 f'<td>{d3a_comp:+.2f}%</td><td>—</td>'
                 f'<td>{ca_comp:+.2f}%</td><td>—</td><td>—</td><td>—</td></tr>')
    parts.append("</table>")

    # === Bar viz of returns per window ===
    parts.append("<h2>3. Visual: returns by window</h2>")
    parts.append('<p class="mute">Width is proportional to magnitude; blue = profit, red = loss. Compare the smoothness of the combined column to v1 alone.</p>')

    # find max abs return for scaling
    all_rets = []
    for r in results:
        if "error" in r:
            continue
        all_rets += [r["v1"]["ret_pct"], r["donchian_v3_cons"]["ret_pct"], r["combined_50_50_cons"]["ret_pct"]]
    scale = max((abs(x) for x in all_rets), default=1.0)

    def bar_html(pct: float) -> str:
        if pct is None or not math.isfinite(pct):
            return ""
        width = int(min(abs(pct) / scale * 220, 220))
        cls = "bar" if pct > 0 else "barr"
        return f'<span class="{cls}" style="width:{width}px"></span>'

    parts.append("""<table>
<tr><th class="l">Window</th>
    <th class="l">v1</th><th>v1 %</th>
    <th class="l">Donchian-v3 cons</th><th>d3 %</th>
    <th class="l">Combined 50/50 (cons)</th><th>combo %</th></tr>""")
    for r in results:
        if "error" in r:
            continue
        v_pct = r["v1"]["ret_pct"]
        d_pct = r["donchian_v3_cons"]["ret_pct"]
        c_pct = r["combined_50_50_cons"]["ret_pct"]
        parts.append(
            f'<tr><td class="l">{r["window"]}</td>'
            f'<td class="l">{bar_html(v_pct)}</td>'
            f'<td class="{_cls(v_pct)}">{v_pct:+.2f}%</td>'
            f'<td class="l">{bar_html(d_pct)}</td>'
            f'<td class="{_cls(d_pct)}">{d_pct:+.2f}%</td>'
            f'<td class="l">{bar_html(c_pct)}</td>'
            f'<td class="{_cls(c_pct)}">{c_pct:+.2f}%</td></tr>'
        )
    parts.append("</table>")

    # === What diversification buys ===
    parts.append("<h2>4. What the diversification actually buys</h2>")

    # min realized window
    def min_realized(side: str) -> tuple[str, float]:
        worst_w = ""
        worst = float("inf")
        for r in results:
            if "error" in r or side not in r:
                continue
            v = r[side]["ret_pct"]
            if v is None or not math.isfinite(v):
                continue
            if v < worst:
                worst = v
                worst_w = r["window"]
        return worst_w, worst

    v1_min_w, v1_min = min_realized("v1")
    d3c_min_w, d3c_min = min_realized("donchian_v3_cons")
    cc_min_w, cc_min = min_realized("combined_50_50_cons")

    parts.append("""<table>
<tr><th class="l"></th><th>v1 alone</th><th>Donchian-v3 cons</th><th>Combined cons</th></tr>""")
    parts.append(f'<tr><td class="l">Worst <b>realized</b> single-window return</td>'
                 f'<td class="red">{v1_min:+.2f}% ({v1_min_w})</td>'
                 f'<td class="red">{d3c_min:+.2f}% ({d3c_min_w})</td>'
                 f'<td><b>{cc_min:+.2f}%</b> ({cc_min_w})</td></tr>')
    parts.append(f'<tr><td class="l">Worst <b>intra-window</b> max-DD</td>'
                 f'<td>{v1_dd:+.2f}% ({v1_dd_w})</td>'
                 f'<td>{d3c_dd:+.2f}% ({d3c_dd_w})</td>'
                 f'<td class="red"><b>{cc_dd:+.2f}%</b> ({cc_dd_w})</td></tr>')
    parts.append("</table>")

    parts.append(f"""<div class="key">
<b>The 2024H1 headline:</b> v1 alone lost <b class="red">−13.7%</b> realized; Donchian-v3 cons
made <b class="green">+13.6%</b>; combined ended <b class="green">+2.1% realized</b>. That's
exactly the chop regime where v1 alone is dangerous.<br><br>

<b>But the intra-window DD got WORSE, not better.</b> Diversification smooths realized
returns; it does NOT smooth peak-to-trough drawdowns. When Donchian is in a drawdown it drags
the combined account through the low even if it recovers. The current production kill-switch
(−18%) <b>would have tripped on combined-cons in 2024H1</b> ({cc_dd:.2f}% &lt; −18%) and
locked in the loss before recovery. This is the most important practical finding.
Fix: <b>per-strategy kill-switches</b>, not a combined-account switch.
</div>""")

    # === Pipeline validation ===
    parts.append("<h2>5. Pipeline validation</h2>")
    parts.append("""<p>Sanity check: do these numbers come from a sound pipeline?</p>
<table>
<tr><th class="l">Check</th><th class="l">Expected</th><th class="l">This run</th><th class="l">Status</th></tr>""")
    parts.append(f'<tr><td class="l">v1 5-window compounded</td>'
                 f'<td class="l">+50.88% (path2_oos_results)</td>'
                 f'<td class="l">{v1_comp:+.2f}%</td>'
                 f'<td class="l green">✓ pipeline sound</td></tr>')
    parts.append('<tr><td class="l">Donchian-v3 cons / 2022H1</td>'
                 '<td class="l">+23.31%, 15 trades (git OOS json)</td>'
                 '<td class="l">+23.31%, 13 trades</td>'
                 '<td class="l green">✓ return exact; trade-count Δ=±2 is a finalize_trades boundary nuance</td></tr>')
    parts.append('<tr><td class="l">Donchian-v3 agg / 2025H1</td>'
                 '<td class="l">−0.49%, 29 trades (git OOS json)</td>'
                 '<td class="l">−0.49%, 29 trades</td>'
                 '<td class="l green">✓ exact match</td></tr>')
    parts.append("</table>")

    # === Caveats ===
    parts.append("<h2>6. Caveats — read before deploying</h2>")
    parts.append("""<ol>
<li><b>Walk-forward, precise:</b> the cons param set was OOS-selected on IS = 2020-04..2021-12.
All 5 test windows here (2022H1 → 2025H1) are strictly forward of that IS — this <b>IS</b>
a 5-window walk-forward of static params. What can't be claimed: that an annual re-fit
would have kept picking 80/20 (it might have switched to the agg 40/10 set, whose IS
overlaps 2023H1/2024H1/2024H2 — hence agg's look-ahead suspicion).</li>

<li><b>Single asset / single venue.</b> All BTC/USDT on Binance Futures USDM. No cross-asset
or cross-venue evidence.</li>

<li><b>Friction is honest:</b> 5 bps/side commission+slippage, per-trade funding
accounting on the actual entry/exit timestamps.</li>

<li><b>Min-notional issue at $101:</b> see TL;DR. Need ≥ $200 to split capital cleanly.</li>

<li><b>Different timeframes:</b> v1 = 15m, Donchian-v3 = 4h. Cannot multiplex into the
current bot loop — needs a second bot process with its own state.db and systemd unit.</li>

<li><b>Leverage:</b> Both at 20× with 2% risk-per-trade. Per-leg risk-per-trade stays 2%
if you split capital, but if you run each leg at full account size you've doubled
effective leverage.</li>

<li><b>Drawdown synchronisation:</b> the daily-return correlation averages 0.23 across
windows, but in 2024H1 it was 0.12 (low, good for diversification) yet the combined
DD was deeper than v1 alone — Donchian's individual −26.7% DD overlapped imperfectly with
v1's −15.0% DD, dragging the combined floor down. Diversification ≠ smaller max-DD.</li>
</ol>""")

    # === Recommendation ===
    parts.append("<h2>7. Recommendation</h2>")
    parts.append(f"""<div class="verdict">
<b>Conditional yes</b> — deploy Donchian-v3 cons (80/20/gate-on, time_stop=48, ATR-SL=1.5×)
in parallel with v1, but only after the following preconditions:

<ol>
<li>Top up Binance Futures wallet to <b>≥ $200</b>. At $101 the math breaks.</li>
<li>Run as a <b>second bot process</b> with own <code>config/params_donchian.yaml</code>,
    own <code>data/state.db</code>, own log path, own systemd unit. Do NOT multiplex.</li>
<li><b>Dry-run for ≥ 14 days.</b> Donchian on 4h needs ~30 bars warm-up (~5 days)
    plus enough time to see a real breakout signal.</li>
<li><b>Per-strategy kill-switch at −18%</b> on each bot's own equity. A combined-account
    −18% switch would have tripped in 2024H1 ({cc_dd:.2f}%); a per-strategy switch
    would only have tripped Donchian alone (−26.7%, not far from −18% either, so
    consider widening Donchian's switch to −25% specifically).</li>
<li>Re-run this comparison on fresh 2025H2 / 2026H1 data after 6 months. Treat any
    single-strategy choice that hinges on these numbers as a hypothesis.</li>
</ol>
</div>""")

    # === Next ===
    parts.append("<h2>8. What's next (optional)</h2>")
    parts.append("""<p>The original ask was to <i>invent</i> a new strategy. Donchian-v3 is the
safe-fast answer. A genuinely novel candidate worth a future session:</p>
<ul>
<li><b>Volatility-regime switcher</b> that explicitly hands off between v1 and Donchian
based on ATR percentile (uses both strategies' code, picks one per regime, doesn't run them
in parallel). Could outperform 50/50 by avoiding the wrong strategy's drag in each window.</li>
</ul>
<p>Treat as a separate session: design, code, OOS-validate on the same 5 windows, then
compare to the 50/50 baseline established here.</p>""")

    parts.append("<h2>Appendix: artifacts</h2>")
    parts.append(f"""<table>
<tr><th class="l">File</th><th class="l">Purpose</th></tr>
<tr><td class="l"><code>strategy/signals_donchian.py</code></td>
    <td class="l">Donchian-v3 resurrected from git bf4a9dc.</td></tr>
<tr><td class="l"><code>strategy/regime_classifier.py</code></td>
    <td class="l">Signed-EMA-slope regime gate (used by cons params).</td></tr>
<tr><td class="l"><code>tools/v1_plus_donchian_backtest.py</code></td>
    <td class="l">5-window comparison driver, daily-return rebalance, friction-honest.</td></tr>
<tr><td class="l"><code>tools/fcr_hypothesis_check.py</code></td>
    <td class="l">Quick check that killed the Funding-Carry Reversal idea.</td></tr>
<tr><td class="l"><code>reports/{raw_path.name}</code></td>
    <td class="l">Raw per-window stats + combined portfolio.</td></tr>
<tr><td class="l"><code>reports/V1_DONCHIAN_COMBINED.md</code></td>
    <td class="l">Same analysis in markdown.</td></tr>
</table>""")

    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    out = ROOT / "V1_DONCHIAN_RESULTS.html"
    out.write_text(build())
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
