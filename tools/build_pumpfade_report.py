"""Render the pump-fade study to a self-contained HTML report.

Loads the faithful intraday-high trade stream (+ daily-close cross-check),
computes the headline / by-year / by-reason / cohort / distribution stats and
the risk-sized equity path, and writes reports/PUMPFADE_REPORT.html with Plotly
(CDN). No plotly install needed to GENERATE — all aggregates are computed in
pandas and inlined as JSON; Plotly.js (CDN) renders them in the browser.

Run:  .venv/bin/python tools/build_pumpfade_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_backtest as pf  # noqa: E402

CACHE = ROOT / "data" / "pumpfade"
OUT = ROOT / "reports" / "PUMPFADE_REPORT.html"
OOS = pd.Timestamp("2025-01-01", tz="UTC")
TAKEN = ["TP", "STOP", "TIME", "SETTLE"]


def taken_of(path: str) -> pd.DataFrame:
    df = pd.read_parquet(CACHE / path)
    t = df[df.reason.isin(TAKEN)].copy()
    t["et"] = pd.to_datetime(t["entry_time"], utc=True)
    t["year"] = t["et"].dt.year
    return t


def agg(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return {"n": 0}
    nr = d["net_ret"]
    return {
        "n": int(len(d)),
        "win": round(100 * (nr > 0).mean(), 1),
        "ev": round(100 * nr.mean(), 2),
        "med": round(100 * nr.median(), 2),
        "worst": round(100 * nr.min(), 1),
        "best": round(100 * nr.max(), 1),
        "stop": round(100 * (d.reason == "STOP").mean(), 1),
        "tp": round(100 * (d.reason == "TP").mean(), 1),
        "fund": round(100 * d["funding_ret"].mean(), 2),
    }


def main() -> int:
    intr = taken_of("trades_intraday.parquet")   # faithful
    close = taken_of("trades_honest.parquet")     # cross-check
    p = pf.Params(cooldown_days=7)

    # headline blocks
    headline = {
        "intraday_all": agg(intr), "intraday_is": agg(intr[intr.et < OOS]), "intraday_oos": agg(intr[intr.et >= OOS]),
        "close_all": agg(close), "close_oos": agg(close[close.et >= OOS]),
        "delisted": agg(intr[intr.is_delisted]), "surviving": agg(intr[~intr.is_delisted]),
    }

    # by year (faithful)
    years = sorted(intr.year.unique())
    by_year = [{"year": int(y), **agg(intr[intr.year == y])} for y in years]

    # by exit reason (faithful)
    by_reason = []
    for r in ["TP", "TIME", "SETTLE", "STOP"]:
        g = intr[intr.reason == r]
        by_reason.append({"reason": r, "n": int(len(g)),
                          "mean": round(100 * g.net_ret.mean(), 1) if len(g) else 0})

    # equity path (faithful, risk-sized)
    ep = pf.equity_path(intr.sort_values("entry_time"), p)
    eq = {"t": [str(x)[:10] for x in ep["entry_time"]], "eq": [round(float(v), 1) for v in ep["equity"]]} \
        if not ep.empty else {"t": [], "eq": []}

    # return distribution (clip for display)
    nr = (intr.net_ret * 100).clip(-220, 100)
    hist, edges = np.histogram(nr, bins=44)
    dist = {"x": [round(float(e), 1) for e in (edges[:-1] + edges[1:]) / 2], "y": [int(c) for c in hist]}

    # scatter: selection magnitude vs outcome (the tail)
    samp = intr.sample(min(len(intr), 900), random_state=1) if len(intr) else intr
    scat = {
        "x": [round(float(v) * 100, 1) for v in samp.day_ret],
        "y": [round(float(v) * 100, 1) for v in samp.net_ret.clip(-220, 100)],
        "c": ["#c62828" if r == "STOP" else ("#2e7d32" if n > 0 else "#9e9e9e")
              for r, n in zip(samp.reason, samp.net_ret)],
    }

    # example trades
    def ex(df, asc, k=5):
        d = df.nsmallest(k, "net_ret") if asc else df.nlargest(k, "net_ret")
        return [{"sym": r.symbol, "day": str(r.day)[:10], "entry": float(r.entry), "peak": float(r.peak),
                 "exit": float(r.exit), "mae": round(float(r.mae) * 100, 0), "net": round(float(r.net_ret) * 100, 0),
                 "reason": r.reason, "dl": bool(r.is_delisted)} for _, r in d.iterrows()]
    examples = {"wins": ex(intr, asc=False), "blowups": ex(intr, asc=True)}

    data = {"headline": headline, "by_year": by_year, "by_reason": by_reason,
            "eq": eq, "dist": dist, "scat": scat, "examples": examples}

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
    print(f"  faithful: n={headline['intraday_all']['n']} EV {headline['intraday_all']['ev']}% "
          f"OOS {headline['intraday_oos']['ev']}%  worst {headline['intraday_all']['worst']}%")
    return 0


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pump-Fade Study — Verdict</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--ink:#1a2230;--mut:#5b6677;--line:#e3e8ef;--red:#c62828;--redbg:#ffebee;--grn:#2e7d32;--grnbg:#e8f5e9;--amb:#f9a825;--ambbg:#fff8e1;--card:#fff;--bg:#f3f5f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:30px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}
.sub{color:var(--mut);margin:0 0 18px}
.verdict{background:var(--redbg);border:1px solid #ef9a9a;border-left:6px solid var(--red);border-radius:10px;padding:16px 18px;margin:18px 0}
.verdict b{color:var(--red)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .v{font-size:26px;font-weight:700;line-height:1.1}.card .l{color:var(--mut);font-size:12.5px;margin-top:3px}
.neg{color:var(--red)}.pos{color:var(--grn)}
.note{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:10px 0}
.note .b{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px;font-size:13.5px}
.note .b b{display:block;margin-bottom:3px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:9px 11px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{background:#eef2f7;font-weight:600;color:var(--mut);font-size:12.5px;text-transform:uppercase;letter-spacing:.02em}
tr:last-child td{border-bottom:0}.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;margin:12px 0}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:760px){.g2{grid-template-columns:1fr}}
.pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;background:#eceff3;color:var(--mut);margin-left:6px}
.foot{color:var(--mut);font-size:12.5px;margin-top:30px}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px}
</style></head><body><div class="wrap">

<h1>Pump-Fade Study</h1>
<p class="sub">“Short the day’s top Binance gainer (+40–100%) after it rolls over, TP at the pre-pump support, stop above the peak.”
&nbsp;·&nbsp; 2026-05-31 &nbsp;·&nbsp; survivorship-safe · point-in-time · adversarially verified (5 agents)</p>

<div class="verdict"><b>VERDICT — NOT a tradeable edge.</b> It has a winning hit-rate (57%) and a positive median (+4.6%),
so most trades work and it <i>feels</i> great — but expectancy is <b>negative</b> (−1.6%/trade) because ~1 in 10 pumps
<b>continues</b> (a 3–5× move against the short) and one blow-out erases dozens of small wins. <b>No stop policy survives</b>
(tight → whipsaw, wide → catastrophe, none → −440% tail). Even at <b>zero friction the signal is negative</b>. Proof-of-edge
research only — not a deployable bot.</div>

<div class="cards" id="cards"></div>

<h2>How this was made trustworthy</h2>
<div class="note">
<div class="b"><b>Survivorship-safe</b>732-coin point-in-time universe incl. <i>delisted</i> coins via <code>data.binance.vision</code> (the live API hides delisted history — and pumped microcaps are exactly what later vanish).</div>
<div class="b"><b>Faithful to the idea</b>Headline uses the <i>intraday rolling-24h</i> leaderboard crossing (your “every 4–8h”), entry only from the cross — captures the spike-then-fade winners a daily-close proxy drops (1354 vs 786 trades).</div>
<div class="b"><b>Honest friction</b>Taker fees both sides, illiquidity-scaled slippage, <i>heavy asymmetric stop slippage</i> (a stop fires into the squeeze), real funding every interval, delisting force-settle.</div>
<div class="b"><b>Adversarially verified</b>5 independent agents tried to refute the verdict (look-ahead/sign, friction realism, alt-mechanizations, data integrity, steelman-bull). All upheld it; the one bug found made results look <i>worse</i>.</div>
</div>

<h2>Headline — faithful intraday selection (dedup, $20M liquidity floor)</h2>
<table id="t_head"></table>

<h2>Equity path (risk-sized 2%/trade, $1000 start)</h2>
<div class="chart" id="c_eq"></div>

<div class="g2">
<div><h2>EV by year</h2><div class="chart" id="c_year"></div></div>
<div><h2>Net return by exit reason</h2><div class="chart" id="c_reason"></div></div>
</div>

<div class="g2">
<div><h2>Per-trade return distribution</h2><div class="chart" id="c_dist"></div></div>
<div><h2>Pump magnitude vs outcome</h2><div class="chart" id="c_scat"></div></div>
</div>

<h2>Survivorship check — delisted vs surviving</h2>
<p class="sub">The coins that pumped and <i>died</i> were profitable shorts (a survivor-only backtest never sees them) — but you can’t know ex-ante which delist, and the surviving majority loses. This <i>validates</i> the method while confirming the strategy fails.</p>
<table id="t_cohort"></table>

<h2>Why it fails — exit economics</h2>
<p class="sub">TP / TIME / SETTLE exits are all <b>positive</b>; STOP exits (≈⅓ of trades at −30%+) are the entire problem. When the short isn’t run over by a continuation, fading the pump works — surviving the continuation is what can’t be solved.</p>

<h2>Can it be tuned? — SL &amp; short-hold sweep</h2>
<p class="sub"><b>Tunable for RISK, not for EDGE.</b> Stops fire fast (61% of stop-outs &amp; ~half the
big losers happen within 48h), so a 1.5-2d cap helps only at the margin. Across the full hold &times; SL
grid <b>every cell is negative-EV</b>. A cap/structure/ATR stop <i>bounds</i> the worst trade to −42% to
−152% (survivable) but EV stays negative; removing the stop explodes the tail to −214% to −526%. The
negative edge is in the <i>signal</i> (price-only ≈ −2.7%), not the exit — tuning only moves the loss
between <i>fat tail</i> and <i>whipsaw</i>.</p>
<div class="g2">
<div><b>Hold &times; SL policy (EV / worst trade)</b>
<table>
<tr><th>SL ↓ / Hold →</th><th>36h</th><th>48h</th><th>72h</th><th>168h</th></tr>
<tr><td>peak (1.06×)</td><td class="neg">−1.47 / −208</td><td class="neg">−1.44 / −208</td><td class="neg">−1.82 / −208</td><td class="neg">−1.62 / −208</td></tr>
<tr><td>no price-stop</td><td class="neg">−1.33 / −214</td><td class="neg">−1.43 / −526</td><td class="neg">−1.52 / −266</td><td class="neg">−1.70 / −440</td></tr>
<tr><td>cap 30%</td><td class="neg">−1.68 / −42</td><td class="neg">−1.68 / −46</td><td class="neg">−1.85 / −84</td><td class="neg">—</td></tr>
</table></div>
<div><b>SL flavors @ 48h hold (EV all / OOS / worst)</b>
<table>
<tr><th>SL design</th><th>EV</th><th>OOS</th><th>Worst</th></tr>
<tr><td>peak (ref)</td><td class="neg">−1.44%</td><td class="neg">−1.44%</td><td class="neg">−208%</td></tr>
<tr><td>local-high +3% (structure)</td><td class="neg">−1.19%</td><td class="neg">−1.37%</td><td class="neg">−121%</td></tr>
<tr><td>local-high +8%</td><td class="neg">−1.46%</td><td class="neg">−1.57%</td><td class="neg">−130%</td></tr>
<tr><td>ATR 2.5×</td><td class="neg">−1.48%</td><td class="neg">−1.64%</td><td class="neg">−121%</td></tr>
<tr><td>ATR 4× <span class="pill">best anywhere</span></td><td class="neg">−1.13%</td><td class="neg">−1.21%</td><td class="neg">−152%</td></tr>
<tr><td>cap 40%</td><td class="neg">−1.36%</td><td class="neg">−1.21%</td><td class="neg">−62%</td></tr>
<tr><td>cap 50%</td><td class="neg">−1.43%</td><td class="neg">−1.33%</td><td class="neg">−74%</td></tr>
</table></div>
</div>

<h2>Example trades</h2>
<div class="g2">
<div><b class="pos">Biggest wins (reversion captured)</b><table id="t_wins"></table></div>
<div><b class="neg">Worst blow-ups (continuation)</b><table id="t_blow"></table></div>
</div>

<p class="foot">Generated by <code>tools/build_pumpfade_report.py</code> from <code>data/pumpfade/trades_intraday.parquet</code>.
Full writeup: <code>PUMPFADE_RESULTS.md</code>. Engine: <code>tools/pumpfade_{data,backtest,phase2}.py</code>.
Numbers are net of modeled friction + funding. Not investment advice.</p>
</div>

<script>
const D = /*__DATA__*/;
const sgn = v => (v>0?'pos':'neg');
const f = v => (v>0?'+':'')+v+'%';

// metric cards
const H=D.headline;
const cards=[
 ['EV / trade', H.intraday_all.ev+'%', sgn(H.intraday_all.ev)],
 ['Win rate', H.intraday_all.win+'%', 'pos'],
 ['Median', f(H.intraday_all.med), sgn(H.intraday_all.med)],
 ['OOS EV (2025–26)', H.intraday_oos.ev+'%', sgn(H.intraday_oos.ev)],
 ['Worst trade', H.intraday_all.worst+'%', 'neg'],
 ['Trades', H.intraday_all.n, ''],
];
document.getElementById('cards').innerHTML = cards.map(c=>
 `<div class="card"><div class="v ${c[2]}">${c[1]}</div><div class="l">${c[0]}</div></div>`).join('');

// headline table
function row(label,a,extra){return `<tr><td>${label}</td><td>${a.n}</td><td>${a.win}%</td>
 <td class="${sgn(a.ev)}"><b>${a.ev}%</b></td><td class="${sgn(a.med)}">${f(a.med)}</td>
 <td class="neg">${a.worst}%</td><td>${a.stop}%</td><td>${a.tp}%</td></tr>`;}
document.getElementById('t_head').innerHTML =
 `<tr><th>Slice</th><th>n</th><th>Win</th><th>EV/trade</th><th>Median</th><th>Worst</th><th>Stop%</th><th>TP%</th></tr>`+
 row('Intraday — ALL',H.intraday_all)+row('Intraday — IS (2020–24)',H.intraday_is)+
 row('Intraday — OOS (2025–26)',H.intraday_oos)+
 row('Daily-close — ALL (cross-check)',H.close_all)+row('Daily-close — OOS',H.close_oos);

document.getElementById('t_cohort').innerHTML =
 `<tr><th>Cohort</th><th>n</th><th>Win</th><th>EV/trade</th><th>Median</th><th>Worst</th><th>Stop%</th><th>TP%</th></tr>`+
 row('Delisted (died)',H.delisted)+row('Surviving',H.surviving);

const PL={displayModeBar:false,responsive:true};
const LAY={margin:{l:48,r:14,t:8,b:36},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{size:12},height:300};

// equity
Plotly.newPlot('c_eq',[{x:D.eq.t,y:D.eq.eq,type:'scatter',mode:'lines',line:{color:'#c62828',width:2},fill:'tozeroy',fillcolor:'rgba(198,40,40,.07)'}],
 {...LAY,height:320,yaxis:{title:'equity ($)'},shapes:[{type:'line',x0:D.eq.t[0],x1:D.eq.t[D.eq.t.length-1],y0:1000,y1:1000,line:{color:'#9e9e9e',dash:'dot',width:1}}]},PL);

// by year
Plotly.newPlot('c_year',[{x:D.by_year.map(r=>r.year),y:D.by_year.map(r=>r.ev),type:'bar',
 marker:{color:D.by_year.map(r=>r.ev>0?'#2e7d32':'#c62828')},text:D.by_year.map(r=>'n='+r.n),textposition:'outside'}],
 {...LAY,yaxis:{title:'EV/trade %',zeroline:true}},PL);

// by reason
Plotly.newPlot('c_reason',[{x:D.by_reason.map(r=>r.reason+' (n='+r.n+')'),y:D.by_reason.map(r=>r.mean),type:'bar',
 marker:{color:D.by_reason.map(r=>r.mean>0?'#2e7d32':'#c62828')},text:D.by_reason.map(r=>f(r.mean)),textposition:'outside'}],
 {...LAY,yaxis:{title:'mean net %',zeroline:true}},PL);

// distribution
Plotly.newPlot('c_dist',[{x:D.dist.x,y:D.dist.y,type:'bar',marker:{color:D.dist.x.map(v=>v>0?'#66bb6a':'#ef5350')}}],
 {...LAY,bargap:0.02,xaxis:{title:'net return % (clipped −220..+100)'},yaxis:{title:'trades'}},PL);

// scatter
Plotly.newPlot('c_scat',[{x:D.scat.x,y:D.scat.y,mode:'markers',type:'scatter',
 marker:{color:D.scat.c,size:5,opacity:.6}}],
 {...LAY,xaxis:{title:'pump size (24h high %)'},yaxis:{title:'net return %',zeroline:true}},PL);

// example tables
function exTable(rows){return `<tr><th>Coin</th><th>Day</th><th>MAE</th><th>Net</th><th>Exit</th></tr>`+
 rows.map(r=>`<tr><td>${r.sym.replace('USDT','')}${r.dl?' <span class="pill">delisted</span>':''}</td>
 <td>${r.day}</td><td class="neg">+${r.mae}%</td><td class="${sgn(r.net)}"><b>${f(r.net)}</b></td><td>${r.reason}</td></tr>`).join('');}
document.getElementById('t_wins').innerHTML = exTable(D.examples.wins);
document.getElementById('t_blow').innerHTML = exTable(D.examples.blowups);
</script>
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
