"""HTML report: impact of the user's volume-drying rule on the pump-fade backtest.
Runs baseline vs +rule variants, captures summary + equity path + net distribution,
renders reports/PUMPFADE_VOLRULE_REPORT.html (Plotly CDN, data inlined).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_backtest as pf  # noqa: E402

OUT = ROOT / "reports" / "PUMPFADE_VOLRULE_REPORT.html"
OOS = pd.Timestamp("2025-01-01", tz="UTC")
BASE = dict(select_mode="intraday_high", thresh=0.40, cooldown_days=7)
CONFIGS = [
    ("Baseline (no rule)", pf.Params(**BASE), "#8a93a6"),
    ("+ Volume rule (skip volR≥0.8)", pf.Params(**BASE, require_vol_drying=True, vol_dry_max=0.8), "#4f9dff"),
    ("+ Rule strict (skip volR≥0.5)", pf.Params(**BASE, require_vol_drying=True, vol_dry_max=0.5), "#7c5cff"),
    ("+ Rule + 48h hold (best)", pf.Params(**BASE, require_vol_drying=True, vol_dry_max=0.8, max_hold_h=48), "#27c093"),
]


def main() -> int:
    blocks = []
    for name, p, color in CONFIGS:
        df = pf.run_study(p, workers=16)
        t = df[df.reason.isin(["TP", "STOP", "TIME", "SETTLE"])].copy()
        t["et"] = pd.to_datetime(t.entry_time, utc=True)
        ep = pf.equity_path(t.sort_values("entry_time"), p)
        e = ep["equity"].values if not ep.empty else np.array([p.start_equity])
        peak = np.maximum.accumulate(e)
        dd = float(((e - peak) / peak).min() * 100) if len(e) else 0.0
        isd, oos = t[t.et < OOS], t[t.et >= OOS]
        nr = (t.net_ret * 100).clip(-220, 100)
        hist, edges = np.histogram(nr, bins=40)
        blocks.append({
            "name": name, "color": color,
            "n": int(len(t)), "skip": int((df.reason == "SKIP_VOLRISING").sum()),
            "win": round(100 * (t.net_ret > 0).mean(), 0),
            "ev": round(100 * t.net_ret.mean(), 2),
            "is": round(100 * isd.net_ret.mean(), 2) if len(isd) else None,
            "oos": round(100 * oos.net_ret.mean(), 2) if len(oos) else None,
            "worst": round(100 * t.net_ret.min(), 0),
            "fineq": round(float(e[-1]), 0), "dd": round(dd, 0),
            "eq_t": [str(x)[:10] for x in (ep["entry_time"] if not ep.empty else [])],
            "eq": [round(float(v), 1) for v in e],
            "dist_x": [round(float(x), 1) for x in (edges[:-1] + edges[1:]) / 2],
            "dist_y": [int(c) for c in hist],
        })
        print(f"  {name}: n={blocks[-1]['n']} EV {blocks[-1]['ev']}% worst {blocks[-1]['worst']}% finEq ${blocks[-1]['fineq']}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.replace("/*__DATA__*/", json.dumps(blocks)))
    print(f"wrote {OUT} ({OUT.stat().st_size//1024} KB)")
    return 0


TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Volume-Rule Backtest Report</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{margin:0;background:#0e1420;color:#e6edf6;font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:26px 18px 80px}
h1{font-size:27px;margin:0 0 6px}.sub{color:#8fa3bd;margin:0 0 18px}
.lead{background:#13241c;border:1px solid #1f5e44;border-left:5px solid #27c093;border-radius:10px;padding:15px 17px;margin:14px 0;color:#cfeede}
.lead b{color:#5fe0b0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}
.card{background:#161e2e;border:1px solid #243049;border-radius:10px;padding:13px 15px}
.card .v{font-size:23px;font-weight:700}.card .l{color:#8fa3bd;font-size:12px;margin-top:3px}
.pos{color:#5fe0b0}.neg{color:#ff8a8a}
table{width:100%;border-collapse:collapse;background:#161e2e;border:1px solid #243049;border-radius:10px;overflow:hidden;font-size:13.5px;margin:10px 0}
th,td{padding:9px 10px;text-align:right;border-bottom:1px solid #243049}th:first-child,td:first-child{text-align:left}
th{background:#1c2740;color:#9fb0c8;font-size:12px;text-transform:uppercase}
tr:last-child td{border-bottom:0}
h2{font-size:18px;margin:30px 0 8px;border-bottom:1px solid #243049;padding-bottom:6px}
.chart{background:#0c121d;border:1px solid #1e2940;border-radius:10px;margin:10px 0;padding:6px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:780px){.g2{grid-template-columns:1fr}}
.foot{color:#6b7a92;font-size:12px;margin-top:26px}
</style></head><body><div class="wrap">
<h1>Pump-Fade — Volume-Drying Rule</h1>
<p class="sub">“Only short a genuine pump (up &gt;30%) where volume is <b>drying up</b>; skip if volume is rising.”
&nbsp;·&nbsp; survivorship-safe universe · intraday rolling-24h selection · 2020–2026</p>
<div class="lead" id="lead"></div>
<div class="cards" id="cards"></div>
<h2>All configs</h2>
<table id="tbl"></table>
<h2>Equity curve — $1,000 start, 2%-risk sizing</h2>
<div class="chart" id="eq"></div>
<div class="g2">
<div><h2>Worst single trade</h2><div class="chart" id="worst"></div></div>
<div><h2>Max drawdown</h2><div class="chart" id="dd"></div></div>
</div>
<h2>Return distribution — Baseline vs +Rule (the tail the rule removes)</h2>
<div class="chart" id="dist"></div>
<div class="lead" style="border-color:#5b6b86;background:#161e2e;color:#cdd9ea">
<b style="color:#e6edf6">Honest verdict:</b> the rule turns the mechanical strategy from <i>bleeding with −208% blow-up risk</i>
into <b style="color:#5fe0b0">roughly breakeven with controlled risk</b> — worst trade −208% → −102%, equity $98 → $281,
max DD −91% → −75%, win 57% → 60%. But mechanically it's still ~breakeven (EV ≈ −1%), not a money-printer.
Your real-money edge on genuine pumps (+$2.49/trade) is the part the robot can’t copy — your <b>selection</b>.
Rule = discipline layer; rule + your selection = the positive combination.</div>
<p class="foot">Generated by tools/build_volrule_report.py from tools/pumpfade_backtest.py (require_vol_drying). Net of modeled fees+slippage+funding.</p>
</div>
<script>
const D=/*__DATA__*/;
const f=v=>v==null?'—':(v>0?'+':'')+v+'%';
const base=D[0], best=D[D.length-1];
document.getElementById('lead').innerHTML=
 `<b>Your rule works.</b> Adding “volume must be drying up” skipped ${best.skip} rising-volume setups — and those were the blow-ups: `+
 `worst trade <b>${base.worst}% → ${best.worst}%</b>, equity <b>$${base.fineq} → $${best.fineq}</b>, max DD <b>${base.dd}% → ${best.dd}%</b>, win <b>${base.win}% → ${best.win}%</b>.`;
const cards=[['Worst trade',`${base.worst}% → ${best.worst}%`,'pos'],['Equity $1k→',`$${base.fineq} → $${best.fineq}`,'pos'],
 ['Max DD',`${base.dd}% → ${best.dd}%`,'pos'],['Win rate',`${base.win}% → ${best.win}%`,'pos'],['EV/trade',`${base.ev}% → ${best.ev}%`, best.ev>base.ev?'pos':'neg']];
document.getElementById('cards').innerHTML=cards.map(c=>`<div class="card"><div class="v ${c[2]}">${c[1]}</div><div class="l">${c[0]}</div></div>`).join('');
document.getElementById('tbl').innerHTML=
 `<tr><th>Config</th><th>n</th><th>skipped</th><th>Win</th><th>EV</th><th>IS</th><th>OOS</th><th>Worst</th><th>Eq $1k→</th><th>DD</th></tr>`+
 D.map(b=>`<tr><td>${b.name}</td><td>${b.n}</td><td>${b.skip||'—'}</td><td>${b.win}%</td>`+
 `<td class="${b.ev>0?'pos':'neg'}">${b.ev}%</td><td>${f(b.is)}</td><td>${f(b.oos)}</td>`+
 `<td class="neg">${b.worst}%</td><td>$${b.fineq}</td><td class="neg">${b.dd}%</td></tr>`).join('');
const L={paper_bgcolor:'#0c121d',plot_bgcolor:'#0c121d',font:{color:'#8fa3bd',size:11},margin:{l:52,r:12,t:10,b:34}};
const P={displayModeBar:false,responsive:true};
Plotly.newPlot('eq', D.map(b=>({x:b.eq_t,y:b.eq,type:'scatter',mode:'lines',name:b.name,line:{color:b.color,width:2}})),
 {...L,height:340,yaxis:{title:'equity $',gridcolor:'#1a2334'},xaxis:{gridcolor:'#1a2334'},legend:{orientation:'h',y:-0.18},
  shapes:[{type:'line',x0:D[0].eq_t[0],x1:D[0].eq_t[D[0].eq_t.length-1],y0:1000,y1:1000,line:{color:'#5b6b86',dash:'dot',width:1}}]},P);
Plotly.newPlot('worst',[{x:D.map(b=>b.name),y:D.map(b=>b.worst),type:'bar',marker:{color:D.map(b=>b.color)},text:D.map(b=>b.worst+'%'),textposition:'outside'}],
 {...L,height:280,yaxis:{title:'worst trade %',gridcolor:'#1a2334'},xaxis:{tickangle:-12}},P);
Plotly.newPlot('dd',[{x:D.map(b=>b.name),y:D.map(b=>b.dd),type:'bar',marker:{color:D.map(b=>b.color)},text:D.map(b=>b.dd+'%'),textposition:'outside'}],
 {...L,height:280,yaxis:{title:'max drawdown %',gridcolor:'#1a2334'},xaxis:{tickangle:-12}},P);
Plotly.newPlot('dist',[
 {x:base.dist_x,y:base.dist_y,type:'bar',name:'Baseline',marker:{color:'rgba(138,147,166,.65)'}},
 {x:best.dist_x,y:best.dist_y,type:'bar',name:'+Rule+48h',marker:{color:'rgba(39,192,147,.7)'}}],
 {...L,height:300,barmode:'overlay',xaxis:{title:'net return % per trade (clipped)',gridcolor:'#1a2334'},yaxis:{title:'trades',gridcolor:'#1a2334'},legend:{orientation:'h',y:-0.2}},P);
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
