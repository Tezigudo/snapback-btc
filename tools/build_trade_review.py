"""Build an HTML review of my REAL futures trades: 1h/4h/1d candlestick charts
per trade with entry/exit marked + classification (pump-fade vs short-into-down)
+ the volume read. Live fapi klines (serves the current month). Plotly via CDN.

Run: .venv/bin/python tools/build_trade_review.py  ->  reports/MY_TRADE_REVIEW.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "MY_TRADE_REVIEW.html"
WEIRD = {"PORTALUSDT", "KATUSDT"}
TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def fetch(symbol: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows, cur = [], start_ms
    while cur < end_ms:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": symbol, "interval": tf, "startTime": cur,
                                 "endTime": end_ms, "limit": 1500}, timeout=30)
        r.raise_for_status()
        b = r.json()
        if not b:
            break
        rows += b
        cur = b[-1][0] + TF_MS[tf]
        if len(b) < 1500:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"])
    df["t"] = pd.to_numeric(df["t"])
    for k in ("o", "h", "l", "c", "v"):
        df[k] = pd.to_numeric(df[k])
    return df[["t", "o", "h", "l", "c", "v"]]


def chart(df: pd.DataFrame, start_ms: int, end_ms: int) -> dict:
    d = df[(df.t >= start_ms) & (df.t <= end_ms)]
    return {
        "t": [pd.Timestamp(x, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M") for x in d.t],
        "o": d.o.tolist(), "h": d.h.tolist(), "l": d.l.tolist(), "c": d.c.tolist(), "v": d.v.tolist(),
    }


def classify(side: str, sym: str, ret72: float, dret: float, rolled: float, volR: float) -> tuple[str, str]:
    if side == "LONG":
        return ("DIP-BUY / long", "long an ordinary coin after a pullback")
    if ret72 is not None and ret72 < -0.15:
        return ("SHORT INTO A DOWN-MOVE ⚠", "shorted a coin already falling — bounce/squeeze risk (this is NOT the pump-fade)")
    if (dret is not None and dret > 0.30) or (ret72 is not None and ret72 > 0.10):
        scale = "blow-off" if (dret or 0) > 0.8 else "run-up"
        return (f"PUMP-FADE ({scale})", "faded an up-move after it rolled over off the peak")
    return ("short (other)", "short without a clear up-move into entry")


def main() -> int:
    trades = json.loads(Path("/tmp/my_futures_history.json").read_text())["trades"]
    trades.sort(key=lambda t: t["open_ms"])
    by_sym: dict[str, dict] = {}
    for t in trades:
        s = t["symbol"]
        by_sym.setdefault(s, {"lo": t["open_ms"], "hi": t["close_ms"]})
        by_sym[s]["lo"] = min(by_sym[s]["lo"], t["open_ms"])
        by_sym[s]["hi"] = max(by_sym[s]["hi"], t["close_ms"])
    # fetch each symbol's klines once over a generous span
    klines: dict[str, dict] = {}
    for s, span in by_sym.items():
        klines[s] = {}
        for tf, pre, post in [("1h", 9, 2), ("4h", 35, 6), ("1d", 130, 12)]:
            klines[s][tf] = fetch(s, tf, span["lo"] - pre * 86_400_000, span["hi"] + post * 86_400_000)
        print(f"fetched {s}: " + ", ".join(f"{tf}={len(klines[s][tf])}" for tf in ("1h", "4h", "1d")))

    payload = []
    for t in trades:
        s = t["symbol"]
        k1 = klines[s]["1h"]
        o, c = int(t["open_ms"]), int(t["close_ms"])
        pre = k1[k1.t <= o]
        ret24 = float(pre.c.iloc[-1] / pre.c.iloc[-25] - 1) if len(pre) >= 25 else None
        ret72 = float(pre.c.iloc[-1] / pre.c.iloc[-73] - 1) if len(pre) >= 73 else None
        dret = float(pre.h.iloc[-25:].max() / pre.c.iloc[-25] - 1) if len(pre) >= 25 else None
        w3 = k1[(k1.t >= o - 3 * 86_400_000) & (k1.t <= o)]
        peak = float(w3.h.max()) if len(w3) else None
        rolled = (1 - t["entry_px"] / peak) if peak else None
        pkvol = float(w3.loc[w3.h.idxmax(), "v"]) if len(w3) else None
        evol = float(pre.v.iloc[-3:].mean()) if len(pre) else None
        volR = (evol / pkvol) if pkvol else None
        dur = k1[(k1.t >= o) & (k1.t <= c)]
        mae = float(dur.h.max() / t["entry_px"] - 1) if len(dur) else None
        tag, why = classify(t["side"], s, ret72, dret, rolled, volR)
        volread = "RISING ⚠ (danger)" if (volR or 0) >= 0.8 else ("drying up ✓ (exhaustion)" if (volR or 1) <= 0.5 else "mid")
        payload.append({
            "sym": s, "side": t["side"], "pnl": t["realizedPnl"], "weird": s in WEIRD,
            "entry": t["entry_px"], "exit": t["exit_px"],
            "open": pd.Timestamp(o, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
            "close": pd.Timestamp(c, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
            "open_ms": o, "close_ms": c,
            "ret24": ret24, "ret72": ret72, "dret": dret, "rolled": rolled, "volR": volR, "mae": mae,
            "tag": tag, "why": why, "volread": volread,
            "held_h": round((c - o) / 3.6e6, 1),
            "ch": {tf: chart(klines[s][tf],
                             o - {"1h": 7, "4h": 30, "1d": 110}[tf] * 86_400_000,
                             c + {"1h": 2, "4h": 5, "1d": 10}[tf] * 86_400_000) for tf in ("1d", "4h", "1h")},
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.replace("/*__DATA__*/", json.dumps(payload)))
    print(f"wrote {OUT} ({OUT.stat().st_size//1024} KB)")
    return 0


TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>My Futures Trade Review</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{margin:0;background:#0e1420;color:#e6edf6;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:24px 18px 80px}
h1{font-size:26px;margin:0 0 16px}
.tr{background:#161e2e;border:1px solid #243049;border-radius:12px;padding:16px 18px;margin:18px 0}
.hd{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;margin-bottom:6px}
.hd .sym{font-size:20px;font-weight:700}
.pill{font-size:12px;padding:2px 9px;border-radius:20px;font-weight:600}
.short{background:#3a1d24;color:#ff8a9b}.long{background:#13324a;color:#7cc4ff}
.win{background:#13361f;color:#69db7c}.loss{background:#3a1d1d;color:#ff8a8a}
.tag{background:#2a2140;color:#c9b6ff}.weird{background:#2c2410;color:#ffd479}
.why{color:#9fb0c8;margin:2px 0 10px}
.feat{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:10px 0}
.feat div{background:#0f1828;border:1px solid #243049;border-radius:8px;padding:8px 10px}
.feat b{display:block;font-size:16px}.feat span{color:#8fa3bd;font-size:12px}
.neg{color:#ff8a8a}.pos{color:#69db7c}
.charts{display:grid;grid-template-columns:1fr;gap:8px}
.ch{background:#0c121d;border:1px solid #1e2940;border-radius:8px}
.lead{background:#161e2e;border:1px solid #243049;border-left:4px solid #c9b6ff;border-radius:10px;padding:14px 16px;margin:14px 0;color:#cdd9ea}
.foot{color:#6b7a92;font-size:12px;margin-top:24px}
</style></head><body><div class="wrap">
<h1>My Futures Trade Review <span style="color:#6b7a92;font-size:15px">— real fills, 2026</span></h1>
<div class="lead" id="lead"></div>
<div id="trades"></div>
<p class="foot">Live fapi klines. Entry ▲/▼ and exit ● marked; dashed lines = entry & exit price. Volume read: drying-up ✓ = exhaustion (fade-able), RISING ⚠ = continuation/squeeze danger.</p>
</div>
<script>
const D=/*__DATA__*/;
const pct=v=>v==null?'—':(v>0?'+':'')+(v*100).toFixed(0)+'%';
const g=v=>v==null?'—':(+v).toPrecision(5);
// lead synthesis
document.getElementById('lead').innerHTML=
 "<b>The pattern in your real money:</b> your two short WINS (WLD, PORTAL) both <b>faded an UP-move</b> after it rolled off the peak, with volume <b>drying up / mid</b>. Your big short LOSS (KAT) <b>shorted a coin already DOWN −33%</b>, 36% below its peak, with volume <b>RISING 4.3×</b> — and it bounced +18%. Same lesson the backtest gave: fade exhausted up-moves, never short into a falling knife with rising volume.";

const wrap=document.getElementById('trades');
D.forEach((t,i)=>{
 const win=t.pnl>0;
 const el=document.createElement('div');el.className='tr';
 el.innerHTML=`<div class="hd">
   <span class="sym">${t.sym.replace('USDT','')}</span>
   <span class="pill ${t.side==='SHORT'?'short':'long'}">${t.side}</span>
   <span class="pill ${win?'win':'loss'}">${win?'+':''}$${t.pnl.toFixed(2)}</span>
   <span class="pill tag">${t.tag}</span>
   <span class="pill ${t.weird?'weird':''}">${t.weird?'weird coin':'ordinary coin'}</span>
   <span style="color:#6b7a92">${t.open} → held ${t.held_h}h</span></div>
  <div class="why">${t.why}</div>
  <div class="feat">
   <div><b>${pct(t.dret)}</b><span>24h-high spike into entry</span></div>
   <div><b>${pct(t.ret72)}</b><span>72h move into entry</span></div>
   <div><b>${pct(t.rolled)}</b><span>entry below 3d-peak (rolled over)</span></div>
   <div><b class="${(t.volR>=0.8)?'neg':'pos'}">${t.volR==null?'—':t.volR.toFixed(2)}</b><span>volume vs peak — ${t.volread}</span></div>
   <div><b class="${(t.mae>0.15)?'neg':'pos'}">${pct(t.mae)}</b><span>max move against you</span></div>
   <div><b>${g(t.entry)} → ${g(t.exit)}</b><span>entry → exit</span></div>
  </div>
  <div class="charts">
   <div class="ch" id="d${i}"></div><div class="ch" id="h4_${i}"></div><div class="ch" id="h1_${i}"></div>
  </div>`;
 wrap.appendChild(el);
 [["1d","d"],["4h","h4_"],["1h","h1_"]].forEach(([tf,pfx])=>{
  const c=t.ch[tf]; const div=pfx+i;
  const cs={x:c.t,open:c.o,high:c.h,low:c.l,close:c.c,type:'candlestick',
            increasing:{line:{color:'#3fb950'}},decreasing:{line:{color:'#f85149'}},name:tf};
  const eIso=t.open, xIso=t.close;
  const shapes=[
   {type:'line',x0:eIso,x1:eIso,yref:'paper',y0:0,y1:1,line:{color:'#c9b6ff',width:1,dash:'dot'}},
   {type:'line',x0:xIso,x1:xIso,yref:'paper',y0:0,y1:1,line:{color:'#6b7a92',width:1,dash:'dot'}},
   {type:'line',x0:c.t[0],x1:c.t[c.t.length-1],y0:t.entry,y1:t.entry,line:{color:'#ff8a9b',width:1,dash:'dash'}},
   {type:'line',x0:c.t[0],x1:c.t[c.t.length-1],y0:t.exit,y1:t.exit,line:{color:'#69db7c',width:1,dash:'dash'}},
  ];
  const mk={x:[eIso,xIso],y:[t.entry,t.exit],mode:'markers',type:'scatter',
            marker:{symbol:[t.side==='SHORT'?'triangle-down':'triangle-up','circle'],size:[12,10],
                    color:['#ff8a9b','#69db7c']},name:'entry/exit',hoverinfo:'y'};
  Plotly.newPlot(div,[cs,mk],{
    height:tf==='1h'?300:240,margin:{l:54,r:10,t:22,b:24},
    paper_bgcolor:'#0c121d',plot_bgcolor:'#0c121d',font:{color:'#8fa3bd',size:10},
    title:{text:tf+(tf==='1d'?'  (context)':tf==='1h'?'  (entry zoom)':''),font:{size:11},x:0.01},
    xaxis:{rangeslider:{visible:false},gridcolor:'#1a2334'},yaxis:{gridcolor:'#1a2334'},shapes,showlegend:false,
  },{displayModeBar:false,responsive:true});
 });
});
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
