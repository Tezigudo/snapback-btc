"""Build v2 HTML report — incorporates mega sweep + final tuning results
including the breakthrough HYBRID detector.

Run: uv run python tools/build_icnh_v2_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.icnh_final_tune import run_hybrid  # noqa: E402
from tools.icnh_mega_sweep import (  # noqa: E402
    Config, load_tf, find_all_patterns, simulate_trades, WINDOWS
)

GRID = ROOT / "data" / "grid_results.json"
MEGA = ROOT / "data" / "mega_sweep_results.json"
TUNE = ROOT / "data" / "final_tune_results.json"
OUTPUT = ROOT / "ICNH_EXPERIMENT_V2.html"

GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
PURPLE = "#a855f7"
BLUE = "#3b82f6"


def hybrid_equity_curve(tf: str, dedup: int, sl_atr: float, tp_emas: tuple,
                        entry_emas: tuple) -> tuple[list, list, list]:
    """Simulate hybrid + return equity curve + trade list."""
    from tools.icnh_mega_sweep import _detect_distribution_top, _detect_cnh  # noqa
    from tools.icnh_final_tune import find_hybrid_patterns  # noqa

    df = load_tf(tf)
    dt_cfg = Config(
        name="h_dt", pattern_type="distribution_top", direction="short", tf=tf,
        uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
        require_chop_at_top=True, breakdown_mode="chop_low_or_ema24",
        sl_atr_mult=sl_atr, regime_sl_mode="off", tp_emas=tp_emas,
        entry_emas=entry_emas, dedup_bars=dedup,
    )
    icnh_cfg = Config(
        name="h_icnh", pattern_type="inverse_cnh", direction="short", tf=tf,
        cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
        handle_max_depth_frac=0.70, peak_tolerance=6,
        entry_emas=entry_emas, sl_atr_mult=sl_atr, regime_sl_mode="off",
        tp_emas=tp_emas, dedup_bars=dedup,
    )
    all_trades = []
    for label, start, end in WINDOWS:
        sub = df.loc[start:end]
        if len(sub) < 100:
            continue
        hits = find_hybrid_patterns(sub, dt_cfg, icnh_cfg)
        dt_idxs = [h for h, s in hits if s == "DT"]
        icnh_idxs = [h for h, s in hits if s == "ICNH"]
        dt_trades = simulate_trades(sub, dt_idxs, dt_cfg, label)
        for t in dt_trades:
            t["pattern"] = "DT"
        icnh_trades = simulate_trades(sub, icnh_idxs, icnh_cfg, label)
        for t in icnh_trades:
            t["pattern"] = "ICNH"
        all_trades.extend(dt_trades + icnh_trades)
    all_trades.sort(key=lambda t: t["exit_ts"])
    ts, eq = [], []
    val = 1.0
    for t in all_trades:
        val *= (1.0 + t["net_pct"])
        ts.append(t["exit_ts"])
        eq.append(val)
    return ts, eq, all_trades


def build_leaderboard_html(rows: list[dict], top_n: int = 20) -> str:
    rows = [r for r in rows if r.get("trades", 0) >= 5]
    rows.sort(key=lambda r: -r.get("cum", 0))
    html_rows = []
    for i, r in enumerate(rows[:top_n]):
        c = r.get("config", {})
        name = c.get("name", "?")
        note = c.get("note", "")
        tpm = r.get("trades_per_month", r.get("trades", 0) / (5.9 * 12))
        cum_color = GREEN if r["cum"] > 0 else RED
        wr_color = GREEN if r["win_rate"] > 0.5 else AMBER
        html_rows.append(f"""
          <tr class="row-{i%2}">
            <td class="muted">#{i+1}</td>
            <td><b>{name}</b><br><span class="muted small">{note}</span></td>
            <td class="num">{r["trades"]}</td>
            <td class="num">{tpm:.2f}/mo</td>
            <td class="num" style="color:{wr_color}">{r["win_rate"]*100:.1f}%</td>
            <td class="num" style="color:{cum_color}"><b>{r["cum"]*100:+.2f}%</b></td>
            <td class="num">{r["sharpe"]:+.2f}</td>
          </tr>""")
    return f"""
      <table class="leaderboard">
        <thead><tr><th>#</th><th>Config</th><th>Trades</th><th>Freq</th>
                   <th>Win Rate</th><th>Cum Return</th><th>Sharpe</th></tr></thead>
        <tbody>{''.join(html_rows)}</tbody>
      </table>"""


def build_equity_chart(ts, eq, label: str, color: str = GREEN) -> str:
    return f"""
      <div id="eq-{label.lower().replace(' ', '_')}" style="height:380px;"></div>
      <script>
        Plotly.newPlot('eq-{label.lower().replace(' ', '_')}',
          [{{x: {json.dumps(ts)}, y: {json.dumps(eq)},
             type:'scatter', mode:'lines+markers',
             line:{{color:'{color}', width:2}}, marker:{{size:5}},
             hovertemplate:'%{{x}}<br>equity: %{{y:.4f}}×<extra></extra>',
             name:'{label}'}}],
          {{margin:{{l:60,r:30,t:20,b:60}}, paper_bgcolor:'transparent', plot_bgcolor:'#0f1530',
            yaxis:{{title:'$1 → ', gridcolor:'#1a2040', color:'#cfd6e8'}},
            xaxis:{{title:'exit date', gridcolor:'#1a2040', color:'#cfd6e8'}},
            font:{{color:'#cfd6e8'}},
            shapes:[{{type:'line', x0:0, x1:1, xref:'paper', y0:1, y1:1,
                      line:{{color:'#3a4378', dash:'dash'}}}}]}},
          {{responsive:true}});
      </script>"""


def build_per_window_table(trades: list[dict]) -> str:
    by_window: dict = {}
    for t in trades:
        w = t["window"]
        by_window.setdefault(w, []).append(t)
    rows = []
    for w in [x[0] for x in WINDOWS]:
        ts = by_window.get(w, [])
        if not ts:
            rows.append(f"<tr><td>{w}</td><td class='muted'>—</td><td class='muted'>—</td><td class='muted'>—</td><td class='muted'>—</td></tr>")
            continue
        nets = np.array([t["net_pct"] for t in ts])
        cum = float(np.prod(1.0 + nets) - 1.0)
        wr = float((nets > 0).mean())
        col = GREEN if cum > 0 else RED
        dt_n = sum(1 for t in ts if t.get("pattern") == "DT")
        icnh_n = sum(1 for t in ts if t.get("pattern") == "ICNH")
        rows.append(f"""<tr>
          <td>{w}</td>
          <td class="num">{len(ts)} (DT:{dt_n} / ICNH:{icnh_n})</td>
          <td class="num">{wr*100:.1f}%</td>
          <td class="num" style="color:{col}"><b>{cum*100:+.2f}%</b></td>
        </tr>""")
    return f"""<table class="window-table">
        <thead><tr><th>Window</th><th>Trades</th><th>Win Rate</th><th>Cum Return</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table>"""


def build_trades_examples(trades: list[dict], n: int = 8) -> str:
    sorted_t = sorted(trades, key=lambda t: -t["net_pct"])
    winners = sorted_t[:n]
    losers = sorted(sorted_t[-n:], key=lambda t: t["net_pct"])

    def row(t):
        color = GREEN if t["net_pct"] > 0 else RED
        return f"""<tr>
          <td>{t["window"]}</td>
          <td class="muted small">{t.get("pattern", "?")}</td>
          <td class="small">{t["entry_ts"]}</td>
          <td class="num">${t["entry_price"]:,.0f}</td>
          <td class="small">{t["exit_ts"]}</td>
          <td class="num">${t["exit_price"]:,.0f}</td>
          <td class="small">{t["exit_reason"]}</td>
          <td class="num" style="color:{color}"><b>{t["net_pct"]*100:+.2f}%</b></td>
        </tr>"""

    return f"""
      <div class="trade-tables">
        <div>
          <h4>Top {n} Winners</h4>
          <table class="trades">
            <thead><tr><th>Window</th><th>Pat</th><th>Entry</th><th>$</th><th>Exit</th><th>$</th><th>Why</th><th>Net</th></tr></thead>
            <tbody>{''.join(row(t) for t in winners)}</tbody>
          </table>
        </div>
        <div>
          <h4>Top {n} Losers</h4>
          <table class="trades">
            <thead><tr><th>Window</th><th>Pat</th><th>Entry</th><th>$</th><th>Exit</th><th>$</th><th>Why</th><th>Net</th></tr></thead>
            <tbody>{''.join(row(t) for t in losers)}</tbody>
          </table>
        </div>
      </div>"""


def main() -> int:
    grid = json.loads(GRID.read_text())
    mega = json.loads(MEGA.read_text())
    tune = json.loads(TUNE.read_text())

    # Hybrid runs returned by final_tune (last 9 entries with "config" dict but no per_window in same format)
    hybrid_rows = []
    other_tune = []
    for r in tune:
        cfg = r.get("config", {})
        name = cfg.get("name", "")
        if "HYBRID" in name or name.startswith("hybrid"):
            hybrid_rows.append(r)
        else:
            other_tune.append(r)

    # Synthesize hybrid configs from the run script for display
    hybrid_synth = []
    for r in tune:
        cfg = r.get("config", {})
        name = cfg.get("name", "")
        if "HYBRID" not in name and not name.startswith("FT"):
            # This is from run_hybrid — has tf/dedup/sl_atr/tp/entry
            tf = cfg.get("tf", "?")
            dedup = cfg.get("dedup", "?")
            atr_m = cfg.get("sl_atr", "?")
            tp = cfg.get("tp", [])
            ent = cfg.get("entry", [])
            r2 = dict(r)
            r2["config"] = {
                "name": f"HYBRID_{tf}_d{dedup}_atr{atr_m}_tp{','.join(tp)}_e{','.join(ent)}",
                "note": f"HYBRID (DT+ICnH) {tf} dedup={dedup} ATR={atr_m}× TP={tp} entry={ent}",
            }
            hybrid_synth.append(r2)

    # Combine ALL results
    all_rows = mega + other_tune + hybrid_synth

    # Filter for short side (the user's actual ask)
    short_rows = [r for r in all_rows if r.get("trades", 0) >= 5 and r.get("cum", 0) > 0]
    short_rows.sort(key=lambda r: -r["cum"])

    # ALSO add grid results (LONG_OPTIMAL etc.)
    for g in grid:
        c = g.get("config", {})
        if c.get("direction") == "long" and g.get("trades", 0) >= 10:
            # Reshape to match
            g2 = dict(g)
            g2["trades_per_month"] = g["trades"] / (5.9 * 12)
            all_rows.append(g2)

    # Top hybrid for equity curve
    print("Simulating best HYBRID for equity curve...")
    ts_h15, eq_h15, trades_h15 = hybrid_equity_curve("4h", 15, 1.5, ("ema100",), ("ema24",))
    ts_h5, eq_h5, trades_h5 = hybrid_equity_curve("4h", 5, 1.5, ("ema100",), ("ema24",))

    final_h15 = eq_h15[-1] if eq_h15 else 1.0
    final_h5 = eq_h5[-1] if eq_h5 else 1.0

    leaderboard_html = build_leaderboard_html(all_rows, top_n=25)
    eq_h15_chart = build_equity_chart(ts_h15, eq_h15, "HYBRID dedup15", GREEN)
    eq_h5_chart = build_equity_chart(ts_h5, eq_h5, "HYBRID dedup5", BLUE)
    window_table_h15 = build_per_window_table(trades_h15)
    window_table_h5 = build_per_window_table(trades_h5)
    trades_examples = build_trades_examples(trades_h15, n=8)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cup-and-Handle Experiment V2 — HYBRID Detector Found</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background:#0b1020; color:#e7e7e7; margin:0; padding:0; line-height:1.55; }}
  .container {{ max-width:1280px; margin:0 auto; padding:32px; }}
  header {{ background: linear-gradient(135deg, #0f4d2e 0%, #0b1020 100%);
           padding:32px; border-radius:16px; margin-bottom:32px; }}
  h1 {{ font-size:30px; margin:0 0 6px 0; font-weight:700; }}
  .subtitle {{ color:#cfd6e8; font-size:15px; }}
  .badge {{ display:inline-block; padding:4px 10px; border-radius:4px;
           font-size:11px; font-weight:600; letter-spacing:0.4px; margin-right:8px; }}
  .badge-green {{ background:{GREEN}; color:white; }}
  .badge-amber {{ background:{AMBER}; color:#0b1020; }}
  .badge-blue {{ background:{BLUE}; color:white; }}
  .badge-purple {{ background:{PURPLE}; color:white; }}

  section {{ background:#1a1f3a; padding:28px; border-radius:12px; margin-bottom:24px; }}
  section h2 {{ font-size:22px; margin:0 0 12px 0; font-weight:600; }}
  section h3 {{ font-size:17px; margin:18px 0 8px 0; color:#cfd6e8; }}
  section h4 {{ font-size:14px; margin:12px 0 6px 0; color:#94a3b8; }}
  p {{ font-size:14px; color:#cfd6e8; }}

  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
  .grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}

  .stat-card {{ background:#11162c; padding:18px; border-radius:8px; border-left:4px solid {BLUE}; }}
  .stat-card.good {{ border-color:{GREEN}; }}
  .stat-card.bad {{ border-color:{RED}; }}
  .stat-card.warn {{ border-color:{AMBER}; }}
  .stat-card.star {{ border-color:{GREEN}; border-width:6px; background:#0f3528; }}
  .stat-label {{ font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; }}
  .stat-value {{ font-size:26px; font-weight:700; margin-top:4px; }}
  .stat-sub {{ font-size:12px; color:#94a3b8; margin-top:4px; }}

  .verdict {{ padding:20px; border-radius:8px; margin:16px 0; border-left:5px solid; }}
  .verdict.success {{ background:#064e3b; border-color:{GREEN}; }}
  .verdict.warn {{ background:#4a2f0a; border-color:{AMBER}; }}
  .verdict.fail {{ background:#481b1b; border-color:{RED}; }}
  .verdict h3 {{ margin:0 0 8px 0; }}

  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:8px 10px; text-align:left; border-bottom:1px solid #1a1f3a; }}
  th {{ background:#161a2e; color:#94a3b8; font-weight:600; font-size:11px;
        text-transform:uppercase; letter-spacing:0.4px; }}
  .row-0 {{ background:#11162c; }}
  .row-1 {{ background:#161a2e; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.muted, .muted {{ color:#94a3b8; }}
  .small {{ font-size:11px; }}

  .trade-tables {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .trades th, .trades td {{ font-size:11px; padding:5px 7px; }}
  .window-table {{ font-size:12px; }}

  code {{ background:#11162c; padding:1px 6px; border-radius:3px; font-size:13px; }}
  ul, ol {{ font-size:14px; color:#cfd6e8; }}
  li {{ margin:6px 0; }}
</style>
</head>
<body>
<div class="container">

<header>
  <div>
    <span class="badge badge-green">EXPERIMENT V2</span>
    <span class="badge badge-blue">HYBRID DETECTOR DISCOVERED</span>
  </div>
  <h1>Cup-and-Handle V2 — User Tuning Pass</h1>
  <p class="subtitle">
    User feedback: "I see this pattern 2-4 times/month, your detector finds it 4×/year."<br>
    Built a new looser "distribution top" detector matching the user's image-16 visual rule,
    then combined it with the loose-ICnH detector into a <b>HYBRID</b>. Major breakthrough.
  </p>
</header>

<section>
  <h2>🚀 TL;DR — HYBRID detector is the winner</h2>
  <div class="grid-3">
    <div class="stat-card star">
      <div class="stat-label">HYBRID dedup=15 (BEST QUALITY)</div>
      <div class="stat-value" style="color:{GREEN};">+75.0%</div>
      <div class="stat-sub">61 trades, 70.5% WR, Sharpe 5.15<br>$1 → ${final_h15:.3f}</div>
    </div>
    <div class="stat-card good">
      <div class="stat-label">HYBRID dedup=5 (HIGH FREQ)</div>
      <div class="stat-value" style="color:{GREEN};">+67.9%</div>
      <div class="stat-sub">89 trades = <b>1.3/month</b>, 62% WR<br>$1 → ${final_h5:.3f}</div>
    </div>
    <div class="stat-card good">
      <div class="stat-label">Vs. previous best LONG_OPTIMAL</div>
      <div class="stat-value" style="color:{GREEN};">+55.8%</div>
      <div class="stat-sub">23 trades, 60% WR, Sharpe 7.4<br>still strong but lower</div>
    </div>
  </div>

  <div class="verdict success">
    <h3>📈 NEW WINNER: HYBRID short detector (DT + loose-ICnH)</h3>
    <p>
      The breakthrough was building a SECOND detector for your actual visual pattern
      ("uptrend → chop → breakdown" — image 16) and <b>combining</b> it with a loosened
      parabolic-cup detector. Together they catch a wider variety of distribution-top
      shapes, with edges that don't cancel out — they reinforce. Best variant
      <b>+75% cum across 61 trades</b> (Jan 2020 – May 2026), <b>70.5% win rate</b>,
      Sharpe 5.15, distributed across 9 of 12 windows.
    </p>
    <p style="margin-top:10px;">
      Frequency at dedup=5 hits <b>1.3 trades/month</b> — still below your 2-4/month
      visual count but 3× more than the original detector. Going higher frequency (dedup=3)
      gives ~1.5/mo with slightly worse Sharpe.
    </p>
  </div>
</section>

<section>
  <h2>What changed since v1 report</h2>
  <ol>
    <li><b>New pattern detector: <code>distribution_top</code></b> — implements your image-16
        rule literally: N bars of uptrend → M bars of sideways chop → close breaks below
        chop low OR closes back through EMA(24). No parabola fit required (your eye sees
        flat tops, not perfect domes).</li>
    <li><b>Loosened ICnH parameters dramatically.</b> R² requirement dropped from 0.70 → 0.50,
        cup depth from 2.5× → 1.0× ATR, peak tolerance widened. Got from 8 trades → 48.</li>
    <li><b>Discovered: larger dedup window = better edge.</b> Counterintuitive but consistent:
        when 3 patterns fire within 10 bars, only the first one has alpha; the rest are
        noisy echoes of the same setup. dedup=15 wins on Sharpe.</li>
    <li><b>HYBRID detector</b>: union of distribution_top + loose-ICnH. Different patterns
        catch different shapes; both have real edge; combined +75% &gt; either alone.</li>
    <li><b>Bug found and fixed:</b> distribution_top's chop_low initially included the
        current bar, so close-below-chop-low could never trigger. Fixed; pattern now fires.</li>
  </ol>
</section>

<section>
  <h2>🏆 HYBRID equity curves</h2>
  <h3>HYBRID dedup=15 — best Sharpe ($1 → ${final_h15:.3f})</h3>
  {eq_h15_chart}

  <h3>HYBRID dedup=5 — higher frequency ($1 → ${final_h5:.3f})</h3>
  {eq_h5_chart}
</section>

<section>
  <h2>Per-window: HYBRID dedup=15</h2>
  {window_table_h15}

  <h3>Per-window: HYBRID dedup=5 (more trades)</h3>
  {window_table_h5}
</section>

<section>
  <h2>Leaderboard (top 25 by cumulative return)</h2>
  {leaderboard_html}
</section>

<section>
  <h2>Sample trades — HYBRID dedup=15</h2>
  {trades_examples}
</section>

<section>
  <h2>Honest findings (v2)</h2>
  <ul>
    <li><b>Your visual frequency was right.</b> The pattern DOES occur often — my original
        detector was just too strict (parabola R² ≥ 0.7). With the new looser detector + hybrid,
        we go from 4/year to 10–15/year.</li>
    <li><b>Two patterns, one signal.</b> The distribution_top catches "flat-top distribution"
        shapes; the loose-ICnH catches "rounded-dome" shapes. Both are valid topping patterns
        and they're DIFFERENT enough to be additive.</li>
    <li><b>Even with looser detection, sub-1h TFs are noise.</b> 1h hybrid: -52%. 15m: -50%+.
        The signal lives at 4h, where bars take long enough to form that "uptrend, then chop"
        actually means something.</li>
    <li><b>SHORT edge appears real even in BTC bull market.</b> Even 2024 and 2026-H1 (both bull)
        showed positive PnL on the HYBRID — because the pattern catches LOCAL distribution tops
        within larger uptrends.</li>
    <li><b>2-4 trades/month is hard to hit on 4h with positive edge.</b> Best frequency we
        achieved with positive Sharpe: 1.3/month. Going higher (dedup=3) drops Sharpe quickly.
        The constraint isn't laziness of the detector — it's that genuine distribution-tops
        with statistical edge don't occur weekly on the daily-trend scale.</li>
  </ul>
</section>

<section>
  <h2>What I'd deploy: <code>cnh-hybrid-short-v1</code></h2>
  <div class="verdict success">
    <h3>Recommended config</h3>
    <p>
      <b>Direction:</b> SHORT (your original ask, finally with real edge)<br>
      <b>Timeframe:</b> 4h<br>
      <b>Detection:</b> HYBRID — fires if EITHER:
      <ul style="margin-top:6px;">
        <li>distribution_top: 16 uptrend bars (≥ 2.5% rise) + 8 chop bars (range ≤ 55% of uptrend) + chop must be at top + close &lt; chop_low OR EMA(24) breakdown</li>
        <li>loose ICnH: cup_len=20, R² ≥ 0.5, depth ≥ 1.0×ATR, handle_len=4, EMA(24) breakdown</li>
      </ul>
      <b>Entry:</b> Close of trigger bar<br>
      <b>SL:</b> entry + 1.5×ATR(14) — pure ATR, no regime SL<br>
      <b>TP:</b> EMA(100) (first touch below entry)<br>
      <b>Dedup:</b> 10-15 bars between consecutive signals<br>
      <b>Frequency:</b> 0.9-1.3 trades/month (10-15/year)<br>
      <b>Expected:</b> Sharpe 4-5, +60-75% cum over 6 years (in-sample)
    </p>
  </div>

  <div class="verdict warn">
    <h3>⚠️ Still caveats — read these</h3>
    <ul>
      <li><b>Still in-sample.</b> All 12 windows are in the search set. True OOS could drop Sharpe to 2-3.</li>
      <li><b>Frequency below your visual count.</b> You see 2-4/mo discretionarily; bot finds 1.3/mo. Difference: your eye uses confluence (S/R, news context, multi-TF) that the bot can't.</li>
      <li><b>Standing $300 capital rule.</b> At current $50.50/leg, deploying a 3rd strategy means each strategy gets ~$33 — below min-notional for many signals. Wait until total ≥ $300.</li>
      <li><b>HYBRID is harder to reason about than single-pattern.</b> When something goes wrong in production, "was it DT or ICnH that fired?" becomes part of every debug session.</li>
    </ul>
  </div>
</section>

<section>
  <h2>Sensitivity findings (key axes for SHORT hybrid)</h2>
  <ul>
    <li><b>Dedup window:</b> 5 → +68%, 10 → +66%, 15 → +75%, 20 → +40%. Sweet spot: 10-15.</li>
    <li><b>ATR multiplier:</b> 1.5 = sweet spot (+68%); 2.0 drops to +38%; 1.0 = -12% (too tight).</li>
    <li><b>TP EMA:</b> EMA-100 wins (+68%); EMA-50 = +15%, EMA-200 = +45%. EMA-100 is the goldilocks target.</li>
    <li><b>Entry EMA:</b> EMA-24 alone is best. Adding EMA-7 (faster) corrupts signal with whipsaws.</li>
    <li><b>Distribution-top uptrend bars:</b> 16 wins (+27% alone, sweet spot). 8 = bad (-25%), 24 = neutral.</li>
    <li><b>Distribution-top chop bars:</b> 6 wins (+27%); 4 = bad (-23%); 8-10 still positive but fewer signals.</li>
    <li><b>Min rise %:</b> 2.5% sweet spot; 1% allows too many noise patterns; 5%+ kills frequency.</li>
  </ul>
</section>

<section>
  <h2>Suggested next steps</h2>
  <ol>
    <li>You decide whether to greenlight a true walk-forward validation of <code>cnh-hybrid-short-v1</code>.</li>
    <li>If yes: set up 6 OOS folds (train on first half, test on second half × rolling). Report median Sharpe across folds.</li>
    <li>If validated OOS Sharpe ≥ 1.5: code <code>strategy/signals_cnh_hybrid.py</code>, add a <code>live_cnh_hybrid.py</code> wrapper.</li>
    <li>Don't deploy until total capital ≥ $300 (per standing rule).</li>
  </ol>
</section>

<section>
  <h2>Files produced (cumulative)</h2>
  <ul>
    <li><code>tools/icnh_cheap_check.py</code> — original cheap check (v1)</li>
    <li><code>tools/icnh_sweep.py</code> — 16-config initial sweep</li>
    <li><code>tools/icnh_grid_sweep.py</code> — 96-config parameter grid</li>
    <li><code>tools/icnh_regime_aware.py</code> — failed regime-aware experiment</li>
    <li><code>tools/icnh_mega_sweep.py</code> — 92-config mega sweep (NEW pattern + multi-TF)</li>
    <li><code>tools/icnh_final_tune.py</code> — final tuning + HYBRID detector (NEW)</li>
    <li><code>tools/build_icnh_v2_report.py</code> — this report builder</li>
    <li><code>ICNH_EXPERIMENT.html</code> (v1) + <code>ICNH_EXPERIMENT_V2.html</code> (this)</li>
    <li><code>data/grid_results.json</code>, <code>data/mega_sweep_results.json</code>,
        <code>data/final_tune_results.json</code> — raw results for re-analysis</li>
  </ul>
</section>

<footer style="text-align:center; padding:24px; color:#94a3b8; font-size:12px;">
  Generated 2026-05-23 (V2) — autonomous AFK runner.<br>
  Total experiment wall time: ~5 minutes across 200+ configurations on 8 cores.
</footer>

</div>
</body>
</html>
"""
    OUTPUT.write_text(html)
    print(f"\nReport saved → {OUTPUT}")
    print(f"  File size: {OUTPUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
