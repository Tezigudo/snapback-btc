"""Build VISUAL_EXPLAINER.html — diagrams of how the bot + strategy work.

Self-contained HTML with inline SVG. No JavaScript. Loads instantly.

Sections:
  1. System architecture (who talks to whom)
  2. Bot main loop (what happens every 5 seconds)
  3. Strategy entry logic (the 4-filter AND-gate)
  4. Sizing math worked out at $60 / $100 / $200 equity
  5. Trade lifecycle (signal → bracket orders → exit)
  6. Safety gate stack (every layer that can stop a trade)
  7. Kill switch math (when bot self-destructs)
  8. State machine view (bot's three states)
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CSS = """
  body { font: 14px/1.55 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1200px; margin: 24px auto; padding: 0 20px; color: #2c2c2c; background: #fafafa; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  h2 { margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
  h3 { margin-top: 26px; color: #555; }
  .sub { color: #666; font-style: italic; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
          padding: 18px 24px; margin: 16px 0; }
  .note { background: #fff8e1; border-left: 4px solid #f57c00; padding: 10px 16px; margin: 14px 0; }
  .good { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 10px 16px; margin: 14px 0; }
  .danger { background: #ffebee; border-left: 4px solid #c62828; padding: 10px 16px; margin: 14px 0; }
  code { background: #f3f3f3; padding: 1px 6px; border-radius: 3px; font-size: 12px;
         font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th, td { padding: 6px 12px; border: 1px solid #ddd; text-align: left; }
  th { background: #eee; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0; font-size: 12px; }
  .legend > div { display: flex; align-items: center; gap: 6px; }
  .swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
  svg { display: block; max-width: 100%; height: auto; background: #fff; border-radius: 6px;
        border: 1px solid #e0e0e0; margin: 8px 0; }
"""

# ----------------------------------------------------------------------------
# Section 1: System architecture
# ----------------------------------------------------------------------------
ARCHITECTURE_SVG = """
<svg viewBox="0 0 1080 480" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#546e7a"/>
    </marker>
  </defs>
  <style>
    .lbl { font: 13px sans-serif; fill: #2c2c2c; }
    .small { font: 11px sans-serif; fill: #666; }
    .box { fill: #fff; stroke: #455a64; stroke-width: 1.5; rx: 8; }
    .exch { fill: #fff3e0; stroke: #e65100; }
    .bot { fill: #e3f2fd; stroke: #1565c0; }
    .data { fill: #f3e5f5; stroke: #6a1b9a; }
    .user { fill: #e8f5e9; stroke: #2e7d32; }
    .link { stroke: #546e7a; stroke-width: 1.5; fill: none; }
    .read { stroke-dasharray: 6 4; }
    .arrow-head { fill: #546e7a; }
  </style>

  <!-- Exchange box -->
  <rect class="box exch" x="40" y="60" width="220" height="160"/>
  <text class="lbl" x="150" y="90" text-anchor="middle" font-weight="600">Binance Futures</text>
  <text class="small" x="150" y="108" text-anchor="middle">USDM perpetuals · mainnet</text>
  <text class="small" x="150" y="135" text-anchor="middle">OHLCV (15m)</text>
  <text class="small" x="150" y="152" text-anchor="middle">funding rate (8h)</text>
  <text class="small" x="150" y="169" text-anchor="middle">balance · positions</text>
  <text class="small" x="150" y="190" text-anchor="middle" fill="#c62828" font-weight="600">create_order ← real money</text>

  <!-- Bot center box -->
  <rect class="box bot" x="400" y="40" width="280" height="380"/>
  <text class="lbl" x="540" y="68" text-anchor="middle" font-weight="700">bot.py (the daemon)</text>
  <text class="small" x="540" y="84" text-anchor="middle">5-second loop</text>

  <rect class="box bot" x="420" y="100" width="240" height="42"/>
  <text class="lbl" x="540" y="126" text-anchor="middle">1. heartbeat + HALT check</text>

  <rect class="box bot" x="420" y="152" width="240" height="42"/>
  <text class="lbl" x="540" y="178" text-anchor="middle">2. fetch equity + kill switch</text>

  <rect class="box bot" x="420" y="204" width="240" height="42"/>
  <text class="lbl" x="540" y="230" text-anchor="middle">3. time-stop check on position</text>

  <rect class="box bot" x="420" y="256" width="240" height="60"/>
  <text class="lbl" x="540" y="280" text-anchor="middle">4. evaluate signal on last</text>
  <text class="lbl" x="540" y="298" text-anchor="middle">closed 15m bar</text>

  <rect class="box bot" x="420" y="326" width="240" height="60"/>
  <text class="lbl" x="540" y="350" text-anchor="middle">5. size + place bracket</text>
  <text class="lbl" x="540" y="368" text-anchor="middle">(market + SL + TP)</text>

  <!-- State store -->
  <rect class="box data" x="820" y="60" width="220" height="160"/>
  <text class="lbl" x="930" y="90" text-anchor="middle" font-weight="600">data/state.db</text>
  <text class="small" x="930" y="108" text-anchor="middle">SQLite WAL</text>
  <text class="small" x="930" y="135" text-anchor="middle">deploy_start_equity</text>
  <text class="small" x="930" y="152" text-anchor="middle">fills (timestamped)</text>
  <text class="small" x="930" y="169" text-anchor="middle">events (JSONL backup)</text>
  <text class="small" x="930" y="190" text-anchor="middle">survives bot restart</text>

  <!-- Logs -->
  <rect class="box data" x="820" y="240" width="220" height="80"/>
  <text class="lbl" x="930" y="270" text-anchor="middle" font-weight="600">logs/</text>
  <text class="small" x="930" y="290" text-anchor="middle">bot.jsonl · console.log</text>
  <text class="small" x="930" y="307" text-anchor="middle">heartbeat (touched each tick)</text>

  <!-- User -->
  <rect class="box user" x="820" y="340" width="220" height="80"/>
  <text class="lbl" x="930" y="370" text-anchor="middle" font-weight="600">You</text>
  <text class="small" x="930" y="390" text-anchor="middle">via email (SMTP)</text>
  <text class="small" x="930" y="407" text-anchor="middle">via tmux session</text>

  <!-- Arrows: Bot ↔ Exchange -->
  <line class="link read" x1="400" y1="160" x2="260" y2="120" marker-end="url(#arrow)"/>
  <text class="small" x="328" y="135" text-anchor="middle">fetch (read)</text>

  <line class="link" x1="400" y1="350" x2="260" y2="190" marker-end="url(#arrow)"/>
  <text class="small" x="324" y="282" text-anchor="middle" fill="#c62828">create_order</text>

  <!-- Arrows: Bot → state.db / logs -->
  <line class="link" x1="680" y1="140" x2="820" y2="140" marker-end="url(#arrow)"/>
  <text class="small" x="750" y="130" text-anchor="middle">record_fill / event</text>

  <line class="link" x1="680" y1="280" x2="820" y2="280" marker-end="url(#arrow)"/>
  <text class="small" x="750" y="270" text-anchor="middle">jsonl + heartbeat</text>

  <!-- Arrow: Bot → User (email) -->
  <line class="link" x1="680" y1="380" x2="820" y2="380" marker-end="url(#arrow)"/>
  <text class="small" x="750" y="370" text-anchor="middle">SMTP alerts</text>

  <!-- Legend bottom -->
  <rect width="14" height="14" fill="#e3f2fd" stroke="#1565c0" x="40" y="450"/>
  <text class="small" x="62" y="461">bot</text>
  <rect width="14" height="14" fill="#fff3e0" stroke="#e65100" x="110" y="450"/>
  <text class="small" x="132" y="461">exchange</text>
  <rect width="14" height="14" fill="#f3e5f5" stroke="#6a1b9a" x="210" y="450"/>
  <text class="small" x="232" y="461">data / logs</text>
  <rect width="14" height="14" fill="#e8f5e9" stroke="#2e7d32" x="330" y="450"/>
  <text class="small" x="352" y="461">you</text>
  <line class="link read" x1="400" y1="457" x2="430" y2="457"/>
  <text class="small" x="440" y="461">read-only</text>
  <line class="link" x1="500" y1="457" x2="530" y2="457" marker-end="url(#arrow)"/>
  <text class="small" x="540" y="461">writes / orders</text>
</svg>
"""

# ----------------------------------------------------------------------------
# Section 2: Main loop flowchart
# ----------------------------------------------------------------------------
LOOP_SVG = """
<svg viewBox="0 0 1080 760" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#455a64"/>
    </marker>
  </defs>
  <style>
    .step { fill: #fff; stroke: #455a64; stroke-width: 1.5; rx: 8; }
    .decision { fill: #fff8e1; stroke: #f57c00; stroke-width: 1.5; }
    .terminal { fill: #ffebee; stroke: #c62828; stroke-width: 1.8; rx: 10; }
    .ok { fill: #e8f5e9; stroke: #2e7d32; stroke-width: 1.5; rx: 8; }
    .lbl { font: 13px sans-serif; fill: #2c2c2c; }
    .small { font: 11px sans-serif; fill: #666; }
    .link { stroke: #455a64; stroke-width: 1.4; fill: none; }
  </style>

  <!-- Start -->
  <rect class="ok" x="450" y="20" width="180" height="40"/>
  <text class="lbl" x="540" y="46" text-anchor="middle" font-weight="600">Loop tick (every 5s)</text>

  <line class="link" x1="540" y1="60" x2="540" y2="84" marker-end="url(#arrow2)"/>

  <!-- Step 1: heartbeat -->
  <rect class="step" x="450" y="86" width="180" height="36"/>
  <text class="lbl" x="540" y="110" text-anchor="middle">touch data/heartbeat</text>

  <line class="link" x1="540" y1="122" x2="540" y2="142" marker-end="url(#arrow2)"/>

  <!-- Step 2: HALT decision -->
  <polygon class="decision" points="540,142 660,180 540,218 420,180"/>
  <text class="lbl" x="540" y="178" text-anchor="middle">HALT file?</text>
  <text class="small" x="540" y="194" text-anchor="middle">data/HALT exists?</text>

  <!-- HALT yes: terminal -->
  <line class="link" x1="660" y1="180" x2="780" y2="180" marker-end="url(#arrow2)"/>
  <text class="small" x="720" y="172" text-anchor="middle">yes</text>
  <rect class="terminal" x="780" y="160" width="240" height="44"/>
  <text class="lbl" x="900" y="186" text-anchor="middle" font-weight="600">flatten · email · exit</text>

  <!-- HALT no: continue -->
  <line class="link" x1="540" y1="218" x2="540" y2="240" marker-end="url(#arrow2)"/>
  <text class="small" x="556" y="232">no</text>

  <!-- Step 3: fetch equity -->
  <rect class="step" x="430" y="242" width="220" height="36"/>
  <text class="lbl" x="540" y="266" text-anchor="middle">fetch equity (USDT)</text>

  <line class="link" x1="540" y1="278" x2="540" y2="298" marker-end="url(#arrow2)"/>

  <!-- Kill switch decision -->
  <polygon class="decision" points="540,298 700,340 540,382 380,340"/>
  <text class="lbl" x="540" y="336" text-anchor="middle">equity &lt; start × 0.82?</text>
  <text class="small" x="540" y="354" text-anchor="middle">(−18% kill switch)</text>

  <line class="link" x1="700" y1="340" x2="780" y2="340" marker-end="url(#arrow2)"/>
  <text class="small" x="740" y="332" text-anchor="middle">yes</text>
  <rect class="terminal" x="780" y="320" width="240" height="44"/>
  <text class="lbl" x="900" y="346" text-anchor="middle" font-weight="600">KILL · HALT · email · exit</text>

  <line class="link" x1="540" y1="382" x2="540" y2="402" marker-end="url(#arrow2)"/>
  <text class="small" x="556" y="396">no</text>

  <!-- Position check -->
  <polygon class="decision" points="540,402 700,440 540,478 380,440"/>
  <text class="lbl" x="540" y="438" text-anchor="middle">in a position?</text>

  <!-- yes → time-stop -->
  <line class="link" x1="700" y1="440" x2="800" y2="440" marker-end="url(#arrow2)"/>
  <text class="small" x="750" y="432" text-anchor="middle">yes</text>
  <rect class="step" x="800" y="420" width="240" height="40"/>
  <text class="lbl" x="920" y="445" text-anchor="middle">held &gt; 14d? → close</text>
  <line class="link" x1="920" y1="460" x2="920" y2="720"/>
  <line class="link" x1="920" y1="720" x2="540" y2="720"/>
  <line class="link" x1="540" y1="720" x2="540" y2="698" marker-end="url(#arrow2)"/>

  <!-- no → fetch bars -->
  <line class="link" x1="540" y1="478" x2="540" y2="500" marker-end="url(#arrow2)"/>
  <text class="small" x="556" y="494">no</text>
  <rect class="step" x="380" y="500" width="320" height="40"/>
  <text class="lbl" x="540" y="525" text-anchor="middle">fetch latest 15m bars + funding rate</text>

  <line class="link" x1="540" y1="540" x2="540" y2="562" marker-end="url(#arrow2)"/>

  <!-- evaluate -->
  <rect class="step" x="380" y="562" width="320" height="40"/>
  <text class="lbl" x="540" y="586" text-anchor="middle">evaluate signal (RSI + vol + EMA + funding)</text>

  <line class="link" x1="540" y1="602" x2="540" y2="624" marker-end="url(#arrow2)"/>

  <!-- signal decision -->
  <polygon class="decision" points="540,624 690,662 540,700 390,662"/>
  <text class="lbl" x="540" y="658" text-anchor="middle">signal fires?</text>
  <text class="small" x="540" y="676" text-anchor="middle">LONG or SHORT?</text>

  <line class="link" x1="390" y1="662" x2="100" y2="662" marker-end="url(#arrow2)"/>
  <text class="small" x="250" y="654" text-anchor="middle">no</text>
  <rect class="step" x="80" y="640" width="200" height="44"/>
  <text class="lbl" x="180" y="666" text-anchor="middle">sleep 5s</text>
  <line class="link" x1="180" y1="640" x2="180" y2="40"/>
  <line class="link" x1="180" y1="40" x2="450" y2="40" marker-end="url(#arrow2)"/>

  <line class="link" x1="690" y1="662" x2="900" y2="662" marker-end="url(#arrow2)"/>
  <text class="small" x="800" y="654" text-anchor="middle">yes</text>
  <rect class="ok" x="900" y="640" width="160" height="44"/>
  <text class="lbl" x="980" y="666" text-anchor="middle" font-weight="600">place bracket</text>
</svg>
"""

# ----------------------------------------------------------------------------
# Section 3: Strategy entry logic (AND gate)
# ----------------------------------------------------------------------------
STRATEGY_SVG = """
<svg viewBox="0 0 1080 460" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#455a64"/>
    </marker>
  </defs>
  <style>
    .filter { fill: #e3f2fd; stroke: #1565c0; stroke-width: 1.5; rx: 8; }
    .gate { fill: #fff8e1; stroke: #f57c00; stroke-width: 2; rx: 10; }
    .long { fill: #e8f5e9; stroke: #2e7d32; stroke-width: 2; rx: 10; }
    .short { fill: #ffebee; stroke: #c62828; stroke-width: 2; rx: 10; }
    .skip { fill: #f5f5f5; stroke: #757575; stroke-width: 1.5; rx: 8; }
    .lbl { font: 13px sans-serif; fill: #2c2c2c; }
    .small { font: 11px sans-serif; fill: #666; }
    .link { stroke: #455a64; stroke-width: 1.4; fill: none; }
  </style>

  <text class="lbl" x="540" y="28" text-anchor="middle" font-weight="700" font-size="16">
    multifactor-v1: ALL 4 conditions must align
  </text>

  <!-- LONG side -->
  <text class="lbl" x="220" y="68" text-anchor="middle" font-weight="600" fill="#2e7d32">LONG entry</text>

  <rect class="filter" x="60" y="80" width="320" height="46"/>
  <text class="lbl" x="220" y="100" text-anchor="middle" font-weight="600">RSI(14) &lt; 40</text>
  <text class="small" x="220" y="116" text-anchor="middle">oversold on 15m bar</text>

  <rect class="filter" x="60" y="138" width="320" height="46"/>
  <text class="lbl" x="220" y="158" text-anchor="middle" font-weight="600">Volume &gt; 2× SMA(20)</text>
  <text class="small" x="220" y="174" text-anchor="middle">real interest, not a wick</text>

  <rect class="filter" x="60" y="196" width="320" height="46"/>
  <text class="lbl" x="220" y="216" text-anchor="middle" font-weight="600">Close &gt; EMA(200)</text>
  <text class="small" x="220" y="232" text-anchor="middle">confirmed uptrend</text>

  <rect class="filter" x="60" y="254" width="320" height="46"/>
  <text class="lbl" x="220" y="274" text-anchor="middle" font-weight="600">funding ≤ +0.05%</text>
  <text class="small" x="220" y="290" text-anchor="middle">not piling into crowded long</text>

  <!-- AND gate -->
  <rect class="gate" x="120" y="320" width="200" height="50"/>
  <text class="lbl" x="220" y="350" text-anchor="middle" font-weight="700">AND (all true)</text>

  <line class="link" x1="380" y1="103" x2="430" y2="320" marker-end="url(#arrow3)"/>
  <line class="link" x1="380" y1="161" x2="220" y2="320"/>
  <line class="link" x1="380" y1="219" x2="220" y2="320"/>
  <line class="link" x1="380" y1="277" x2="220" y2="320"/>

  <rect class="long" x="120" y="390" width="200" height="50"/>
  <text class="lbl" x="220" y="420" text-anchor="middle" font-weight="700" fill="#2e7d32">LONG</text>
  <line class="link" x1="220" y1="370" x2="220" y2="390" marker-end="url(#arrow3)"/>

  <!-- SHORT side -->
  <text class="lbl" x="860" y="68" text-anchor="middle" font-weight="600" fill="#c62828">SHORT entry</text>

  <rect class="filter" x="700" y="80" width="320" height="46"/>
  <text class="lbl" x="860" y="100" text-anchor="middle" font-weight="600">RSI(14) &gt; 70</text>
  <text class="small" x="860" y="116" text-anchor="middle">overbought on 15m bar</text>

  <rect class="filter" x="700" y="138" width="320" height="46"/>
  <text class="lbl" x="860" y="158" text-anchor="middle" font-weight="600">Volume &gt; 2× SMA(20)</text>
  <text class="small" x="860" y="174" text-anchor="middle">real interest, not a wick</text>

  <rect class="filter" x="700" y="196" width="320" height="46"/>
  <text class="lbl" x="860" y="216" text-anchor="middle" font-weight="600">Close &lt; EMA(200)</text>
  <text class="small" x="860" y="232" text-anchor="middle">confirmed downtrend</text>

  <rect class="filter" x="700" y="254" width="320" height="46"/>
  <text class="lbl" x="860" y="274" text-anchor="middle" font-weight="600">funding ≥ −0.05%</text>
  <text class="small" x="860" y="290" text-anchor="middle">not piling into crowded short</text>

  <rect class="gate" x="760" y="320" width="200" height="50"/>
  <text class="lbl" x="860" y="350" text-anchor="middle" font-weight="700">AND (all true)</text>

  <line class="link" x1="700" y1="103" x2="860" y2="320"/>
  <line class="link" x1="700" y1="161" x2="860" y2="320"/>
  <line class="link" x1="700" y1="219" x2="860" y2="320"/>
  <line class="link" x1="700" y1="277" x2="860" y2="320"/>

  <rect class="short" x="760" y="390" width="200" height="50"/>
  <text class="lbl" x="860" y="420" text-anchor="middle" font-weight="700" fill="#c62828">SHORT</text>
  <line class="link" x1="860" y1="370" x2="860" y2="390" marker-end="url(#arrow3)"/>

  <!-- Middle: SKIP -->
  <rect class="skip" x="420" y="390" width="240" height="50"/>
  <text class="lbl" x="540" y="412" text-anchor="middle" font-weight="600">any condition fails →</text>
  <text class="lbl" x="540" y="430" text-anchor="middle" font-weight="700">SKIP (no trade)</text>
</svg>
"""

# ----------------------------------------------------------------------------
# Section 4: Sizing math
# ----------------------------------------------------------------------------
SIZING_SVG = """
<svg viewBox="0 0 1080 380" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#455a64"/>
    </marker>
  </defs>
  <style>
    .input { fill: #e3f2fd; stroke: #1565c0; stroke-width: 1.5; rx: 8; }
    .formula { fill: #f3e5f5; stroke: #6a1b9a; stroke-width: 1.8; rx: 8; }
    .check { fill: #fff8e1; stroke: #f57c00; stroke-width: 1.5; rx: 8; }
    .pass { fill: #e8f5e9; stroke: #2e7d32; stroke-width: 2; rx: 10; }
    .fail { fill: #ffebee; stroke: #c62828; stroke-width: 2; rx: 10; }
    .lbl { font: 13px sans-serif; fill: #2c2c2c; }
    .mono { font: 12px ui-monospace, "SF Mono", Menlo, monospace; fill: #2c2c2c; }
    .small { font: 11px sans-serif; fill: #666; }
    .link { stroke: #455a64; stroke-width: 1.4; fill: none; }
  </style>

  <text class="lbl" x="540" y="28" text-anchor="middle" font-weight="700" font-size="16">
    How position size is computed
  </text>

  <!-- Inputs -->
  <rect class="input" x="40" y="60" width="200" height="42"/>
  <text class="lbl" x="140" y="80" text-anchor="middle">equity (USDT)</text>
  <text class="mono" x="140" y="96" text-anchor="middle">e.g. $60, $100, $200</text>

  <rect class="input" x="40" y="118" width="200" height="42"/>
  <text class="lbl" x="140" y="138" text-anchor="middle">price (BTC USDT)</text>
  <text class="mono" x="140" y="154" text-anchor="middle">e.g. $100,000</text>

  <rect class="input" x="40" y="176" width="200" height="42"/>
  <text class="lbl" x="140" y="196" text-anchor="middle">risk_per_trade_pct</text>
  <text class="mono" x="140" y="212" text-anchor="middle">2% (config)</text>

  <rect class="input" x="40" y="234" width="200" height="42"/>
  <text class="lbl" x="140" y="254" text-anchor="middle">sl_pct</text>
  <text class="mono" x="140" y="270" text-anchor="middle">1.5% (config)</text>

  <rect class="input" x="40" y="292" width="200" height="42"/>
  <text class="lbl" x="140" y="312" text-anchor="middle">leverage cap</text>
  <text class="mono" x="140" y="328" text-anchor="middle">20× (risk.py)</text>

  <!-- Formula -->
  <rect class="formula" x="320" y="100" width="380" height="80"/>
  <text class="lbl" x="510" y="124" text-anchor="middle" font-weight="700">target_qty (BTC)</text>
  <text class="mono" x="510" y="148" text-anchor="middle">= (equity × 0.02) / (price × 0.015)</text>
  <text class="mono" x="510" y="166" text-anchor="middle">= 1.33 × equity / price</text>

  <rect class="formula" x="320" y="200" width="380" height="80"/>
  <text class="lbl" x="510" y="224" text-anchor="middle" font-weight="700">cap_qty (BTC)</text>
  <text class="mono" x="510" y="248" text-anchor="middle">= (equity × 20 × 0.95) / price</text>
  <text class="mono" x="510" y="266" text-anchor="middle">= 19 × equity / price</text>

  <line class="link" x1="240" y1="140" x2="320" y2="140" marker-end="url(#arrow4)"/>
  <line class="link" x1="240" y1="240" x2="320" y2="240" marker-end="url(#arrow4)"/>

  <!-- Result -->
  <rect class="check" x="320" y="298" width="380" height="42"/>
  <text class="lbl" x="510" y="318" text-anchor="middle" font-weight="700">qty = min(target, cap)</text>
  <text class="mono" x="510" y="332" text-anchor="middle">(risk-based usually wins → ~1.33× effective leverage)</text>

  <line class="link" x1="510" y1="180" x2="510" y2="298" marker-end="url(#arrow4)"/>
  <line class="link" x1="510" y1="280" x2="510" y2="298"/>

  <!-- Exchange minimum check -->
  <rect class="check" x="740" y="100" width="300" height="120"/>
  <text class="lbl" x="890" y="124" text-anchor="middle" font-weight="700">exchange minimums</text>
  <text class="mono" x="890" y="148" text-anchor="middle">qty ≥ 0.001 BTC</text>
  <text class="mono" x="890" y="166" text-anchor="middle">notional ≥ $50</text>
  <text class="small" x="890" y="190" text-anchor="middle">if fail → skip signal</text>
  <text class="small" x="890" y="206" text-anchor="middle">(never scale up!)</text>

  <line class="link" x1="700" y1="319" x2="740" y2="180" marker-end="url(#arrow4)"/>

  <!-- Pass / Fail -->
  <rect class="pass" x="740" y="240" width="140" height="44"/>
  <text class="lbl" x="810" y="266" text-anchor="middle" font-weight="700" fill="#2e7d32">place order</text>

  <rect class="fail" x="900" y="240" width="140" height="44"/>
  <text class="lbl" x="970" y="266" text-anchor="middle" font-weight="700" fill="#c62828">skip signal</text>

  <line class="link" x1="810" y1="220" x2="810" y2="240" marker-end="url(#arrow4)"/>
  <line class="link" x1="970" y1="220" x2="970" y2="240" marker-end="url(#arrow4)"/>
</svg>
"""

# ----------------------------------------------------------------------------
# Section 5: Trade lifecycle
# ----------------------------------------------------------------------------
LIFECYCLE_SVG = """
<svg viewBox="0 0 1100 380" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#455a64"/>
    </marker>
  </defs>
  <style>
    .step { fill: #e3f2fd; stroke: #1565c0; stroke-width: 1.5; rx: 8; }
    .order { fill: #fff3e0; stroke: #e65100; stroke-width: 1.5; rx: 8; }
    .exit-sl { fill: #ffebee; stroke: #c62828; stroke-width: 1.8; rx: 8; }
    .exit-tp { fill: #e8f5e9; stroke: #2e7d32; stroke-width: 1.8; rx: 8; }
    .exit-time { fill: #f5f5f5; stroke: #757575; stroke-width: 1.5; rx: 8; }
    .exit-halt { fill: #fff8e1; stroke: #f57c00; stroke-width: 1.8; rx: 8; }
    .lbl { font: 13px sans-serif; fill: #2c2c2c; }
    .small { font: 11px sans-serif; fill: #666; }
    .link { stroke: #455a64; stroke-width: 1.4; fill: none; }
  </style>

  <text class="lbl" x="550" y="28" text-anchor="middle" font-weight="700" font-size="16">
    Trade lifecycle: from signal to exit
  </text>

  <!-- Signal fires -->
  <rect class="step" x="40" y="60" width="180" height="50"/>
  <text class="lbl" x="130" y="84" text-anchor="middle" font-weight="600">signal fires</text>
  <text class="small" x="130" y="100" text-anchor="middle">15m bar closes</text>

  <line class="link" x1="220" y1="85" x2="270" y2="85" marker-end="url(#arrow5)"/>

  <!-- Compute size + check minimums -->
  <rect class="step" x="270" y="60" width="180" height="50"/>
  <text class="lbl" x="360" y="84" text-anchor="middle" font-weight="600">size + check</text>
  <text class="small" x="360" y="100" text-anchor="middle">qty + notional minimums</text>

  <line class="link" x1="450" y1="85" x2="500" y2="85" marker-end="url(#arrow5)"/>

  <!-- 3 orders placed -->
  <rect class="order" x="500" y="60" width="220" height="50"/>
  <text class="lbl" x="610" y="84" text-anchor="middle" font-weight="700">place 3 orders</text>
  <text class="small" x="610" y="100" text-anchor="middle">market entry + SL stop + TP stop</text>

  <line class="link" x1="720" y1="85" x2="770" y2="85" marker-end="url(#arrow5)"/>

  <!-- Position open -->
  <rect class="step" x="770" y="60" width="180" height="50"/>
  <text class="lbl" x="860" y="84" text-anchor="middle" font-weight="700">position open</text>
  <text class="small" x="860" y="100" text-anchor="middle">bot waits, polls every 5s</text>

  <!-- Four exit paths -->
  <line class="link" x1="860" y1="110" x2="860" y2="140" marker-end="url(#arrow5)"/>
  <text class="lbl" x="860" y="158" text-anchor="middle" font-weight="600">which exit fires first?</text>

  <!-- SL -->
  <rect class="exit-sl" x="40" y="200" width="220" height="64"/>
  <text class="lbl" x="150" y="222" text-anchor="middle" font-weight="700" fill="#c62828">price hits SL (−1.5%)</text>
  <text class="small" x="150" y="240" text-anchor="middle">exchange triggers</text>
  <text class="small" x="150" y="254" text-anchor="middle">market close · reduce-only</text>

  <!-- TP -->
  <rect class="exit-tp" x="280" y="200" width="220" height="64"/>
  <text class="lbl" x="390" y="222" text-anchor="middle" font-weight="700" fill="#2e7d32">price hits TP (+3.0%)</text>
  <text class="small" x="390" y="240" text-anchor="middle">exchange triggers</text>
  <text class="small" x="390" y="254" text-anchor="middle">market close · reduce-only</text>

  <!-- Time stop -->
  <rect class="exit-time" x="520" y="200" width="220" height="64"/>
  <text class="lbl" x="630" y="222" text-anchor="middle" font-weight="700">held &gt; 14 days</text>
  <text class="small" x="630" y="240" text-anchor="middle">bot fires close_position</text>
  <text class="small" x="630" y="254" text-anchor="middle">on next 5s tick</text>

  <!-- HALT / kill -->
  <rect class="exit-halt" x="760" y="200" width="280" height="64"/>
  <text class="lbl" x="900" y="222" text-anchor="middle" font-weight="700">HALT or kill switch</text>
  <text class="small" x="900" y="240" text-anchor="middle">−18% equity OR data/HALT</text>
  <text class="small" x="900" y="254" text-anchor="middle">flatten + email + exit</text>

  <line class="link" x1="860" y1="170" x2="150" y2="200" marker-end="url(#arrow5)"/>
  <line class="link" x1="860" y1="170" x2="390" y2="200" marker-end="url(#arrow5)"/>
  <line class="link" x1="860" y1="170" x2="630" y2="200" marker-end="url(#arrow5)"/>
  <line class="link" x1="860" y1="170" x2="900" y2="200" marker-end="url(#arrow5)"/>

  <!-- Logging / state.db / email -->
  <rect class="step" x="280" y="300" width="540" height="46"/>
  <text class="lbl" x="550" y="320" text-anchor="middle" font-weight="600">every exit → record_fill → state.db, jsonl, email alert</text>
  <text class="small" x="550" y="336" text-anchor="middle">bot returns to "no position" state, ready for next signal</text>

  <line class="link" x1="150" y1="264" x2="280" y2="320"/>
  <line class="link" x1="390" y1="264" x2="400" y2="300"/>
  <line class="link" x1="630" y1="264" x2="600" y2="300"/>
  <line class="link" x1="900" y1="264" x2="820" y2="320"/>
</svg>
"""

# ----------------------------------------------------------------------------
# Section 6: Safety gate stack
# ----------------------------------------------------------------------------
SAFETY_SVG = """
<svg viewBox="0 0 1080 540" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#455a64"/>
    </marker>
  </defs>
  <style>
    .gate { fill: #fff8e1; stroke: #f57c00; stroke-width: 1.5; rx: 8; }
    .gate b { fill: #e3f2fd; stroke: #1565c0; }
    .pass { fill: #e8f5e9; stroke: #2e7d32; stroke-width: 2; rx: 10; }
    .fail { fill: #ffebee; stroke: #c62828; stroke-width: 1.8; rx: 8; }
    .lbl { font: 13px sans-serif; fill: #2c2c2c; }
    .small { font: 11px sans-serif; fill: #666; }
    .link { stroke: #455a64; stroke-width: 1.4; fill: none; }
  </style>

  <text class="lbl" x="540" y="28" text-anchor="middle" font-weight="700" font-size="16">
    Safety gate stack — a trade survives ALL these to actually place
  </text>

  <!-- Signal at top -->
  <rect class="pass" x="430" y="50" width="220" height="40"/>
  <text class="lbl" x="540" y="76" text-anchor="middle" font-weight="700">candidate signal</text>

  <line class="link" x1="540" y1="90" x2="540" y2="110" marker-end="url(#arrow6)"/>

  <!-- Gate 1: env -->
  <rect class="gate" x="380" y="110" width="320" height="36"/>
  <text class="lbl" x="540" y="134" text-anchor="middle">1. mainnet lockfile present  (env.py)</text>

  <line class="link" x1="540" y1="146" x2="540" y2="160" marker-end="url(#arrow6)"/>

  <!-- Gate 2: HALT -->
  <rect class="gate" x="380" y="160" width="320" height="36"/>
  <text class="lbl" x="540" y="184" text-anchor="middle">2. no data/HALT file  (bot.py loop)</text>

  <line class="link" x1="540" y1="196" x2="540" y2="210" marker-end="url(#arrow6)"/>

  <!-- Gate 3: kill switch -->
  <rect class="gate" x="380" y="210" width="320" height="36"/>
  <text class="lbl" x="540" y="234" text-anchor="middle">3. equity ≥ 82% × deploy_start  (kill switch)</text>

  <line class="link" x1="540" y1="246" x2="540" y2="260" marker-end="url(#arrow6)"/>

  <!-- Gate 4: position state -->
  <rect class="gate" x="380" y="260" width="320" height="36"/>
  <text class="lbl" x="540" y="284" text-anchor="middle">4. no existing position  (single-position rule)</text>

  <line class="link" x1="540" y1="296" x2="540" y2="310" marker-end="url(#arrow6)"/>

  <!-- Gate 5: bar dedupe -->
  <rect class="gate" x="380" y="310" width="320" height="36"/>
  <text class="lbl" x="540" y="334" text-anchor="middle">5. new 15m bar (one signal per bar)</text>

  <line class="link" x1="540" y1="346" x2="540" y2="360" marker-end="url(#arrow6)"/>

  <!-- Gate 6: symbol allowlist -->
  <rect class="gate" x="380" y="360" width="320" height="36"/>
  <text class="lbl" x="540" y="384" text-anchor="middle">6. symbol = BTC/USDT:USDT  (risk.py allowlist)</text>

  <line class="link" x1="540" y1="396" x2="540" y2="410" marker-end="url(#arrow6)"/>

  <!-- Gate 7: exchange minimums -->
  <rect class="gate" x="380" y="410" width="320" height="36"/>
  <text class="lbl" x="540" y="434" text-anchor="middle">7. qty ≥ 0.001 BTC AND notional ≥ $50</text>

  <line class="link" x1="540" y1="446" x2="540" y2="460" marker-end="url(#arrow6)"/>

  <!-- Gate 8: notional cap -->
  <rect class="gate" x="380" y="460" width="320" height="36"/>
  <text class="lbl" x="540" y="484" text-anchor="middle">8. notional ≤ $500  (risk.py MAX_NOTIONAL)</text>

  <line class="link" x1="540" y1="496" x2="540" y2="510" marker-end="url(#arrow6)"/>

  <!-- Final -->
  <rect class="pass" x="430" y="510" width="220" height="24"/>
  <text class="lbl" x="540" y="527" text-anchor="middle" font-weight="700">→ place market + SL + TP</text>

  <!-- Sidebar: failing any gate -->
  <rect class="fail" x="780" y="160" width="240" height="120"/>
  <text class="lbl" x="900" y="184" text-anchor="middle" font-weight="700" fill="#c62828">FAIL any gate?</text>
  <text class="small" x="900" y="208" text-anchor="middle">skip this signal,</text>
  <text class="small" x="900" y="224" text-anchor="middle">log to state.db,</text>
  <text class="small" x="900" y="240" text-anchor="middle">wait for next bar.</text>
  <text class="small" x="900" y="262" text-anchor="middle" fill="#c62828">never bypass.</text>

  <line class="link" x1="700" y1="128" x2="780" y2="180"/>
  <line class="link" x1="700" y1="478" x2="780" y2="260"/>
</svg>
"""

# ----------------------------------------------------------------------------
# Section 7: Kill switch math
# ----------------------------------------------------------------------------
KILL_SWITCH_SVG = """
<svg viewBox="0 0 1080 360" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis { stroke: #455a64; stroke-width: 1.5; fill: none; }
    .grid { stroke: #e0e0e0; stroke-width: 1; fill: none; }
    .start { stroke: #1565c0; stroke-width: 2; stroke-dasharray: 6 4; fill: none; }
    .kill { stroke: #c62828; stroke-width: 2.5; fill: none; }
    .curve { stroke: #1976d2; stroke-width: 2.5; fill: none; }
    .area { fill: #e3f2fd; opacity: 0.4; }
    .danger { fill: #ffcdd2; opacity: 0.5; }
    .lbl { font: 13px sans-serif; fill: #2c2c2c; }
    .small { font: 11px sans-serif; fill: #666; }
  </style>

  <text class="lbl" x="540" y="28" text-anchor="middle" font-weight="700" font-size="16">
    Kill switch zone: $60 deploy, −18% threshold
  </text>

  <!-- Background -->
  <rect class="area" x="100" y="60" width="900" height="160"/>
  <rect class="danger" x="100" y="220" width="900" height="50"/>

  <!-- Grid -->
  <line class="grid" x1="100" y1="60" x2="100" y2="270"/>
  <line class="grid" x1="1000" y1="60" x2="1000" y2="270"/>
  <line class="grid" x1="100" y1="270" x2="1000" y2="270"/>

  <!-- Axis labels -->
  <text class="small" x="80" y="65" text-anchor="end">$72 (+20%)</text>
  <text class="small" x="80" y="140" text-anchor="end">$60 (start)</text>
  <text class="small" x="80" y="220" text-anchor="end">$49.20 (−18%)</text>
  <text class="small" x="80" y="270" text-anchor="end">$48</text>

  <!-- Start line -->
  <line class="start" x1="100" y1="140" x2="1000" y2="140"/>
  <text class="small" x="1010" y="145" fill="#1565c0">deploy_start = $60</text>

  <!-- Kill line -->
  <line class="kill" x1="100" y1="220" x2="1000" y2="220"/>
  <text class="small" x="1010" y="225" fill="#c62828" font-weight="700">KILL = $49.20</text>

  <!-- Example equity curve (illustrative wiggle) -->
  <polyline class="curve" points="100,140 200,128 280,148 360,135 440,160 520,150 600,180
                                  680,170 760,195 820,175 900,158 1000,148"/>
  <text class="small" x="100" y="55" fill="#1565c0">example equity curve over 30 days</text>

  <!-- Annotations -->
  <text class="small" x="540" y="105" text-anchor="middle" font-weight="600" fill="#1565c0">
    SAFE zone (bot trades normally)
  </text>
  <text class="small" x="540" y="252" text-anchor="middle" font-weight="700" fill="#c62828">
    KILL zone → flatten + HALT + email + exit
  </text>

  <!-- Bottom math -->
  <text class="lbl" x="100" y="310" font-weight="700">Kill triggers when:</text>
  <text class="lbl" x="100" y="330" font-family="ui-monospace,Menlo,monospace">
    current_equity &lt; deploy_start × 0.82
  </text>
  <text class="lbl" x="100" y="350" font-family="ui-monospace,Menlo,monospace" fill="#c62828">
    $60 × 0.82 = $49.20  →  max tolerable loss = $10.80
  </text>
</svg>
"""

# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Snapback-BTC — Visual Explainer</title>
<style>__CSS__</style></head><body>

<h1>Snapback-BTC — Visual Explainer</h1>
<p class="sub">How the bot, the strategy, and the safety stack actually work — in pictures.</p>

<h2>1. System architecture</h2>
<p>Four boxes talk to each other. The bot is the only thing that places orders. Everything else is read-only, logging, or notifications.</p>
__ARCH__
<div class="note">
The exchange is the only external dependency. If Binance is down, the bot logs the error
and keeps polling — it doesn't crash or panic-close.
</div>

<h2>2. The main loop (what happens every 5 seconds)</h2>
__LOOP__
<div class="card">
<b>Tick budget:</b> ~1–2 seconds per loop iteration in normal conditions (one HTTP request to fetch
bars, one to fetch balance). The 5-second poll interval gives lots of headroom. If the network is
slow and a tick takes 8 seconds, the next tick fires immediately — there's no drift.
</div>

<h2>3. The strategy decision (entry logic)</h2>
<p>multifactor-v1 is a 4-filter AND gate. To enter LONG, all four conditions must be true on the
last closed 15m bar. SHORT is the mirror. If even one condition fails, the bot does nothing.</p>
__STRATEGY__
<div class="good">
<b>Why these four filters together?</b> Each one rejects a different bad-trade pattern:
<ul>
  <li><b>RSI(14) &lt; 40</b> — wait for a real pullback, not a top.</li>
  <li><b>Volume &gt; 2× SMA</b> — make sure there's real interest behind the move.</li>
  <li><b>Close &gt; EMA(200)</b> — only buy dips inside an uptrend, never in a downtrend.</li>
  <li><b>Funding ≤ +0.05%</b> — if everyone else is already piled into longs, don't add to the crowd.</li>
</ul>
Each filter trims the trade count and lifts win rate a few points. Together they capture
"oversold inside a confirmed trend with real participation and uncrowded positioning."
</div>

<h2>4. Sizing — how big is each trade?</h2>
__SIZING__
<div class="card">
<b>Worked example at three capital levels (BTC = $100,000):</b>
<table>
<tr><th>Equity</th><th>target_qty</th><th>cap_qty (20× lev)</th><th>final qty</th><th>notional</th><th>passes minimums?</th></tr>
<tr><td>$60</td><td>0.0008 BTC</td><td>0.0114 BTC</td><td>0.0008 BTC</td><td>$80</td>
    <td style="color:#c62828">NO (qty &lt; 0.001 min)</td></tr>
<tr><td>$100</td><td>0.00133 BTC</td><td>0.019 BTC</td><td>0.001 BTC*</td><td>$100</td>
    <td style="color:#2e7d32">YES</td></tr>
<tr><td>$200</td><td>0.00267 BTC</td><td>0.038 BTC</td><td>0.002 BTC</td><td>$200</td>
    <td style="color:#2e7d32">YES</td></tr>
</table>
<small>* rounded DOWN to exchange step (0.001 BTC). Risk per trade may be slightly below target as a result.</small>
</div>
<div class="note">
<b>This is why $60 is awkward:</b> the strategy wants to size at $80 notional at current BTC prices,
but the exchange minimum order is $50 notional / 0.001 BTC. At $60 equity, every signal that
fires gets SKIPPED because rounding the position down lands below the minimum. Bot logs it,
moves on. No money lost, but no money made either.
</div>

<h2>5. Trade lifecycle — from signal to exit</h2>
__LIFECYCLE__
<div class="card">
<b>The bracket order trick.</b> When the bot opens a position, it places THREE orders at the same time:
<ol>
  <li><b>Market order</b> — buys/sells immediately at the bid/ask.</li>
  <li><b>Stop-market (SL)</b> — sits at price × (1 − 1.5%), reduce-only. Fires automatically if price drops to it.</li>
  <li><b>Take-profit-market (TP)</b> — sits at price × (1 + 3.0%), reduce-only. Fires automatically if price reaches it.</li>
</ol>
This means the SL and TP work even if the bot itself crashes or loses internet — the exchange
manages them. The bot only handles time-stop (close after 14 days) and HALT.
</div>

<h2>6. Safety gate stack — 8 things that can block a trade</h2>
__SAFETY__
<div class="danger">
<b>The principle:</b> every check is a fail-closed gate. The bot would rather miss a trade than
place an unsafe one. There are no "override" paths — if any gate trips, the signal is skipped,
period. The user gets visibility via state.db + email, but the bot doesn't ask for permission;
it just doesn't trade.
</div>

<h2>7. Kill switch — when the bot self-destructs</h2>
__KILL__
<div class="card">
<b>What "kill" actually means:</b>
<ol>
  <li>Bot detects equity has crossed below the threshold ($49.20 for a $60 start).</li>
  <li>Touches <code>data/HALT</code> file (this acts as a permanent flag).</li>
  <li>Calls <code>close_position</code> (reduce-only market order at current price).</li>
  <li>Sends "BOT KILL SWITCH FIRED" email with the exact numbers.</li>
  <li>Logs the event to state.db.</li>
  <li>Exits the process cleanly (return 0).</li>
</ol>
After a kill, you must MANUALLY remove <code>data/HALT</code> and re-run the bot to restart
trading. This is intentional — the bot won't recover automatically from a thesis-busting drawdown.
</div>

<h2>8. The big picture — bot states</h2>
<div class="card">
<table>
<tr><th>State</th><th>What's happening</th><th>What can change it</th></tr>
<tr><td><b>booting</b></td><td>load .env, ccxt, market specs, state.db, set leverage</td>
    <td>any check fails → exit with error</td></tr>
<tr><td><b>idle</b></td><td>no position. polling 5s. evaluating signal on each new 15m bar.</td>
    <td>signal fires → open. HALT/kill → exit.</td></tr>
<tr><td><b>in position</b></td><td>position is open. SL/TP managed by exchange. polling for time-stop.</td>
    <td>SL/TP/time-stop → close → idle. HALT/kill → flatten → exit.</td></tr>
<tr><td><b>exiting</b></td><td>flattening, closing orders, sending final email, terminating.</td>
    <td>process ends (return code 0).</td></tr>
</table>
</div>

<h2>Putting it together</h2>
<div class="good">
<b>One sentence:</b> Every 5 seconds the bot asks "should I trade right now?" by checking 4
market filters against the latest 15m bar, but ONLY if 8 safety gates (env, halt, kill-switch,
position-state, bar-dedupe, symbol-allowlist, exchange-minimums, notional-cap) all pass — and
when it does trade, the SL/TP brackets are pre-placed on the exchange so it stays safe even if
the bot crashes.
</div>
<div class="note">
<b>Three things that aren't here on purpose:</b>
<ul>
  <li>No LLM in the trading loop. Every decision is deterministic Python.</li>
  <li>No machine learning, no neural networks. Just RSI + volume + EMA + funding.</li>
  <li>No "smart" overrides. The bot doesn't second-guess its own filters.</li>
</ul>
</div>

</body></html>
"""


def build() -> str:
    # Use literal placeholder tokens to avoid %-formatting issues with CSS / SVG content.
    out = HTML
    parts = [
        ("__CSS__", CSS),
        ("__ARCH__", ARCHITECTURE_SVG),
        ("__LOOP__", LOOP_SVG),
        ("__STRATEGY__", STRATEGY_SVG),
        ("__SIZING__", SIZING_SVG),
        ("__LIFECYCLE__", LIFECYCLE_SVG),
        ("__SAFETY__", SAFETY_SVG),
        ("__KILL__", KILL_SWITCH_SVG),
    ]
    for token, value in parts:
        out = out.replace(token, value)
    return out


if __name__ == "__main__":
    out = ROOT / "VISUAL_EXPLAINER.html"
    out.write_text(build())
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
