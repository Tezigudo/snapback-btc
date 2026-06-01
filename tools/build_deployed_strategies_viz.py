"""Build DEPLOYED_STRATEGIES.html — illustrative walkthrough of the live
multifactor-v1 + donchian-v3 cons pair, with real recent BTC data and
actual signal fires annotated on the charts.

Sections:
  1. Overview + comparison table
  2. multifactor-v1: decision flow + 30-day 15m chart with indicators
     + every signal fire in the window
  3. donchian-v3 cons: decision flow + 90-day 4h chart with Donchian
     channel + EMA(120) + slope + every signal fire in the window
  4. Combined: how they complement each other (live correlation evidence)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy.live_donchian_v3 import (  # noqa: E402
    _ema_slope_signed, evaluate_signal_donchian_v3,
)
from strategy.live_multifactor_v1 import evaluate_signal  # noqa: E402
from strategy.live_cnh_hybrid_short import (  # noqa: E402
    evaluate_signal_cnh_hybrid_short,
)
from strategy.indicators import atr as _atr_helper  # noqa: E402
from strategy.indicators import ema as _ema_helper  # noqa: E402
from strategy.indicators import rsi as _rsi_helper  # noqa: E402
from strategy.indicators import sma as _sma_helper  # noqa: E402

DATA = ROOT / "data" / "historical"
OUT = ROOT / "DEPLOYED_STRATEGIES.html"

DAYS_15M = 30   # 30 days of 15m bars for v1 chart
DAYS_4H = 120   # 120 days of 4h bars for donchian + cnh-hybrid chart


def load_data():
    df15 = pd.read_parquet(DATA / "BTC_USDT_USDT_15m.parquet").sort_index()
    df15 = df15[df15["volume"] > 0].copy()
    df4h = pd.read_parquet(DATA / "BTC_USDT_USDT_4h.parquet").sort_index()
    df4h = df4h[df4h["volume"] > 0].copy()
    funding = pd.read_parquet(DATA / "BTC_USDT_USDT_funding.parquet").sort_index()
    # normalize column on funding
    fcol = next((c for c in funding.columns if c.lower() in
                 ("rate", "fundingrate", "funding")), None)
    if fcol and fcol != "funding_rate":
        funding = funding.rename(columns={fcol: "funding_rate"})
    return df15, df4h, funding


def funding_at(funding: pd.DataFrame, t: pd.Timestamp) -> float:
    """Most recent funding settlement at or before t."""
    try:
        return float(funding["funding_rate"].asof(t))
    except (KeyError, ValueError):
        return 0.0


def scan_v1(df15: pd.DataFrame, funding: pd.DataFrame, params: dict,
            days: int) -> list[dict]:
    """Walk recent N days of 15m bars, call live evaluator at each closed bar.
    Returns deduped list of first-fire signals (consecutive same-side merged)."""
    end_ts = df15.index[-1]
    start_ts = end_ts - pd.Timedelta(days=days)
    # Need warmup behind start_ts
    warm = 250
    warm_start = df15.index[max(0, df15.index.get_indexer([start_ts], method="bfill")[0] - warm)]
    sub = df15.loc[warm_start:end_ts].rename(columns=str.capitalize)

    fires: list[dict] = []
    last_side: str | None = None
    for i in range(warm, len(sub)):
        slice_df = sub.iloc[:i + 1]
        ts = sub.index[i]
        if ts < start_ts:
            continue
        fr = funding_at(funding, ts)
        side, dbg = evaluate_signal(slice_df, fr, params)
        if side is None:
            last_side = None
            continue
        if side == last_side:
            continue  # dedup consecutive same-side fires
        last_side = side
        fires.append({
            "ts": ts, "side": side, "price": float(dbg["cur_close"]),
            "rsi": float(dbg["rsi"]),
            "trend_ema": float(dbg["trend_ema"]),
            "vol_sma": float(dbg["vol_sma"]),
            "cur_vol": float(dbg["cur_vol"]),
            "funding_rate": fr,
        })
    return fires


def scan_donchian(df4h: pd.DataFrame, params: dict, days: int) -> list[dict]:
    end_ts = df4h.index[-1]
    start_ts = end_ts - pd.Timedelta(days=days)
    warm = 200
    warm_start = df4h.index[max(0, df4h.index.get_indexer([start_ts], method="bfill")[0] - warm)]
    sub = df4h.loc[warm_start:end_ts].rename(columns=str.capitalize)

    fires: list[dict] = []
    last_side: str | None = None
    for i in range(warm, len(sub)):
        slice_df = sub.iloc[:i + 1]
        ts = sub.index[i]
        if ts < start_ts:
            continue
        side, sl_d, tp_d, dbg = evaluate_signal_donchian_v3(slice_df, 0.0, params)
        if side is None:
            last_side = None
            continue
        if side == last_side:
            continue
        last_side = side
        fires.append({
            "ts": ts, "side": side, "price": float(dbg["cur_close"]),
            "upper": float(dbg["upper"]), "lower": float(dbg["lower"]),
            "atr": float(dbg["atr"]),
            "sl_dist": float(sl_d), "tp_dist": float(tp_d),
            "slope": float(dbg["slope"]) if dbg.get("slope") is not None else None,
        })
    return fires


def scan_cnh_hybrid_short(df4h: pd.DataFrame, params: dict, days: int) -> list[dict]:
    """Replay the live cnh-hybrid-short evaluator bar-by-bar over the recent
    window, dedup consecutive same-side fires, return a list of fire dicts.
    """
    end_ts = df4h.index[-1]
    start_ts = end_ts - pd.Timedelta(days=days)
    # Needs 250+ bars of warmup (EMA200 + cup_len + handle_len + buffer).
    warm = 260
    warm_start = df4h.index[max(0, df4h.index.get_indexer([start_ts], method="bfill")[0] - warm)]
    sub = df4h.loc[warm_start:end_ts].rename(columns=str.capitalize)

    fires: list[dict] = []
    last_pattern_idx: int | None = None
    for i in range(warm, len(sub)):
        slice_df = sub.iloc[:i + 1]
        ts = sub.index[i]
        if ts < start_ts:
            continue
        side, sl_d, tp_d, dbg = evaluate_signal_cnh_hybrid_short(slice_df, 0.0, params)
        if side is None:
            continue
        # Dedup at the pattern-bar level — evaluator already does pattern-level
        # dedup, but multiple bars within an ICnH lookback can all fire if the
        # admitted ICnH pattern's bar is within entry_max_bars_after_handle.
        pattern_bar_str = dbg.get("pattern_bar") or dbg.get("ts")
        if pattern_bar_str == last_pattern_idx:
            continue
        last_pattern_idx = pattern_bar_str
        fires.append({
            "ts": ts, "side": side,
            "price": float(dbg["close"]),
            "ema24": float(dbg["ema24"]),
            "ema100": float(dbg["ema100"]),
            "atr": float(dbg["atr"]),
            "sl_dist": float(sl_d), "tp_dist": float(tp_d),
            "pattern": dbg.get("pattern", "?"),
            "pattern_bar": dbg.get("pattern_bar", ""),
        })
    return fires


def build_cnh_chart_data(df4h: pd.DataFrame, params: dict, days: int) -> dict:
    """4h candles + EMA(24)/EMA(100)/EMA(200) overlays + ATR(14) panel.
    The EMA stack is the visual proxy for the regime context the
    cnh-hybrid-short detectors operate in (uptrend → chop → breakdown for DT;
    cup → handle → cross for ICnH)."""
    end_ts = df4h.index[-1]
    start_ts = end_ts - pd.Timedelta(days=days)
    warm_start = start_ts - pd.Timedelta(days=40)
    sub = df4h.loc[warm_start:end_ts].copy()
    sub["ema24"] = _ema_helper(sub["close"], 24)
    sub["ema100"] = _ema_helper(sub["close"], 100)
    sub["ema200"] = _ema_helper(sub["close"], 200)
    sub["atr14"] = _atr_helper(sub["high"], sub["low"], sub["close"], 14)
    sub = sub.loc[start_ts:]
    return {
        "ts": [t.isoformat() for t in sub.index],
        "open": sub["open"].round(2).tolist(),
        "high": sub["high"].round(2).tolist(),
        "low": sub["low"].round(2).tolist(),
        "close": sub["close"].round(2).tolist(),
        "ema24": sub["ema24"].round(2).tolist(),
        "ema100": sub["ema100"].round(2).tolist(),
        "ema200": sub["ema200"].round(2).tolist(),
        "atr14": sub["atr14"].round(2).tolist(),
    }


def build_v1_chart_data(df15: pd.DataFrame, funding: pd.DataFrame,
                        params: dict, days: int) -> dict:
    """Compute everything needed to render the v1 multi-panel chart."""
    s = params["strategy"]
    end_ts = df15.index[-1]
    start_ts = end_ts - pd.Timedelta(days=days)
    warm_start = start_ts - pd.Timedelta(days=4)
    sub = df15.loc[warm_start:end_ts].copy()
    sub["rsi"] = _rsi_helper(sub["close"], s["rsi_period"])
    sub["ema200"] = _ema_helper(sub["close"], s["mf_trend_ema_period"])
    sub["vol_sma"] = _sma_helper(sub["volume"], s["volume_ma_period"])
    sub = sub.loc[start_ts:]
    # Funding aligned to bar timestamp
    sub["funding"] = [funding_at(funding, t) for t in sub.index]
    return {
        "ts": [t.isoformat() for t in sub.index],
        "open": sub["open"].round(2).tolist(),
        "high": sub["high"].round(2).tolist(),
        "low": sub["low"].round(2).tolist(),
        "close": sub["close"].round(2).tolist(),
        "ema200": sub["ema200"].round(2).tolist(),
        "rsi": sub["rsi"].round(2).tolist(),
        "volume": sub["volume"].round(2).tolist(),
        "vol_sma": sub["vol_sma"].round(2).tolist(),
        "vol_sma_2x": (sub["vol_sma"] * 2).round(2).tolist(),
        "funding": [round(v * 100, 4) for v in sub["funding"].tolist()],  # bp/8h
    }


def build_donchian_chart_data(df4h: pd.DataFrame, params: dict, days: int) -> dict:
    s = params["strategy"]
    end_ts = df4h.index[-1]
    start_ts = end_ts - pd.Timedelta(days=days)
    warm_start = start_ts - pd.Timedelta(days=20)
    sub = df4h.loc[warm_start:end_ts].copy()
    # 80-bar entry channel (high/low rolled max/min, shifted 1 to exclude current bar)
    pent = int(s["donchian_period_entry"])
    pexit = int(s["donchian_period_exit"])
    sub["upper"] = sub["close"].rolling(pent, min_periods=pent).max().shift(1)
    sub["lower"] = sub["close"].rolling(pent, min_periods=pent).min().shift(1)
    sub["exit_upper"] = sub["close"].rolling(pexit, min_periods=pexit).max().shift(1)
    sub["exit_lower"] = sub["close"].rolling(pexit, min_periods=pexit).min().shift(1)
    sub["ema120"] = _ema_helper(sub["close"], int(s["regime_ema_period"]))
    sub["atr20"] = _atr_helper(sub["high"], sub["low"], sub["close"], int(s["atr_period"]))
    # Slope at each bar: linear regression of last 30 EMA values, normalized to mean
    slope_window = int(s["regime_slope_window"])
    slope_series = []
    x = np.arange(slope_window, dtype=float)
    e = sub["ema120"].values
    for i in range(len(sub)):
        if i < slope_window:
            slope_series.append(np.nan)
            continue
        w = e[i - slope_window + 1: i + 1]
        if not np.all(np.isfinite(w)):
            slope_series.append(np.nan)
            continue
        slp = np.polyfit(x, w, 1)[0]
        mid = float(np.mean(w))
        if mid <= 0:
            slope_series.append(np.nan)
        else:
            slope_series.append(float(slp * slope_window / mid))
    sub["slope_pct"] = [s * 100 if not np.isnan(s) else None for s in slope_series]
    sub = sub.loc[start_ts:]
    return {
        "ts": [t.isoformat() for t in sub.index],
        "open": sub["open"].round(2).tolist(),
        "high": sub["high"].round(2).tolist(),
        "low": sub["low"].round(2).tolist(),
        "close": sub["close"].round(2).tolist(),
        "upper": sub["upper"].round(2).tolist(),
        "lower": sub["lower"].round(2).tolist(),
        "exit_upper": sub["exit_upper"].round(2).tolist(),
        "exit_lower": sub["exit_lower"].round(2).tolist(),
        "ema120": sub["ema120"].round(2).tolist(),
        "atr20": sub["atr20"].round(2).tolist(),
        "slope_pct": [round(v, 3) if v is not None and not np.isnan(v) else None
                      for v in sub["slope_pct"].tolist()],
    }


def fmt_signal_annot(fire: dict, kind: str) -> dict:
    """Plotly annotation dict for a single signal."""
    color = "#2e7d32" if fire["side"] == "long" else "#c62828"
    symbol = "▲" if fire["side"] == "long" else "▼"
    label = f"{fire['side'].upper()}  ${fire['price']:.0f}"
    return {"ts": fire["ts"].isoformat(), "price": fire["price"],
            "color": color, "symbol": symbol, "label": label, "side": fire["side"]}


def build_html(v1_chart: dict, don_chart: dict, cnh_chart: dict,
               v1_fires: list[dict], don_fires: list[dict],
               cnh_fires: list[dict],
               v1_params: dict, don_params: dict, cnh_params: dict) -> str:
    v1_annots = [fmt_signal_annot(f, "v1") for f in v1_fires]
    don_annots = [fmt_signal_annot(f, "don") for f in don_fires]
    cnh_annots = [fmt_signal_annot(f, "cnh") for f in cnh_fires]

    cnh_fires_table = "\n".join(
        f"<tr><td>{f['ts'].strftime('%Y-%m-%d %H:%M')} UTC</td>"
        f"<td style='color:#c62828;font-weight:600'>SHORT</td>"
        f"<td><b>{f['pattern']}</b></td>"
        f"<td class='num'>${f['price']:,.0f}</td>"
        f"<td class='num'>${f['ema24']:,.0f}</td>"
        f"<td class='num'>${f['ema100']:,.0f}</td>"
        f"<td class='num'>${f['atr']:,.0f}</td>"
        f"<td class='num'>${f['sl_dist']:,.0f}</td>"
        f"<td class='num'>${f['tp_dist']:,.0f}</td></tr>"
        for f in cnh_fires
    )
    if not cnh_fires_table:
        cnh_fires_table = "<tr><td colspan='9' style='text-align:center;color:#999'>No signals in this window — cnh-hybrid-short fires ~7–10 times per year by design.</td></tr>"

    v1_fires_table = "\n".join(
        f"<tr><td>{f['ts'].strftime('%Y-%m-%d %H:%M')} UTC</td>"
        f"<td style='color:{'#2e7d32' if f['side']=='long' else '#c62828'};font-weight:600'>"
        f"{f['side'].upper()}</td>"
        f"<td class='num'>${f['price']:,.0f}</td>"
        f"<td class='num'>{f['rsi']:.1f}</td>"
        f"<td class='num'>${f['trend_ema']:,.0f}</td>"
        f"<td class='num'>{f['cur_vol']/f['vol_sma']:.2f}×</td>"
        f"<td class='num'>{f['funding_rate']*100:+.4f}%</td></tr>"
        for f in v1_fires
    )
    if not v1_fires_table:
        v1_fires_table = "<tr><td colspan='7' style='text-align:center;color:#999'>No signals in this window — all four gates rarely line up at once. (That's the point.)</td></tr>"

    don_fires_table = "\n".join(
        f"<tr><td>{f['ts'].strftime('%Y-%m-%d %H:%M')} UTC</td>"
        f"<td style='color:{'#2e7d32' if f['side']=='long' else '#c62828'};font-weight:600'>"
        f"{f['side'].upper()}</td>"
        f"<td class='num'>${f['price']:,.0f}</td>"
        f"<td class='num'>${f['upper'] if f['side']=='long' else f['lower']:,.0f}</td>"
        f"<td class='num'>${f['atr']:,.0f}</td>"
        f"<td class='num'>${f['sl_dist']:,.0f}</td>"
        f"<td class='num'>${f['tp_dist']:,.0f}</td>"
        f"<td class='num'>{(f['slope']*100 if f['slope'] is not None else 0):+.2f}%</td></tr>"
        for f in don_fires
    )
    if not don_fires_table:
        don_fires_table = "<tr><td colspan='8' style='text-align:center;color:#999'>No signals in this window — Donchian is by design rare. (≈25 fires/year is normal.)</td></tr>"

    v1_s = v1_params["strategy"]
    don_s = don_params["strategy"]
    cnh_s = cnh_params.get("strategy", {})

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Snapback BTC — Deployed Strategies Walkthrough</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body {{ font: 14px/1.6 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1300px; margin: 24px auto; padding: 0 20px;
         color: #2c2c2c; background: #fafafa; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  h2 {{ margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 6px; font-size: 22px; }}
  h3 {{ margin-top: 24px; color: #333; font-size: 17px; }}
  .sub {{ color: #666; font-style: italic; margin-top: 0; }}
  .strategy-box {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
                   padding: 20px 24px; margin: 16px 0; }}
  .chart {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
            padding: 10px; margin: 16px 0; }}
  .compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .leg-tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
              font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
              text-transform: uppercase; }}
  .leg-v1 {{ background: #1976d2; color: white; }}
  .leg-don {{ background: #f57c00; color: white; }}
  .leg-cnh {{ background: #00838f; color: white; }}
  .leg-both {{ background: #6a1b9a; color: white; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 8px 0; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .gate-box {{ background: #f9f9f9; border-left: 3px solid #1976d2;
               padding: 8px 14px; margin: 6px 0; font-size: 13px; }}
  .gate-box.don {{ border-left-color: #f57c00; }}
  .gate-box.cnh {{ border-left-color: #00838f; }}
  .gate-name {{ font-weight: 700; color: #444; }}
  .gate-rule {{ color: #666; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }}
  .thesis-box {{ background: #f1f8e9; border-left: 3px solid #2e7d32;
                 padding: 12px 16px; margin: 12px 0; }}
  .fail-box {{ background: #ffebee; border-left: 3px solid #c62828;
               padding: 12px 16px; margin: 12px 0; font-size: 13px; }}
  pre.mermaid {{ background: #fff; padding: 12px; border-radius: 6px;
                 border: 1px solid #e0e0e0; }}
  .compare-table th {{ font-size: 11px; }}
  .footer {{ margin: 40px 0 12px; color: #999; font-size: 12px; text-align: center; }}
  .winner-cell {{ background: #e8f5e9; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
          font-size: 12px; }}
</style></head>
<body>

<h1>Snapback BTC — Deployed Strategies</h1>
<p class="sub">How the three live bots make decisions. As of {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')} UTC.
Each strategy runs in its own Binance sub-account, isolated wallet, dry-run.
The newest leg — <span class="leg-tag leg-cnh">cnh-hybrid-short</span> —
went live 2026-05-27 on the third sub-account.</p>

<h2>The trio, at a glance</h2>
<table class="compare-table">
<tr><th></th>
  <th><span class="leg-tag leg-v1">multifactor-v1</span></th>
  <th><span class="leg-tag leg-don">donchian-v3 cons</span></th>
  <th><span class="leg-tag leg-cnh">cnh-hybrid-short-v1</span></th></tr>
<tr><th>Live file</th>
  <td><code>strategy/live_multifactor_v1.py</code></td>
  <td><code>strategy/live_donchian_v3.py</code></td>
  <td><code>strategy/live_cnh_hybrid_short.py</code></td></tr>
<tr><th>Config</th>
  <td><code>config/params.yaml</code></td>
  <td><code>config/params_donchian.yaml</code></td>
  <td><code>config/params_cnh_hybrid_short.yaml</code></td></tr>
<tr><th>Signal class</th>
  <td>Mean reversion <i>within</i> trend</td>
  <td>Breakout follow-through</td>
  <td>Pattern-based short (DT + ICnH)</td></tr>
<tr><th>Direction</th>
  <td>Long &amp; short</td>
  <td>Long &amp; short</td>
  <td>Short-only</td></tr>
<tr><th>Timeframe</th>
  <td>15-minute bars</td>
  <td>4-hour bars</td>
  <td>4-hour bars</td></tr>
<tr><th>Time-stop</th>
  <td>{v1_s['time_stop_bars']} bars = 14 days</td>
  <td>{don_s['time_stop_bars']} bars = 8 days</td>
  <td>{cnh_s.get('max_hold_bars', 96)} bars = 16 days</td></tr>
<tr><th>SL method</th>
  <td>Fixed % ({v1_s['sl_pct']*100:.1f}%)</td>
  <td>{don_s['atr_sl_multiple']}× ATR(20)</td>
  <td>1.5× ATR(14)</td></tr>
<tr><th>TP method</th>
  <td>Fixed % ({v1_s['tp_pct']*100:.1f}%, 2:1 R:R)</td>
  <td>5× ATR(20) (~3.3:1 R:R)</td>
  <td>Distance to EMA(100) (skip if EMA100 ≥ entry)</td></tr>
<tr><th>Hit rate (backtest)</th>
  <td>~55%</td>
  <td>~40%</td>
  <td>~60% (sparse but high-quality fires)</td></tr>
<tr><th>Fires/year (live estimate)</th>
  <td>~100</td>
  <td>~25</td>
  <td>~9</td></tr>
<tr><th>Phase-1 OOS cum (BTC)</th>
  <td>+21.5%</td>
  <td>-26.9% (overfit warning)</td>
  <td>+18.8% (3/4 windows positive)</td></tr>
</table>

<h2><span class="leg-tag leg-v1">multifactor-v1</span> Confluence on 15-minute bars</h2>

<div class="strategy-box">
<h3>The four gates (LONG variant)</h3>

<div class="gate-box">
<div class="gate-name">Gate 1 — Oversold short-term</div>
<div class="gate-rule">RSI(14) on 15m close &lt; {v1_s['rsi_long_threshold']}</div>
<p>Standard 14-period RSI. Threshold is {v1_s['rsi_long_threshold']} (not 30) — we want noticeable
weakness but not panic. RSI(14)&lt;40 happens often, so by itself it's noise.</p>
</div>

<div class="gate-box">
<div class="gate-name">Gate 2 — Higher-timeframe trend up</div>
<div class="gate-rule">close &gt; EMA(200) on 15m</div>
<p>200 bars × 15min = <b>50 hours</b> of trend context. Filters out "buying the dip"
in a downtrend (the #1 newbie killer). If close &lt; EMA(200), no longs allowed.</p>
</div>

<div class="gate-box">
<div class="gate-name">Gate 3 — Real volume conviction</div>
<div class="gate-rule">current volume &gt; {v1_s['volume_multiple']:.1f} × SMA(20-bar volume)</div>
<p>20 × 15min = 5 hours of baseline volume. The dip bar must have ≥2× that volume.
Filters out low-liquidity wicks that look like dips on the chart but had no real
selling. This is what separates a real signal from chart noise.</p>
</div>

<div class="gate-box">
<div class="gate-name">Gate 4 — Funding not extreme</div>
<div class="gate-rule">funding_rate ≤ +{v1_s['funding_extreme_threshold']*100:.4f}% / 8h</div>
<p>If longs are paying shorts more than +0.05%/8h, the long side is over-crowded.
Block new longs. This is the "don't fight positioning" filter — extreme funding
often marks the top, not a dip-to-buy.</p>
</div>

<div class="thesis-box">
<b>Edge thesis:</b> any one filter alone loses money. Their <i>conjunction</i> wins
because the false signals get rejected. Out of (say) 100 bars where RSI&lt;40 fires,
maybe 30 also have close&gt;EMA(200), maybe 15 of those also have 2× volume, and
maybe 10 of those also have non-extreme funding. Those 10 are the high-confidence
setups. Mirror logic for shorts (RSI&gt;{v1_s['rsi_short_threshold']}, close&lt;EMA(200), etc.).
</div>

<div class="fail-box">
<b>Failure modes:</b>
<ul style="margin: 6px 0 0 20px">
<li>Chop above EMA(200) with no follow-through: TP at +3% rarely hits, SL at −1.5% does</li>
<li>News-driven gap below entry: 1.5% SL gets blown through; actual loss is bigger</li>
<li>Trending market where EMA(200) is the support that's about to crack — buys the last dip before the trend break</li>
</ul>
</div>
</div>

<h3>Live indicator panel — last {DAYS_15M} days of 15m bars</h3>
<p class="sub">Price + EMA(200), then RSI(14) panel, then volume vs SMA(20) panel, then funding.
Triangles mark actual signal fires from the live evaluator over this window.</p>
<div class="chart" id="v1-chart" style="height: 720px;"></div>

<h3>Signal fires this window</h3>
<table>
<tr><th>Timestamp</th><th>Side</th><th>Price</th><th>RSI(14)</th><th>EMA(200)</th><th>Vol ratio</th><th>Funding</th></tr>
{v1_fires_table}
</table>

<h2><span class="leg-tag leg-don">donchian-v3 cons</span> Trend-confirmed breakout on 4-hour bars</h2>

<div class="strategy-box">
<h3>The two gates (LONG variant)</h3>

<div class="gate-box don">
<div class="gate-name">Gate 1 — Donchian channel breakout</div>
<div class="gate-rule">close &gt; max(high, last {don_s['donchian_period_entry']} 4h bars)</div>
<p>{don_s['donchian_period_entry']} × 4h = {don_s['donchian_period_entry']*4/24:.1f} days
of price range. Breaking above that = something has changed. Classic Turtle Trader
rule. Note: the rolling max excludes the current bar (no peek).</p>
</div>

<div class="gate-box don">
<div class="gate-name">Gate 2 — Slope is genuinely up (the modern addition)</div>
<div class="gate-rule">slope of EMA({don_s['regime_ema_period']}) over last {don_s['regime_slope_window']} bars ≥ +{don_s['slope_trend_threshold_pct']:.2f}</div>
<p>EMA({don_s['regime_ema_period']}) = ~{don_s['regime_ema_period']*4/24:.0f} days of average price.
Take the LAST {don_s['regime_slope_window']} bars (~{don_s['regime_slope_window']*4/24:.1f} days)
and linear-regress them. The slope (normalized to % of EMA mid-value) must be
≥ +3% over those {don_s['regime_slope_window']} bars.</p>
<p>This is the <b>slope gate</b> — it kills the classic Turtle failure mode (false
breakouts in chop). Donchian on BTC without a slope filter gets killed
in 2018, 2022 Q4, etc. With it, the breakouts that fire are nearly all in
established trends.</p>
</div>

<div class="thesis-box">
<b>Edge thesis:</b> trend persistence in crypto. When BTC breaks a 13-day high
<i>while a 5-day uptrend already exists</i>, the move continues more often than
chance. The trade hunts the <b>fat right tail</b>: ~60% of trades hit SL
(small loss), but the ~40% that work run 5× ATR ≈ $7,500–$15,000 absolute.
At R:R 3.3:1, a 40% hit rate is comfortably profitable. Mirror for shorts.
</div>

<div class="fail-box">
<b>Failure modes:</b>
<ul style="margin: 6px 0 0 20px">
<li>Whipsaw: price pokes above 80-bar high, slope gate passes, reverses immediately. SL hits within 1-2 bars.</li>
<li>Trend exhaustion: breakout was the climax, not the start. SL hits within a few days.</li>
<li>Slope misread: bumpy EMA looks trending by linear-regression but isn't really.</li>
<li>Time-stop at 8 days: trade exits flat because the trend stalled.</li>
</ul>
</div>
</div>

<h3>Live indicator panel — last {DAYS_4H} days of 4h bars</h3>
<p class="sub">Price + 80-bar Donchian channel (shaded) + EMA(120). Bottom panel shows the EMA-slope% and the ±3% threshold lines.
Triangles mark actual signal fires from the live evaluator over this window.</p>
<div class="chart" id="don-chart" style="height: 720px;"></div>

<h3>Signal fires this window</h3>
<table>
<tr><th>Timestamp</th><th>Side</th><th>Price</th><th>Channel edge</th><th>ATR(20)</th><th>SL dist</th><th>TP dist</th><th>EMA slope</th></tr>
{don_fires_table}
</table>

<h2><span class="leg-tag leg-cnh">cnh-hybrid-short-v1</span> Pattern-based short on 4-hour bars</h2>

<div class="strategy-box">
<h3>The newest leg, in one sentence</h3>
<p>
Looks for <b>two specific chart patterns</b> on the 4h timeframe — <b>Distribution Top</b> (DT)
and <b>Inverse Cup-and-Handle</b> (ICnH) — and shorts BTC when one fires. Stop-loss is
1.5× ATR(14) above entry; take-profit is the distance down to EMA(100); the trade is
<i>skipped</i> if EMA(100) sits above the current close (no valid SHORT TP target).
Each pattern is given a single shot per 15-bar (~2.5-day) window — repeats inside
that window are ignored, even if they would have fired on their own.
</p>

<h3>Pattern 1 — Distribution Top (DT)</h3>
<p class="sub">"Uptrend → chop → breakdown" — classic topping structure on a multi-bar scale.</p>

<div class="gate-box cnh">
<div class="gate-name">Rule 1 — Uptrend phase</div>
<div class="gate-rule">close[i − {cnh_s.get('uptrend_bars', 16)} − {cnh_s.get('chop_bars', 8)}] → close[i − {cnh_s.get('chop_bars', 8)}] rises ≥ {cnh_s.get('min_rise_pct', 2.5):.1f}%</div>
<p>Over the {cnh_s.get('uptrend_bars', 16)} bars (= {cnh_s.get('uptrend_bars', 16)*4} hours ≈ {cnh_s.get('uptrend_bars', 16)*4/24:.1f} days) before the chop window, price rose at least
{cnh_s.get('min_rise_pct', 2.5):.1f}%. Filters out range-bound noise — DT only makes sense as the
exhaustion of a prior up-move.</p>
</div>

<div class="gate-box cnh">
<div class="gate-name">Rule 2 — Chop phase at the top</div>
<div class="gate-rule">range(close, last {cnh_s.get('chop_bars', 8)} bars) ≤ {cnh_s.get('max_chop_ratio', 0.55):.2f} × range(close, prior uptrend bars)</div>
<p>The last {cnh_s.get('chop_bars', 8)} bars (= {cnh_s.get('chop_bars', 8)*4} hours ≈ {cnh_s.get('chop_bars', 8)*4/24:.1f} days) drift sideways — narrow range relative to the uptrend that came
before. This is the "distribution" — smart money offloading without driving price further
up. Optionally requires the chop to actually be at the local peak of the uptrend.</p>
</div>

<div class="gate-box cnh">
<div class="gate-name">Rule 3 — Breakdown trigger on the firing bar</div>
<div class="gate-rule">close[i] &lt; min(chop_lows)  OR  close[i] &lt; EMA(24)[i]</div>
<p>Current 4h close breaks below either the chop range's low <i>or</i> the 24-bar EMA
(= 4 days of price). Either condition counts ("<code>chop_low_or_ema24</code>" mode in
the config). The break is the "actually selling now" confirmation — without it,
the chop is just chop.</p>
</div>

<h3>Pattern 2 — Inverse Cup-and-Handle (ICnH)</h3>
<p class="sub">A bullish-looking cup that <i>fails</i>. The handle's failed retest is the short signal.</p>

<div class="gate-box cnh">
<div class="gate-name">Rule 1 — Cup shape ({cnh_s.get('cup_len', 20)} bars)</div>
<div class="gate-rule">parabolic R² ≥ {cnh_s.get('min_r2', 0.50):.2f}  AND  depth ≥ {cnh_s.get('min_cup_depth_atr', 1.0)}× ATR  AND  rim heights within ±{cnh_s.get('peak_tolerance', 6)} bars</div>
<p>Fit a parabola to the last {cnh_s.get('cup_len', 20)} bars (= {cnh_s.get('cup_len', 20)*4} hours ≈ {cnh_s.get('cup_len', 20)*4/24:.1f} days). The fit's R² must be ≥ {cnh_s.get('min_r2', 0.50):.2f} (it's actually parabolic, not just noisy), the trough must be
at least 1.0 ATR below both rims (real depth), and the two rims must be within ±{cnh_s.get('peak_tolerance', 6)} bars of each other (symmetric, not a one-sided slope).</p>
</div>

<div class="gate-box cnh">
<div class="gate-name">Rule 2 — Handle ({cnh_s.get('handle_len', 4)} bars)</div>
<div class="gate-rule">handle depth ≤ {cnh_s.get('handle_max_depth_frac', 0.70)*100:.0f}% of cup depth</div>
<p>The {cnh_s.get('handle_len', 4)} bars right after the cup must form a "handle" — a shallower retracement (≤ {cnh_s.get('handle_max_depth_frac', 0.70)*100:.0f}% of the cup's depth).
Visually: the cup tries to recover, the handle gives a bit back. Bullish setup —
which is exactly why fading it works when the pattern <i>fails</i>.</p>
</div>

<div class="gate-box cnh">
<div class="gate-name">Rule 3 — EMA-cross trigger (lookback)</div>
<div class="gate-rule">close[i] crosses below EMA({(cnh_s.get('entry_emas') or ['ema24'])[0].replace('ema','')})  AND  an ICnH pattern was admitted within the last 8 bars</div>
<p>ICnH doesn't fire on the pattern bar itself — it waits up to 8 bars (= {8*4} hours ≈ {8*4/24:.0f} days) for a close
below EMA(24) to confirm that the cup is failing. If that cross doesn't happen
inside the lookback, the setup expires.</p>
</div>

<h3>Stop, take-profit, and the skip rule</h3>

<div class="gate-box cnh">
<div class="gate-name">Stop-loss</div>
<div class="gate-rule">SL = 1.5 × ATR(14) above entry</div>
<p>Adaptive volatility-scaled stop. At a typical BTC ATR of ~$1,500/4h, that's a ~$2,250 stop —
roughly 2-3% of price. Bigger than donchian's stop, smaller than v1's fixed-percent stop.</p>
</div>

<div class="gate-box cnh">
<div class="gate-name">Take-profit</div>
<div class="gate-rule">TP = distance from entry down to EMA(100)</div>
<p>EMA(100) over 4h ≈ 16 days of average price. The trade closes when price retraces back to
that mean — variable distance depending on how stretched the breakdown is. Long-tailed
trades (where price keeps going) catch the EMA on the way down.</p>
</div>

<div class="gate-box cnh">
<div class="gate-name">Skip-no-TP rule</div>
<div class="gate-rule">if EMA(100) ≥ close: skip the trade entirely</div>
<p>If the take-profit target sits <i>above</i> the entry price, there's no valid downside target —
the trade would be opened with TP unreachable. The evaluator returns <code>side=None</code> with
<code>reason="dt_admitted_but_no_tp"</code> or <code>"icnh_admitted_but_no_tp"</code>. Costs us
some signals but the alternative (no take-profit) is unacceptable for a short.</p>
</div>

<h3>Stateful pattern-level dedup (the load-bearing innovation)</h3>
<div class="thesis-box">
<p>Both DT and ICnH detectors can fire on consecutive bars during a single
exhaustion event. Without dedup, the same topping structure produces 4-6 entries
back-to-back, all hitting the same drawdown. The Phase 3b validation showed
this caused <b>~40% over-fire</b> versus the backtest's intended behaviour,
dragging portfolio Sharpe by -0.36.</p>

<p>The fix: <b>once a pattern is admitted, no further pattern (DT or ICnH) is
admitted for {cnh_s.get('dedup_bars', 15)} bars (= {cnh_s.get('dedup_bars', 15)*4/24:.1f} days)</b>. This is
tracked stateful inside the live evaluator by replaying the admission state from
the visible bars window each call — matching <code>find_hybrid_patterns</code>'s
backtest behaviour exactly. Phase 4 portfolio sim confirmed: stateful dedup
restores the +0.26 live Sharpe lift that the BTC validation locked in.</p>
</div>

<div class="thesis-box">
<b>Edge thesis:</b> short-only on a long-biased asset is contrarian — most of
crypto is in uptrend, so unconditional shorts lose. But topping structures (DT +
ICnH) appear <i>only</i> at exhaustion, when the rally is structurally over for
a few days. Catching those moments — and only those — gives ~60% win rate at
1.5×ATR vs EMA(100) R:R. Validated in Phase 1 walk-forward with all 3 dedup
variants passing the +18.8% / 4-window-positive gate; dedup=15 won on Sharpe
and worst-window.
</div>

<div class="fail-box">
<b>Failure modes:</b>
<ul style="margin: 6px 0 0 20px">
<li><b>No TP slot</b>: in a downtrend, EMA(100) often sits above current close. Strategy correctly skips, but you lose what would have been valid pattern fires.</li>
<li><b>Strong uptrend resumes</b>: pattern fires, price snaps back, SL hits at 1.5×ATR. Happens ~40% of the time (the lose rate).</li>
<li><b>Pattern-level dedup blocks a real second signal</b>: rare but possible — two genuinely distinct topping structures within 15 bars get reduced to one fire.</li>
<li><b>Cross-coin transfer is uneven</b>: same params on ETH/ADA fail Phase 1; SOL/WLD pass cleanly. The pattern detector is asset-sensitive in a way that v1's RSI/funding gates are not.</li>
</ul>
</div>
</div>

<h3>Live indicator panel — last {DAYS_4H} days of 4h bars</h3>
<p class="sub">Price + EMA(24)/EMA(100)/EMA(200) overlays + ATR(14) panel. Triangles mark
actual SHORT fires from the live evaluator over this window, labeled by the pattern
type that admitted them (DT or ICnH).</p>
<div class="chart" id="cnh-chart" style="height: 720px;"></div>

<h3>Signal fires this window</h3>
<table>
<tr><th>Timestamp</th><th>Side</th><th>Pattern</th><th>Price</th><th>EMA(24)</th><th>EMA(100)</th><th>ATR(14)</th><th>SL dist</th><th>TP dist</th></tr>
{cnh_fires_table}
</table>

<h2><span class="leg-tag leg-both">Why the trio works</span></h2>

<div class="strategy-box">
<p>The three strategies are <b>structurally orthogonal</b> on six axes — each occupies
a different point in strategy-space. v1 is a short-bar counter-trend trader; donchian-v3
is a long-bar trend-follower; cnh-hybrid-short is a long-bar pattern-fade short.</p>

<table>
<tr><th>Axis</th>
  <th><span class="leg-tag leg-v1">v1</span></th>
  <th><span class="leg-tag leg-don">donchian-v3</span></th>
  <th><span class="leg-tag leg-cnh">cnh-hybrid-short</span></th></tr>
<tr><td>Timeframe</td>
  <td>15m (fast)</td>
  <td>4h (slow)</td>
  <td>4h (slow)</td></tr>
<tr><td>Signal direction</td>
  <td>Counter (fade the dip)</td>
  <td>With (ride the breakout)</td>
  <td>Counter (fade the top)</td></tr>
<tr><td>Allowed sides</td>
  <td>Long + short</td>
  <td>Long + short</td>
  <td><b>Short only</b></td></tr>
<tr><td>Signal logic</td>
  <td>Indicator confluence</td>
  <td>Channel breakout + slope</td>
  <td>Multi-bar pattern shape</td></tr>
<tr><td>R:R</td>
  <td>2:1 (symmetric)</td>
  <td>~3.3:1 (asymmetric, fat tail)</td>
  <td>~2:1 (entry-to-EMA100)</td></tr>
<tr><td>SL method</td>
  <td>Fixed % of price</td>
  <td>Adaptive ATR(20)</td>
  <td>Adaptive ATR(14)</td></tr>
<tr><td>Fire rate</td>
  <td>~100/year</td>
  <td>~25/year</td>
  <td>~9/year</td></tr>
</table>

<p>The empirical proof of orthogonality for the v1 ↔ donchian-v3 pair:
<b>daily P&amp;L correlation of −0.01</b> in the realistic 6.7-year simulation
(n=2,025 common trading days). The cnh-hybrid-short leg is too new for a live
correlation measurement, but its Phase 4 portfolio sim measured a
<b>+0.26 Sharpe lift</b> on top of the existing pair — meaning it adds
diversification, not redundancy.</p>

<p>Losing strategies that are independent can be <i>combined</i> into a winning
portfolio — and winning strategies that are independent <i>compound</i> beyond
their individual contributions. A potential 4th leg in this codebase would need
correlation &lt; ~0.3 with all three current legs to add real diversification.
The Phase-1 cross-coin scan (<code>tools/cross_coin_backtest.py</code>) flagged
<b>SOL × cnh-hybrid-short</b> as the strongest candidate
(4/4 OOS positive, +42% cum); ETH/ADA on the same strategy didn't transfer
cleanly and would need per-coin param tuning.</p>
</div>

<h3>What runs every 5 seconds (bot loop)</h3>
<pre style="background: #2d2d2d; color: #f0f0f0; padding: 16px; border-radius: 6px; font-size: 12px; overflow-x: auto;">while not halted:
    touch(data/heartbeat)                    # 1. liveness signal
    if exists(data/HALT): flatten(); exit()  # 2. operator kill
    equity = client.fetch_equity_usdt()
    if equity / deploy_start &lt; 0.645: HALT() # 3. -35.5% kill switch

    if NOT in_position:
        bars = client.fetch_ohlcv(timeframe) # 4. recent N bars
        funding = client.fetch_funding()
        decision = evaluate_for_strategy(name, bars, funding, params)
        if decision.side is not None:
            qty = (risk_pct * equity) / decision.sl_distance
            place_bracket_order(side, qty, sl_price, tp_price)

    if in_position and bars_held &gt; time_stop_bars:
        close_position()                     # 5. time stop

    log_to_jsonl()
    push_consolidate_event()                 # 6. dashboard heartbeat
    sleep(poll_interval_s)                   # 5s for v1; 60s for donchian + cnh-short
</pre>

<p>All three legs run this same loop. The only difference is which
<code>evaluate_for_strategy</code> branch executes (selected by <code>strategy_name</code>
in the config) and which <code>.env.&lt;instance&gt;</code> file supplies the
sub-account API key. The bot is otherwise strategy-agnostic — adding a 4th
leg requires only:</p>
<ul>
<li>A new <code>strategy/live_*.py</code> module + dispatch branch in <code>bot_internals.py</code></li>
<li>A new entry in <code>bot.INSTANCE_PROFILES</code> (config / state DB / log / heartbeat paths)</li>
<li>A new <code>.env.&lt;instance&gt;</code> with the new sub-account's keys</li>
<li>A new entry in <code>tools.watchdog.LEGS</code></li>
<li>A new <code>BotSource</code> union entry in the consolidate web dashboard</li>
</ul>

<p class="footer">Generated by tools/build_deployed_strategies_viz.py. Source: BTC/USDT:USDT, Binance Futures USDM.</p>

<script>
mermaid.initialize({{startOnLoad: true, theme: 'default'}});

const V1 = {json.dumps(v1_chart)};
const DON = {json.dumps(don_chart)};
const CNH = {json.dumps(cnh_chart)};
const V1_ANNOTS = {json.dumps(v1_annots)};
const DON_ANNOTS = {json.dumps(don_annots)};
const CNH_ANNOTS = {json.dumps(cnh_annots)};

// ===== v1 multi-panel chart =====
// Panel 1 (60%): candles + EMA200
// Panel 2 (15%): RSI(14)
// Panel 3 (15%): volume bars + SMA(20) + 2×SMA(20)
// Panel 4 (10%): funding rate
{{
  const v1Traces = [
    // Candles
    {{type: 'candlestick', name: 'BTC 15m',
      x: V1.ts, open: V1.open, high: V1.high, low: V1.low, close: V1.close,
      increasing: {{line: {{color: '#26a69a'}}}}, decreasing: {{line: {{color: '#ef5350'}}}},
      xaxis: 'x', yaxis: 'y'}},
    {{type: 'scatter', name: 'EMA(200)', x: V1.ts, y: V1.ema200,
      mode: 'lines', line: {{color: '#1976d2', width: 1.5}}, xaxis: 'x', yaxis: 'y'}},

    // RSI
    {{type: 'scatter', name: 'RSI(14)', x: V1.ts, y: V1.rsi,
      mode: 'lines', line: {{color: '#7b1fa2', width: 1.2}}, xaxis: 'x', yaxis: 'y2'}},

    // Volume
    {{type: 'bar', name: 'Volume', x: V1.ts, y: V1.volume,
      marker: {{color: V1.volume.map((v,i) => v > V1.vol_sma_2x[i] ? '#2e7d32' : '#bbb')}},
      xaxis: 'x', yaxis: 'y3'}},
    {{type: 'scatter', name: '2× SMA(20) vol', x: V1.ts, y: V1.vol_sma_2x,
      mode: 'lines', line: {{color: '#d32f2f', width: 1, dash: 'dash'}}, xaxis: 'x', yaxis: 'y3'}},

    // Funding
    {{type: 'scatter', name: 'funding %/8h', x: V1.ts, y: V1.funding,
      mode: 'lines', line: {{color: '#f57c00', width: 1.2}}, xaxis: 'x', yaxis: 'y4'}},
  ];

  // Signal annotations: triangles + text
  for (const a of V1_ANNOTS) {{
    v1Traces.push({{type:'scatter', x:[a.ts], y:[a.price],
      mode:'markers+text', text:[a.symbol], textposition: a.side==='long'?'bottom center':'top center',
      textfont:{{size:24, color:a.color}},
      marker:{{size:14, color:a.color, line:{{color:'#fff', width:1.5}}}},
      name: a.label, xaxis:'x', yaxis:'y', showlegend:false}});
  }}

  const layout = {{
    grid: {{rows: 4, columns: 1, pattern: 'independent',
            rowheights: [0.55, 0.18, 0.17, 0.10]}},
    margin: {{t: 10, r: 10, b: 30, l: 60}},
    showlegend: true,
    legend: {{x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.85)'}},
    xaxis: {{rangeslider: {{visible: false}}, anchor: 'y'}},
    xaxis2: {{matches: 'x', anchor: 'y2', showticklabels: false}},
    xaxis3: {{matches: 'x', anchor: 'y3', showticklabels: false}},
    xaxis4: {{matches: 'x', anchor: 'y4'}},
    yaxis: {{title: 'BTC/USDT', side: 'left', autorange: true}},
    yaxis2: {{title: 'RSI', range: [0, 100], side: 'left',
              showgrid: true,
              tickvals: [0, 30, 40, 50, 70, 100]}},
    yaxis3: {{title: 'Vol', side: 'left'}},
    yaxis4: {{title: 'Fund %/8h', side: 'left'}},
    shapes: [
      // RSI thresholds
      {{type:'line', xref:'x2', yref:'y2', x0:V1.ts[0], x1:V1.ts[V1.ts.length-1],
        y0:40, y1:40, line:{{color:'#2e7d32', width:1, dash:'dot'}}}},
      {{type:'line', xref:'x2', yref:'y2', x0:V1.ts[0], x1:V1.ts[V1.ts.length-1],
        y0:70, y1:70, line:{{color:'#c62828', width:1, dash:'dot'}}}},
      // Funding extreme thresholds
      {{type:'line', xref:'x4', yref:'y4', x0:V1.ts[0], x1:V1.ts[V1.ts.length-1],
        y0:0.05, y1:0.05, line:{{color:'#c62828', width:1, dash:'dot'}}}},
      {{type:'line', xref:'x4', yref:'y4', x0:V1.ts[0], x1:V1.ts[V1.ts.length-1],
        y0:-0.05, y1:-0.05, line:{{color:'#2e7d32', width:1, dash:'dot'}}}},
    ],
  }};
  Plotly.newPlot('v1-chart', v1Traces, layout, {{displayModeBar: false, responsive: true}});
}}

// ===== donchian multi-panel chart =====
// Panel 1 (70%): candles + 80-bar Donchian channel (filled) + EMA(120)
// Panel 2 (30%): EMA-slope% with ±3% threshold lines
{{
  const donTraces = [
    // Filled Donchian channel
    {{type:'scatter', name:'80-bar Upper', x:DON.ts, y:DON.upper,
      mode:'lines', line:{{color:'rgba(245,124,0,0.5)', width:1}},
      xaxis:'x', yaxis:'y'}},
    {{type:'scatter', name:'80-bar Lower', x:DON.ts, y:DON.lower,
      mode:'lines', line:{{color:'rgba(245,124,0,0.5)', width:1}},
      fill:'tonexty', fillcolor:'rgba(245,124,0,0.08)',
      xaxis:'x', yaxis:'y'}},
    // Candles
    {{type:'candlestick', name:'BTC 4h',
      x:DON.ts, open:DON.open, high:DON.high, low:DON.low, close:DON.close,
      increasing:{{line:{{color:'#26a69a'}}}}, decreasing:{{line:{{color:'#ef5350'}}}},
      xaxis:'x', yaxis:'y'}},
    // EMA(120)
    {{type:'scatter', name:'EMA(120)', x:DON.ts, y:DON.ema120,
      mode:'lines', line:{{color:'#6a1b9a', width:1.6}}, xaxis:'x', yaxis:'y'}},
    // Slope panel
    {{type:'scatter', name:'EMA-slope %', x:DON.ts, y:DON.slope_pct,
      mode:'lines', line:{{color:'#1976d2', width:1.4}},
      fill:'tozeroy', fillcolor:'rgba(25,118,210,0.15)',
      xaxis:'x', yaxis:'y2'}},
  ];

  for (const a of DON_ANNOTS) {{
    donTraces.push({{type:'scatter', x:[a.ts], y:[a.price],
      mode:'markers+text', text:[a.symbol], textposition: a.side==='long'?'bottom center':'top center',
      textfont:{{size:28, color:a.color}},
      marker:{{size:18, color:a.color, line:{{color:'#fff', width:1.5}}}},
      name: a.label, xaxis:'x', yaxis:'y', showlegend:false}});
  }}

  const donLayout = {{
    grid: {{rows: 2, columns: 1, pattern: 'independent', rowheights: [0.70, 0.30]}},
    margin: {{t: 10, r: 10, b: 30, l: 60}},
    showlegend: true,
    legend: {{x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.85)'}},
    xaxis: {{rangeslider: {{visible: false}}, anchor: 'y'}},
    xaxis2: {{matches: 'x', anchor: 'y2'}},
    yaxis: {{title: 'BTC/USDT', side: 'left'}},
    yaxis2: {{title: 'EMA slope (%)', side: 'left'}},
    shapes: [
      {{type:'line', xref:'x2', yref:'y2', x0:DON.ts[0], x1:DON.ts[DON.ts.length-1],
        y0:3, y1:3, line:{{color:'#2e7d32', width:1, dash:'dot'}}}},
      {{type:'line', xref:'x2', yref:'y2', x0:DON.ts[0], x1:DON.ts[DON.ts.length-1],
        y0:-3, y1:-3, line:{{color:'#c62828', width:1, dash:'dot'}}}},
      {{type:'line', xref:'x2', yref:'y2', x0:DON.ts[0], x1:DON.ts[DON.ts.length-1],
        y0:0, y1:0, line:{{color:'#666', width:0.5}}}},
    ],
  }};
  Plotly.newPlot('don-chart', donTraces, donLayout, {{displayModeBar: false, responsive: true}});
}}

// ===== cnh-hybrid-short multi-panel chart =====
// Panel 1 (75%): candles + EMA(24) + EMA(100) + EMA(200)
// Panel 2 (25%): ATR(14) line
{{
  const cnhTraces = [
    {{type:'candlestick', name:'BTC 4h',
      x:CNH.ts, open:CNH.open, high:CNH.high, low:CNH.low, close:CNH.close,
      increasing:{{line:{{color:'#26a69a'}}}}, decreasing:{{line:{{color:'#ef5350'}}}},
      xaxis:'x', yaxis:'y'}},
    {{type:'scatter', name:'EMA(24)', x:CNH.ts, y:CNH.ema24,
      mode:'lines', line:{{color:'#00838f', width:1.5}}, xaxis:'x', yaxis:'y'}},
    {{type:'scatter', name:'EMA(100)', x:CNH.ts, y:CNH.ema100,
      mode:'lines', line:{{color:'#f57c00', width:1.5, dash:'dash'}}, xaxis:'x', yaxis:'y'}},
    {{type:'scatter', name:'EMA(200)', x:CNH.ts, y:CNH.ema200,
      mode:'lines', line:{{color:'#6a1b9a', width:1.2, dash:'dot'}}, xaxis:'x', yaxis:'y'}},
    {{type:'scatter', name:'ATR(14)', x:CNH.ts, y:CNH.atr14,
      mode:'lines', line:{{color:'#1976d2', width:1.4}},
      fill:'tozeroy', fillcolor:'rgba(25,118,210,0.1)',
      xaxis:'x', yaxis:'y2'}},
  ];

  // SHORT-only annotations: red ▼ on the entry bar.
  for (const a of CNH_ANNOTS) {{
    cnhTraces.push({{type:'scatter', x:[a.ts], y:[a.price],
      mode:'markers+text', text:[a.symbol], textposition:'top center',
      textfont:{{size:28, color:a.color}},
      marker:{{size:18, color:a.color, line:{{color:'#fff', width:1.5}}}},
      name: a.label, xaxis:'x', yaxis:'y', showlegend:false}});
  }}

  const cnhLayout = {{
    grid: {{rows: 2, columns: 1, pattern: 'independent', rowheights: [0.75, 0.25]}},
    margin: {{t: 10, r: 10, b: 30, l: 60}},
    showlegend: true,
    legend: {{x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.85)'}},
    xaxis: {{rangeslider: {{visible: false}}, anchor: 'y'}},
    xaxis2: {{matches: 'x', anchor: 'y2'}},
    yaxis: {{title: 'BTC/USDT', side: 'left'}},
    yaxis2: {{title: 'ATR (USDT)', side: 'left'}},
  }};
  Plotly.newPlot('cnh-chart', cnhTraces, cnhLayout, {{displayModeBar: false, responsive: true}});
}}
</script>
</body></html>
"""


def main() -> int:
    df15, df4h, funding = load_data()
    v1_params = yaml.safe_load(open(ROOT / "config" / "params.yaml"))
    don_params = yaml.safe_load(open(ROOT / "config" / "params_donchian.yaml"))
    cnh_params = yaml.safe_load(open(ROOT / "config" / "params_cnh_hybrid_short.yaml"))

    print(f"Data: 15m {len(df15):,} bars, 4h {len(df4h):,} bars, funding {len(funding):,} settlements")
    print(f"Window: 15m last {DAYS_15M}d, 4h last {DAYS_4H}d")

    print("Scanning v1 signals (this is the slow part)...")
    v1_fires = scan_v1(df15, funding, v1_params, DAYS_15M)
    print(f"  found {len(v1_fires)} v1 first-fire signals (deduped)")

    print("Scanning donchian signals...")
    don_fires = scan_donchian(df4h, don_params, DAYS_4H)
    print(f"  found {len(don_fires)} donchian first-fire signals (deduped)")

    print("Scanning cnh-hybrid-short signals...")
    cnh_fires = scan_cnh_hybrid_short(df4h, cnh_params, DAYS_4H)
    print(f"  found {len(cnh_fires)} cnh-hybrid-short pattern-deduped fires")

    print("Building chart data...")
    v1_chart = build_v1_chart_data(df15, funding, v1_params, DAYS_15M)
    don_chart = build_donchian_chart_data(df4h, don_params, DAYS_4H)
    cnh_chart = build_cnh_chart_data(df4h, cnh_params, DAYS_4H)

    print("Composing HTML...")
    html = build_html(
        v1_chart, don_chart, cnh_chart,
        v1_fires, don_fires, cnh_fires,
        v1_params, don_params, cnh_params,
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
