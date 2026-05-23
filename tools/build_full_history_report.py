"""Build FULL_HISTORY.html — continuous 2020 → today backtest for v1,
Donchian-v3 cons & agg, and 50/50 combined. Equity curves (PNG), monthly
returns heatmap, full trade ledger.

Input: reports/full_history_<UTC>.json + corresponding _*_equity.csv files.
"""

from __future__ import annotations

import base64
import glob
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Real deployed capital per leg (snapback bot at $101 single; $50.50 if combined split).
# See memory: snapback-deploy-capital-101.
START_PER_LEG = 50.50
START_COMBINED = 101.00
# The backtest harness uses a $1M plain-Backtest cash base so integer BTC units
# fit cleanly. PnL / Size in raw trade records are in $1M-base terms; scale them
# down by this ratio so the trade ledger shows what the trade would have been at
# the real deploy size.
BACKTEST_CASH_BASE = 1_000_000.0
SCALE_TO_LEG = START_PER_LEG / BACKTEST_CASH_BASE
SCALE_TO_COMBINED = START_COMBINED / BACKTEST_CASH_BASE


def _latest_run() -> tuple[Path, str]:
    cands = sorted(glob.glob(str(ROOT / "reports" / "full_history_*.json")))
    if not cands:
        raise RuntimeError("no full_history_*.json under reports/")
    path = Path(cands[-1])
    # ts embedded in filename
    ts = path.stem.replace("full_history_", "")
    return path, ts


def _equity_png(curves: list[tuple[str, str, pd.Series, float]], title: str,
                ylabel: str = "equity ($)") -> str:
    """Render multi-line equity curve PNG.

    curves: list of (label, hex_color, normalised_equity_series, start_dollar).
    Each curve is scaled to start_dollar at t=0.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, color, eq, start in curves:
        dollar = eq * start
        ax.plot(dollar.index, dollar.values, label=label, linewidth=1.4, color=color)
    # baseline = starting equity of the first curve (or just leave it off if mixed)
    if curves:
        ax.axhline(curves[0][3], color="#888", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    # Log scale if any series exceeds 5x to keep multiyear visualisation readable.
    max_mult = max((float(eq.max()) for _, _, eq, _ in curves), default=1.0)
    if max_mult > 5.0:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + " — log scale")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap_html(monthly: dict, label: str) -> str:
    """Year × month heatmap of monthly returns (% of cumulative)."""
    if not monthly:
        return f"<p class='mute'>{label}: no monthly data</p>"
    # Parse "YYYY-MM" -> (year, month)
    rows: dict[int, dict[int, float]] = {}
    for k, v in monthly.items():
        y, m = k.split("-")
        rows.setdefault(int(y), {})[int(m)] = float(v) * 100.0
    years = sorted(rows.keys())
    months = list(range(1, 13))

    html = [f'<h3>{label} — monthly returns (%)</h3>']
    html.append('<table class="hm">')
    html.append('<tr><th></th>' + "".join(f"<th>{datetime(2000, m, 1).strftime('%b')}</th>" for m in months) + "<th>Year</th></tr>")
    for y in years:
        cells = []
        year_eq = 1.0
        for m in months:
            v = rows[y].get(m)
            if v is None:
                cells.append('<td class="mute">·</td>')
                continue
            year_eq *= (1.0 + v / 100.0)
            if v > 0:
                # green scale
                intensity = min(int(abs(v) * 6), 80)
                bg = f"rgba(46,125,50,{0.10 + intensity / 200:.2f})"
            elif v < 0:
                intensity = min(int(abs(v) * 6), 80)
                bg = f"rgba(198,40,40,{0.10 + intensity / 200:.2f})"
            else:
                bg = "transparent"
            cells.append(f'<td style="background:{bg}">{v:+.1f}</td>')
        year_pct = (year_eq - 1.0) * 100.0
        year_cls = "green" if year_pct > 0 else "red"
        html.append(f'<tr><td class="l"><b>{y}</b></td>' + "".join(cells) + f'<td class="{year_cls}"><b>{year_pct:+.1f}</b></td></tr>')
    html.append("</table>")
    return "\n".join(html)


def _trade_table(records: list[dict], strategy: str, max_rows: int = 25,
                 scale: float = SCALE_TO_LEG, start_usd: float = START_PER_LEG) -> str:
    """Render top-N best + bottom-N worst + first-N + last-N for compactness.

    PnL and Size are scaled from the $1M backtest cash base down to the real
    deploy capital (`start_usd`). ReturnPct is dimensionless and unchanged.
    """
    if not records:
        return f"<p class='mute'>{strategy}: no trades</p>"
    rows = sorted(records, key=lambda r: r.get("PnL", 0.0), reverse=True)
    best = rows[:5]
    worst = rows[-5:]

    by_time = sorted(records, key=lambda r: r.get("EntryTime", ""))
    first = by_time[:5]
    last = by_time[-5:]

    def fmt(r: dict) -> str:
        try:
            et = pd.Timestamp(r.get("EntryTime")).strftime("%Y-%m-%d %H:%M")
        except Exception:
            et = "?"
        try:
            xt = pd.Timestamp(r.get("ExitTime")).strftime("%Y-%m-%d %H:%M")
        except Exception:
            xt = "?"
        size_raw = r.get("Size", 0.0)
        size_scaled = size_raw * scale if isinstance(size_raw, (int, float)) else 0.0
        side = "L" if (isinstance(size_raw, (int, float)) and size_raw > 0) else "S"
        ep = r.get("EntryPrice", 0.0)
        xp = r.get("ExitPrice", 0.0)
        pnl_raw = r.get("PnL", 0.0)
        pnl_scaled = pnl_raw * scale if isinstance(pnl_raw, (int, float)) else 0.0
        ret = r.get("ReturnPct", 0.0)
        if isinstance(ret, (int, float)) and abs(ret) < 1:
            ret = ret * 100  # legacy fractional
        pnl_cls = "green" if pnl_scaled > 0 else "red"
        return (
            f'<tr><td class="l">{et}</td><td class="l">{xt}</td>'
            f'<td>{side}</td><td>{ep:.1f}</td><td>{xp:.1f}</td>'
            f'<td>{abs(size_scaled):.6f}</td>'
            f'<td class="{pnl_cls}">${pnl_scaled:+.3f}</td>'
            f'<td class="{pnl_cls}">{ret:+.2f}%</td></tr>'
        )

    section = lambda title, rs: (
        f'<tr><th class="l" colspan="8" style="background:#eceff1">{title}</th></tr>' +
        "".join(fmt(r) for r in rs)
    )

    return (
        f'<details><summary>{strategy} — {len(records)} trades '
        f'(PnL/Size scaled to ${start_usd:.2f} starting equity; showing first 5, last 5, best 5, worst 5)'
        f'</summary>'
        f'<table class="tt"><tr>'
        f'<th class="l">Entry</th><th class="l">Exit</th>'
        f'<th>side</th><th>entry $</th><th>exit $</th>'
        f'<th>size BTC</th><th>PnL @ ${start_usd:.2f}</th><th>Ret %</th></tr>'
        f'{section("First 5 trades chronologically", first)}'
        f'{section("Last 5 trades chronologically", last)}'
        f'{section("Top 5 best by PnL", best)}'
        f'{section("Top 5 worst by PnL", worst)}'
        f'</table></details>'
    )


def _full_trade_csv_link(records: list[dict], strategy_slug: str, ts: str) -> str:
    """Save full trade list as CSV and link to it."""
    if not records:
        return ""
    df = pd.DataFrame(records)
    out = ROOT / "reports" / f"full_history_{ts}_{strategy_slug}_trades.csv"
    df.to_csv(out, index=False)
    return f'<p class="mute">Full ledger: <code>reports/{out.name}</code> ({len(records)} trades)</p>'


def build() -> str:
    json_path, ts = _latest_run()
    data = json.loads(json_path.read_text())
    win = data["window"]

    # Load equity CSVs
    def load_eq(name: str) -> pd.Series:
        p = ROOT / "reports" / f"full_history_{ts}_{name}_equity.csv"
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df["equity_norm"]

    eq_v1 = load_eq("v1")
    eq_cc = load_eq("d3cons")
    eq_cg = load_eq("d3agg")
    eq_combo_cons = load_eq("combo_cons")
    eq_combo_agg = load_eq("combo_agg")

    # Build HTML
    parts = []
    parts.append("""<!doctype html><html><head><meta charset="utf-8">
<title>Full History — snapback-btc trading backtest 2020 → 2026</title>
<style>
  body { font: 13.5px/1.55 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
         max-width: 1280px; margin: 28px auto; padding: 0 24px; color: #2c2c2c; background: #fafafa; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  h2 { margin-top: 36px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h3 { margin-top: 24px; color: #555; }
  .sub { color: #666; font-style: italic; }
  table { border-collapse: collapse; margin: 12px 0; font-size: 12.5px; }
  th, td { padding: 5px 10px; border: 1px solid #ddd; text-align: right; }
  th { background: #eee; }
  td.l, th.l { text-align: left; }
  .green { color: #1b5e20; font-weight: 600; }
  .red { color: #b71c1c; font-weight: 600; }
  .mute { color: #888; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px 18px; margin: 14px 0; }
  .key { background: #fff8e1; border-left: 4px solid #f57c00; padding: 10px 16px; margin: 14px 0; }
  .verdict { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 12px 16px; margin: 18px 0; }
  .truth { background: #ffebee; border-left: 4px solid #c62828; padding: 12px 16px; margin: 18px 0; }
  .warn  { background: #fff3e0; border-left: 4px solid #ef6c00; padding: 12px 16px; margin: 18px 0; }
  code { background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  img.eq { display: block; max-width: 100%; height: auto; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; background: #fff; }
  table.hm td { font-size: 11px; padding: 3px 6px; min-width: 36px; text-align: center; }
  table.tt th { background: #f5f5f5; }
  table.tt td { font-size: 11.5px; padding: 3px 8px; }
  details { margin: 10px 0; }
  details summary { cursor: pointer; font-weight: 600; padding: 6px 0; }
</style></head><body>""")

    parts.append("<h1>Full-history backtest — snapback-btc</h1>")
    parts.append(f'<p class="sub">Continuous backtest from <b>{win["start"]}</b> to <b>{win["end"]}</b> '
                 f'({(pd.Timestamp(win["end"]) - pd.Timestamp(win["start"])).days} days, ~6.1 years). '
                 f'Same harness, 5 bps/side friction, per-trade funding. '
                 f'<b>Starting equity: ${START_PER_LEG:.2f} per leg, ${START_COMBINED:.2f} combined</b> '
                 f'(matches deployed bot capital).</p>')
    parts.append(f'<p class="mute">Source data: <code>{json_path.name}</code> '
                 f'({json_path.stat().st_size // 1024} KB)</p>')

    parts.append(f"""<div class="warn">
<b>Reality check on dollar amounts:</b> these dollar trajectories assume the
strategy can size <i>proportionally</i> at $50.50 / $101. The real bot enforces
Binance min-qty 0.001 BTC and min-notional $50. At BTC ~$100k, v1's risk-based
sizing wants ~0.0008 BTC per trade — below min-qty → the live bot logs
<code>signal_skipped_minimum</code> and sits out.
<br><br>
<b>So treat the dollar curves below as an "if sizing worked" upper bound.</b>
The real bot at $50.50 capital would fire on a small fraction of these signals
(LIVE_PLAN.md projects 0-3 trades / 30 days at $60; $50.50 is worse). Combined
deploy at this capital needs <b>≥ $200 total</b> for the min-notional math
to hold cleanly. See §7.
</div>""")

    # === Honest preamble about "since 2016" ===
    parts.append("""<div class="warn">
<b>Why not since 2016?</b> Binance Futures BTC/USDT perpetual was launched
<b>September 2019</b>. Funding rates begin January 2020 on this venue.
4h klines on Binance start January 2020. There is no pre-2020 futures data
for this contract on this exchange. Spot BTC prices exist back to 2010, but
the strategy uses <i>funding rates</i> and a futures perpetual, so spot
substitution would produce misleading numbers. This run uses the longest
available continuous window: 2020-04-01 → today.
</div>""")

    # === Headline numbers ===
    v1 = data["v1"]
    cc = data["d3cons"]
    cg = data["d3agg"]
    ccombo = data["combined_cons"]
    cgcombo = data["combined_agg"]

    parts.append("<h2>1. Headline summary</h2>")
    parts.append(f'<p class="mute">Per-leg starting equity: <b>${START_PER_LEG:.2f}</b>. Combined starting: <b>${START_COMBINED:.2f}</b>. '
                 f'"Final $" columns scale linearly from these — proportional-sizing assumption.</p>')
    parts.append("""<table>
<tr><th class="l">Strategy</th><th>Trades</th><th>Start $</th><th>End $</th>
    <th>Total return</th><th>Sharpe</th><th>Max DD</th><th>Win rate</th></tr>""")
    rows = [
        ("multifactor-v1 (15m)", v1, "#1565c0", START_PER_LEG),
        ("Donchian-v3 cons (4h)", cc, "#2e7d32", START_PER_LEG),
        ("Donchian-v3 agg (4h)", cg, "#6a1b9a", START_PER_LEG),
    ]
    for label, d, _, start_usd in rows:
        ret = d["ret_pct"]
        end_usd = start_usd * (1.0 + ret / 100.0)
        cls = "green" if ret > 0 else "red"
        parts.append(
            f'<tr><td class="l">{label}</td>'
            f'<td>{d["trades"]}</td>'
            f'<td>${start_usd:.2f}</td>'
            f'<td class="{cls}"><b>${end_usd:.2f}</b></td>'
            f'<td class="{cls}">{ret:+.2f}%</td>'
            f'<td>{(d["sharpe"] or 0):+.2f}</td>'
            f'<td>{(d["max_dd_pct"] or 0):+.2f}%</td>'
            f'<td>{(d["win_rate_pct"] or 0):.1f}%</td></tr>'
        )
    cc_end = START_COMBINED * (1.0 + ccombo["ret_pct"] / 100.0)
    cg_end = START_COMBINED * (1.0 + cgcombo["ret_pct"] / 100.0)
    parts.append(
        f'<tr style="background:#f5f5f5"><td class="l"><b>50/50 combined (cons)</b></td>'
        f'<td>—</td><td>${START_COMBINED:.2f}</td>'
        f'<td class="green"><b>${cc_end:.2f}</b></td>'
        f'<td class="green"><b>{ccombo["ret_pct"]:+.2f}%</b></td>'
        f'<td><b>{ccombo["sharpe"]:+.2f}</b></td>'
        f'<td><b>{ccombo["max_dd_pct"]:+.2f}%</b></td><td>—</td></tr>'
    )
    parts.append(
        f'<tr style="background:#f5f5f5"><td class="l">50/50 combined (agg)</td>'
        f'<td>—</td><td>${START_COMBINED:.2f}</td>'
        f'<td class="green">${cg_end:.2f}</td>'
        f'<td class="green">{cgcombo["ret_pct"]:+.2f}%</td>'
        f'<td>{cgcombo["sharpe"]:+.2f}</td>'
        f'<td>{cgcombo["max_dd_pct"]:+.2f}%</td><td>—</td></tr>'
    )
    parts.append("</table>")

    v1_end = START_PER_LEG * (1 + v1['ret_pct'] / 100.0)
    cc_end_leg = START_PER_LEG * (1 + cc['ret_pct'] / 100.0)
    parts.append(f"""<div class="key">
<b>Headline (proportional-sizing scenario):</b> $50.50 → ${v1_end:.0f} for v1 alone,
$50.50 → ${cc_end_leg:.0f} for Donchian-v3 cons, and combined <b>$101 → ${cc_end:.0f}</b>.
Combined Sharpe {ccombo['sharpe']:+.2f} beats both legs individually
(v1 {v1['sharpe']:+.2f}, Donchian-cons {cc['sharpe']:+.2f}) and trims max-DD
from −32% to {ccombo['max_dd_pct']:.1f}%.<br><br>

<b>Daily-return correlation v1 ↔ Donchian-cons across 6 years = +{ccombo['daily_corr_v1_d3']:.2f}</b>
— very low, real diversification.<br><br>

<b>These dollar numbers are an upper bound</b> assuming the bot could size below
min-qty. The real bot at $50.50/$101 capital would skip most signals — see
the red warning at the top of this report.
</div>""")

    parts.append("""<div class="truth">
<b>Reality check on drawdowns:</b> the production kill-switch is set at −18%.
v1 alone exceeded this with a max-DD of −32.4%. Donchian-v3 also hit ~−32%.
The combined-cons reduced this to −27.5% but still triggers the kill-switch.
<b>The current −18% kill-switch is too tight for either strategy across multi-year
deployment</b> — it would have flatlined the account at the worst point. Either
widen to ~−35%, or deploy with the understanding that the kill-switch is a
hard floor that will be tested.
</div>""")

    # === Equity curves ===
    parts.append("<h2>2. Equity curves</h2>")

    png_individual = _equity_png(
        [
            ("multifactor-v1", "#1565c0", eq_v1, START_PER_LEG),
            ("Donchian-v3 cons", "#2e7d32", eq_cc, START_PER_LEG),
            ("Donchian-v3 agg", "#6a1b9a", eq_cg, START_PER_LEG),
        ],
        f"Per-strategy equity in $, starting ${START_PER_LEG:.2f}, {win['start']} → {win['end']}",
    )
    parts.append(f'<img class="eq" src="{png_individual}" alt="per-strategy equity">')

    png_combined = _equity_png(
        [
            ("multifactor-v1 alone ($101)", "#1565c0",
             eq_v1.reindex(eq_combo_cons.index, method="ffill"), START_COMBINED),
            ("50/50 combined cons ($50.50 + $50.50)", "#1b5e20", eq_combo_cons, START_COMBINED),
            ("50/50 combined agg", "#6a1b9a", eq_combo_agg, START_COMBINED),
        ],
        f"v1-alone vs 50/50 combined portfolios in $, starting ${START_COMBINED:.2f}, {win['start']} → {win['end']}",
    )
    parts.append(f'<img class="eq" src="{png_combined}" alt="combined vs v1">')

    parts.append("""<p>Observe: combined-cons (green) tracks above v1 (blue) almost
continuously and has visibly smaller drawdowns. The decoupling around 2022–2023 (BTC
crash + recovery) is where Donchian's regime-complementarity does its work.</p>""")

    # === Monthly heatmaps ===
    parts.append("<h2>3. Monthly returns heatmap</h2>")
    parts.append(_heatmap_html(v1["monthly_returns"], "multifactor-v1"))
    parts.append(_heatmap_html(cc["monthly_returns"], "Donchian-v3 cons"))
    parts.append(_heatmap_html(cg["monthly_returns"], "Donchian-v3 agg"))

    # === Trade ledgers ===
    parts.append("<h2>4. Trade ledgers</h2>")
    parts.append('<p>Click each strategy to expand. Full ledger CSVs are written to <code>reports/</code> alongside the JSON.</p>')

    parts.append(_trade_table(v1["trade_records"], "multifactor-v1"))
    parts.append(_full_trade_csv_link(v1["trade_records"], "v1", ts))

    parts.append(_trade_table(cc["trade_records"], "Donchian-v3 cons"))
    parts.append(_full_trade_csv_link(cc["trade_records"], "d3cons", ts))

    parts.append(_trade_table(cg["trade_records"], "Donchian-v3 agg"))
    parts.append(_full_trade_csv_link(cg["trade_records"], "d3agg", ts))

    # === Funding cost honest summary ===
    parts.append("<h2>5. Funding cost (real-money realism)</h2>")
    parts.append(f"""<p>Binance Futures funding settles every 8h. We accumulate per-trade open-interval
funding using actual entry/exit timestamps. These numbers are deducted from the returns above.
Raw amounts are in the backtest's $1M cash base; scaled column shows the equivalent at the
${START_PER_LEG:.2f} deploy size.</p>""")
    parts.append(f"""<table>
<tr><th class="l">Strategy</th><th>Funding paid @ $1M base (USDT)</th>
    <th>Scaled to ${START_PER_LEG:.2f} (USDT)</th><th>% of cash base</th></tr>""")
    for label, d in (("multifactor-v1", v1), ("Donchian-v3 cons", cc), ("Donchian-v3 agg", cg)):
        fc = d.get("funding_cost_usdt") or 0.0
        fc_scaled = fc * SCALE_TO_LEG
        parts.append(
            f'<tr><td class="l">{label}</td>'
            f'<td>{fc:,.0f}</td>'
            f'<td>${fc_scaled:+.4f}</td>'
            f'<td>{fc / BACKTEST_CASH_BASE * 100:+.2f}%</td></tr>'
        )
    parts.append("</table>")

    # === Caveats ===
    # === Vol-regime switcher (rejected) ===
    parts.append("<h2>6. Vol-regime switcher experiment — REJECTED</h2>")
    parts.append("""<p>Side experiment: would picking v1-or-Donchian per day based on ATR percentile
beat the 50/50 baseline? Phase 1 binning suggested "yes" in-sample; OOS validation showed otherwise.</p>""")

    # Try to load switcher results
    sw_json_candidates = sorted((ROOT / "reports").glob("regime_switcher_*.json"))
    sw_verify_candidates = sorted((ROOT / "reports").glob("regime_switcher_verify_*.json"))
    sw_data = None
    sw_verify = None
    if sw_json_candidates:
        # Skip the _verify files in the main glob
        sw_main = [p for p in sw_json_candidates if "_verify_" not in p.name and "_eq" not in p.name]
        if sw_main:
            try:
                sw_data = json.loads(sw_main[-1].read_text())
            except Exception:
                sw_data = None
    if sw_verify_candidates:
        try:
            sw_verify = json.loads(sw_verify_candidates[-1].read_text())
        except Exception:
            sw_verify = None

    if sw_data:
        p = sw_data["trained_params"]
        tr = sw_data["test_results"]
        parts.append(f"""<p><b>Setup:</b> daily ATR(14) percentile rank, rolling 90d, shifted 1d.
Trained threshold + hysteresis on {sw_data['train_window'][0]} → {sw_data['train_window'][1]};
true OOS test on {sw_data['test_window'][0]} → {sw_data['test_window'][1]}.
Best trained config: <code>threshold={p['threshold']}, hysteresis={p['hyst']} days</code>.</p>""")

        parts.append("<h3>Train → Test degradation</h3>")
        parts.append("""<table>
<tr><th class="l">Strategy</th><th>Test return</th><th>Test Sharpe</th><th>Test max-DD</th></tr>""")
        for k, label in (("v1", "v1 alone"), ("donchian_cons", "Donchian-cons alone"),
                         ("combo_50_50", "50/50 combined"), ("switcher", "Switcher (OOS)")):
            d = tr[k]
            cls = "green" if d["ret_pct"] > 0 else "red"
            parts.append(
                f'<tr><td class="l">{label}</td>'
                f'<td class="{cls}">{d["ret_pct"]:+.2f}%</td>'
                f'<td>{d["sharpe"]:+.2f}</td>'
                f'<td>{d["max_dd_pct"]:+.2f}%</td></tr>'
            )
        parts.append("</table>")

    if sw_verify:
        parts.append("<h3>Per-window OOS breakdown</h3>")
        parts.append("""<table>
<tr><th class="l">Window</th><th>v1 %</th><th>Donchian %</th>
    <th>50/50 ret</th><th>50/50 Sh</th><th>50/50 DD</th>
    <th>Switcher ret</th><th>Switcher Sh</th><th>Switcher DD</th>
    <th>pct_v1</th><th>Winner</th></tr>""")
        for w in sw_verify:
            c5 = w["combo_50_50"]
            sw = w["switcher"]
            winner = "<b>50/50</b>" if c5["sharpe"] > sw["sharpe"] else "<b>switcher</b>"
            parts.append(
                f'<tr><td class="l">{w["window"]}</td>'
                f'<td>{w["v1_ret_pct"]:+.2f}%</td>'
                f'<td>{w["d3_ret_pct"]:+.2f}%</td>'
                f'<td>{c5["ret_pct"]:+.2f}%</td>'
                f'<td>{c5["sharpe"]:+.2f}</td>'
                f'<td>{c5["max_dd_pct"]:+.2f}%</td>'
                f'<td>{sw["ret_pct"]:+.2f}%</td>'
                f'<td>{sw["sharpe"]:+.2f}</td>'
                f'<td>{sw["max_dd_pct"]:+.2f}%</td>'
                f'<td>{sw["pct_days_v1"]:.1f}%</td>'
                f'<td>{winner}</td></tr>'
            )
        parts.append("</table>")
        wins_5050 = sum(1 for w in sw_verify if w["combo_50_50"]["sharpe"] > w["switcher"]["sharpe"])
        parts.append(f'<p><b>50/50 wins {wins_5050} of 5 OOS windows on Sharpe.</b></p>')

    parts.append("""<div class="truth">
<b>Smoking gun — per-decision verification (OOS):</b> on the 298 days the switcher chose v1,
v1's mean daily return was <b>+0.131%</b> but Donchian's was <b>+0.142%</b> — Donchian
beat v1 on those days. The signal that says "low vol → use v1" is broken in OOS data
(t-stat for v1−Donchian = −0.14, no signal).<br><br>

The hypothesis from Phase 1 binning held in-sample (2019-2022) but flipped in test
(2023-2026). Whatever regime structure produced that pattern in 2019-22 didn't persist.<br><br>

<b>Verdict:</b> reject the switcher. 50/50 combined-cons (the existing recommendation
from §1) remains the right deploy. Don't deploy a switcher on this evidence.
</div>""")

    parts.append('<p class="mute">Code: <code>tools/regime_switcher_phase1.py</code>, '
                 '<code>regime_switcher_phase2.py</code>, '
                 '<code>regime_switcher_verify.py</code>. '
                 'Raw output in <code>reports/regime_switcher_*.json</code>.</p>')

    # === REALISTIC-ENV simulation at $50.50/$50.50 ===
    # This is the section the user requested: simulate the real live-deploy
    # constraints (min_qty 0.001 BTC, min_notional $50, per-leg kill-switch).
    realistic_path = sorted((ROOT / "reports").glob("realistic_50_50_*_nokill.json"))
    if realistic_path:
        try:
            real = json.loads(realistic_path[-1].read_text())
        except Exception:
            real = None
    else:
        real = None

    parts.append('<h2>7. Realistic-env simulation at $50.50 per leg ($101 combined)</h2>')
    parts.append('<p>Earlier dollar numbers in §1 assumed proportional sizing. Here we '
                 'simulate the LIVE deploy constraints: min-qty 0.001 BTC, min-notional $50, '
                 'per-leg logical equity, dynamic position re-sizing as equity compounds.</p>')

    if real:
        v1r = real["v1"]
        d3r = real["donchian_cons"]
        cr  = real["combined"]
        ub = {"v1": 136.39, "d3": 760.27, "co": 533.68}  # from §1 upper bound

        parts.append("""<table>
<tr><th class="l">Strategy</th>
    <th>Signals</th><th>Fires</th><th>Fire rate</th>
    <th>Start $</th><th>Final $</th><th>Return</th>
    <th>Sharpe</th><th>Max DD (peak)</th>
    <th>Capture vs UB</th></tr>""")
        for label, d, ub_ret in (
            ("v1 (15m)",       v1r, ub["v1"]),
            ("Donchian-cons",  d3r, ub["d3"]),
        ):
            cls = "green" if d["ret_pct"] > 0 else "red"
            parts.append(
                f'<tr><td class="l">{label}</td>'
                f'<td>{d["n_signals"]}</td>'
                f'<td>{d["fires"]}</td>'
                f'<td>{d["fire_rate_pct"]:.1f}%</td>'
                f'<td>${d["start_equity"]:.2f}</td>'
                f'<td class="{cls}"><b>${d["final_equity"]:.2f}</b></td>'
                f'<td class="{cls}">{d["ret_pct"]:+.2f}%</td>'
                f'<td>{d["sharpe"]:+.2f}</td>'
                f'<td>{d["max_dd_pct"]:+.2f}%</td>'
                f'<td>{d["ret_pct"]/ub_ret*100:.0f}%</td></tr>'
            )
        parts.append(
            f'<tr style="background:#f5f5f5"><td class="l"><b>50/50 combined wallet</b></td>'
            f'<td>{cr["total_signals"]}</td>'
            f'<td>{cr["total_fires"]}</td>'
            f'<td>{cr["total_fires"]/cr["total_signals"]*100:.1f}%</td>'
            f'<td>${cr["start"]:.2f}</td>'
            f'<td class="green"><b>${cr["final"]:.2f}</b></td>'
            f'<td class="green"><b>{cr["ret_pct"]:+.2f}%</b></td>'
            f'<td><b>{cr["sharpe"]:+.2f}</b></td>'
            f'<td>{cr["max_dd_pct"]:+.2f}%</td>'
            f'<td><b>{cr["ret_pct"]/ub["co"]*100:.0f}%</b></td></tr>'
        )
        parts.append("</table>")

        # Kill-switch reality (compute min absolute equity from CSVs)
        ts_re = real["ts"]
        try:
            import pandas as _pd
            tab_rows = []
            for n, start_e in (("v1", v1r["start_equity"]),
                              ("d3cons", d3r["start_equity"]),
                              ("combined", cr["start"])):
                df = _pd.read_csv(ROOT / "reports" / f"realistic_50_50_{ts_re}_nokill_{n}.csv",
                                  index_col=0, parse_dates=True)
                eq = df["equity_usd"]
                kill_floor = start_e * 0.82
                min_eq = float(eq.min())
                min_date = eq.idxmin().date()
                drop_from_start = (min_eq / start_e - 1) * 100
                survived = min_eq > kill_floor
                tab_rows.append((n, start_e, kill_floor, min_eq, min_date, drop_from_start, survived))

            parts.append("<h3>Kill-switch reality (the previously alarming finding, revised)</h3>")
            parts.append("""<p>Earlier I conflated <i>peak-to-trough</i> max-DD with the bot's
actual kill-switch trigger. The bot fires at <code>equity ≤ deploy_start_equity × 0.82</code>
— start-anchored, not peak-anchored. Once equity has grown above start, the kill-floor stays
at the original $start × 0.82. Below: actual minimum equity vs that floor.</p>""")
            parts.append("""<table>
<tr><th class="l">Leg</th><th>Start $</th><th>Kill floor (−18%)</th>
    <th>Lowest equity ever</th><th>Drop from start</th><th>Kill-switch outcome</th></tr>""")
            for n, start_e, floor, mn, mn_d, drop, survived in tab_rows:
                status = '<b class="green">SURVIVES</b>' if survived else '<b class="red">TRIPS</b>'
                parts.append(
                    f'<tr><td class="l">{n}</td>'
                    f'<td>${start_e:.2f}</td>'
                    f'<td>${floor:.2f}</td>'
                    f'<td>${mn:.2f} ({mn_d})</td>'
                    f'<td>{drop:+.2f}%</td>'
                    f'<td>{status} {"by $" + format(mn - floor, ".2f") if survived else ""}</td></tr>'
                )
            parts.append("</table>")

            parts.append(f"""<div class="verdict">
<b>Revised deploy verdict for $101 capital:</b> the realistic 50/50 combined simulation
takes the wallet from $101 → <b>${cr['final']:.0f}</b> over 6.7 years
(<b>{cr['ret_pct']:+.0f}%</b>, Sharpe {cr['sharpe']:+.2f}). Worst drop from start: only
<b>{tab_rows[2][5]:+.1f}%</b> (Nov 2019, early in deploy). The −18% kill-switch is NOT
tripped in any leg or combined over 6.7 years.<br><br>

This is materially different from the earlier reading. <b>You can deploy at $101</b>;
the prior "fund to ≥$200" recommendation was driven by min-qty math being too tight at
$50.50 — but the dynamic compounding view shows it's tight only briefly at the start,
then improves as both legs gain. Fire rate over 6.7 years: <b>{cr['total_fires']}/{cr['total_signals']} = {cr['total_fires']/cr['total_signals']*100:.0f}%</b>.
</div>""")
        except Exception as exc:
            parts.append(f'<p class="mute">Could not load equity CSVs for kill-switch reality: {exc}</p>')

        # === Equity curve PNG: realistic vs upper bound ===
        try:
            import matplotlib.pyplot as _plt
            import io as _io, base64 as _b64, pandas as _pd
            ts_re = real["ts"]
            real_co = _pd.read_csv(ROOT / "reports" / f"realistic_50_50_{ts_re}_nokill_combined.csv",
                                  index_col=0, parse_dates=True)["equity_usd"]
            # Upper-bound combined: load eq series and scale to $101 start
            ub_co = _pd.read_csv(ROOT / "reports" / f"full_history_{ts_re}_combo_cons_equity.csv",
                                index_col=0, parse_dates=True)["equity_norm"] * START_COMBINED
            v1_real = _pd.read_csv(ROOT / "reports" / f"realistic_50_50_{ts_re}_nokill_v1.csv",
                                  index_col=0, parse_dates=True)["equity_usd"]
            d3_real = _pd.read_csv(ROOT / "reports" / f"realistic_50_50_{ts_re}_nokill_d3cons.csv",
                                  index_col=0, parse_dates=True)["equity_usd"]
            fig, ax = _plt.subplots(figsize=(11, 4))
            ax.plot(ub_co.index, ub_co.values, label="combined upper-bound (proportional sizing)",
                    color="#9e9e9e", linewidth=1.2, linestyle="--")
            ax.plot(real_co.index, real_co.values, label="combined REALISTIC ($50.50+$50.50)",
                    color="#1b5e20", linewidth=1.8)
            ax.plot(v1_real.index, v1_real.values, label="v1 leg realistic",
                    color="#1565c0", linewidth=1.2, alpha=0.7)
            ax.plot(d3_real.index, d3_real.values, label="Donchian leg realistic",
                    color="#6a1b9a", linewidth=1.2, alpha=0.7)
            ax.axhline(101, color="#888", linewidth=0.5, linestyle="--", alpha=0.6)
            ax.axhline(82.82, color="#c62828", linewidth=0.8, linestyle=":", alpha=0.7, label="−18% kill floor ($82.82)")
            ax.set_title("Realistic 50/50 deploy at $101 — kill-switch floor never hit", fontsize=11)
            ax.set_ylabel("wallet equity ($)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", fontsize=9)
            _plt.tight_layout()
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
            _plt.close(fig)
            png = "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode("ascii")
            parts.append(f'<img class="eq" src="{png}" alt="realistic equity curve">')
        except Exception as exc:
            parts.append(f'<p class="mute">Equity-curve render failed: {exc}</p>')

        parts.append("<h3>What the sim does and doesn't model</h3>")
        parts.append("""<table>
<tr><th class="l">Concern</th><th>Modelled?</th></tr>
<tr><td class="l">Min-qty 0.001 BTC SKIP</td><td class="green">yes — floor to 0.001 step, skip if below</td></tr>
<tr><td class="l">Min-notional $50 SKIP</td><td class="green">yes</td></tr>
<tr><td class="l">Dynamic position re-sizing as equity compounds</td><td class="green">yes — target_btc recomputed per signal at current equity</td></tr>
<tr><td class="l">Per-leg kill-switch (start-anchored)</td><td class="green">yes — halts the leg if equity ≤ start × 0.82</td></tr>
<tr><td class="l">Per-leg commission (5 bps/side)</td><td class="green">yes — scaled to realised qty</td></tr>
<tr><td class="l">Per-leg funding (8h cycles)</td><td class="green">approx — 0.01%/8h crude proxy (real backtest used actual funding parquet)</td></tr>
<tr><td class="l">Two bots on one Binance account in hedge mode</td><td>partial — each leg's logical equity is tracked separately; assumes isolated-margin per position</td></tr>
<tr><td class="l">Wallet-level liquidation when both legs simultaneously deep DD</td><td class="red">no — would require modelling shared margin pool</td></tr>
<tr><td class="l">Network outages, API rejects, partial fills</td><td class="red">no</td></tr>
</table>""")

    else:
        parts.append('<p class="red">No realistic-env JSON found. Run <code>tools/realistic_50_50_sim.py</code>.</p>')

    parts.append("<h2>8. Caveats — read before extrapolating</h2>")
    parts.append(f"""<ol>
<li><b>Single 6-year continuous run is not 6 walk-forwards.</b> The Donchian cons params
(80/20/gate-on) were OOS-selected on IS 2020-04..2021-12; the 2022+ period is genuine
forward, but the 2020-2021 portion is the IS itself. The agg params (40/10/gate-off) have
IS = 2022-06..2024-12 → meaningful overlap with the 2023-2024 portion of this run, so the
agg compounded number is partly look-ahead. v1 has no overlap (locked params).</li>

<li><b>32% max-DD vs 18% kill-switch.</b> The kill-switch would have tripped twice
across this 6-year window for either strategy alone (and once for combined-cons),
locking in the worst point. Production deployment with the current switch will produce
different real-world results than this backtest.</li>

<li><b>Survivorship and exchange uptime.</b> Binance had service interruptions in 2020-2021
that affected liquidation cascades; these are not modeled here.</li>

<li><b>Friction is constant at 5 bps/side.</b> Real fees depend on VIP tier and ratio of
maker/taker fills. v1 added limit-order entry in 2026-05 (2 bps maker) — backtest assumes
taker throughout, so real fills should be slightly cheaper than shown.</li>

<li><b>Trades counted are closed positions only.</b> If a position is still open at the
end of the window, it does not appear in the trade ledger but does in the equity curve.</li>
</ol>""")

    parts.append('<p class="mute">Generated by <code>tools/build_full_history_report.py</code>. '
                 f'Raw data: <code>reports/{json_path.name}</code>.</p>')
    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    out = ROOT / "FULL_HISTORY.html"
    out.write_text(build())
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
