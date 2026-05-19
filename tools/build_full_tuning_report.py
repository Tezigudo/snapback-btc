"""
Comprehensive tuning report:

(a) Walk-forward: v3-all + multifactor-v1 across 5 OOS windows
(b) ATR-aware exit analysis for v3-all (bucket by return magnitude, not fixed-pct)
(c) require_trend on/off ablation (multifactor-v1)
(A) Trend-break debounce variants (N=2, 3, 4 bars below EMA before exit)
(C) Loss-floor variants (only exit on trend-cross if PnL > floor)
    + combined "deluxe" variant (debounce N=2 + floor=-0.5%)

All consolidated into reports/TUNING_REPORT.html
"""

from __future__ import annotations

import dataclasses
import html
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "15m"
CASH = 10_000.0
LEVERAGE = 20
DEFAULT_RISK_PCT = 1.0

# 5 OOS windows over 2024-2026 — matches MEMORY.md's prior methodology.
WINDOWS = [
    ("W1 2024-H1", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC)),
    ("W2 2024-H2", datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)),
    ("W3 2025-H1", datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    ("W4 2025-H2", datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
    ("W5 2026-YTD", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 5, 18, tzinfo=UTC)),
]

OUT = REPO_ROOT / "reports" / "TUNING_REPORT.html"


def reset_classes() -> None:
    import importlib

    import strategy.signals_multifactor_v2 as v2
    import strategy.signals_multifactor_v3 as v3
    import strategy.signals_multifactor_tuned as tn
    importlib.reload(v2)
    importlib.reload(v3)
    importlib.reload(tn)
    STRATEGIES["multifactor-v2-loose"] = v2.DayTradeMultiFactorBTCv2Loose
    STRATEGIES["multifactor-v2-strict"] = v2.DayTradeMultiFactorBTCv2Strict
    STRATEGIES["multifactor-v3"] = v3.DayTradeMultiFactorBTCv3
    STRATEGIES["v3-dist-ema-only"] = v3.V3DistEmaOnly
    STRATEGIES["v3-vol-regime-only"] = v3.V3VolRegimeOnly
    STRATEGIES["v3-atr-stops-only"] = v3.V3AtrStopsOnly
    STRATEGIES["v3-all"] = v3.V3All
    STRATEGIES["v1-debounce-2"] = tn.V1Debounce2
    STRATEGIES["v1-debounce-3"] = tn.V1Debounce3
    STRATEGIES["v1-debounce-4"] = tn.V1Debounce4
    STRATEGIES["v1-floor-0.5"] = tn.V1Floor005
    STRATEGIES["v1-floor-1.0"] = tn.V1Floor010
    STRATEGIES["v1-deluxe"] = tn.V1Deluxe


def make_params(risk_pct: float, **overrides) -> StrategyParams:
    base = StrategyParams.from_yaml()
    return dataclasses.replace(base, risk_per_trade_pct=risk_pct, leverage=LEVERAGE, **overrides)


def run(strategy: str, start: datetime, end: datetime,
        risk_pct: float = DEFAULT_RISK_PCT, **param_overrides) -> dict:
    reset_classes()
    p = make_params(risk_pct, **param_overrides)
    try:
        r = run_backtest(
            strategy_name=strategy, symbol=SYMBOL, timeframe=TIMEFRAME,
            start=start, end=end, cash=CASH, leverage=LEVERAGE,
            quiet=True, params_override=p, return_trades=True,
        )
        r["error"] = None
    except Exception as e:
        r = {"strategy": strategy, "error": str(e), "trades": 0,
             "backtest_return_pct": float("nan"), "after_funding_pct": float("nan"),
             "sharpe": float("nan"), "max_drawdown_pct": float("nan"),
             "win_rate_pct": float("nan"), "profit_factor": float("nan"),
             "trades_df": None}
    return r


# --- (b) ATR-aware exit classification --------------------------------------
def classify_exit_magnitude(t) -> str:
    """Coarse classification by return magnitude — works for ATR + fixed."""
    ret = float(t.get("ReturnPct", 0))
    if ret >= 0.015:
        return "big-win (>+1.5%)"
    if ret >= 0.003:
        return "small-win (+0.3%..+1.5%)"
    if ret <= -0.012:
        return "big-loss (<-1.2%)"
    if ret <= -0.003:
        return "small-loss (-0.3%..-1.2%)"
    return "near-flat (±0.3%)"


def exit_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["bucket"] = df.apply(classify_exit_magnitude, axis=1)
    df["return_pct"] = df["ReturnPct"].astype(float) * 100
    grouped = df.groupby("bucket").agg(
        count=("PnL", "size"),
        mean_ret=("return_pct", "mean"),
        total_pnl=("PnL", "sum"),
    )
    grouped["pct_of_trades"] = grouped["count"] / len(df) * 100
    bucket_order = ["big-win (>+1.5%)", "small-win (+0.3%..+1.5%)",
                    "near-flat (±0.3%)", "small-loss (-0.3%..-1.2%)",
                    "big-loss (<-1.2%)"]
    grouped = grouped.reindex([b for b in bucket_order if b in grouped.index])
    return grouped


# --- HTML helpers -----------------------------------------------------------
def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def fmt_num(v, p: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{p}f}"


def cls(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "ok" if v >= 0 else "bad"


# --- Sections ---------------------------------------------------------------
def section_walk_forward() -> tuple[str, dict]:
    print("\n[a] Walk-forward (5 OOS windows)")
    strategies = ["multifactor-v1", "multifactor-v2-strict", "v3-all", "v3-vol-regime-only"]
    grid: dict[tuple[str, str], dict] = {}
    for s in strategies:
        for label, start, end in WINDOWS:
            print(f"    {s} · {label} ...", end="", flush=True)
            t0 = time.time()
            r = run(s, start, end)
            grid[(s, label)] = r
            print(f" {fmt_pct(r.get('after_funding_pct'))} ({time.time()-t0:.1f}s)")

    # Build summary table: rows = strategies, cols = windows + summary
    header_cols = "".join(f"<th>{html.escape(w)}</th>" for w, _, _ in WINDOWS)
    rows_html = []
    for s in strategies:
        cells = []
        wins = 0
        total_ret = 0.0
        for label, _, _ in WINDOWS:
            r = grid[(s, label)]
            v = r.get("after_funding_pct")
            if v is not None and not pd.isna(v):
                if v > 0:
                    wins += 1
                total_ret += v
            cells.append(f"<td class='{cls(v)}'>{fmt_pct(v)}</td>")
        consistency = f"{wins}/{len(WINDOWS)}"
        sum_cls = "ok" if total_ret >= 0 else "bad"
        rows_html.append(f"<tr><td><strong>{html.escape(s)}</strong></td>"
                         + "".join(cells)
                         + f"<td><strong>{consistency}</strong></td>"
                         + f"<td class='{sum_cls}'><strong>{fmt_pct(total_ret)}</strong></td></tr>")
    table = f"""
<section>
  <h2>(a) Walk-forward — 5 OOS windows @ 1% risk</h2>
  <p class="muted">Per-window post-funding return. "Consistency" = # windows with positive return.
     "Cumulative" = sum of per-window returns (not compounded).</p>
  <table>
    <thead><tr><th>Strategy</th>{header_cols}<th>Consistency</th><th>Cumulative</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</section>
"""
    return table, grid


def section_atr_exit_analysis(walk_grid: dict) -> str:
    print("\n[b] ATR-aware exit analysis (v3-all)")
    # Aggregate trades across all windows for v3-all
    all_trades = []
    for label, _, _ in WINDOWS:
        r = walk_grid.get(("v3-all", label))
        if r and isinstance(r.get("trades_df"), pd.DataFrame):
            all_trades.append(r["trades_df"])
    if not all_trades:
        return "<section><h2>(b) ATR-aware exit analysis</h2><p>No trades.</p></section>"
    trades = pd.concat(all_trades, ignore_index=True)
    breakdown = exit_breakdown(trades)

    rows_html = []
    for bucket, row in breakdown.iterrows():
        rows_html.append(
            f"<tr><td>{html.escape(bucket)}</td>"
            f"<td>{int(row['count'])}</td>"
            f"<td>{row['pct_of_trades']:.1f}%</td>"
            f"<td class='{cls(row['mean_ret'])}'>{row['mean_ret']:+.2f}%</td>"
            f"<td class='{cls(row['total_pnl'])}'>${row['total_pnl']:+,.2f}</td></tr>"
        )

    return f"""
<section>
  <h2>(b) ATR-aware exit analysis — v3-all aggregated across all OOS windows</h2>
  <p class="muted">v3-all uses ATR-based stops, so fixed ±1.5% buckets don't apply.
     Classifying by return magnitude instead (works for both fixed and ATR strategies).</p>
  <table>
    <thead><tr><th>Bucket</th><th># Trades</th><th>% of all</th>
      <th>Mean return</th><th>Total PnL</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</section>
"""


def section_require_trend_ablation() -> str:
    print("\n[c] require_trend ablation (multifactor-v1)")
    start = WINDOWS[0][1]
    end = WINDOWS[-1][2]
    results = []
    for label, override in [("require_trend=ON", {}), ("require_trend=OFF", {"require_trend": False})]:
        print(f"    {label} ...", end="", flush=True)
        t0 = time.time()
        r = run("multifactor-v1", start, end, **override)
        r["__label"] = label
        results.append(r)
        print(f" {fmt_pct(r.get('after_funding_pct'))} ({time.time()-t0:.1f}s)")

    rows_html = []
    for r in results:
        ret = r.get("after_funding_pct")
        rows_html.append(
            f"<tr><td><strong>{html.escape(r['__label'])}</strong></td>"
            f"<td>{r.get('trades', 0)}</td>"
            f"<td class='{cls(ret)}'>{fmt_pct(ret)}</td>"
            f"<td>{fmt_num(r.get('sharpe'), 2)}</td>"
            f"<td>{fmt_num(r.get('max_drawdown_pct'), 1)}%</td>"
            f"<td>{fmt_num(r.get('win_rate_pct'), 1)}%</td>"
            f"<td>{fmt_num(r.get('profit_factor'), 2)}</td></tr>"
        )
    return f"""
<section>
  <h2>(c) require_trend ablation — multifactor-v1 across full window</h2>
  <p class="muted">Tests whether removing the trend-cross exit entirely (relying only on SL/TP)
     is better than the default behavior.</p>
  <table>
    <thead><tr><th>Config</th><th># Trades</th><th>Return</th><th>Sharpe</th>
      <th>Max DD</th><th>Win rate</th><th>Profit factor</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</section>
"""


def section_tuned_variants() -> str:
    print("\n[A+C] Tuned-variant comparison (multifactor-v1 baseline + 6 variants)")
    start = WINDOWS[0][1]
    end = WINDOWS[-1][2]
    variants = [
        ("multifactor-v1", "baseline (N=1, no floor)"),
        ("v1-debounce-2", "A: debounce N=2"),
        ("v1-debounce-3", "A: debounce N=3"),
        ("v1-debounce-4", "A: debounce N=4"),
        ("v1-floor-0.5",  "C: loss-floor -0.5%"),
        ("v1-floor-1.0",  "C: loss-floor -1.0%"),
        ("v1-deluxe",     "A+C: N=2 + floor -0.5%"),
    ]
    rows_html = []
    for name, label in variants:
        print(f"    {name} ...", end="", flush=True)
        t0 = time.time()
        r = run(name, start, end)
        ret = r.get("after_funding_pct")
        rows_html.append(
            f"<tr><td><strong>{html.escape(name)}</strong></td>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{r.get('trades', 0)}</td>"
            f"<td class='{cls(ret)}'>{fmt_pct(ret)}</td>"
            f"<td>{fmt_num(r.get('sharpe'), 2)}</td>"
            f"<td>{fmt_num(r.get('max_drawdown_pct'), 1)}%</td>"
            f"<td>{fmt_num(r.get('win_rate_pct'), 1)}%</td>"
            f"<td>{fmt_num(r.get('profit_factor'), 2)}</td></tr>"
        )
        print(f" {fmt_pct(ret)} ({time.time()-t0:.1f}s)")

    return f"""
<section>
  <h2>(A+C) Trend-exit tuning — multifactor-v1 variants, full window</h2>
  <p class="muted">All variants share v1's signal logic; only the trend-cross exit behavior differs.</p>
  <table>
    <thead><tr><th>Variant</th><th>Config</th><th># Trades</th><th>Return</th>
      <th>Sharpe</th><th>Max DD</th><th>Win rate</th><th>Profit factor</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</section>
"""


def render_html(walk_table: str, atr_section: str, trend_ablation: str,
                tuned_section: str) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Tuning report — walk-forward + trend-exit ablation</title>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --ok: #3fb950; --bad: #f85149;
  }}
  * {{ box-sizing: border-box }}
  body {{
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    max-width: 1400px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ font-size: 22px; margin: 0 0 6px 0 }}
  h2 {{ font-size: 16px; margin: 30px 0 10px 0 }}
  .muted, p.muted {{ color: var(--muted); font-size: 12.5px; margin: 0 0 12px 0 }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin-bottom: 18px;
           padding: 12px; background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
           font-size: 12.5px }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid var(--border);
            text-align: left; vertical-align: top }}
  th {{ background: #0a0f17; color: var(--muted); font-weight: 500; font-size: 11px;
        text-transform: uppercase; letter-spacing: .04em; white-space: nowrap }}
  tr:last-child td {{ border-bottom: none }}
  .ok  {{ color: var(--ok) }}
  .bad {{ color: var(--bad) }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 30px; text-align: center }}
  code {{ background: rgba(255,255,255,.05); padding: 1px 5px; border-radius: 4px }}
</style>
</head>
<body>

<h1>Tuning report — walk-forward + trend-exit ablation</h1>

<div class="meta">
  Five non-overlapping OOS windows over 2024-01-01 → 2026-05-18.
  <code>$10k</code> cash, <code>20×</code> leverage, fees + slippage + funding all subtracted.
  All runs use <code>risk_per_trade_pct = 1.0%</code> (baseline sizing).
</div>

{walk_table}

{atr_section}

{trend_ablation}

{tuned_section}

<div class="footer">
  Generated {generated} · <code>tools/build_full_tuning_report.py</code>
</div>
</body>
</html>
"""


def main() -> None:
    t_total = time.time()
    walk_table, walk_grid = section_walk_forward()
    atr_section = section_atr_exit_analysis(walk_grid)
    trend_ablation = section_require_trend_ablation()
    tuned_section = section_tuned_variants()

    html_out = render_html(walk_table, atr_section, trend_ablation, tuned_section)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"\nTotal: {time.time()-t_total:.1f}s")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
