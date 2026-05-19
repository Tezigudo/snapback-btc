"""
Sweep position sizing levels (5%, 7.5%, 10% margin per trade) across all strategies.

User asked: "what about 5-10% sizing of sizing" — explore the middle ground
between the original 1%-risk default and the catastrophic 20%-margin test.

Margin math (sl_pct=1.5%, leverage=20×):
    margin/equity = (risk_pct/100) / (sl_pct × leverage) = risk_pct / 30

So:
    5%  margin → risk_pct = 1.5%
    7.5%        → 2.25%
    10%         → 3.0%

We also include the original 1%-risk run (~3.3% margin) as the baseline,
and skip buy-and-hold (broken at leverage as we saw — gets liquidated).

For v3-atr-stops the SL is ATR-based, not 1.5%, so margin floats with vol —
those rows are advisory, not directly comparable to the fixed-pct strategies.
"""

from __future__ import annotations

import dataclasses
import html
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

# --- Configuration -----------------------------------------------------------
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "15m"
CASH = 10_000.0
LEVERAGE = 20

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2026, 5, 18, tzinfo=UTC)

# (label, risk_pct, derived_margin_pct)
SIZINGS: list[tuple[str, float, float]] = [
    ("1% risk (baseline)", 1.0, 3.3),
    ("5% margin", 1.5, 5.0),
    ("7.5% margin", 2.25, 7.5),
    ("10% margin", 3.0, 10.0),
]

# Strategies — exclude buy-and-hold (broken at 20× leverage).
STRATEGY_NAMES = [
    "snapback-v1",
    "multifactor-v1",
    "multifactor-v2-loose",
    "multifactor-v2-strict",
    "multifactor-v3",
    "v3-dist-ema-only",
    "v3-vol-regime-only",
    "v3-atr-stops-only",
    "v3-all",
]

OUT = REPO_ROOT / "reports" / "SIZING_SWEEP_RESULTS.html"


def reset_classes() -> None:
    """Reload variant modules so class-level defaults survive between runs."""
    import importlib

    import strategy.signals_multifactor_v2 as v2
    import strategy.signals_multifactor_v3 as v3
    importlib.reload(v2)
    importlib.reload(v3)
    STRATEGIES["multifactor-v2-loose"] = v2.DayTradeMultiFactorBTCv2Loose
    STRATEGIES["multifactor-v2-strict"] = v2.DayTradeMultiFactorBTCv2Strict
    STRATEGIES["multifactor-v3"] = v3.DayTradeMultiFactorBTCv3
    STRATEGIES["v3-dist-ema-only"] = v3.V3DistEmaOnly
    STRATEGIES["v3-vol-regime-only"] = v3.V3VolRegimeOnly
    STRATEGIES["v3-atr-stops-only"] = v3.V3AtrStopsOnly
    STRATEGIES["v3-all"] = v3.V3All


def make_params(risk_pct: float) -> StrategyParams:
    base = StrategyParams.from_yaml()
    return dataclasses.replace(base, risk_per_trade_pct=risk_pct, leverage=LEVERAGE)


def run_one(name: str, risk_pct: float) -> dict:
    print(f"    {name} @ risk={risk_pct}% ...", end="", flush=True)
    t0 = time.time()
    reset_classes()
    try:
        result = run_backtest(
            strategy_name=name,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start=START,
            end=END,
            cash=CASH,
            leverage=LEVERAGE,
            quiet=True,
            params_override=make_params(risk_pct),
            return_trades=True,
        )
        result["error"] = None
    except Exception as e:
        traceback.print_exc()
        result = {
            "strategy": name,
            "error": str(e),
            "trades": 0,
            "backtest_return_pct": float("nan"),
            "after_funding_pct": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown_pct": float("nan"),
            "win_rate_pct": float("nan"),
            "profit_factor": float("nan"),
            "avg_trade_pct": float("nan"),
            "trades_df": None,
            "leverage": LEVERAGE,
        }
    result["risk_pct"] = risk_pct
    print(f" done in {time.time() - t0:.1f}s")
    return result


# --- Rendering ---------------------------------------------------------------
def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def fmt_num(v, p: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{p}f}"


def cls_for(v: float) -> str:
    if v is None or pd.isna(v):
        return ""
    return "ok" if v >= 0 else "bad"


def summary_table_for_sizing(label: str, margin_pct: float, risk_pct: float,
                             results: list[dict]) -> str:
    rows = []
    for r in results:
        ret = r.get("backtest_return_pct")
        after = r.get("after_funding_pct")
        dd = r.get("max_drawdown_pct")
        kill_hit = "⚠️" if (isinstance(dd, (int, float)) and not pd.isna(dd) and dd <= -18) else ""
        rows.append(f"""
<tr>
  <td><strong>{html.escape(r['strategy'])}</strong></td>
  <td>{r.get('trades', 0)}</td>
  <td class="{cls_for(ret)}">{fmt_pct(ret)}</td>
  <td class="{cls_for(after)}">{fmt_pct(after)}</td>
  <td>{fmt_num(r.get('sharpe'), 2)}</td>
  <td>{fmt_num(dd, 1)}% {kill_hit}</td>
  <td>{fmt_num(r.get('win_rate_pct'), 1)}%</td>
  <td>{fmt_num(r.get('profit_factor'), 2)}</td>
</tr>""".strip())
    return f"""
<section>
  <h2>{html.escape(label)} <span class="muted">· risk_pct={risk_pct}% · margin≈{margin_pct}%</span></h2>
  <table>
    <thead><tr>
      <th>Strategy</th><th># Trades</th><th>Return (pre-fund)</th>
      <th>Return (post-fund)</th><th>Sharpe</th><th>Max DD</th>
      <th>Win rate</th><th>Profit factor</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def render_trades_block(strategy: str, sizing_label: str, df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return ""
    rows = []
    df = df.copy().reset_index(drop=True)
    for i, t in df.iterrows():
        side = "LONG" if float(t.get("Size", 0)) > 0 else "SHORT"
        pnl = float(t.get("PnL", 0))
        pnl_pct = float(t.get("ReturnPct", 0)) * 100
        cls = "ok" if pnl >= 0 else "bad"
        entry = pd.to_datetime(t.get("EntryTime")).strftime("%Y-%m-%d %H:%M")
        exit_ = pd.to_datetime(t.get("ExitTime")).strftime("%Y-%m-%d %H:%M")
        rows.append(
            f"<tr class='{cls}'><td>{i+1}</td><td>{side}</td>"
            f"<td>{entry}</td><td>{exit_}</td>"
            f"<td>${float(t.get('EntryPrice', 0)):,.2f}</td>"
            f"<td>${float(t.get('ExitPrice', 0)):,.2f}</td>"
            f"<td>${pnl:+,.2f}</td>"
            f"<td>{pnl_pct:+.2f}%</td></tr>"
        )
    summary_pnl = float(df["PnL"].sum()) if "PnL" in df.columns else 0.0
    summary_class = "ok" if summary_pnl >= 0 else "bad"

    return f"""
<details>
  <summary><strong>{html.escape(strategy)}</strong> · {html.escape(sizing_label)} · {len(df)} trades ·
    <span class="{summary_class}">net P&amp;L ${summary_pnl:+,.2f}</span>
  </summary>
  <table class="trades">
    <thead><tr><th>#</th><th>Side</th><th>Entry</th><th>Exit</th>
      <th>Entry $</th><th>Exit $</th><th>P&amp;L</th><th>Return</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</details>"""


def render_html(all_runs: dict[str, list[dict]]) -> str:
    sizing_sections = []
    for label, risk_pct, margin_pct in SIZINGS:
        sizing_sections.append(summary_table_for_sizing(label, margin_pct, risk_pct, all_runs[label]))

    trade_blocks = []
    for label, _, _ in SIZINGS:
        for r in all_runs[label]:
            trade_blocks.append(render_trades_block(r["strategy"], label, r.get("trades_df")))

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Sizing sweep — 5–10% margin across all strategies</title>
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
  h2 {{ font-size: 16px; margin: 24px 0 10px 0 }}
  .muted {{ color: var(--muted); font-weight: normal }}
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
  details {{ background: var(--panel); border: 1px solid var(--border);
             border-radius: 8px; padding: 10px 14px; margin: 8px 0 }}
  summary {{ cursor: pointer; padding: 4px 0; user-select: none }}
  details[open] summary {{ margin-bottom: 8px; border-bottom: 1px solid var(--border);
                            padding-bottom: 8px }}
  table.trades {{ border: 1px solid var(--border); border-radius: 6px;
                  font-size: 11.5px; margin-top: 6px }}
  table.trades th, table.trades td {{ padding: 5px 9px }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 30px; text-align: center }}
  code {{ background: rgba(255,255,255,.05); padding: 1px 5px; border-radius: 4px }}
  h3 {{ font-size: 14px; color: var(--muted); margin: 30px 0 12px 0;
        text-transform: uppercase; letter-spacing: .05em }}
</style>
</head>
<body>

<h1>Position sizing sweep — 5%, 7.5%, 10% margin per trade</h1>

<div class="meta">
  <strong>Window:</strong> {START.date()} → {END.date()} ·
  <strong>Symbol:</strong> {SYMBOL} · <strong>TF:</strong> {TIMEFRAME} ·
  <strong>Cash:</strong> ${CASH:,.0f} · <strong>Leverage:</strong> {LEVERAGE}× ·
  Friction: 5 bps/side + funding subtracted.<br><br>
  Sizing is controlled by <code>risk_per_trade_pct</code>. At sl_pct=1.5%, lev=20×, the
  resulting per-position margin is approximately <code>risk_pct ÷ 30 × 100</code>%.
  Strategies with ATR-based stops (v3, v3-atr-stops, v3-all) use volatility-adaptive
  sizing and their effective margin floats — not directly comparable to fixed-pct stops.
  <strong>⚠️ in Max DD column</strong> = drawdown breached -18% (live kill switch would have fired).
</div>

{''.join(sizing_sections)}

<h3>Trade history (click to expand)</h3>
{''.join(trade_blocks)}

<div class="footer">
  Generated {generated} · <code>tools/build_sizing_sweep.py</code>
</div>
</body>
</html>
"""


def main() -> None:
    print(f"Sizing sweep · {START.date()} → {END.date()}")
    print(f"  Strategies: {len(STRATEGY_NAMES)} · Sizings: {len(SIZINGS)}")
    print()

    all_runs: dict[str, list[dict]] = {}
    t_total = time.time()
    for label, risk_pct, margin_pct in SIZINGS:
        print(f"  ▶ {label}  (risk_pct={risk_pct}%  margin≈{margin_pct}%)")
        all_runs[label] = [run_one(name, risk_pct) for name in STRATEGY_NAMES]
        print()

    print(f"All {len(SIZINGS) * len(STRATEGY_NAMES)} backtests done in {time.time() - t_total:.1f}s")
    html_out = render_html(all_runs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
