"""
Backtest every available strategy with 20%-of-equity margin per position.

User asked: "put margin 20% of future port with risk managed, backtest all,
open result + trade history."

Margin math (testing only, NOT promoted to params.yaml):

    margin_per_position = qty × price / leverage
    qty = (equity × risk_pct/100) / (sl_pct × price)        [risk-based]
    ⇒ margin = (equity × risk_pct/100) / (sl_pct × leverage)
    ⇒ margin / equity = (risk_pct/100) / (sl_pct × leverage)

At sl_pct = 1.5%, leverage = 20×, target margin = 20% of equity:
    0.20 = (risk_pct/100) / (0.015 × 20) = (risk_pct/100) / 0.30
    risk_pct = 0.20 × 0.30 × 100 = 6.0%

So we override `risk_per_trade_pct = 6.0` for every strategy run, keeping
SL/TP/leverage/kill-switch behaviour intact (risk-managed).

For multifactor-v3 + variants the SL is ATR-based, not fixed-pct, so margin
won't be exactly 20% (it floats with volatility). That's correct — the user
asked for risk-managed sizing, not fixed dollar margin.

Output: reports/MARGIN20_RESULTS.html with summary table + collapsible trade
history per strategy.
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
RISK_PCT = 6.0  # ⇒ ~20% margin per position at sl=1.5%, lev=20×

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2026, 5, 18, tzinfo=UTC)

# Order matters for the report — sized strategies first, B&H last as anchor.
ALL_STRATEGIES = [
    "snapback-v1",
    "multifactor-v1",
    "multifactor-v2-loose",
    "multifactor-v2-strict",
    "multifactor-v3",
    "v3-dist-ema-only",
    "v3-vol-regime-only",
    "v3-atr-stops-only",
    "v3-all",
    "buy-and-hold",
]

OUT = REPO_ROOT / "reports" / "MARGIN20_RESULTS.html"


# --- Per-strategy run --------------------------------------------------------
def reset_classes() -> None:
    """Reset class attributes so ablation-variant flags survive between runs."""
    # signals_multifactor_v3 variant classes carry CLASS-LEVEL enable_* flags
    # that must not be overwritten by _apply_params_to_class. Just re-import.
    import importlib

    import strategy.signals_multifactor_v3 as v3
    importlib.reload(v3)
    STRATEGIES["multifactor-v3"] = v3.DayTradeMultiFactorBTCv3
    STRATEGIES["v3-dist-ema-only"] = v3.V3DistEmaOnly
    STRATEGIES["v3-vol-regime-only"] = v3.V3VolRegimeOnly
    STRATEGIES["v3-atr-stops-only"] = v3.V3AtrStopsOnly
    STRATEGIES["v3-all"] = v3.V3All


def make_params() -> StrategyParams:
    base = StrategyParams.from_yaml()
    return dataclasses.replace(base, risk_per_trade_pct=RISK_PCT, leverage=LEVERAGE)


def run_one(name: str) -> dict:
    print(f"  → running {name} ...", flush=True)
    t0 = time.time()
    reset_classes()
    params_override = None if name == "buy-and-hold" else make_params()
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
            params_override=params_override,
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
            "trades_df": None,
            "leverage": LEVERAGE,
        }
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return result


# --- HTML rendering ----------------------------------------------------------
def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def fmt_num(v, p: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{p}f}"


def render_summary_row(r: dict) -> str:
    ret = r.get("backtest_return_pct")
    after = r.get("after_funding_pct")
    pnl_class = ""
    if isinstance(after, (int, float)) and not pd.isna(after):
        pnl_class = "ok" if after >= 0 else "bad"
    err = r.get("error")
    return f"""
<tr>
  <td><strong>{html.escape(r['strategy'])}</strong></td>
  <td>{r.get('trades', 0)}</td>
  <td class="{pnl_class}">{fmt_pct(ret)}</td>
  <td class="{pnl_class}">{fmt_pct(after)}</td>
  <td>{fmt_num(r.get('sharpe'), 2)}</td>
  <td>{fmt_num(r.get('max_drawdown_pct'), 2)}%</td>
  <td>{fmt_num(r.get('win_rate_pct'), 1)}%</td>
  <td>{fmt_num(r.get('avg_trade_pct'), 3)}%</td>
  <td>{fmt_num(r.get('profit_factor'), 2)}</td>
  <td>{('<span class="bad">'+html.escape(err)+'</span>') if err else '✓'}</td>
</tr>
""".strip()


def render_trades_table(strategy: str, df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return f'<details><summary><strong>{html.escape(strategy)}</strong> — no trades</summary></details>'

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
            f"<td>{float(t.get('Size', 0)):.4f}</td>"
            f"<td>${pnl:+,.2f}</td>"
            f"<td>{pnl_pct:+.2f}%</td></tr>"
        )

    summary_pnl = float(df["PnL"].sum()) if "PnL" in df.columns else 0.0
    summary_class = "ok" if summary_pnl >= 0 else "bad"

    return f"""
<details>
  <summary>
    <strong>{html.escape(strategy)}</strong> · {len(df)} trades ·
    <span class="{summary_class}">net P&amp;L ${summary_pnl:+,.2f}</span>
  </summary>
  <table class="trades">
    <thead><tr>
      <th>#</th><th>Side</th><th>Entry time</th><th>Exit time</th>
      <th>Entry $</th><th>Exit $</th><th>Size BTC</th><th>P&amp;L</th><th>Return</th>
    </tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</details>
""".strip()


def render_html(results: list[dict]) -> str:
    summary_rows = "\n".join(render_summary_row(r) for r in results)
    trades_sections = "\n".join(
        render_trades_table(r["strategy"], r.get("trades_df")) for r in results
    )
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Margin-20% backtest — all strategies</title>
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
  h2 {{ font-size: 16px; color: var(--muted); margin: 30px 0 12px 0;
        text-transform: uppercase; letter-spacing: .05em }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin-bottom: 18px;
           padding: 12px; background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px }}
  .meta code {{ color: var(--text) }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
           font-size: 12.5px }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left;
            vertical-align: top }}
  th {{ background: #0a0f17; color: var(--muted); font-weight: 500; font-size: 11px;
        text-transform: uppercase; letter-spacing: .04em; white-space: nowrap }}
  tr:last-child td {{ border-bottom: none }}
  .ok  {{ color: var(--ok) }}
  .bad {{ color: var(--bad) }}
  details {{ background: var(--panel); border: 1px solid var(--border);
             border-radius: 8px; padding: 10px 14px; margin: 10px 0 }}
  summary {{ cursor: pointer; padding: 4px 0; user-select: none }}
  details[open] summary {{ margin-bottom: 10px; border-bottom: 1px solid var(--border);
                            padding-bottom: 10px }}
  table.trades {{ border: 1px solid var(--border); border-radius: 6px;
                  font-size: 11.5px; margin-top: 6px }}
  table.trades th, table.trades td {{ padding: 5px 9px }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 30px; text-align: center }}
  code {{ background: rgba(255,255,255,.05); padding: 1px 5px; border-radius: 4px }}
</style>
</head>
<body>

<h1>Margin-20% backtest — all strategies</h1>

<div class="meta">
  <strong>Window:</strong> {START.date()} → {END.date()} ·
  <strong>Symbol:</strong> {SYMBOL} ·
  <strong>Entry TF:</strong> {TIMEFRAME} ·
  <strong>Cash:</strong> ${CASH:,.0f} ·
  <strong>Leverage:</strong> {LEVERAGE}× ·
  <strong>Risk per trade:</strong> {RISK_PCT}% (→ ~20% margin per position at sl=1.5%)
  <br><br>
  <strong>Caveat:</strong> This is a "what if I sized larger" test, not a config change.
  For multifactor-v3 + variants, SL is ATR-based, so margin floats with vol — not a
  fixed 20%. The risk-management math still bounds per-trade loss to ~6% of equity at SL.
  Funding cost is subtracted from returns. Fees + slippage modelled at 5 bps/side.
</div>

<h2>Summary</h2>
<table>
  <thead><tr>
    <th>Strategy</th>
    <th># Trades</th>
    <th>Return (pre-funding)</th>
    <th>Return (post-funding)</th>
    <th>Sharpe</th>
    <th>Max DD</th>
    <th>Win rate</th>
    <th>Avg trade</th>
    <th>Profit factor</th>
    <th>Status</th>
  </tr></thead>
  <tbody>
    {summary_rows}
  </tbody>
</table>

<h2>Trade history (click to expand)</h2>
{trades_sections}

<div class="footer">
  Generated {generated} · <code>tools/build_margin20_results.py</code>
</div>
</body>
</html>
"""


# --- Main --------------------------------------------------------------------
def main() -> None:
    print(f"Running {len(ALL_STRATEGIES)} strategies, {START.date()} → {END.date()}")
    print(f"  risk_per_trade_pct = {RISK_PCT}%  ⇒  ~20% margin per position")
    print(f"  leverage = {LEVERAGE}×  cash = ${CASH:,.0f}")
    print()

    t_total = time.time()
    results = [run_one(name) for name in ALL_STRATEGIES]
    print(f"\nAll backtests done in {time.time() - t_total:.1f}s")

    html_out = render_html(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
