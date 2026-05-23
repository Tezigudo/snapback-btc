"""
Friction stress test — does v3-all hold up when fees + slippage are worse?

We default to 5 bps/side (4 bps Binance taker fee + 1 bps slip proxy). Live
execution can be worse: ill-timed fills near a tick, partial fills, depth
shortfall. This sweep tests 5 / 10 / 15 / 20 bps/side to see how each
strategy degrades.

Sizing: 5% margin and 10% margin (per user request — risk_pct 1.5 / 3.0
at sl_pct 1.5%, lev 20×). The live bot already uses risk_pct=2.0 (≈6.7%
margin), which sits inside this range.

5 OOS windows, 2 strategies (v3-all + multifactor-v1 baseline), 4 friction
levels, 2 sizings → 80 backtests.
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

WINDOWS = [
    ("W1 2024-H1", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC)),
    ("W2 2024-H2", datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)),
    ("W3 2025-H1", datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    ("W4 2025-H2", datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
    ("W5 2026-YTD", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 5, 18, tzinfo=UTC)),
]

# (label, commission_per_side)
FRICTION = [
    ("baseline 5bps", 0.0005),
    ("stress 10bps",  0.0010),
    ("stress 15bps",  0.0015),
    ("worst 20bps",   0.0020),
]

# (label, risk_pct → margin approx at sl=1.5%, lev=20×)
SIZINGS = [
    ("5% margin",   1.5),
    ("10% margin",  3.0),
]

STRATEGIES_UNDER_TEST = ["v3-all", "multifactor-v1"]

OUT = REPO_ROOT / "reports" / "FRICTION_STRESS_RESULTS.html"


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


def make_params(risk_pct: float) -> StrategyParams:
    base = StrategyParams.from_yaml()
    return dataclasses.replace(base, risk_per_trade_pct=risk_pct, leverage=LEVERAGE)


def run(strategy: str, start: datetime, end: datetime,
        risk_pct: float, commission: float) -> dict:
    reset_classes()
    p = make_params(risk_pct)
    try:
        r = run_backtest(
            strategy_name=strategy, symbol=SYMBOL, timeframe=TIMEFRAME,
            start=start, end=end, cash=CASH, leverage=LEVERAGE,
            quiet=True, params_override=p, commission=commission,
        )
        r["error"] = None
    except Exception as e:
        r = {"strategy": strategy, "error": str(e),
             "backtest_return_pct": float("nan"), "after_funding_pct": float("nan"),
             "sharpe": float("nan"), "max_drawdown_pct": float("nan")}
    return r


def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def cls(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "ok" if v >= 0 else "bad"


def build_sizing_section(label: str, risk_pct: float,
                         grid: dict[tuple[str, str, str], dict]) -> str:
    """Build a section per sizing showing strategy × friction matrix."""
    out = [f'<h2>{html.escape(label)}<span class="muted"> · risk_pct={risk_pct}%</span></h2>']
    for strategy in STRATEGIES_UNDER_TEST:
        # Per-strategy: friction levels × OOS windows
        rows = []
        for fric_label, _ in FRICTION:
            cells = []
            wins = 0
            total = 0.0
            for win_label, _, _ in WINDOWS:
                r = grid.get((strategy, fric_label, win_label))
                v = r.get("after_funding_pct") if r else None
                if v is not None and not pd.isna(v):
                    if v > 0:
                        wins += 1
                    total += v
                cells.append(f"<td class='{cls(v)}'>{fmt_pct(v)}</td>")
            rows.append(f"<tr><td><strong>{html.escape(fric_label)}</strong></td>"
                        + "".join(cells)
                        + f"<td><strong>{wins}/{len(WINDOWS)}</strong></td>"
                        + f"<td class='{cls(total)}'><strong>{fmt_pct(total)}</strong></td></tr>")
        header_cols = "".join(f"<th>{html.escape(w)}</th>" for w, _, _ in WINDOWS)
        out.append(f"""
<h3>{html.escape(strategy)}</h3>
<table>
  <thead><tr><th>Friction</th>{header_cols}<th>Win/5</th><th>Cumulative</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
""")
    return "\n".join(out)


def main() -> None:
    print(f"Friction stress: {len(STRATEGIES_UNDER_TEST)} strategies × "
          f"{len(FRICTION)} friction levels × {len(SIZINGS)} sizings × "
          f"{len(WINDOWS)} windows")
    print()
    t_total = time.time()
    sections_per_sizing: dict[str, dict] = {label: {} for label, _ in SIZINGS}

    for siz_label, risk_pct in SIZINGS:
        print(f"  ▶ {siz_label}")
        siz_grid: dict[tuple[str, str, str], dict] = {}
        for fric_label, commission in FRICTION:
            for strategy in STRATEGIES_UNDER_TEST:
                for win_label, start, end in WINDOWS:
                    print(f"    {strategy} · {fric_label} · {win_label} ...",
                          end="", flush=True)
                    t0 = time.time()
                    r = run(strategy, start, end, risk_pct, commission)
                    siz_grid[(strategy, fric_label, win_label)] = r
                    print(f" {fmt_pct(r.get('after_funding_pct'))} ({time.time()-t0:.1f}s)")
        sections_per_sizing[siz_label] = siz_grid
        print()

    section_html_parts = []
    for siz_label, risk_pct in SIZINGS:
        section_html_parts.append(
            build_sizing_section(siz_label, risk_pct, sections_per_sizing[siz_label])
        )

    print(f"All backtests done in {time.time()-t_total:.1f}s")

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    html_out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Friction stress test — v3-all vs v1 at 5-10% margin</title>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --ok: #3fb950; --bad: #f85149;
  }}
  * {{ box-sizing: border-box }}
  body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    max-width: 1400px; margin-left: auto; margin-right: auto; }}
  h1 {{ font-size: 22px; margin: 0 0 6px 0 }}
  h2 {{ font-size: 16px; margin: 30px 0 6px 0 }}
  h3 {{ font-size: 14px; margin: 18px 0 8px 0; color: var(--text) }}
  .muted {{ color: var(--muted); font-weight: normal; font-size: 12px }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin-bottom: 18px;
           padding: 12px; background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
           font-size: 12.5px; margin-bottom: 14px }}
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
<h1>Friction stress — does v3-all hold up at higher fees?</h1>
<div class="meta">
  Same 5 OOS windows as walk-forward. Cash $10k, 20× leverage. Risk_pct chosen
  to land ~5% or ~10% margin per position (at sl=1.5%). For v3-all (ATR stops)
  the effective margin floats with volatility. <strong>Baseline 5bps</strong> is
  the same friction used in our prior walk-forward report. Stress levels double
  and triple it; <strong>20bps</strong> is a near-worst-case daytime liquidity
  scenario.
</div>
{''.join(section_html_parts)}
<div class="footer">Generated {generated} · <code>tools/build_friction_stress.py</code></div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
