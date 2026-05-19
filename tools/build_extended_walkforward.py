"""
Extended walk-forward: 2020-2026, v3-all + wider-stop variants under friction.

Goals:
1. Test across 6 years (12 half-year OOS windows) instead of just 28 months.
2. Stress-test wider ATR stops as a friction-tolerance improvement to v3-all.
3. Validate at user's actual sizing (5-10% margin with $100 port).

Each strategy is tested:
  - Across all available OOS windows (data starts 2020-05-09)
  - At baseline 5 bps and stress 10 bps friction
  - At 7.5% margin (risk_pct = 2.25%) — the midpoint of user's 5-10% target
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
RISK_PCT = 2.25  # ~7.5% margin per position at sl=1.5%, lev=20×

# Data starts 2020-05-09. We start OOS at 2020-07-01 to allow warm-up.
WINDOWS = [
    ("W01 2020-H2", datetime(2020, 7, 1, tzinfo=UTC), datetime(2020, 12, 31, tzinfo=UTC)),
    ("W02 2021-H1", datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 6, 30, tzinfo=UTC)),
    ("W03 2021-H2", datetime(2021, 7, 1, tzinfo=UTC), datetime(2021, 12, 31, tzinfo=UTC)),
    ("W04 2022-H1", datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 6, 30, tzinfo=UTC)),
    ("W05 2022-H2", datetime(2022, 7, 1, tzinfo=UTC), datetime(2022, 12, 31, tzinfo=UTC)),
    ("W06 2023-H1", datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
    ("W07 2023-H2", datetime(2023, 7, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)),
    ("W08 2024-H1", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC)),
    ("W09 2024-H2", datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)),
    ("W10 2025-H1", datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    ("W11 2025-H2", datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
    ("W12 2026-YTD", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 5, 18, tzinfo=UTC)),
]

FRICTION = [
    ("5 bps (baseline)", 0.0005),
    ("10 bps (stress)", 0.0010),
]

STRATEGIES_UNDER_TEST = [
    "multifactor-v1",
    "v3-all",
    "v3-all-wider-2",
    "v3-all-wider-3",
    "v3-all-wider-4",
]

OUT = REPO_ROOT / "reports" / "EXTENDED_WALKFORWARD.html"


def reset_classes() -> None:
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
    STRATEGIES["v3-all-wider-2"] = v3.V3AllWider2
    STRATEGIES["v3-all-wider-3"] = v3.V3AllWider3
    STRATEGIES["v3-all-wider-4"] = v3.V3AllWider4


def make_params() -> StrategyParams:
    base = StrategyParams.from_yaml()
    return dataclasses.replace(base, risk_per_trade_pct=RISK_PCT, leverage=LEVERAGE)


def run(strategy: str, start: datetime, end: datetime, commission: float) -> dict:
    reset_classes()
    try:
        r = run_backtest(
            strategy_name=strategy, symbol=SYMBOL, timeframe=TIMEFRAME,
            start=start, end=end, cash=CASH, leverage=LEVERAGE,
            quiet=True, params_override=make_params(), commission=commission,
        )
        r["error"] = None
    except Exception as e:
        r = {"strategy": strategy, "error": str(e),
             "after_funding_pct": float("nan"), "max_drawdown_pct": float("nan")}
    return r


def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def cls(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "ok" if v >= 0 else "bad"


def build_friction_section(fric_label: str, commission: float,
                           grid: dict[tuple[str, str], dict]) -> str:
    header_cols = "".join(f"<th>{html.escape(w.split(' ', 1)[1])}</th>" for w, _, _ in WINDOWS)
    rows = []
    for strategy in STRATEGIES_UNDER_TEST:
        cells = []
        wins = 0
        total = 0.0
        kill_count = 0
        for win_label, _, _ in WINDOWS:
            r = grid.get((strategy, win_label))
            v = r.get("after_funding_pct") if r else None
            dd = r.get("max_drawdown_pct") if r else None
            if v is not None and not pd.isna(v):
                if v > 0:
                    wins += 1
                total += v
            if dd is not None and not pd.isna(dd) and dd <= -18:
                kill_count += 1
            cells.append(f"<td class='{cls(v)}'>{fmt_pct(v)}</td>")
        wins_cls = "ok" if wins >= 8 else ("bad" if wins <= 4 else "")
        total_cls = cls(total)
        rows.append(
            f"<tr><td><strong>{html.escape(strategy)}</strong></td>"
            + "".join(cells)
            + f"<td class='{wins_cls}'><strong>{wins}/{len(WINDOWS)}</strong></td>"
            + f"<td class='{total_cls}'><strong>{fmt_pct(total)}</strong></td>"
            + f"<td>{kill_count}/{len(WINDOWS)}</td></tr>"
        )
    return f"""
<section>
  <h2>{html.escape(fric_label)} <span class="muted">· commission_per_side={commission*1e4:.0f}bps</span></h2>
  <table>
    <thead><tr><th>Strategy</th>{header_cols}<th>Wins</th><th>Cumulative</th><th>Kill-switch hits</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def main() -> None:
    print(f"Extended walk-forward: {len(STRATEGIES_UNDER_TEST)} strategies × "
          f"{len(FRICTION)} friction × {len(WINDOWS)} windows = "
          f"{len(STRATEGIES_UNDER_TEST) * len(FRICTION) * len(WINDOWS)} backtests")
    print()

    t_total = time.time()
    grids: dict[str, dict[tuple[str, str], dict]] = {}
    for fric_label, commission in FRICTION:
        print(f"  ▶ {fric_label}")
        g: dict[tuple[str, str], dict] = {}
        for strategy in STRATEGIES_UNDER_TEST:
            for win_label, start, end in WINDOWS:
                t0 = time.time()
                r = run(strategy, start, end, commission)
                g[(strategy, win_label)] = r
                print(f"    {strategy} · {win_label} ... {fmt_pct(r.get('after_funding_pct'))}"
                      f" (DD {r.get('max_drawdown_pct', 0):.1f}%, {time.time()-t0:.1f}s)")
        grids[fric_label] = g
        print()

    sections = "\n".join(
        build_friction_section(label, c, grids[label]) for label, c in FRICTION
    )

    print(f"All backtests done in {time.time()-t_total:.1f}s")

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    html_out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Extended walk-forward 2020-2026 · v3-all wider-stop variants</title>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --ok: #3fb950; --bad: #f85149;
  }}
  * {{ box-sizing: border-box }}
  body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    max-width: 1500px; margin-left: auto; margin-right: auto; }}
  h1 {{ font-size: 22px; margin: 0 0 6px 0 }}
  h2 {{ font-size: 16px; margin: 30px 0 10px 0 }}
  .muted {{ color: var(--muted); font-weight: normal; font-size: 12px }}
  .meta {{ color: var(--muted); font-size: 12.5px; margin-bottom: 18px;
           padding: 12px; background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
           font-size: 11.5px }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid var(--border);
            text-align: center; vertical-align: top; white-space: nowrap }}
  th:first-child, td:first-child {{ text-align: left }}
  th {{ background: #0a0f17; color: var(--muted); font-weight: 500; font-size: 10.5px;
        text-transform: uppercase; letter-spacing: .04em }}
  tr:last-child td {{ border-bottom: none }}
  .ok  {{ color: var(--ok) }}
  .bad {{ color: var(--bad) }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 30px; text-align: center }}
  code {{ background: rgba(255,255,255,.05); padding: 1px 5px; border-radius: 4px }}
</style>
</head>
<body>
<h1>Extended walk-forward · 2020-H2 → 2026-YTD · v3-all wider-stop variants</h1>
<div class="meta">
  12 non-overlapping OOS windows over 6 years (data starts 2020-05-09).
  <code>$10k</code> cash, <code>20×</code> leverage, <code>risk_pct={RISK_PCT}%</code> (~7.5% margin).
  Friction includes Binance taker fee (4 bps) + slippage proxy.
  <strong>Wider-stop variants</strong> trade ATR multipliers up to test friction tolerance:
  baseline V3All uses sl_k=1.5/tp_k=3.0; wider-2 = 2.0/4.0; wider-3 = 3.0/6.0; wider-4 = 4.0/8.0.
  Same 2:1 R:R maintained throughout. <br>
  <strong>Kill-switch hits</strong> column counts windows where Max DD ≤ -18% (live bot would HALT).
</div>
{sections}
<div class="footer">Generated {generated} · <code>tools/build_extended_walkforward.py</code></div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
