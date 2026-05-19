"""
multifactor-v1 friction stress — 12 OOS windows × 3 friction levels (4/8/12 bps).

Insurance check before going live. v1 is the ONLY validated strategy after
the causal-swing fix invalidated all v2/v3 backtests (see LUNCH_SUMMARY.html).
v1 doesn't use swing detection — its numbers are unaffected by the fix. This
sweep confirms v1 still profits under elevated trading costs.

3 friction levels (per-side commission):
  - 4 bps: Binance taker fee only (best case, no slippage)
  - 8 bps: 2× stress (Binance + meaningful slippage)
  - 12 bps: 3× stress (worst-case illiquid fill)

If v1 still wins ≥6/12 windows at 12 bps with positive cumulative, we have
real headroom before going to mainnet.
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
RISK_PCT = 2.0  # matches config/params.yaml — the live bot's actual sizing

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
    ("4 bps (Binance fee only)", 0.0004),
    ("8 bps (2× stress)",        0.0008),
    ("12 bps (3× worst-case)",   0.0012),
]

STRATEGY = "multifactor-v1"

OUT = REPO_ROOT / "reports" / "V1_FRICTION_STRESS.html"


def make_params() -> StrategyParams:
    base = StrategyParams.from_yaml()
    return dataclasses.replace(base, risk_per_trade_pct=RISK_PCT, leverage=LEVERAGE)


def run(start: datetime, end: datetime, commission: float) -> dict:
    try:
        r = run_backtest(
            strategy_name=STRATEGY, symbol=SYMBOL, timeframe=TIMEFRAME,
            start=start, end=end, cash=CASH, leverage=LEVERAGE,
            quiet=True, params_override=make_params(), commission=commission,
        )
        r["error"] = None
    except Exception as e:
        r = {"strategy": STRATEGY, "error": str(e),
             "after_funding_pct": float("nan"), "max_drawdown_pct": float("nan"),
             "win_rate_pct": float("nan"), "trades": 0}
    return r


def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def cls(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "ok" if v >= 0 else "bad"


def main() -> None:
    print(f"v1 friction stress: {len(FRICTION)} friction × {len(WINDOWS)} windows = "
          f"{len(FRICTION) * len(WINDOWS)} backtests")
    print()

    t_total = time.time()
    grid: dict[tuple[str, str], dict] = {}
    for fric_label, commission in FRICTION:
        print(f"  ▶ {fric_label}")
        for win_label, start, end in WINDOWS:
            t0 = time.time()
            r = run(start, end, commission)
            grid[(fric_label, win_label)] = r
            print(f"    {win_label} ... {fmt_pct(r.get('after_funding_pct'))}"
                  f" (DD {r.get('max_drawdown_pct', 0):.1f}%, "
                  f"WR {r.get('win_rate_pct', 0):.1f}%, "
                  f"trades {r.get('trades', 0)}, "
                  f"{time.time()-t0:.1f}s)")
        print()

    # Build per-friction rows
    header_cols = "".join(f"<th>{html.escape(w.split(' ', 1)[1])}</th>" for w, _, _ in WINDOWS)
    rows = []
    for fric_label, commission in FRICTION:
        cells = []
        wins = 0
        total = 0.0
        kill = 0
        all_trades = 0
        for win_label, _, _ in WINDOWS:
            r = grid.get((fric_label, win_label))
            v = r.get("after_funding_pct") if r else None
            dd = r.get("max_drawdown_pct") if r else None
            n = r.get("trades", 0) if r else 0
            if v is not None and not pd.isna(v):
                if v > 0:
                    wins += 1
                total += v
            if dd is not None and not pd.isna(dd) and dd <= -18:
                kill += 1
            all_trades += int(n or 0)
            cells.append(f"<td class='{cls(v)}'>{fmt_pct(v)}</td>")
        wins_cls = "ok" if wins >= 8 else ("bad" if wins <= 4 else "")
        total_cls = cls(total)
        rows.append(
            f"<tr><td><strong>{html.escape(fric_label)}</strong></td>"
            + "".join(cells)
            + f"<td class='{wins_cls}'><strong>{wins}/{len(WINDOWS)}</strong></td>"
            + f"<td class='{total_cls}'><strong>{fmt_pct(total)}</strong></td>"
            + f"<td>{kill}/{len(WINDOWS)}</td>"
            + f"<td>{all_trades}</td></tr>"
        )

    print(f"All backtests done in {time.time()-t_total:.1f}s")

    # Verdict line
    fric_lookup = {label: 0.0 for label, _ in FRICTION}
    for fric_label, _ in FRICTION:
        s = 0.0
        for win_label, _, _ in WINDOWS:
            r = grid.get((fric_label, win_label))
            v = r.get("after_funding_pct") if r else None
            if v is not None and not pd.isna(v):
                s += v
        fric_lookup[fric_label] = s

    verdict_cls = "ok" if fric_lookup["12 bps (3× worst-case)"] > 0 else "bad"
    verdict_text = (
        f"v1 holds up — even at 12 bps/side (3× normal friction) "
        f"cumulative is {fmt_pct(fric_lookup['12 bps (3× worst-case)'])}."
        if fric_lookup["12 bps (3× worst-case)"] > 0
        else f"v1 falls apart at 12 bps/side "
             f"(cumulative {fmt_pct(fric_lookup['12 bps (3× worst-case)'])}). "
             f"Beware deploying to thin liquidity or wide spread venues."
    )

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    html_out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>multifactor-v1 friction stress — 4/8/12 bps</title>
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
  .meta, .verdict {{ font-size: 12.5px; margin-bottom: 18px;
           padding: 12px; background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px }}
  .meta {{ color: var(--muted) }}
  .verdict.ok  {{ border-left: 4px solid var(--ok); color: var(--text) }}
  .verdict.bad {{ border-left: 4px solid var(--bad); color: var(--text) }}
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
<h1>multifactor-v1 friction stress · 12 OOS windows × 4/8/12 bps/side</h1>
<div class="meta">
  Insurance check before going live. v1 is the ONLY validated strategy
  post-causal-fix (see LUNCH_SUMMARY.html). v1 doesn't use swing detection —
  it's unaffected by the lookahead bias that killed v2/v3.<br><br>
  <code>$10k</code> cash, <code>20×</code> leverage, <code>risk_pct={RISK_PCT}%</code>
  (matches config/params.yaml — the actual live bot sizing).
  <strong>Kill-switch hits</strong> = windows where Max DD ≤ -18% (live bot would HALT).
</div>
<div class="verdict {verdict_cls}"><strong>Verdict:</strong> {html.escape(verdict_text)}</div>

<table>
  <thead><tr><th>Friction</th>{header_cols}
    <th>Wins</th><th>Cumulative</th><th>Kill-hits</th><th>Total trades</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>

<div class="footer">Generated {generated} · <code>tools/build_v1_friction_stress.py</code></div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
