"""
Granular ATR-multiplier sweep to find the true optimum.

Tests k = 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0 (TP = 2 × SL throughout).
Same 12 OOS windows, same 7.5% margin, both 5 bps and 10 bps friction.
"""
from __future__ import annotations
import dataclasses, html, sys, time
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

SYMBOL, TIMEFRAME, CASH, LEVERAGE, RISK_PCT = "BTC/USDT:USDT", "15m", 10_000.0, 20, 2.25

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

VARIANTS = [
    ("v3-all",         1.5),
    ("v3-all-k-2.5",   2.5),
    ("v3-all-wider-2", 2.0),
    ("v3-all-k-3.5",   3.5),
    ("v3-all-wider-3", 3.0),
    ("v3-all-wider-4", 4.0),
    ("v3-all-k-5.0",   5.0),
    ("v3-all-k-6.0",   6.0),
]

FRICTION = [("5 bps", 0.0005), ("10 bps stress", 0.0010)]
OUT = REPO_ROOT / "reports" / "ATR_OPTIMUM.html"


def reset_classes():
    import importlib
    import strategy.signals_multifactor_v3 as v3
    importlib.reload(v3)
    STRATEGIES["v3-all"] = v3.V3All
    STRATEGIES["v3-all-k-2.5"] = v3.V3AllK_2_5
    STRATEGIES["v3-all-wider-2"] = v3.V3AllWider2
    STRATEGIES["v3-all-k-3.5"] = v3.V3AllK_3_5
    STRATEGIES["v3-all-wider-3"] = v3.V3AllWider3
    STRATEGIES["v3-all-wider-4"] = v3.V3AllWider4
    STRATEGIES["v3-all-k-5.0"] = v3.V3AllK_5_0
    STRATEGIES["v3-all-k-6.0"] = v3.V3AllK_6_0


def make_params():
    return dataclasses.replace(StrategyParams.from_yaml(),
                               risk_per_trade_pct=RISK_PCT, leverage=LEVERAGE)


def run(strategy, start, end, commission):
    reset_classes()
    try:
        return run_backtest(strategy_name=strategy, symbol=SYMBOL, timeframe=TIMEFRAME,
                            start=start, end=end, cash=CASH, leverage=LEVERAGE,
                            quiet=True, params_override=make_params(),
                            commission=commission, return_trades=True)
    except Exception as e:
        return {"strategy": strategy, "error": str(e),
                "after_funding_pct": float("nan"), "max_drawdown_pct": float("nan"),
                "win_rate_pct": float("nan"), "trades": 0}


def fmt(v): return "—" if v is None or pd.isna(v) else f"{v:+.2f}%"
def cls(v): return "" if v is None or pd.isna(v) else ("ok" if v >= 0 else "bad")


def main():
    print(f"ATR-multiplier sweep · {len(VARIANTS)} variants × {len(FRICTION)} friction × {len(WINDOWS)} windows")
    t0 = time.time()
    grids = {}
    for fric_label, commission in FRICTION:
        print(f"\n  ▶ {fric_label}")
        g = {}
        for strategy, k in VARIANTS:
            ret_per_window, dd_per_window = [], []
            wins = 0
            kill_hits = 0
            n_trades_total = 0
            n_wins_total = 0
            for win_label, start, end in WINDOWS:
                r = run(strategy, start, end, commission)
                g[(strategy, win_label)] = r
                v = r.get("after_funding_pct")
                dd = r.get("max_drawdown_pct")
                wr = r.get("win_rate_pct")
                nt = r.get("trades", 0)
                ret_per_window.append(v)
                dd_per_window.append(dd)
                if v is not None and not pd.isna(v) and v > 0:
                    wins += 1
                if dd is not None and not pd.isna(dd) and dd <= -18:
                    kill_hits += 1
                if nt and wr is not None and not pd.isna(wr):
                    n_trades_total += nt
                    n_wins_total += int(nt * wr / 100)
            cumulative = sum(v for v in ret_per_window if v is not None and not pd.isna(v))
            mean_dd = sum(d for d in dd_per_window if d is not None and not pd.isna(d)) / max(len([d for d in dd_per_window if d is not None and not pd.isna(d)]), 1)
            agg_win_rate = (n_wins_total / max(n_trades_total, 1) * 100) if n_trades_total else 0
            g[(strategy, "__summary__")] = {
                "k": k, "wins": wins, "cumulative": cumulative,
                "kill_hits": kill_hits, "n_trades": n_trades_total,
                "agg_win_rate": agg_win_rate, "mean_dd": mean_dd,
            }
            print(f"    k={k:>3.1f}  cum={cumulative:>+7.2f}%  wins={wins}/12  "
                  f"kill={kill_hits}/12  trades={n_trades_total}  wr={agg_win_rate:.1f}%")
        grids[fric_label] = g
    print(f"\nTotal: {time.time()-t0:.1f}s")

    # Render
    sections = []
    for fric_label, _ in FRICTION:
        g = grids[fric_label]
        rows = []
        for strategy, k in VARIANTS:
            s = g.get((strategy, "__summary__"), {})
            rows.append(
                f"<tr><td><strong>{html.escape(strategy)}</strong></td>"
                f"<td>{s.get('k', 0):.1f}</td>"
                f"<td>{s.get('n_trades', 0)}</td>"
                f"<td>{s.get('agg_win_rate', 0):.1f}%</td>"
                f"<td>{s.get('mean_dd', 0):.1f}%</td>"
                f"<td class='{cls(s.get('cumulative', 0))}'><strong>{fmt(s.get('cumulative', 0))}</strong></td>"
                f"<td>{s.get('wins', 0)}/12</td>"
                f"<td>{s.get('kill_hits', 0)}/12</td></tr>"
            )
        sections.append(f"""
<h2>{html.escape(fric_label)}</h2>
<table>
  <thead><tr><th>Variant</th><th>k</th><th># Trades</th><th>Win rate</th>
    <th>Mean DD</th><th>Cumulative</th><th>Win windows</th><th>Kill-switch hits</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
""")

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>ATR multiplier optimum sweep</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--ok:#3fb950;--bad:#f85149}}
*{{box-sizing:border-box}}body{{margin:0;padding:24px;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,sans-serif;max-width:1300px;margin:auto}}
h1{{font-size:22px;margin:0 0 6px 0}}h2{{font-size:16px;margin:30px 0 10px 0}}
.meta{{color:var(--muted);font-size:12.5px;padding:12px;background:var(--panel);border:1px solid var(--border);border-radius:8px;margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;font-size:12.5px}}
th,td{{padding:8px 12px;border-bottom:1px solid var(--border);text-align:left}}
th{{background:#0a0f17;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}
</style></head><body>
<h1>ATR-multiplier optimum search · v3-all variants</h1>
<div class="meta">8 stop multipliers tested across 12 OOS windows (2020-2026), at 2 friction levels.
TP held at 2× SL throughout (constant 2:1 R:R). risk_pct={RISK_PCT}% (~7.5% margin).</div>
{''.join(sections)}
<p style="color:var(--muted);font-size:11px;margin-top:30px;text-align:center">
Generated {generated} · <code>tools/build_atr_optimum.py</code></p>
</body></html>""", encoding="utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
