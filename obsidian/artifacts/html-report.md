---
tags: [artifact, report]
---

# HTML comparison report — `reports/HYBRID_VS_ALL.html`

Comprehensive backtest report comparing
[[HYBRID-short-strategy|cnh-hybrid-short-v1]] against
[[strategies/multifactor-v1]] and [[strategies/donchian-v3]] on shared
ground (2020-2026, 13 bps friction, daily-P&L basis).

Built by `../../tools/build_hybrid_comparison_report.py`.

## Contents (9 sections)

1. TL;DR — verdict + hold-time caveat
2. Per-leg headlines — trades, WR, cum, Sharpe, max DD, mean per trade
3. Equity curves — inline SVG, log-scale, all 5 series
4. Per-year cum return matrix
5. Daily-P&L correlation matrix
6. HYBRID deep dive — pattern attribution + hold-time distribution
7. Realistic deploy matrix (capital × risk grid)
8. Files index with hyperlinks to all artifacts
9. Related reports cross-reference

## Style convention

Matches existing project reports (PATH2_RESULTS, V1_DONCHIAN_RESULTS):
pure HTML+CSS with calm callout cards, color-coded values. No JavaScript,
no external deps. Inline SVG for the equity-curve chart.

## How to regenerate

```bash
uv run python tools/build_hybrid_comparison_report.py
```

Picks up the latest data files automatically (`reports/full_history_*` for
v1+Donchian trades; replays HYBRID via the live evaluator).

## See also

- `../../reports/HYBRID_VS_ALL.html` — the file
- [[phases/phase-4-portfolio-sim]] — the data behind §1-§5
- [[artifacts/realistic-deploy-matrix]] — the data behind §7
