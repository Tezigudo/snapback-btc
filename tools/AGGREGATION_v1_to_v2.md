# Aggregation v1 → v2 — reader's guide

Methodology debt #1 fix: how to read old vs new `reports/*.json` produced
by the post-fractional runners.

## The two families

Every runner now emits an `aggregation_method` tag.  Two families exist
and **must not be cross-compared**.

| Tag                                       | Family               | Per-window return = | Multi-window aggregate |
|-------------------------------------------|----------------------|----------------------|------------------------|
| `v2_equity_curve`                         | 5-OOS / fixed-window | `stats["Return [%]"]` | `prod(1 + return_pct/100) - 1` |
| `v2_walkforward`                          | Walk-forward (rolling train/test) | `prod(1 + ReturnPct_OOS_filtered/100) - 1` per quarter | same |
| `v2_equity_curve_funding_adjusted`        | Soft-drift (funding-net) | net-of-funding `PnL/cash + funding` | same |

The walk-forward family legitimately uses stitched `prod(1+ReturnPct)`
per quarter because the underlying `backtesting.py` headline spans
train+OOS — the runner cannot read `stats["Return [%]"]` for OOS-only
attribution.  Tag it `v2_walkforward` so it never gets cross-compared
with 5-OOS numbers.

## What `compounded_pct` means in each emit

- `compounded_pct` (canonical v2) — `prod(1 + per_window_return_pct/100) - 1`,
  where per-window return is the family's canonical headline.
- `legacy_compounded_pct` (legacy v1) — `prod(1 + per_trade_pnl_pct/100) - 1`
  over stitched per-trade `ReturnPct`.  Sizing-blind.  Kept for diff only.
- `legacy_delta_pp` — `compounded_pct - legacy_compounded_pct`.

If the runner already used `stats["Return [%]"]` (most siblings), the two
numbers match to rounding.  The drifters (`rv_band`, `kc_squeeze`,
`turn_of_candle_15m`) historically only had the legacy form — their `delta_pp`
can be large.

## PSR: which one to trust

Two PSR series can appear:

1. `legacy_psr_stitched` — `compute_psr` on stitched per-trade ReturnPct
   union.  **N-INFLATED**: treats trades from disjoint windows as if drawn
   from one process.  Carries a `deprecation` flag.  Use only for diff.
2. `canonical.psr_walkforward` — `compute_psr` on the n-WINDOW return
   series (e.g. n=5 for 5-OOS).  Defeats N-inflation.  **This is the
   primary verdict input.**
3. `canonical.psr_per_window[]` — `compute_psr` on equity-impact returns
   (PnL / equity-at-entry) within a single contiguous window.  Per-window
   evidence.  Equity-impact, not ReturnPct — sizing-aware.

Small sample warning: `psr_walkforward` with n=5 has wider error bars
than the historical stitched PSR with n=100+.  A small drop is small-sample
property of the new metric, NOT evidence the strategy degraded.

## PRICE_SCALE is orthogonal

PRICE_SCALE only fixes integer-unit truncation under the
`backtesting.py` harness.  Neither the v2 compounded nor the v2 PSR
changes between PRICE_SCALE=1 and PRICE_SCALE=1000 (BTC at $50 vs $50k)
given the same per-window dicts.  This is codified in
`tests/test_aggregate.py::test_price_scale_invariance`.

## Phase 2 baseline gate

`tools.aggregate.phase2_gate(v1_locked, v2_result, deployed=True)`
returns `"HALT_AND_SURFACE"` if any of:
- `v2_result["psr_walkforward"]["psr_vs_hurdle"] < psr_floor` (default 0.90)
- `v2_result["windows_positive_pct"] / 100 < wf_pos_floor` (default 0.70)

If it trips on the deployed `multifactor_v1 + 4H gate`, the operator MUST
NOT flip any runner defaults or touch the live bot.  Per CLAUDE.md
"BEFORE DEPLOY/PUSH", surface to the user with the small-sample framing.

## Reading an old `reports/*.json`

- Has `compounded_pct` only, no `aggregation_method` → pre-fix v1.
  Treat `compounded_pct` as `prod(1+ReturnPct)` (sizing-blind).
- Has `canonical` block AND `psr` block → post-fix.  Use `canonical.*`
  for verdicts; keep `psr` for diff.
- `aggregation_method` is the firewall — if missing, treat as v1.

## Coverage status

Patched to canonical v2 dual-emit:
- `_postfrac_adaptrend_v1_rv_band.py` (DRIFTER)
- `_postfrac_kc_squeeze.py` (DRIFTER)
- `_postfrac_turn_of_candle_15m.py` (DRIFTER)
- `_postfrac_adaptrend_v1_funding_skip.py` (soft-DRIFTER, funding-adjusted)
- `_postfrac_adaptrend_v1.py` (Phase 2 baseline)
- `_postfrac_mf_baseline.py` (Phase 2 baseline)
- `_postfrac_mf_4h_btc_run.py` (Phase 2 baseline — DEPLOYED)
- `_postfrac_walkforward_adaptrend_v1_volsize.py` (WF family example)

Deferred (numerically identical dual-emit cosmetic — adopt
`build_canonical_block()` incrementally):
- 30+ other `_postfrac_*` and `_adaptrend_*` runners that already use
  `stats["Return [%]"]`. Their v2 numbers will match v1 to rounding.

Deferred (genuinely hard, tied to debt #2):
- `_postfrac_wf_mf_4h_btc_sol_portfolio.py` — requires true weighted-portfolio
  equity-curve rewrite. Separate workstream.
