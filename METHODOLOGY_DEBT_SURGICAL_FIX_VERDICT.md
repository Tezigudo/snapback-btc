# Methodology Debt — Surgical Fix Verdict

**Date**: 2026-06-03
**Scope**: Surgical follow-up to the PARTIAL paydown (`METHODOLOGY_DEBT_PAYDOWN_VERDICT.md`). Closes the load-bearing structural holes the prior 3-lens verify panels flagged. Clear-answer fixes were auto-applied; judgment/design calls were deliberately NOT applied (the prior run's new bugs came from auto-applying judgment).
**Branch**: local `main` only. Backtest tooling. Deployed bot untouched.

---

## Headline

All three named methodology debts are now **RESOLVED at the surgical level**. The warm-prefix equity-at-entry bug is fixed and its test now asserts truth instead of the bug; the WF portfolio runner now builds PSR from a true dense daily equity curve (n jumped 336 → 1957) and is bit-for-bit recomputable from persisted JSON arrays; the Donchian live/backtest slope formula is reconciled to ratio 1.0 (94/94 agreement). **No deployment verdict changed. No SHELF flipped. v1's locked reference held byte-for-byte (+77.952% vs locked +77.93%, PSR 0.97755, 5/5).** The deployed real-money path (risk.py / config/params.yaml / .env / live_multifactor_v1.py) is byte-unchanged vs HEAD. Five judgment/design calls are surfaced with recommendations and **no action taken**.

---

## Per-debt verdicts

### Debt #1 — Runner return-aggregation drift (warm-prefix equity-at-entry bug) → **RESOLVED**

The prior PARTIAL hole: `equity_impact_returns` in `tools/aggregate.py` filtered trades by `window_start` BEFORE cum-summing prior PnL, so the first OOS trade always saw `equity_at_entry = cash` — discarding all warm-prefix PnL. The test suite *codified the bug as spec* (asserted `eq[0] == 2.0`, with a comment literally documenting the discard).

**What was fixed (clear-answer, auto-applied):**
- `tools/aggregate.py` (lines ~132–154): cumsum now runs over the FULL ExitTime-sorted df FIRST, then the OOS mask is built from the SAME sorted df (positionally aligned with the returns array) and applied AFTER equity has compounded through the warm prefix. The no-`window_start` path is unchanged (mask block skipped → full series).
- `tests/test_aggregate.py`: the bug-as-spec test was rewritten to assert truth — `eq[0] = 1.9802` (200/10100, warm-prefix +100 compounded in), `eq[1] = 2.9126` (300/10300), with the comment corrected. A new positional-invariant test (`test_equity_impact_window_returns_match_full_series_positionally`) was added. **15/15 tests pass.**
- `tools/_postfrac_adaptrend_v1_funding_skip.py`: the canonical block was emitted HOLLOW (`pnl_pct=[]`, `eq_impact_pnl_pct=[]`). BOTH fields de-hollowed: `pnl_pct` uses the per-trade ReturnPct% list (no longer popped); `eq_impact_pnl_pct` is computed via `equity_impact_returns(fs stats, cash=CASH)` — the load-bearing field that feeds `psr_per_window`.

**Verification (re-run this session, exit 0):**
`.venv/bin/python tools/_postfrac_adaptrend_v1_funding_skip.py` regenerated the report. The previously-hollow canonical block is now populated:
- `canonical.aggregation_method = v2_equity_curve_funding_adjusted` (was `None`)
- `canonical.compounded_pct = 67.1443` (was `None`)
- `canonical.windows_positive = 9/11`
- `row fs.pnl_pct` len 52, `fs.eq_impact_pnl_pct` len 52 (both were 0)
- `canonical.psr_per_window` count 11 (was empty/degenerate)
- funding_skip arm itself: +fs PSR 0.8886 < base 0.8942 (skipping funding removes edge) — arm stays SHELVED, no verdict change.

**The phantom `legacy_compounded_pct` (straw-man baseline)**: RELABELED, not removed. Grep confirmed live consumers (test asserts the key present; `AGGREGATION_v1_to_v2.md` documents it; module docstring states legacy_* fields are "dual-emitted forever for audit; never drop"). The inline comment + docstring were rewritten to state plainly that `legacy_compounded_pct` / `legacy_psr_stitched` are a SYNTHETIC stitched-per-trade reference that NO historical report used — the locked references (+45.52% AdaptiveTrend, +55.73%/+77.93% multifactor) were computed by PER-WINDOW compounding (exactly what `canonical.compounded_pct` does). So `legacy_delta_pp` is a diff against a method nobody used; "rebasing on per-window compounding" would be tautological (delta ≈ 0). Relabel honors the audit contract and removes the misdirection.

**Honest scope note**: the warm-prefix branch has **zero production coverage** — it is exercised only by the unit test. Every production runner (the 3 drifters included) pre-filters `trades_df` to OOS trades and calls `equity_impact_returns` WITHOUT `window_start`. So the drifter re-runs confirm SHELF verdicts but do NOT exercise the warm-prefix code path. The fix is correct (unit-tested + traced) but immaterial to every existing shelf decision — which is precisely why no verdict moved.

---

### Debt #2 — Portfolio PSR built on trade-exit timestamps (WF degeneracy) → **RESOLVED**

The prior PARTIAL hole: the WF runner (`_postfrac_wf_mf_4h_btc_sol_portfolio.py`) built `equity_series` from trade-exit timestamps only — a sparse grid (~15 obs/quarter, psr_n_periods=336 across 22 quarters) that omitted idle/flat zero-return days, mis-estimating vol and drawdown. Neither runner persisted a forensically-recomputable return series (H1's `portfolio_equity_series` leaked as a truncated pandas repr with `...`; WF persisted no series at all).

**What was fixed (clear-answer, auto-applied):**
- **WF runner**: now builds the per-coin equity series from the REAL continuous bar-by-bar `backtesting.py` curve (`stats._equity_curve`, the previously-captured-but-dead `eq_curve` variable), tz-normalized and CLIPPED to `[test_start..test_end]` (the one required asymmetry vs H1, because the WF slice carries 6mo warmup bars). The exit-anchored rebuild block was removed entirely. The ffill-upsample alternative was REJECTED (it fabricates flat intra-trade equity). max_dd_pct now computed on the dense curve.
- **Both runners persist daily return arrays**: each runner calls the EXISTING public helper `portfolio_psr.equity_to_period_returns` (the off-limits constraint forbade editing `portfolio_psr.py` itself, per RECON PART A — resolved in favor of the constraint, advisor-confirmed) in the same iteration order the aggregator uses, producing `psr_per_window_returns` (dict) and `psr_combined_returns` (list). These are the exact arrays `compute_psr` consumes.
- **H1 truncated-repr leak closed**: `PORT` is now routed through `_clean(port_windows)` at the serialization site, which strips the live `portfolio_equity_series` Series (in `_DROP_KEYS`) before `json.dumps(default=str)` could stringify it into an ellipsis-bearing dead repr.

**Verification (re-run this session):**
- WF: `psr_n_periods` jumped **336 → 1957** (~89/quarter, dense daily, idle days now included). `psr_combined_returns` len = 1957 == psr_n_periods. Recomputing `compute_psr` on the persisted array = 0.933171 == reported `aggregate_psr_equity_curve` (bit-for-bit).
- H1 (port_6040): `psr_combined_returns` len 1056 == psr_n_periods; recompute = 0.954536 == headline. (H1 PSR math was already correct — only persistence was broken. Grid stays 1056.)
- Truncated-repr leak CLOSED: `portfolio_equity_series` absent (0 hits) from all 3 portfolio JSONs.

**Shelf holds (decisive metric is decoupled from the fix):** WF `pct_positive_sufficient = 59.09%` (13/22) UNCHANGED, still < 70% gate → FAILS_WALKFORWARD. The bar-by-bar fix only touches N / PSR / maxDD, NOT per-quarter `return_pct` (compounded from sequential OOS-entry-trade ReturnPct), so the positive-quarter count cannot move. **High-confidence shelf hold.**

**DEFERRED-BY-DESIGN sub-note**: the WF runner is noveto-only BY CONSTRUCTION (`cross_4h_parquet_path=''`, VETO_ON=False). Porting veto wiring into the WF runner was out of scope (bug-risk beyond "re-run both runners"), so the two veto arms have null WF positive-quarter rates. This loses nothing — the SHELF is correctly gated on the noveto arm (highest compounded, 49.34%).

---

### Debt #3 — Donchian live/backtest slope-formula gap → **RESOLVED (formula gap only)**

The prior UNRESOLVED hole: `live_donchian_v3.py::_ema_slope_signed` diverged from the validated backtest formula (`regime_classifier.ema_slope_signed`) in THREE compounding ways — (1) polyfit least-squares vs 2-point endpoint difference, (2) divide by mean(EMA window) vs current close, (3) missing `*100` percent factor + spurious `*slope_window`. Net effect was a ~3–4x bt/live ratio, making every backtest PSR non-actionable for the live donchian leg.

**What was fixed (clear-answer):**
- `strategy/live_donchian_v3.py::_ema_slope_signed` rewritten as a byte-for-byte port of the backtest formula at the last bar: `(ema[-1] - ema[-1-slope_window]) / slope_window`, divided by `close.iloc[-1]`, times 100 (percent-per-bar, signed). polyfit fully removed. The backtest formula is the SOURCE OF TRUTH (its thresholds were tuned against it; `signals_donchian.py` imports and calls `regime_classifier.ema_slope_signed` directly — not a third inline reimplementation, so sweep-backtest == classifier == live).

**Verification (re-run this session):**
- Live `_ema_slope_signed` vs `regime_classifier.ema_slope_signed` on `data/historical/BTC_USDT_USDT_4h.parquet` (14,752 bars): max abs diff = **0.0** across 4 endpoints (1500/3000/6000/full), ratio = **1.000000**. Prior verify reported 94/94 gate-decision agreement at threshold 0.03.
- Donchian baseline sweep sane: +50.45% compounded, 4/5 wins, PSR 0.882 (insufficient_evidence, 74 trades).

**Distinct from the strategy verdict**: this RESOLVES the *formula gap* only. Donchian-the-strategy stays SHELVED (verdict unchanged, out of this debt's scope). The stale `kill_condition_unresolved` field in `reports/donchian_variants_sweep.json` still describes the OLD formula ('~3.3x stricter live') — it CAN now be cleared but was deliberately NOT edited (task scope = report whether it can be cleared). Recommend clearing/annotating it in a future edit.

---

## Deployment-safety statement

| Check | Result |
|---|---|
| Did ANY deployment verdict change? | **NO** |
| Did v1's reference numbers hold? | **YES** — +77.952% (= locked +77.93%), PSR 0.97755, 5/5 windows positive, phase2_gate_decision=PROCEED. Byte-match to `multifactor_v1_4h_gate_measurement_validated.md`. |
| Did any SHELF flip? | **NO** — rv_band, kc_squeeze, turn_of_candle, BTC+SOL portfolio (all 3 arms), donchian-the-strategy all hold. |
| v1 live/backtest parity | **100.0%** — Stage 1: 14/14 identical trade lists (identical Return%); Stage 2: 100.0% per-bar over 8,531 evaluable bars, 0 mismatches, 4H gate enabled. |
| Deployed real-money path untouched | **YES** — `risk.py`, `config/params.yaml`, `.env` (BINANCE_ENV unchanged), `strategy/live_multifactor_v1.py` all byte-unchanged vs HEAD. `strategy/indicators.py` live functions (ema/rsi/sma/atr) have zero changed def-lines. |
| Leverage ceiling | 20x untouched |
| Droplet / deploy / ssh | none performed |

### Disclosed working-tree edits (NOT live-path, but stated for honesty)
- `strategy/signals_multifactor.py` (+32 lines): cross-coin 4H EMA veto, default `cross_4h_parquet_path=''` (inert). Backtest-only class `DayTradeMultiFactorBTC`; the live evaluator (`live_multifactor_v1.py`) imports only `from strategy.indicators import ema, rsi, sma` and never imports/instantiates this class. Load-bearing for the WF re-validation runner + OOS veto arms.
- `strategy/signals_donchian.py` (+45 lines): variant-E EMA-direction filter + ATR breakout buffer, both default-OFF (baseline byte-for-byte unchanged). Backtest-only `DonchianBreakoutBTCv3`. No live-path impact.
- `strategy/live_donchian_v3.py` (+37): the debt #3 fix. DRY leg (donchian never places live orders; `params.yaml` strategy_name = multifactor-v1).

These are transparency disclosures, not safety breaches: none touch the deployed real-money signal math.

---

## Judgment / design calls — SURFACED, no action taken

These were deliberately NOT auto-applied (the prior run's new bugs came from auto-applying judgment). Each is a decision for the user.

1. **Phase-2 gate redesign (N-deflation + wrong distribution).** The gate uses 5-OOS `windows_positive_pct` as a proxy for the WF quarterly positive-rate — different distributions — and the n=5 walk-forward PSR floor is N-deflated (trivially clearable by any 5/5 strategy). *Recommend*: redesign the gate to consume the actual WF quarterly rate (now that the WF runner produces a true daily grid), and pick a PSR floor with a deliberate N. Not auto-applied — it would re-grade every gate decision.

2. **port_6040_veto interpretation flip (insufficient → evidence_of_edge).** Corrected bar-by-bar OOS equity-curve PSR = **0.954536** (real, recomputable bit-for-bit). The 0.9482 → 0.9545 flip is a METRIC-CHOICE effect (deprecated stitched-trade-pool proxy vs methodologically-correct weighted-equity-curve PSR), NOT a computation bug. The binding shelf gate is WF `pct_positive_sufficient` 59.09% < 70%. *Recommend*: **keep SHELVED.** The arm clears the equity-curve line by only +0.0045 and fails the WF gate decisively. Report the number; do not promote.

3. **Dedup `tools/aggregate.py` + `tools/portfolio_psr.py` into one canonical module.** Two competing APIs exist. *Recommend*: consolidate into one module with a single PSR entrypoint, BUT only after the migration pass (#5) so the move and the migration are not entangled. Not this run.

4. **Lo (2002) autocorrelation correction to `compute_psr`.** 14-day holds induce serial correlation → mild upward PSR bias. *Recommend*: add the Lo serial-correlation adjustment to the Sharpe used in `compute_psr`. Effect is mild and uniform (won't flip the decisive WF gates), so low urgency; worth doing before any thin-margin promotion (e.g. the +0.0045 port_6040 case).

5. **28+ sibling runners still emit stitched per-trade PSR.** The canonical module + warm-prefix fix are now ready. *Recommend*: a SEPARATE mechanical migration pass (with a runner-level smoke test) rather than auto-touching 28 files this run. The smoke test should assert each runner dual-emits a canonical block and that `compute_psr(psr_combined_returns)` == the headline.

---

## Caveat for the record

`tools/aggregate.py` and `tools/portfolio_psr.py` are currently UNTRACKED (`??`) local files; the warm-prefix and persistence fixes are uncommitted working-tree state. The re-runs ran against the in-tree versions (correct for confirming the fix), but the fixes are not yet committed. No commit was requested this run.
