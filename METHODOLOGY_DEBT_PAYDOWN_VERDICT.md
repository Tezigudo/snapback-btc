# Methodology Debt Paydown — Consolidated Verdict

**Date**: 2026-06-03
**Scope**: Three coupled methodology defects identified via 3-lens verify panels (correctness / completeness / alternative-explanation). Decision rule: 2+ panel votes of `wrong` or `partial` → not resolved.

---

## Headline

Three debts attempted, **zero fully resolved**. Debt #1 and #2 land at PARTIAL — code and tests landed, locked-baseline reproductions held within rounding, but verify panels surfaced load-bearing structural holes (warm-prefix equity-at-entry bug, WF-runner equity-curve degeneracy, N-deflation gate vacuity, parallel canonical modules, 28+ sibling runners still emitting stitched PSR). Debt #3 lands UNRESOLVED — best candidate (variant A, threshold 0.015) cleared PSR 0.940 / +29.69pp lift but failed WF (35.71% < 70%) and worst-window DD (-32.79% << -15%), AND a confirmed backtest↔live formula gap (~3.3x stricter on live longs, ~6.4x on shorts) means every backtest PSR is non-actionable until the formula is reconciled. **No historical verdict changes**. Live BTC-only mainnet bot (multifactor-v1 + 4H gate, deployed 2026-06-02 12:07 UTC) is unaffected — no defaults flipped, no risk.py touched, no live config changed.

---

## Per-debt verdicts

### Debt #1 — Runner return-aggregation drift → **PARTIAL**

**Verify panel**: correctness=partial, completeness=partial, alternative-explanation=partial.

**What was delivered**:
- `tools/aggregate.py` canonical aggregator (window_return_pct, equity_impact_returns, aggregate_windows, legacy_stitched_psr) + `AGGREGATION_VERSION="v2_equity_curve"`
- 10 runners migrated to dual-emit (canonical + legacy fields)
- 14 deterministic tests in `tests/test_aggregate.py`, all passing
- `backcompat_baselines.json` Phase 2 re-baseline table
- `tools/AGGREGATION_v1_to_v2.md` reader's guide
- Phase 2 reruns: mf_baseline 50.48 → v2 50.54 (+0.06pp); mf_4h_btc 77.93 → v2 77.95 (+0.02pp); adaptrend_v1 45.52 → v2 45.52 (+0.002pp). Phase2 gate: PROCEED on DEPLOYED multifactor-v1+4H (v2 psr_walkforward 0.9966 ≥ 0.90, windows_positive 5/5 ≥ 70%).

**Load-bearing holes flagged by verify**:
1. **Warm-prefix equity-at-entry bug** — `equity_impact_returns` filters by window_start *before* cum-summing prior PnL, so first OOS trade always sees `equity_at_entry = cash`. Per-window PSR is systematically wrong for any warm-prefix runner. The test suite *codifies* the bug as spec.
2. **Phase 2 rerun selection bias** — three runners re-run (mf_baseline, mf_4h_btc, adaptrend_v1) were never drifters; they already used `stats['Return [%]']`. The plan's root-cause section identifies them as such. The three *actual* drifters (rv_band, kc_squeeze, turn_of_candle_15m) had reruns explicitly DEFERRED. Empirical evidence that the fix actually moves the drifters' numbers does not exist yet.
3. **Phase 2 gate is N-deflated, not corrected** — n=5 walk-forward PSR floor 0.90 is trivially clearable by any 5/5-positive strategy. The fix swapped N-inflation (stitched per-trade) for N-deflation (5-bucket window series) without testing intermediate compromises.
4. **Phase 2 gate input semantics wrong** — `phase2_gate` receives `windows_positive_pct` from the 5-OOS family as a proxy for the WF positive-quarter rate. These are different distributions; the DEPLOYED-strategy PROCEED verdict was made on the wrong metric.
5. **28+ sibling runners still drift on PSR axis** — only the compounded-return axis was cosmetic in those runners; their PSR is still computed on stitched per-trade ReturnPct, which aggregate.py's own docstring declares N-inflated and sizing-blind.
6. **Phantom legacy_compounded_pct** — `build_canonical_block` defines `legacy_compounded_pct` as prod over stitched per-trade ReturnPct, but no historical report ever computed it that way. The `legacy_delta_pp = +39pp` figures in backcompat_baselines.json compare v2 against a straw-man v1 that never existed.
7. **Warm-prefix runners use undocumented third aggregation method** inline (not in ALLOWED_METHODS, not in AGGREGATION_v1_to_v2.md, no test).
8. **No runner-level smoke integration tests** — tests target aggregate.py functions directly; a runner that drops dual-emit would not be caught.
9. **funding_skip canonical emit is hollow** — passes empty `pnl_pct: []` to build_canonical_block; legacy_compounded_pct is 0.0; only return_pct survives.
10. **11-OOS extended baseline drifted +8.5pp on AdaptiveTrend V1** without forensic reconciliation; attributed to "different harness" but unverified.

**Historical verdict changes**: NONE. Within-runner deltas use identical calc on both arms; SHELF verdicts (rv_band, adx-dual-regime, divergence) stand on their internal-integrity verify lens. Locked baselines reproduce v1 within rounding.

**Status**: Methodology debt #1 is **NOT closed**. Code is patched and observability dataset is in place, but the drifters were not re-run, the warm-prefix path has a bug encoded as test spec, and the Phase 2 gate metric does not measure what the plan promised to gate on.

---

### Debt #2 — Portfolio PSR (stitched-trade union) → **PARTIAL**

**Verify panel**: correctness=partial, completeness=partial, alternative-explanation=partial.

**What was delivered**:
- `tools/portfolio_psr.py` — sum-then-diff weighted-equity-curve aggregator (build_portfolio_equity_curve, equity_to_period_returns, aggregate_portfolio_psr)
- Both H1 (`_postfrac_mf_4h_btc_sol_portfolio.py`) and WF (`_postfrac_wf_mf_4h_btc_sol_portfolio.py`) runners migrated
- 4 unit tests, all passing (including load-bearing perfect-correlation regression guard)
- Headline rerun: port_5050_noveto_v2 old proxy 0.9717 → new equity-curve 0.9606 (delta -0.0111); psr_n_periods=1056 ≈ 6 H1 windows × ~180 daily obs; psr_trade_pool_proxy preserved under new key
- WF gate verdict-invariance confirmed: 13/22 (59.09%) < 70% gate stands

**Load-bearing holes flagged by verify**:
1. **WF runner is structurally NOT a daily-equity-curve PSR** — `_postfrac_wf_mf_4h_btc_sol_portfolio.py` builds equity_series from trade-exit timestamps only, not from continuous bar-by-bar equity. WF psr_n_periods=336 across 22 quarters (~15 obs/quarter vs expected ~90 trading days/quarter — a 6x undersample). The "mirror identical wiring" claim is satisfied syntactically but not semantically.
2. **Direction reversal contradicts stated mechanism** — only port_5050_noveto_v2 moves the predicted direction (delta -0.011). port_5050_veto_v2 (+0.005) and port_6040_veto_v2 (+0.006) move the OPPOSITE way. If N-inflation was the bug, all arms should move down. The headline "delta documents the n-inflation magnitude" framing is selective.
3. **Interpretation flip on port_6040_veto_v2 is a silent deploy-decision risk** — old psr_trade_pool_proxy=0.9482 was "insufficient_evidence"; new psr_equity_curve=0.9545 flips to "evidence_of_edge". The fix that was sold as verdict-invariant actually moves a gate-cross for one arm; no regression test catches this.
4. **No Lo (2002) autocorrelation correction** — compute_psr's varterm covers only skew/kurtosis. With max_hold_bars=1344 (14 days), each trade contributes to ~14 daily-return observations → strong positive serial correlation → PSR mildly upward-biased on real data.
5. **Test C (frequency invariance) is theater** — tolerance < 0.2 covers most of PSR (0,1); under IID, z = SR·sqrt(N−1) is approximately invariant in N. Test passes trivially. Tests B uses degenerate identical-leg setup that cannot fail by construction.
6. **Skew correction is the likely confound, not N-inflation** — with point_sharpe_period ~0.05 and PSR ~0.96 on n=1056, varterm < 1 due to positive skew amplifying PSR. The "fix" trades one statistical artifact (N-inflation) for another (residual skew sensitivity at daily aggregation).
7. **Parallel module duplicates aggregate.py contract** — repo now has TWO competing canonical PSR-aggregation modules (aggregate.py and portfolio_psr.py) with different APIs, no shared invariants. Future portfolio claims will pick whichever is more familiar. Will require methodology-debt-#3 to deduplicate.
8. **BTC-solo per-leg PSR drifted from 0.978 to 0.942** — the plan promised "no per-leg regression risk" but the v2 JSON for port_5050_noveto_v2 shows BTC_solo psr_vs_hurdle = 0.9419 vs historically cited 0.978. Unflagged audit-invariant violation.
9. **10+ sibling runners still feed stitched per-trade arrays into compute_psr** — taker_flow, run_mf_deepening, adaptrend_v2 family, donchian_variants_sweep, _postfrac_adaptrend_v1_rv_band, _postfrac_wf_adaptrend_v1_rv_band, _postfrac_mf_baseline. Systemic methodology debt is unresolved.
10. **JSON drops portfolio_equity_series at serialization** — reviewers cannot recompute PSR at alternative resample periods from the published artifact.

**Historical verdict changes**: NONE on the BTC+SOL portfolio (WF 13/22 < 70% gate is the binding constraint, unchanged). However, the verify panel surfaces that port_6040_veto_v2's interpretation flipped — if any FUTURE portfolio claim is made on the 6040 weighting, the new metric would label it "evidence_of_edge" where the old labeled it "insufficient_evidence". Not a verdict change on the deployed strategy, but a methodology-watch item.

**Status**: Methodology debt #2 is **NOT closed**. The primary H1 runner's equity-curve PSR is structurally sound; the WF runner's is not. Parallel canonical modules and sibling-runner systemic drift mean a future portfolio claim still cannot be evaluated on a uniformly defensible metric.

---

### Debt #3 — Donchian slope variants → **UNRESOLVED**

**Verify panel**: correctness=partial, completeness=partial, alternative-explanation=**wrong**.

**Best variant**: A_lower_threshold_0.015 — PSR 0.940, compounded +80.14% (+29.69pp lift over baseline), 4/5 wins, 81 trades. **Recommendation: SHELVE** (do NOT add to validation queue).

**Why no variant promotes**:
- All 5 variants AND the baseline fail the WF positive-quarter gate (23-43% vs 70% floor)
- All 5 variants AND the baseline breach the -15% worst-window kill-switch (DD -32.79% to -39.26%)
- A's PSR 0.940 carries `interpretation: insufficient_evidence` in the JSON (n=81 stitched trades, min_trl=91)
- A's +2.38pp WF lift over baseline is 1-of-14 quarters — well below binomial detection threshold

**Load-bearing kill-condition (alternative-explanation = wrong)**:
- **Backtest↔live formula gap is asymmetric and worse than payload's 3.3x claim**: empirical bt/live ratio median=3.18, p10=2.65, p90=3.81. Long gate: backtest fires 47% of bars vs live 22% (~2.1x stricter). Short gate: backtest 24% vs live 3.8% (~6.4x stricter). At threshold 0.015 in backtest, live would map to ~0.0045 — essentially gate-off → A behaves like D (pure Turtle) live. The PSR ranking A > D is an artifact of the backtest-only formula.
- **D actually outperforms A on directional regimes** (D 2022_H1 +67.5% vs A +41.7%; D 2023_H1 +32.7% vs A +17.4%). The 4/5-vs-3/5 win-count tiebreak is decided by 0.3pp swing in 2024_H2 — noise.
- **E (EMA200 binary filter) is a false literature comparison**: E trade count 102 ≈ D 104, per-window returns within ±1.5pp in 4/5 windows. EMA200 filter on 4H entry TF rejects almost no trades — E is essentially D relabeled.
- **Locked baseline itself blew every gate** in this harness (PSR 0.882, DD -34.94%, WF 33.33%). The interpretation_rules.none_pass clause ("locked baseline stays, debt #3 documented as 'defensible'") is incoherent — the locked baseline numbers refute "defensible".

**Methodology debt #2 recurrence**: Sweep runner pools per-trade ReturnPct across 5 non-contiguous OOS windows (n=74-104) into a single compute_psr call — the SAME stitched-stream defect that just got "fixed" in debt #2. PSR_min=0.90 gate would mis-promote a future borderline variant on this metric.

**Historical verdict changes**: NONE. Locked donchian-v3 baseline numbers in this harness diverge slightly from prior shelf references (50.45% vs unknown locked figure), but baseline_reference fields in the gate config are null — no cross-validation occurred. Locked donchian-v3 remains DRY-deployed on the droplet alongside multifactor-v1; the variants sweep does not change that posture.

**Status**: Methodology debt #3 is **NOT closed**. The slope-gate axis is exhausted on BTC: pure Turtle (D), magnitude threshold (baseline/A), faster EMA (B), shorter window (C), and binary EMA200 (E) all sit in the same WF-failure / kill-switch-breach cluster. Per plan kill_conditions item 2 ("if 3+ of 5 variants fail all gates: gate-axis is exhausted"), escalate: the slope gate is not the right lever for this strategy. Debt #3 requires either strategy-level rework (different gate axis: vol regime, ADX-replacement on a different feature, MTF) or acceptance that donchian-v3 stays DRY-only.

**Action required before any variant work resumes**: Reconcile `strategy/live_donchian_v3.py:55-66` polyfit-fraction formula with `strategy/regime_classifier.py:66-74` ewm-diff-percent formula. Until backtest==live or both are dual-calibrated against a held-out reference threshold, every variant PSR is misleading for deploy.

---

## Cross-debt observations

1. **Same defect recurs across debts**: stitched per-trade PSR is the bug debt #2 set out to fix. Debt #1's `build_canonical_block` adds `legacy_psr_stitched` as observability but its `legacy_compounded_pct` mathematically uses the same stitched-stream construction. Debt #3's sweep runner still uses pooled per-trade compute_psr. The fix corpus has not converged on a single canonical PSR pipeline.
2. **Parallel canonical modules**: `tools/aggregate.py` (debt #1) and `tools/portfolio_psr.py` (debt #2) duplicate intent without cross-referencing. Sibling runners that should use one of these still emit ad-hoc inline math.
3. **Phase 2 / Phase 3 / Variant gates all rely on small-N walk-forward positive-rate metrics** (n=5 windows or n=9-14 sufficient quarters) where the 70% threshold is statistically equivalent to a coin-flip test at α=0.09. All three debts ship gates that look stringent but admit borderline strategies through.
4. **Live↔backtest formula drift (debt #3) is the kind of defect debts #1 and #2 cannot detect** — both assume the engine output is ground truth. A separate methodology-debt #4 may be needed: "live execution path ↔ backtest path numerical equivalence".

---

## Live deploy posture (unchanged)

- multifactor-v1 + 4H gate: LIVE on mainnet (since 2026-06-02 12:07 UTC, commit `b83e20f`). v2 reruns confirm +77.95% reproduces the locked +77.93% within rounding; PSR/WF gates not threatened by debt #1 paydown.
- donchian + cnh_short: DRY on droplet. Debt #3 outcome does not change this; live formula gap (kill_condition_unresolved) is now formally documented and blocks any promotion attempt.
- BTC+SOL portfolio: SHELF stands. Debt #2 fix does not move the binding WF constraint.
- risk.py, leverage ceiling 20x, kill switch -15%, HALT poll, mainnet lock — all untouched per CLAUDE.md hard rules.

---

## Files written by this synthesis

- `/Users/god/Desktop/work/snapback-btc/METHODOLOGY_DEBT_PAYDOWN_VERDICT.md` (this file)
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/methodology_debt_1_paydown.md`
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/methodology_debt_2_paydown.md`
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/methodology_debt_3_paydown.md`
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/MEMORY.md` (one-line indexes appended)
