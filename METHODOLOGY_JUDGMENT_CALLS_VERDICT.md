# Methodology Judgment Calls — Final Synthesis Verdict

**Date:** 2026-06-03
**Base commit:** `10adaef` (`[fix] methodology-debt surgical fixes + PSR canonical tooling`)
**Scope:** 5 judgment-call executions on the PSR/gate methodology paydown, LOCAL `main` working tree only.
**Deployed strategy:** multifactor-v1 + 4H gate, real money $101.95 anchor — must stay byte-unchanged.

---

## Bottom line

| Question | Answer |
|---|---|
| Did ANY deployment verdict change? | **No.** Deployed multifactor-v1+4H returns PROCEED via both the new gate path and the backcompat proxy path — byte-identical. Every `verdict_changed=true` is on a NON-deployed candidate. |
| Did the v1 uncorrected reference hold? | **Yes — empirically reproduced.** `psr_vs_hurdle = 0.97755` (rounds to locked 0.978 / +77.93%), 121 trades, 5/5 windows positive, read fresh from `reports/postfrac_mf_4h_btc.json`. Lo is additive: `psr_lo_adjusted = 0.97755` (no-op, `lo_eta = 1.0`). |
| Did any SHELF flip? | **Yes — one, benign, non-deployed.** `_divergence_final_verdict.py` v2_loose: shelf → marginal (canonical PSR 0.523 crosses the 0.5 marginal floor that stitched 0.36 failed). Interpretation still `insufficient_evidence`; strategy NOT promoted; compounded/wins unchanged. This is a migration-*correct* metric fix, not a regression. |
| Did the gate redesign change the DEPLOYED decision? | **No.** The sole deployed caller (`tools/_postfrac_mf_4h_btc_run.py:241`) passes no WF series → backcompat proxy branch (`aggregate.py:421-427`) → PROCEED. Replicating the old committed gate on the same inputs also yields PROCEED. |
| Safe to commit? | **Yes** (with one caveat — see below). Live path byte-clean by blob hash, verify panel clean (no `wrong`), v1 held, 37 PSR tests pass. |

**Commit caveat:** committing the *code* is safe. But the redesigned `phase2_gate` must NOT be treated as an active live guardrail yet — it is on the backcompat path for the deployed strategy and its convenience carrier is field-name-broken (see residual #1). Wire a runner + fix the carrier before relying on it to protect the deployed strategy.

---

## Per-call status

### #1 GATE REDESIGN — **PARTIAL**
`phase2_gate` was genuinely rewritten (not renamed) to recompute both inputs from a true walk-forward quarterly series: `wf_pos_pct = mean(series > 0)` (replacing the optimistic 5-OOS positive-rate proxy) and `wf_psr = compute_psr(series)` at n≈25 (replacing the N-deflated n=5 read). Scope-decision-B adopted: floor checks moved out of the `if deployed:` short-circuit so they run for ANY strategy; `deployed` is now a severity tag (HALT_AND_SURFACE vs SHELF). Guard D: `n_q < 8 → insufficient_wf_evidence`. PSR floor kept at 0.90 (meaningful only now that n changed). Mutation-tested (dropping the PSR term fails `test_phase2_gate_psr_floor_breach`). 19/19 gate tests pass.

**Why PARTIAL (two confirmed residual defects):**
1. **Carrier field-name bug (functional, medium).** `aggregate.py:404` extracts `e["return_pct"]` from the `wf_result` carrier, but the real adaptrend WF JSON windows key on `net_return_pct` (verified: 0/20 windows have `return_pct`, 20/20 have `net_return_pct`). Feeding the real JSON via `wf_result=` resolves `series → None` → **silent** fallback to the optimistic 5-OOS proxy → PROCEED — the exact masked-failure the redesign claims to kill. A silent fallback to the wrong answer is worse than an exception. The documented adaptrend PROCEED→SHELF regrade only fires via the explicit `wf_quarterly_returns=net_return_pct` kwarg (verified: 11/20 = 55% < 70% → SHELF). The synthetic test only exercises `{'return_pct': r}`, so it never catches the mismatch.
2. **Teeth inert in production.** The only production caller passes neither kwarg → always backcompat. The adaptrend regrade is a hand/test computation, not pipeline output any runner emits. `any_deployed_decision_changed=false` is true partly because the new logic never runs on real data.

Verified-correct on the deployed strategy: the mf+4H WF JSON genuinely keys on `return_pct` (n=25, 20/25 = 80% positive, WF-PSR 0.9987), so both gate paths agree on PROCEED.

### #2 port_6040 KEEP-SHELVED — **DONE** (document-only)
Decision already made: corrected equity-curve PSR 0.9545 is real but fails the walk-forward 59% positive-rate gate (< 70%). Stays shelved. No code action required; documented only. Verified consistent with the portfolio shelf sample (`postfrac_wf_mf_4h_btc_sol_portfolio_port_5050_noveto_v2.json`, FAILS_WALKFORWARD, PSR 0.9332 recomputed bit-for-bit).

### #3 DEDUP (canonical core) — **DONE**
`tools/portfolio_psr.py` reduced to a pure re-export shim of `build_portfolio_equity_curve`, `equity_to_period_returns`, `aggregate_portfolio_psr` from `tools.aggregate` — verified at runtime as the SAME objects (`is`-identity) via `test_unified_psr_equivalence.py`. New `aggregate_psr(...)` dispatcher is additive sugar with literal delegation (no input-shape fusion); the two pre-existing public functions stay callable unchanged. Pre-merge golden fixture generated from the ACTUAL base-commit modules (extracted via `git show`), so the equivalence check is behavior-preserving, not tautological. `psr_vs_hurdle` byte-identical pre/post in both paths. 7 equivalence tests pass.

### #4 LO CORRECTION — **DONE**
Lo (2002) is strictly additive: `compute_psr` emits BOTH `psr_vs_hurdle` (uncorrected, locked-ref family, NEVER reassigned) AND `psr_lo_adjusted`. Mechanism: deflates the `psr_z` denominator via a Bartlett/Newey-West VIF; `psr_z_lo = psr_z/sqrt(VIF)`; written into NEW variables only. Gated: fires only when `contiguous AND n >= lo_min_n(=20)`. Empirically confirmed BOTH directions:
- **No-op** on the deployed stitched series (`lo_eta = 1.0`, `psr_lo_adjusted = 0.97755 == psr_vs_hurdle`) because the series has net non-positive autocorrelation and the VIF is floored at 1.0 (intentional one-sided deflation-only deviation, documented).
- **Genuine deflation** on positive-autocorr windows: `postfrac_mf_4h_btc.json` window[1] shows `lo_eta = 0.857`, `psr_lo_adjusted = 0.9153 < psr_vs_hurdle = 0.9456`. AR(1) φ=0.6 n=200 contiguous → 0.9386 deflates to 0.8091; contiguous=False → exact no-op.

7 Lo tests pass. Minor cosmetic: the two n<2 stub shapes omit sibling keys `sr_lo_adjusted`/`lo_eta` (schema asymmetry, not a correctness problem — no gate reads them).

### #5 MIGRATE (31 runners) — **PARTIAL**
31 runners migrated from stitched-per-trade headline PSR (N-inflated) to canonical `psr_walkforward` (window-level, `contiguous=False`), keeping `legacy_psr_stitched` for observability, with per-runner round-trip asserts. `all_match=true` across all 31; every persisted headline PSR matches independent `compute_psr` recompute bit-for-bit. Two `verdict_changed=true` cases, both correctly attributed:
- `adaptrend_oos_sweep` ETH/SOL coin_specific→transfers — NOT a shelf flip; attributed to the base-commit `10adaef` compute_psr rewrite (proven: unmigrated legacy call yields the identical value), migration-neutral.
- `_divergence_final_verdict` v2_loose shelf→marginal — genuine migration-driven correction (canonical 0.523 crosses the 0.5 marginal floor stitched 0.36 failed); benign (non-deployed, still insufficient_evidence, not promoted).

**Why PARTIAL (coverage gap, confirmed by direct read):** the working tree has 38 PSR-referencing runners; 8 are unmigrated and absent from the 31-runner manifest. Two of them still gate their verdict on the stitched PSR the migration was meant to retire:
- `tools/_postfrac_adaptrend_v1_rv_band.py` — `verdict()` at lines 343-367 reads `base["psr"]["psr_vs_hurdle"]` / `rv["psr"]["psr_vs_hurdle"]` (stitched); canonical block emitted at line 322 but unused. Emits a live `PROMOTE_CANDIDATE` report.
- `tools/_postfrac_kc_squeeze.py` — `psr_cleared` at line 302 gates on stitched `psr_vs_hurdle` vs 0.97; canonical at line 322 unused.

These are structurally identical to siblings that WERE migrated, so "scope" does not cleanly excuse them. **Mitigation (caps severity at medium):** both were explicitly DEFERRED in a prior `METHODOLOGY_DEBT_PAYDOWN_VERDICT.md`, and rv_band's FINAL disposition is correctly SHELVED via its migrated walk-forward sibling (`_postfrac_wf_adaptrend_v1_rv_band.py`, FAILS_WALKFORWARD 45% positive) — the unmigrated OOS SMOKE PROMOTE_CANDIDATE never escalated toward deploy. No live or promotion error.

---

## Verification evidence (independently reproduced this session)

- **HEAD** = base `10adaef` (0 commits past).
- **Live path byte-clean by blob hash:** `config/params.yaml`=`4b73237`, `strategy/live_multifactor_v1.py`=`a3f2dd5`, `risk.py`=`492e8e6`, `strategy/indicators.py`=`db08e0a`. None appear in `git diff HEAD --name-only`.
- **v1 reference** read fresh from `reports/postfrac_mf_4h_btc.json`: `psr_vs_hurdle 0.97755`, `psr_lo_adjusted 0.97755`, 121 trades, 5/5 windows [+20.01, +33.18, +0.67, +1.76, +8.68], all windows have trades > 0.
- **Gate carrier mismatch** reproduced: adaptrend WF JSON has 0/20 `return_pct`, 20/20 `net_return_pct`, 11/20 = 55% positive.
- **rv_band / kc_squeeze stitched-gated verdicts** read directly from source.
- **Safety scan:** `git diff HEAD` contains no `subprocess|ssh|scp|152.42|confirm_mainnet|BINANCE_ENV|place_order|create_order|verify_identity|data/HALT`.
- **Tests:** `tests/test_aggregate.py` + `tools/tests/test_portfolio_psr.py` + `tools/tests/test_lo_correction.py` + `tests/test_unified_psr_equivalence.py` = **37 passed**.

## Residual methodology debt
1. Gate `wf_result` carrier reads `return_pct`; real adaptrend WF JSON keys on `net_return_pct` → silent fallback to optimistic proxy. (medium, functional)
2. Gate teeth inert in production — no runner wired to pass the WF series; deployed strategy stays on backcompat. (medium)
3. 8 PSR-referencing runners unmigrated; 2 still stitched-gated: `_postfrac_adaptrend_v1_rv_band.py` (OOS), `_postfrac_kc_squeeze.py`. (medium, mitigated — known-deferred, final dispositions correct)
4. No runner-level smoke test; only primitive-level tests + per-runner embedded round-trip asserts (fire on manual run, not under pytest). (low)
5. Lo n<2 stub shapes omit `sr_lo_adjusted`/`lo_eta` keys (cosmetic schema asymmetry). (low)

**Scoped-out (NOT residual debt):** Methodology debt #2 portfolio per-leg solo PSRs (BTC 0.9419 / SOL 0.8663) intentionally left stitched — `_aggregate_book` design target is the portfolio headline PSR only.
