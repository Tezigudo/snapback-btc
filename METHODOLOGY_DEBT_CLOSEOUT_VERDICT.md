# Methodology-Debt Closeout — Verdict

**Date:** 2026-06-04
**Base:** `f995dfe` (clean revert point) → working tree
**Scope:** Backtest-tooling on LOCAL main only. Deployed bot untouched.
**Headline:** Methodology debt is **substantially closed to self-verifying** — but **NOT 100% closed**. Smoke test runs GREEN (60 passed) and the deployed-strategy verdict is unchanged. Residual coverage gaps and one stale-artifact display remain; the ONE gate live-wiring step is **deferred by design** (awaits user PSR-floor sign-off post-trip).

---

## 1. Debt status: REMAINING-RESIDUALS

The dedup + Lo correction + PSR canonical-migration work is done and verified for the discovered, smoke-covered corpus. It is NOT a full 38/38 close. Residuals below are disclosed, low-severity, and **none touch the live trading path or any promotion/deploy decision**.

### Residuals (accidental debt, not by design)
1. **8 runners unmigrated (residual debt #3).** `kc_squeeze` (1 runner, now de-fanged on disk) + `rv_band` OOS (1 runner + 9 `RV_TAG` variant reports: `pinned_25_75` / `short_lkb_180` / `smoke_pinned` / `tight_30_70` / `wide_20_80` + 4 `wp_*` twins). These are whitelisted in `KNOWN_DEFERRED_PREFIXES`.
2. **Nested-canonical coverage gap.** `reports/postfrac_adaptrend_v1.json` and `postfrac_adaptrend_v1_volsize.json` carry canonical blocks under `set_5_OOS.canonical` / `set_11_OOS_extended.canonical` — a layout the smoke test's `_canonical_units` does NOT read (it reads top-level + nested `base`/`rv` only). `postfrac_adaptrend_v1.json` is the AdaptiveTrend-V1 **base anchor** cited across many shelved-arm verdicts. Verified independently: persisted == `compute_psr(per_window, contiguous=False)` bit-for-bit on all 4 nested blocks, and Lo keys present — so the migration WAS done correctly here; it is a pure **coverage** gap with no live or promotion consequence.
3. **Stale on-disk rv_band artifacts still DISPLAY a stitched-driven PROMOTE.** `postfrac_adaptrend_v1_rv_band_smoke_pinned.json` (also `wp_tight_30_70`, `wp_wide_20_80`) on disk show `decision="PROMOTE_CANDIDATE"`, `psr_basis=None`, `base_psr=0.939795` (the STITCHED value). These are OLD-CODE writes, never regenerated. Current code re-runs them to SHELF; the WF sibling `postfrac_walkforward_adaptrend_v1_rv_band.json` = `FAILS_WALKFORWARD` shelves rv_band regardless. De-fang is in runner CODE, not on disk. `kc_squeeze` BY CONTRAST is de-fanged on disk (`kc_squeeze_5oos.json`: `gates.psr_actual=0.446298==canonical_wf`, `psr_basis=canonical_psr_walkforward`).
4. **Smoke test is uncommitted.** The primary self-verifying deliverable, `tools/tests/test_runner_smoke.py`, exists only in the working tree. It must be committed for the "self-verifying state" to be durable.

### Self-verification posture
- **Code-level claim — TRUE:** Every migrated runner gates on CANONICAL in code (`canonical[psr_walkforward][psr_vs_hurdle]`); stitched is demoted to `*_psr_stitched_legacy`. Verified at source across rv_band, time_stop, adx, and the 6 stale-ablation runners. `ZERO_STITCHED_GATING` is a runner-CODE property — NOT a clean-disk property.
- **Disk-level reality — NOT fully clean:** Pre-migration artifacts still DISPLAY stitched-gated decisions (rv_band, item 3 above). These are not outputs of current code.

---

## 2. Plainly stated

| Question | Answer |
|---|---|
| **Deploy verdict changed?** | **NO.** `phase2_gate_decision = PROCEED` on disk, unchanged. The deployed caller passes no WF series → hits the BACKCOMPAT path → byte-identical decision. |
| **v1 held?** | **YES.** `reports/postfrac_mf_4h_btc.json`: canonical `psr_walkforward.psr_vs_hurdle = 0.996555`, `lo_adjusted = 0.996555`; uncorrected/stitched `psr = 0.97755`; both clear the 0.90 floor; 5/5 windows positive; compounded +77.952%. |
| **Any shelf flip?** | **YES — benign, non-deployed.** rv_band pinned-band configs flip PROMOTE_CANDIDATE (stitched basis, base 0.940 < rv 0.994 → `psr_not_worse=True`) → SHELF (canonical basis, base 0.99995 > rv 0.97–0.98 → `psr_not_worse=False`). Structural / parameter-independent: the base arm is identical AdaptiveTrend-V1 with constant canonical PSR 0.999951 across every rv variant; rv canonical PSRs (0.973–0.98) cannot reach it, so no variant promotes under canonical. The migrated WF sibling already SHELVES rv_band, so nothing escalates toward deploy. Same granularity/benignness as the previously-recorded divergence v2_loose flip. |
| **Gate wired to deployed?** | **NO** (the new WF-series gate is NOT wired). The `phase2_gate(..., deployed=True)` CALL exists at `_postfrac_mf_4h_btc_run.py:241`, but it was introduced in commit `10adaef`, which is an **ancestor of clean base f995dfe** — this run added zero diff (`git diff f995dfe -- _postfrac_mf_4h_btc_run.py` and `git diff HEAD -- ...` are both empty). The call hits the BACKCOMPAT path (no `wf_quarterly_returns`/`wf_result` passed) and behaves as the OLD gate. The forbidden action — wiring the NEW gate into the deployed caller this run — did NOT occur. |

---

## 3. Is the corpus self-verifying? — Partially. Sound where covered; NOT all-runner.

`corpus_self_verifying = FALSE` against the task's bundled definition (*smoke green AND covers all runners*).

- **Smoke GREEN:** `tools/tests/test_runner_smoke.py` → 60 passed. Invariant (b) `headline == compute_psr(persisted, contiguous=False)` is anti-tautological (injecting a fabricated headline fails the assert; deployed v1 matched 0.996555 bit-for-bit). The footgun test has teeth (injecting a non-whitelisted footgun raises). Lo + gate tests: `tools/tests/test_lo_correction.py` + `tests/test_aggregate.py` → 28 passed.
- **NOT all-runner:** (a) the discovery is a disk-glob, not a runner census — the anti-vacuous guard is a COUNT floor (≥16), not a per-runner roll-call; (b) nested-canonical `set_5_OOS` reports (incl. the base anchor) escape all three invariants; (c) the footgun decision-detector recognizes only `gates.psr_actual` and `psr_not_worse`-gated verdicts — other decision shapes return None = silently counted clean; (d) the "38/38" denominator is unsubstantiated (the closeout's own summary said both "31 migrated" and "8 unmigrated" → 39, an arithmetic mismatch; there is no defined 38-member manifest the test checks against).

**Net:** the test is real and green where it covers, and no NON-deferred stitched-driven PROMOTE escapes (the corpus is safe). But "self-verifying over the WHOLE corpus" is not yet demonstrated. To earn a clean TRUE: extend `_canonical_units` to read the nested `set_5_OOS`/`set_11_OOS_extended` layout, regenerate the stale rv_band artifacts, then re-run smoke. That is scope creep on this reporting run and risks the 60-green state — recommend deferring.

---

## 4. Deferred-by-design (NOT a failure)

**Gate live-wiring / sign-off — the ONE intentional deferral.** The redesigned WF-series `phase2_gate` path is correct-when-called but NOT wired into the deployed strategy's validation caller. This is an explicit user sign-off item: the user must choose a PSR floor (post-trip) before the gate evaluates a real walk-forward quarterly series for the deployed strategy. Until then, the deployed caller intentionally hits the byte-identical backcompat path. This was OUT OF SCOPE this run by mandate; leaving it un-wired is the correct, intended state — not incomplete work.

---

## 5. Safety attestation (live path byte-clean)

`git diff f995dfe` AND `git diff HEAD` are both EMPTY for: `config/params.yaml`, `strategy/live_multifactor_v1.py`, `risk.py`. `.env` is gitignored/untracked (no blob to diff — not an edit). No `RISK_REVIEW` edit to `risk.py`. The deployed caller `tools/_postfrac_mf_4h_btc_run.py` is byte-untouched since base. Deployed bot (multifactor-v1+4H, real $101.95 anchor) stays byte-unchanged.

---

## 6. Recommendations

- **Safe to commit: YES — with a scoped `git add` (NEVER `git add -A`).** Commit by explicit path: `tools/aggregate.py`, `tools/tests/test_runner_smoke.py` (currently untracked — this IS the self-verifying deliverable), the migrated runners, `tests/test_aggregate.py`, and the verdict docs. EXCLUDE the accidental shell-redirect junk files `0.645`, `35`, `50%`. Confirm `git diff --cached` shows NONE of `config/params.yaml`, `risk.py`, `strategy/live_multifactor_v1.py` before committing.
- **Gate live-wiring (2-minute steps, when user picks a PSR floor post-trip):**
  1. In `tools/_postfrac_mf_4h_btc_run.py:241`, change the call from
     `phase2_gate(out["locked_reference"], canon, deployed=True)` to
     `phase2_gate(out["locked_reference"], canon, deployed=True, wf_quarterly_returns=<deployed quarterly net returns>, psr_floor=<user's chosen floor>)`.
  2. Re-run the runner; confirm `phase2_gate_decision` still reads PROCEED (or surfaces HALT_AND_SURFACE intentionally if the real WF series breaches the chosen floor).
  3. **Latent footgun to fix at the same time:** `phase2_gate`'s WF-series path calls `compute_psr(arr, ...)` with the DEFAULT `contiguous=True` on a disjoint walk-forward quarterly series (inconsistent with `aggregate_windows`, which uses `contiguous=False`). It is benign TODAY because the gate reads `psr_vs_hurdle` (invariant to the flag), but a future switch to `psr_lo_adjusted` would get a spurious Lo deflation. Pass `contiguous=False` when wiring.
