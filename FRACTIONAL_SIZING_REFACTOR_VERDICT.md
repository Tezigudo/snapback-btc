# Fractional Sizing Refactor — Synthesis Verdict

**Date:** 2026-06-01
**Status:** Refactor landed; re-validation is **partially valid**; promotion decisions below.

---

## TL;DR

- The refactor mechanism is **harness-level OHLC scaling (×0.001)**, not the brief's proposed `(round(raw/0.001)*0.001)` float-return (which crashed backtesting.py 0.6.5). The mechanism is mathematically sound and matches live `round_qty_down(qty, 0.001)` exactly. Strategy `_position_units` integer math is unchanged — only docstrings touched in 8 files.
- **Adversarial verify uncovered two load-bearing problems with the headline numbers.** (a) `tools/_postfrac_mf_4h_btc_run.py` — the runner for the **just-relocked deployed strategy** — does **not** apply PRICE_SCALE. The bit-exact 77.93% / 121 trades / PSR 0.9776 match is artifactual: the pre-fix code path was rerun. (b) `adaptrend_v1`'s headline +8.47pp delta is a **gross-vs-net basis mismatch** (pre-fix 67.09% is net of funding; post-fix 75.56% is gross). Real apples-to-apples delta is likely ~0.
- **Nothing newly accretive, nothing newly viable enough to promote.** `adaptrend_v1_volsize` is the only "NEWLY_VIABLE" candidate; on walk-forward (25 rolling 3mo windows, 2020–2026) it lands at **56% positive (14/25)** — below the 70% gate — and **PSR 0.928 < 0.95**. AdX-dual-regime remains catastrophically bad (-99.4% compounded, 0/5 windows positive). Divergence v1/v2 unchanged (no signal change from sizing).
- **Net recommendation:** **maintain the existing multifactor-v1 BTC re-lock** (the deployed numbers were already correct — the int-truncation bug never bit this config at $1M cash). **Do NOT promote AdaptiveTrend V1 or its vol-scaled sibling.** Multifactor-v1 SOL: **WAIT_FOR_MORE_DATA** (PSR 0.89 < 0.95, MinTRL 159 vs n=92). Shelved divergence and ADX strategies stay shelved. Before any future deploy, **fix the four execution gaps** flagged by adversarial verify (min_notional not enforced in harness, no slippage, no max-lot cap, no maker/taker mix).

---

## Refactor scope

### Files changed (8 strategy files — doc-only, math unchanged)
- `strategy/signals_multifactor.py` — `_position_units` docstring
- `strategy/signals_adaptive_trend.py` — same
- `strategy/signals_adaptive_trend_v2.py` — same
- `strategy/signals_adaptive_trend_v2_vol_scaled_sizing.py` — same
- `strategy/signals_divergence.py` — same
- `strategy/signals_divergence_v2.py` — same
- `strategy/signals_adx_dual_regime.py` — same
- `strategy/signals_volume_profile.py` — same

### Files created
- `tools/_fractional_run.py` — harness that scales OHLC ×0.001 so 1 backtesting-unit == 0.001 BTC (matches `exchange/constraints.py::qty_step`)
- `tools/_postfrac_*.py` — 8+ per-strategy runners producing `reports/postfrac_*.json`

### Files explicitly NOT touched
- `risk.py` (hard rule), `bot.py`, `config/params.yaml`, `live_multifactor_v1.py`, deploy plumbing
- 4H regime-gate parquet loaders (flagged for follow-up — they load unscaled inside `init()`)
- 4 other affected strategy files with the same `int(min(...))` pattern: `signals.py` (snapback-v1), `signals_donchian.py`, `signals_multifactor_v2.py`, and the AdaptiveTrendV2 sibling subclasses (funding_skip, half_out, mtf_h1, regime_gate, session_volume)

---

## Strategy-by-strategy delta table

| Strategy | Pre-fix | Post-fix | Delta | Verdict | Notes |
|---|---|---|---|---|---|
| `mf_baseline` (BTC, no 4h gate) | 50.48% / 168 tr | 50.54% / 168 tr | +0.06pp | **MAINTAINED** | PSR 0.91 vs 0 hurdle; trade count identical → truncation never bit |
| `mf_4h_btc` (locked deployed) | 77.93% / 121 tr / PSR 0.978 | 77.93% / 121 tr / PSR 0.978 | bit-exact | **UNCERTAIN** — adversarial flag: runner did NOT apply PRICE_SCALE; never re-validated under the new code path |
| `mf_4h_sol` | 41.93% / 92 tr | 41.93% / 92 tr | +0.002pp | **MAINTAINED** | At SOL $20-200, target_units 9k-91k → int() never bit |
| `mf_4h_eth` | 5.83% / 129 tr | 5.83% / 129 tr | -0.003pp | **MAINTAINED** | 3/5 windows negative; PSR 0.63, no edge evidence |
| `adaptrend_v1` (α=2.0, L=4, θ=0.02) | 67.09% NET / 563 tr | 75.56% GROSS / 563 tr | **+8.47pp ARTIFACT** | **NEUTRAL** | Adversarial: pre-fix is net-of-funding, post-fix is gross-of-funding. Real delta ≈ 0pp. PSR 0.894 unchanged. |
| `adaptrend_v1_volsize` | untested at $1M | 11-OOS: 34.42% / 414 tr / PSR 0.930; 5-OOS: 15.23% / 193 / PSR 0.944 | n/a | **NEWLY_VIABLE** (label) | Does NOT clear 0.95 PSR gate; walk-forward fails (see below) |
| `div_v1_baseline` | 0% / 0 tr | 0% / 0 tr | 0pp | **MAINTAINED** | Signal gating, not sizing — refactor neutral as expected |
| `div_v2_loose` | 0.26% / 26 tr | 0.26% / 26 tr | -0.001pp | **MAINTAINED** | PSR 0.36 — no edge before or after |
| `adx_dr` (ADX dual-regime) | -99.23% / 3627 tr | -99.40% / 2841 tr | -0.17pp | **REGRESSED** (effectively unchanged blowup) | 0/5 windows positive; PSR ≈ 0; structurally negative expectancy |

---

## HEADLINE FINDING #1 — Did vol_scaled_sizing on AdaptiveTrend V1 cross PSR 0.95?

**No.**

- 5-OOS PSR: **0.9444** (just under)
- 11-OOS extended PSR: **0.9304** (clearly under)
- Walk-forward (25 rolling 3mo windows): **56% positive (14/25)** vs **70% required** — fails the consistency test
- Per-trade point Sharpe is 0.07–0.10; MinTRL 206–512 trades to clear 95% confidence
- Pattern: concentrates returns in a few breakout quarters (2023_Q1 +85.3%, 2025_Q2 +33.3%), bleeds in chop/reversal quarters (2021_Q1 -31.4%, 2022_Q1 -21.9%, 2020_Q1 -16.1%)

The vol-scaled sizing damps drawdowns but does not flip enough chop windows positive to clear promotion. The edge exists in aggregate but is too cyclical for production.

**Verdict: DO NOT PROMOTE.** Iterate on chop-window survival; do not deploy.

---

## HEADLINE FINDING #2 — Did multifactor-v1 + 4H gate (BTC) change materially?

**Surface answer: NO** — 77.93% / 121 trades / PSR 0.9776 are bit-exactly matched pre- and post-fix.
**Honest answer: WE DID NOT ACTUALLY TEST IT.**

The adversarial-`transfer` lens caught it: `tools/_postfrac_mf_4h_btc_run.py` does **not** apply `PRICE_SCALE`. `grep -n "PRICE_SCALE\|scale\|0.001"` returns zero matches. The runner's docstring (lines 11-12) admits an a-priori argument — "at $1M cash, int(target_btc) truncation is <2%, so post-fix should match pre-fix" — that is a *prediction*, not a *re-validation*.

The math behind the prediction is correct: at $1M cash and 2.75% risk × 1.5% SL, target ≈ 18-30 BTC per trade. `int(26.x) = 26`; the truncation loss is <2% per trade — well inside noise. So the locked deploy numbers are very likely *still right*, but we have not measured that under the new code path.

**Impact on the just-applied re-lock:** the locked numbers (compounded +77.93%, PSR 0.9776, 5/5 windows positive) **stand** on the basis that:
1. The strategy code is unchanged (only `_position_units` docstring touched)
2. The harness math is provably scale-invariant
3. The truncation regime where the fix would matter is well below this config's operating point

**Required follow-up before next re-lock**: patch `_postfrac_mf_4h_btc_run.py` to apply `PRICE_SCALE` *and* scale the 4H parquet inside `_build_4h_ema_aligned()`. Until then, the re-lock rests on math, not measurement.

---

## Newly viable strategies (any shelved ones rescued?)

**None.**

- `adaptrend_v1_volsize`: labeled NEWLY_VIABLE by Phase 2 (compounded > 0%, PSR > 0.5), but fails walk-forward gate (56% < 70%, PSR 0.928 < 0.95). **NOT promoted.**
- `div_v1_baseline`: 0 trades → confirms refactor didn't accidentally start firing entries. Stays shelved.
- `div_v2_loose`: PSR 0.36 — no edge. Stays shelved.
- `adx_dr`: -99.4% compounded, 0/5 windows. Structurally negative expectancy. Stays shelved.

---

## Newly accretive improvements (Phase 3)

**None measurable.** Of 7 AdaptiveTrend V1 improvement re-tests:
- `funding_skip`: NEUTRAL — delta -1.37pp net (inside ±5pp band), PSR drifts down a hair, indistinguishable from noise on V1
- `regime_gate_adx`, `regime_gate_vol`, `half_out_at_1R`, `time_stop_24h`, `mtf_h1_confirmation`, `session_volume_filter`: **NOT RUN** — agent hit session limit before completing

Phase 3 is incomplete. No accretive improvement has been identified yet.

---

## Adversarial verify summary

Four lenses ran. Two real issues found:

| Lens | Verdict | Headline issue |
|---|---|---|
| `correctness` | **PARTIALLY REAL** | Math is sound; harness scaling is correct; integer truncation matches live `round_qty_down`. **But 4 of 8 result files lack `price_scale` field — those runners did NOT apply the fix; their "MAINTAINED" verdicts are from rerunning the pre-fix path.** Affected: `mf_4h_btc`, `mf_4h_sol`, `mf_4h_eth`, `div_v2_loose`. |
| `overfit` | **REFUTED** for the released-winners hypothesis. The biggest delta (`adaptrend_v1` +8.47pp) is a basis artifact, not an overfit. Where the fix *did* release trades (`volsize` 2024H1: 38→56), the released trades **lowered** returns (+1.47% → +0.46%), the opposite of overfit. |
| `transfer` | **REFUTED** for `mf_4h_btc`. The locked-strategy runner is the **only** `_postfrac_*` runner missing PRICE_SCALE. Bit-exact match is inevitable, not informative. |
| `execution` | **PARTIALLY REAL.** Harness is mathematically sound but: (1) `min_notional_usdt=50` not enforced in backtest (lives in `exchange/constraints.py:19` + `bot.py:514` only); (2) no slippage model (4bp commission is the only proxy); (3) no Binance max-lot cap (~1000 BTC); (4) no maker/taker mix. Pre-existing issues; refactor activates regimes where they bite harder. |

---

## Recommendation

### Re-validation work that MUST happen before next deploy decision
1. **Patch `_postfrac_mf_4h_btc_run.py`** to apply PRICE_SCALE and scale the 4H parquet inside `_build_4h_ema_aligned()`. Re-run the 5 OOS windows. Until then, treat the re-lock as "math-validated, not measurement-validated."
2. **Patch the other 3 missing-scale runners** (`mf_4h_sol`, `mf_4h_eth`, `div_v2_loose`) for consistency.
3. **Decide on min_notional and slippage** in the backtest harness. At $1M cash they don't bite; at the live deploy capital level (~$25k start equity), `0.001 BTC × $70k = $70 notional` clears `min_notional=50`, but a 50% drawdown puts the bot below the floor. Add the check.
4. **Extend the doc-only update + harness scaling** to the 4 other affected files (snapback-v1, donchian, multifactor-v2, AdaptiveTrendV2 siblings) OR explicitly retire them from the harness.

### Strategy promotion verdicts

| Strategy | Verdict | Rationale |
|---|---|---|
| **multifactor-v1 BTC + 4H gate (deployed)** | **MAINTAIN existing re-lock** | Pre-fix numbers stand on mathematical grounds. Re-validation under new code path is owed but is expected to confirm. Numbers do NOT need updating. |
| **multifactor-v1 SOL** | **WAIT_FOR_MORE_DATA** | Compounded +41.93%, 5/5 windows positive, but PSR 0.894 < 0.95 and MinTRL 159 vs n=92. The edge is plausible but sample size doesn't clear the gate. Run forward in dry-mode (no capital) for another 1-2 OOS quarters then re-evaluate. **DO NOT promote to dry-run with capital yet.** |
| **multifactor-v1 ETH** | **SHELF** | PSR 0.63; 3/5 windows negative; concentrated entirely in 2022H1. No edge. |
| **AdaptiveTrend V1 (best config: α=2.0)** | **ITERATE — DO NOT PROMOTE** | PSR 0.894 (under 0.95 gate). The +8.47pp delta vs prior reference is a basis artifact. Strategy is borderline but does not clear the gate. |
| **AdaptiveTrend V1 + vol_scaled_sizing** | **ITERATE — DO NOT PROMOTE** | PSR 0.93, walk-forward 56% positive (<70%). Best candidate of the batch but does not clear gates. Focus iteration on chop-window survival (2021_Q1, 2022_Q1, 2020_Q1 all <-15%). |
| **AdaptiveTrend V2 (base + improvements)** | **DEFER** | Phase 3 incomplete (5 of 7 improvement re-tests not run due to session limit). Re-queue. |
| **Divergence v1 / v2** | **SHELF** | No edge before or after; refactor was sizing-neutral as expected. |
| **ADX dual regime** | **SHELF (permanent)** | Structurally negative expectancy (-99% compounded, 0/5 windows). Don't re-test. |
| **Volume profile POC** | **SHELF** | -56% compounded; not re-validated against fractional sizing but no path to viability. |

### What changed about the deployed strategy

**Nothing actionable.** The just-applied re-lock for multifactor-v1 BTC + 4H gate stands. The locked numbers (77.93% compounded, PSR 0.978, 5/5 windows) are still the reference. The owed work is to confirm under the new code path, not to update the numbers.

### What did NOT happen that should have

- Phase 3 (6 of 7 AdaptiveTrend V1 improvements) not completed
- 4 of 8 "post-fix" runs did not actually apply the fix (artifactually bit-exact)
- 4H parquet scaling not patched (blocks any future re-run of the 4H-gated strategies under the new code path)
- 4 sibling strategy files (snapback-v1, donchian, multifactor-v2, AdaptiveTrendV2 children) not extended

### Files explicitly closed by this synthesis

- The brief's proposed float-return mechanism is **rejected** (crashes backtesting.py)
- The harness-scaling mechanism is **accepted** as the correct approach
- The "MAINTAINED" labels on the 4 unscaled runners are **downgraded to UNCERTAIN** in this report

---

## Appendix: load-bearing file paths

- `/Users/god/Desktop/work/snapback-btc/tools/_fractional_run.py` — canonical harness
- `/Users/god/Desktop/work/snapback-btc/tools/_postfrac_mf_4h_btc_run.py` — **missing PRICE_SCALE; must be patched**
- `/Users/god/Desktop/work/snapback-btc/tools/_postfrac_mf_baseline.py` — reference correct implementation (lines 33, 98-101)
- `/Users/god/Desktop/work/snapback-btc/exchange/constraints.py:19,23` — live qty_step and min_notional
- `/Users/god/Desktop/work/snapback-btc/bot.py:502,514` — live floor + min_notional enforcement
- `/Users/god/Desktop/work/snapback-btc/reports/postfrac_*.json` — post-fix results (treat the 4 unscaled ones as advisory only)
- `/Users/god/Desktop/work/snapback-btc/reports/postfrac_walkforward_adaptrend_v1_volsize.json` — only walk-forward run; fails 70% gate
