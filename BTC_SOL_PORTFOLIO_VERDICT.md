# BTC + SOL Multifactor-v1 4H-Gate Cross-Asset Portfolio — Verdict

**Date:** 2026-06-03
**TODO_LEG:** cross_asset_btc_sol_portfolio
**Strategy under test:** multifactor-v1 with 4H EMA200 regime gate, applied to BTC + SOL
**Reference doc:** `obsidian/.../todo_leg_cross_asset_btc_sol_portfolio.md`

---

## DECISION: **SHELF_ALL — pivot to next TODO_LEG**

Walk-forward gate (>=70% positive quarters among sufficient) **fails at 59.09%**. The headline portfolio PSR of 0.97 is an n-inflated stitched-stream artifact, not a true portfolio Sharpe. Cost-stress at 15bps was never measured — the load-bearing gate explicitly named by the TODO_LEG. SOL-solo PSR 0.866 falls below the 0.90 portfolio bar and well below the 0.978 BTC-solo bar that authorized current mainnet deployment. No promote candidate clears its own gate. Pivot to the next TODO_LEG (recommend `turn-of-15m-candle` or the queued `funding-extreme contrarian` re-run if data window has refreshed).

---

## Pre-requisite numbers (reconfirmed)

- SOL fresh 2026H1: **+14.79%**, 11 trades, PSR proxy **0.8815**
- BTC↔SOL 4H Pearson correlation: **0.681** (regime agreement 83.82% → cross-veto would bind ~16% of the time)

## Post-fix portfolio arms

| Arm | PSR | Comp % | Max DD % | Wins | BTC-solo PSR | SOL-solo PSR | Lift vs better solo |
|---|---|---|---|---|---|---|---|
| port_5050_noveto_v2 | **0.9717** | +49.34% | -9.78% | 5/6 | 0.9419 | 0.8663 | +0.0298 |
| port_5050_veto_v2   | 0.9584 | +31.00% | **-5.39%** | 4/6 | 0.8888 | 0.9024 | +0.0560 |
| port_6040_veto_v2   | 0.9482 | +32.32% | -6.31% | 4/6 | 0.8888 | 0.9024 | +0.0459 |

Walk-forward (best arm, port_5050_noveto_v2): **22 quarters, 13 positive = 59.09%**, aggregate PSR 0.9378 across 388 trades. **Misses the 70% gate by -10.91pp.** Worst quarters: 2024Q3 -6.35%, 2025Q3 -4.15%, 2023Q3 -3.62% (recurring summer-chop signature).

---

## Verifier panel (3 lenses)

### Lens 1 — Fix correctness: **partial / high confidence**

The PSR-propagation fix at `tools/_postfrac_mf_4h_btc_sol_portfolio.py:304-306` works mechanically: all three v2 arms now report portfolio_psr > 0 (previously 0). But the proxy is **not a true equity-curve PSR**:

- It computes PSR on a *concatenated* per-trade stream (BTC trades stacked with SOL trades, each scaled by its allocation), then runs `compute_psr` on the pooled array.
- Because `compute_psr` is scale-invariant (`psr_eval.py:144`), per-slice weighting has zero effect on per-slice SR/skew/kurt — weighting only matters at the pool boundary.
- The metric should be labelled **pooled-trade-proxy PSR**, NOT portfolio PSR, in any downstream report.
- Solo-recovery sanity (w_sol=0) is approximate, not exact; defensive guard needed at `_synth_portfolio_window:304-306`.

Fix is fit for evidence-of-edge gating but **does not change the promotion decision** — the WF gate is still the binding constraint and it fails.

### Lens 2 — Completeness vs TODO_LEG promotion bar: **partial / high confidence**

Gate-by-gate:

1. SOL 2026H1 re-validation — **MET** (PSR proxy 0.8815 > 0.85 halt-floor)
2. Portfolio PSR > max(BTC-solo, SOL-solo) — **MET** for noveto (+0.0298 cushion); marginal for others
3. BTC↔SOL 4H corr < 0.95 — **MET** (0.681)
4. Portfolio max DD < -25% — **MET comfortably** (best -5.39%, worst -9.78%)
5. 15bps cost-stress, portfolio PSR > 0.90 both legs — **NOT MEASURED** (TODO_LEG names this as load-bearing)
6. Walk-forward >= 70% positive quarters — **NOT MET** (59.09%)

SOL-as-separate-leg: SOL-solo PSR 0.866 below the 0.90 portfolio comfort bar; would deploy a second-class leg under the first-class brand. Not justified at this evidence level.

### Lens 3 — Alternative explanation: **partial / high confidence**

The portfolio PSR headline is a **measurement artifact**, not evidence of diversification.

- PSR scales with n. Portfolio n=236 mechanically out-prints BTC-solo n=138 (~sqrt(236/138) inflation factor) regardless of trade quality.
- Portfolio compounded **49.34% < BTC-solo 60.71%** — adding SOL **destroys 11.37pp** of compounded return on the n-invariant metric that matters for capital allocation.
- 5/6 windows-positive vs BTC-solo 4/6 traces entirely to SOL rescuing 2024H1; in 2025H1 SOL clearly drags BTC.
- Cross-veto LOWERS PSR/comp/wins vs noveto on n-contaminated PSR. On risk-adjusted terms it is a *risk-reduction lever* (DD halved -9.78 → -5.39) not an edge improver.
- **Methodology debt**: SOL standalone (`reports/postfrac_mf_4h_sol.json`) and SOL portfolio-slice report **different trade counts per shared window** (e.g. 2024H1: 18 vs 14, 2022H1: 11 vs 10) at the same commission. Different signal/config/code version. Resolve before citing either as canonical baseline.

---

## Top 3 caveats

1. **WF 59% < 70%** — the binding gate. Recurring summer-chop signature 3 of 5 years (2023Q3, 2024Q3, 2025Q3) means promoting now would deploy into a known regime weakness with insufficient cushion.
2. **Pooled-trade-stream PSR is not a portfolio Sharpe** — the 0.97 headline is n-inflated. Any future cross-coin report must use a time-aligned weighted equity curve PSR.
3. **SOL standalone vs portfolio-slice trade-count divergence is unresolved** — affects which SOL number is canonical for any SOL-as-second-leg decision. Cost-stress at 15bps was never run on either leg.

---

## Suggested next step

1. **Mark `cross_asset_btc_sol_portfolio` SHELVED.** Update MEMORY.md index.
2. **Do NOT deploy SOL as a separate leg.** SOL-solo PSR 0.866 + 4/6 windows + no standalone WF + unresolved trade-count divergence = does not clear the bar that gated BTC deployment (PSR 0.978, 5/5, WF 80%).
3. **Pivot to next TODO_LEG.** Candidates queued: `turn-of-15m-candle` (untested), `funding-extreme contrarian` (data may have refreshed), `KC squeeze breakout` (practitioner-backed). Multi-feature stacks remain the higher-EV branch.
4. **Methodology fix to land before next cross-coin attempt:**
   - Refactor `tools/_postfrac_mf_4h_btc_sol_portfolio.py` to compute portfolio PSR on a time-aligned weighted bar-equity curve, not stitched trade returns.
   - Add `tools/cost_stress_psr.py` invocation to the portfolio harness so gate 5 always runs.
   - Reconcile SOL standalone vs portfolio-slice trade-count divergence — pick one canonical signal definition.

Deployed BTC-only multifactor-v1 + 4H gate remains the production leg. No change to mainnet config.
