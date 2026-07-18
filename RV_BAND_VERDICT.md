# RV-Band Gate — Post-Patch Verdict

**DECISION: SHELF (ESCALATE methodology to user)**

AdaptiveTrend V1 + realized-vol band gate (`rv_lo=0.25 / rv_hi=0.75 / lookback_days=365 / rv_window_hours=720`) fails the accretive promotion bar after warm-prefix patch. PSR/lift gates met, robustness within envelope, but walk-forward 45.0% sufficient-trade quarter-positivity (vs 70% gate, 56% volsize baseline -> **-11pp WORSE than baseline**) refutes the regime-separation hypothesis. Three independent verification lenses returned `partial / partial / wrong`. Memo TODO_LEG line 54 named walk-forward <60% the explicit shelf trigger. SHELF.

This is the 10th consecutive AdaptiveTrend single-feature variant to fail accretive promotion (V2 + 8 imp + RV-band). Pattern: 5-OOS lift survives, walk-forward refutes. AdaptiveTrend single-feature search is **EXHAUSTED**.

---

## Headline Numbers (Pinned Arm: `wp_pinned_25_75`)

| Metric | BASE | RV-BAND | Delta |
|---|---|---|---|
| PSR (5-OOS) | 0.939795 | 0.994299 | **+0.0545** |
| Compounded (5-OOS) | (inflated*) | (inflated*) | **+94.70pp** |
| Trades | 254 | 125 | **-129 (-50.8%)** |
| Hit rate | 42.13% | 44.00% | +1.87pp |
| Windows positive | 3/5 | 4/5 | +1 |
| Walk-forward sufficient-trade positive | - | **45.0% (9/20)** | vs 56% baseline = **-11pp** |
| Walk-forward all-quarter positive | - | 45.45% (10/22) | vs 70% gate = **-25pp** |
| Walk-forward aggregate compounded | - | +46.35% | - |
| Walk-forward PSR | - | 0.812 | - |

*Absolute compounded returns are inflated by undisclosed methodology change in patched runner (see Lens 1). Within-runner BASE vs RV delta is internally consistent and the gate effect is directionally valid; absolute numbers are NOT comparable to locked +45.52% reference baseline.

---

## All Arms (5-OOS)

| Arm | PSR | dPSR | dcomp (pp) | dtrades | Windows | Hit rate |
|---|---|---|---|---|---|---|
| `wp_pinned_25_75` | 0.994299 | +0.0545 | +94.70 | -129 | 3/5 -> 4/5 | 42.13% -> 44.00% |
| `wp_wide_20_80` | 0.992320 | +0.0525 | +107.97 | -109 | 3/5 -> 5/5 | 42.13% -> 42.76% |
| `wp_tight_30_70` | 0.995097 | +0.0553 | +72.70 | -145 | 3/5 -> 5/5 | 42.13% -> 43.12% |
| `wp_short_lkb_180` | 0.955064 | +0.0153 | **-24.19** | -134 | 3/5 -> 4/5 | 42.13% -> 40.83% |

Robustness envelope: max deviation 0.039 from pinned (within +/-0.05 target). Pinned <-> wide <-> tight cluster within 0.003 PSR. `short_lkb_180` confirms TODO_LEG's flagged risk -- shorter lookback degrades and locks 365d as the correct setting, but does not save the candidate.

---

## Gate-by-Gate (TODO_LEG Bar)

| # | Gate | Threshold | Result | Status |
|---|---|---|---|---|
| 1 | 5-OOS PSR | > 0.95 | 0.994 | **MET** |
| 2 | Walk-forward | >= 70% (kill at <= 60%) | 45.0% | **FAILED (kill condition)** |
| 3 | PSR lift | >= +0.03 | +0.0545 | MET (caveat below) |
| 4 | Robustness | within +/-0.05 | 0.039 max | MET |
| 5 | Correlation to multifactor-v1 | < +0.2 | NOT MEASURED | moot |
| 6 | Live-backtest parity | 1000+ bars | NOT MEASURED | moot |

Gate 2 was the load-bearing test for this candidate's reason-to-exist. Gates 5/6 are moot given gate 2 failure (memo line 54 mandate).

Gate 3 caveat: lift numerically clears, but coexists with -129 trades and only +1.87pp hit-rate change. PSR rise is partially a trade-count-shrinkage / variance-shrinkage artifact, not pure regime separation.

---

## Verification (3 Lenses)

### Lens 1 -- Correctness-of-patch: **PARTIAL**

Warm-prefix mechanics are correct (slice, `EntryTime` filter, `max_dd` slice all sound). HOWEVER, the patched runner silently introduced a SECOND change: it switched return computation from `stats["Return [%]"]` (used by reference and all sibling postfrac runners) to `prod(1 + ReturnPct) - 1` (lines 190-200 of `_postfrac_adaptrend_v1_rv_band.py`).

- Reference BASE = +45.52% / 255 trades / PSR 0.905 (at 10bps RT)
- Patched BASE = +134.63% / 254 trades / PSR 0.940 (at 15bps RT)

Same strategy class, ~same trade count, HIGHER commission -- yet compounded return jumped +89pp. That is methodology, not commission. `Trade.ReturnPct` is per-trade price-return on entry notional; compounding 254 of them assumes 100%-of-equity sizing, while the strategy actually sizes at fractional notional via `risk_per_trade_pct=1%`. The patched method materially OVERSTATES absolute compounded return and PSR vs the equity-curve-based reference. The runner's `reference_postfrac_base` note attributes the gap to commission, which is misleading.

Sub-issue (look-ahead): trades during the 395d warm prefix are correctly excluded from attribution, but `self.equity` at the first OOS bar reflects warm-prefix P&L -- OOS sizing depends on pre-window outcomes. Subtle equity-state coupling, not a strict look-ahead.

**What this means:**
- Within-runner BASE vs RV delta IS internally consistent (both arms use the same buggy compounding), so the +94.7pp / +0.054 PSR gate effect is directionally valid.
- Absolute +134.63% BASE and +229.33% RV numbers are inflated and CANNOT be compared to the +45.52% locked baseline.
- Walk-forward 45% vs 56% baseline IS apples-to-apples (volsize WF baseline uses the same `prod` method). FAILS_WALKFORWARD stands.

Recommended fix: switch the runner to `stats["Return [%]"]` so it matches reference/sibling runners (preferred), OR re-anchor BASE expectations explicitly and update the note to call out the methodology change.

### Lens 2 -- Completeness vs TODO_LEG bar: **PARTIAL -> SHELF**

Hit OOS-PSR + lift + robustness, but FAILED the explicit walk-forward kill condition. Memo line 54 named this exact outcome the explicit shelf trigger: vol alone is not the orthogonal axis hoped for.

### Lens 3 -- Alternative-explanation: **WRONG (refuted)**

RV-band gate REFUTED on every alternative test:
1. **2024_H1 regime leakage**: base -5.60% rescued to +22.7-39.4% across 3 arms with trades cut 67-69%. Strip 2024_H1 and base compounds to ~+154%, pinned RV to ~+135% -- RV UNDERPERFORMS base on the other 4 windows. Entire +95pp delta is single-window rescue.
2. **Hit-rate denominator artifact**: WR delta only +0.6 to +1.9pp (and -1.3pp on `short_lkb_180`). PSR lift is variance shrinkage from -43% to -57% trade cut, not per-trade edge.
3. **4-arm direction NOT robust**: `short_lkb_180` delivers NEGATIVE -24pp compounded vs +95/+108/+73pp for other arms. 120pp swing on lookback hyperparameter = parameter-grabs-regime-noise.
4. **Walk-forward refutes the hypothesis DIRECTLY**: 45.0% sufficient-trade positive vs 56% volsize baseline. Quarter-positivity went DOWN 11pp, not UP. 2024 quarters are 4/4 positive (regime cherry); 2021 H2 + 2022 H1 are 1/8; 2025-26 is 2/6 sufficient-trade positive.
5. **TODO_LEG prediction** = "gate IMPROVES quarter-consistency by blocking dead-vol AND blow-off." **Observation** = "gate REDUCES quarter-consistency by 11pp." This is REFUTATION, not parameter-tuning territory.

---

## Decision Path

- Pinned PSR > 0.95: YES
- Lift >= +0.03: YES (with denominator-artifact caveat)
- Robustness variance <= 0.05: YES
- Walk-forward >= 70%: NO (45.0%, below 56% baseline)
- 2+ verifies wrong/partial: YES (partial/partial/wrong)

**Walk-forward gate fails AND verification panel is 2-partial-1-wrong -> SHELF.**

Memo line 54 prohibits parameter rescue at this stage. Do NOT sweep lookback != 180. Do NOT chase finer band edges. The hypothesis ("realized-vol bands carve a regime where AdaptiveTrend wins more, blocking dead-vol AND blow-off") was empirically refuted.

---

## Methodology Issue -- Surface to User

The patched RV-band runner uses `prod(1+ReturnPct)-1` while reference/sibling runners use `stats["Return [%]"]`. Within-runner deltas remain valid; absolute numbers do not match the locked baselines. Two paths:

1. **Recommended:** patch the runner to `stats["Return [%]"]` and re-run for clean apples-to-apples archives. WF verdict won't change (volsize WF uses same `prod` method, so the 45% vs 56% comparison is consistent regardless).
2. Re-anchor BASE expectations and document the methodology change explicitly in `reference_postfrac_base`.

Either way the SHELF verdict is unaffected.

---

## Pattern Alert

AdaptiveTrend single-feature search is **EXHAUSTED** -- 10 consecutive failures:
- V2 (Algorithm 2 monthly re-opt)
- 8 V1 improvements (funding-skip, half-out 1R, MTF H1, regime-gate ADX, regime-gate vol, session-volume, time-stop, vol-scaled-sizing)
- RV-band gate (this verdict)

Recommendation per memo: STOP searching single-feature ablations on AdaptiveTrend V1. Pivot to either (a) different feature class (OI-divergence, taker-flow imbalance), (b) different strategy class entirely (KC squeeze, intraday TSMOM, funding-extreme contrarian) -- all four remain in TODO_LEG queue.

---

## Files

- 5-OOS: `reports/postfrac_adaptrend_v1_rv_band_wp_{pinned_25_75,wide_20_80,tight_30_70,short_lkb_180}.json`
- Walk-forward: `reports/postfrac_walkforward_adaptrend_v1_rv_band.json`
- Runners: `tools/_postfrac_adaptrend_v1_rv_band.py`, `tools/_postfrac_wf_adaptrend_v1_rv_band.py`
