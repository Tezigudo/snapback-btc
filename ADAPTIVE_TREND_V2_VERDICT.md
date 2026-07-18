# AdaptiveTrend-v2 — Verdict

**Date:** 2026-06-01
**Scope:** Algorithm 2 (monthly per-asset re-optimization, arXiv 2602.11708 §3.3) on single-asset BTC perp + 8 improvement experiments + stacking trial.
**Baseline of record:** AdaptiveTrendV1 at 11 OOS windows = +67.09% compounded, 9/11 wins, 563 trades, per-trade SR 0.050, PSR 0.894, MinTRL 976 (see `ADAPTIVE_TREND_EXTENDED_VERDICT.md`).

---

## TL;DR

- **Algorithm 2 on single-asset BTC is a net loss vs static-param V1.** Monthly per-asset re-opt of (L, theta) over a 6-month rolling window: 5-OOS compounded drops 33.57% → 22.03%, PSR 0.944 → 0.798. Implementation is correct (lookahead-clean, prefix-buffered, v1 reproducible); the result is the finding.
- **Zero of 8 improvement experiments cleared the accretive bar** (Δcompounded > +5pp AND PSR improved). Best Δcomp: regime_gate_vol at +1.94pp with PSR -0.060. Best PSR: vol_scaled_sizing at +0.177 PSR but -13.26pp comp. Five experiments hurt by >5pp comp; one was a no-op (time_stop never bound).
- **No combined stack built.** Per spec guardrail — manufacturing a stack from neutral-or-worse improvements is overfitting theater. The two highest-signal levers (regime_gate_vol, vol_scaled_sizing) both reduce position aggressiveness from different angles; stacking would compound the comp drag.
- **The architectural finding matches the paper.** The paper's 1.34 → 2.41 Sharpe lift comes from the multi-asset selection layer (Sharpe-ranked top-K across 150+ perps + 70/30 long-short allocation), NOT from the single-asset monthly re-opt. On a single-asset BTC port the re-opt is noise on top of an already-converged static optimum at (L=4, theta=0.02, alpha=2.0).

---

## Algorithm 2 base result

**Implementation summary**
- `AdaptiveTrendV2` strategy in `strategy/signals_adaptive_trend_v2.py`.
- Per-month re-fit at the first 15m bar of each UTC month: grid sweep over (L, theta) on the trailing 6 months of H6 bars; alpha fixed at 2.0 (extended sweep showed monotone in alpha across all cells).
- Inner sim is pure-numpy H6 replay (MOM entry + ATR-trail-stop), ranked by per-trade Sharpe. Avoids 360 nested Backtest constructions.
- Lookahead-clean: fit slice strict less-than month-start, MOM/ATR recomputed *within* slice. 3/3 boundary lookahead test pass.
- Load-bearing fix: 6-month prefix buffer + `trade_start_ns` entry guard so v2 doesn't carry a Dec position into Jan OOS while v1 starts flat.

**5-OOS sanity (vs V1, $1M)**

| Window  | V1 trades | V1 net% | V2 trades | V2 net% |
|---------|-----------|---------|-----------|---------|
| 2022_H1 |        40 |   -1.03 |        52 |   -2.15 |
| 2023_H1 |        33 |  +17.58 |        46 |  +10.50 |
| 2024_H1 |        39 |   +4.38 |        54 |   +5.64 |
| 2024_H2 |        43 |   +6.54 |        57 |   +3.17 |
| 2025_H1 |        38 |   +3.22 |        44 |   +3.55 |
| **TOTAL** | **193** | **+33.57** | **253** | **+22.03** |

- Re-opt is firing (e.g. 2024_H1: Jan=(4,0.025), Feb=(3,0.015), Mar=(3,0.025), Apr=(4,0.02), May=(3,0.015), Jun=(6,0.025)).
- Trade count up +31% from regime-adapted params (expected behavior).
- Compounded return DROPS 11.5pp; PSR drops 0.146.
- Root cause: fit Sharpes are almost universally negative across all (L, theta) cells every month. The algorithm picks the "least bad" candidate, which drifts away from the static plateau optimum (4, 0.02) that the extended sweep proved dominant.

**11-OOS extended baseline (used for improvement comparisons)**
- AdaptiveTrendV2: +43.90% comp, 8/11 wins, 555 trades, per-trade Sharpe 0.0356, PSR 0.806, MinTRL 2010.
- (V1 at same 11 OOS: +67.09%, 9/11, 563 trades, PSR 0.894 — V2 is materially worse on every metric.)

---

## Improvement ranking

Ranked by Δ compounded vs AdaptiveTrendV2 base over 11 OOS windows at $1M with real Binance funding.

| Improvement              | Δcomp (pp) | Δ PSR  | Verdict   | Notes |
|--------------------------|-----------:|-------:|-----------|-------|
| regime_gate_vol          |     +1.94  | -0.060 | NEUTRAL   | comp drift up, PSR/SR/wins all worse |
| time_stop_24h            |      0.00  |  0.000 | NEUTRAL   | lever never binds (ATR-trail closes first) |
| funding_skip             |     -4.42  | -0.074 | NEUTRAL   | threshold (5bps) rarely fires; PSR meaningfully worse |
| vol_scaled_sizing        |    -13.26  | +0.177 | HURTING   | PSR crossed 0.95, but `int()` truncation kills small positions; comp loses 13pp |
| half_out_at_1R           |    -13.48  | -0.027 | HURTING   | caps right tail; per-entry expectancy falls |
| regime_gate_adx          |    -14.87  | -0.109 | HURTING   | redundant trend-on-trend filter; throws out winners |
| session_volume_filter    |    -17.69  | -0.412 | HURTING   | clock-time gate; PSR collapses to 0.394, MinTRL degenerate |
| mtf_h1_confirmation      |    -30.59  | -0.273 | HURTING   | extreme H1 RSI correlates with best trend entries — filter inverted the signal |

**Best single lever by Δcomp:** regime_gate_vol (+1.94pp). Fails PSR support.
**Best single lever by ΔPSR:** vol_scaled_sizing (+0.177, lands at 0.983). Fails comp gate (lost 13pp).
**Accretive count:** 0 / 8.

---

## Combined stack result + final verdict

**No combined stack was built.** Per the spec guardrail: when accretive_count < 2, do not manufacture a stack from neutral improvements. This is the correct call here — every lever either failed to fire (time_stop), failed to lift comp (regime_gate_vol, vol_scaled_sizing), or actively destroyed comp. The two highest-PSR-signal levers both reduce position aggressiveness; stacking would compound the drag without lifting comp toward the 0.95 PSR gate at a meaningful return level.

Result file: `/Users/god/Desktop/work/snapback-btc/reports/adaptrend_v2_combined_stack.json` (accretive_count=0, verdict=shelf).

---

## Promotion recommendation

**SHELF.**

Reasoning:
- AdaptiveTrendV2 base loses 23pp of compounded return vs V1 at the 11-OOS comparison (+43.90% vs +67.09%) and loses on PSR (0.806 vs 0.894). The monthly re-opt does not pay for itself on single-asset BTC.
- No improvement closed the gap. Best comp delta was +1.94pp (regime_gate_vol), which still leaves V2 ~21pp below V1.
- The PSR > 0.95 promote gate is not met by V2-base (0.806) nor by any single improvement except vol_scaled_sizing (0.983) — and that variant fails the comp > 50% gate (lands at 30.64%).
- Stacking is not a viable path because the only lever that lifts PSR (vol_scaled_sizing) does so by shrinking position size, which is incompatible with the other PSR-positive lever (regime_gate_vol's selectivity).

Promoting V2 as a dry-run leg would put a strategy with worse risk-adjusted return AND worse compounded return into deploy. There is no defensible reason to ship it.

---

## Operator playbook

**Do not promote V2 as-is.** Recommended next moves, in order of expected payoff:

1. **Test the paper's actual edge: multi-asset port.** The arXiv 2602.11708 Sharpe lift comes from the cross-sectional selection layer, not the single-asset re-opt. Build a basket runner (BTC + ETH + SOL at minimum) with:
   - Per-leg AdaptiveTrendV2 monthly re-opt (already implemented).
   - Sharpe-ranked top-K selection (paper's gamma_L = 1.3 long, gamma_S = 1.7 short post-fit gates).
   - 70/30 long-short allocation across the active leg.
   - Inverse-vol within-leg sizing + correlation overlay CF = sqrt(N / (N + N(N-1)*rho_bar)) — at BTC/ETH/SOL rho_bar ~0.65, CF ~0.66.
   This is the load-bearing experiment. If it doesn't lift, the architecture genuinely doesn't transfer to crypto perps.

2. **Re-tune vol_scaled_sizing as a standalone iteration on V1.** It crossed PSR 0.95 — that's signal. Two fixes to try:
   - Raise vol target 0.15 → 0.20 (paper-anchored, Bitwise-validated).
   - Replace `int(min(target_btc, max_btc))` truncation with fractional sizing — the int() rounds small positions to 0, biasing toward survivor entries.
   This is a single-lever re-tune of a known-good base; cheap, ~1 day.

3. **Drop the single-asset V2 line entirely.** Keep V1 as the BTC trend leg. Reallocate research budget to (1) above.

If user explicitly wants V2 in deploy despite this verdict:
- Do NOT add to live `config/params.yaml`.
- Create `config/params_adaptive_trend_v2.yaml` as a dry-run-only config (already exists).
- Run paper trading in `deploy/` for >= 60 days before any live capital.
- Set kill switch at -10% equity DD (V2 max DD across 11 OOS is comparable to V1, but PSR is lower — be more aggressive on cutoff).

---

## Open questions / what wasn't tested

- **Multi-asset port (BTC + ETH + SOL + ...).** The single biggest unanswered question. Paper architecture lives here.
- **Compounded-return fitness instead of per-trade Sharpe.** V2 has the config knob (`fit_metric="compounded_return"`); never swept. Cheap test, may stabilize noisy monthly picks.
- **12-month fit window instead of 6.** Reduces re-opt frequency (~80 trades/fit vs ~30), less noise, slower regime adaptation. Untested.
- **TPE (Optuna) instead of grid sweep.** Research suggests ~90% of grid-optimal in ~10-13 trials. Untested on this codebase.
- **Asymmetric funding filter (block 0 to +120 min post-settlement, not -30 to +30).** The 30-min symmetric block was tested and failed; the MDPI-anchored asymmetric variant aligned to the actual spread-peak window was not tested.
- **Vol_scaled_sizing re-tune with raised target + fractional sizing.** Highest-EV iteration; not done.
- **Stack of regime_gate_vol + vol_scaled_sizing.** Both PSR-positive directionally, but both shrink exposure — explicitly excluded per stacking guardrail. If user wants the data point anyway, it's a 1-hour run.
- **What V2 does in a strong-trend year (2024_H2 full / 2025_H2).** OOS windows cover 2020_H2 through 2025_H1; nothing past Jun 2025 in the test set.

---

**Files of record:**
- Implementation: `strategy/signals_adaptive_trend_v2.py`, `tools/_adaptrend_v2_run.py`
- Config: `config/params_adaptive_trend_v2.yaml` (dry-run only, do not promote)
- Sanity (5 OOS): `reports/adaptrend_v2_sanity.json`
- Improvements: `reports/adaptrend_v2_imp_*.json` (8 files)
- Combined stack: `reports/adaptrend_v2_combined_stack.json`
- This verdict: `ADAPTIVE_TREND_V2_VERDICT.md`
