# Phase 3 Validation Synthesis

**Date**: 2026-06-02
**Scope**: 4 TODO_LEG candidates + 3 AdaptiveTrend V1 single-feature ablations
**Promotion bars**: PSR >= 0.95 + compounded delta >= +5pp + non-decreasing PSR vs locked base

---

## 1. Headline

**0 of 7 candidates cleared promotion. 0 cleared ITERATE. 5 SHELF, 2 DATA_BLOCKED (PARTIAL_RESULT).**

| # | Candidate | Verdict | PSR | Compounded |
|---|-----------|---------|-----|------------|
| 1 | TSMOM intraday (half-day) | SHELF | 0.069 | -10.45% |
| 2 | Funding-extreme contrarian | SHELF | 0.736 | +2.46% |
| 3 | OI vs price divergence | DATA_BLOCKED | n/a | n/a |
| 4 | Taker-flow imbalance | DATA_BLOCKED | n/a | n/a |
| 5 | AdaptiveTrend + regime_gate_vol | SHELF | 0.898 (base 0.905) | +14.92% (base +45.52%) |
| 6 | AdaptiveTrend + time_stop_24h | SHELF | 0.278 | +2.25% |
| 7 | AdaptiveTrend + session_volume_filter | SHELF | 0.904 | +46.60% (delta +1.08pp) |

Locked production reference (multifactor-v1 + 4H gate, separate track): +77.93% compounded, PSR 0.978.

---

## 2. TODO_LEG verdicts

- **TSMOM intraday (half-day clock)** — SHELF. Compounded -10.45%, PSR 0.069, 1/5 windows positive. Horizon-mismatch: stretching a published half-hour effect to 12h dilutes SNR below detectability. Verifier CONFIRM. Do not iterate.
- **Funding-extreme contrarian** — SHELF (verging on REFUTE). PSR 0.736, compounded +2.46% but 2024 H1+H2 both negative (carry-decay null not rejected). Engulfing+volume gate is a chop killer (3.4% arm-to-fire ratio, biases entries late). n=64 with one 2025 trade swinging +1.4pp. Cost-stressed at 15bps round-trip flips compounded to -1.5% to -2.5%.
- **OI vs price divergence** — DATA_BLOCKED. Binance public OI endpoint capped at ~30 days; no local OI parquet anywhere in repo. PARTIAL_RESULT correct. Vendor procurement (CoinGlass/Coinalyze, $500-$2k) NOT recommended yet — same pivot-overfit failure mode as divergence-v1/v2 (PSR 0.36) is unrebutted by spec author.
- **Taker-flow imbalance** — DATA_BLOCKED (PARTIAL_RESULT). `exchange/data.py` already patched (task #21 actually complete, prompt was stale); blocker is the stale 5-column cached parquet. Fix is a cache rebuild, NOT a code patch.

**Walk-forward next**: nothing from this batch promotes. Taker-flow is the only one with a cheap path to a real backtest (cache rebuild ~minutes, no vendor cost). That goes first. Everything else stays shelved.

---

## 3. AdaptiveTrend ablation closure

**No lever cleared +5pp. All 3 SHELF. With prior 5 V2 ablations (paper Algorithm 2) also all SHELF, the single-feature overlay search on AdaptiveTrend is exhausted (8 in a row failed).**

- **regime_gate_vol**: -30.60pp compounded, -0.007 PSR. Gate over-suppresses runners; V1's edge is distributed across regimes, not concentrated in high-vol windows. Same failure pattern as ADX gate ablation.
- **time_stop_24h**: -43.27pp compounded, -0.627 PSR. The 24h cap chops the right tail that funds the strategy — top-decile per-trade mean drops +11.75% -> +5.77%. The base's alpha=2.0 trailing stop IS the source of edge; time-capping pre-empts it.
- **session_volume_filter**: +1.08pp compounded (below +5pp bar), -0.0018 PSR. 1.2x bucket-mean gate suppresses ~equal share of winners and losers; only pays off in trending 2024 H2. Magnitude insufficient.

**Verdict**: stop tuning single overlays on AdaptiveTrend V1. The locked +45.52% / PSR 0.905 base stands as the candidate ceiling on BTC alone. Next move on AdaptiveTrend is multi-coin stacking (SOL transferred at +41.93%, PSR 0.894 per prior memory), not more BTC single-feature gates.

---

## 4. What we learned (insights that survive even though nothing promoted)

- **Horizon-mismatch breaks published-effect ports.** TSMOM half-hour -> half-day was the cleanest test: spec implemented exactly, lookahead audit passed, still -10.45%. When porting a paper, preserve the original horizon or treat it as a fresh candidate, not an extension.
- **Asymmetric arm-to-fire ratios diagnose gate quality.** Funding-extreme armed 1,210 shorts but only 64 entries fired (3.4%) because engulfing+volume confirmation only triggers after part of the move is gone. Honest version of a contrarian thesis fires at the extreme print, not after the candle confirms. Any future contrarian leg should be tested *without* same-direction confirmation gates first.
- **AdaptiveTrend's edge is in the trailing exit, not the entries.** time_stop_24h failure proved this — capping holds drops top-decile per-trade mean by half. Future improvements should target *entry quality* (regime gate, signal filtering) only if they preserve the right tail; any change that shortens average hold is a near-certain SHELF.
- **Single-feature overlays on a tuned base rarely add +5pp.** 8/8 SHELF on AdaptiveTrend V1+V2 single overlays. Stacking 2-3 weak overlays would compound the trade-count collapse problem; stop sweeping single levers, move to either multi-coin or fundamentally different signal architecture.
- **Data-blocked verdicts have second-order value.** The taker-flow block surfaced a stale-cache vs patched-code mismatch (task #21 actually complete) that would have caused 30+ min of confused re-debugging on the next attempt. The OI block surfaced an unrebutted graveyard-adjacency concern (pivot-overfit) before $500-$2k of vendor spend.

---

## 5. Next actions (ordered)

1. **Rebuild 15m parquet cache** (~5 min, zero risk):
   ```
   rm data/historical/BTC_USDT_USDT_15m.parquet
   .venv/bin/python -m exchange.data --symbol BTC/USDT:USDT --tf 15m --days 2400
   ```
   Then re-run `tools/_postfrac_taker_flow.py`. Stress at COMMISSION=0.00075 (15bps round-trip) in the same pass.

2. **Run AdaptiveTrend multi-coin stack** (BTC + SOL, locked V1 params, no overlays). Prior memory shows SOL transferred at +41.93% / PSR 0.894. Test whether equal-weight or vol-weighted stack clears PSR 0.95 on combined equity. This is the cheapest +5pp path remaining.

3. **Draft KC squeeze candidate spec** (validator's own recommendation closing the OI block). Runnable today on existing parquets. Different signal family from everything shelved.

4. **Do NOT iterate on**:
   - TSMOM intraday at other horizons (would be in-sample fit on already-negative data)
   - Funding-extreme with parameter tweaks (carry decay null not rejected)
   - AdaptiveTrend single-feature overlays (8/8 SHELF, pattern decisive)
   - OI divergence vendor procurement (graveyard-adjacency not refuted)

5. **Update `DEPLOYED_STRATEGIES.html`** to reflect: locked = multifactor-v1+4H gate (+77.93%, PSR 0.978), candidate ceiling = AdaptiveTrend V1 base (+45.52%, PSR 0.905), 8 ablations SHELVED, 2 candidates DATA_BLOCKED pending cache rebuild.

6. **Memory write**: `phase3_validation_synthesis` fact summarizing the 0/7 promotion rate, the AdaptiveTrend single-overlay exhaustion, and the redirect to multi-coin stacking + KC squeeze. Tags: `snapback-btc, phase3, validation-synthesis`.
