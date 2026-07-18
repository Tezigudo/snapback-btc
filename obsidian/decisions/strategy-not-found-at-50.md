---
tags: [decision, learning]
locked: 2026-05-26
---

# No strategy fits at $50/leg — the constraint is capital

Structural finding after four pitches in three days:

1. [[strategies/chop-reverter-FAILED]] — failed Phase 1 (53.5 vs 60/yr gate)
2. [[HYBRID-short-strategy]] — strategy works, but Phase 2 sizing requires ≥$100/leg
3. LONG_OPTIMAL (parked) — wider stops than HYBRID, even worse at $50/leg
4. [[strategies/retest-resume-FAILED]] — 25.2% WR, cum -61.7% — concept broken

## Why this is structural, not bad luck

At BTC $80-100k × Binance 0.001 BTC min-qty = $80-100 notional minimum.

For a position to fit:
- `equity × risk_pct / stop_pct ≥ 80`
- At equity=$50 and risk=1.5%: stop_pct must be ≤ 0.94% — very tight
- Tight stops (≤1%) get killed by 4h BTC noise; that's why
  [[strategies/retest-resume-FAILED|Retest-Resume]] failed

The only ways out:
- (a) Larger equity per leg (≥$80, ideally $100+)
- (b) Wider stops + higher risk per trade (3%+) — bad VaR
- (c) Scalping (1m/5m) — fee-dominated, separately rejected

## Resolution

[[HYBRID-short-strategy]] deploys at $100/leg per
[[decisions/deploy-capital-floor]]. No further 3rd-leg search at $50/leg.

If user has $50 only, run v1 + Donchian as-is; don't add a 3rd leg.

## See also

- [[concepts/min-qty-floor]]
- Memory: `snapback_chopreverter_phase1_failed.md`,
  `snapback_retest_resume_failed.md`,
  `snapback_hybrid_short_phase2_sizing_floor.md`
