# Short-TF / Microstructure / Multi-Feature-Stack Research Synthesis

**Date**: 2026-06-02
**Inputs**: 3 research streams (short-TF literature, multi-feature stack designs, microstructure / orderbook plays).
**Output**: 5 TODO_LEG candidates filed to memory. 4 strong candidates from stream pool of 14 (3 short-TF + 6 stacks + 5 microstructure plays) explicitly rejected here as inferior to the 5 promoted.

---

## Decision

Promote 5 candidates to TODO_LEG. Reject 9. Validate in this order: **RV-surprise gate → cross-asset BTC+SOL → turn-of-15m candle → VPIN persistence → OI-velocity divergence**. The order is set by `(token cost) × (data readiness) × (priors strength)`, not by upside ceiling. Cheap and quickly-killable comes first; expensive new-infrastructure comes last.

---

## Promoted candidates

| # | Name | File | Class | Token cost | Data ready? | In-house priors? |
|---|------|------|-------|------------|-------------|------------------|
| 1 | RV-surprise gate on AdaptiveTrend V1 | `todo_leg_rv_surprise_gate.md` | vol-regime gate | ~6k | Yes (1h cached) | Strong (V1 PSR 0.93, V2 vol_scaled 0.983) |
| 2 | Cross-asset BTC+SOL portfolio | `todo_leg_cross_asset_btc_sol_portfolio.md` | cross-sectional | ~9k | Mostly (SOL may need build) | Strong (SOL +41.93%, 5/5 wins) |
| 3 | Turn-of-15m-candle seasonality | `todo_leg_turn_of_candle_15m.md` | time-of-candle | ~18k | Need 1m parquet | None — pure external |
| 4 | VPIN persistence (order-flow toxicity) | `todo_leg_vpin_persistence.md` | microstructure (persistence) | ~10k | Yes (patched 15m parquet) | Negative — taker-flow already shelved on same data |
| 5 | OI-velocity / price divergence | `todo_leg_oi_velocity_divergence.md` | positioning | ~14k | NO — needs Tardis ingestion | None (BIS WP 1087 only mechanism) |

Files (absolute paths):
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_rv_surprise_gate.md`
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_cross_asset_btc_sol_portfolio.md`
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_turn_of_candle_15m.md`
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_vpin_persistence.md`
- `/Users/god/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_oi_velocity_divergence.md`

---

## Validation order (rationale)

### FIRST: RV-surprise gate (~6k tokens)
- Cheapest. Bolts onto existing PSR-0.93 base. No new data, no new infrastructure.
- Decisive failure mode (gate-without-lift) shelved in 1 sweep — won't get stuck in parameter rescue.
- Best information yield per token in batch.

### SECOND: Cross-asset BTC+SOL (~9k tokens)
- Composition over validated parts. Lowest novelty risk.
- Surfaces a portfolio engineering decision that's overdue regardless (the SOL transfer finding has been on shelf for at least weeks).
- Re-validating SOL on 2026 H1 is non-optional anyway; this strategy is just the natural use of that re-validation.

### THIRD: Turn-of-15m-candle (~18k tokens)
- Single-most-orthogonal feature class in batch. Time-of-candle seasonality is unrepresented in graveyard.
- High kill probability (crowding decay is the explicit mechanism) but the 2023-2026 OOS replication is conclusive in either direction. Result is information either way.
- 1m parquet build cost is non-trivial but reusable for future microstructure-adjacent strategies.

### FOURTH: VPIN persistence (~10k tokens)
- Data already on disk (patched taker columns in 15m parquet from previous taker-flow attempt).
- Second attempt at order-flow on this data — bar of evidence is higher than for a fresh class. The persistence-vs-instantaneous refutation test (gate #6) makes the shelf decision cheap.
- Deferred to fourth because the prior failure on the same data raises base-rate expected loss.

### FIFTH (DEFERRED-PENDING-DATA-CHECK): OI-velocity divergence (~14k tokens)
- Requires Tardis.dev ingestion (new infrastructure). Validation gate #6 is a data-quality sanity check that MUST run before strategy build.
- Tardis ingestion is reusable for funding history archives and liquidation streams — amortizes across future candidates.
- Defer until first 4 candidates resolve. If 2+ promote, this fifth slot may not be needed in current capacity.

---

## Rejected candidates (and why)

| Stream | Candidate | Reason rejected |
|--------|-----------|-----------------|
| Short-TF #2 | Drift bursts in pure jumps (JBES 2025) | Detection paper, not strategy paper. Authors say hedge funds don't profit. No Sharpe basis. Would require strategy construction from zero. |
| Short-TF #3 | Funding-rate arbitrage (Pythagoras) | Cross-exchange edge, not single-venue. Not implementable on Binance-only architecture without spot leg + cross-venue infra. |
| Short-TF #4 | Volume-weighted TSMOM (SSRN 4825389) | Daily frequency in paper; 30m port is speculative; TSMOM family already in graveyard via Shen/Urquhart half-day. |
| Stack #1 | Multifactor + OI/funding Z-score gate | Subsumed by promoted OI-velocity candidate (#5) — same data dependency, more sophisticated signal. |
| Stack #3 | Multifactor + spot-vs-perp CVD divergence | Evidence is anecdotal chart commentary. Needs sub-bar trade tape parity test before evidence. Defer until parity infrastructure exists. |
| Stack #5 | Fresh 30m 3-feature voting base | Too speculative. Three new components with no in-house priors AND novel voting wrapper. Token cost / failure-mode count too high. |
| Stack #6 | AdaptiveTrend + VolZ trigger + 4H gate | session_volume already shelved on AdaptiveTrend as filter — same volume class. Weak orthogonality argument. |
| Microstructure #3 | Liquidation-cascade trigger | Binance public WS truncates to largest liquidation per 1000ms — severe measurement bias documented by Binance docs. Tardis archives mitigate but Tardis dependency overlaps with #5 OI-velocity. Defer. |
| Microstructure #4 | Multi-level order book imbalance | Heavy book-replay infrastructure (~25k tokens). Not justified without 2+ planned book-based strategies. Defer. |
| Microstructure #6 | Funding composite (level + ROC) | Too close to shelved funding-extreme candidate. ROC overlay is marginal differentiation. |

---

## Portfolio shape (5 candidates)

The promoted set covers **5 structurally different feature classes**:
- vol-regime gate (RV-surprise)
- cross-sectional (BTC+SOL)
- time-of-candle seasonality (turn-of-15m)
- microstructure persistence (VPIN)
- positioning state (OI-velocity)

No two candidates collide on feature class. Each has at least one defined refutation gate that triggers shelf without parameter rescue. Expected hit rate given graveyard base rate (5/9 shelved + recent 5/5 TODO_LEG validations failed): **1-2 will promote**, **3-4 will shelf**. Validating cheapest-first means the shelf decisions cost <10k tokens each.

---

## What this report explicitly does NOT do

- Does not commit to any strategy. All 5 candidates are flagged "NOT ready to deploy" in their respective files.
- Does not pre-commit Tardis budget. OI-velocity Tardis ingestion is gated on the data-quality sanity check; that decision surfaces to user before commitment.
- Does not modify MEMORY.md (main loop owns the index).
- Does not propose any change to deployed multifactor-v1 + 4H gate parameters or live config.
- Does not promote drift-bursts, funding arb, CVD divergence, liquidation cascade, multi-level book, or fresh 30m voting base. Each is rationalized above; revisit only if 4+ of the promoted 5 shelf and capacity remains.

---

## Cost note for orchestrator

If token capacity is tight and only ONE candidate can be validated this cycle: **RV-surprise gate first**. 6k tokens, in-house priors, cheapest possible shelf decision, and a non-trivial chance of moving AdaptiveTrend across the promotion bar.
