---
tags: [concept, anti-pattern]
---

# Tuning into compliance (the anti-pattern)

The temptation to *move the gate* until a strategy passes, instead of
accepting failure and pivoting.

## Examples in this session

- [[strategies/chop-reverter-FAILED]]: at the project-canonical chop
  threshold of slope<0.05, the strategy produced **53.5 chop-trade-days/yr**
  vs the gate's 60/yr. Loosening to slope<0.07 would have made it pass.
  Rejected — that's tuning into compliance.

- The earlier "≥$150 capital" relaxation that this plan started with —
  I had relaxed memory's "≥$300 for 3 legs" rule to "≥$150 for 1 new
  leg" by analogy. Phase 2 sizing audit then showed the actual floor
  was ~$200. The plan was updated honestly.

## Why this is load-bearing

Memory `snapback_promotion_gate_broken` documents an earlier instance
where median Sharpe passed a strategy with -18.6% CAGR. That happened
because the gate was the wrong metric for the question. The strict
per-window gates in this work's HYBRID plan exist BECAUSE of that prior
failure.

If a gate fails, the choices are:
1. Honor the failure → abandon or re-pitch
2. Diagnose the gate → did the gate measure the right thing?
3. NEVER: silently widen the gate to declare pass

## How to recognise it in oneself

When a numerical result lands close to (but on the wrong side of) a
gate, you'll feel pressure to:
- "Just bump the threshold a tiny bit"
- "Use a slightly different formula that happens to favor passing"
- "Aggregate differently (median vs mean) to smooth the bad result"

Each of these is the same trap. Either the gate was correct → strategy
fails. Or the gate was wrong → fix the gate FIRST (with reasoning),
then re-run on the original strategy.

## Discipline applied here

- chop-reverter FAILED — abandoned, didn't tune the threshold
- retest-resume FAILED at 25% WR — abandoned, didn't widen the WR gate
- HYBRID Phase 2 sizing failed at $50/leg — accepted the structural floor,
  documented as [[decisions/deploy-capital-floor|≥$100/leg required]] rather
  than fudging the risk parameters

## See also

- [[strategies/chop-reverter-FAILED]]
- [[strategies/retest-resume-FAILED]]
- Memory: `snapback_promotion_gate_broken.md` (the original incident)
