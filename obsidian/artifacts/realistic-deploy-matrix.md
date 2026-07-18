---
tags: [artifact, data]
---

# Realistic deploy matrix

Capital × risk sweep showing what HYBRID looks like under live constraints
(min-qty / min-notional skips, compounding equity, killswitch).

Tool: `../../tools/hybrid_realistic_deploy_sim.py`
Data: `../../data/hybrid_realistic_deploy_results.json`

## Result table

| $start | risk% | $final | cum | kept/81 | skip (n,q) | WR | killed? |
|------:|-----:|------:|----:|--------:|--------:|---:|:-------:|
| 50 | 1.50 | $50.40 | +0.8% | 4 | (69,8) | 75% | no |
| 50 | 2.75 | $69.56 | +39.1% | 44 | (28,9) | 65.9% | no |
| 80 | 2.75 | $148.95 | +86.2% | 74 | (4,2) | 64.9% | no |
| 100 | 1.50 | $121.95 | +22.0% | 45 | (26,10) | 66.7% | no |
| 100 | 2.00 | $160.23 | +60.2% | 70 | (8,2) | 67.1% | no |
| **100** | **2.75** | **$197.31** | **+97.3%** | **77** | **(3,0)** | **64.9%** | **no** |
| 150 | 2.75 | $296.05 | +97.4% | 80 | (0,0) | 65.0% | no |
| 200 | 2.75 | $394.74 | +97.4% | 80 | (0,0) | 65.0% | no |

Skip column shows (min_notional, min_qty) — see [[concepts/min-qty-floor]].

## Saturation point

$150 and $200 produce identical 80/81 trades. The strategy SATURATES at
~$100/leg: above that, capital just sits idle most of the time.

## Killswitch never trips

At -35.5% threshold, no scenario crosses it. The drawdown profile is
mild — the HYBRID leg's path is monotone-positive on rolling 6-mo
windows for all eight tested configs.

## Deploy decision

Per [[decisions/deploy-capital-floor]], the locked target is **$100/leg
at risk 2.75%**.

## See also

- [[decisions/deploy-capital-floor]]
- [[concepts/min-qty-floor]]
- [[phases/phase-2-friction-sizing]]
- Memory: `snapback_hybrid_short_realistic_deploy_matrix.md`
