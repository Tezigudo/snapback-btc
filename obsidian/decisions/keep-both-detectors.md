---
tags: [decision, locked]
locked: 2026-05-26
---

# Keep both detectors — DT and ICnH are complementary

Decision: never drop either [[detectors/distribution-top|DT]] or
[[detectors/inverse-cup-handle|ICnH]] in a "simplification" pass. They
target different regimes.

## Evidence (from `../../tools/hybrid_dt_vs_icnh.py`)

Per-year cum return for each variant:

| Year | HYBRID | DT-only | ICnH-only | Regime |
|---:|--:|--:|--:|:--|
| 2020 | +2.6% | -0.4% | +4.6% | ICnH wins |
| 2021 (raging bull) | +13.2% | **+34.3%** | +0.7% | **DT crushes** |
| 2022 | +7.6% | +6.4% | +11.7% | ICnH |
| 2023 | +18.1% | +1.4% | **+12.4%** | **ICnH dominates** |
| 2024 | +16.6% | +0.7% | **+14.8%** | **ICnH dominates** |
| 2025 | +0.4% | +0.7% | -4.1% | DT marginally |

DT carries strong-trend years (2021 distribution-top in the bull market
top). ICnH carries chop/transition years (2023 recovery, 2024 mixed).
[[concepts/regime-complementarity]] — this is the textbook example.

## Aggregate (2020-2026)

| Variant | Trades | WR | Cum |
|---|--:|--:|--:|
| HYBRID (both) | 80 | 65.0% | **+82.1%** |
| DT-only | 70 | 60.0% | +51.2% |
| ICnH-only | 57 | 66.7% | +53.4% |

HYBRID > either alone. The patterns don't substitute; they cover
different regimes.

## What this means for monitoring

If Phase 6 dry-run shows one detector underperforming the live window,
DO NOT disable it. That window's regime may favor the other detector;
the disabled one will be carrying the next regime.

Wait for ≥6 months of mixed-regime live data before any pruning.

## See also

- [[detectors/distribution-top]]
- [[detectors/inverse-cup-handle]]
- [[concepts/regime-complementarity]]
- Memory: `snapback_hybrid_short_dt_icnh_complementary.md`
