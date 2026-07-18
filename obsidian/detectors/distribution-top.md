---
tags: [detector]
---

# Distribution-Top (DT) detector

One of the two pattern detectors in [[HYBRID-short-strategy]]. Matches the
user's image-16 visual rule: a 16-bar uptrend followed by an 8-bar sideways
chop near the top, ending in a breakdown bar (close below chop_low OR close
below EMA(24)).

## Defining params (locked)

| | value |
|---|--:|
| uptrend_bars | 16 |
| chop_bars | 8 |
| min_rise_pct | 2.5% |
| max_chop_ratio | 0.55 (chop range / uptrend range) |
| require_chop_at_top | true |
| breakdown_mode | chop_low_or_ema24 |

## Implementation

`detect_distribution_top` in `../../strategy/cnh_detectors.py`. Pure function,
takes a dataframe and an end_idx, returns a dict if the pattern terminates
at that bar.

## Regime fit

Per the [[decisions/keep-both-detectors|DT vs ICnH ablation experiment]], DT dominates
**strong-trend years**:

- 2021 raging bull: DT-only +34.3% vs ICnH-only +0.7%
- DT mean per-trade edge: **+114.8 bps** (sharper individual signal)
- DT win rate: 65.4%

## Why "distribution"?

The pattern reflects classical Wyckoff/Schabacker distribution: smart money
unloading into a topping range while retail piles in, then a clean breakdown
through the range low. The 8-bar chop at the top is the "distribution
phase"; the close-below-chop-low is the "markdown."

## See also

- [[detectors/inverse-cup-handle]] — complementary detector
- [[detectors/pattern-dedup]] — how the two are merged
- [[concepts/regime-complementarity]]
