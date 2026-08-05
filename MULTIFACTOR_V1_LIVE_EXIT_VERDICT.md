# multifactor-v1 re-validated against the exit model that actually runs

**Date:** 2026-08-01
**Tool:** `tools/multifactor_v1_live_exit_revalidation.py`
**Artifact:** `reports/multifactor_v1_live_exit_revalidation.json`
**Verdict: FAILS_REVALIDATION** — 3 of 6 gates fail.

---

## The gap being tested

`DayTradeMultiFactorBTC.next()` (`strategy/signals_multifactor.py:310-326`) closes the
position on an adverse EMA200 cross whenever `require_trend=True`. The deployed config
sets it. **The live bot never runs that rule**: `bot.py:1154` gates the hook on
`bot_internals.strategy_uses_trend_exit()`, which returns True only for `donchian-v3`
and `supertrend` (`bot_internals.py:54`). Confirmed on droplet @ `9612480`.

So live multifactor-v1 exits on the exchange bracket (SL/TP) or the time stop, and
nothing else. Every prior v1 sign-off measured the *with*-trend-exit model.
`tools/multifactor_validate.py` does not close the gap either — its stage 2 compares
**entry signal bars only**, so its 100% parity result says nothing about exits.

`LiveExitMultiFactorBTC` in the tool is the parent class with that one branch removed.
Entries, sizing and bracket geometry are inherited untouched, so the existing entry
parity still holds.

## Harness control (must pass, or nothing below is trustworthy)

Re-ran the ORIGINAL walk-forward method (rolling 90-day windows, quarterly advance,
2020-01-01..2026-02-28) with the ORIGINAL locked params and the as-validated model:

> **20/25 positive (80.0%), compounded +243.81%**

Byte-matches `reports/multifactor_v1_4h_gate_walk_forward.json`. Harness verified.

## Attribution — params vs exit model

Same original method, four arms:

| arm | windows positive | compounded | 70% gate |
|---|---|---|---|
| old params + as-validated *(= signed-off artifact)* | 20/25 (80.0%) | +243.81% | PASS |
| deployed params + as-validated | 19/25 (76.0%) | +412.68% | PASS |
| old params + as-live-runs | 19/25 (76.0%) | +425.08% | PASS |
| **deployed params + as-live-runs** *(what is running)* | **16/25 (64.0%)** | +601.24% | **FAIL** |

The param drift since the sign-off (vol 2.0→1.5, funding 0.0005→0.0015, risk 2.75→3.5)
costs one window on its own. The exit model costs one more on old params. **Together they
cost four and cross the gate.** The consistency gate is exactly what catches this: the
live-exit arm makes *more* money (+601% vs +413%) out of *fewer, fatter* wins.

## The 5 locked OOS windows (5 bps/side, funding attached)

| window | as-validated | as-live-runs |
|---|---|---|
| 2022H1 | +19.71% | +22.90% |
| 2023H1 | +39.54% | +38.18% |
| 2024H1 | +20.60% | **−10.43%** |
| 2024H2 | +20.74% | +61.00% |
| 2025H1 | +8.05% | **−3.86%** |
| **compounded** | **+162.83%, 5/5** | **+135.45%, 3/5** |
| canonical PSR | 0.982 — `evidence_of_edge` | 0.9415 — **`insufficient_evidence`** |

## Kill switch — the load-bearing failure

`deploy.kill_switch_equity_fraction` (0.645) is anchored to **deploy-start equity, not the
running peak**, so peak-to-trough max drawdown is the wrong test. The right metric is the
start-anchored drawdown: for every possible deploy date, `min(equity thereafter) /
equity at that date`.

| | as-validated | as-live-runs |
|---|---|---|
| worst start-anchored DD | −24.84% | **−38.35%** (deploy 2023-12-09) |
| share of start dates breaching −35.5% | **0.00%** | **0.41%** |
| worst if deployed 2025 or later | −16.14% | **−37.44%** |

`config/params.yaml` claims *"Realistic-sim verified kill never fires."* That is true for
the model that was validated and **false for the model that runs** — including a breach
cluster in April 2025, which is recent. A breach means the leg flattens and HALTs, so the
+639% full-period figure is not even attainable in live operation.

## Full period 2020-01-01 → 2026-07-25

| | as-validated | as-live-runs |
|---|---|---|
| return | +381.27% | +639.81% |
| profit factor | 1.397 | 1.328 |
| Sharpe | 0.746 | 0.679 |
| win rate | 30.2% | 42.3% |
| median hold | 0.18 d | **0.48 d** |
| funding carry on $1M | $268k | **$822k** |
| **2025 alone** | **+16.4%** | **−21.3%** |

Holding 2.7× longer triples the funding carry — a drag the entry-only parity check never
saw. And the two models diverge hardest in the most recent full year.

## Cost stress (live-exit model, 5 OOS)

| bps/side | compounded | windows | PSR |
|---|---|---|---|
| 5 | +135.45% | 3/5 | 0.9415 |
| 10 | +72.62% | 3/5 | 0.8651 |
| 15 | +25.77% | 3/5 | 0.7223 |

Survives cost stress. Costs are not the problem; consistency and tail risk are.

## Gates

| gate | result |
|---|---|
| harness reproduces signed-off artifact | PASS |
| OOS compounded positive | PASS |
| OOS PSR `evidence_of_edge` | **FAIL** (`insufficient_evidence`) |
| walk-forward ≥70% positive | **FAIL** (64.0%) |
| cost stress 15 bps positive | PASS |
| kill switch respected | **FAIL** (0.41% of start dates breach) |

## Conclusion

v1 as it actually runs is **not a validated configuration**. It is still net-positive and
survives cost stress, so this is not "turn it off today" — but it does not clear the bar
its own deploy was signed off against, and its worst-case start-anchored drawdown breaches
the kill floor the sign-off asserted would never fire.

Two ways to close it, not decided here:

1. **Make live match the validation** — add the adverse-trend exit to v1 by extending
   `strategy_uses_trend_exit()` and adding a v1 branch to `trend_exit_signal()`. Restores
   the model that passes every gate. Cost: a new live code path on a real-money leg, and
   `boot()` flattens open positions so it needs a flat-leg deploy window.
2. **Make the validation match live** — accept the live exit model and re-tune to it
   (the SL/TP geometry and the 1344-bar time stop were never optimised for a
   no-trend-exit system). Cost: a full re-tune plus fresh OOS, and nothing is validated
   until it lands.

Option 1 is the smaller change and restores a known-good configuration; option 2 is the
more honest one if the trend exit turns out to be hard to reproduce faithfully live
(it reads the 15m EMA200 every bar, which the loop already has).

Regardless of which: `tools/multifactor_validate.py` should grow an exit-parity stage,
and its `LOCKED` import from `tools/run_mf_deepening.py` is stale (vol 2.0, funding
0.0005, risk 2.75 vs deployed 1.5 / 0.0015 / 3.5) — re-running it today validates a
config that is not in production.
