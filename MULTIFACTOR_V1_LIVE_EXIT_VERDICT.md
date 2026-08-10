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

---

## RESOLUTION — Option 1 taken, 2026-08-10

Branch `fix/v1-adverse-trend-exit`. **Code complete and tested; NOT yet deployed** —
`boot()` flattens open positions and v1 has been LONG 0.005 @ 64,911.2 since
2026-08-08 23:30 UTC, so the restart must wait for a flat leg.

**What changed**

| file | change |
|---|---|
| `strategy/live_multifactor_v1.py` | new `trend_exit_signal_multifactor_v1()` — the pure-function port of `next()`'s adverse-EMA branch |
| `bot_internals.py` | `strategy_uses_trend_exit()` gains `multifactor-v1`; `trend_exit_signal()` dispatches to it; new `trend_exit_fill_reason()` |
| `bot.py` | the trend-exit hook records the per-strategy fill reason and carries `rule`/`trend_ema` in the exit payload |
| `config/params.yaml` | comments corrected — the kill-switch "never fires" claim and `require_trend` both described a model that wasn't running |
| `tests/test_v1_trend_exit.py` | 24 tests, incl. bar-for-bar exit parity vs the backtest branch |
| `tests/test_channel_exit.py` | `test_noop_for_non_donchian_strategy` was parameterised on v1 — re-pointed at legs that genuinely have no trend exit, plus a positive v1-reaches-the-hook test |

Full suite: **346 tests, 0 failures, 1 pre-existing skip.**

**The parity gap that hid this is now closed in the suite.** The old harness compared
entry signal bars only; `TestTrendExitParity` compares the EXIT decision bar-for-bar
against a reference rebuilt the way `init()` builds `_trend_ema`, so a change to the
backtest's indicator fails the test instead of silently re-opening the divergence.

**Two behaviours deliberately preserved rather than "fixed":**

- A NaN close HOLDS. `next()` guards only `t`, so a NaN close falls through both strict
  comparisons and the backtest holds too. Labelled `nan_close` for observability; the
  behaviour is unchanged and matches.
- donchian-v3 and supertrend keep the `channel_exit` fill reason. It is already in the
  fills table and the consolidate fixtures; v1 gets a new `trend_exit` value instead of
  renaming shared history. `exitReason` is a pass-through string consumer-side with no
  enum validation, so the new value renders as-is.

**Known divergence, documented not closed:** backtesting.py executes `position.close()`
at the next bar's open; the bot closes at market as soon as it sees the triggering CLOSED
bar. Sub-bar timing only — the decision is identical. Shared with the donchian channel exit.

**Tooling debt CLOSED, same day.** `tools/multifactor_validate.py` grew the two things
the section above asked for:

- **stage 3 — exit parity.** Compares `next()`'s in-position branch against
  `trend_exit_signal_multifactor_v1` bar by bar over the stage-2 window, probing a
  hypothetical long AND short at every bar, in two shapes: `full_prefix` (algebraic,
  gate 100%) and `rolling_1500` (the shape `bot.py` actually fetches, whose EMA is
  seeded at the window start rather than at genesis — gate 99.5%). A `non_vacuous`
  guard fails the stage if fewer than 10 exits fire on either side, so it cannot pass
  by seeing nothing. **Result on 2025-04-01..06-30: 100.0000% on both shapes**,
  3,598 long / 4,939 short exit bars over 8,537 evaluable bars — so the rolling fetch
  window introduces ZERO decision drift on real data, not merely tolerable drift.
- **stage 0 — params provenance.** Params now come from `config/params.yaml`, the file
  the bot reads; `LOCKED` is imported for COMPARISON ONLY and its divergence is reported.
  It found the three keys this verdict flagged: `volume_multiple` 2.0→1.5,
  `funding_extreme_threshold` 0.0005→0.0015, `risk_per_trade_pct` 2.75→3.5. Stages 1–2
  now run on the deployed config too, and keys a Strategy class can't accept are
  reported as `deployed_keys_dropped_*` rather than silently filtered.

Overall verdict on the refreshed run: **PASS** (stage 0 PASS / 1 PASS / 2 100% / 3 100%).

`tests/test_multifactor_validate_wiring.py` (14 tests) pins the wiring, because the
lesson of both this bug and the supertrend flip-exit bug is that **the suite never runs
tools** — a checker nobody checks is how a green run stays green while the thing it
covers is broken. Mutation-tested: reverting stage 0 to `DEPLOYED = LOCKED` fails with
`assert 2.75 == 3.5`, and deleting the recorder's `require_trend` guard fails too.
