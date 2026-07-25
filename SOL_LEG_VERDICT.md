# SOL Leg — Return-Ranked Strategy Search — Verdict

**Date:** 2026-07-25
**Asset:** SOL/USDT:USDT perpetual, native 4h
**Objective (per God's explicit instruction):** rank on **return**, not win rate
**Harnesses:** `tools/sol_leg_return_search.py`, `tools/sol_leg_confirm.py`
**New strategy module:** `strategy/signals_sol_trend_rider.py`

---

## DECISION: **PROMOTE `rider-v1` to paper-trade as the SOL leg**

Config: `rider_donchian_n=34`, `rider_ema_period=200`, `rider_sl_atr=1.0`,
`rider_tp_atr=12.0`, `rider_trail_atr=0.0`, **risk 1.5%/trade**, leverage cap 3x.

Frozen-param continuous run, 2022-04-01 → 2026-07-25 (4.32 yr), 5 bps/side + funding:

| | |
|---|---|
| Return | **+144.7%** |
| CAGR | **+23.0%** |
| Max drawdown | **-30.4%** (inside the -35.5% kill switch, ~5pp cushion) |
| Profit factor | 1.84 |
| **Win rate** | **14.4%** — 1 win in 7 |
| Trades | 97 (~22/yr) |
| SOL buy-and-hold, same span | **-56.4%**, maxDD -93.6% |

This is not beta. SOL *lost* 56% over the test span; the leg made 145%.

**Runner-up, and it is close:** `sol-trend-rider` at risk 1.25% returns
**+155.8%** (CAGR +24.3%, maxDD -31.1%, PF 1.89, WR 19.8%, n=106). At *matched
drawdown* the two are within 11pp over 4.32 years — effectively tied. `rider-v1`
gets the nod on walk-forward consistency and kill-switch cushion, not on return.
See the head-to-head below; running both is a legitimate option.

---

## Why this and not the bigger headline number

Two candidates print more than `rider-v1` on compounded walk-forward return.
Both fail the concentration test — delete each candidate's single best 6-month
fold and see what is left:

| Candidate | Compounded OOS | +folds | Median fold | **Ex-best-fold** | Median PF |
|---|---|---|---|---|---|
| st-donchexit::long | +194.7% | 4/9 | **-0.66%** | +27.8% | 0.86 |
| st-adx-donchexit | +135.7% | 4/9 | **-4.82%** | **-0.0%** | 0.66 |
| **rider-v1** | **+132.2%** | **6/9** | **+6.54%** | **+57.0%** | **1.37** |
| sol-trend-rider | +123.5% | 6/9 | +3.45% | **-0.9%** | 1.07 |
| st-adx | +60.2% | 5/9 | +2.90% | -1.5% | 0.82 |
| st-dual | +59.9% | 5/9 | +1.25% | -0.7% | 1.02 |
| donchian-v3 | +19.5% | 5/9 | +4.80% | -21.1% | 1.06 |

`rider-v1` is the only candidate in this table whose walk-forward return
survives losing its best window. The others collapse to roughly zero — their
compounded return *is* the single 2023-10 → 2024-03 SOL explosion (SOL B&H did
+790% in that one fold).

**Caveat on that table:** those rows re-select parameters every fold, so they
measure *strategy + selection process* together. `sol-trend-rider`'s -0.9% is
mostly a selection failure, not a strategy failure — return-max selection kept
switching its optional donch/trail exits between folds and picked badly. With
parameters **frozen**, it holds up far better. The fair comparison is below.

### Frozen head-to-head (no reselection, matched risk)

Both configs frozen, run on each of the 9 walk-forward test windows:

| risk | Candidate | Per-window returns (%) | Chained | Ex-best-fold | +folds |
|---|---|---|---|---|---|
| 1.25% | sol-trend-rider | -8 +26 -3 **+98** +1 +33 +7 -17 -3 | +152.7% | +27.9% | 5/9 |
| 1.25% | rider-v1-alt | -12 +2 +9 **+42** +7 +25 +19 -7 -4 | +97.6% | **+39.3%** | **6/9** |
| 1.50% | sol-trend-rider | -10 +31 -4 **+116** +1 +40 +8 -20 -4 | +184.2% | +31.6% | 5/9 |
| 1.50% | rider-v1-alt | -15 +2 +11 **+51** +8 +30 +23 -8 -5 | +121.4% | **+46.4%** | **6/9** |

### Continuous run, matched risk

| risk | Candidate | Return | CAGR | Max DD | PF | WR | n |
|---|---|---|---|---|---|---|---|
| 1.25% | sol-trend-rider | **+155.8%** | +24.3% | -31.1% | 1.89 | 19.8% | 106 |
| 1.25% | rider-v1-alt | +116.2% | +19.5% | **-25.9%** | 1.84 | 14.4% | 97 |
| 1.50% | sol-trend-rider | **+188.2%** | +27.8% | **-35.9% ✗** | 1.89 | 19.8% | 106 |
| 1.50% | rider-v1-alt | +144.7% | +23.0% | -30.4% ✓ | 1.84 | 14.4% | 97 |

### The two robustness tests disagree — and why

| Measure | rider-v1-alt | sol-trend-rider | Winner |
|---|---|---|---|
| Ex-best **fold** (frozen, @1.5%) | +46.4% | +31.6% | rider-v1 |
| Positive folds | 6/9 | 5/9 | rider-v1 |
| Ex-best **year** (frozen) | +7.3% | **+30.2%** | sol-trend-rider |
| Negative years (of 5 slices) | 2 | 2 | tie |
| SOL−BTC gap | +100.3pp | +260.0pp | sol-trend-rider |
| Runs at 1.5% inside kill switch | ✓ (-30.4%) | ✗ (-35.9%) | rider-v1 |

They disagree because they cut time differently: `sol-trend-rider` spreads its
gains more evenly across *calendar years* but has three losing 6-month folds;
`rider-v1` has fewer losing folds but leans harder on one year.

**Tie-break → `rider-v1`,** on two operational grounds rather than return:
it fits at 1.5% risk with ~5pp of kill-switch cushion (sol-trend-rider must be
sized down to 1.25%, which erases most of its 40pp return lead), and 6/9
positive folds is the better proxy for "deploy it and let it run through
regimes you have not seen."

For the record, the earlier `rider-v1` variant `n=34 / ema=100` at risk 2.0%
has ex-best-year **-27.6%** — the ema-200 filter, not the channel, is what makes
the recommended config's return hold up across years.

---

## Search scope

- **16 candidates**, **1,148 parameter combinations**, **10,332 train
  backtests** + 288 out-of-sample runs.
- Walk-forward: 18mo train / 6mo test / 6mo step → 9 folds, OOS span
  2022-04-01 → 2026-07-25 (the last fold is a 3.8-month partial).
- Per fold: grid-search on TRAIN maximising compounded return after fees +
  slippage + funding, subject to ≥8 trades; freeze; run once on TEST.
- Risk normalised to **2.0%/trade at 3x for every candidate** so the ranking
  measures strategy, not sizing.
- Friction: 4 bps taker + 1 bp slippage per side, plus real funding history.
- Second objective (`ret_dd`, same but rejecting train max-DD worse than -45%)
  changed only `donchian-v3`'s rank — the ranking is not sensitive to it.

Candidates: `supertrend` ±long-only, `st-adx` ±long, `st-ema`, `st-dual`,
`st-donchexit` ±long, `st-voladapt`, `st-adx-donchexit` ±long, `st-trail` ±long,
`rider-v1`, `donchian-v3`, `sol-trend-rider`.

---

## Confirmation gates (all on the frozen deploy config)

**1. Cost stress — passes to 25 bps/side.** The gate
`BTC_SOL_PORTFOLIO_VERDICT.md` flagged as never-measured.

| bps/side | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| Return | +144.7% | +130.3% | +116.7% | +103.9% | +91.9% |
| PF | 1.84 | 1.77 | 1.71 | 1.65 | 1.59 |

By contrast `donchian-v3` on SOL dies at 15 bps (+4.5%, PF 0.99) — it is a
fee-fragile 514-trade churner and is **not** a SOL candidate.

**2. Parameter plateau, not a spike.** Full 72-point `rider-v1` grid re-run on
the continuous span: **67/72 positive (93%)**, median +59.9%. The 5 losers are
all `rider_donchian_n=55` — the only real cliff is "channel too slow for SOL".

**3. Cross-asset control — the edge is SOL-specific.** Identical params on BTC:

| | SOL | BTC | Gap |
|---|---|---|---|
| rider-v1 (deploy cfg) | +144.7% | +44.4% | **+100.3pp** |
| st-donchexit::long | +250.4% | -9.7% | +260.0pp |
| donchian-v3 | +158.4% | +142.6% | +15.8pp |

`donchian-v3`'s SOL result is ~the same as its BTC result — adding it on SOL
would duplicate an already-deployed leg. `rider-v1` is genuinely additive.

**4. Risk ladder — 1.5% is the frontier.** Kill switch fires at -35.5%:

| risk/trade | 0.75% | 1.00% | 1.25% | **1.50%** | 1.75% | 2.00% |
|---|---|---|---|---|---|---|
| Return | +64.0% | +89.2% | +116.2% | **+144.7%** | +174.3% | +204.8% |
| Max DD | -16.3% | -21.2% | -25.9% | **-30.4%** | -34.6% | -38.6% |

1.75% leaves only 0.9pp of kill-switch cushion. **1.5% is the recommendation**;
1.25% if you want the donchian-v3 leg's DD profile.

**5. Annual behaviour — bull amplifier, bear survivor.**

| Year | Leg | SOL B&H |
|---|---|---|
| 2022-04 → 2023-04 | -16.9% | -78.4% |
| 2023-04 → 2024-04 | +203.5% | +803.8% |
| 2024-04 → 2025-04 | +23.8% | -42.9% |
| 2025-04 → 2026-04 | +9.7% | -34.5% |
| 2026-04 → 2026-07 (partial) | -4.8% | -11.8% |

Beats B&H in 4 of 5 slices; only one negative full year, and that year SOL fell
78%. The two flat-to-down SOL years (2024-25, 2025-26) were both positive for
the leg — that is where the alpha actually shows.

---

## Finding: `st-donchexit`'s Donchian exit is dead code

Investigating why `st-donchexit::long` topped the raw ranking turned up a bug
that also affects the **BTC** bake-off in `EXTENDED_WALKFORWARD.html`.

`SupertrendDonchExit.next()` (and `SupertrendADXDonchExit`) read
`self.data.STDonchLower[-1]` **unshifted**. `donchian_channel()` includes the
current bar and its own docstring says the caller must shift by 1. So the long
exit test `close <= rolling_min(low, N)` can only be true when the close equals
the window's lowest low. Measured on the SOL 4h frame:

| donch period | 3 | 20 | 30 |
|---|---|---|---|
| Unshifted `close <= lower` fires | **0** / 9,457 bars | **0** | **0** |
| Correctly shifted, fires | 1,284 | 422 | 331 |

Zero fires at period 3, where a 3-bar low would obviously bite. The exit never
runs. `st_tp_atr` is separately and deliberately unused in that class. So
`st-donchexit` long-only is really *"Supertrend flip entry, 1×ATR stop, **no
take-profit**, exit only on the trend flipping back"* — and that accidental
configuration beat every deliberately-parameterised variant on SOL.

`strategy/signals_sol_trend_rider.py` states that geometry explicitly (parity
verified exact: +155.8% / -31.1% / PF 1.89 / n=106 at risk 1.25%) and adds the
exits the variant *meant* to test, shifted correctly. Result — **the bug was
accidentally beneficial**; every tighter exit clips winners:

| Variant (risk 1.25%) | Return | Max DD | PF |
|---|---|---|---|
| A — flip-exit only (the accidental winner) | **+155.8%** | -31.1% | 1.89 |
| B — + shifted Donchian exit, N=10 | +56.3% | -28.6% | 1.32 |
| B — + shifted Donchian exit, N=20 | +143.3% | -31.1% | 1.84 |
| B — + shifted Donchian exit, N=30 | +160.8% | -31.1% | 1.90 |
| C — + chandelier trail 3×ATR | +2.5% | -28.8% | 0.98 |
| C — + chandelier trail 5×ATR | +160.0% | -31.1% | 1.90 |
| D — longs **and** shorts | +103.1% | -35.6% | 1.28 |

Two transferable lessons for SOL: **do not take profit on a SOL trend**, and
**do not short SOL trends** (adding shorts cost 53pp and breached the kill
switch).

**Recommended follow-up (not done — needs your call):** the shift bug is in
research code, not the deployed legs, but it invalidates the `st-donchexit` and
`st-adx-donchexit` rows of the BTC extended walk-forward. Fixing it changes
published BTC baselines, so it is left untouched pending your decision.

---

## Caveats

1. **Return is concentrated even for the winner.** 2023-04→2024-04 alone is
   +203.5% of the +144.7% total. Ex-best-year is **+7.3%** — positive, but thin.
   This is a bull-market amplifier that survives bears, not an all-weather leg.
   `sol-trend-rider` is materially better on this specific axis (+30.2%); if
   year-level evenness matters more to you than fold count, it is the better
   pick and the recommendation flips.
2. **14.4% win rate.** ~6 losers per winner, and the losing streaks are long.
   This is the shape you asked for, but it is operationally punishing to watch.
3. **`rider_ema_period=200` on 4h with no pre-roll.** `_prepare_tf_agnostic_data`
   deliberately sets `warmup_bars = 0` for rider (matching the standalone
   validator), so EMA200 needs ~33 days of the visible window before any signal
   can fire. Harmless on the continuous run; it eats ~18% of each 6-month fold,
   making the fold results mildly conservative.
4. **Walk-forward param instability.** `rider-v1`'s per-fold picks move within a
   tight family (`n ∈ {20,34}`, `ema ∈ {100,200}`, `sl ∈ {1.0,1.5}`,
   `tp ∈ {8,12}`) rather than jumping around, and the 93% plateau says the
   family is flat — but the exact deploy triple is a modal choice, not a
   uniquely-selected one.
5. **Wide short TPs are unreachable on early SOL.** 96 of 1,296 `supertrend`
   train runs errored in folds 0-1 (`price < 0`: a 10×ATR short target below a
   $2 SOL). Those combos were skipped in those folds, so long+short candidates
   searched a slightly smaller grid there than long-only ones. Does not affect
   the winner (long-only).
6. **No portfolio check.** Return correlation of this leg against the live
   BTC v1 and donchian-v3 legs was not measured. Do that before sizing it
   inside the shared book.

---

## Next steps if you want it live

1. Paper/dry-run: needs a `strategy/live_rider_v1.py` (live analogue of
   `live_donchian_v3.py`) — does not exist yet. That is the real build cost.
2. Sub-account + `.env.sol_rider`, per-leg HALT (the 2026-07-01 cascade lesson).
3. Measure return correlation vs the BTC legs before allocating capital.
4. Suggested first funding: small. ~22 trades/yr means the sample builds slowly.
5. **Consider paper-running `sol-trend-rider` alongside it.** Different entry
   signal (Supertrend flip vs Donchian breakout), similar cadence, and the pair
   disagree about which windows they win in — so the combination may be steadier
   than either alone. Not measured yet; both are long-only SOL, so the combined
   book would double SOL exposure and needs correlation work first.

Artifacts: `reports/sol_leg_return_search_oos.csv`,
`reports/sol_search_train/*.json`, `reports/sol_leg_confirm.json`.
No live config changed. No orders placed. `risk.py` untouched.
