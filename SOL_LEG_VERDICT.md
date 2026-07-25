# SOL Leg — Strategy Search — Verdict

**Date:** 2026-07-25
**Asset:** SOL/USDT:USDT perpetual, native 4h
**Harnesses:** `tools/sol_leg_return_search.py`, `tools/sol_leg_confirm.py`,
`tools/sol_leg_blend_confirm.py`
**New strategy module:** `strategy/signals_sol_trend_rider.py`

**Round 2 objective:** rank on return, not win rate (God's instruction).
**Round 3 objective:** *blend* win rate and return — God's follow-up after seeing
round 2's 14% win rate ("this is a heart attack"). Return is still what gets
maximised; it is now maximised only among configs clearing a win-rate floor.

---

# ROUND 5: four-leg comparison + the donchian drawdown, resolved

Harness `tools/leg_comparison.py`. Every leg over 2022-04-01 → 2026-07-25
(4.32 yr), **each at its own deployed/recommended sizing** — so `ret%`/`maxDD%`
compare legs as they would actually run, not strategy quality at equal risk.
WR, PF and the cadence columns are sizing-independent and directly comparable.

| Leg | ret% | CAGR% | maxDD% | WR% | PF | n | **n/yr** | **med gap** | **max gap** | lose streak | days UW | +months |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SOL supertrend** *(candidate)* | **+673.4** | **60.6** | -29.9 | **37.0** | **1.63** | 119 | 27.5 | **12.2 d** | 43 d | **6** | **187** | **51.0%** |
| donchian-v3 (BTC 4h) | +251.7 | 33.8 | -32.9 | 31.9 | 1.45 | 135 | 31.2 | 9.8 d | 51 d | 8 | 872 | 37.3% |
| multifactor-v1 (BTC 15m) | +126.4 | 20.8 | **-24.8** | 30.7 | 1.38 | 218 | 50.5 | 4.4 d | 52 d | 12 | 191 | 47.1% |
| cnh-hybrid-short (BTC 4h) | +38.1 | 7.8 | **-8.7** | **70.0** | **2.02** | 50 | 11.6 | 22.3 d | 130 d | **2** | 410 | 57.1% |
| cnh-hybrid-short (SOL 4h) | +5.5 | 1.3 | -41.4 | 61.4 | 1.11 | 57 | 13.2 | 15.3 d | 116 d | 3 | 1360 | 65.2% |

Sizing: v1 = `params.yaml` risk 3.5% lev 20 · donchian-v3 = `params_donchian.yaml`
risk 2.75% lev 20 · SOL supertrend = risk 4.0% lev 3 · cnh = its own harness,
which compounds a fixed fractional `net_pct` per trade instead of risk-sizing
off a stop (so its DD is a realised-equity floor, not a true peak-to-trough —
intra-trade excursions are invisible to that simulator).

## Summary

* **Order cadence is the answer to "how often does it fire":** v1 is the busy
  one (50/yr, an order every ~4 days), donchian-v3 and the SOL candidate are
  both ~30/yr (every ~10-12 days), cnh-hybrid-short is a sniper (12/yr, median
  22 days apart, and up to **130 days** with nothing to do).
* **cnh-hybrid-short on BTC is the comfort king and the return dog:** 70% win
  rate, PF 2.02, worst losing streak **2**, max DD only **-8.7%** — but CAGR
  7.8%. It is short-only, so it just does not fire much. If drama is the
  problem, this is the shape you like; it will not compound.
* **cnh-hybrid-short on SOL is the worst leg here** — 61% win rate but CAGR
  1.3% and **1,360 days underwater** (nearly 4 years below high-water). High win
  rate with no edge. Consistent with the 2026-07-25 finding that the SOL
  premium had gone; this says don't fund it.
* **donchian-v3 grinds:** best BTC return (CAGR 33.8%) but **872 days
  underwater**, the longest of the BTC legs, and only 37.3% positive months.
* **The SOL candidate leads on return, win rate, PF, losing streak and time
  underwater simultaneously** — but see the round-3 bear-bias caveat, which the
  single-window numbers here do not show.

## The -63.6% donchian drawdown: my error, not your leg

I flagged this last round. It was a bug in **my harness**, not your config.

`StrategyParams.from_yaml()` has no donchian fields in its constructor block, so
loading `config/params_donchian.yaml` through it **silently drops two keys**:

| Key | YAML says | from_yaml gives |
|---|---|---|
| `donchian_period_entry` | **80** | 20 (dataclass default) |
| `slope_trend_threshold_pct` | **0.03** | **0.0 → regime gate OFF** |

That converts the deployed 80-bar gated breakout into a 20-bar ungated one.
Isolating each key:

| Config | ret% | maxDD% | n | n/yr | med gap |
|---|---|---|---|---|---|
| as I ran it in round 4 (from_yaml) | +101.3 | **-63.6** | 454 | 105.1 | 3.2 d |
| + entry 80 only | +178.6 | -49.0 | 219 | 50.7 | 6.2 d |
| + gate 0.03 only | +129.1 | -40.4 | 187 | 43.3 | 6.6 d |
| **DEPLOYED: entry 80 + gate 0.03** | **+233.5** | **-32.9** | **135** | **31.2** | 9.8 d |

**Your live leg is fine.** `strategy/live_donchian_v3.py:119,125` reads the YAML
dict directly (`s.get("donchian_period_entry", ...)`,
`s.get("slope_trend_threshold_pct", ...)`), so the bot runs 80 / 0.03 as
intended. The correct backtest is **-32.9% max DD — inside the -35.5% kill
switch**, with ~2.6pp of cushion, and 31.2 orders/yr matches the repo's
"~26 signals/yr" expectation. The 105/yr churn was the tell that I had it wrong.

**Real bug still open:** `from_yaml` cannot load any donchian config correctly.
Any harness that calls it on `params_donchian*.yaml` measures the wrong system.
`tools/leg_comparison.py::deployed_donchian_params()` is the workaround; the
proper fix is adding the donchian fields to `from_yaml`, which would move
published donchian baselines, so I have not made it.

## Correction to round 4's book numbers

Round 4's book table used the mis-loaded donchian, so it overstated both the
existing book's drawdown and the benefit of adding SOL. Corrected:

| Book | ret% | CAGR% | maxDD% | +months | days UW | monthly vol |
|---|---|---|---|---|---|---|
| BTC only (v1 + donchian-v3) | +207.0 | 29.6 | -22.3 | 45.1% | 253 | 9.8% |
| **+ SOL supertrend** | +219.0 | **30.8** | **-13.5** | **62.7%** | **166** | **6.8%** |
| + SOL st-dual | +191.2 | 28.1 | -13.4 | 60.8% | 166 | 6.5% |

The return benefit is far smaller than I said (CAGR 29.6% → 30.8%, not
25.7% → 39.5%). **The risk benefit is the real case:** max drawdown -22.3% →
-13.5%, monthly vol 9.8% → 6.8%, positive months 45% → 63%, time underwater
253 → 166 days. Correlation is unchanged and still the reason: SOL vs v1
**-0.02**, SOL vs donchian **+0.06**, while v1 vs donchian is **0.43**.

Artifacts: `reports/leg_comparison.json`, `reports/sol_leg_basket_wf.json`.

---

# ROUND 4: the roller coaster is a single-leg illusion

> **Superseded in part by round 5.** The correlation finding stands; the book
> table below used a mis-loaded donchian-v3 and is corrected in round 5.

God's round-3 read: still a roller coaster — is there more on SOL, other coins,
or another BTC leg? Harnesses `tools/sol_leg_basket.py`,
`tools/sol_leg_basket_wf.py`. 13 coins, native 4h, same span, no coin selection
in the headline basket.

## The answer: it is uncorrelated with your BTC book, so it *removes* drama

Monthly-return correlation of the SOL leg against the two legs actually running:

| | SOL supertrend | SOL st-dual | BTC mf-v1 | BTC donch-v3 |
|---|---|---|---|---|
| SOL supertrend | 1.00 | 0.20 | **-0.02** | **0.10** |
| SOL st-dual | 0.20 | 1.00 | **-0.11** | **0.08** |
| BTC mf-v1 | -0.02 | -0.11 | 1.00 | *0.40* |
| BTC donch-v3 | 0.10 | 0.08 | *0.40* | 1.00 |

The SOL leg is ~uncorrelated with both BTC legs. Your two BTC legs correlate
**0.40 with each other** — so a third BTC leg would stack risk, while this one
offsets it. Effect on the book (equal weight, each leg at its own risk):

| Book | ret% | CAGR% | **maxDD%** | pos months | **days underwater** | monthly vol |
|---|---|---|---|---|---|---|
| BTC only (mf-v1 + donch-v3) | +168.5 | 25.7 | **-37.7** | 49.0% | **439** | 11.3% |
| **+ SOL supertrend** | **+321.3** | **39.5** | **-26.4** | 58.8% | **193** | **8.9%** |
| + SOL + ADA | +278.6 | 36.1 | -26.3 | **62.7%** | 194 | 8.7% |
| + SOL + ADA + AVAX | +257.1 | 34.3 | -26.8 | 58.8% | 204 | 8.4% |

Adding one SOL leg raises book CAGR 25.7% → 39.5% **and** cuts max drawdown
-37.7% → -26.4%, monthly vol 11.3% → 8.9%, and time underwater 439 → 193 days.
The lumpiness that looks alarming leg-by-leg is what smooths the portfolio,
because it lands in different months than the BTC P&L.

## More on SOL? No — SOL is the outlier, not a sample

Per coin at risk 2% (`supertrend` geometry). SOL is the best of 13 by a wide
margin, and a third of the field loses money:

| | SOL | ADA | AVAX | DOT | ATOM | LINK | NEAR | BNB | ETH | XRP | BCH | LTC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ret% | **+210** | +144 | +138 | +54 | +52 | +40 | +22 | +21 | -4 | -6 | -27 | -41 |
| WR% | 37.0 | 39.0 | 38.5 | 33.3 | 29.5 | 27.7 | 28.1 | 26.0 | 31.7 | 27.9 | 24.8 | 25.7 |

That the geometry loses on 4 of 12 coins is a real caveat on round 3's
robustness claim: this is not a universal trend edge, it is one that works on
high-beta L1s. Mean monthly correlation between coin legs is only 0.22, so
diversification does work statistically — but the spread in leg *quality*
dominates it.

## Other coins? Only ADA and AVAX, and they buy smoothness, not return

Walk-forwarding the **coin choice** (rank coins by TRAIN return each fold,
equal-weight exactly those in TEST) — so this is not hindsight:

| Variant | chained% | +folds | median% | worst% | ex-best% |
|---|---|---|---|---|---|
| SOL alone | **+141.7** | 7/9 | +6.8 | -4.1 | **+84.6** |
| top-6 by train | +52.4 | **8/9** | +4.4 | -6.0 | +30.4 |
| top-4 by train | +43.4 | **8/9** | +3.6 | -7.1 | +15.3 |
| all 12, no selection | +37.9 | 7/9 | +2.0 | -3.8 | +20.5 |

Coin selection is a real lever (top-6 +52.4% beats all-12 +37.9%, and the same
holds for `st-dual`: +71.8% vs +30.8%), **but no basket beats SOL alone.**
ADA and AVAX were train-picked in 7-9 of 9 folds, so they are persistently good
rather than cherry-picked. What they buy is a calmer ride, at a return cost —
alt sleeves each sized to -30% DD:

| Sleeve | ret% | CAGR% | pos months | days UW | **monthly vol** |
|---|---|---|---|---|---|
| SOL | +676.9 | **60.7** | 51.0% | 187 | **12.4%** |
| SOL+ADA | +529.3 | 53.1 | 56.9% | **159** | 8.7% |
| **SOL+ADA+AVAX** | +441.5 | 47.8 | **58.8%** | 174 | **7.3%** |
| SOL+ADA+AVAX+ATOM+LINK | +335.9 | 40.6 | 56.9% | 159 | 7.5% |

**SOL+ADA+AVAX cuts monthly volatility 41% (12.4% → 7.3%) for 13pp of CAGR.**
If the ride is the problem, that is the trade to make.

## Another BTC leg? No

BTC under these geometries, sized to the same -30% DD: `supertrend` **+4.9%**
(CAGR 1.1%) and `st-dual` **-7.5%** over 4.32 years. BTC's trend edge is already
taken by multifactor-v1 (+119.3%, DD -24.8%, WR 30.7% over this span). Adding
BTC exposure means 0.40 correlation with what you already hold.

## Recommendation

**Run ONE alt sleeve alongside the BTC book — `supertrend` on SOL, or on
SOL+ADA+AVAX if the ride matters more than the return.** Do not build the
13-coin basket (dilutes into 4 losers) and do not add a BTC leg.

## Flag on your existing book (not part of this task)

At deployed params (`config/params_donchian.yaml`: 80/10 channel, gate 0.03,
risk 2.75%, lev 20), **donchian-v3 backtests to -63.6% max drawdown** over
2022-04 → 2026-07 (+101.3%, WR 29.1%, 454 trades), and the two-leg BTC book to
-37.7%. Both exceed the -35.5% kill-switch fraction. My run does not model the
live bot's per-leg HALT, principal anchoring, or limit-order fills, so this is
not a live-equivalent figure — but it is worth a look independently of the SOL
work. Round-4 artifacts: `reports/sol_leg_basket.json`,
`reports/sol_leg_basket_wf.json`.

---

# ROUND 3 DECISION: **`supertrend` long+short**

**Supersedes the round-2 `rider-v1` decision below.** Round 3 dominates round 2
on *every* axis God cares about, at the same drawdown budget.

Config: `st_period=14`, `st_multiplier=3.5`, `st_sl_atr=2.0`, `st_tp_atr=10.0`,
`allow_shorts=True`, **risk ~4.0%/trade**, leverage cap 3x. Native 4h.
(Modal parameter set chosen by the walk-forward under `blend40`.)

| Metric | Round 3 `supertrend` | Round 2 `rider-v1` |
|---|---|---|
| Win rate | **37.0%** | 14.4% |
| **Max consecutive losses** | **6** | **17** |
| **Max days underwater** | **187** | **457** |
| Return (matched -30% DD) | **+662.8%** | +142.2% |
| CAGR | **+60.1%** | +22.7% |
| Profit factor | 1.63 | 1.84 |
| Walk-forward folds positive | **9 / 9** | 6 / 9 |
| Trades/yr | 27.5 | 22.5 |

Both sized to the same -30% max drawdown, so the return column is a fair
comparison rather than a risk-appetite artifact. **9/9 walk-forward folds
positive** (+11 +13 +11 +0 +6 +1 +13 +6 +7 at risk 2%) with parameters
re-selected every fold — no losing 6-month window in 4.3 years. Plateau: 170 of
180 grid points positive (94%), and the chosen point ranks 11th of 180, i.e. it
sits mid-plateau rather than on the peak.

**The load-bearing caveat: this is a bear-biased vehicle.** Short trades
contributed +571% vs longs' +108%, and the per-year split shows why:

| Year | SOL | Leg total | from longs | from shorts | short WR |
|---|---|---|---|---|---|
| 22/23 | -78% | +63.3% | +7.2% | +53.6% | 50% |
| 23/24 | **+804%** | **-5.0%** | +18.5% | -20.7% | 14% |
| 24/25 | -43% | +195.5% | +47.0% | +149.2% | 75% |
| 25/26 | -35% | +12.3% | -15.3% | +29.1% | 39% |

SOL fell 56% net over the test span, and the short side is where the money came
from. In SOL's one huge bull year the leg was flat-to-negative. **Do not
underwrite CAGR 60% — underwrite "makes money when SOL falls, roughly flat when
it rips."** Against a long-biased BTC book that is a genuine diversifier, but it
is a regime bet, not a symmetric edge. The +195% in 24/25 rests on a 75% short
win rate that appears in exactly one of four years.

### Regime-balanced alternative: `st-dual` fast

`st_period=7`, `st_multiplier=2.0`, `st_slow_period=30`, `st_sl_atr=1.0`,
`st_tp_atr=6.0`, shorts on, risk ~2.9%. WF `blend30`, 7/9 folds.

+202.0% / CAGR +29.2% / WR 27.5% / PF 1.30 at the same -30% DD. **Positive in
all four full years (+43.6, +9.1, +55.0, +60.0) with both directions
contributing** — longs carried 25/26 and 23/24, shorts carried 22/23. Best
ex-best-year in the field (+133.6%), so it is the least regime-dependent
candidate. Costs: lower win rate (27.5%), 9-trade losing streak, 388 days
underwater, and it is the most fee-sensitive finalist (+83.6% at 20bps vs +202%
at 5bps, 32 trades/yr).

**Pick `supertrend` for comfort + return; pick `st-dual` fast if you would
rather not bet on SOL continuing to fall.**

### Full matched-drawdown table (all sized to maxDD ≈ -30%)

| Candidate | Provenance | risk% | ret% | CAGR% | WR% | PF | n |
|---|---|---|---|---|---|---|---|
| supertrend L+S | WF blend40 9/9 | 4.02 | **+662.8** | +60.1 | 37.0 | 1.63 | 119 |
| rider-v1 midWR | ⚠ SCAN, in-sample | 3.88 | +248.6 | +33.5 | 35.2 | 1.52 | 105 |
| st-dual fast | WF blend30 7/9 | 2.94 | +202.0 | +29.2 | 27.5 | 1.30 | 138 |
| rider-v1 round-2 | WF ret round-2 | 1.48 | +142.2 | +22.7 | 14.4 | 1.84 | 97 |
| rider-v1 tightTP | WF blend40 5/9 | 2.65 | +85.5 | +15.4 | 40.8 | 1.31 | 142 |
| st-dual slow | WF blend50 7/9 | 3.80 | +69.2 | +12.9 | 40.3 | 1.40 | 77 |
| sol-trend-rider | WF blend40 7/9 | 1.63 | +55.7 | +10.8 | 27.8 | 1.57 | 79 |
| rider-v1 wideSL | WF blend30 6/9 | 1.61 | +46.0 | +9.2 | 21.4 | 1.44 | 84 |
| SOL buy-and-hold 1x | benchmark | — | -56.4 | -17.5 | — | — | — |

`rider-v1 midWR` is flagged: I found it by sweeping the reported span, so its
numbers are in-sample. It is a hypothesis, not a validated candidate. Every
other row's parameters came from walk-forward selection on train windows only.

### Two findings worth keeping

1. **The win-rate floor improved out-of-sample return.** It is a regulariser,
   not a tax: `rider-v1` went from +132.2% (pure return) to **+192.3%** under
   `blend30`. Constraining away lottery-ticket geometry made selection more
   robust. Round 2's premise — that win rate had to be sacrificed for return —
   was wrong.
2. **Widening the stop is why.** Round 2's grids were all narrow-SL / wide-TP,
   which forces win rate down *and* drawdown up. A 2×ATR stop cuts max drawdown
   roughly in half, and a smaller drawdown buys size inside the same kill-switch
   budget. Higher win rate and higher return were never in conflict; the round-2
   grid just could not reach that region. The walk-forward independently
   converged on `sl_atr=2.0` for three separate strategy families.

Cost stress (matched-DD size): supertrend +476% at 20bps; st-dual fast +83.6% at
20bps. BTC control: supertrend -9.9% on BTC (+672pp gap), st-dual fast -26.1%
(+228pp gap) — both SOL-specific.

Round-3 artifacts: `reports/sol_leg_blend_confirm.json`, objectives
`blend30/40/50` in `reports/sol_leg_return_search_oos.csv`.

---

# ROUND 2 (superseded — kept for the record)

## Round-2 decision: `rider-v1` on pure-return ranking

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
