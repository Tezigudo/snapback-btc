# Trailing stop on the snapback legs — verdict (2026-09-23)

**Question (God):** should any leg run a trailing stop, and if so what % / what style is best?

**Answer: no trailing stop on any leg.** Nothing passed the adoption rule written
before any results were seen. One related idea, a one-time move of SOL's stop to
**breakeven at +2R (≈+10% from entry)**, is a real plateau. It passes out of sample
on SOL's full 2020+ history but not on the pre-registered 2022+ span. It is the one
candidate, and adopting it is God's call.

Tool: `tools/trailing_stop_study.py` (research only: no config, bot or droplet
change). Results: `reports/trailing_stop_study.json` (+ `.probe.json`,
`.donchian_timestop.json`, and `.equity.json` for curves).

## Method

- Each leg's **deployed** backtest class gets one mixin. On every bar **close** it
  ratchets the resting stop and never loosens it. The new stop fills intrabar from
  the next bar on. That is what a live bot gets by cancel/replacing its STOP_MARKET
  once per bar. Binance's native `TRAILING_STOP_MARKET` moves intrabar and is **not**
  modelled here.
- Families tested:
  - breakeven at +X R
  - trail in R, with and without an activation threshold
  - ATR chandelier
  - fixed % trail
  - a resting stop at the leg's own exit line (donchian exit channel, SOL Supertrend line)
  - TP removal
- **Parity controls passed exactly** before any arm was trusted:

  | Leg | Return | Trades | PF |
  |---|---|---|---|
  | v1 | +373.2% | 370 | 1.392 |
  | donchian | +1399.02% | 219 | 1.607 |
  | SOL | +515.0% net | 119 | 1.63 |

- Data is the cached set: BTC to 2026-08-11, SOL to 2026-07-25. Each leg keeps its own commission.
- **Pre-registered adoption rule.** An arm must pass all five:
  1. Walk-forward selection beats always-baseline.
  2. Beats baseline on net return **and** on MAR, and wins at least 60% of years.
  3. Its grid neighbours also beat baseline (plateau).
  4. Start-anchored drawdown is no more than 2pp worse, with 0% kill-switch breaches.
  5. Still beats baseline at 2× costs.

## What 1R means in % (so the answers below read in %)

| Leg | Initial stop = 1R | ≈ % of price |
|---|---|---|
| v1 (15m) | `sl_pct` | **1.5%** |
| donchian (4h) | 1.5 × ATR20 | **≈2.1%** (ATR median 1.41%) |
| SOL (4h) | 2 × ATR14 | **≈5.1%** (IQR 4.3–6.4%) |

## Results, net of funding (full period)

### v1 (baseline +346%, maxDD −24.8%, MAR 1.02)

| Arm | Net | MAR |
|---|---|---|
| breakeven 0.5R / 1.0R / 1.25R | +161% / +124% / +286% | 0.66 / 0.53 / 0.93 |
| breakeven **1.5R** | **+370%** | 1.08 |
| breakeven 1.75R | +320% | 0.97 |
| trail 0.5R (0.75%) after +1R | +56% | 0.27 |
| trail 1R (1.5%) after +1R, with / without TP | +105% / +114% | 0.43 / 0.44 |
| trail 1R after +1.5R, no TP | +282% | 0.88 |
| trail 2R (3%) after +1R, no TP | +216% | 0.54, DD −35.3% (at the kill floor) |

**Verdict: no.** Breakeven at 1.5R is a lone spike: 1.25R and 1.75R both lose to
baseline. Every trail loses. v1 is a mean-reversion entry, and trades routinely come
back to entry before reaching the 2R TP. The adverse-EMA exit (62% of exits) already
cuts the losers.

### donchian (as-live baseline +1790%, DD −30.0%, MAR 1.77)

| Arm | Net | DD |
|---|---|---|
| ATR trail 2 / 3 / 4 / 5 / 6 | +135% / +260% / +924% / +888% / +1082% | |
| ATR trail 8 | identical to baseline (never fires) | |
| % trail 3 / 5 / 8 / 12 / 16 % | +208% / +321% / +1151% / +1485% / +1581% | |
| breakeven 1R | +1755% | −39.0% |
| breakeven 2R | +1698% | |
| resting stop at the 10-bar exit channel | +693% | −35.6% (wicks stop it out) |
| trails without the time stop | all worse | |

**Verdict: no.** The 10-bar exit channel already *is* the trailing stop. Every
tighter trail cuts the trend trades that pay for the strategy: the top 5 trades carry
more than 100% of the P&L.

⚠️ **Live/backtest gap found.** `DonchianBreakoutBTCv3.next()` ignores `time_stop_bars`,
but the live bot closes at 48 bars (`_maybe_time_stop`). So the published +1399%
(+1244% net) is **not what runs**. Modelled as live: +1790% net, DD −30.0%.

Time-stop sweep:

| Bars | Net |
|---|---|
| 24 | +661% |
| 36 | +1129% |
| 42 | +1745% |
| **48** | **+1790%** |
| 54 | +1424% |
| ≥60 | never fires (= +1244%) |

It only touches about 11 of 222 trades, so it is a narrow bump, not an edge. It says
the live exit is not worse than the backtest. It is not a reason to tune the time stop.

### SOL (baseline +515%, DD −26.6%, MAR 1.97)

| Arm | Net | Notes |
|---|---|---|
| ATR trail 2 / 2.5 / 3 / 3.5 / 4.5 | −1% / +29% / +44% / +120% / +265% | |
| % trail 5 / 8 | +28% / +148% | |
| % trail 12 | +561% | lone spike |
| % trail 16 / 20 | +361% / +366% | |
| resting stop at the Supertrend line | +183% | DD −37.8%, breaches kill floor |
| remove the 10-ATR TP | +82% | |
| **breakeven 1.5 / 1.75 / 2 / 2.25 / 2.5 / 3R** | **+778 / +762 / +838 / +831 / +778 / +721%** | MAR 2.30–2.57, DD ≤ −28.4% |
| breakeven 1R | +418% | |
| breakeven 4R | = baseline | |
| breakeven 2R at 2× costs | +762% | baseline at 2× costs: +467% |

Breakeven is a real **in-sample plateau**: every level from 1.5R to 3R (+7.7% to
+15% of price) beats baseline. Most of the gain is on the **long** side, where long
P&L triples. It turns SOL longs that run and then fail into scratches.

**Out of sample, on the pre-registered span (2022-04+): not confirmed.**

| Test | Picked | Baseline |
|---|---|---|
| Pre-registered WF, picking among *all* arms | +134% | +236% |
| Post-hoc, breakeven-only WF, 18-month train | +242% | +236% (lost 4 of 6 folds) |
| Post-hoc, breakeven-only WF, 12-month train | +326% | +323% (lost 4 of 7 folds) |

The pre-registered test fails. The post-hoc ones are a tie.

**Post-hoc extension to SOL's full cached history (2020-10-01 → 2026-07-25)**, 9–10
folds instead of 6 (`reports/trailing_stop_study.sol_be_2020.json`). This is **not**
the +515% parity span. One harness change was needed: the 10-ATR short TP goes
negative in 2020–21, so a non-positive TP is treated as "no TP". Parity was
re-verified unchanged.

Full period, net of funding:

| Arm | Net | At 2× costs | Years beating baseline | Max DD |
|---|---|---|---|---|
| baseline | +918% | +824% | — | −26.6% |
| breakeven 1R | +630% | | | |
| breakeven 1.5R | +1090% | | 4/7 | |
| breakeven 1.75R | +1534% | +1374% | 6/7 | −26.5% |
| breakeven 2R | +1344% | +1204% | 5/7 | −26.5% |
| breakeven 2.25R | +1370% | | 4/7 | |
| breakeven 2.5R | +1299% | | 3/7 | |
| breakeven 3R | +1511% | | 3/7 | |
| breakeven 4R | = baseline | | | |

No kill-switch breach anywhere.

Walk-forward, re-picking the breakeven level (including "none") each fold:

| Train window | Picked | Baseline | Folds won / lost / tied |
|---|---|---|---|
| 12 months | **+850%** | +724% | 4 / 2 / 4 |
| 18 months | **+678%** | +544% | 4 / 2 / 3 |

The losing folds are 2023-10 (−8.6pp) and 2024-10 (−24pp); 2026-04 is about −4pp.

On the longer history it clears all five gates. The one borderline is 2.25R, at
4 of 7 years.

**Standing: the strongest lead in the study, but found by looking.** It was spotted
in-sample on 2022+ and then confirmed on a span that overlaps it. It rests on about
160 trades, and 85–90% of the P&L sits in the top 5 trades. Breakeven rarely fires,
which is why most walk-forward folds tie. The downside is bounded: it only acts after
a trade reaches +2R, and it never widened drawdown in any test.

## Recommendation

1. **Do not add a trailing stop to any leg.** On donchian and SOL the existing exit
   (channel / Supertrend flip) already trails. Any tighter trail cuts the fat tail
   both legs live on. On v1, trails turn eventual TP hits into scratches.
2. **SOL breakeven at +2R (≈+10% from entry) is the only candidate. It is God's
   call.** It passes on the longer history and fails on the pre-registered one. If
   adopted, the build is:
   - a **one-shot** SL move per position: cancel/replace the algo STOP once, at the
     first 4h close with MFE ≥ 2R
   - new bot code on a leg with a history of bracket/boot bugs, so it needs the
     usual review + `leg_safe_restart.py`

   Alternative: have the bot log the move without making it, for ~30 trades first.
3. **The donchian time-stop gap should be written down, not "fixed".** Research
   harnesses for donchian should add the 48-bar time stop so their baselines match
   live.

   Checked: live enters on the **level** (`cur_close > upper`,
   `live_donchian_v3.py:173`), with no cooldown after a bot close. So re-entry after
   a time stop matches the backtest, except that live can re-enter up to one bar
   earlier, because the in-position early return never advances `_last_signal_ts`.

## Caveats

- Stops are modelled as bar-close ratchets filled intrabar. A 15m/1m sub-bar test
  would be needed for Binance's native trailing order.
- SOL trades: n = 119 over 4.3 years, with the top 5 trades carrying 85–90% of P&L.
  Any SOL exit change has wide error bars.
- Walk-forward selection runs on one continuous equity curve per arm. Positions open
  at a fold boundary are carried, not re-simulated.
