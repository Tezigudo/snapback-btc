# Consecutive-loss breaker: should `MAX_CONSECUTIVE_LOSSES` be wired?

Status: **PLAN ONLY.** Written 2026-09-29. No code, config, droplet or git state changed.

## 0. Recommendation

**Do not wire it as a trading breaker, on any leg.**
- On v1 and SOL, loss streaks occur at the rate independent coin-flips predict.
- donchian has somewhat *more* ≥4-loss streaks than chance (18 vs 14.5 expected), but the trade
  after those streaks is **better** than average, not worse.

So there is no exploitable "keep losing" dependence on any leg. On the two trend legs the breaker's
main effect is to lock out the recovery.

Instead:

1. **Resolve the dead constant.** Either delete `MAX_CONSECUTIVE_LOSSES` from `risk.py` or re-label
   it as unenforced. The same applies to the `consecutive_losses` meta key documented in the
   `exchange/state.py` schema docstring (line 8), which nothing writes or reads either. Today it reads as a safety net that does not exist. This is a `risk.py` edit,
   so it needs `RISK_REVIEW=1`.
2. **Optionally, add an alert-only streak monitor** that never blocks, with a per-leg threshold
   set **above** the leg's backtest maximum streak. That is the one streak length that actually
   carries information: "this is worse than anything in 6 years of history".

### Correcting the premise

`MAX_CONSECUTIVE_LOSSES = 4` lives in **`risk.py` → `RiskCeilings`**, the hard-ceiling module
headed "DO NOT EDIT without git env RISK_REVIEW=1". It is **not** in `config/*.yaml`. Its comment
says "Trip kill-switch after this many losses in a row". The kill switch in this bot flattens,
touches `HALT_<leg>` and exits, and a human then has to remove the file. The comment gives **no
reset rule**, so its semantics are underspecified. The closest reading is a halt that latches until
a human intervenes. §2b models that with an explicit, stated assumption. A grep of `bot.py` on
`main`, `droplet` and `f83b8a3` finds no reader.

---

## 1. Constraints

- **Streaks are expected.** Low win-rate trend legs *must* produce long losing streaks. v1 wins
  30% of trades, so any four-trade window is all-loss 0.7⁴ = 24% of the time.
- **The input would be the least reliable data in the system.** A streak count needs a correct
  per-trade win/loss. The leg `fills` tables have known holes: the donchian 2026-09-04 close was
  never written (its backfill, event 17188, is not in the leg DB), exits during downtime are never
  written, and infrastructure closes (`boot_flatten`, time-stop-zero) are not strategy outcomes.
  Memory records the rule: "reconcile via exchange income, not `fills.pnl_usd`".
- **Two breakers already exist, and both are principled.** The per-leg kill switch at
  `kill_switch_equity_fraction: 0.645` vs principal caps cumulative ruin. The daily-loss breaker
  (4.5%, droplet only) blocks entries for a UTC day and was sized deliberately to tolerate exactly
  one full SL. A streak breaker adds a third, and unlike those two it has no ruin argument behind it.
- **v1 live ≈ backtest.** The 2026-09-23 review: −17.5% live vs −15.3% backtest since 07-12, and
  13/13 entries match. The live losing streak is the strategy behaving as modelled in its first
  losing backtest year. It is not a broken leg.

---

## 2. Evidence

**Method.** `tools/trailing_stop_study.py`'s parity-checked harness was used to regenerate each
leg's deployed baseline trade list locally from the cached parquet, with the network
monkey-patched off. Parity: v1 **370 trades / +373.2%**, matching the published baseline on return and trade count.
Its profit factor reproduced at **1.309 vs the published 1.392** (`trailing_stop_study.py:31`).
That gap is **unexplained**, most likely a different PF definition, but not verified. The
streak statistics use only per-trade sign and return, so they don't depend on PF.
donchian (as-live, 48-bar time stop) 226 trades / +2205%. SOL 119 trades / +527% pre-funding, vs
the published 515% after funding. Per-trade return is `PnL / equity-before`. Breaker arms were then
applied **ex-post** to that sequence. The scratch script was not committed. The method is fully
stated here so `engineer` can reproduce it as `tools/consecutive_loss_study.py` if God wants it
on record.

### 2a. No exploitable clustering

Streaks of ≥4 losses, observed vs expected if every trade were an independent draw at the leg's
own win rate (≈ n·p·(1−p)⁴):

| leg | trades | win rate | streaks ≥4, observed | expected if independent | longest streak |
|---|---|---|---|---|---|
| v1 | 370 | 30.0% | **25** | **26.7** | 12 |
| donchian | 226 | 34.1% | 18 | 14.5 | 8 |
| SOL | 119 | 37.0% | 6 | 6.9 | 6 |

P(next trade wins | previous k all lost), with a 2,000-shuffle permutation p-value (one-sided,
"lower than chance"):

| leg | uncond. | after 2L | after 3L | after 4L | mean return of trade after 4L | perm p (k=4) |
|---|---|---|---|---|---|---|
| v1 | 0.300 | 0.264 | 0.269 | 0.255 | **+0.45%** (uncond. +0.49%) | 0.12 |
| donchian | 0.341 | 0.383 | 0.414 | **0.529** | **+5.26%** (uncond. +1.77%) | 0.99 |
| SOL | 0.370 | 0.463 | 0.409 | 0.462 | +1.46% (uncond. +1.84%) | 0.73 |

- **v1** shows a slight, non-significant dip after losses (the best is p = 0.077 at k = 2, which
  does not survive testing five k's). The trade after a 4-streak still has **positive expectancy,
  about equal to average**. Skipping it deletes an average trade.
- **donchian** is mixed. It has more ≥4 streaks than chance (18 vs 14.5), so losses do bunch
  somewhat. But the trade *after* a losing run is better, not worse: after 4 losses the next trade
  wins 53% of the time and averages +5.3%, three times the leg's mean.
  **Caveat:** that 53% rests on 34 **overlapping** observations drawn from only 18 streaks (a
  6-streak contributes three k=4 windows), so the effective sample is closer to 18 and the
  permutation p-value is optimistic. The direction is suggestive, but it is not proof of
  anti-clustering. This is the trend-follower signature: chop produces
  the losing run, and the breakout that ends the chop is the big winner. **A breaker here is
  aimed precisely at the recovery.**
- **SOL** has too few events (13 at k = 4) to say anything, and leans the same way as donchian.

### 2b. Simulated breaker arms

Total return over the full backtest (baseline in the header), for three arm families at trip
thresholds K = 3 / 4 / 5 (these are the arms that correspond to real designs):

| arm | v1 (373%) | donchian (2205%) | SOL (527%) |
|---|---|---|---|
| skip next 2 trades | 190 / **440** / 280 | 766 / 327 / 1846 | 292 / 587 / 404 |
| pause 7 days | 392 / 365 / 376 | 2845 / 2496 / 2349 | 680 / 625 / 574 |
| **latch until a (shadow) win**, *assumed* reset rule (see below) | **210 / 226 / 170** | **370 / 412 / 957** | 299 / 436 / 394 |

These are read against the trailing-stop study's adoption rule: beat baseline on return **and**
drawdown, **and** have both K-neighbours also beat it. That rule was pre-registered for the
trailing-stop study. Here it was **applied post hoc**, after the results had been seen. The verdict
does not depend on that, because the latch and skip-N arms fail by wide margins.

- **Latch: fails everywhere, badly, under a stated assumption.** `risk.py:46` says only "trip
  kill-switch" and gives no reset rule. This arm **assumes** the breaker releases on the first
  winning *shadow* trade after it trips. By construction, that skips each streak's ending winner,
  which is the worst case for a latch. A human-reset latch could release earlier (losing less) or
  later (losing more). Treat the numbers as the cost of that specific rule, not of the constant.
  Under it, the latch cuts v1's
  return by 40%, donchian's by 81% and SOL's by 17%, and makes drawdown *worse* on v1 and donchian
  (v1 −27.2% vs −23.8%, donchian −33.9% vs −27.3%). With K = 4 on v1 it would have tripped 25
  times in 6.5 years, about 4 times a year, including 2026-04-23 and 2026-07-26. Each trip would
  have been a manual HALT removal.
- **Skip-N: no plateau.** v1 "skip 2 at K = 4" looks good (+440%), but its neighbours K = 3
  (190%) and K = 5 (280%) are both far below baseline. That is a lone peak, i.e. noise. On
  donchian, skip-N destroys 15–85% of the return at every K.
- **Pause 7 days: the only arm that doesn't hurt, and it isn't evidence.**
  - On donchian and SOL it "wins" by skipping just 2–16 trades, which is too few to separate
    from luck.
  - On v1 it is flat on return (365–392% vs 373%) with a ~3pp shallower drawdown.
  - **The simulation flatters pauses in two ways.**
    - It gates on each trade's **ExitTime**, not EntryTime. A trade that *entered* during the
      pause but exited after it counts as taken, although a real entry-blocking breaker would have
      refused it.
    - A skipped trade can't free up flat time for a different signal.

    Treat the pause numbers as an upper bound.
  - A mechanism worth ~0 in return and ~3pp of drawdown on one leg does not pay for:
    - a third breaker on the hot entry path
    - an input (the streak count) that depends on the unreliable `fills` ledger
    - a `RISK_REVIEW` change

### 2c. The live record argues the same way

- **SOL:** its first entries were a losing run, but 3 of its 5 entries died to infrastructure
  (time-stop-zero, 2× `boot_flatten`), not to signals. A fills-based K = 4 breaker would have
  tripped on **operations noise** and then held the leg out of the market. The two real strategy
  trades are n = 2 (memory: "do not re-open the leg question").
- **v1:** a losing streak that matches its backtest trade-for-trade is not a malfunction for a
  breaker to catch. The backtest says 2026 is its first losing year, and that recoveries after
  streaks carry average expectancy. A breaker tripping now would lock v1 out of exactly the trades
  that historically end a drawdown.
- **The real risk a streak rule is reaching for** is "the edge has died". The tool for that already
  exists and was run on 2026-09-23: a live-vs-backtest parity review. A streak count is a much
  noisier proxy for it.

---

## 3. If God still wants something: an alert-only monitor

Cost-free in lock-out terms. It notifies and never blocks.

- **Threshold per leg = backtest longest streak + 1.** That is v1 **13**, donchian **9**, SOL
  **7**. Reaching it means "worse than anything in the backtest". That is a statistically
  meaningful event, and the prompt is to re-run the parity review, not to stop trading. A threshold
  of 4 would fire on v1 about 4×/year and train God to ignore it.
- **Input: exchange realized PnL, not `fills`.** Count strategy closes only. Take the leg's
  `REALIZED_PNL` income rows grouped per position, or the `exit` bot_events cross-checked against
  income. Exclude closes tagged `bf` (boot_flatten), `h` (halt) and `k` (kill). Infra deaths are not
  strategy losses.
- **Where:** `monitor.py` or the daily digest, **not** `bot.py`. It is a reporting concern, so it
  needs no `RISK_REVIEW` and no restart of a live leg, and it sits outside the hot path.
- **Reset:** the streak resets on the first winning strategy close. The alert fires once per
  streak (latched in the monitor's own state) and again at +3 beyond the threshold.

## 4. If God overrides and wants a blocking breaker anyway

Design it to minimise lock-out, and put it where it has to live.

- **Branch:** the `droplet` line only. It must sit beside `_daily_loss_blocks_entry`
  (droplet bot.py:783), which does not exist on `main`. Build it on a branch off current `droplet`
  and cherry-pick nothing from `main`.
- **Semantics: pause, never latch, never flatten.** Block **new entries** for 7 days after K
  consecutive strategy losses. Open positions keep their brackets. This is the only arm in §2b
  that did not damage any leg. A HALT-style latch (the constant's current comment) is ruled out by
  the data.
- **K:** per leg, from its own history. Not a global 4. v1 ≥ 6, donchian ≥ 5, SOL ≥ 5. Even so,
  §2 says it earns roughly nothing.
- **State:** persist `consec_loss_count`, `consec_loss_last_signal_id` and `consec_pause_until` in
  `state.meta`, the same pattern as `daily_loss_breaker_date`, so a restart neither resets nor
  re-trips it. Increment only on a strategy close whose `signal_id` differs from the last one
  counted. That makes it idempotent against the retried exit path (`_hold_unrecorded_exit`) and
  against a replacement exit being recorded twice.
- **Interactions:**
  - **HALT / kill switch:** unchanged, and they dominate. HALT flattens and exits regardless.
    The pause only ever blocks entries.
  - **Daily-loss breaker:** independent. Both are entry-only gates, OR'ed in `_maybe_enter` after
    the flat check. The daily one keeps its UTC-midnight reset. The streak pause is not reset at
    midnight.
  - **Boot-resume:** adopting a position is not an entry, so it is unaffected.
- **Reset:** automatic at `consec_pause_until`, or manually by deleting the meta key (documented
  command). A win resets the count. A `bf`/`h`/`k` close neither increments nor resets it.
- **Ceiling vs knob:** put the *cap* (the largest K allowed and the longest pause) in `risk.py`
  under `RISK_REVIEW`. Put the per-leg K in each params YAML. This mirrors how
  `MAX_DAILY_LOSS_PCT` relates to sizing.

## 5. Test plan

- **Research (before any code), local `.venv` only, network stubbed:**
  - Commit the §2 study as a tool that reproduces the two tables.
  - Add a proper replay arm to the harness, where the pause suppresses entries inside the
    backtest rather than deleting trades ex-post. This removes the flat-time bias noted in §2b.
  - Walk-forward: choose K and the pause on 18-month train windows, and apply them to the next 6
    months.
  - Adopt only if the pre-registered rule passes: return, drawdown, plateau, ≥60% of years, and
    2× commission stress.
- **Alert-only monitor (§3):** unit tests on a fixture income ledger:
  - infra-tagged closes are excluded
  - a win resets the streak
  - the alert fires exactly once at the threshold and once at +3
  - the counter is idempotent across a re-run over the same rows
- **Blocking breaker (§4), if built:** unit tests with the droplet test layout:
  - it trips at K
  - it blocks `_maybe_enter` and not exits
  - it persists across a simulated restart
  - the same `signal_id` is not double-counted
  - `bf`/`h`/`k` closes are ignored
  - it auto-releases at `pause_until`
  - it coexists with a latched daily-loss breaker on the same day
  - the full suite passes on the merged droplet tree via `.venv/bin/python -m pytest`, never on
    the droplet

## 6. Decisions God needs to make

- **E1:** Accept "do not wire it"? *Recommended: yes.*
- **E2:** The dead constant, plus the equally dead `consecutive_losses` meta key in the
  `exchange/state.py` docstring. Delete it from `risk.py` (RISK_REVIEW) or re-comment it as
  "UNENFORCED — see docs/CONSECUTIVE_LOSS_BREAKER_PLAN.md". *Recommended: re-comment.* That keeps
  the decision discoverable where someone will grep for it.
- **E3:** Build the alert-only streak monitor (thresholds 13 / 9 / 7)? *Optional. Low value, zero
  lock-out.*

## 7. Not done here

- No code, no `risk.py` edit, no study tool committed.
- The ex-post simulation is an approximation. The in-harness replay (§5) is the rigorous version,
  and it is only worth running if God wants to reopen E1.


## Decisions — God, 2026-09-29 ("4 defaults")

- D1 build Phase B on the `droplet` line only — **YES**
- D2 refuse to adopt positions opened before Phase A (no stored qty) — **YES, flatten as today**
- D3 read-only probe only, no testnet run — **PROBE ONLY**
- D4 accept a possible duplicate SL on an adopted position — **YES**
- D5 crash-loop guard: max 3 adopts / 30 min, then stop adopting — **YES**
- E1 do NOT wire MAX_CONSECUTIVE_LOSSES as a trading breaker — **AGREED**
- E2 re-comment the constant (and the dead `consecutive_losses` meta key) as UNENFORCED — **YES** (via RISK_REVIEW)
- E3 alert-only streak monitor at 13 (v1) / 9 (donchian) / 7 (sol) — **YES**

Nothing is built yet; implementation needs a separate go and goes through review + tests before any droplet change.

## Implementation deviation (E3), 2026-09-29 - SIGNED OFF by God 2026-09-29 (local fills accepted for this advisory alert)

The alert-only monitor (`monitor.py`, `_check_streak`) does NOT use exchange
realized PnL as section 3 specifies. Its input is the leg's local `fills`
table (`side='close'`, sign of gross `pnl_usd`), because the cron monitor has
no Binance client and no local REALIZED_PNL store exists (`principal_ledger`
holds transfers only). Consequences: the count is an estimate that can run
late or early (gross of fees; exits during downtime, and pre-23-Aug NULL-pnl
closes, are invisible). Restart/HALT/kill closes write no `fills` row, so they
are excluded by construction. Alerts follow section 3: at the threshold, then
again at each +3, re-armed by a win. Thresholds merge per leg from
`config/monitor.yaml`.
