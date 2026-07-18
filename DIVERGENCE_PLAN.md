# DIVERGENCE_PLAN.md — RSI + OBV divergence strategy

Status: **RESEARCH + DESIGN ONLY**. No code yet, no config edits, no deploy
plumbing. Authored 2026-06-01 at user request ("research and design a
divergence-based strategy, look into volume too").

This document is a contract: it describes exactly what the strategy detects,
where the traps are, how it's coded, and how it's judged. Implementation
starts only after user OKs the plan.

---

## TL;DR

**Name (proposed):** `divergence-v1`. 15m strategy that trades the four
classical price-vs-indicator divergences, paired with **OBV divergence** as
volume confirmation, on causally-detected swing fractals.

**Why this design:**
1. **RSI divergence** is the canonical momentum-exhaustion signal — price
   keeps printing new highs/lows but RSI fails to confirm.
2. **OBV divergence** is the canonical volume-exhaustion signal — same
   geometry, but the "indicator" is cumulative signed volume. When *both*
   diverge the same way, it's price + momentum + volume all agreeing the
   move is hollow. That's much stronger than RSI alone (which is famous for
   firing on every minor wiggle) and replaces the cruder "volume > N × SMA"
   gate.
3. The repo already has `swing_high_low()`, `rsi()`, `atr()`, `macd()`. We
   only need to add `obv()` and one divergence-detector helper.

**The one trap that decides whether this works:**
`swing_high_low()` is a **centered** k-bar fractal — a swing at bar `i` is
not known until bar `i+k`. Naively precomputing the swing mask on the full
series and indexing it in `next()` peeks `k` bars into the future. Divergence
backtests are notorious for looking spectacular and dying live because of
exactly this. The design pins this down: swings register at `swing_bar + k`,
entry fires no earlier than `swing_bar_2 + k`, and the live evaluator detects
swings causally bar-by-bar. **This is the whole ballgame; it gets its own
section below.**

**Volume:** OBV-*divergence*, not a `volume > 1.5×SMA` gate. The latter is a
different (weaker, coincident) idea bolted onto the side.

---

## Status — Phase 4 results (2026-06-02)

**Outcome: SHELVED.** 42-config OOS fleet (baseline → 6-combo ablation → 36-cell
sweep → 2-config trend-filter check) on BTC/USDT 15m across 5 OOS windows
(2022_H1, 2023_H1, 2024_H1, 2024_H2, 2025_H1). Final winner
`filter:trend_with` (RSI+OBV+MACD AND-gated, RSI 40/60, sep 3-40 bars,
SL 1×ATR / TP 3×ATR, trend filter ON) produced:

- compounded return **-2.24%**, worst window **-1.25%**
- **0 of 5 windows positive**, **0%** win rate, **only 2 trades total** across ~2.5y of data
- 3 of 5 windows fired zero trades (strategy dormant 60% of the time)

Adversarial verification (2 lenses: overfit, execution) **both refuted** the
winner — same root cause: AND-gate stack drives fire rate to near-zero, and
the orchestrator rewarded inactivity (0% beats any negative compounded). The
"sweep winner" by orchestrator label (rsi40-60 sep3-40 sl1tp3 no-filter) was
actually the worst-performing active config in the entire fleet at -9.11%.

Promotion gate failed on every metric except max DD, and max DD only passed
because the strategy refused to trade. Not promoting to dry-run. Not iterating
on the gate stack — the search space is structurally wrong, not the parameters.
Full synthesis in `reports/divergence_v1_oos.html`.

Next-step candidates (not started): weaken AND to OR-of-2, drop MACD per
original v1 spec, scale to 1h/4h, add a hard ≥30-trade floor to the
orchestrator, look-ahead audit, cross-coin sanity (ETH/SOL 15m).

---

## What this is NOT

- **Not deploy-ready.** No `.service` unit, no live wiring, no `params.yaml`
  edit. Locked strategy `multifactor-v1` and the active deploy path are not
  touched.
- **Not a replacement for the multifactor leg.** This is a *new candidate*
  to be judged on the same OOS yardstick (5 windows) as the rest of the
  family. It earns its way to mainnet through the standard gate, not
  by being interesting.
- **Not a chop-survivor by default.** Divergence is a reversal pattern; it
  bleeds in strong trends if forced to enter against them. Phase 4 decides
  whether we ship a trend-filter variant or accept the chop tax.

---

## The four divergences (what we're detecting)

| Type | Price | Indicator (RSI / OBV) | Trade |
|---|---|---|---|
| **Regular bullish** | Lower low (LL) | Higher low (HL) | LONG — momentum/flow fail to confirm new low → reversal up |
| **Regular bearish** | Higher high (HH) | Lower high (LH) | SHORT — momentum/flow fail to confirm new high → reversal down |
| **Hidden bullish** | Higher low (HL) | Lower low (LL) | LONG continuation — pullback in uptrend resumes |
| **Hidden bearish** | Lower high (LH) | Higher high (HH) | SHORT continuation — pullback in downtrend resumes |

**v1 ships regular bullish/bearish only.** Hidden divergences are
continuation patterns whose interpretation depends on a separately-defined
trend regime, which adds a config dimension that should be tuned after the
core detector is proven. They're on the `divergence-v2` slate, not v1.

---

## Lookahead safety — the section that decides whether this works

### The trap

`strategy/indicators.py::swing_high_low(high, low, k=3)` defines a swing at
bar `i` as `h[i] == max(h[i-k:i+k+1])`. That reads `k` bars to the **right**
of `i`. The swing is therefore not knowable in real time until bar `i+k`.

If we copy the multifactor-v1 init/next pattern — precompute the swing mask
on the full series in `init()`, then index `self._swing[i]` in `next()` —
we are peeking `k` bars ahead. RSI and EMA don't have this problem (they
only use past data), so the multifactor precompute pattern is **not safe
to reuse here** without modification.

This is exactly why most published divergence backtests don't reproduce live.

### The fix

**Definition (use everywhere):** a swing detected at bar `i` "registers"
(becomes knowable) at bar `i + k`. We call `i` the **swing bar** and
`i + k` the **confirmation bar**.

In code:

- Compute `swing_high_mask`, `swing_low_mask` on the full series as
  the repo already does (cheap, vectorized).
- **Shift both masks forward by `k`**: `registered_low = swing_low_mask.shift(k)`.
  Now `registered_low[j] == True` iff bar `j - k` was a confirmed swing low,
  and that information was first available at bar `j`.
- In `next()`, only ever read `registered_low[: current_bar + 1]`. Use this
  to find the two most-recent registered swings — each is by construction
  at least `k` bars old.
- Entry fires on the current bar's close (or the bar after — see Phase 4),
  no earlier than the confirmation bar of the more recent swing.

The detector compares `swing_bar_1` and `swing_bar_2` (the actual swing
indices, both already `≥ k` bars in the past) — but only after both have
been registered.

### The live evaluator

`live_divergence.py` cannot precompute and must detect swings **causally
bar-by-bar**. On each tick: take the last `lookback_bars` of OHLCV, scan
backwards for two swing fractals whose right-side `k`-bar windows have all
elapsed (i.e. their confirmation bars are `≤ current_bar`), then run the
same divergence check. The live evaluator's docstring must call out this
parity explicitly, following the `live_donchian_v3.py` convention.

### How we prove it's right

`tools/divergence_validate.py` (new): replay 90 days of cached 15m bars
through (a) the backtest path and (b) the live evaluator, expect ≥99%
signal-bar parity. Any divergence is a lookahead bug. This is the same
ritual `hybrid_phase3_validate.py` runs for the hybrid-short leg.

---

## Signal logic — concrete

### Regular bullish divergence → LONG

Trigger conditions (ALL true on the current bar):

1. The most recent two **registered** swing lows are at bars `b1 < b2`
   with `min_swing_separation_bars ≤ (b2 - b1) ≤ max_swing_separation_bars`.
2. `low[b2] < low[b1]` — price made a lower low.
3. `rsi[b2] > rsi[b1]` — RSI made a higher low (RSI divergence).
4. `rsi[b2] < rsi_oversold_zone` — the more recent swing is in the
   oversold zone. Divergences in extreme zones are higher quality;
   divergences mid-range are noise.
5. `obv[b2] > obv[b1]` — OBV made a higher low (OBV divergence, same
   geometry — volume flow is bottoming out before price does).
6. **Confirmation:** current bar's close `>` high of bar `b2`. We do not
   enter into the swing low itself — we wait for price to break the
   swing's high before committing.
7. `current_bar > b2 + k` — the swing at `b2` has been registered for at
   least one full bar (no entering on the registration bar itself, to
   keep the live evaluator and the backtest cleanly aligned).

### Regular bearish divergence → SHORT (mirror)

1. Two registered swing highs `b1 < b2`, separation in `[min, max]`.
2. `high[b2] > high[b1]` — higher high in price.
3. `rsi[b2] < rsi[b1]` — lower high in RSI.
4. `rsi[b2] > rsi_overbought_zone`.
5. `obv[b2] < obv[b1]` — lower high in OBV.
6. Close `<` low of bar `b2`.
7. `current_bar > b2 + k`.

### Exits

- **Initial stop:** for LONG, `entry - sl_atr_multiple × ATR(14)`, but no
  tighter than 0.1% below the swing low at `b2`. For SHORT, mirror.
- **Take profit:** `entry ± tp_atr_multiple × ATR(14)` (single target, no
  scaling for v1 — adds backtest complexity that isn't load-bearing yet).
- **Time stop:** close the position after `max_hold_bars` regardless. If
  divergence reversal hasn't paid by then, the thesis was wrong.
- **No trailing stop in v1.** Adds variance that's hard to attribute when
  judging whether divergence detection itself has edge.

---

## Default params (proposed for `config/params.yaml` — new section)

```yaml
# NEW section — does not touch multifactor-v1 or the active deploy path.
divergence_v1:
  timeframe: "15m"

  # Swing detection (fractal). k=3 matches indicators.swing_high_low default.
  # Larger k = fewer, "more major" swings, more lag before confirmation.
  swing_k: 3

  # Window in which two swings are paired for divergence.
  # Adjacent swings are noise; ancient swings are stale.
  min_swing_separation_bars: 5     # ~75 min on 15m
  max_swing_separation_bars: 60    # ~15 hours on 15m

  # RSI divergence.
  rsi_period: 14
  rsi_oversold_zone: 35            # for longs — slightly above 30 catches
                                   # divergences completing just outside extreme
  rsi_overbought_zone: 65          # for shorts (asymmetric like multifactor-v1)

  # OBV divergence — boolean gate, no threshold (sign of (obv[b2] - obv[b1])).

  # Exits.
  atr_period: 14
  sl_atr_multiple: 1.5
  tp_atr_multiple: 4.5             # target ~3:1 R:R before slippage/fees
  max_hold_bars: 96                # 24 hours on 15m

  # Sizing — kept conservative because divergence is high-variance.
  risk_per_trade_pct: 1.0
  leverage: 20                     # repo ceiling, not a recommendation
  allow_shorts: true
```

These are **starting points for Phase 3 tuning**, not final values.

---

## Phases

| # | Phase | Gate to next | Output |
|---|---|---|---|
| 0 | This plan + user OK | User explicit "go" | `DIVERGENCE_PLAN.md` (this file), user sign-off |
| 1 | Add `obv()` + `find_divergence()` to `indicators.py`. Unit tests. | All tests green, hand-verified swing-shift on a 50-bar fixture. | Code + tests, no strategy yet. |
| 2 | `signals_divergence.py` (backtest) + `tools/divergence_validate.py` (live↔backtest parity). | Parity ≥99% on a 90-day cached window. | Backtest runs, validator passes. |
| 3 | Single-window backtest on 2024 H2 (calm). Eyeball sanity: does it fire? Are signals clustered around obvious chart reversals? | Reasonable trade count (~5–25/window), no obvious garbage. | Eyeball report + adjustment notes. |
| 4 | 5 OOS windows — same windows multifactor-v1 was judged on. Decide trend-filter on/off based on per-window slices. | Compounded return > 0, ≥3/5 windows positive, worst window > -15%. | OOS report (HTML), explicit per-window table. |
| 5 | `live_divergence.py` causal evaluator + tests. | Validator parity still ≥99% on a fresh 30-day window. | Live evaluator + tests. |
| 6 | User review of Phase 4 numbers vs multifactor-v1 baseline. | **User decision** — promote to dry-run leg, shelf, or revise. | Decision recorded. |

No phase implies the next will be approved. Each is a stop-and-show.

---

## Files this plan creates / touches

### Adds
- `DIVERGENCE_PLAN.md` (this file)
- `strategy/signals_divergence.py` (backtest strategy — Phase 2)
- `strategy/live_divergence.py` (live evaluator — Phase 5)
- `tools/divergence_validate.py` (parity check — Phase 2)
- `tests/test_divergence.py` (Phase 1 + 2 + 5)
- `reports/divergence_v1_oos.html` (Phase 4 output)

### Edits
- `strategy/indicators.py` — add `obv(close, volume)` and a divergence
  helper. Pure-numpy/pandas, no pandas-ta (per the file's existing rule).
- `config/params.yaml` — new `divergence_v1:` section only.

### Off-limits (do not touch under this plan)
- `risk.py` (always — repo rule, enforced by pre-commit)
- `strategy/live_multifactor_v1.py` and `signals_multifactor.py` (locked)
- Any deploy unit, `BINANCE_ENV`, `data/HALT`, `confirm_mainnet.lock`
- Existing dry-run / live leg configs (cnh-hybrid-short, donchian-v3, etc.)

---

## Risks & how the plan addresses each

| Risk | Mitigation |
|---|---|
| Centered-fractal lookahead | Explicit `+k` shift, written into both backtest and live evaluator. Validator (Phase 2) gates everything else. |
| Divergence backtests look great, die in chop | OOS judged across the same 5 windows multifactor-v1 was judged on, including chop windows. Per-window slice mandatory in Phase 4. |
| OBV is noisy and crypto-volume is unreliable | OBV used as a **divergence boolean**, not a magnitude — we only ask "did OBV's swing-low move up or down relative to the previous swing's OBV?". Sign-only is robust to volume noise. |
| Overfit to one indicator choice (RSI) | Phase 4 also evaluates **MACD-histogram divergence** as a side-comparison (no extra code — `macd()` already exists; same detector helper). If MACD-div materially outperforms RSI-div, we switch indicators before declaring v1 done. |
| Capital requirements as with hybrid-short ($150–300 floor) | Out of scope for this plan — capital gates are a deploy decision, not a strategy-design decision. Address in Phase 6 if the strategy survives Phase 4. |

---

## Open questions for the user (before Phase 1 starts)

1. **Trend filter default.** Pure regular divergences are reversal plays
   that work *against* the trend by definition. Three options:
   - (a) No trend filter — take every regular divergence (highest count,
     bleeds in strong trends).
   - (b) Trend filter ON — only take longs above EMA(200), shorts below.
     Cuts the count but filters out fighting-the-tape losses.
   - (c) Tune it in Phase 4 across both. **(Recommended.)**
2. **Timeframe.** 15m to match multifactor-v1, or 1h for fewer, cleaner
   swings? 15m proposed by default; the swing-confirmation lag means 1h
   would feel sluggish (3 × 1h = 3h after-the-fact).
3. **Indicator choice.** RSI primary (this plan), or also accept MACD-hist
   divergence in v1? Adding MACD doubles signal frequency at the cost of
   more false positives. Plan defaults to RSI-only for v1, MACD compared
   side-by-side in Phase 4 but not traded.

User answers shape Phase 1; defaults above are sensible if you say "just
pick."
