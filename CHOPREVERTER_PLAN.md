# CHOPREVERTER_PLAN.md — 3rd-leg research plan

**STATUS: FAILED at Phase 1 (2026-05-25).** Kept for historical reference only.
Do not implement Phase 2+. See bottom of file for the failure record, or
[memory: snapback-chopreverter-phase1-failed]. Pivot for the 3rd-leg slot:
already-validated C&H candidates in `snapback_cnh_long_candidate` memory.

---

Status when written: **RESEARCH ONLY**. Deploy was deferred until capital math
worked (≥1×$50 new USDT for a third sub-account). No `.service` units, no
sub-account creation, no mainnet wiring without user approval.

Authored 2026-05-25 in response to: *"can we fit one more strategy? more frequency,
more winrate, day-trade flavor — research and plan."*

---

## TL;DR

**Name (working):** `chop-reverter-v1`. Mean-reversion on 15m BTCUSDT perp, gated by
a 1-day chop classifier so it ONLY runs when both `multifactor-v1` and `donchian-v3`
are structurally silent.

**Why:** The two live legs cover trend-pullback (v1) and breakout (Donchian). They
both lose money or sit out in chop — that's exactly when 2024 H1 ate -12.56% on v1.
A chop-only reverter is the missing third quadrant, not a fourth pull on the same
rope.

**Diversification target:** equity-curve correlation with v1 and Donchian < 0.3.

**Capital gate:** add this leg only when total Futures capital ≥$150 (the relaxed
form of the memory rule "≥$300 for 3 legs" — relaxed because we're adding ONE leg,
not three, and keeping the existing legs at $50.50 each).

---

## Why this and not something else

I considered (and rejected) these alternatives before landing on chop-mean-reversion:

| Candidate | Verdict | Why rejected |
|---|---|---|
| Higher-freq trend-following (5m EMA cross) | Reject | Same shape as v1, just faster. Not diversifying. Fees dominate at 5m. |
| Funding-carry aligned long | Reject | FCR family already failed (memory: `snapback_fcr_killed_by_data`). |
| Liquidation-cascade fade | Reject | Tried at $50.50/leg, rejected (memory: `snapback_3leg_search_dead_ends`). |
| Opening-range breakout (US session) | Hold | Plausible, but LOWER winrate than mean-reversion — wrong fit for the stated goal. Park as Plan B. |
| C&H LONG (already-researched candidate) | Defer | Memory says blocked by $300 capital rule and already validated; can deploy later when capital allows. Keep it warm. |
| **Chop-conditioned mean reversion** | **GO** | Orthogonal to both legs by design. High winrate matches the stated goal. Capital-feasible at $50/leg if maker-only. |

---

## Strategy design

### Regime gate (PRECONDITION — without this, do NOT fire)

The leg evaluates signals only when the daily classifier says CHOP:

- `daily ADX(14) < 20` AND
- `4h Donchian-20 width as % of close < pct30(90-day)` (no breakout regime) AND
- `daily realized vol (5d close-to-close std) < pct50(90-day)`

Extend `strategy/regime_classifier.py` with `is_chop_day(df_daily, df_4h) -> bool`.
The gate refreshes once per day at 00:00 UTC; signals on 15m within that day either
all see CHOP=True or all see CHOP=False. Cheap, no look-ahead.

### Entry rules (15m)

LONG candidate, ALL true:
1. Regime gate: CHOP=True for the current day
2. `Close < BB_lower(20, 2.0)` on 15m
3. `RSI(2) < 10` on 15m (Connors-style oversold)
4. `Close > EMA(50)` on 1h (don't catch a falling knife — fade WITH the medium trend)
5. No open position on this leg (one-at-a-time)

SHORT = mirror (upper band + RSI(2)>90 + 1h close < 1h EMA(50)).

Entry is **limit-only**, placed at `BB_lower - 5 bps` (long) — sit on the bid as
price tags the band. Cancel after 30 min if not filled. No market fallback (mean-
reverters that chase get run over).

### Exit rules

- **TP**: return to BB middle (the 20-SMA). Posted as a limit order at fill time.
  Typical distance: 60–120 bps from entry.
- **SL**: `entry - 1.0 × ATR(14)` (long) — market order. Typical 40–80 bps.
- **Time stop**: 16 bars (= 4 hours). If neither TP nor SL hit, close at market.
- **Regime exit**: if `is_chop_day` flips to False mid-trade, exit at next bar close.

### Sizing

- Risk per trade: **1.5%** (lower than v1's 2.75% — more trades means more aggregate
  exposure; we want similar daily VaR).
- Leverage ceiling: 20× (already in `risk.py`, do not touch).
- Effective leverage at typical 0.6% stop: ~2.5× — well within ceiling.

### Fee budget (this is the gate, not min-qty)

Binance Futures USDM: taker 4 bps, maker 2 bps. Round-trip taker = 8 bps.

| | LONG TP | LONG SL |
|---|---|---|
| Move | +90 bps (BB_mid) | -60 bps (1×ATR) |
| Entry fee (maker) | -2 bps | -2 bps |
| Exit fee | -2 bps (TP limit, maker) | -4 bps (SL market, taker) |
| **Net per trade** | **+86 bps** | **-66 bps** |

Breakeven winrate = 66 / (66+86) = **43%**. Target winrate = 60–65% (mean-reversion
norm with trend filter). Safety margin is large enough that a 10pp winrate haircut
in live still keeps EV positive.

**Hard rule for this strategy: limit-only entries.** A market-entry version would
flip net per trade by ~6 bps and erase most of the edge. If limit-fill rate falls
below 60% in dry-run, the strategy is broken; do not deploy.

### Expected frequency

Chop days historically run ~30–40% of the calendar. Within chop, 2–3 signals/week.
So **~40–60 trades/year**, vs v1's ~50/year. Frequency is comparable, **the win
is in timing**: it fires when v1 is silent or losing.

---

## Validation gates (load-bearing — do NOT skip)

The promotion gate is currently broken (memory: `snapback_promotion_gate_broken` —
median Sharpe passes a -18.6% CAGR strategy). For this strategy, use the strict
form:

- **Per-window compounded return > 0** on all 5 OOS windows (2022 H1 → 2025 H1).
  Aggregate Sharpe is not enough — must pass each window individually.
- **Worst window drawdown > -10%** (not the wider -15% killswitch).
- **After-fee edge per trade > 3× round-trip taker fee** (≥ 24 bps net).
- **Limit-entry fill rate ≥ 60%** in backtest (model: limit fills if low-of-bar
  touches the limit price; otherwise cancels).
- **Equity-curve correlation < 0.3** with both `multifactor-v1` and `donchian-v3`
  on the same windows.
- **Combined portfolio Sharpe** (v1 + Donchian + ChopReverter) > combined Sharpe
  (v1 + Donchian) alone. If adding the leg doesn't improve the combined Sharpe,
  the leg is noise and we reject it regardless of its standalone numbers.
- **Realistic-sim stress**: add 5 bps slippage on top of fees. Must still pass all
  the gates above. If only the no-slippage backtest passes, kill it.

If any gate fails, **stop and re-pitch** rather than tuning until it passes.
Tuning into compliance is how the broken-gate problem got created.

---

## Phases

Each phase has a **gate** — if the gate fails, stop, don't roll into the next.

### Phase 1 — Feasibility audit (no new code, ~1 hour)

Run a quick notebook-style script that:
- Loads `data/historical/BTC_USDT_USDT_4h.parquet` + daily resample.
- Classifies each day as CHOP or not using the proposed gate.
- Reports: % of days that are CHOP, % of those days that also had no v1 signal
  AND no Donchian breakout.
- Output: a single number — `expected_chop_trade_days_per_year`.

**Gate:** ≥ 60 chop-trade days/year, OR the strategy can't generate meaningful PnL
regardless of edge. If gate fails, abandon and pivot to Plan B (ORB).

### Phase 2 — Signal prototype + 2024 H1 backtest

- Write `strategy/signals_chopreverter.py` mirroring `signals_multifactor.py` shape.
- Extend `strategy/regime_classifier.py` with `is_chop_day`.
- Run backtest on **2024 H1 only** first (the chop window that ate v1).

**Gate:** 2024 H1 returns > 0 with the strict fee model. If 2024 H1 is flat or
negative, the whole thesis is wrong; stop here.

### Phase 3 — OOS walk-forward (5 windows)

- Run all 5 OOS windows from `PATH2_RESULTS.html` setup.
- Generate `reports/CHOPREVERTER_OOS.html` parallel to existing reports.

**Gates:** all six validation criteria above. Reject if any fails.

### Phase 4 — Portfolio simulation

- Combine v1 + Donchian + ChopReverter equity curves across the 5 windows.
- Verify Sharpe lift and correlation criteria.
- Report combined CAGR, DD, killswitch trip count.

**Gate:** combined Sharpe improvement ≥ 0.1, no new killswitch trip.

### Phase 5 — Param sweep (LIGHT touch)

- Sweep ONLY 3 params with ≤5 values each: BB lookback (10/20/40),
  RSI(2) threshold (5/10/15), ADX chop threshold (15/20/25).
- Pick by **worst-window compounded return**, NOT mean Sharpe.
- If best sweep result is materially better than default config, refresh Phase 3
  numbers; otherwise keep defaults (avoid overfit).

**Gate:** chosen params must still pass Phase 3 gates on the held-out window. Use
the most recent OOS window as the held-out one and sweep on the prior four.

### Phase 6 — Deploy plumbing (DEFERRED, user-approval required)

ONLY when user says go AND total capital ≥ $150:
- Create Binance sub-account #3 (`snapback-chopreverter`).
- `.env.chopreverter`, `config/params_chopreverter.yaml`,
  `strategy/live_chopreverter.py`, `deploy/snapback-btc-chopreverter.service`.
- `bot_internals.py` dispatch entry for `strategy_name="chopreverter-v1"`.
- Dry-run ≥ 7 days. Watch limit-fill rate, regime gate flips, and confirm no
  desync with v1/Donchian legs.
- Pre-flight via `tools/preflight_live.py --strategy chopreverter`.
- `confirm_mainnet.lock` for the new sub-account.

Nothing in Phase 6 happens without explicit user approval per fact in the chain.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Chop classifier lags — fires after chop ends | Medium | Daily refresh, no intraday flip. Time stop at 4h caps damage when wrong. |
| Mean-reversion fails when chop has structural break (e.g., ETF news drop) | Medium | Tight ATR stop, regime exit, killswitch -35.5% covers tail. |
| Limit orders don't fill at extremes (price snaps back without tagging band) | Low–Med | Phase 3 fill-rate gate (≥60%). If broken, kill in dry-run not live. |
| Overfit to historical chop windows | High (mean-rev norm) | Strict per-window gate, light param sweep, held-out window in Phase 5. |
| Correlation with v1 ends up >0.3 in live (regime overlap) | Low | Phase 4 gate. If correlation is too high, kill. |
| Adding a 3rd leg complicates monitoring | Certain | Mirror existing systemd + log file pattern. `/status` reads all three. |
| Capital math: $50 leg + tight stop ⇒ position near min-notional | Medium | Effective position ~$125 at 0.6% stop, above $50 min. Verify via `tools/preflight_live.py` before deploy. |

---

## Out of scope (explicit)

- **No leverage change.** Stays at 20× ceiling per `snapback_leverage_20x`.
- **No edits to `risk.py`.** Hard wall.
- **No retuning of multifactor-v1 or Donchian-v3.** They run as-is. This leg is
  purely additive.
- **No mainnet.** Phase 6 only after user approval + capital.
- **No new historical data fetch.** Use the existing 15m / 1h / 4h parquets.
- **No C&H LONG.** That's a separate parked candidate (memory:
  `snapback_cnh_long_candidate`); revisit when capital ≥ $300.

---

## Files this plan touches (when executed)

NEW:
- `strategy/signals_chopreverter.py`
- `strategy/live_chopreverter.py` (Phase 6)
- `config/params_chopreverter.yaml` (Phase 6)
- `tools/build_chopreverter_report.py`
- `tools/sweep_chopreverter.py`
- `reports/CHOPREVERTER_OOS.html` (generated)
- `tests/test_chopreverter_signals.py`
- `tests/test_chop_regime.py`
- `deploy/snapback-btc-chopreverter.service` (Phase 6)

EDIT:
- `strategy/regime_classifier.py` (add `is_chop_day`)
- `bot_internals.py` (Phase 6, dispatch)
- `bot.py` (Phase 6, `--instance chopreverter` flag)

---

## Next concrete action

Start **Phase 1** (feasibility audit). One-off Python script reading the 4h parquet,
estimating chop-day counts and v1-silence overlap. Output is a single
go/no-go number. No commits, no plumbing — pure measurement. About one hour of
work, and it answers whether the rest of this plan is worth doing.

When ready: `say "start Phase 1"` and I'll write the audit script.

---

## Phase 1 failure record (2026-05-25)

Audit: `tools/phase1_chop_feasibility.py` over 2022-01-01 → 2025-06-30.

| Chop threshold (EMA-slope-strength, % per 4h bar) | chop-trade-days/yr | Verdict |
|---|---|---|
| slope < 0.05 (project-canonical "trending" cutoff) | **53.5** | FAIL (gate 60/yr) |
| slope < 0.07 | 65 | passes only by moving the line |
| slope < 0.10 | 83 | reframes to "not screaming trend" — different strategy |
| ER < 0.25 (broken on BTC perp) | 110 | filter not filtering (94% of days "chop") |

Per-year at canonical threshold: 2022=44, 2023=68, 2024=45, 2025 H1=30 (≈60 ann.).

Decision per the plan's own gate: **abandon, do not Phase 2.** Loosening to 0.07
or reframing to 0.10 would be tuning into compliance — the exact anti-pattern
this plan's "Validation gates" section forbids.

The 3rd-leg slot remains open. Compare any new pitch against the already-validated
C&H candidates (memory `snapback-cnh-long-candidate`): classic LONG Sharpe 7.4 /
9-of-11 windows, or HYBRID SHORT 70.5% WR / +75% cum / 1.3 trades/month. Both
blocked only by the same capital rule this plan tried to relax to $150 — that
relaxation applies to them too.
