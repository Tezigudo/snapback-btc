# HYBRID_SHORT_PLAN.md — 3rd-leg research plan (replaces failed chop-reverter)

Status: **RESEARCH ONLY**. No deploy plumbing, no `.service` units, no
mainnet wiring until the user explicitly approves Phase 6 AND total Futures
capital reaches the relaxed gate (≥$150 — see [Capital](#capital)).

Authored 2026-05-25 after `CHOPREVERTER_PLAN.md` failed Phase 1. User picked
HYBRID SHORT over classic LONG because the stated goal is "more frequency +
higher winrate." HYBRID SHORT delivers both; LONG_OPTIMAL is parked.

---

## TL;DR

**Name (proposed):** `cnh-hybrid-short-v1`. 4h short signal combining two
pattern detectors (inverse C&H + distribution-top) with EMA(24) breakdown
entry, ATR-based stop, EMA(100) take-profit. Already-prototyped in
`tools/icnh_final_tune.py::find_hybrid_patterns`.

**Best config in-house** (`HYBRID_4h_dedup15_atr1.5_tpema100_entema24` in
`data/final_tune_results.json`):

- 61 trades over 2020-H2 → 2026-H1 (≈ **9.4 trades/yr, 0.78/month**)
- Win rate **70.5%**
- Cum return +75% (unleveraged, in-sample-selected)
- Sharpe 5.15
- 12 per-window slices: 9 positive, 1 flat (2022-H2 +0.0%), 1 near-flat
  (2025-H2 -0.1%), **1 losing (2020-H2 -12.0% on 6 trades, 1/6 WR)**

**Note vs memory `snapback_cnh_long_candidate`**: that memory says HYBRID
runs "1.3 trades/month" — accurate for the dedup=5 variant (89 trades,
1.14/month), NOT for the highest-quality dedup=15 variant. Frequency vs
quality is a real trade-off this plan addresses in Phase 5.

**Diversification:** SHORT-direction strategy. v1 trades both sides;
Donchian trades both sides. Adding a SHORT-only leg deliberately tilts the
portfolio's directional exposure — the bet is the SHORT-direction edge in
BTC's pattern grammar is what's mispriced (per the in-house experiment;
not yet independently validated OOS).

**Capital gate (corrected by Phase 2):** the hybrid leg needs **≥$100
equity** to be tradeable — at $50/leg, 100% of OOS signals get rejected
by Binance's 0.001 BTC min-qty at current BTC prices. With v1 + Donchian
held at $50.50 each, real total capital floor is **~$200 (50% skip rate)
or ~$300 (negligible skips)**. The original `snapback-3leg-search-dead-ends`
rule of ≥$300 is conservative-correct; the earlier ≥$150 relaxation in
this plan was wrong and is hereby withdrawn.

---

## Status of existing research (don't redo what's done)

What's already in the repo (do NOT re-run from scratch):

| Artifact | What it is | Trust level |
|---|---|---|
| `data/final_tune_results.json` (24 runs) | Final tuning grid incl. 9 HYBRID variants with per-window breakdown across 12 H1/H2 slices 2020-2026 | High — concrete config & metrics |
| `tools/icnh_final_tune.py` | Source for the HYBRID detector + final-tune driver | High — runnable now |
| `tools/icnh_grid_sweep.py`, `icnh_mega_sweep.py` | Earlier sweeps (96 + 92 configs) producing `grid_results.json`, `mega_sweep_results.json` | Medium — historical, used to find HYBRID |
| `ICNH_EXPERIMENT.html`, `ICNH_EXPERIMENT_V2.html` | Plotly HTML reports of the experiments | High — readable summary |

What's NOT done (this plan addresses):

1. **True walk-forward.** The 12 per-window numbers above were used to SELECT the
   config (in-sample). A true OOS test = fit on first N windows, test on later
   ones, never look at the test slice during config selection.
2. **Fee/slippage stress.** Existing runs use `FRICTION_BPS` constant in
   `icnh_final_tune.py` — verify it matches Binance Futures USDM live (taker 4
   bps, maker 2 bps). Stress with +5 bps slippage.
3. **Live signal evaluator.** Nothing in `strategy/live_*.py` for HYBRID yet.
   Needs a pure-function port of `find_hybrid_patterns` for `bot.py` dispatch.
4. **Portfolio simulation.** No combined v1 + Donchian + HYBRID equity-curve
   sim has been run. Correlation and combined-Sharpe lift are unknown.
5. **Frequency vs quality decision.** Pick dedup=15 (0.78/mo, 70.5% WR) or
   dedup=5 (1.14/mo, 61.8% WR) — needs explicit user call after Phase 4
   numbers are in.

---

## Strategy design

### Patterns combined

The HYBRID detector is the union (deduplicated) of two BEARISH patterns
defined in `tools/icnh_final_tune.py`:

1. **Inverse Cup-and-Handle** (`_detect_cnh` flipped to SHORT) — concave-down
   parabola where price tops out, retraces with a small bullish handle,
   then breaks down. Strict pattern: `cup_len=20`, `handle_len=5`,
   `R²≥0.7`, depth ≥ 2.5×ATR, handle range ≤ 45%.
2. **Distribution-top** (`_detect_distribution_top`) — user's image-16
   visual rule: uptrend bars (`up16` = 16 of last N rising) followed by
   sideways chop, ending in a breakdown bar. No parabola fit; structural
   only.

Why both: in-sample tests showed each alone produces ~50 trades / Sharpe
2-3; the union dedup'd produces higher-WR composite hits. The hypothesis
is that the two detectors flag *different* visual archetypes of the same
underlying "buyers exhausted" structural moment.

Dedup window: `dedup=15` means if two patterns fire within 15 4h bars
(2.5 days) of each other, take only the first. Higher dedup = fewer trades
but cleaner exits. Lower dedup = more trades, more whipsaw.

### Entry / exit (best in-house config)

- **Timeframe:** 4h (15m for tick management within an open position)
- **Trigger:** pattern detected AND 4h close crosses BELOW EMA(24) on the
  breakdown bar
- **Entry order:** market on next 4h open. (Memory's "limit-only entry"
  rule for chop-reverter does NOT apply — these are breakdown trades,
  limit entries miss the move. Validate this trade-off in Phase 2 fee
  stress.)
- **SL:** `entry + 1.5 × ATR(14, 4h)` — bracket-style, fired at any 15m bar
  in `bot.py`'s normal SL/TP loop
- **TP:** distance to `EMA(100, 4h)` BELOW entry. If EMA(100) is above
  entry (uptrend), the TP slot is unfilled and the trade rides on the
  time stop / SL only — the per-window data above includes these.
- **Time stop:** TBD. The existing runs don't expose this. Default
  candidate: 96 bars (16 days at 4h). Verify in Phase 2.

### Sizing

- Risk per trade: **1.5%** (same as chop-reverter pitch; lower than v1's
  2.75% because winrate is high so single-trade size matters less).
- Leverage ceiling: 20× (hard wall in `risk.py`; do not edit).
- At 1.5%-risk and 1.5×ATR stop on 4h BTC (typical ATR ~1.5%) → notional
  ≈ $50 × 0.015 / 0.0225 = ~$33 — **BELOW Binance's $50 min-notional**.

This is a real problem. Mitigations to evaluate in Phase 2:

- (a) Increase risk to 2.5% per trade → notional ≈ $55. Workable but
  raises portfolio VaR.
- (b) Wait for capital to reach $80/leg → notional at 1.5% risk = $53.
- (c) Tighten SL to 1.0×ATR — fewer winners survive deep dips. Probably
  worsens the strategy.
- (d) Accept that some signals will skip min-notional (the existing
  `exchange/constraints.py` already handles this). Measure how often.

Pick after Phase 2 backtest shows the actual ATR distribution.

### Fee budget

Binance Futures USDM: taker 4 bps, maker 2 bps. SHORT entry + SL/TP both
typically taker (market entry + bracket order) → round-trip 8 bps minimum.
Existing experiments used a `FRICTION_BPS` constant — verify in Phase 2
that the +75% cum survives a fee bump from `FRICTION_BPS` value to 8 bps
+ 5 bps slippage = 13 bps round-trip.

The strategy can absorb significant fee drag: 70.5% WR with 1.5×ATR SL and
EMA(100) TP. Avg win is roughly 2-3× avg loss based on per-window cum/N
math, so EV per trade ~ 100-200 bps gross — plenty of room for 13 bps fees.

---

## Validation gates (strict — promotion gate is broken, don't rely on Sharpe)

All gates must pass before Phase 5 → Phase 6 transition.

- **Walk-forward per-window:** fit dedup/SL/TP knobs on windows 1-8, test on
  windows 9-12. Every TEST window's compounded return > 0. (Existing data
  fails this — 2020-H2 is -12% in-sample, will be worse OOS.)
- **2020-H2 explicit treatment:** the 1-of-6 WR window. Either the
  strategy contains a regime gate that would have skipped 2020-H2 entirely,
  or we accept it as an unrecoverable tail and budget the killswitch to
  -15% (current is -35.5% — wider than needed, but acceptable).
- **After-fee edge per trade > 30 bps** at 13 bps friction (high winrate
  means edge per trade can be small).
- **Combined-portfolio Sharpe (v1 + Donchian + HYBRID)** > Sharpe (v1 +
  Donchian) alone. If not, the leg is noise.
- **Equity-curve correlation < 0.3** with v1 and Donchian over the same
  windows.
- **Realistic-sim** at 13 bps friction + 5 bps slippage + Binance
  min-notional skip-rate from `exchange/constraints.py`. Must still meet
  all of the above. If only the no-friction backtest passes, kill it.

**Anti-pattern guard:** if a gate fails, do NOT loosen the gate, do NOT
re-pick the config, do NOT add a new pattern. Stop and re-pitch. (Same
discipline the chop-reverter plan failed honestly.)

---

## Phases

Each phase has a gate. Failed gate = stop, do not roll forward.

### Phase 1 — Walk-forward audit (the only one that matters before commitment)

1. Modify `tools/icnh_final_tune.py` (or write a sibling
   `tools/hybrid_walkforward.py`) to:
   - Hold out windows 9-12 (2024-H1 → 2026-H1) as test.
   - For each candidate dedup ∈ {5, 10, 15}, fit the rest of the knobs
     (SL ATR mult, TP ema, entry ema) on windows 1-8 only.
   - Score the chosen config on the held-out windows.
2. Produce `reports/HYBRID_WALKFORWARD.html` with per-window OOS bars.

**Gate:** held-out 4-window cum return > 0 for at least one of dedup ∈
{5, 10, 15}, with worst window > -15%. If all three fail, abandon HYBRID
SHORT and pivot to classic LONG (LONG_OPTIMAL grid result).

### Phase 2 — Friction stress + sizing reality

- Compare `icnh_final_tune.py`'s `FRICTION_BPS` to Binance live (8 bps taker
  round-trip). If it's lower, re-run the best config with corrected fees.
- Add `+5 bps slippage` stress.
- Measure the ATR(14, 4h) distribution at the entry bars and compute how
  often the resulting position notional falls below $50.
- Output: a one-page report card. Fees survived? % skipped by min-notional?

**Gate:** after-fee edge ≥ 30 bps/trade AND ≤ 30% of signals lost to
min-notional skipping (or have a documented mitigation).

### Phase 3 — Live signal evaluator

- Port `find_hybrid_patterns` from `tools/icnh_final_tune.py` to a
  pure-function `strategy/live_cnh_hybrid_short.py` mirroring the shape of
  `live_donchian_v3.py` and `live_multifactor_v1.py`.
- Unit tests in `tests/test_cnh_hybrid_short.py`: cover both detectors,
  the dedup window, the EMA(24) breakdown trigger, and a few canned
  fixture bars (sample from the actual bars where the backtest fired).
- DO NOT modify `bot.py` or `bot_internals.py` yet — pure module, no
  dispatch.

**Gate:** unit tests pass; running the live evaluator over the historical
parquet reproduces ≥ 95% of the backtest's signals (some divergence is OK
due to bar alignment).

### Phase 4 — Portfolio simulation

- Run a combined v1 + Donchian + HYBRID short equity-curve sim across the
  5 OOS windows used by `PATH2_RESULTS.html`.
- Compute equity-curve correlation matrix.
- Output: `reports/HYBRID_COMBINED.html`.

**Gate:** combined Sharpe lift ≥ 0.1, correlation < 0.3 with both other
legs.

### Phase 5 — Frequency-vs-quality decision

Once Phases 1-4 are clean, surface to user: dedup=15 (high WR, low freq)
vs dedup=10 (middle) vs dedup=5 (high freq, lower WR). Show their Phase 1
OOS numbers head-to-head and let the user pick consciously.

**Gate:** user explicit choice. No default.

### Phase 6 — Deploy plumbing (DEFERRED until capital ≥ $150)

ONLY when user says go AND total Futures ≥ $150 AND Phase 5 picked a
config:
- Binance sub-account #3 (`snapback-cnh-hybrid-short`).
- `.env.cnh_short`, `config/params_cnh_hybrid_short.yaml`,
  `strategy/live_cnh_hybrid_short.py` (already exists from Phase 3),
  `deploy/snapback-btc-cnh-hybrid-short.service`.
- `bot_internals.py` dispatch entry: `strategy_name="cnh-hybrid-short-v1"`.
- `bot.py --instance cnh_short` flag.
- Dry-run ≥ 14 days (longer than v1/Donchian because frequency is lower —
  need at least 2-3 signals in the dry-run window).
- Pre-flight via `tools/preflight_live.py --strategy cnh_hybrid_short`.
- New `confirm_mainnet.lock` for the sub-account.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Walk-forward OOS Sharpe drops 30-50% (memory warning) | High | Strict Phase 1 gate. If even one of three dedup variants fails OOS, the design is fragile. |
| 2020-H2 -12% wasn't a sample fluke — there's a regime where the detector consistently mis-fires | Medium | Phase 1 looks at 2024-H1 / 2025 / 2026 holdout — if any look like 2020-H2, kill. |
| Min-notional skips kill 30%+ of signals at $50/leg | Medium | Phase 2 measures the actual rate. Mitigation = wait for $80/leg or raise risk per trade. |
| 4 hour timeframe means a single bad trade blocks new entries for days | Inherent | No mitigation needed — low frequency is baked into the 4h design. Accept. |
| Pattern detector silently breaks on a future bar shape change | Low | Live evaluator's Phase 3 reproduction-rate test surfaces this if it happens. |
| SHORT-only direction loses badly in a sustained bull leg | Medium | Phase 1 explicitly tests bull windows (2021-H1/H2, 2024-H2, 2026-H1) — must all be positive in OOS. |
| `FRICTION_BPS` constant is stale (older taker rate) | Medium | Phase 2 audits this. One-line fix if wrong. |
| Bot's existing one-position-per-instance enforcement may double-book if HYBRID fires while v1 is also short | Low | Each leg has its own sub-account = isolated position state by design (per `DEPLOY_COMBINED.md`). |

---

## Out of scope (explicit)

- **No detector retuning** beyond the dedup variable in Phase 5. The
  pattern parameters (cup_len, R², depth, etc.) are frozen at the
  in-house best values; tuning them further would be in-sample bias.
- **No 1h timeframe variant.** `HYBRID_1h_dedup10_atr1.5_tpema100_entema24`
  in the data is Sharpe -3.01 (-51.7% cum). Abandoned. Do not re-pitch.
- **No LONG/SHORT combination.** The HYBRID is SHORT-only. The
  LONG_OPTIMAL candidate from `grid_results.json` (memory says Sharpe 7.4)
  is parked separately; merging them would create a new strategy
  requiring its own validation.
- **No mainnet.** Phase 6 only after user approval + capital + completed
  dry-run.
- **No `risk.py` edits.** Hard wall.
- **No leverage change.** 20× ceiling stays.

---

## Files this plan touches (when executed)

NEW:
- `tools/hybrid_walkforward.py` (Phase 1)
- `tools/hybrid_friction_stress.py` (Phase 2)
- `strategy/live_cnh_hybrid_short.py` (Phase 3)
- `tests/test_cnh_hybrid_short.py` (Phase 3)
- `reports/HYBRID_WALKFORWARD.html` (Phase 1)
- `reports/HYBRID_COMBINED.html` (Phase 4)
- `config/params_cnh_hybrid_short.yaml` (Phase 6)
- `deploy/snapback-btc-cnh-hybrid-short.service` (Phase 6)

EDIT (Phase 6 only):
- `bot_internals.py` (strategy dispatch)
- `bot.py` (`--instance cnh_short` flag)

---

## Next concrete action

Start **Phase 1** — walk-forward audit. Holds out the last 4 windows
(2024-H1 → 2026-H1), fits on 2020-H2 → 2023-H2, scores honestly on the
held-out slice. Output `reports/HYBRID_WALKFORWARD.html` and a pass/fail
verdict.

This is the only Phase that decides whether HYBRID SHORT is real or just
an in-sample artifact. ~1-2 hours of work.

When ready: `say "start hybrid Phase 1"`.

## Capital

The relaxed memory rule used here: **≥ $150 total Futures capital** for a
3rd leg (was ≥ $300 for three new legs). Justification: we're ADDING one
leg, not three, and the existing v1 + Donchian legs are untouched at
$50.50 each. Memory `snapback_3leg_search_dead_ends` is the original rule;
the relaxation applies the same logic the chop-reverter pitch used.

Today: $101 total ($50.50 v1 + $50.50 Donchian). Need ~$50 new for HYBRID.

This is the SAME capital gate that's blocking `LONG_OPTIMAL`. So
whichever wins the OOS test in Phase 1, the deploy can happen at the same
capital threshold.
