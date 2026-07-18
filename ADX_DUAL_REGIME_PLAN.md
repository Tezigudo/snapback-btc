# ADX_DUAL_REGIME_PLAN.md — regime-switched RSI MR + Donchian-20 breakout

Status: **RESEARCH ONLY, FAIL-FAST.** Authored 2026-06-02 after divergence-v1
shelved (see DIVERGENCE_PLAN.md Phase 4 status). Recommended by
FUTURE_DIRECTIONS.md as the #4 EV-ranked candidate (medium confidence) and
the most structurally complementary to where divergence-v1 bled.

This plan is short by design. Lesson from divergence-v1: spending 3M+ tokens
on a wide OOS sweep before proving a strategy fires meaningfully on a single
window is wasteful. Smoke first, sweep only if smoke survives.

---

## TL;DR

**Name (proposed):** `adx-dual-regime-v1`. Regime-switched strategy on BTC 15m:

| ADX(14) | Regime | Leg |
|---|---|---|
| ≤ adx_chop_threshold (default 25) | Range / chop | RSI(2) snap mean-reversion |
| > adx_chop_threshold | Trend | Donchian-20 channel breakout |

Both legs allow long + short. Only one leg's signal is active at a time per
the ADX gate. Mutually exclusive — never both signals fire on the same bar.

**Why this matches our gap:**
- divergence-v1 bled in trending windows (counter-trend shorts at modest OB);
  this strategy *follows* trends when they're real (ADX > 25).
- divergence-v1 needed both indicators to agree (AND-gate) and ended up dormant;
  this strategy uses ADX as a *switch*, not an extra gate.
- Both legs are well-understood, single-indicator entries — no theater gates.

**Promotion gate (same as multifactor-v1):**
- Compounded return > 0 across 5 OOS windows (2022_H1, 2023_H1, 2024_H1, 2024_H2, 2025_H1)
- ≥ 3 of 5 windows positive
- Worst window > -15%
- Total trades ≥ 50 across all windows (no inactivity-wins like divergence-v1)
- ≥ 30 bps per-trade EV survives at 15 bps roundtrip slippage stress

If any gate fails, shelf and move to next candidate.

---

## Why not just patch divergence-v1?

FUTURE_DIRECTIONS.md flagged 3 structural bugs (cumulative-OBV, trivial
breakout confirm, default-unsafe). The Phase 4 empirical result showed
under-firing, not over-firing — the AND-gate stack is too restrictive.

OR-of-2 (relaxing AND to OR) and dropping MACD are the only patches with a
plausible chance of helping, but the literature is broadly anti-divergence
on crypto (PMC paper "strongly advises against"; same study found
long-only RSI divergence underperforms buy-and-hold). The expected EV of
patches is modest at best. The expected EV of a new structural-edge
strategy (ADX dual-regime) is higher with similar build cost.

User can choose to revisit divergence-v1 OR/MACD-drop patches later;
this plan goes elsewhere.

---

## Build plan — single engineer dispatch, three deliverables

### A. `strategy/indicators.py` additions

```python
def adx(high, low, close, period=14) -> pd.Series:
    """Wilder ADX. Returns ADX line only (not +DI/-DI). Causal."""

def donchian_channel(high, low, period=20) -> tuple[pd.Series, pd.Series]:
    """Returns (upper, lower) = (rolling max(high, period), rolling min(low, period)).
    Shift by 1 in the strategy so the channel from the PRIOR period is used —
    a breakout at bar i is judged against bars [i-period .. i-1], not [i-period+1 .. i].
    """
```

Pure pandas/numpy. Follow indicators.py conventions (Wilder smoothing where
applicable, NaN at the head, no fill-mid-series).

Tests: `tests/test_adx_donchian.py` — Wilder math, prior-bar shift causality,
NaN warmup.

### B. `strategy/signals_adx_dual_regime.py`

Class `ADXDualRegimeV1(Strategy)`. Default class attrs:

```python
adx_period = 14
adx_chop_threshold = 25.0      # ADX ≤ → range leg, > → trend leg

# Range leg: RSI(2) snap MR (Larry Connors style)
range_rsi_period = 2
range_rsi_long_threshold = 10.0
range_rsi_short_threshold = 90.0

# Trend leg: Donchian-20 breakout
donchian_period = 20

# Exits — same ATR contract as divergence-v1 (lessons reused)
atr_period = 14
sl_atr_multiple = 1.5
tp_atr_multiple = 3.0           # 2:1 R:R (lower than divergence's 3:1 — Donchian
                                # breakouts have higher hit rate than divergence reversals,
                                # don't need the same R:R cushion)
max_hold_bars = 96              # 24h

# Sizing — STRICTER defaults than divergence-v1 (lesson from FUTURE_DIRECTIONS)
risk_per_trade_pct = 1.0
leverage = 5                    # NOT 20 — see FUTURE_DIRECTIONS bug #3
allow_shorts = True
```

`_long_signal(i)`:
1. ADX is finite at bar i.
2. If `adx[i] ≤ adx_chop_threshold`: RSI(2) < range_rsi_long_threshold AND `close[i] > ema(close, 200)[i]` (trade range MR only on the right side of the long-trend EMA, avoids fading nukes)
3. If `adx[i] > adx_chop_threshold`: `high[i] > donchian_upper_prior[i]` (breakout above prior 20-bar channel)
4. Position sizing identical to `signals_divergence.py::_position_units` (copy verbatim).

`_short_signal(i)`: mirror.

Exits: SL/TP at fixed ATR multiples; time stop at max_hold_bars. Same as
divergence-v1.

### C. `config/params_adx_dual_regime.yaml`

Match the layout of `config/params_donchian.yaml`. Include the strategy block
above + symbol + timeframe entries.

### D. Smoke check (one shot, two windows)

Reuse `tools/run_divergence_experiment.py`. The engineer will need to
generalize it to take `--strategy-class <module:Class>` so a single runner
can run any strategy. Default = DivergenceV1 (back-compat).

Then run **two** quick backtests:
1. 2024 H1 (the chop window divergence-v1 bled in)
2. 2024 H2 (trend window divergence-v1 bled in)

Print metrics for both. Eyeball test:
- ≥ 20 trades per window? If < 5, the strategy is over-gated like divergence-v1 → shelf immediately, do not sweep.
- Mixed wins / losses, not all-losing? If 0% WR like divergence-v1's winner → shelf.
- Compounded return positive across both windows? Soft requirement — single-window
  positive is enough to justify the OOS sweep.

If both pass, proceed to OOS sweep (Phase 5, separate dispatch).
If either fails the trade-count floor, do NOT sweep — shelf with a one-paragraph
note and pivot to next candidate (Volume Profile POC #2, or AdaptiveTrend #3).

---

## Out of scope for this plan

- Multi-coin sweep (only after BTC OOS gate passes).
- Live evaluator (`live_adx_dual_regime.py`) — Phase 6 if v1 survives.
- Funding cost modeling — Phase 5 prep.
- Adaptive ADX threshold — v2 question.
- Multiple Donchian periods, multiple RSI thresholds — only sweep if smoke survives.

## Hard rules carried from CLAUDE.md / FUTURE_DIRECTIONS

- Default leverage 5x, NOT 20x. risk.py 20x ceiling is the hard cap, not the default.
- Default `allow_shorts = True` only because both legs (RSI MR and Donchian) have
  symmetric long/short logic — no asymmetric expectancy like divergence's
  counter-trend shorts.
- Trend-EMA filter on the range MR leg by default (lesson from divergence-v1
  bug #3) — don't fade strong directional moves with RSI(2).
- No edits to `risk.py`. No edits to `config/params.yaml` (multifactor-v1 locked).
- No commits during research phase.

---

## Files this plan creates / touches

| File | Action |
|---|---|
| `ADX_DUAL_REGIME_PLAN.md` | This file |
| `strategy/indicators.py` | Append `adx()` and `donchian_channel()` |
| `strategy/signals_adx_dual_regime.py` | New |
| `config/params_adx_dual_regime.yaml` | New |
| `tools/run_divergence_experiment.py` | Generalize: rename to `run_strategy_experiment.py`, add `--strategy-class` arg, default = DivergenceV1 for back-compat |
| `tests/test_adx_donchian.py` | New |
| `reports/adx_smoke_2024h1h2.json` | Smoke check output |

Off-limits: `risk.py`, `config/params.yaml`, `strategy/live_multifactor_v1.py`,
`strategy/signals_multifactor.py`, any deploy plumbing.
