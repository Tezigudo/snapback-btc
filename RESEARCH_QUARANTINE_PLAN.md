# Research Quarantine Plan — snapback-btc

**Status:** PLAN ONLY. Nothing in this document has been executed. Do not run any
moves until after the trip and after the bot is stopped cleanly.

**Generated:** 2026-06-05 (pre-trip hygiene pass, read-only)

## Why this exists

The repo root and `strategy/`, `tools/`, `reports/`, `tests/`, `config/` hold ~148
untracked research artifacts (verdict docs, plan docs, experiment runners, signal
modules, backtest JSONs, HTML reports). They clutter `git status` and the root
listing. They are safe to relocate into a `research/` subtree **eventually**, but
NOT blindly: a large fraction of them are imported or read-by-name by other
untracked tooling, and a handful of `reports/*.json` are read by **tracked**
(committed) code and the test suite. Moving those breaks imports and tests.

This plan separates the files into **DO-NOT-MOVE** (referenced) and
**SAFE-TO-MOVE** (leaf artifacts, no inbound references), with the grep evidence.

## Hard guardrails (apply to ANY future execution)

- Do this only when the bot is **stopped** (no live process importing modules).
- The LIVE strategy chain is `bot.py` → `strategy/live_multifactor_v1.py` →
  `strategy/signals_multifactor.py` → `strategy/indicators.py`. **None** of these
  import any untracked research signal module. Verified: the only `divergence`
  mentions in `signals_multifactor.py` are code comments (lines 75, 187), not
  runtime imports. Do not touch the four live files regardless.
- `bot.py` loads these configs by path — all tracked, all DO-NOT-MOVE:
  `config/params.yaml`, `config/params_donchian.yaml`,
  `config/params_cnh_hybrid_short.yaml`, `config/params_cnh_hybrid_short_sol.yaml`.
- After any move, run `pytest` and re-run `tools/aggregate.py`'s test
  (`tests/test_aggregate.py`, `tools/tests/test_runner_smoke.py`) BEFORE committing.
- Move by `git mv` only for tracked files; these are untracked, so a plain `mv`
  is fine — but update every referencing path in the same change.

## Category A — DO NOT MOVE (referenced by code/tests/tooling)

### A1. The entire `strategy/signals_*.py` research cluster — DO NOT MOVE
Every untracked signal module is imported by at least one untracked runner/test,
and many import each other (v2 variants import `signals_adaptive_trend_v2`, which
imports `signals_adaptive_trend`). Moving any one breaks the import chain of the
others. Treat as one coupled unit. Referenced modules (each has inbound imports):

- `signals_adaptive_trend.py` (imported by ~40 runners + sibling signals + `tests/test_adaptive_trend.py`)
- `signals_adaptive_trend_v1_funding_skip.py`, `_v1_rv_band.py`, `_v1_vol_scaled_sizing.py`
- `signals_adaptive_trend_v2.py` and all `_v2_*` variants (funding_skip, half_out_at_1R,
  mtf_h1_confirmation, regime_gate_adx, regime_gate_vol, session_volume_filter, vol_scaled_sizing)
- `signals_adx_dual_regime.py` (imported by `tools/run_strategy_experiment.py`, `tests/test_adx_donchian.py`, etc.)
- `signals_divergence.py`, `signals_divergence_v2.py` (imported by `tools/run_strategy_experiment.py`,
  `tests/test_divergence.py`, `_divergence_final_verdict.py`)
- `signals_kc_squeeze.py` (imported by `tools/_postfrac_kc_squeeze.py`)
- `signals_turn_of_candle_15m.py` (imported by `tools/_postfrac_turn_of_candle_15m.py`)
- `signals_volume_profile.py` (imported by `tools/_fractional_run.py`, `tests/test_volume_profile.py`)

If you ever DO move this cluster, move `strategy/signals_*` + their `tools/` runners
+ their `tests/` together, and run `pytest` after. Safest option: **leave in place**.

### A2. `reports/*.json` read by TRACKED code — DO NOT MOVE
These are referenced by name in committed tooling / the test suite. Moving them
breaks `tools/aggregate.py` and tests:
- `reports/postfrac_mf_4h_btc.json` — read by **tracked** `tools/aggregate.py`,
  `tests/test_aggregate.py`, `tools/portfolio_psr.py`, `tools/cost_stress_psr.py`,
  `tools/tests/test_runner_smoke.py`, plus several untracked runners.
- `reports/turn_of_candle_15m_oos.json` — read by `tools/tests/test_runner_smoke.py` (tracked test harness).
- `reports/adaptive_trend_walk_forward*.json` — referenced by name in `tools/aggregate.py` docstring/logic.
- Any `reports/*.json` referenced by name in a runner you intend to keep runnable —
  grep before moving (see "Execution recipe" below).

### A3. Untracked runners/tests that reference kept artifacts — DO NOT MOVE casually
`tools/_*.py`, `tools/adaptrend_*.py`, `tools/run_*.py`, and `tests/test_*.py`
import the A1 signals and read/write the A2 reports. They are the consumers; moving
them without moving their inputs (or vice versa) breaks paths. Move only as whole
experiment bundles, post-trip, with `pytest` as the gate.

## Category B — SAFE TO MOVE (leaf artifacts, no inbound code references)

These have **no** inbound import or read-by-name from any `.py`. They are
human-readable outputs. Verified unreferenced:
- `config/params_adaptive_trend_v2.yaml`, `config/params_adx_dual_regime.yaml`,
  `config/params_divergence.yaml` — grep `*.py` for each: zero references; NOT loaded by `bot.py`.
- Root `*_VERDICT.md`, `*_PLAN.md`, and research reports:
  `ADAPTIVE_TREND_*_VERDICT.md`, `ADX_DUAL_REGIME_PLAN.md`, `AFK_REPORT.md`,
  `BTC_SOL_PORTFOLIO_VERDICT.md`, `CHOPREVERTER_PLAN.md`, `DIVERGENCE_PLAN.md`,
  `FRACTIONAL_SIZING_REFACTOR_VERDICT.md`, `FUTURE_DIRECTIONS.md`,
  `HYBRID_SHORT_PLAN.md`, `PUMPFADE_RESULTS.md`, `RESEARCH_PNL_FINDINGS.md`,
  `RV_BAND_VERDICT.md`, `SHORT_TF_RESEARCH_REPORT.md`.
  (Confirm none are linked from `README.md`/`DEPLOY.md` you want to keep clickable
  before moving — `*_VERDICT.md` are pure prose, low risk.)
- Root research HTML: `ICNH_EXPERIMENT.html`, `ICNH_EXPERIMENT_V2.html`.
- `reports/*.json` with NO inbound reference, e.g. `reports/volume_profile_poc_1M.json`
  (grep returned `<none>`). **Each report must be individually grep-checked** — do
  not move the `reports/` folder wholesale; A2 lives inside it.

**Rule for B:** even "safe" moves should be grep-confirmed per file at execution
time (files may gain references between now and then), and done in a separate
commit from any code change.

## Execution recipe (POST-TRIP, bot stopped)

1. `mkdir -p research/{verdicts,plans,html,configs}`
2. For each candidate file, re-confirm it's unreferenced across ALL referencing
   file types (not just `.py` — a config/report can be named in a shell, deploy,
   Makefile, YAML, or doc):
   ```
   grep -rln "BASENAME" . \
     --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' \
     --include='*.json' --include='*.md' --include='*.toml' --include='Makefile' \
     | grep -v '/.venv/' | grep -v '/.git/'
   ```
   Empty result → safe. Any hit → leave it (it's a consumer's input).
3. Move Category B files only. Plain `mv` (they're untracked).
4. Run `pytest -q` and `python tools/aggregate.py` smoke — must stay green.
5. Add a `.gitignore` entry or `git add` the relocated tree, commit by explicit path.
6. Leave Category A entirely in place until you decide to relocate whole bundles.

## One-line summary

~148 untracked research files: the `strategy/signals_*.py` cluster and the
`tools/`+`tests/` runners are tightly import-coupled and several `reports/*.json`
are read by tracked `aggregate.py`/tests — all DO-NOT-MOVE; only leaf docs
(`*_VERDICT.md`, `*_PLAN.md`, research `*.html`, the 3 unreferenced
`config/params_*.yaml`, and individually-grep-clean `reports/*.json`) are
safe to relocate into `research/`, and only with a per-file grep re-check and
`pytest` gate after the trip.
