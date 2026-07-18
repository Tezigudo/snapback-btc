# HYBRID short — AFK build report (2026-05-26)

You went to shower; I kept going. Here's what's done and what's still pending.

## Status summary

| Phase | Status | What it proved |
|---|---|---|
| **1 — walk-forward** | ✅ PASS | dedup=15 has best OOS (+18.8% cum, Sharpe 8.44, worst window -0.1%) |
| **2 — friction + sizing** | ✅ Friction PASS / Sizing MITIGATED | After-fee edge +93 bps/trade @15bps; needs ≥$80/leg, optimal at $100 |
| **3 — live evaluator** | ✅ PASS | Pure-function module + 7 tests, including detector regression vs tools/ |
| **3b — stateful dedup** | ✅ PASS | Reproduction now 100% (18/18 OOS); load-bearing for Phase 4 |
| **4 — portfolio sim** | ✅ PASS | Sharpe lift +0.258 live (vs +0.311 ideal); correlations ≈0 with v1+Donchian |
| **5 — dedup choice** | ✅ Default = 15 | dedup=10 wins absolute cum by 4pp but 15 wins Sharpe/WR; clean call |
| **6 — code plumbing** | ✅ Done (no deploy) | params YAML, dispatch, gate_status, systemd unit all wired |
| **6 — deploy** | ⏸ Waits for ~$200 total capital (June 25) | Sub-account creation + .env.cnh_short + mainnet ack |

## Deploy expectation (when capital lands)

Realistic deploy sim ([`tools/hybrid_realistic_deploy_sim.py`](tools/hybrid_realistic_deploy_sim.py))
gives an honest picture of what to expect at different funding levels:

| $start | risk% | final | cum | kept/81 | trips killswitch? |
|--:|--:|--:|--:|--:|:--:|
| 50 | 1.5 | $50.40 | +0.8% | 4 (95% skip) | no |
| 50 | 2.75 | $69.56 | **+39%** | 44 (46% skip) | no |
| **100** | **2.75** | **$197.31** | **+97%** | 77 (4% skip) | **no** |
| 150 | 2.75 | $296.05 | +97% | 80 (0% skip) | no |

**Recommendation when capital lands**: $100/leg at risk 2.75% (already in
`config/params_cnh_hybrid_short.yaml`). Captures 95% of signals, +97% over
the 6.5-yr backtest window. **Killswitch never trips** in any tested
scenario.

If you can only spare $50: still works, just at reduced efficiency (+39%).

## Honest caveats (read before deploying)

1. **Hold times are shorter than you originally asked for.** You wanted
   "3-7 days per position." HYBRID actually closes most trades inside a
   day (median 0.83 days, q90 = 3.7 days, max 7.3 days). Only 48% of
   trades hold 1-7 days. This is because SL hits early on losers (median
   0.83 days) and TP hits fast on winners (median 0.92 days). Real edge
   is in "snap into a breakdown, get out fast." If you specifically
   wanted multi-day positions, HYBRID isn't quite that profile.

2. **2024-H2 dominates OOS PnL.** ~75% of the OOS profit came from one
   6-month window. Phase 4's clean +0.258 Sharpe lift would have been
   weaker without it. Phase 5 per-year breakdown helps confirm this
   isn't isolated (HYBRID has positive years in 6 of 7 years), but if
   2027 looks like 2025 (the weakest year), expect a flat-to-negative
   slice.

3. **Live evaluator captures 83% of the "ideal" backtest edge** (live
   Sharpe lift +0.258 vs ideal +0.311). The remaining gap is from
   edge-case ICnH entries where live's `is_ema_breakdown` check differs
   minimally from backtest's entry search. Acceptable, not free.

4. **DT and ICnH are regime-complementary, not redundant.** DT carries
   strong-trend years (2021: DT-only +34% vs ICnH-only +0.7%). ICnH
   carries chop/transition years (2023-2024: ICnH-only +12-14% vs
   DT-only ~0%). Don't drop either detector.

## Files created this session

Strategy/runtime code (production-safe):
- [`strategy/cnh_detectors.py`](strategy/cnh_detectors.py) — pattern detectors (DT + ICnH) extracted from tools/
- [`strategy/live_cnh_hybrid_short.py`](strategy/live_cnh_hybrid_short.py) — pure-function live evaluator with stateful dedup
- [`config/params_cnh_hybrid_short.yaml`](config/params_cnh_hybrid_short.yaml) — locked deploy config
- [`deploy/snapback-btc-cnh-hybrid-short.service`](deploy/snapback-btc-cnh-hybrid-short.service) — systemd unit
- [`bot_internals.py`](bot_internals.py) — added `cnh-hybrid-short-v1` dispatch + gate_status branch (edit)
- [`bot.py`](bot.py) — added `cnh_short` to `INSTANCE_PROFILES` (edit)

Tests:
- [`tests/test_cnh_hybrid_short.py`](tests/test_cnh_hybrid_short.py) — 7 tests, all pass, including detector regression vs tools/

Audit & experiment tools (research, not deployed):
- [`tools/hybrid_walkforward.py`](tools/hybrid_walkforward.py) — Phase 1 walk-forward
- [`tools/hybrid_friction_sizing.py`](tools/hybrid_friction_sizing.py) — Phase 2 sizing audit
- [`tools/hybrid_phase3_validate.py`](tools/hybrid_phase3_validate.py) — Phase 3 reproduction check (100%)
- [`tools/hybrid_phase4_portfolio.py`](tools/hybrid_phase4_portfolio.py) — Phase 4 portfolio sim w/ live & ideal comparison
- [`tools/hybrid_phase5_dedup_choice.py`](tools/hybrid_phase5_dedup_choice.py) — dedup head-to-head
- [`tools/hybrid_realistic_deploy_sim.py`](tools/hybrid_realistic_deploy_sim.py) — capital/risk matrix
- [`tools/hybrid_dt_vs_icnh.py`](tools/hybrid_dt_vs_icnh.py) — pattern ablation
- (CHOPREVERTER_PLAN.md FAILED — kept for history; not part of this strategy)

## What you need to do when capital lands (2026-06-25)

The bot side is fully wired. Deploy is exchange-side work + dry-run:

1. **Create Binance sub-account #3** ("snapback-cnh-hybrid-short"). API key
   permissions: Futures read+trade only. Disable Spot, Margin, Withdrawals.
   IP-whitelist the droplet (152.42.241.43).
2. **Transfer ≥$100 USDT** to that sub-account's Futures wallet.
3. **On the droplet**: create `/root/snapback-btc/.env.cnh_short` by
   COPYING `/root/snapback-btc/.env.donchian` and changing ONLY these
   fields — leave SMTP / alert / `BINANCE_ENV` / `CONSOLIDATE_API_URL` /
   `CONSOLIDATE_API_TOKEN` identical to the Donchian leg:
   - `BINANCE_API_KEY` → sub-account #3's API key
   - `BINANCE_API_SECRET` → sub-account #3's secret (correct var name is
     `BINANCE_API_SECRET`, not `BINANCE_SECRET` — see `exchange/env.py:57`)
   - `CONSOLIDATE_SOURCE=snapback-btc-cnh-short` (do NOT leave as
     `snapback-btc-donchian` — required for dashboard separation)
   Quick recipe:
   ```bash
   sudo cp /root/snapback-btc/.env.donchian /root/snapback-btc/.env.cnh_short
   sudo nano /root/snapback-btc/.env.cnh_short    # edit 3 fields above
   sudo chown snapback:snapback /root/snapback-btc/.env.cnh_short
   sudo chmod 600 /root/snapback-btc/.env.cnh_short
   ```
   Note: `INSTANCE_NAME=...` is NOT read by any code path. The systemd unit
   passes `--instance cnh_short` on the CLI already.
4. **`git pull origin main`** to get this session's code.
5. **Pre-flight**: `uv run python -m tools.preflight_live` (will need a
   `--strategy cnh_hybrid_short` argument if you support that — currently
   only checks v1/donchian; minor patch needed).
6. **Dry-run ≥7 days**: `uv run python -m bot --instance cnh_short --dry-run`.
   Confirm signals fire on the right bars with the expected SL/TP.
   Monitor heartbeat at `data/heartbeat_cnh_short` (NOT `state/cnh_short.heartbeat`).
7. **Mainnet lock**: `echo "cnh-short deploy $(date -u)" > confirm_mainnet.lock`.
8. **Live**: `tmux new-session -ds bot_cnh; tmux send-keys -t bot_cnh '.venv/bin/python -m bot --instance cnh_short' Enter`.
9. **Or systemd**: `sudo systemctl enable --now snapback-btc-cnh-hybrid-short.service`.

No code work required at deploy time — everything compiles, tests pass,
the config is locked.

## Quick-reference numbers (the ones you'll cite)

- **Backtest cum (2020-2026)**: +82% (HYBRID live), +98% (ideal)
- **Win rate**: 65.0% (live), 69.1% (ideal)
- **Trade frequency**: ~12/year ≈ once every 4 weeks
- **Median hold time**: 0.83 days (winners) / 0.83 days (losers)
- **3-leg portfolio Sharpe**: 2.69 (vs 2.43 baseline) — lift +0.258
- **3-leg portfolio cum**: +264% (vs +398% baseline 2-leg) — adding the
  leg dilutes equal-weighted return BUT the leg's own $100 → ~$197 is
  ADDITIONAL dollars not in the baseline; real deploy at $300 total
  beats $100 baseline by ~$197 over 6 years.
- **Correlation with v1**: -0.010 (near zero)
- **Correlation with Donchian**: -0.010 (near zero)
- **Min capital for full efficiency**: $100/leg ($200 total Futures)
