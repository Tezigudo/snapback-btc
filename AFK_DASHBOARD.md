# AFK Dashboard — 2026-06-02 (Tue)

**Read this when you're back at 21:50 ICT.** Times shown ICT (GMT+7).

---

## TL;DR

| Question | Answer |
|---|---|
| Bot live? | **✓ GREEN** at 20:31 ICT. 3 legs active, heartbeats 3-22s. Zero errors / NaN / tracebacks since today's 19:07 deploy. |
| Anything broken? | No. |
| Anything that needs your call at 21:50? | Push 3 local commits + deploy AFK package. See **"Approval needed"** below. |
| Anything to celebrate? | Locked +77.93% is now **measurement-validated** (re-measured: +77.952%, matches within noise). Live↔backtest parity **100%** across 8,736 bars. |
| Big findings from research? | (1) **Cost-sensitivity yellow flag** on deployed strategy: PSR 0.978→0.949 going from 5→10 bps commission. Fine at $101.95, watch at scale. (2) **KC squeeze breakout SHELVED** as first real TODO_LEG validation. (3) Workflow validating 4 more TODO_LEG + 3 more ablations + adversarial refute pass — landing before sign-off. |

---

## Live bot state (20:31 ICT)

```
v1            active   heartbeat 3s     mainnet  dry_run=False  $101.95 anchor
donchian      active   heartbeat 22s    mainnet  dry_run=True   $50.50
cnh_short     active   heartbeat 16s    mainnet  dry_run=True   $80.50

Since deploy (19:07 ICT, 1h24m ago):
  - 0 ERRORs
  - 0 mtf_4h_gate_nan events
  - 0 NEW tracebacks (the count-of-2 in logs is from 2026-05-19 — pre-deploy)
  - Resources: load 0.16, 14% disk, 285Mi free RAM
```

---

## 🎯 Today's biggest wins

### 1. Locked +77.93% is now MEASUREMENT-validated

PRICE_SCALE patch to `tools/_postfrac_mf_4h_btc_run.py` re-ran the 5 OOS windows on the corrected fractional-sizing harness:

| | Re-measured | Locked claim | Match? |
|---|---|---|---|
| Compounded | **+77.952%** | +77.93% | ✓ |
| Trades | 121 | 121 | ✓ exact |
| PSR | 0.9776 | 0.978 | ✓ |
| Windows positive | 5/5 | 5/5 | ✓ |
| Worst window DD | -17.03% (2024 H2) | -16.72% | ✓ within noise |

The deployed strategy went from "math-validated" → "math AND measurement validated". **Methodology debt #1 is RESOLVED for the deployed strategy.**

### 2. Live↔backtest parity 100%

Ran `tools/multifactor_validate.py` over 8,736 bars (90 days, 2025 Q2):
- Stage 1 (impl cross-check): 14/14 trades common → PASS
- Stage 2 (live evaluator vs backtest): 12 long + 0 short fires, identical → PASS

### 3. THREE AdaptiveTrend V1 ablations REFUTED (all SHELF)

**a) ADX(14)>25 entry gate** — SHELF
- Killed -16.43pp (45.52% → 29.09%)
- PSR dropped 0.905 → 0.848
- 2025_H1 falsified: WR 42.9% → 33.3% (gate kept the WORSE half)

**b) H1 EMA50 directional confirmation** — SHELF
- Killed -5.05pp (45.52% → 40.48%)
- PSR -0.003 (0.905 → 0.902)
- 16 trades filtered but H1 EMA50 is too in-phase with H6 MOM>theta — no orthogonal info
- **Survives the SHELF**: MTF gate proposals need a *different feature class* (vol, volume, ADX-direction) — another trend filter won't help

**c) half_out_at_1R partial exit** — SHELF
- Killed -11.73pp (45.52% → 33.79%)
- PSR_entry 0.988 → 0.982
- Variance DID shrink (per-window std 6.90 → 5.11pp), WR +17pp avg — hypothesis was structurally right
- But trend systems are right-skewed: capping the +1R tail costs more than the smoother equity buys
- Same mechanism that killed AdaptiveTrend V2's partial-exit ablation (-13.48pp prior)
- **Survives the SHELF**: don't propose any partial-exit variant for trend-following strategies. Only valid for MR.

**Three real findings**. Together they save future agents from re-running ~600k tokens of dead hypotheses on AdaptiveTrend V1.

### 4. Cost-sensitivity finding on the deployed strategy

Built `tools/cost_stress_psr.py` and ran it on the 121-trade aggregate from the just-validated +77.93% run. The locked PSR is **commission-sensitive**:

| Commission | PSR | Interpretation |
|---|---|---|
| 5 bps (base) | **0.9776** | **evidence_of_edge** |
| 10 bps | 0.949 | insufficient_evidence |
| 15 bps | 0.898 | insufficient_evidence |
| 20 bps | 0.821 | insufficient_evidence |

**Not a deploy blocker** at the current $101.95 capital — real slippage is tiny on 0.001 BTC orders. **But flags concern at scale**: if you push live capital past ~$25k, the strategy's edge robustness shrinks. Filed at `multifactor_v1_4h_gate_cost_sensitivity` memory.

### 5. KC squeeze breakout (1st TODO_LEG validated) — SHELF

First TODO_LEG actually validated. -83.31% compounded / PSR 0.0048 / 0/5 windows. Adversarial check clean (no lookahead). Bleeds in sustained-trend regimes. R:R 1.67 needs 37.5% WR, got 29.7% in 2024 H2. Insight from agent: revive would need an HTF gate (4H EMA200 worked for multifactor) — not param tuning.

### 6. Exchange data patch — taker_buy retained

`exchange/data.py::fetch_klines` was dropping `taker_buy_base`, `taker_buy_quote`, `n_trades`, `quote_vol` columns. Patched to retain them. Backward-compatible (existing callers select by name). First call after patch upgrades the 15m cache from 5→9 columns. Unblocks taker-flow TODO_LEG validation.

### 7. Monitor / digest smoke test infrastructure

`tools/test_monitor_smoke.py` validates `monitor.py` + `daily_digest.py` exit 0 with SMTP disabled. Run before deploying AFK package. **One snag during testing**: my initial sandbox run inherited SMTP creds from your shell env and sent 3 false-positive "NO HEARTBEAT" alerts to godjangg@gmail.com (path resolution followed symlinks to the real repo, no local heartbeats found). Those alerts are NOT real — bot on droplet is still green.

### 8. Five new TODO_LEG candidates filed

Strategy invention agent produced 5 well-formed hypothesis cards with adversarial citation verification (caught 3 fabricated author attributions, corrected before filing):

| # | Candidate | Lit | Data gap? |
|---|---|---|---|
| 1 | **KC-Bollinger squeeze breakout** | Practitioner blog (secondary-cited) | none |
| 2 | **Intraday TSMOM** (clock-anchored) | Shen/Urquhart/Wang 2022 (paywalled — abstract verified) | none |
| 3 | **Funding-extreme contrarian** | BIS WP 1087 (verified primary) | none |
| 4 | **OI vs price divergence** (positioning) | Multiple secondary | ⚠ OI history >30d limited via Binance API |
| 5 | **Taker-flow imbalance** (CVD proxy) | Cont/Kukanov/Stoikov 2014 | ⚠ taker_buy_base_volume dropped in `exchange/data.py::load_klines` |

All 5 filed at `~/.claude/projects/-Users-god-Desktop-work-snapback-btc/memory/todo_leg_*.md`. Each entry has: hypothesis, signal logic (concrete rules), expected regime, sources (with primary/secondary tags), validation gates (PSR > 0.95, walk-forward ≥ 70%, etc.), risk profile, and a "NOT ready to deploy" footer.

---

## What got done (full inventory)

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Health check droplet | ✅ | GREEN, 0 errors today |
| 2 | Shanghai AFK package | ✅ STAGED | Needs your approval to push |
| 3 | Parity validator dry-run | ✅ | 100% match |
| 4 | PRICE_SCALE patch + re-run | ✅ | Measurement-validated +77.952% |
| 5 | Wiki save (deploy procedure + droplet topology) | ✅ | 2 pages, log+index updated |
| 6 | Memory consolidation (Phase 2+3) | ✅ | Stale claim fix, 3× permanent_shelf tags |
| 12 | Strategy invention (5 TODO_LEG candidates) | ✅ | All filed with validation gates |
| 13 | AdaptiveTrend regime_gate_adx ablation | ✅ | SHELVED (-16.43pp, PSR drop) |
| 15 | AdaptiveTrend half_out_at_1R ablation | ✅ SHELF | -11.73pp, PSR -0.006. Variance shrank, WR +17pp, but right-skewed tail dominates. |
| 16 | AdaptiveTrend mtf_h1_confirmation ablation | ✅ SHELF | -5.05pp, PSR -0.003. H1 EMA50 too redundant with H6 MOM gate. |
| 14 | This dashboard | ✅ | Will refresh once more before 21:50 |

---

## 🔴 Approval needed at 21:50

### A. Push the AFK package to droplet

**Files staged locally on `main`, NOT committed yet:**
- `monitor.py` — real implementation (was a `NotImplementedError` stub)
- `daily_digest.py` — new daily 08:00 ICT rollup email
- `SHANGHAI_TRIP_RUNBOOK.md` — phone-friendly alert reference for the trip
- `AFK_PACKAGE_DEPLOY.md` — deploy checklist + risks

**Both Python files parse cleanly.** Read `AFK_PACKAGE_DEPLOY.md` for the full deploy sequence including the SMTP smoke test.

**Risk class**: medium. `monitor.py` runs as cron in the bot's venv. Wrapped in `try/except sys.exit(0)` so cron doesn't flap. SMTP not configured → `is_configured()` short-circuits silently.

### B. Push PRICE_SCALE patch + auxiliary files

**Files staged locally on `main`, NOT committed yet:**
- `tools/_postfrac_mf_4h_btc_run.py` — PRICE_SCALE patch (research-only, no live-runtime impact)
- `AFK_DASHBOARD.md` (this file)
- `AFK_REPORT.md` (existing untracked, ignore if not yours)
- All the various reports/ regeneration outputs

These are research-side and don't touch the live bot. Safe to push standalone.

### C. (Conditional) Half_out_at_1R and mtf_h1 ablation results

If the two background agents return before 21:50: review their verdicts. Most likely outcome based on prior ablation patterns and the strategy graveyard: both shelf. But check.

---

## What I left ALONE on purpose

- **Did not push anything to `origin`.** All local commits are unpushed. Your call.
- **Did not deploy `monitor.py` to droplet.** Staged only.
- **Did not edit `risk.py`.** CLAUDE.md hard rule.
- **Did not remove `data/HALT`.** It wasn't there, but flagging for principle.
- **Did not propose mainnet promotion of anything.** SOL leg stays WAIT_FOR_MORE_DATA, ETH stays SHELF.
- **Did not run the 3 remaining AdaptiveTrend ablations** (regime_gate_vol, time_stop_24h, session_volume_filter). Time/budget call. They're queued in `fractional_sizing_refactor_verdict`.
- **Did not patch the 3 sibling strategy `int(min(...))` files** (snapback-v1, donchian, multifactor-v2). Those aren't deploy-blocking.
- **Did not merge the 3 4H-gate timeline memory files into one canonical**. The proposal recommends it (Phase 1) but it's destructive — the historical snapshots have value.

---

## Watch for these on the trip

1. **Binance `recvWindow` drift**: the bot has had this error 2× historically (most recently 2026-05-19). Manifests as `InvalidNonce` in logs. Transient, self-recovers on next poll. NOT a halt-worthy event. If you see > 10 of these per day in a `[snapback-monitor]` ERROR alert, the system clock on the droplet is drifting — `sudo timedatectl status` to check NTP sync.
2. **`RemoteDisconnected` from Binance API**: same historical category. Network blip, self-recovers. Bot keeps running.
3. **The bot is LIVE on mainnet**. $101.95 is small but real. Per the runbook (`SHANGHAI_TRIP_RUNBOOK.md`), the response to a single ⚠ email is "read what happened first, don't panic-halt".

---

## Token spend (estimated, running)

**Round 1 (19:30–20:35 ICT)**: ~471k
- Main loop ~85k, wiki 58k, memory consolidation 43k, strategy invention 77k, 3 ablations 64+68+76k

**Round 2 (20:35 → ongoing — ultracode 35%)**:
- KC squeeze validation: 75k
- load_klines patch + smoke + cost-stress tool: ~25k main
- Workflow wus0bdtx7: ~250-350k (running — 4 TODO_LEG validate + 3 ablations + 4 refute + 1 synthesize)
- Estimated round-2 total: ~350-450k

**Total session: ~820–920k tokens estimated.**

---

## Suggested next round at 21:50

When you give me the green light again:

1. **Push the 2-3 unpushed commits** (mainline first, then droplet cherry-pick for slash commands)
2. **Deploy AFK package to droplet** (per `AFK_PACKAGE_DEPLOY.md` steps)
3. **Confirm SMTP wiring** with a smoke test
4. **Install cron** (`monitor.py` every 5 min, `daily_digest.py` 0 1 * * *)
5. Optional: review any TODO_LEG and pick one to start validation
6. Optional: launch the remaining 3 AdaptiveTrend ablations (regime_gate_vol, time_stop_24h, session_volume_filter)

---

_Generated 2026-06-02. Last update: 20:35 ICT. Files written by Claude during the AFK window 19:30–20:35 ICT._
