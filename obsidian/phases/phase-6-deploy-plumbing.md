---
tags: [phase, pending]
gate_result: Code done; exchange-side work pending
deploy_target: 2026-06-25 (capital ETA)
---

# Phase 6 — Deploy plumbing

The code side is wired. The exchange-side steps wait for the user's capital
top-up.

## Code wired (this session)

- `../../config/params_cnh_hybrid_short.yaml` — locked deploy config
- `../../strategy/live_cnh_hybrid_short.py` — live evaluator
- `../../strategy/cnh_detectors.py` — detectors (production-safe)
- `../../bot_internals.py` — added dispatch + `gate_status` branch
- `../../bot.py` — added `"cnh_short"` to `INSTANCE_PROFILES`
- `../../deploy/snapback-btc-cnh-hybrid-short.service` — systemd unit
- `../../tests/test_cnh_hybrid_short.py` — 7 tests passing

## Exchange-side pending

When capital arrives:

1. Create Binance sub-account #3 (label: `snapback-cnh-hybrid-short`)
2. API key: Futures read+trade only, IP-whitelist droplet 152.42.241.43
3. Transfer ≥$100 USDT to that sub-account's Futures wallet
4. On droplet: create `/root/snapback-btc/.env.cnh_short` (template from
   `.env.donchian`)
5. `git pull` to get this session's code
6. Pre-flight: `uv run python -m tools.preflight_live` (may need a
   `--strategy cnh_hybrid_short` patch; check before running)
7. Dry-run ≥7 days: `uv run python -m bot --instance cnh_short --dry-run`
8. Mainnet lock: `echo "cnh-short deploy $(date -u)" > confirm_mainnet.lock`
9. Live: `tmux new-session -ds bot_cnh; tmux send-keys -t bot_cnh '.venv/bin/python -m bot --instance cnh_short' Enter`
10. Or systemd: `sudo systemctl enable --now snapback-btc-cnh-hybrid-short.service`

## Capital prerequisite

≥$100 to the sub-account. See [[decisions/deploy-capital-floor]] for why.

## See also

- `../../AFK_REPORT.md` — has the same checklist with step numbers
- [[decisions/deploy-capital-floor]]
- [[HYBRID-short-strategy]] — root index
