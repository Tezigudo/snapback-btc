---
description: Guided pre-mainnet checklist — testnet to real money
---

**This command moves the bot to REAL MONEY. Stop if any item below is unchecked.**

**Time display rule:** any timestamps in the checklist output (deploy timestamp, first-heartbeat-on-mainnet, alert times) are shown to the user in GMT+7 ICT. UTC stays in the bot/config.

Pre-flight:
1. `verify_identity(action_type="bot_mainnet")` — if BLOCKED, STOP and ask the user.
2. Confirm 7+ days of clean testnet operation: `/status` shows ≥10 trades, zero unhandled errors, heartbeat continuous.
3. Confirm walk-forward + OOS metrics are within tolerance (see `/backtest`).
4. Confirm `risk.py` ceilings are appropriate for the user's intended capital. Read them back to the user explicitly:
   - `MAX_NOTIONAL_USD`
   - `MAX_LEVERAGE`
   - `MAX_DAILY_LOSS_PCT`
5. Confirm VPS deploy is ready: `systemd` unit installed, IP-whitelisted on Binance, secrets in `/etc/snapback-btc/.env` (mode 600).
6. Confirm Telegram alerting is working — send a test alert before proceeding.

Activation (user must do these manually — do NOT do them yourself):
1. User edits `.env` on VPS: `BINANCE_ENV=mainnet`
2. User creates lockfile: `touch confirm_mainnet.lock` on VPS
3. User restarts: `systemctl restart snapback-btc`
4. User confirms first heartbeat on mainnet via Telegram alert.

Initial cap: **$100 seed only.** No exceptions for the first 30 days. Increase only after a clean month.

After activation: monitor every few hours for the first 48h. Anything weird → `/halt`.
