# DEPLOY_COMBINED.md — Two-bot deploy via Binance sub-account isolation

## Why sub-accounts

I previously designed for "single account + hedge mode" to isolate the two bots. **That was wrong.** Binance hedge mode has ONE LONG slot + ONE SHORT slot per symbol per account — `positionSide` identifies a slot, not a bot. Two bots both going long would share the same slot, with desynced state.db's against the exchange. See the design-bug discussion in the chat log on 2026-05-23.

**Real isolation requires separate accounts.** Binance sub-accounts cost nothing, take ~15 minutes to set up, and are the recommended path for running multiple algorithms on the same Binance umbrella.

## Final deploy plan

| | Main account (v1) | Sub-account (Donchian) |
|---|---|---|
| Wallet | $50.50 USDT | $50.50 USDT |
| Strategy | `multifactor-v1` (15m) | `donchian-v3` cons (4h) |
| Risk per trade | 2.75% | 2.75% |
| Kill-switch | −35.5% start-anchored | −35.5% start-anchored |
| Position mode | one-way (default) | one-way (default) |
| API key | existing key | NEW sub-account API key |
| Env file on droplet | `/root/snapback-btc/.env` | `/root/snapback-btc/.env.donchian` |
| State.db | `data/state.db` | `data/state_donchian.db` |
| Log file | `logs/bot.jsonl` | `logs/donchian.jsonl` |
| Systemd unit | `snapback-btc.service` (existing) | `snapback-btc-donchian.service` (new) |
| clientOrderId prefix | `snap-v1-` | `snap-d3-` |

Expected on $101 over 6.7-yr backtest analogue: $101 → **$812.79**, **$8.90/mo**, Sharpe 0.85, peak-DD −33%, kill never trips. (`FULL_HISTORY.html` §7.)

## Step 1 — Create the sub-account in Binance

(Web UI, ~5 minutes)

1. Binance home → **Account** → **Sub-Accounts** (left sidebar). If you don't see it, you may need to apply for the feature — Binance approves most retail accounts within 24h.
2. Click **Create Sub Account**. Pick any email alias and a label like `snapback-donchian`. Note the email — you can't change it.
3. Once created, click the sub-account row → **Transfer**. Transfer **$50.50 USDT** from main → sub-account's **Futures wallet** (not Spot). Use the "USDM-Futures" wallet on the sub-account side.
4. Verify in the sub-account's Futures wallet shows $50.50.

Then withdraw $50.50 from the main Futures wallet → leaves $50.50 in main Futures for the v1 bot.

After this step, main has $50.50 and sub-account has $50.50. Both in Futures wallets.

## Step 2 — Generate the sub-account's API key

In the sub-account view:

1. **API Management** → **Create API**.
2. Permissions: **Enable Futures** (read + trade). DISABLE Spot, Margin, Withdrawals.
3. **IP whitelist**: add the droplet's static IP (152.42.241.43). This is critical — never expose a production key without IP whitelisting.
4. Copy the API key + secret. **You only see the secret ONCE.**

Save these securely. Don't commit to git.

## Step 3 — On the droplet, create the second env file

```bash
sudo cp /root/snapback-btc/.env /root/snapback-btc/.env.donchian
sudo chmod 600 /root/snapback-btc/.env.donchian
sudo chown snapback:snapback /root/snapback-btc/.env.donchian  # or root:root, match the main env file
sudo nano /root/snapback-btc/.env.donchian
```

Replace `BINANCE_API_KEY` and `BINANCE_API_SECRET` with the sub-account's new keys. Keep `BINANCE_ENV=mainnet` and the rest of the file the same.

## Step 4 — Pull the new code on the droplet

```bash
cd /root/snapback-btc  # or wherever the live deploy lives (e.g., /root/snapback-btc)
git pull origin main
.venv/bin/pip install -e .  # no new deps expected, safe to run
```

Files added/changed by this work (see `git log` for the deploy commit):
- `config/params.yaml` — risk 2.75%, kill −35.5%, coid prefix
- `config/params_donchian.yaml` — NEW Donchian leg config
- `strategy/live_donchian_v3.py` — NEW live signal evaluator
- `bot_internals.py` — added donchian-v3 dispatch
- `bot.py` — added `--instance v1|donchian` (canonical) plus `--config / --state-db / --log-file / --heartbeat` overrides
- `exchange/state.py` — `set_db_path()` for second-instance redirect
- `exchange/binance_client.py` — configurable coid_prefix + hedge-mode plumbing (unused in sub-account path; kept for future use)
- `deploy/snapback-btc-donchian.service` — NEW systemd unit

## Step 5 — Stop the existing v1 bot cleanly

```bash
# If running under tmux:
tmux send-keys -t bot 'C-c' && sleep 5
tmux kill-session -t bot

# If running under systemd:
sudo systemctl stop snapback-btc
```

Then update the main account's wallet to $50.50 (Step 1 already does this — verify in Binance UI).

## Step 6 — Restart v1 with the new config (still on main account)

```bash
# Tmux path:
tmux new-session -ds bot_v1
tmux send-keys -t bot_v1 \
  '.venv/bin/python -m bot --dry-run' Enter
```

The `--dry-run` flag is for the 14-day soak. Drop it later for live.

After 60 seconds, verify:
```bash
tail -20 logs/bot.jsonl  # should show "snapback-btc booting strategy=multifactor-v1..."
                          # "Recorded deploy_start_equity=50.50 USDT" (NOT 101)
stat -c '%Y' data/heartbeat  # should be < 30s old
```

## Step 7 — Install the Donchian systemd unit

```bash
sudo cp deploy/snapback-btc-donchian.service /etc/systemd/system/
sudo systemctl daemon-reload
```

Open the unit file and double-check `EnvironmentFile=/root/snapback-btc/.env.donchian` matches what you created in Step 3.

## Step 8 — Start the Donchian bot

```bash
# Tmux path (recommended for the 14-day dry-run):
tmux new-session -ds bot_donchian
tmux send-keys -t bot_donchian \
  'set -a; source /root/snapback-btc/.env.donchian; set +a; \
   .venv/bin/python -m bot --dry-run --instance donchian' Enter
```

`--instance donchian` derives the config path, state.db path, log file, and heartbeat file from the named profile in `bot.py:INSTANCE_PROFILES`. The `set -a; source ...; set +a` exports the sub-account credentials so the bot reads the right API key.

After 60 seconds:
```bash
tail -20 logs/donchian.jsonl
# Should see: "snapback-btc booting strategy=donchian-v3 config=config/params_donchian.yaml ..."
#             "Recorded deploy_start_equity=50.50 USDT"  ← sub-account's wallet
stat -c '%Y' data/heartbeat_donchian
```

If both bots are running cleanly, you have two independent processes against two independent Binance accounts. Their state.db's never overlap, their orders go to different exchange accounts, their kill-switches are independent.

## Step 9 — The 14-day dry-run watch

While both run in `--dry-run`:

```bash
# Hourly: both heartbeats fresh
watch -n 60 'stat -c "%Y" data/heartbeat data/heartbeat_donchian; date +%s'

# Both log files growing, no ERROR
tail -F logs/bot.jsonl logs/donchian.jsonl
grep ERROR logs/*.jsonl

# Each bot's logical equity in state.db
sqlite3 data/state.db 'SELECT * FROM meta;'
sqlite3 data/state_donchian.db 'SELECT * FROM meta;'

# Would-have-traded events (proves signals fire)
grep -E "would_long|would_short|dry_run_signal" logs/*.jsonl | wc -l
```

What's good:
- Both bots tick heartbeats every 5-60s (v1=5s, Donchian=60s by config)
- v1 fires signals on a 15m boundary (mostly logging skips at very low equity at first — that's fine, expected)
- Donchian fires signals on a 4h boundary (slower — expect 0-2 entries in 14 days at $50.50)
- No ERROR or stale-heartbeat from either

What's bad and means STOP:
- One bot's logs disappear / heartbeat goes stale → process crashed; debug
- Both bots logging the SAME `deploy_start_equity` value at boot → env files mixed up; check Step 3
- ERROR in `fetch_balance` → API key on the wrong account, or IP whitelist incomplete

## Step 10 — Promote to live

After 14 clean days:

```bash
tmux kill-session -t bot_v1
tmux kill-session -t bot_donchian

# v1 — same command, without --dry-run
tmux new-session -ds bot_v1
tmux send-keys -t bot_v1 '.venv/bin/python -m bot' Enter

# Donchian — same command, without --dry-run
tmux new-session -ds bot_donchian
tmux send-keys -t bot_donchian \
  'set -a; source /root/snapback-btc/.env.donchian; set +a; \
   .venv/bin/python -m bot --instance donchian' Enter
```

For real systemd-managed deploy later: `sudo systemctl enable --now snapback-btc-donchian` (after editing the unit's ExecStart to drop `--dry-run` if you added it).

## Rollback

If anything looks wrong on live:

```bash
# Immediate halt
touch /root/snapback-btc/data/HALT          # v1 polls and exits
tmux kill-session -t bot_donchian           # Donchian halt (HALT file would need wiring; simplest = kill tmux)

# Flatten any open positions manually in Binance UI for each account.
# Main account: BTC/USDT-perp page → close LONG / close SHORT
# Sub-account: switch to sub-account view first, then same
```

To revert the config:
```bash
git checkout HEAD~1 config/params.yaml config/params_donchian.yaml
```

## Risk reminders

- **Kill-switch is per-bot**. v1 at −35.5% halts ONLY v1. Donchian keeps trading on its own account. Intentional.
- **First live trade is real money.** Watch the first entry from each bot for fill price, qty, SL/TP placement, clientOrderId tagging. If anything looks wrong, halt that bot and reconcile.
- **Live diverges from backtest:** Donchian leg uses fixed 5×ATR TP rather than the backtest's Donchian channel-exit. The 14-day dry-run will show any drift.
- **Don't manually intervene mid-trade.** If you close a position by hand on the exchange, the bot's state.db won't know — it'll think it has a position that no longer exists. Use `touch data/HALT` to stop the bot first, then close manually.

## What I changed in the code today (summary for reviewers)

| File | Why |
|---|---|
| `config/params.yaml` | risk 2.0 → 2.75, kill 0.82 → 0.645, hedge.enabled=false + coid prefix |
| `config/params_donchian.yaml` | new — Donchian-v3 cons params, 4h, risk 2.75, kill 0.645, sub-account uses one-way mode |
| `strategy/live_donchian_v3.py` | new — pure-function live signal evaluator (channel breakout + slope gate + ATR stops) |
| `bot_internals.py` | added donchian-v3 case in `evaluate_for_strategy` |
| `bot.py` | new `--instance v1\|donchian` (canonical), plus path-override flags; reads `hedge` block from params |
| `exchange/state.py` | added `set_db_path()` for multi-instance |
| `exchange/binance_client.py` | configurable `coid_prefix` per instance; `positionSide` plumbing kept for future hedge-mode use (unused in sub-account path) |
| `deploy/snapback-btc-donchian.service` | new systemd unit pointing at `.env.donchian`, separate state/logs/heartbeat |

Memory snapshot: `snapback-deploy-config-chosen.md`, `snapback-deploy-capital-101.md`.
