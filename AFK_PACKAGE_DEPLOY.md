# AFK package deploy — staged but NOT pushed

**Status:** staged locally on `main`. **NOT yet pushed, NOT yet on droplet.**

This is the deploy procedure for the Shanghai AFK safety package. Three new files:

- `monitor.py` — replaces the `NotImplementedError` stub with a real cron-invoked health checker (5-min interval)
- `daily_digest.py` — new file, sends a single rollup email at 08:00 ICT daily
- `SHANGHAI_TRIP_RUNBOOK.md` — operator phone-friendly reference for what each alert means

## Why this is staged (not deployed)

The bot is **LIVE on mainnet**. Per the advisor's invariant for this AFK window: nothing mutates the live droplet without explicit operator approval. The three files above don't modify bot.py or strategy code, but `monitor.py` will run inside the same Python venv and import `alerts.py` (which imports from `exchange/`). If any of that breaks, the alert system fails silently, which itself is a class of risk.

## Pre-deploy checklist (user approves at 21:50)

```
[ ] Read SHANGHAI_TRIP_RUNBOOK.md to confirm the alert kinds match what you want
[ ] Decide if monitor cooldown 30min is right (in DEFAULTS at top of monitor.py)
[ ] Confirm SMTP_HOST/PORT/USER/PASSWORD/ALERT_EMAIL_FROM/ALERT_EMAIL_TO are set in droplet /root/snapback-btc/.env
[ ] (Optional) Set a per-leg ALERT_TAG override if you want separate Gmail threads
```

## Deploy sequence (operator-run after approval)

1. **Verify SMTP wiring works first** — never deploy alerting code if alerts can't actually send.
   ```
   ssh root@152.42.241.43
   cd /root/snapback-btc
   .venv/bin/python alerts.py "smoke test from main" "SMTP wiring check"
   # → expects "ok" output AND an email in your inbox
   ```
   If FAILED: fix `.env` SMTP_* first, do not proceed.

2. **Commit + push from local main** (Claude can do this once approved):
   ```
   git add monitor.py daily_digest.py SHANGHAI_TRIP_RUNBOOK.md AFK_PACKAGE_DEPLOY.md
   git commit -m "[feat] AFK package: real monitor.py + daily digest + Shanghai runbook"
   git push origin main
   ```

3. **Cherry-pick onto droplet branch** (preserves the strip pattern):
   ```
   git checkout droplet
   git cherry-pick <sha>
   git push origin droplet
   git checkout main
   ```

4. **Pull on droplet** (no bot restart — these files don't touch the runtime):
   ```
   ssh root@152.42.241.43
   cd /root/snapback-btc
   git pull --ff-only origin droplet
   .venv/bin/python monitor.py     # dry-run; should report no alerts on green state
   .venv/bin/python daily_digest.py  # send one digest right now to confirm
   ```

5. **Install cron** on the droplet:
   ```
   crontab -e
   # add:
   */5 * * * * /root/snapback-btc/.venv/bin/python /root/snapback-btc/monitor.py >> /root/snapback-btc/logs/monitor.log 2>&1
   0 1 * * *   /root/snapback-btc/.venv/bin/python /root/snapback-btc/daily_digest.py >> /root/snapback-btc/logs/digest.log 2>&1
   # 1 UTC = 08:00 ICT
   ```

6. **Confirm the first cron fire** at the next 5-min boundary:
   ```
   ssh root@152.42.241.43
   tail -f /root/snapback-btc/logs/monitor.log
   # wait up to 5 min — should see "monitor: ..." log lines and silence (no alerts in green state)
   ```

## Backout

If `monitor.py` starts spamming false-positive emails:
```
ssh root@152.42.241.43
crontab -e   # comment out the monitor.py line (leave digest)
```
That's it — no bot restart needed, monitor runs in its own process.

## Risks

| Risk | Mitigation |
|---|---|
| monitor.py crashes inside cron | Wrapped in `try/except` at module level — `sys.exit(0)` on any unhandled exception. Cron sees clean exit, doesn't retry-loop. |
| monitor.py spams emails | 30-min cooldown per alert kind (configurable in monitor.yaml). Plus is_configured() short-circuits if SMTP not set. |
| Log offset state corrupt | `_load_state` catches JSONDecodeError, starts fresh. Worst case: monitor re-alerts on old events once. |
| systemd is-active flapping | Captured via subprocess timeout; on subprocess error, _systemd_active returns True (don't alert on transient failure). |
| Equity DB locked by live bot | `connect(..., mode=ro)` + 2s timeout; on any sqlite3.Error, drops to "unknown anchor", no alert. |
| Daily digest sends at restart-prone hour | 08:00 ICT is mid-day in business hours; bot is mid-loop. Read-only access only. |

## What's NOT in this package (deliberately)

- **No Telegram bot wiring**. Email-only. Telegram adds a webhook + token, doubles the failure surface. If you want Telegram, file as a TODO_LEG.
- **No automated `/halt`** on equity drop. The kill switch at -15% already exists in `risk.py`. Layered automation = layered failure modes.
- **No mobile app push**. Email is the universal denominator.
- **No re-validation against backtest**. Parity checks are operator-triggered via `tools/multifactor_validate.py`, not cron'd. Catching parity drift overnight is fine.

---

_Staged 2026-06-02 by Claude. Deploy after 21:50 ICT operator approval._
