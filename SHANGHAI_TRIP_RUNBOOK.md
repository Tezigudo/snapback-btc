# Shanghai trip runbook — snapback-btc

**Trip dates:** 2026-06-06 to 2026-06-10 (4–8 Jun)
**Bot state at trip start:** v1 LIVE mainnet on droplet `152.42.241.43`, multifactor-v1 + 4H EMA200 regime gate. Anchor equity $101.95.

This document is what you check from your phone in Shanghai. Times below are ICT (GMT+7).

---

## TL;DR — daily routine

| Time | Action | Why |
|---|---|---|
| ~08:30 ICT | Open inbox, find the **`[snapback-digest] daily digest …`** email | Auto-rolled 24h summary. If subject contains `⚠`, read closer; if `✓ all green`, you can move on. |
| Any time | Check `[snapback-monitor]` emails | Anomaly-triggered. None → nothing happened. |

That's it for daily. If you get an `⚠` email, see the **What does X mean** section.

---

## The 4 alert kinds you might see

All come from `monitor.py` running every 5 min, or `daily_digest.py` running once at 08:00 ICT.

### 1. `DOWN: snapback-btc` (or `-donchian` / `-cnh-hybrid-short`)

Meaning: systemd reports the unit is no longer active.
Severity: **high for v1** (real money idle, no kill-switch protection), low for the DRY legs.

Action:
```
ssh root@152.42.241.43
systemctl status snapback-btc      # see why it exited
journalctl -u snapback-btc --since "1 hour ago" -p err
```
If it's a transient crash, `systemctl restart snapback-btc` brings it back. If it crashed repeatedly, **do not auto-restart** — text Claude with the journal output.

### 2. `STALE HEARTBEAT: v1 (Ns)`

Meaning: heartbeat file hasn't been touched in `N` seconds. Bot loop may be hung on an exchange API call or a deadlocked thread.
Threshold: 120s (v1 polls 5s, so 24× expected).
Severity: **medium** — bot may still be holding positions but not monitoring them.

Action: same as DOWN. SSH and check `systemctl status` + the last 200 log lines.

### 3. `ERROR x{N}: v1` / `TRACEBACK: v1`

Meaning: `N` ERROR-level events or `N` Python tracebacks logged.
Severity: **depends on content**. A single ERROR could be a rate-limit (harmless), a traceback usually isn't.

Action:
```
ssh root@152.42.241.43
tail -n 50 /root/snapback-btc/logs/bot.jsonl | grep -E '"level":"ERROR"|Traceback'
```
Read what blew up. If it's a network blip, you can ignore. If it's a strategy logic error, halt and call Claude.

### 4. `EQUITY DROP {pct}%: v1`

Meaning: current equity dropped >5% (warn) or >10% (alert) from the $101.95 anchor.
Severity: **high at 10%, watch at 5%**. Kill switch fires at 15% automatically.

Action: do NOT panic-halt. The strategy backtests show -16.72% max DD in the worst window (2024 H2). A 5-10% drawdown is within expected variance.

If alert AND you've had multiple alerts in a few days AND no obvious cause:
```
ssh root@152.42.241.43
cd /root/snapback-btc
.venv/bin/python tools/multifactor_validate.py --recent 7d
```
This compares the last week's live decisions to what the backtest would have done on the same bars. If parity is < 100%, escalate to Claude.

### 5. `4H GATE NAN: v1`

Meaning: the 4H EMA200 isn't being computed. Signals are silently skipped.
Severity: **medium** — bot isn't doing anything wrong, but the new regime gate isn't working.

Action: report to Claude. The 4H parquet feed may be stale. Bot is safe; just less productive.

### 6. `KILL SWITCH FIRED: v1`

Meaning: the bot's own -15% kill switch tripped. It has flattened positions, exited, and `data/HALT` exists.

Severity: **highest**. By definition something went unexpectedly wrong.

Action: do NOT remove `data/HALT`. SSH and check:
```
ssh root@152.42.241.43
cat /root/snapback-btc/data/HALT     # should exist
tail -n 100 /root/snapback-btc/logs/bot.jsonl | grep -iE 'kill|flatten|error'
```
Then text Claude. The bot is safely stopped — no urgency to restart from Shanghai.

---

## What to NOT do from Shanghai

- **Do NOT `rm data/HALT`** — see CLAUDE.md hard rule. The bot polls every 5s; HALT exists for a reason.
- **Do NOT edit `.env`** to flip `DRY_RUN` or `BINANCE_ENV`. Both have been audited.
- **Do NOT run `git stash -u`** on the droplet — sweeps the `.env.donchian` / `.env.cnh_short` files and silently flips them LIVE on their sub-accounts. See `snapback_droplet_branch_and_stash_gotchas` memory.
- **Do NOT promote to mainnet** anything that wasn't already promoted before the trip. SOL leg is `WAIT_FOR_MORE_DATA`.
- **Do NOT panic-halt on a single down email**. Read what happened first.

---

## Sub-account snapshot at trip start

| Leg | Sub-account | Balance | Mode | Strategy |
|---|---|---|---|---|
| v1 | `.env` | ~$101.95 | LIVE | multifactor-v1 + 4H gate |
| donchian | `.env.donchian` | $50.50 | DRY | donchian-v3 |
| cnh_short | `.env.cnh_short` | $80.50 | DRY | cnh-hybrid-short-v1 |

Total real-money exposure on the box: **~$101.95** (only v1 trades live).

---

## Emergency contacts (you, your future self)

- Droplet host: `root@152.42.241.43` (DigitalOcean)
- Repo: https://github.com/Tezigudo/snapback-btc (branches: `main`, `droplet`)
- Deploy procedure: `concept-snapback-droplet-deploy-procedure` in the second-brain wiki
- This runbook: `SHANGHAI_TRIP_RUNBOOK.md` in repo root

---

_Generated 2026-06-02 by Claude during the AFK window before the Shanghai trip. Times ICT throughout._
