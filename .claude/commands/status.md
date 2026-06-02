---
description: Snapshot the last 24h — open positions, fills, P&L, equity delta, heartbeat age
---

Read-only operator command. NEVER place an order.

**Time display rule:** the bot stores timestamps in UTC (do not change that — backtest joins depend on it). When showing times to the user in this command's output, convert to **GMT+7 (Asia/Bangkok)** and suffix with `ICT`. Internal log greps, DB queries, and file paths stay UTC.

1. Check `data/heartbeat` mtime. If >90s old, lead with **🔴 BOT DOWN**. Show last-tick age as "Ns ago" (relative), not absolute.
2. Query `data/state.db`:
   - Open positions (symbol, side, size, entry, unrealized PnL). Show entry time as GMT+7 ICT.
   - Closed trades in last 24h: count, winrate, net PnL, avg R. Show last fill time as GMT+7 ICT.
   - Equity now vs equity 24h ago (% delta).
3. `grep -c '"level":"ERROR"' logs/bot.jsonl logs/donchian.jsonl logs/cnh_short.jsonl` — flag any non-zero per leg.
4. Output as a compact markdown table. No prose, no preamble. Footer line: `_Times shown in GMT+7 (ICT). Bot stores UTC._`

If `data/state.db` doesn't exist yet, say "no live data yet — bot hasn't run".
