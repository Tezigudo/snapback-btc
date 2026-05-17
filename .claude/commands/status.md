---
description: Snapshot the last 24h — open positions, fills, P&L, equity delta, heartbeat age
---

Read-only operator command. NEVER place an order.

1. Check `data/heartbeat` mtime. If >90s old, lead with **🔴 BOT DOWN**.
2. Query `data/state.db`:
   - Open positions (symbol, side, size, entry, unrealized PnL)
   - Closed trades in last 24h: count, winrate, net PnL, avg R
   - Equity now vs equity 24h ago (% delta)
3. `grep -c '"level":"ERROR"' logs/bot-$(date -u +%Y-%m-%d).jsonl` — flag if non-zero
4. Output as a compact markdown table. No prose, no preamble.

If `data/state.db` doesn't exist yet, say "no live data yet — bot hasn't run".
