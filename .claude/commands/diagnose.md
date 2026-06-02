---
description: Triage an anomaly — heartbeat, errors, position drift, recent signals
---

Read-only triage. Propose fixes, do NOT apply them.

**Time display rule:** bot stores UTC, operator sees **GMT+7 ICT**. Convert every timestamp on display. Internal greps/file paths stay UTC. Heartbeat is on Linux → use `stat -c %Y` (NOT macOS `stat -f %m`).

1. **Heartbeat:** `stat -c %Y data/heartbeat` → if >90s old, bot is DOWN. Note last mtime as GMT+7 ICT.
2. **Errors:** tail last 200 lines of `logs/bot.jsonl` (rolling file — no date suffix). Surface every `level=ERROR` and any Python traceback with file:line. Show event times as GMT+7 ICT.
3. **Position drift:** compare `state.db` open positions vs Binance REST `/fapi/v2/positionRisk`. Flag any mismatch in symbol/side/size.
4. **Drawdown:** equity_now vs equity_day_start vs `risk.MAX_DAILY_LOSS_PCT`.
5. **Recent signals:** last 20 entries in `state.db` signals table — did the bot see the right setups but skip them? Why?
6. **CogniLayer memory:** `memory_search("snapback <error keyword>")` for past similar incidents.
7. Output:
   - Root cause hypothesis (most likely)
   - Proposed fix (config tweak, restart, code patch — be specific)
   - Whether `/halt` is warranted (yes only if losses approaching ceiling, position drift, or repeated crashes)
   - Footer: `_Times shown in GMT+7 (ICT). Bot stores UTC._`
8. If novel root cause, end by suggesting `memory_write(type="error_fix", tags="snapback-btc,<topic>")` with the keywords.

**Do not edit `risk.py`. Do not place orders. Do not remove `data/HALT`.**
