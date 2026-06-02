---
description: Kill switch — write data/HALT so the bot closes all and exits clean
---

Use when:
- Drawdown approaching `risk.MAX_DAILY_LOSS_PCT`
- Position drift between state.db and exchange
- Unexpected error pattern in logs
- User explicitly asks to stop

Action:
1. `touch data/HALT`
2. Tail logs for ~15s to confirm bot saw the HALT and started closing positions.
3. Wait for `bot.py` process to exit cleanly (or report if it didn't).
4. Report: "HALT engaged. Bot closed N positions, exited at HH:MM:SS ICT (GMT+7). Remove with `rm data/HALT` when ready to resume."

**Time display rule:** convert all exit/event times shown to the user from UTC to GMT+7 ICT.

**Do NOT remove `data/HALT` yourself, ever. The user must do it.**

If the bot doesn't respond to HALT within 30s, suggest manual `kill <pid>` and a position audit on Binance directly.
