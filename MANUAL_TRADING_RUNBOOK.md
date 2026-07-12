# Manual trading vs the bots — separation runbook

Date: 2026-07-12 · Owner: God · Status: **DONE 2026-07-12** (inverted variant)

> **Executed with the roles flipped:** instead of moving manual play off the
> main account, the **v1 bot moved to its own new sub-account** and the main
> account became the manual-trading account. Cutover 06:09 UTC: new key in
> `.env.v1` (+ base `.env` rotated to the same sub-account key so the hourly
> consolidate push follows the bot), old `state.db` archived at
> `archive/state.db.v1-mainaccount.pre-migration-20260712`, fresh principal
> P=$142.93 / kill floor $92.19. The transfer-while-running tripped a
> false-positive kill switch (equity 0 vs stale principal) — expected,
> HALT_v1 cleared during migration. Remaining: God deletes the old
> main-account API key on Binance. The "standing rules" below still apply,
> with v1/donchian sub-accounts as the protected zone.

## Why this exists

The 2026-07-12 PnL audit found manual trades (HOMEUSDT, WLDUSDT, TACUSDT,
VELVETUSDT, BNB) executed on the **same Binance sub-account as the v1 bot**.
Since 2026-06-01 the bot's own contribution was **+$5.46 (one BTC trade)**;
everything else in the account's ±$41 swing was manual. Manual trading on a
bot account causes three concrete problems:

1. **It freezes the bot.** On 2026-06-30 the manual TAC/VELVET losses pushed
   intraday drawdown past the daily-loss breaker (19 trigger events in
   `data/state.db`), blocking bot entries for the rest of the day. The breaker
   cannot tell a manual loss from a bot loss — equity is equity.
2. **It distorts the safety anchors.** The principal ledger, daily anchor, and
   kill-switch floor all key off account equity. Manual PnL shifts them in
   ways the backtests never modeled.
3. **It leaks funding fees.** −$7.35 of funding since 06-01, mostly on manually
   held alt positions, silently drags the account the bot is measured on.

## One-time fix (do on Binance app/web, master account)

1. Create a new sub-account (suggested name: `manual-play`).
   Binance app → Profile → Sub Accounts → Create.
2. Transfer your manual-trading capital from the **v1 bot sub-account**
   futures wallet → master → `manual-play` futures wallet.
   Leave the bot's principal (currently ~$101.71) plus its accumulated bot
   profit in place. The bots' principal ledgers are transfer-immune
   (TRANSFER-only income filter), so this withdrawal will NOT trip the
   breaker or the kill switch — but do it while the bot is flat if possible
   (check the dashboard or `data/state.db` positions first).
3. Trade HOME/WLD/TAC/whatever only from `manual-play` from now on.

## Standing rules after the fix

- **Never place manual orders on the v1 or donchian sub-accounts.** Not even
  "just one scalp" — one losing scalp ≥ breaker threshold freezes the leg
  for the rest of the UTC day.
- Depositing/withdrawing bot capital is fine (principal ledger absorbs
  TRANSFER rows); manual *positions* are not.
- BNB held on bot accounts for the fee discount is fine — it's small and
  intentional. Don't trade it.

## How to verify it worked

After moving capital, run (read-only) from the repo on the droplet:

```
.venv/bin/python /tmp/pull_income.py   # or tools/consolidate_futures_push.py path
```

The v1 account's income history should show only BTCUSDT REALIZED_PNL rows
(client order ids prefixed `snap-v1-`) going forward.
