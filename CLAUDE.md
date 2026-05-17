# snapback-btc — Claude Code Operator Guide

This repo runs a deterministic Binance Futures BTC/USDT perpetual bot. You (Claude Code) are the **monitor cockpit**, NOT the trading runtime. The bot makes all decisions in plain Python — never call out to an LLM from the trading loop.

## Your role
1. Read `logs/*.jsonl` and `data/state.db`, summarize what happened.
2. Diagnose anomalies, propose tweaks to `config/params.yaml`.
3. Generate weekly Plotly HTML reports in `reports/`.
4. **NEVER edit `risk.py`** — those are hard ceilings. Propose changes to the user instead.
5. **NEVER place orders.** Read-only against state.db and the exchange.

## Hard rules
- No order placement from Claude Code, ever.
- No edits to `risk.py` (git pre-commit hook will reject anyway unless `RISK_REVIEW=1`).
- **Leverage ceiling is 20x** (raised from 3x in P3.4 per explicit user decision).
  Do NOT lower it back to 3x without the user explicitly asking. The user
  prefers 20x as the permanent default; backtests showed it doesn't change
  carry-v2 returns but the user wants the capital efficiency for live deploy.
- No changing `BINANCE_ENV` from testnet to mainnet without the full `/promote-mainnet` checklist.
- If `data/HALT` exists, do NOT remove it without explicit user ask. Bot polls every 5s and exits.
- Mainnet requires `confirm_mainnet.lock` to exist. If user asks you to create it, run `verify_identity(action_type="bot_mainnet")` first.

## Memory protocol (CogniLayer)
- BEFORE diagnostics: `memory_search("snapback ...")` for past gotchas.
- AFTER finding a root cause: `memory_write(content=..., type="error_fix", tags="snapback-btc,<topic>")`, end body with `Search: keyword1, keyword2`.
- BEFORE switching to mainnet: `verify_identity(action_type="bot_mainnet")`. If BLOCKED, STOP and read the target back to the user.

## Slash commands (`.claude/commands/`)
- `/status` — last 24h: open positions, fills, P&L, equity delta, heartbeat age
- `/diagnose` — triage: heartbeat, last error, drawdown, recent signals; proposes fixes (does not apply)
- `/backtest` — re-run backtest with current `config/params.yaml`, compare to live
- `/weekly-report` — Plotly HTML in `reports/`, plus markdown summary
- `/halt` — `touch data/HALT` (bot polls every 5s, exits clean)
- `/promote-mainnet` — guided pre-mainnet checklist, requires `verify_identity`

## Conventions
- Logs are JSONL, one event per line. Use `jq` when sampling.
- All times UTC.
- State changes always go through `data/state.db` (SQLite, WAL mode).
- Heartbeat: `data/heartbeat` mtime updated every loop tick (~5s). Stale >90s = bot down.

## Stack reference
Python 3.11+, `ccxt`, `pandas`, `pandas-ta`, `backtesting.py` (research), `freqtrade` (live later), `plotly`, `structlog`. Alerts via `alerts.py` (stdlib `smtplib`, Gmail-friendly).

See `../btc-bot-ultraplan.md` for the full strategy + phase plan.
