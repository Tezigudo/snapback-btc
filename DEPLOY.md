# DEPLOY.md — 1-month real-money plan (small capital)

Updated 2026-05-17. Strategy: **multifactor-v1**. Capital: **$60 USDT** (real). Duration: **30 days**.

Backtest evidence: `PATH2_RESULTS.html` + `TRADING_HISTORY.html` (6 OOS windows, +55% compounded, 4 of 6 positive).

## Why real money (not testnet)

Binance Futures testnet/sandbox for **signed/private** endpoints was deprecated in ccxt late 2025 — only public market data still works on the sandbox URLs. We can no longer paper-trade against a free practice account through this codebase.

The honest paths:
- **Dry-run mode**: bot fetches REAL market data + REAL balance from your live account but never calls `create_order`. Use this for as long as you need to feel comfortable.
- **Live with small capital ($60)**: real fills, real fees, real PnL — but the dollar amounts are small enough that a worst-case loss is bounded.

You started with $60. The kill switch caps absolute loss at ~$11 (−18%) before forced flatten. Most likely 30-day outcome is between −$10 and +$15.

---

## What changed since the testnet plan

| Item | Old (testnet) | New (real $60) |
|---|---|---|
| Environment | testnet | mainnet (real money) |
| Capital | $1M paper | $60 real |
| Duration | 90 days | 30 days |
| Kill switch | -15% | **-18%** (widened — see PATH2_RESULTS for why) |
| Pre-flight | none | `tools/preflight_live.py` (mandatory) |
| Dry-run mode | n/a | **mandatory first** before live orders |
| Min-capital warning | none | warn below $100 |
| Exchange minimums | ignored | enforced (`exchange/constraints.py`) |
| Tooling | pip / `.venv/bin/python` | **uv** (`uv run python -m bot`) |
| Linting | informal | `uv run ruff check` (clean) |

---

## $60 capital sizing reality check

Binance Futures BTC/USDT:USDT exchange minimums:
- Minimum order quantity: **0.001 BTC**
- Minimum notional cost: **$50 USDT**
- Taker fee: 0.05% per side

At $60 equity with our params (2% risk, 1.5% SL):
- target_btc = ($60 × 0.02) / (price × 0.015) → at $100k BTC: **0.0008 BTC**
- That's below the 0.001 BTC minimum → **the bot will SKIP these signals.**

What the bot does: when sizing comes out below exchange minimum, it logs `signal_skipped_minimum` and waits for the next signal. It does NOT scale up to meet the minimum (that would violate the risk budget).

**At $60 you will skip many signals.** To make this strategy size correctly, fund **at least $100** (ideally $200+). Estimated signal frequency:
- $60: ~0-3 trades per month
- $100: ~3-7 trades per month
- $200+: matches backtest density (~8-12 trades/month)

You can run dry-run at $60 to confirm the wiring without losing money to fees on rare fills.

---

## Pre-flight checklist (do these once)

### 1. Update `.env` with your **real mainnet** Binance Futures API key

```bash
# Edit .env:
BINANCE_ENV=mainnet
BINANCE_API_KEY=<your real key>
BINANCE_API_SECRET=<your real secret>
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=godjangg@gmail.com
SMTP_PASSWORD=<app password>
ALERT_EMAIL_FROM=godjangg@gmail.com
ALERT_EMAIL_TO=godjangg@gmail.com
```

**API key permissions on Binance:**
- ENABLE: Read, Enable Futures
- DISABLE: Withdrawals, Spot, Margin
- Restrict to your IP (recommended)

### 2. Create the mainnet lockfile (one-line ack that you mean it)

```bash
echo "Snapback-btc mainnet $60 deploy $(date -u +%Y-%m-%dT%H:%M:%SZ)" > confirm_mainnet.lock
```

Without this file, `exchange/env.py` will refuse to boot in mainnet mode.

### 3. Run pre-flight

```bash
uv run python -m tools.preflight_live --send-test-email
```

Expected output: every check shows ✓ green. If anything fails, fix before continuing.

### 4. Start in DRY-RUN first (NO real orders)

```bash
uv run python -m bot --dry-run
```

Let it run for **at least one full 15m bar cycle** (15–60 min). Watch:
- `logs/bot.jsonl` — every event
- `logs/console.log` — human-readable
- Email — you should get a "Bot deploy start [DRY-RUN]" message

If a signal fires during dry-run, you'll get a "DRY-RUN: would LONG/SHORT" email with the sizing details. **No order is placed.**

### 5. Verify dry-run state, THEN go live

After watching dry-run for as long as you need:
```bash
# Stop dry-run (Ctrl+C)
# Optionally reset deploy-start equity (recommended — you want the LIVE run to anchor at the equity you go live with):
rm -f data/state.db
# Start live:
uv run python -m bot
```

You'll get a "Bot deploy start [LIVE]" email.

---

## Run it under tmux so it survives shell exit

```bash
tmux new -s snapback 'uv run python -m bot 2>&1 | tee -a logs/console.log'
# Attach later: tmux attach -t snapback
# Detach without killing: Ctrl+B then D
```

---

## What to watch

| Source | What you see |
|---|---|
| Email inbox | deploy start, every entry, time-stop closes, HALTs, kill-switch |
| `logs/bot.jsonl` | structured JSONL per event |
| `logs/console.log` | human-readable mirror |
| `data/state.db` | SQLite. Tables: `meta`, `fills`, `events` |
| `data/heartbeat` | mtime updated each ~5s. Stale >90s = bot died. |

**Quick health check** (run anytime):
```bash
uv run python -c "
import sqlite3, time
from pathlib import Path
db = sqlite3.connect('data/state.db')
print('--- meta ---')
for k, v in db.execute('SELECT key, value FROM meta'):
    print(f'  {k}: {v}')
print('--- last 10 fills ---')
for r in db.execute('SELECT ts, side, qty, price, reason, equity_after FROM fills ORDER BY id DESC LIMIT 10'):
    print(' ', r)
print('--- last 10 events ---')
for r in db.execute('SELECT ts, level, kind, msg FROM events ORDER BY id DESC LIMIT 10'):
    print(' ', r)
hb = Path('data/heartbeat')
age = (time.time() - hb.stat().st_mtime) if hb.exists() else float('inf')
print(f'heartbeat age: {age:.1f}s')
"
```

---

## How to stop

| Goal | Command |
|---|---|
| Clean stop (leaves position open) | `Ctrl+C` in bot terminal |
| Stop AND flatten | `touch data/HALT` — bot closes position, emails, exits in ~5s |
| Emergency manual flatten | `uv run python -c "from exchange.binance_client import BinanceClient; BinanceClient.from_env().close_position('BTC/USDT:USDT')"` |

---

## Safety stack (all wired in)

| Gate | Where | Behavior |
|---|---|---|
| Hard symbol allowlist | `risk.py` | crash on attempt to trade anything but BTC |
| Leverage ceiling 20x | `risk.py` | reject setLeverage above 20 |
| Notional cap $500/order | `risk.py` | reject oversized orders |
| Mainnet requires lockfile | `exchange/env.py` | crash on boot without `confirm_mainnet.lock` |
| **Exchange minimums** | `exchange/constraints.py` | skip signal if qty < 0.001 BTC or notional < $50 |
| **-18% equity kill switch** | `bot.py Bot._check_kill_switch` | flattens, touches HALT, emails, exits |
| HALT file polling (5s) | `bot.py Bot.loop` | clean exit on `data/HALT` |
| Single position exclusive | `bot.py Bot._maybe_enter` | won't open new position if one exists |
| One signal per 15m bar | `bot.py Bot._maybe_enter` | dedupe on `_last_signal_ts` |
| Bracket SL/TP on every entry | `bot.py Bot._maybe_enter` | reduce-only STOP_MARKET + TAKE_PROFIT_MARKET |
| **Dry-run mode** | `bot.py` `--dry-run` | observes & logs everything but places NO orders |

---

## 30-day live test plan

### Week 1: dry-run only
Run the bot in `--dry-run` mode. Confirm:
- Heartbeat freshness > 99%
- No unhandled exceptions in `logs/bot.jsonl`
- If a signal fires, the "DRY-RUN: would LONG/SHORT" email arrives within seconds
- The dry-run sizing math at YOUR equity (try with $60) — does it skip signals because of exchange minimums? If yes, decide whether to top up.

### Week 2-4: live (real orders)
Switch off `--dry-run`. Watch:
- Trade-by-trade outcomes match expected SL/TP geometry
- Drawdown stays inside -18%
- Email alerts arrive for every fill

### 30-day promotion criteria

To increase capital to $500 or more, all of these must be true:

1. **No kill-switch fire** (-18% never breached).
2. **No silent crash** — bot's heartbeat freshness > 99%.
3. **Realized return within ±10% of backtest expectation.** Backtest mean for ~1 month windows is roughly +0% to +5%. Acceptable real outcome: roughly -10% to +15%. The SHAPE of the equity curve matters more than the number — does it look like a smaller version of the backtest curves in `TRADING_HISTORY.html`?
4. **Email alerts arrived for every fill** (so you trust the monitoring).

If 1–4 pass → top up to $500–$1000 and continue with same params.
If 1–4 fail → STOP. Read `logs/bot.jsonl`, reconcile against expectations, decide.

### Honest expectation

The strategy is **regime-dependent**:
- Trending months: +3% to +8% likely
- Choppy months: -3% to +3% likely
- Worst case in backtest: -12% (single month inside a 6-month chop window)

You may finish 30 days at break-even or small loss. That is NOT failure — it's the wrong half of the regime coin. The honest deploy criterion is "did the bot behave as designed", not "did it make money in 30 days".

---

## Future work (NOT in this deploy)

- **Regime detector**: investigated and rejected as a fixed-threshold filter (ADX, EMA-slope, ATR — none cleanly separate chop from trend on BTC). The shape of "chop" is too varied. If you want regime gating in v2, the path is probably ML-based (random forest on ADX + slope + ATR + funding + 30d return). Several days of work.
- **Multi-asset extension**: ETH, SOL would need their own backtests. v1 is BTC-only.
- **News/sentiment filter**: explicitly out of scope (no LLM in trading loop per repo rules).

---

## File map (post-cleanup)

```
bot.py                              # live trading loop (--dry-run flag)
alerts.py                           # SMTP email
risk.py                             # hard ceilings (DO NOT casually edit)
backtest.py                         # research only
config/params.yaml                  # locked strategy config (-18% kill switch)
exchange/
  binance_client.py                 # ccxt wrapper, fetch_equity/position, bracket orders
  constraints.py                    # exchange minimums (live + fallback)
  state.py                          # SQLite state store
  env.py                            # env + mainnet lockfile gates
  data.py                           # historical klines + funding (research)
strategy/
  signals.py                        # StrategyParams + snapback-v1 + data prep
  signals_multifactor.py            # multifactor-v1 (deployable)
  indicators.py                     # rsi/ema/atr/macd/sar/swing/trendline
tools/
  preflight_live.py                 # MANDATORY before live (verify everything)
  diagnose_trades.py                # EV math + CI on win rate
  chart_trade.py                    # per-trade chart (price+RSI+volume)
  extract_trades.py                 # run backtest, dump trade JSON
  chart_mtf.py                      # multi-TF chart
  analyze_mtf.py                    # confluence scoring (research only)
  build_path2_report.py             # PATH2_RESULTS.html generator
  build_trading_history.py          # TRADING_HISTORY.html generator
data/
  historical/                       # cached parquet OHLCV + funding
  state.db                          # bot's SQLite state (created on first run)
  heartbeat                         # mtime touched each loop tick
  HALT                              # touch this to flatten + exit
logs/
  bot.jsonl                         # structured event log
  console.log                       # stdout/stderr mirror (if you tee)
PATH2_RESULTS.html                  # deploy-decision report
TRADING_HISTORY.html                # full 288-trade ledger
DEPLOY.md                           # this file
LIVE_PLAN.md                        # summary of changes for review
CLAUDE.md                           # operator guide for Claude-Code monitor cockpit
README.md                           # project overview
.env / .env.example                 # API keys (real .env never committed)
confirm_mainnet.lock                # must exist for mainnet (you create it)
pyproject.toml                      # uv-managed
uv.lock                             # uv lockfile
```
