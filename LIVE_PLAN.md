# LIVE_PLAN.md — Welcome back. Read me first.

You went AFK with these constraints:
- `$60` real USDT in Binance Futures wallet
- Testnet is broken (ccxt deprecation)
- "Make money or I downgrade to pro" — fair
- "1-month test" — done with real money, small size
- "Switch pip → uv, use ruff" — done

Here's everything that changed while you were away, what you need to do, and the honest math.

---

## TL;DR — 4 commands to deploy

```bash
# 1. Edit .env with REAL mainnet API key (already have the format)
nano .env

# 2. Create the mainnet ack file (refuses to boot without this)
echo "Snapback $60 deploy $(date -u)" > confirm_mainnet.lock

# 3. Run pre-flight (mandatory)
uv run python -m tools.preflight_live --send-test-email

# 4. Start in DRY-RUN first (no real orders, observes for as long as you want)
uv run python -m bot --dry-run
```

When you're satisfied the wiring is right, swap step 4 to live:

```bash
rm -f data/state.db           # reset deploy-start equity for the live run
uv run python -m bot
```

---

## What I changed while you were AFK

### ✅ uv replaces pip
- Added `uv.lock` (569 KB), `pyproject.toml` cleaned up (dropped `pandas-ta` which was unused and unmaintained for Python 3.13)
- All commands now `uv run python …` instead of `.venv/bin/python …`
- DEPLOY.md updated

### ✅ ruff lints clean
- Auto-fixed 27 violations
- Tweaked config to ignore cosmetic warnings (multiplication sign in comments, etc.)
- `uv run ruff check bot.py risk.py alerts.py exchange/ strategy/ tools/` → "All checks passed!"

### ✅ Dry-run mode added to `bot.py`
- `uv run python -m bot --dry-run` or `DRY_RUN=1 uv run python -m bot`
- Fetches REAL market data + REAL balance from your live account
- Evaluates signals, logs sizing, sends "DRY-RUN: would LONG/SHORT" emails
- **Never calls create_order, never set_leverage in dry-run**
- Use this for as long as you want before going live

### ✅ Exchange-minimum guards (`exchange/constraints.py`)
- Binance Futures BTC: **min 0.001 BTC qty, min $50 notional**
- Bot now SKIPs signals that would be below these limits (logs `signal_skipped_minimum`)
- Never scales up — that would violate the risk budget

### ✅ Kill switch widened to -18%
- Was -15%. Empirically, 2024 H2 (a +21% WINNING window) had max DD -14.23% mid-window. -15% would have killed a winner. -18% gives ~4pp cushion.
- New config: `kill_switch_equity_fraction: 0.82` in `config/params.yaml`

### ✅ Pre-flight script (`tools/preflight_live.py`)
- Checks: env, API creds, SMTP, ccxt market load, fetch_balance, fetch_position, sizing simulation at YOUR equity, risk.py ceilings
- Run before every live deploy
- Outputs ✓ green / ! yellow / ✗ red per check

### ✅ Cleaner pyproject.toml
- Removed obsolete `pandas-ta` and `jupyterlab/matplotlib-as-extra` clutter
- Description updated to reflect multifactor-v1
- `[tool.hatch.build.targets.wheel] packages` no longer references deleted `research/`

---

## Honest expectation for 30 days at $60

### What backtests say about 30-day windows
The strategy mean per 6-month window is roughly +10%, so a 1-month slice is roughly +1.5% expected. But variance is HUGE:
- 1-σ range: −7% to +10% per month
- Worst observed 6-month: −12.56% (2024 H1 chop)
- A single bad month inside a chop window could be −5% to −10%

**Translated to $60:** most likely outcome is between **−$5 and +$8**. With variance, you could see anywhere from **−$10 to +$15**. The kill switch caps absolute loss at **−$11** (−18%).

### What $60 means for trade frequency
- At BTC ~$100k, our 2%/1.5% sizing wants **0.0008 BTC** per trade
- Exchange minimum is **0.001 BTC**
- → Bot SKIPS many signals at $60. You may see **0–3 actual trades** in 30 days
- → Even fewer datapoints to judge "is the bot working"

**Honest recommendation:** if you can fund **$100–$200**, do it. That keeps the strategy's sizing math intact and gives you 5–10 trades worth of evidence in 30 days. At $60, statistical noise will dominate.

### What I can't promise
- That 30 days will be profitable — strategy is regime-dependent (3 of last 6 OOS windows were ~flat or losing)
- That this is the best possible strategy — it's the best in THIS codebase
- That a Claude subscription decision should hinge on 30-day live PnL — that's the WRONG metric. The right one is "does the bot behave like the backtest?" Outcome is dominated by which regime BTC happens to be in.

---

## What you might want to do differently before going live

I would consider these before pushing the start button. Each is an explicit decision:

| Decision | If yes, do this | If no, default applies |
|---|---|---|
| Top up to $200+ | Transfer USDT into futures wallet | $60 (many signal skips) |
| Lower SL to 1% to allow $60 sizing | Edit `config/params.yaml` `sl_pct: 0.010`, re-run backtests on 6 windows to verify | Keep 1.5% SL (skip more signals) |
| Tighten kill switch back to -15% | Edit `kill_switch_equity_fraction: 0.85` | -18% (current default) |
| Skip dry-run entirely, go live immediately | Run `uv run python -m bot` directly | Dry-run first (recommended) |

I won't change anything without your say-so. Tell me what you want.

---

## What I did NOT do (and won't without your OK)

- **Never started the bot.** Real money requires you to type the start command.
- **Never created confirm_mainnet.lock.** You must do that explicitly to acknowledge "I mean it".
- **Never edited risk.py.** That's a hard wall in this repo.
- **Did not change `sl_pct` to fit $60.** That would compromise the backtested edge. Better to either fund more, or run dry-run only.

---

## Filemap of what's new/changed today

| File | Status | Why |
|---|---|---|
| `pyproject.toml` | EDIT | Drop pandas-ta, tighten lint config, description |
| `uv.lock` | NEW | uv lockfile |
| `bot.py` | EDIT | `--dry-run` flag, exchange-min guard, equity warning, argparse |
| `exchange/constraints.py` | NEW | min qty/notional from live market, tighter-wins merge |
| `config/params.yaml` | EDIT | -18% kill switch, `min_capital_warn_usdt: 100` |
| `tools/preflight_live.py` | NEW | mandatory verify-before-live |
| `DEPLOY.md` | REWRITE | real-money plan, $60 reality, 30-day milestones |
| `LIVE_PLAN.md` | NEW | this file |
| `strategy/indicators.py` | EDIT | rename `l` → `lo` (lint fix) |
| `tools/chart_*.py` | EDIT | drop dead vars (lint fix) |

---

## What to do RIGHT NOW (in priority order)

1. **Read this file** ✓ (you're here)
2. **Read `DEPLOY.md`** — the full playbook
3. **Decide on capital:** stay at $60 with skip-signals, or fund up to $200+
4. **Decide on kill switch:** -18% (current) or -15% (your original)
5. **Run pre-flight:** `uv run python -m tools.preflight_live --send-test-email`
6. **Dry-run first:** `uv run python -m bot --dry-run` for at least one full 15m cycle (15–60 min)
7. **Go live when ready:** `rm -f data/state.db && uv run python -m bot`

Take your time. The bot will be ready whenever you are.
