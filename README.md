# snapback-btc

Deterministic Binance Futures BTC/USDT perpetual bot. RSI(2) extreme + EMA(200) trend filter + relative-volume confirmation + funding-rate edge. Testnet-first, with Claude Code as the human-loop monitor cockpit.

> ⚠️ **Research project.** Not financial advice. May lose money. Testnet only until §6 of the [ultraplan](../btc-bot-ultraplan.md) checklist passes.

## Stack
- Python 3.11+
- `ccxt` (Binance Futures REST + WS)
- `pandas`, `pandas-ta` (indicators)
- `backtesting.py` (research), `freqtrade` (live, later)
- `plotly` (HTML reports)
- SQLite (state), SMTP email (alerts — Gmail app password works)

## Quickstart (testnet)
```bash
git clone https://github.com/Tezigudo/snapback-btc
cd snapback-btc
cp .env.example .env             # fill testnet keys
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "from exchange.env import get_env; print(get_env())"   # -> 'testnet'
```

## Layout
| Path | Purpose |
|---|---|
| `bot.py` | Daemon entrypoint |
| `strategy/` | Deterministic signal + execution logic |
| `risk.py` | **Hard ceilings — do not edit** |
| `exchange/` | Binance client + env/lockfile gate |
| `monitor.py` | Cron health checker (email alerts via `alerts.py`) |
| `backtest.py` | `backtesting.py` harness |
| `research/` | Walk-forward + OOS + pluggable researcher seam (zero API cost by default) |
| `report.py` | Plotly HTML generator |
| `.claude/commands/` | Slash commands for the Claude Code cockpit |
| `deploy/` | systemd unit for VPS |

## Safety model
1. Defaults to **testnet** via `BINANCE_ENV`
2. Mainnet requires `confirm_mainnet.lock` file to exist
3. `risk.py` constants are absolute ceilings (no YAML override)
4. `data/HALT` file = bot closes all and exits clean
5. Git pre-commit hook rejects `risk.py` edits without `RISK_REVIEW=1`

## Status
- [x] P0 — scaffold
- [x] P1 — data + backtester (honest fees/slippage/funding)
- [x] P2 — strategy v1 (RSI(2) + EMA(200) + volume + funding)
- [x] P3 — walk-forward + OOS (deterministic researcher; LLM seam ready)
- [x] P3.2 — strategy bake-off: snapback v1/v2 retired, Donchian + carry promising
- [x] P3.3 — v2 refinements + ensemble: carry-v2 leads (2/3 promotion checks pass)
- [x] P3.4 — leverage + TF sweeps: **carry-v2 PROMOTES at 15m, 3x** (leverage proved cosmetic)
- [ ] P4 — live testnet (7-day soak with carry-v2 at 3x)
- [ ] P5 — monitor + viz
- [ ] P6 — mainnet gate

## Strategy zoo (P3.4 — see `STRATEGY_NOTES.md` for full table)

All ran on the same 2022-06 → 2024-12 walk-forward with fees + slippage + funding.

| Strategy | TF | Stability | Sharpe | Drift | Promotion |
|---|---|---:|---:|---:|:---:|
| snapback-v1 / v2 | 15m | 19-21% | −3.58 / −0.22 | huge | ❌ retired |
| donchian-v1 | 15m | 53% | +0.40 | +58% | ❌ |
| donchian-v2 | 15m | 59% | +0.47 | +64% | ❌ |
| donchian-v2 | 4h | 55% | +0.67 | +70% | ❌ drift only |
| carry-v1 | 15m | 50% | −0.13 | +2% | ❌ Sharpe |
| carry-v2 (P3.3) | 15m | 58% | +0.96 | +54% | ❌ drift only |
| **carry-v2 (P3.4 phaseC)** | **15m** | **60%** | **+0.70** | **+40%** | **✅ PASS at 3x** |
| ensemble(d2 + c2) | 15m | 48% | −0.30 | n/a | ❌ falsified |

P3.4 leverage ablation: phase C pass at 20x and at 3x produced *identical*
numbers (Sharpe +0.70, stability 60%, drift +40%). The pass came from
`test_days=30 + min_trades=6`, NOT from leverage. Carry-v2 graduates at
the safe 3x. Live bot needs no `RISK_REVIEW` override.

## Research

P3 walk-forward sweeps a param grid (`config/sweep.yaml`) across rolling
train/test windows and writes JSON/MD/HTML reports under `reports/`.

```bash
python -m research.walk_forward \
    --sweep config/sweep.yaml \
    --start 2025-01-01 --end 2025-12-31
```

The researcher that comments on results is pluggable via
`research/agents/`. Default is `DeterministicResearcher` — pure stats,
**zero API cost, no network**. See `research/agents/AGENT_ROLES.md` for
the role taxonomy borrowed from TradingAgents / AgentQuant and the list
of opt-in LLM-backed researchers that can be added later.
