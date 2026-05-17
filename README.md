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
- [ ] P4 — live testnet (7-day soak)
- [ ] P5 — monitor + viz
- [ ] P6 — mainnet gate

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
