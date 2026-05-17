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
- [x] P3.4 — leverage + TF sweeps (leverage set to 20x permanent)
- [x] P3.5 — **promotion gate was broken**; rebuilt with tail-risk checks; carry-v4 + trend gate + tail-aware selection lifted CAGR -18.6% → +4.95% (still at boundary, NOT P4-ready)
- [ ] P4 — blocked on OOS validation of carry-v4 on 2025 data
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
`test_days=30 + min_trades=6`, NOT from leverage.

**P3.5 correction:** the "PASS" was a broken-gate artefact. Replaying
the 28 fold returns serially produces $100 → $62.30 over 2.3 years —
**CAGR −18.6%/year**. The 3-check median-based gate didn't see three
tail folds (−47%, −25%, −21%) wipe out 25 small winners. Rebuilt the
gate with strict mean/CAGR/max-loss checks (`min_compounded_cagr_pct`,
`min_mean_test_return_pct`, `max_single_fold_loss_pct`). Architectural
fixes — trend EMA gate (carry-v4), tail-aware combo selection
(`research/scoring.py: tail_aware_score`) — lifted CAGR to **+4.95%**
on the same fold set, but with **negative median Sharpe (−0.05)** and
**48% fold-positive rate**, this is at the floor of noise, not a real
edge. Live deployment is blocked on **out-of-sample validation against
2025 data** before P4 testnet.

**Per-user-decision:** leverage permanently set to 20x in
`config/params.yaml` and `risk.py: MAX_LEVERAGE`. Ablation showed
leverage doesn't change carry-v2/v4 returns (cap doesn't bind for the
winning combos) but live tail-risk at 20x is real (flash crash past SL
= liquidation).

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
