# Strategy notes — can we adopt anything from the 7 LLM trading-agent repos?

> Direct, blunt answer to the user's question. Updated 2026-05-17 after P3 +
> v2 redesign + long-history walk-forward.

## The honest answer

**No drop-in adoption.** All 7 repos require paid LLM APIs at runtime to do
the thing they're famous for. We have one hard constraint that rules that
out: *no LLM in the trading loop, zero API cost at init*. What we *can* and
*do* borrow is **patterns** — not code, not runtime, not infrastructure.

| Repo | What it does | What it costs to run | What we borrowed | Why we didn't borrow more |
|---|---|---|---|---|
| **TradingAgents** | Multi-LLM agents (fundamental / sentiment / technical / risk / trader) on LangGraph; Ollama-optional | OpenAI / Claude / Gemini tokens per decision (~$0.10–$1 per "trade thought"). Ollama path = free but slower / weaker | **Role taxonomy** mirrored in `research/agents/AGENT_ROLES.md`. The seam (Researcher protocol) is shaped after their analyst/researcher/trader/risk decomposition. | Their entire value-add is the LLM reasoning. Strip that out and you're left with a LangGraph runtime we don't need. |
| **AgentQuant** | Autonomous quant research: LLM proposes params, deterministic backtester evaluates, regime detected via VIX/momentum | OpenAI tokens for the proposer step. Lower than TradingAgents because LLM is called once per research cycle, not per trade. | **(1)** Deflated-Sharpe scoring with N-trial penalty in `research/scoring.py`. **(2)** Regime-detection-then-adapt pattern → implemented deterministically in `strategy/regime.py` (funding/volatility/trend regimes from cached features, no LLM). **(3)** v2 strategy uses regime SIGN as a gate (`SnapbackBTCv2`). | The LLM proposer is the differentiator; without it, this collapses to grid search + walk-forward — which is exactly what we already built. |
| **FinRobot** | Equity-research HTML/PDF reports (10-Ks, fundamentals) via LLM director + sub-agents | Heavy GPT-4 dependence | Nothing applicable. | Equities research is a completely different problem than BTC perp execution. Reports are also not the bottleneck; live execution is. |
| **FinMem** | Layered memory (sensory / short / long-term) for self-evolving LLM trading agent | OpenAI / HuggingFace TGI per decision | Nothing yet. | We already use CogniLayer (per `CLAUDE.md`) for cross-session memory — same role, already integrated. Their layered-memory innovation is for the LLM, not for us. |
| **AgenticTrading / FinAgent** | A2A protocol + MCP + Neo4j graph memory + vector retrieval for "similar past setups" | Heavy infra (Neo4j, vector DB) + LLM tokens | Nothing. | Massive infra for one symbol / one strategy is wildly over-engineered. If we ever go multi-symbol with a strategy zoo, revisit. |
| **StockAgent** | Simulated market environment for behavioural studies on LLM agents | LLM tokens for each simulated agent | Nothing. | Research tool, not a library. The point of the paper is studying LLM behaviour, not producing alpha. |
| **TradingGoose** | Hosted product / workflow studio for non-coders | Subscription | Nothing. | Closed-source-ish, not adoptable. |
| **QuantAgent + forks** | Various OS quant-agent forks (top has 1★, no canonical maintainer) | LLM tokens | Nothing. | Fragmentation = no stable reference implementation worth borrowing. |

## What this looks like in our code

Each adoption is concrete, not aspirational:

- **`research/agents/AGENT_ROLES.md`** — role taxonomy from TradingAgents,
  documents the *shape* of future researcher implementations (Claude Code
  cockpit, Ollama-local, opt-in Anthropic API). None of those ship yet.
- **`research/agents/deterministic.py`** — `DeterministicResearcher`: pure
  stats, no LLM, no API cost. The default. Plug-replaceable.
- **`research/scoring.py`** — `deflated_sharpe`, `bootstrap_sharpe_p5`,
  `fold_stability_score`. Pattern source: AgentQuant + López de Prado.
- **`strategy/regime.py`** — deterministic regime features (funding sign +
  magnitude percentile, ATR percentile, EMA slope). Pattern source:
  AgentQuant's regime-aware param-proposer step, executed without LLM.
- **`strategy/signals_v2.py`** — `SnapbackBTCv2` uses funding *regime sign*
  to gate entries: long only when shorts are paying funding, short only
  when longs are paying. That's the "trade with the carry" thesis dressed
  in regime-gate clothes.

## What we explicitly DID NOT adopt — and why each is dangerous for us

- **An LLM in the trading loop.** Latency, cost, non-determinism. A market
  order during a flash crash cannot wait 4 seconds for an LLM to "reason".
  Even with caching, even with local Ollama, the failure mode is
  "non-deterministic order placement under stress" which makes incidents
  unreviewable.
- **Multi-agent debate frameworks (TradingAgents, AgenticTrading).** The
  number of LLM calls per trade scales with the number of agents. For a
  bot that backtests over 175k bars, this is 175k × N agents × token cost.
  Even at the cheapest rates this is hundreds of dollars per backtest.
- **Vector memory / graph memory of past trades.** Useful for human-style
  pattern recognition. Useless for a deterministic rule-based bot — the
  rules don't "remember", they just trigger on current bar features.
  Adds infra (Neo4j, embeddings) that breaks our zero-cost constraint.
- **LLM-proposed parameters.** AgentQuant's pitch is that an LLM
  reasoning about regime context proposes better starting params than a
  blind grid search. We replaced this with a hardcoded reasonable grid +
  walk-forward + a Researcher seam that can later use Claude Code (free
  under your existing subscription) to *suggest* the next sweep range
  based on this fold's winners. No paid API needed.

## Where an LLM could earn its cost later, opt-in

Three concrete future implementations listed in
`research/agents/AGENT_ROLES.md`:

1. **`ClaudeCodeResearcher`** — emits a markdown block addressed to you,
   to be pasted into Claude Code. You're already paying for Claude Code,
   so this is $0 incremental. Use case: after a walk-forward, "open
   `reports/walk_forward_*.md` and call `/diagnose-folds`" — Claude Code
   reads the markdown and answers in chat.
2. **`OllamaResearcher`** — calls a local Ollama model via HTTP. $0 API
   cost (RAM + electricity instead), no network. Reasoning quality is
   weaker but the use case is just narrating fold tables, not reasoning
   about live trades.
3. **`AnthropicAPIResearcher`** — direct Anthropic API. Costs real money.
   Opt-in only, behind `RESEARCHER_API_ALLOWED=1` env flag. Listed for
   completeness but not recommended until P5+.

## Bottom line for "can I adopt a strategy from any of them?"

**No.** None of the 7 repos contains an *open-sourced, deterministic,
audited strategy with edge*. They contain frameworks for LLMs to reason
about markets. The strategy IS the LLM's reasoning. Without the LLM,
there's no strategy — there's a tool that built it.

What you actually want — a deterministic Python strategy with a real edge
on BTC perp — does not exist in any of those repos. It also does not
currently exist in this one: v1 fails walk-forward, v2 redesign is
in-progress. The honest path forward is to keep iterating on the
strategy family (see "Next experiments" below), use the seam to optionally
let an LLM critique fold results, and accept that the LLM is a *research
assistant*, never a *trading agent*.

## Strategy bake-off (P3.2 + P3.3 — 7-way walk-forward, BTC perp, 2022-2024)

All ran on the **same** 30-month window (2022-06 → 2024-12), 60d
train / 20d test / 30d step, same `Backtest` harness with fees + slippage
+ funding accounting, same deflated-Sharpe scoring. Different param grids
chosen per strategy's natural free knobs. Honest, brutal numbers:

| Strategy | Hypothesis | Stability | Median OOS Sharpe | Median OOS return | Drift | Promotion |
|---|---|---:|---:|---:|---:|:---:|
| snapback-v1 | RSI(2) mean-revert in EMA-trend | 21% | −3.58 | −5.13% | +122% | ❌ (3/3 fail) |
| snapback-v2 | + regime gating | 19% | −0.22 | −0.06% | +566% | ❌ (3/3 fail) |
| donchian-v1 | 1h breakout trend-follow | 53% | +0.40 | +1.35% | +58% | ❌ (Sharpe + drift) |
| donchian-v2 | + ATR trailing + wider grid | 59% | +0.47 | +2.37% | +64% | ❌ (Sharpe + drift) |
| carry-v1 | funding-rate harvester | 50% | −0.13 | +0.55% | +2% | ❌ (Sharpe only) |
| **carry-v2** | + 24h fast-move skip filter + tighter SL | **58%** | **+0.96** | **0.00%*** | +54% | ❌ (drift only) |
| ensemble(d2+c2) | 50/50 capital split | 48% | −0.30 | −0.32% | n/a | ❌ (Sharpe + stability) |

\* carry-v2 median return is exactly 0.00% because 5/29 folds had zero
trades (funding never crossed the chosen threshold on those test
windows). Those zeros sit at the centre of the sort, pinning the
median. Trade-conditional median is materially positive; the +0.96
Sharpe captures the actual win-vs-loss asymmetry.

### What the table says (after P3.3 refinement cycle)

- **snapback (both versions) has no edge** on BTC 15m. v2's regime
  gating turns it from catastrophic to merely losing — useful diagnostic,
  not a real strategy.
- **Donchian v2 didn't move the needle as much as hoped.** The ATR
  trailing stop bumped stability 53% → 59% and median return +1.35% →
  +2.37%, but median Sharpe only crept 0.40 → 0.47 — still under the 0.5
  bar. Drift got *worse* (58% → 64%), because the trailing stop sometimes
  exits before the move completes, making train-window picks slightly
  more optimistic than test results. Conclusion: trailing helps the
  *median* but doesn't fix the *tail*.
- **Carry v2 is the headline result.** Adding the `max_24h_change_pct`
  fast-move skip filter plus a wider `sl_pct` grid (now 0.005 / 0.010 /
  0.015) lifted median Sharpe from −0.13 → **+0.96** and stability 50%
  → 58%. It now PASSES median-Sharpe AND stability gates; only drift
  (+54%) keeps it under the promotion bar. Picking apart the winners:
  - `max_24h_change_pct`: 14 folds chose 100 (filter off), 15 chose
    some filter (8 at 3.0, 7 at 5.0). Slight plurality for "filter
    off"; the filter wins when it wins, but it isn't a uniform
    improvement.
  - `sl_pct`: dominant winner was **0.015** (23/29 folds) — the
    *loosest* stop in the v2 grid, looser than v1's 0.010 default.
    0.005 (the tightest) only won 3/29. The v2 win didn't come from
    tighter risk; it came from a wider grid that included a looser
    stop matched to BTC's typical 24h range.
  - 5/29 folds had zero trades (folds 6, 18, 21, 22, 25 — funding
    never crossed threshold). Those zeros are why the median return
    pins to exactly 0.00% despite a positive Sharpe; the
    trade-conditional median is materially positive (e.g. fold 24:
    +20.6%, fold 0: +11.1%).
- **The ensemble HYPOTHESIS FAILED.** Combining Donchian v2 and Carry v2
  at 50/50 capital did NOT produce a smoother equity curve. Median
  combined Sharpe (−0.30) is worse than either standalone, and
  stability (48%) is worse than either too. Two failure modes drove it:
  (a) several test windows had carry-v2 doing 0 trades, leaving the
  ensemble as a half-sized Donchian; (b) when both strategies trade,
  they sometimes lose together (folds 15, 27, 28 had both members
  negative), so the diversification benefit was zero. The "uncorrelated
  strategies" hypothesis is empirically wrong for this pair on this
  asset/timeframe.

### What this changes about the project

After P3.3, **carry-v2 is the lead candidate** for paper trading on
testnet. It's the only strategy that passes 2 of 3 promotion checks. The
remaining drift gap is small enough that a stricter `min_trades` filter
or longer test windows (30d instead of 20d) might close it — both worth
trying before declaring victory.

Donchian v2 stays in the zoo as the second-best, but its inability to
pass median-Sharpe at this grid size means more tuning has diminishing
returns; the next move is structural (e.g. add a regime filter that
gates entries by ATR percentile) not parametric.

The ensemble path is **deprioritised**. Equal-weight 50/50 is a strong
form of "the priors are right"; the data says they aren't right here.
Future ensemble work should be variance-weighted (carry gets more
capital because its Sharpe is better) or regime-switched (use carry in
low-vol, Donchian in high-vol). Neither will be revisited until P5 at
the earliest.

The walk-forward engine + Researcher seam built in P3 paid for itself
*on the first weekend* by separating "noisy losing strategy"
(snapback) from "real-but-rough strategy" (Donchian, carry). That's
exactly what it was supposed to do.

## Next experiments (in honest priority order, post-P3.3)

1. **Close carry-v2's drift gap.** Two micro-experiments, each ~1 hour
   of walk-forward: (a) bump `min_trades_train` from 3 to 6 to penalise
   train picks with too few samples; (b) try 30d/45d test windows
   instead of 20d. Either should compress drift without lying about the
   underlying edge.
2. **Try regime-gated carry.** Use `strategy/regime.py`'s ATR
   percentile as an entry gate — only take carry trades when ATR is
   below median. The failure folds were almost all high-vol windows.
3. **THEN proceed to P4 testnet** with carry-v2 (best gridpoint from the
   walk-forward report) for 7-day soak. By then carry-v2 either passes
   promotion or we know it doesn't and we stop pretending.
4. **Defer indefinitely:** ensemble work. The data says equal-weight
   doesn't help; variance-weighted or regime-switched ensembles are
   research projects of their own and don't gain us anything if our
   single-strategy edge is already solid.

Snapback as a research subject is **done** — declared a negative result.
We keep the code and the v1/v2 baselines as comparison points; we do not
deploy them. Donchian v2 stays in the zoo but is no longer a P4
candidate. Carry-v2 is the **lead candidate** for testnet.
