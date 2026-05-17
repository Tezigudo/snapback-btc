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

## Strategy bake-off (P3.2 — 4-way walk-forward, BTC perp, 2022-2024)

All four ran on the **same** 36-month window (2022-01 → 2024-12), 60d
train / 20d test / 30d step, same `Backtest` harness with fees + slippage
+ funding accounting, same deflated-Sharpe scoring. Different param grids
chosen per strategy's natural free knobs. Honest, brutal numbers:

| Strategy | Hypothesis | Stability | Median OOS Sharpe | Median OOS return | Drift | Promotion |
|---|---|---:|---:|---:|---:|:---:|
| snapback-v1 | RSI(2) mean-revert in EMA-trend | 21% | −3.58 | −5.13% | +122% | ❌ (3/3 fail) |
| snapback-v2 | + regime gating | 19% | −0.22 | −0.06% | +566% | ❌ (3/3 fail) |
| **donchian-v1** | 1h breakout trend-follow | **53%** | **+0.40** | **+1.35%** | +58% | ❌ (Sharpe + drift) |
| carry-v1 | funding-rate harvester | 50% | −0.13 | +0.55% | **+2%** | ❌ (Sharpe only) |

### What the table says

- **snapback (both versions) has no edge** on BTC 15m. v2's regime
  gating turns it from catastrophic to merely losing — useful diagnostic,
  not a real strategy.
- **Donchian breakout is the strongest** — 53% folds positive, median
  return ≈ +25% annualised if compounded, only Sharpe (0.40 < 0.5) and
  drift (58% > 50%) keep it from passing promotion outright. Tune
  Donchian periods + add an ATR-based trailing stop and this likely
  clears the gate.
- **Carry harvester has the best generalisation** — drift of +2% means
  train and test perform nearly identically (no overfitting), and 50%
  fold stability. But median Sharpe is essentially zero because some
  folds get smashed by fast price moves the 1% SL doesn't catch fast
  enough. Add a tighter SL or a price-direction filter and this becomes
  a real carry strategy.

### What this changes about the project

Two strategies (Donchian, carry) now have **structural signal** in walk-forward,
neither pass promotion outright but both are within a refinement cycle of doing
so. snapback-v1/v2 are dead-ends and can be retired or kept as the engine's
default cautionary baseline.

The walk-forward engine + Researcher seam built in P3 paid for itself
*on the first weekend* by separating "noisy losing strategy"
(snapback) from "real-but-rough strategy" (Donchian, carry). That's
exactly what it was supposed to do.

## Next experiments (in honest priority order, post-bake-off)

1. **Tune Donchian: trailing stop + period sweep on wider grid.** With
   the structural signal proven, the work now is parameter refinement,
   not new strategy invention. ~half-day of walk-forward.
2. **Tune carry: tighter SL, optional price-direction filter, longer
   funding lookback.** Same ~half-day of walk-forward.
3. **Try an ensemble.** Both Donchian and carry trade on different
   features; in principle they should be lowly correlated. Run them on
   the same equity curve (split capital 50/50) and see if combined
   Sharpe beats either alone. ~1 day.
4. **THEN proceed to P4 testnet** for live execution plumbing — by then
   we have a real strategy worth executing live.
5. **Defer:** multi-symbol cross-sectional momentum. Wait until P4 is
   stable and a single-symbol strategy is profitable in paper.

Snapback as a research subject is **done** — declared a negative result.
We keep the code and the v1/v2 baselines as comparison points; we do not
deploy them.
