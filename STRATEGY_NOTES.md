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

## Next experiments (in honest priority order)

1. **Different timeframe family.** Move from 15m mean-reversion to 1h or
   4h trend-following. 15m on BTC has a brutal signal-to-noise ratio,
   confirmed by both v1 walk-forwards.
2. **Different signal family.** Bollinger band squeeze + breakout instead
   of RSI extreme. Donchian channel breakouts. Volume Profile POC reversion.
3. **Asset rotation.** Multi-symbol portfolio (BTC + ETH + 2-3 alt perps)
   with cross-sectional momentum. Spreads single-strategy risk.
4. **Accept the result.** If three more redesigns also fail OOS, the
   honest conclusion is that retail-accessible BTC perp does not have
   easy edge for a deterministic mean-reverter, and we should EITHER
   shift focus to capturing funding (a carry trade, not an alpha trade)
   OR retire the project as "operationally complete, financially
   inconclusive". The walk-forward engine has done its job in that case.
