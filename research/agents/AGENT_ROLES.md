# Researcher agent roles

This file documents the role taxonomy the `Researcher` protocol is shaped
after. It is **design intent**, not a contract. The only thing the
walk-forward engine ever calls is `Researcher.commentary(folds)` and
`Researcher.next_sweep_ranges(folds)`.

## Hard rules (from `CLAUDE.md`)

- **No Researcher is ever invoked from the trading loop.** They consume
  *fold results* after walk-forward completes. The trading code path
  (`strategy/`, `exchange/`, `bot.py`, `monitor.py`) doesn't even import
  the `research/` package.
- **The default researcher (`DeterministicResearcher`) makes zero API
  calls and depends on zero external services.** That's the contract for
  "no paid API as init".
- **Any LLM-backed implementation is opt-in.** It must be a new file in
  `research/agents/`, must not be the default, and must be selectable via
  CLI or config, never by import order.

## Role taxonomy (borrowed shape)

Mirroring the analyst/researcher/trader/risk taxonomy from
[TradingAgents](https://github.com/TauricResearch/TradingAgents) and the
LLM-optional regime-aware pattern from
[AgentQuant](https://github.com/OnePunchMonk/AgentQuant). Below is the
list of *future* researcher implementations we'd want; none of them ship
in this PR.

| Role | What it would do | Inputs | Output |
|---|---|---|---|
| `TechnicalAnalyst` | Look at per-fold winning params + indicator distributions, comment on whether the strategy is leaning trend-follow vs mean-revert | FoldResult list + indicator distributions | natural-language summary |
| `RiskReviewer` | Surface tail-risk concerns: max DD, deepest losing fold, consecutive loss runs | FoldResult list + per-trade PnL series | risk bulletins |
| `FundingRegimeAnalyst` | Map fold-by-fold funding-rate regime (positive/negative/neutral) and ask whether the winning combos co-vary with regime | FoldResult list + funding history | regime table + caveats |
| `DebateModerator` | When multiple researchers disagree, call out the disagreement and ask the user to break the tie | researcher commentaries | meta-commentary |

## Concrete future implementations (NOT in this PR)

Each of these is a planned file that does NOT exist yet. Listed so the
seam shape stays honest about where the eventual code lives.

- `research/agents/claude_code_researcher.py` — meta-implementation that
  emits a markdown block addressed to *you* opening Claude Code: "open
  `reports/walk_forward_*.md`, run `/diagnose-folds`". Zero API cost
  because it leverages the user's existing Claude Code subscription.
- `research/agents/ollama_researcher.py` — calls a local Ollama model
  via HTTP. Zero API cost (RAM + electricity instead). Requires Ollama
  installed locally.
- `research/agents/anthropic_api_researcher.py` — direct Anthropic API
  via `anthropic` SDK. Costs money per call. Opt-in only, behind an env
  flag like `RESEARCHER_API_ALLOWED=1`.

## Why the seam exists at all

Without it, "add an LLM later" becomes a refactor. With it, "add an LLM
later" becomes a new file in `research/agents/` and a `--researcher`
flag. The trading code never knows or cares.

This is also the only sustainable answer to the user's question
*"how do we adopt these 7 LLM trading-agent repos?"*: we don't import
their runtimes, we adopt their role taxonomy behind a stable interface,
and any cost-incurring implementation must be an opt-in subclass.
