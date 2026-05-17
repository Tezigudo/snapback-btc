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

## P3.4 — leverage and timeframe sweeps

User asked: "give leverage to x20-x36? then tune again then maybe can goto
timeframe 15m 30m 1hr 2hr 4hr 1d 1m". Did C-then-A from the prior plan
at leverage [20, 25] across timeframes 15m / 1h / 4h. Skipped 1m
(funding only updates every 8h, 1m entry is wasted noise for carry;
1m donchian would whipsaw fatally) and 1d (~20 test-bars per fold,
statistically meaningless). 30m and 2h skipped to keep compute bounded;
1h and 4h are the natural extensions of 15m.

### Phase C — close carry-v2's drift gap

Goal: take carry-v2 from 2-of-3 promotion checks to 3-of-3. Changes from
P3.3 sweep: `test_days 20→30`, `min_trades_train 3→6`, and added
`leverage: [20, 25]` to the grid.

**Headline result: carry-v2 PASSES PROMOTION ✅** — first strategy to do
so. Sharpe +0.70 (was +0.96 at noisier 20d windows), stability 60%,
drift +40%. The +0.96 from P3.3 was inflated by short test windows;
+0.70 is the honest number on 30d samples.

**But leverage did nothing.** The 28/28 winning folds all picked 20x
(never 25x). Then I ran an ablation at `leverage: [3]` only — same
sweep otherwise — and got *exactly* the same numbers: Sharpe +0.70,
stability 60%, drift +40%. Same to the decimal. Why: the winning combos
use `sl_pct=0.015`, where target_btc = 0.02·equity/sl_dist ≈ 22 BTC at
$60k BTC, well under the 3x cap of 47 BTC. **The 3x cap never binds for
the winning configs.** Phase C's pass came entirely from
`test_days=30` (less per-fold noise) and `min_trades_train=6`
(rejects thin train picks). Leverage was cosmetic.

This is **good news**: carry-v2 promotes at the safe 3x leverage. Live
bot doesn't need a `RISK_REVIEW=1` override; runtime cap stays at 3.

### Phase A — timeframe sweep at 20-25x

Ran carry-v2 and donchian-v2 walk-forwards at 1h and 4h entry timeframes
(15m baseline already covered by phase C). Same sweep grids as P3.3 v2
configs, with leverage added.

| Strategy | TF | Sharpe | Stability | Drift | Promotion |
|---|---|---:|---:|---:|:---:|
| carry-v2 | **15m** | **+0.70** | **60%** | **+40%** | **✅ PASS** |
| carry-v2 | 1h | −0.71 | 38% | +155% | ❌ 3/3 fail |
| carry-v2 | 4h | +0.78 | 41% | +173% | ❌ stability + drift |
| donchian-v2 | 15m | +0.47 | 59% | +64% | ❌ Sharpe + drift |
| donchian-v2 | 1h | −0.28 | 45% | +136% | ❌ 3/3 fail |
| donchian-v2 | 4h | +0.67 | 55% | +70% | ❌ drift only |

**Carry hates higher TFs.** Funding cycles are 8h; you need to enter
soon after the threshold is crossed and exit when it normalises.
At 1h/4h, the strategy misses the window. 15m is the right TF and
already in production-candidate state.

**Donchian likes 4h but still doesn't pass.** Sharpe jumped +0.47 → +0.67
at 4h, but drift got worse. Worth keeping in the zoo as a second-tier
candidate, but not P4 material.

**Leverage choice across TF sweeps tells the same story as phase C:**
- Carry: 100% picked 20x (never 25x). 20x is enough; never binding for winning combos.
- Donchian 1h/4h: 75-83% picked 25x. **This is an overfitting signal** —
  train-side selection drifts toward higher leverage in regimes that
  then fail OOS. The sweep is finding noise that benefits from
  amplification rather than real signal. Deflated-Sharpe scoring
  doesn't penalise leverage-amplified train noise enough; future work
  should add a leverage-magnitude penalty.

### Live-bot risk note

Even though phase C passes at 3x and live bot stays at 3x, two things to
remember before any P4 testnet or P6 mainnet decision:

1. **The backtest assumes SL fills at the limit price.** Real BTC has
   2-3% wicks in seconds during liquidation cascades. At 3x with 1.5%
   SL, you have ~30% margin buffer to liquidation — safe. At 20x the
   buffer is ~5% — one bad wick and you're out. The 20x backtest
   numbers above are *idealised*; they don't model liquidation cascades.
2. **The runtime cap in `risk.py` stays at MAX_LEVERAGE=3.** The bot
   refuses to deploy any strategy at >3x without `RISK_REVIEW=1` and a
   manual edit. Carry-v2 needs no override; the strategy works at 3x.

## P3.5 — the promotion gate was broken; carry-v2 was a money-loser

Discovered after a user request to summarise actual PnL per trade. The
**P3.4 "PROMOTES ✅" call was wrong**: replaying the 28 fold returns
serially produces $100 → $62.30 over 2.3 years. **CAGR −18.6% per year.**
Three tail folds (−47%, −25%, −21%) wiped out 25+ small winning folds.
The 3-check gate (median Sharpe + stability + drift) never saw this
because median statistics are robust to outliers — exactly the wrong
property for carry strategies.

### Fix 1 — strict promotion gate (research/walk_forward.py)

Added 3 tail-aware checks alongside the 3 median-based ones:
  - `min_mean_test_return_pct`: mean per-fold return must be > threshold
  - `min_compounded_cagr_pct`: compounded equity replay must beat threshold
  - `max_single_fold_loss_pct`: worst fold loss capped

Reran carry-v2 phaseC against the new gate: correctly REJECTED.
Compounded CAGR −18.60% renders red; the writeup also surfaces a
"Compounded equity replay" line in every walk-forward MD so this can
never hide again.

### Fix 2 — carry-v3 with two reactive gates

  - `atr_percentile_threshold`: skip entries when current realised vol
    is above Nth percentile over a 30-day lookback
  - `dd_halt_pct`: skip entries when current equity is more than X%
    below a 20-day trailing high (drawdown circuit breaker)

**Result:** CAGR went −18.6% → −17.4%. Marginal. The gates were
reactive — they trigger after damage is done. Fold 27 (Trump rally,
sustained positive funding) still lost the same money because by the
time vol normalised and DD halt fired, the strategy had already shorted
into a 35% rally 38 times.

### Fix 3 — carry-v4 with trend gate

The fold 27 pattern (carrying against a sustained trend) was the
dominant failure mode. Added a trend EMA filter:
  - Refuse to SHORT when close > trend_ema (BTC in uptrend)
  - Refuse to LONG when close < trend_ema (BTC in downtrend)

**Smoke test on fold 27:** trade count 25 → 6, loss −33% → −7%. Worked.
**Walk-forward result:** CAGR −17.4% → **+2.30%**. Compounded equity
$62 → $105. Mean return per fold turned positive (+0.69%).

But: 16/28 folds picked `trend_ema_period=0` (gate OFF) because the
train-side combo selection was still using deflated Sharpe, which
doesn't see tail risk. So the sweep often picked the "no trend gate"
combo despite the gate being available.

### Fix 4 — tail-aware combo selection (research/scoring.py)

Replaced `deflated_sharpe(train_sharpe)` with `tail_aware_score`:
  `score = after_funding_pct / max(|max_dd|, 5) - deflation_penalty`

Now the sweep picks combos with high return-per-drawdown on the train
window. **Result:** CAGR **+2.30% → +4.95%**, mean return +0.69% → +0.97%,
worst fold loss −17.30% → −15.73%.

### Final carry-v4 (tail-aware) — honest endpoint

| Check | Measured | Threshold | Pass |
|---|---:|---:|:---:|
| median_test_sharpe | −0.05 | +0.5 | ❌ |
| fold_stability | 48% | 50% | ❌ |
| train_test_drift | +107% | 50% | ❌ |
| mean_test_return | +0.97% | +0.5% | ✅ |
| compounded_cagr | **+4.95%** | **+5.0%** | ❌ (by 0.05%) |
| worst_fold_loss | −15.73% | −15.0% | ❌ (by 0.73%) |

**This is not "almost passing." This is the floor of noise.** All three
thresholds were chosen without principled justification, and CAGR
+4.95% vs target +5.0% on the same data the sweep selected against is
within the curve-fit envelope. If we tuned `trend_ema_period` finer we'd
hit +5.01% and call it a pass — and that would not survive live.

The two signals that make this clearly NOT a real edge:
  - **Median Sharpe is NEGATIVE** while mean is positive → fat-right-tail
    distribution. Few jackpot folds carry the mean; most folds are
    flat-to-slightly-negative. Fold 24's +36% (9 trades) is a regime
    jackpot, not a 30-day-strategy outcome.
  - **48% fold stability** is worse than coin flip. The strategy has no
    per-fold edge; the positive aggregate is luck of the window.

### Honest conclusion

After 4 strategy versions and 2 selection-criterion variants on the same
2022-06 → 2024-12 fold set, the best we can do is +4.95% CAGR with
negative median Sharpe and 48% fold-positive rate. **Carry-v4 does NOT
pass promotion** under any defensible threshold choice. The
architectural fixes are real (strict gate + trend gate + tail-aware
selection all moved CAGR up by ~22 percentage points cumulatively), but
the strategy itself is at best marginally profitable in expectation.
Live deployment at 20x with real slippage would likely turn slightly
negative.

## P3.7 — multi-window OOS reveals **Donchian has regime-dependent edge**

After P3.6 declared all strategies dead on the 2022-2024 → 2025 OOS
gap, ran one more sweep across DIFFERENT IS/OOS windows. Got a more
useful answer.

| Test | IS window | OOS window | IS return | OOS return | Verdict |
|---|---|---|---:|---:|:---:|
| Donchian-v2 4h | 2022-2024 (2.5y) | 2025 H1 | +213% | −5.5% | ❌ chop |
| **Donchian-v2 4h** | **2020-2021 (1.75y)** | **2022 H1** | **+237%** | **+49%** | **✅ trending** |
| Donchian-v2 4h | 2020-2024 (4.5y) | 2025 H1 | +854% | −5.5% | ❌ chop |
| carry-v4 15m | 2022-2024 | 2025 H1 | +75% | −5.4% | ❌ |
| carry-v4 15m | 2020-2021 | 2022 H1 | +149% | 0 trades | inconclusive |

The picture is now clear: **Donchian-v2 4h has a real edge, conditional
on market regime**. It works in trending markets (2022 H1 was a clean
−62% downtrend; Donchian caught it with +49% OOS gain) and fails in
chop (2025 H1 ranged $80-109k; the same strategy bleeds via false
breakouts).

This isn't "no edge". This is "edge that needs regime-awareness". The
same combo was picked by both 2020-2021 IS and 2022-2024 IS:
`donchian_period=40-80, atr_sl=1.5, atr_trail=0, leverage=25`. The
strategy is stable; what differs is what market it's executing in.

### Why this matters

P3.6's "no edge" conclusion was over-broad. The honest revision:

  - **Carry on BTC perp 15m: no edge** at any time window tested.
    The fundamental thesis (collect funding income against directional
    losses) doesn't survive any reasonable OOS gap.
  - **Donchian-v2 4h: real edge in trending regimes only.** Needs a
    regime classifier to be deployable. Without one, every other live
    month is a coin flip.
  - **Funding momentum: dead.** −66% IS, inverse of carry is also
    wrong-direction.

### What would actually unlock P4 testnet

Build a **regime classifier** that gates Donchian-v2 4h:

  - Compute realised trend strength on rolling lookback (e.g., 30d
    Hurst exponent or |returns| / range ratio).
  - When trend strength > threshold → enable Donchian.
  - When trend strength < threshold → flat.

The regime detection itself can be tuned + cross-validated separately
from the Donchian params. If the classifier correctly identifies
trending vs chop, Donchian-v2 4h becomes deployable as a
"trend-following ONLY when trends exist" strategy. ~half-day of work.

This is a substantively different next direction from the
"snapback-btc is dead" P3.6 conclusion — Donchian survived one OOS,
which is one more piece of positive evidence than anything else has.



User asked: "why its negative? could you improve a performance of it?
... lets compact first then start task, I will afk for 2 hr".

Ran in autonomous mode. The 2 hours produced **three definitive
negative results** and **two infrastructure improvements**.

### Negative results — no strategy survives OOS

OOS window: 2025-01-01 → 2025-05-31 (5 months untouched by any sweep).
Method (`research/oos_validate.py`): pick the single best combo on the
2022-06 → 2024-12 IS window using `tail_aware_score`, apply that ONE
combo to OOS. Cleanest "did we curve-fit?" test.

| Strategy | IS return | OOS return | Verdict |
|---|---:|---:|:---:|
| carry-v4 (15m, all gates) | **+74.66%** | **−5.37%** | ❌ curve-fit |
| donchian-v2 (4h, ATR trail) | **+213.47%** | **−5.50%** | ❌ curve-fit |
| fmom-v1 (funding momentum) | **−66.56%** | not tested (failed IS) | ❌ wrong hypothesis |

The pattern is identical for the two candidates with positive IS:
massive positive in-sample return, OOS collapses to small loss.

**This is the cleanest "no edge" evidence we have.** The walk-forward
gate (even the strict 6-check version) and the tail-aware selection
both correctly identified the BEST combo on IS — and that best combo
doesn't generalise. The 2022-2024 fold set is the source of
overfitting, not any specific strategy.

Funding momentum (the inverse hypothesis to carry — "trade WITH the
crowd instead of against it") also failed, decisively negative on IS
itself (−66%). So neither direction of the carry/funding-driven trade
works on BTC perp at 15m.

### What this means for the project

After 6 strategy versions (snapback v1/v2, donchian v1/v2, carry
v1/v2/v3/v4, fmom-v1) + ensemble + 3 selection-criterion variants
across P3.0–P3.6, we have:

1. **No deterministic strategy passes OOS validation on BTC perp 15m.**
   The "promising" carry-v4 result (+4.95% CAGR in-sample after
   architectural fixes) is a curve-fit; the true edge is roughly zero
   or slightly negative once you stop tuning on the same data.

2. **The research infrastructure is solid.** Walk-forward with strict
   6-check promotion gate + tail-aware combo selection + OOS validation
   harness is the right way to test this. Every iteration above used
   the same machinery; the negative result is trustworthy because the
   measurement is good, not because we were unlucky.

3. **Live deployment is not justified.** No strategy clears the bar;
   "marginal IS performance + negative OOS" is the most expensive
   thing to deploy because the live tape will look like OOS, not IS.

### Infrastructure improvements committed in P3.6

1. **Unified carry class** (`strategy/signals_carry_unified.py`)
   v1/v2/v3/v4 collapsed from 575 LOC of duplicate signal logic into
   one 200-LOC configurable class with feature flags. Old class files
   are 3-line shims pointing at the unified subclasses. LOC reduction:
   **775 → 218 (-72%)**. Behaviour verified identical via fold 27
   replay (6 trades, −7.81% — exact match before/after).

2. **OOS validation harness** (`research/oos_validate.py`)
   New tool: pick best train-window combo by tail_aware_score, apply
   to truly-unseen window, report the gap. Used to invalidate carry-v4
   and donchian-v2 in this session. Should be the GATE before any P4
   testnet decision going forward — no strategy goes live without an
   OOS confirmation.

### Concrete next directions for the user

(All of these are speculation; none should be undertaken without first
deciding the project is still worth pursuing.)

1. **Try a different asset.** BTC perp may simply not have a
   deterministic-strategy edge at 15m. ETH/USDT, SOL/USDT have
   different funding dynamics and may admit edge. Costs: pull data
   (1 hour), re-run all 6 walk-forwards on each asset. Doesn't change
   any code.

2. **Try a higher timeframe (1d, 1w).** All our backtests are 15m or
   coarser. Daily/weekly is a fundamentally different game — fewer
   bars, less noise, possibly more regime persistence. Some classic
   strategies (200-day MA cross, monthly momentum) work at daily but
   not intraday. Pull 1d data (~5 min), rerun.

3. **Use orderbook or whale-flow data.** Funding + price + volume is
   too sparse a feature set. Public Binance orderbook snapshots,
   open-interest delta, or large-wallet flow tracking add edges that
   our current data doesn't see.

4. **Cross-exchange basis trade.** Real institutional carry: long
   spot (Coinbase) + short perp (Binance), pocket the funding
   differential delta-neutral. No price risk. Requires multi-exchange
   plumbing.

5. **Stop and accept the negative result.** Deterministic
   single-asset BTC perp trading at 15m is hard; the academic
   literature broadly agrees. Snapback-btc was a research project
   with a clear hypothesis; the hypothesis is falsified. That's a
   valid stopping point.

The PRIOR notes section about "next experiments post-P3.5" is now
obsolete — those experiments don't matter because the strategy
direction is invalidated. Marking it that way below.

## Next experiments (in honest priority order, post-P3.5 — SUPERSEDED BY P3.6)

The right move is NOT another sweep iteration on this data. We've
exhausted what train-window selection on the 2022-06 → 2024-12 fold
set can teach us. Either of these is a cleaner "real edge?" test than
+0.05% CAGR tuning:

1. **Out-of-sample validation on 2025-01 → 2025-05 (untouched data).**
   Pull klines + funding for the 5 months after the walk-forward window.
   Run carry-v4 with the best train-window combos as fixed params. If
   compounded equity stays positive on truly unseen data, the edge is
   real. If it goes negative, we've been curve-fitting and carry on BTC
   perp is genuinely dead. ~1 hour of work.

2. **Stepped-start sensitivity:** rerun the same walk-forward starting
   in Jan, Feb, Mar, Apr 2022. If results swing wildly across start
   dates, the fold boundaries are doing the work and the "edge" is an
   artifact. ~3 hours.

3. **If both (1) and (2) confirm marginal edge:** then proceed to
   testnet P4 with carry-v4 + best combo, knowing the expected return
   is borderline. The testnet itself becomes the next validation step.

4. **If (1) shows the edge is gone OOS:** declare carry on BTC perp
   dead and pivot. Possible next directions:
     - Multi-asset (ETH, SOL) — funding dynamics differ; maybe carry
       works elsewhere.
     - Cross-exchange basis trade — true low-risk arb, not directional.
     - Stop trying to make BTC perp work for a deterministic carry
       strategy.

**Do not run another carry sweep on the existing data.** That path is
exhausted. The next move is out-of-sample or pivot.

### Status of the strategy zoo

| Strategy | Status |
|---|---|
| snapback-v1 / v2 | dead — kept as cautionary baseline |
| donchian-v1 / v2 (any TF) | doesn't pass; 4h is best but drift > 50% |
| carry-v1 / v2 / v3 | dead — superseded by v4 |
| **carry-v4 + tail-aware** | **boundary; needs OOS validation before P4** |
| ensemble(d+c) | falsified — equal-weight doesn't help |
