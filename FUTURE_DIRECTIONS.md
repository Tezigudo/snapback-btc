# FUTURE_DIRECTIONS.md — Where divergence-v1 sits, and what comes after

Date: 2026-06-01
Author: Claude Code (synthesis of pre-mortem + alternatives scan + EV ranking)
Status: Decision support — not a deploy plan. Do not commit code based on this alone.

---

## TL;DR

- **Recommendation:** Do **not** ship divergence-v1 with its current defaults. Backtest math (+3.61% / 12 trades / 6mo / 41.7% WR / -4.66% MaxDD) is borderline at best and three structural bugs in the signal gate (cumulative-OBV degeneracy, trivially-satisfied breakout confirmation, default-on shorts + no trend filter at 20x) credibly turn a marginal backtest into a -25% to -40% live result in a trending 60-day window. Fix the gates, re-test on stress windows, AND ship a leg-local circuit breaker before any live capital.
- **Biggest single risk:** Cascade-buying-the-knife — in a March-2020-style monotonic crash, the centered-fractal detector + RSI-pinned-at-floor + cumulative-OBV trivially satisfy the long-entry gate on the first relief bounce, ATR(14) underestimates SL distance by ~5x, and 20x leverage liquidates before the SL fires. Backtest reports -1%; reality is full-margin wipe.
- **Next strategy to build:** **ADX Dual-Regime (RSI MR when ADX≤25, Donchian-20 breakout when ADX>25).** All indicators already exist in `strategy/indicators.py`, it activates exactly when divergence-v1 should be benched, and it's the only candidate with low build cost AND clean structural complement.

---

## Pre-mortem summary — Top 5 failure scenarios across all lenses

Ranked by probability × impact. Drawn from `live_60d_loss`, `black_swan`, `whipsaw`, `slippage` lenses.

### 1. Default config ships with allow_shorts=True + trend_filter=False + leverage=20 + RSI-OB=65
- **Lens:** live_60d_loss
- **Probability × impact:** high × catastrophic
- **Why it kills you:** Counter-trend divergence shorts at modest overbought (65, not 70) in a 60-day BTC uptrend with no regime guard. Asymmetric expectancy: full SLs hit, partial TPs hit, time-stops close the rest near breakeven. -25 to -40% credibly even before signal-gate bugs compound it.
- **Fix:** Ship v1 with `trend_filter_enabled=True` by default, RSI-OB=70 / OS=30, cap leverage at 5x for v1, add hard "no shorts above EMA200 / no longs below EMA200" gate, add max-3-losing-shorts-per-day circuit breaker.

### 2. Cumulative OBV gate is structurally degenerate; AND-gate collapses to RSI-only
- **Lens:** live_60d_loss
- **Probability × impact:** high × severe
- **Why it kills you:** OBV is an unbounded cumulative sum. In any window with net directional drift (≈every 60-day BTC window), OBV at later swing-low b2 is mechanically greater than at earlier swing-low b1 — the "OBV higher low" check is near-tautological for longs in an uptrend. The team mentally modeled that OBV would cut ~half of RSI signals; live trade frequency is 2-3x what they expect.
- **Fix:** Replace cumulative OBV with windowed OBV-slope or OBV-residual: regress OBV on time across the b1..b2 window, require slope < 0 for bearish setups (> 0 for bullish). Add a unit test that injects steady drift and asserts the gate REJECTS it. Telemetry: log what % of RSI signals OBV actually filters out — if < 15%, gate is broken.

### 3. `close > high[b2]` confirmation is trivially satisfied by the firing-bar delay itself
- **Lens:** live_60d_loss
- **Probability × impact:** high × severe
- **Why it kills you:** `b2` is by definition the local low (bottom of a 7-bar centered-fractal valley). The strategy fires at `j = b2 + k = b2 + 3` and requires `close[j] > high[b2]` — but `high[b2]` is structurally lower than the high of every nearby bar. Almost every post-swing recovery candle clears it. Result: 5-10x signal frequency vs intent, with no actual breakout filter.
- **Fix:** Replace with `close > max(high[b2:j+1])` or `close > recovery_max + atr_break_buffer * atr[j]`. Better: require N consecutive closes above the most-recent registered swing-high above b2. Telemetry: median `(close[j] - high[b2]) / atr[j]` — if < 0.3, gate is doing nothing.

### 4. Cascade entry: tight ATR-based SL + 20x leverage = liquidation before SL fires
- **Lens:** black_swan
- **Probability × impact:** high × catastrophic
- **Why it kills you:** In March-2020-style cascades, ATR(14) Wilder lags the realized vol explosion. SL at entry - 1.5*ATR(14) sits ~1-1.5% below entry while next-bar low prints 5-8% below. Backtesting.py fills SL at the bar open or SL price; Binance perp slips 1-5% beyond trigger. With 20x leverage, a 5% slip = full margin wipe. The strategy thinks worst-trade is bounded; it isn't.
- **Fix:** ATR/close > 0.015 hard veto on new entries. Cascade veto: block longs for 4h after 5 consecutive lower closes or 20-bar drawdown > 12%. Slippage stress model in backtest: any bar with H-L > 3*ATR adds 1.5% adverse slippage to stop fills. Hard-cap divergence-v1 leverage at 5x. Max-loss-per-trade auto-HALT at -3% equity.

### 5. No per-strategy circuit breaker between "normal loss" and the -35.5% portfolio kill switch
- **Lens:** whipsaw
- **Probability × impact:** high × moderate (compounds to severe under chop)
- **Why it kills you:** `signals_divergence.py` has no consecutive-loss cooldown, no daily-loss cap, no leg-local HALT, no entry rate-limiter. The portfolio kill at -35.5% is calibrated for combined-deploy catastrophe, not for a single new leg bleeding in chop. A 5-10 false-signal chop streak quietly bleeds 5-10% equity while every guardrail says "within limits." By the time the kill fires from divergence-v1 alone, the leg has lost ~33% and recovery needs ~50% gross return.
- **Fix:** Leg-local `data/HALT_DIVERGENCE` flag. Consecutive-loss cooldown: 3 losses in 24h → suppress new entries for 96 bars. Leg-local daily DD cap: -4% in 48h → leg HALT. Both are ~20 lines in the strategy module.

### Honorable mention — slippage understatement
The 5 bps commission constant in backtest is ~3-10x too low for momentum-break entries (the strategy enters on `close > high[b2]` which by construction has moved through a swing level — next-bar opens routinely slip 1-3 bps in benign conditions, 5-15 bps in low-liquidity hours, 20-60 bps in stressed conditions for $1M+ notional). Combined with stop-cascade slippage on stop-outs (50-150 bps possible), the realized per-trade economics are 10-30 bps worse than backtest. Promotion gate: must stay net-positive at 15 bps roundtrip across all 5 OOS windows.

---

## Alternative strategies summary — Top 5 by EV

From the cross-topic scan (reversal_crypto, exhaustion_15m, regime_switch, volume_alpha, order_flow). Sharpe/winrate numbers are as reported by source; treat as unverified until re-run on the same 5 OOS windows multifactor-v1 used.

| Rank | Strategy | Reported metric | Source | Why it matters |
|---|---|---|---|---|
| 1 | **AdaptiveTrend (arXiv 2602.11708)** | Sharpe 2.41 | arxiv.org/pdf/2602.11708 | Highest credible Sharpe from non-marketing source. Pure trend-follower, structurally avoids divergence-v1's lane. Penalty: high complexity, unknown 15m trade frequency. |
| 2 | BTC-cascade → altcoin MR basket | Sharpe 3.58 OOS 2021-2025 | medium.com/@tigroblanc (2026-02) | Loudest claim, but author concedes ~54% is BTC beta and alpha p=0.18 (not significant). Wrong instrument for this repo (alts, not BTC perp). |
| 3 | Local-extremum BTD MR (10-day lookback) | Return/vol 2.06, AR 98.43%, MDD -37.67% | quantpedia.com | Strongest academically-flavored number, but daily not 15m, and return/vol ≠ Sharpe. Worth re-running on 15m. |
| 4 | **ADX Dual-Regime: RSI MR (ADX≤25) + Donchian-20 (ADX>25)** | No published number | medium.com/@FMZQuant | Boring but lowest build cost — both legs use indicators already in `strategy/indicators.py`. The trend leg activates exactly when divergence-v1 bleeds. Modest per-trade EV, high portfolio EV. |
| 5 | Stacked bid/ask imbalance (footprint, freqtrade OOTB) | No published number | freqtrade.io/en/stable/advanced-orderflow/ | Orthogonal signal class (continuation, order-flow). Cost: freqtrade trade-feed wiring you don't run live yet. Modest expected 15m edge. |

**Honorable mentions out of top 5:**
- Volume Profile POC bounce (24h rolling) — structural complement to chop, but qualitative literature only.
- StochRSI + EMA(50) — PF 1.6 on BTC 4H (non-marketing source), not portable to 15m without re-test.
- Hurst-Exponent regime router — clean concept, but it's a router not a strategy; EV depends on underlying legs.

**Explicit anti-evidence to respect:**
- PMC paper (Effectiveness of RSI Signals, 2023) found classic RSI OB/OS reversal AND long-only RSI divergence both lose to buy-hold on crypto. Authors "strongly advise against." This is the divergence family that includes divergence-v1.
- Curupira walk-forward (2025-2026) on liquidation-cascade fades: works on ETH/SOL, **fails on BTC** (PF ~1.5, MDD 8x worse). BTC's deep book absorbs forced flow; cascade fades are not a BTC strategy.

---

## Cross-strategy ranking from the compare step

Full 32-entry EV ranking is in the EV-comparison input; here's the structurally important slice:

| Rank | Strategy | EV/trade (bps) | Confidence | Notes |
|---|---|---|---|---|
| 1 | AdaptiveTrend (arXiv) | 25 | low | High build cost |
| 2 | BTC→alts MR basket | 30 | low | Wrong instrument |
| 3 | Local-extremum BTD (daily) | 20 | low | Wrong timeframe |
| 4 | ADX Dual-Regime | 15 | medium | **Highest confidence + lowest build cost** |
| 5 | Footprint imbalance | 12 | low | Needs freqtrade wiring |
| 6 | Volume Profile POC | 10 | low | Qualitative only |
| 7 | StochRSI + EMA(50) | 10 | low | Wrong timeframe (4H) |
| 8 | Hurst router | 10 | low | Router, not strategy |
| 9 | HMM regime-switch | 10 | low | High build cost, fragile refit |
| 10 | Connors RSI(2) + MA | 8 | low | Cheap; in-trend pullback lane |
| ... | ... | ... | ... | ... |
| **17** | **divergence-v1 (current)** | **3** | **low** | **~30 bps equity/trade gross of fees, ~0 net of 4-6 bps roundtrip slippage. n=12 sample indistinguishable from breakeven.** |
| 19-20 | RSI OB/OS + RSI divergence (PMC) | 1 | medium | Explicit anti-evidence |
| 28-29 | Liquidation cascade fade / sub-second OFI | 0 | medium | Wrong instrument or wrong timeframe |

**Critical reads:**
- `divergence_v1_rank: 17` of 32 — middle of the pack, sub-3-bps per-trade net edge, n=12.
- `most_complementary_alternative: ADX Dual-Regime` — confirmed by topic-level top pick AND cross-topic ranking.
- The strategies that beat divergence-v1 on EV all either (a) need higher build cost, (b) need re-testing on 15m, or (c) are wrong-instrument/wrong-timeframe — i.e. none are drop-in superior; divergence-v1 isn't dominated, it's just borderline.

---

## Top 3 next strategies to build (prioritized)

### 1. ADX Dual-Regime — RSI MR (ADX≤25) + Donchian-20 breakout (ADX>25)
- **Why now:** Lowest build cost (all indicators exist), highest structural complementarity to divergence-v1 (activates in trends, where divergence-v1 bleeds), only candidate with `medium` confidence in the EV ranking.
- **Build plan:** ~1 day of work. ADX threshold is the single knob to walk-forward tune on the same 5 OOS windows multifactor-v1 used. Run alongside divergence-v1 ONLY after measuring monthly-return correlation < 0.4.
- **Promotion gate:** Backtest must clear net-positive on all 5 OOS windows at 15-bps roundtrip; max-window DD ≤ multifactor-v1's worst window; trade frequency between 30-200 trades/window (sanity).
- **Risks to flag in pre-mortem:** ADX whipsaws around the threshold (regime-thrashing); breakout leg eats slippage on the entry bar; Donchian-20 false breakouts in low-vol regimes.

### 2. Volume Profile POC bounce on 24h rolling fixed-range
- **Why second:** Structural-level signal (institutional defense of HVN) complements divergence-v1's chop failure mode. Implementable in pure Python without tick data. Best candidate for the "what fires when both divergence-v1 and ADX dual-regime are quiet" gap.
- **Build plan:** ~3-5 days. Rolling 24h volume-by-price histogram, POC = highest-volume bin, LVN above/below as structural risk. Entry: retest of prior session's POC on declining volume; exit at next HVN or LVN break.
- **Promotion gate:** Same 5 OOS windows; additionally must show non-zero trade count in the 2024 H1 chop window (where multifactor-v1 and divergence-v1 both struggle).
- **Risks to flag:** POC drift in trending markets makes the level meaningless; declining-volume detection is noise-prone on 15m; "session" definition is arbitrary (UTC day vs Asian/Europe/US sessions).

### 3. AdaptiveTrend (arXiv 2602.11708) — pure trend-follower with vol/correlation-scaled sizing
- **Why third:** Highest credible Sharpe (2.41) in the scan, structurally avoids divergence-v1's lane, and a real published paper means the parameter choices are documented. Penalty: highest build cost of the top 3, unknown 15m trade frequency, and uncertain whether the paper's vol-scaling generalizes to BTC perp's funding regime.
- **Build plan:** ~1-2 weeks. Implement core trend signal first (no vol/correlation scaling), validate on OOS windows, then layer the adaptive sizing. Funding cost MUST be modeled — a 24h+ trend hold on 20x with -0.05%/8h funding eats ~1.5%/day of notional.
- **Promotion gate:** Same as above, AND funding-adjusted PnL must remain positive across all 5 windows; trade frequency must allow Sharpe estimation (n ≥ 50 per window).
- **Risks to flag:** Paper's number is Sharpe 2.41 across the asset universe, not BTC-only — single-asset Sharpe will be lower; vol-scaling may chop the position in BTC's wild vol regime; the paper doesn't model perp funding.

---

## What would change about divergence-v1 if we knew today what we'll know in 6 months

Speculative but useful — taking the pre-mortem seriously as ex-ante evidence of what live will teach us.

1. **The cumulative-OBV gate would be the first thing rewritten.** In 6 months we'll have telemetry showing OBV filters < 15% of RSI signals (or worse, < 5%). We'll know the gate was theater. Fix today: windowed OBV-slope regression across b1..b2; unit test with steady drift injection.

2. **The default config would have shipped with `trend_filter_enabled=True` and `leverage=5`.** In 6 months the live equity curve from `trend_filter=False, leverage=20` will look like a stair-step down with 3-5 catastrophic single-trade losses ≈ -3 to -8% equity each. The "we'll choose per-window" framing in DIVERGENCE_PLAN.md will be retroactively obviously wrong: ship the safe variant by default, let Phase-4 OOS choose if the unfiltered one is worth the risk.

3. **The confirmation gate would compare against the recovery window's high, not the swing-bar high.** In 6 months we'll have a telemetry record showing median `(close[j] - high[b2]) / atr[j]` ≈ 0.1-0.2 — i.e. the "breakout confirmation" was usually a 0.1-ATR move, which is noise. Fix today: `close > max(high[b2:j+1]) + 0.25 * atr[j]`.

4. **Funding cost will dominate trade economics on 24h holds.** In 6 months the backtest-vs-live PnL gap will be ~1-2% per month, and ~half of it will trace to funding the backtest didn't model. Fix today: subtract realized 8h funding × notional from each trade's PnL in the backtest. Add a funding-extreme veto: block longs when last 24h funding < -0.05% per 8h (heavy long-pay regime).

5. **The leg-local circuit breaker will have saved at least one bleed.** In 6 months we'll have at least one 5-10 false-signal chop streak where the portfolio kill at -35.5% didn't fire and the user had to manually halt at -7% leg DD. The leg-local HALT_DIVERGENCE flag and consecutive-loss cooldown will be retroactively obvious. Fix today: 20 lines in `signals_divergence.py`.

6. **Slippage assumptions will have been wrong by 10-30 bps per trade.** In 6 months we'll have a live-vs-backtest fill-quality reconciler showing median entry slippage 8-15 bps and stop-slippage 20-60 bps. Promotion gate at 15 bps roundtrip will have been the right line; the current 5-bps assumption will look naive.

7. **Time stop at 96 bars will have closed at least one local capitulation low.** In 6 months we'll have a "puked the low" log line from the cascade case where 24h after a cascade entry was the local bottom and the bounce followed in hour 25. Vol-adjusted exit (close if loss > 1.5×sl_dist OR 24h passed AND realized vol below entry vol) would have held that one.

8. **Backtest parity at 99% will have hidden a chop-cluster of 1% mismatched bars.** In 6 months `divergence_validate.py` parity report will look fine in aggregate but per-month it'll show one bad chop month at 98.5% parity — and that month will be where the bleed happened live. Tighten today: per-month parity reporting; block promotion if any single month < 99.5%.

9. **In 6 months the divergence family will still be net-positive across the BTC literature only because the few public studies all had trend filters and lower leverage.** The PMC paper's "strongly advise against" finding for RSI divergence on crypto isn't going to age out. The honest framing: divergence-v1 is a coin-flip strategy whose edge comes from the AND-gates working as designed. If the AND-gates are theater (issues #1 and #3 above), it's a coin flip at fee cost — i.e. negative EV.

10. **Multifactor-v1 + divergence-v1 correlation will turn out to be > 0.4 in chop.** In 6 months the combined-leg drawdown in a 2024-H1-style chop month will be ~ -15% to -20% (vs ~ -12.5% multifactor-only) because both legs share the chop failure mode at slightly different decision points. The "run both legs simultaneously" framing in DIVERGENCE_PLAN.md needs a portfolio-level cap (divergence-v1 capped at 25% of portfolio equity) before live.

---

## Bottom line

Divergence-v1 sits 17th of 32 candidates on per-trade EV at the threshold of breakeven. The Phase-2 backtest (+3.61%, n=12) is too small a sample to distinguish from noise; three structural gate bugs (cumulative-OBV degeneracy, trivial breakout confirmation, default-unsafe config) credibly turn it negative live. None of the 31 alternatives is drop-in superior — but the #4-ranked **ADX Dual-Regime** is cheaper to build, structurally complementary (activates when divergence-v1 should be benched), and has higher confidence in its EV estimate.

**Decision-support recommendation:** Finish divergence-v1's Phase-4 OOS gate with the three structural fixes applied AND the leg-local circuit breaker shipped. In parallel, start a one-page spike on ADX Dual-Regime so it can enter Phase-3 OOS as soon as divergence-v1 either passes or is shelved. Do not start AdaptiveTrend or Volume Profile POC until one of the first two is proven on the same 5 OOS windows multifactor-v1 was judged on.
