# RESEARCH_PNL_FINDINGS.md — PNL optimization research

Date: 2026-06-02

## TL;DR — 5 bullets max

- **divergence-v1 (if revived): replace cumulative OBV with MFI(14)** as the divergence indicator (bounded → cleaner pivots, most-cited divergence indicator in the literature) and **add a 4H EMA200 regime gate** (long-only when 15m close > 4H EMA200). Expect win-rate lift from ~44% → ~55–60% but signal count to drop ~70%; budget for that in significance testing.
- **divergence-v1 (if revived): do NOT shelve based on the n=12 bootstrap CI.** At n=12 every bootstrap method is known-undercovered; the defensible verdict is "insufficient evidence to greenlight" via PSR/MinTRL, not "evidence of no edge." Re-state the shelving memo accordingly before killing the strategy.
- **adx-dual-regime: drop any "skip top-decile vol" filter and instead filter the BOTTOM of the vol distribution** (skip when ATR(14) < MA50(ATR), or vol z-score < −1). Nagel (2012, RFS) shows reversal Sharpe *increases* with vol — top-decile skip removes your best trades; compressed-vol skip removes your worst.
- **adx-dual-regime: replace any absolute funding threshold (e.g. >0.05%/8h) with a symbol-specific 30d Z-score gate (|Z| ≥ 2).** Presto 2024 shows single-asset funding has ~0 linear predictive power, but as a veto layer the Z-score formulation generalizes across BTC/ETH/SOL where absolute thresholds do not.
- **Cross-strategy: the literature broadly does NOT support divergence-family strategies on crypto.** The only peer-reviewed crypto RSI-divergence study found it returned −31.85% vs +243.79% buy-and-hold (Chmielewski & Wójtowicz 2023, PMC). Hidden divergence has no peer-reviewed support either. If divergence-v1 is revived it should be on a tight leash with an explicit "kill if not beating buy-and-hold by month N" stop.

## Per-topic summary

### 1. obv_alternatives
Cumulative OBV is unbounded → poor for systematic divergence pivot detection across regimes. The strongest replacement candidates are **MFI(14)** (bounded 0–100, RSI-like, most-cited divergence indicator) and **CMF(20)** (zero-centered, volume-normalized). Avoid A/D Line as a drop-in — StockCharts ChartSchool itself flags that the Money Flow Multiplier ignores inter-period price gaps, propagating into CMF too. No published BTC 15m head-to-head backtest exists across these indicators — all crypto sources are educational. **Apply: swap OBV → MFI(14) in divergence-v1 if revived; use CMF(20) as a secondary zero-line confirmation gate.**

### 2. donchian_retest
Practitioner consensus says retest entries improve R/R via tighter stops but the only quantitative head-to-head I could find (DailyForex 2017, FX H4, 15.5y) found **raw breakout beat pullback entries at every R/R below 20:1**. Donchian's own original Rule #20 already encodes a "1-day reversal" filter so a minimal retest concept is native to the system. Standard parameters from practitioner sources: retest window 1–3 bars, proximity ~0.5–1.5× ATR(14) or ~0.8% for crypto, rejection candle close back beyond level required. **Apply: if testing retest, build as A/B against raw breakout on the same 5 OOS windows, track win-rate and R-multiple separately, consider hybrid split-entry as fallback.**

### 3. volume_profile_poc
No peer-reviewed or vendor-published backtest reports a Sharpe ratio for a BTC 15m POC-bounce system. Every credible source is qualitative. Session definition (00:00 UTC default for crypto), bin width sensitivity, and trend-regime failure are the three dominant pitfalls. Fading the POC during trending regimes is the documented kill mode — multiple sources warn "POC reversion trades will get stopped out repeatedly" in trend. **Look-ahead trap**: naive "current bar's POC" includes the current bar's volume and must be shifted by ≥1 bar. **Apply: low priority for now — too many free parameters and no quantitative basis; if prototyped, require previous-session POC + ADX<20 or balanced-day regime gate, and run a bin-width sensitivity sweep.**

### 4. mtf_divergence
The only peer-reviewed crypto study (Chmielewski & Wójtowicz, PMC 2023) tested daily RSI divergence on BTC 2018–2022 and found −31.85% vs +243.79% buy-and-hold — but did NOT test MTF, so MTF remains an open question. QuantPedia shows HTF trend filter materially improves LTF MACD entries on BTC (Sharpe 0.33 → 1.07), but that result is on MACD crossover, not divergence. Stratbase shows the divergence quality/sample-size tradeoff cleanly: 15m gives 412 signals at 44% WR; 4H gives 38 signals at 58% WR; the celebrated "double divergence at 75% WR" runs on N=12 over 3 years (statistically meaningless). All "80% false signal reduction" vendor claims are unsubstantiated. **Apply: 4H EMA200 regime gate is the highest-evidence MTF change; require N≥100 signals per regime before believing results.**

### 5. funding_filter
Q3 2025 BitMEX data confirms funding >0.05%/8h on Binance BTC is genuinely a tail event (BTC funding exceeds even 0.01%/8h only ~31% of the time). But Presto Research (2024) found ~zero R² for single-asset lagged funding → next-period price, and the broader carry alpha has decayed: cross-sectional carry Sharpe fell from 6.45 (2020–2025) to 4.06 (2024) to **negative in 2025**. Crowding is the cited cause. Practitioner consensus uses 0.05–0.10%/8h as "extreme" but cross-asset researchers prefer percentile/Z-score normalization. 2026 has been a sustained negative-funding regime — a stress test for symmetric, direction-aware gates. **Apply: use 30d Z-score (|Z| ≥ 2) per symbol as a veto on same-direction entries, not as a standalone signal; re-test only on 2024–2026 data; make the gate direction-aware so it doesn't mute legitimate contrarian longs during the 2026 negative-funding regime.**

### 6. vol_regime_filter
**This is the topic where the academic literature most strongly contradicts the practitioner intuition.** Nagel (2012, *Review of Financial Studies*) — reversal strategy Sharpe and expected returns *increase* with VIX/realized vol because high vol = withdrawn liquidity = higher compensation for the reversal trade. Avramov-Chordia-Goyal (2006): largest reversals occur in high-turnover, low-liquidity stress regimes. Nam et al. (ANST-GARCH): contrarian profits concentrate in the high-vol-following-down-move state — suggesting any vol filter should be **directional**, not symmetric. Practitioner filters that *do* help target the LOW end (e.g., ATR(14) > MA50(ATR)). **Apply: do NOT add a top-decile-vol skip; bin existing trades by realized-vol decile and confirm Sharpe is highest in upper deciles; if filtering, kill compressed-vol regime entries; rely on the existing −15% kill switch for genuine tail risk.**

### 7. bootstrap_lowN
At n=12 trades, **no bootstrap method gives reliable CIs**. Block-bootstrap coverage breaks down badly at small n, the Politis-White optimal block length itself becomes unestimable, and the Lo (2002) asymptotic SE for SR=1, n=12 gives a 95% CI of roughly ±0.69 — wider than most plausible point estimates. The defensible tool is **Probabilistic Sharpe Ratio + Minimum Track Record Length** (Bailey & López de Prado 2012): for SR≈1 with normal returns, MinTRL is ~30 observations; with negative skew/fat tails it jumps to 50–100+. **Apply: do not lead with a bootstrap CI for shelving divergence-v1. Report (1) point Sharpe + Lo asymptotic SE, (2) PSR vs threshold 0 and vs deployment hurdle, (3) MinTRL — "we'd need X more trades to reject SR≤0." Reframe shelving as "insufficient evidence" not "evidence of no edge." If bootstrap is used, run StationaryBootstrap on bar returns (not trade P&Ls) via `arch.bootstrap` with studentized intervals and B≥5000.**

### 8. hidden_divergence
The peer-reviewed PMC 2023 paper tested **regular** RSI divergence only, so its negative verdict does not directly transfer to hidden divergence. Hidden divergence is mechanically a pullback-continuation pattern, structurally distinct from a reversal signal. However, **all positive empirical claims (50–65% WR, "+25% alpha in bull regimes") trace to blogs/vendor content with no traceable primary backtest** — the +25% figure looks like AI-generated SEO copy. Hidden divergence is essentially equivalent to "buy the dip in an uptrend," which a simple pullback-to-MA rule already captures. **Apply: prototype only with a pre-committed fixed-parameter test (RSI-14, pivot lookback 5, EMA-200 trend filter), restricted to hidden bullish on 4H/1D BTC+ETH+SOL, and require it to beat both regular divergence AND a vanilla pullback-to-MA baseline on the same OOS windows before earning any more tuning budget.**

## Cross-topic contradictions

1. **Vol-filter direction (vol_regime_filter):** academic literature (Nagel 2012; Avramov et al. 2006) says reversal Sharpe *increases* with vol; many practitioner write-ups and the user's framing implicitly assume skipping high-vol regimes helps. The academic side wins on rigor — skipping top-decile vol likely removes the best trades, not the worst.

2. **Donchian retest vs raw breakout (donchian_retest):** practitioner consensus universally favors retest entries; the only quantitative head-to-head I found (DailyForex, FX H4) favors raw breakout at all R/R targets below 20:1. Unresolved on crypto.

3. **Divergence on crypto (mtf_divergence + hidden_divergence):** peer-reviewed academic work says it doesn't work; the vendor/blog ecosystem (and our own divergence-v1 hypothesis) says MTF/hidden variants rescue it. The vendor numbers are unsubstantiated; the burden of proof for "MTF or hidden saves divergence on crypto" is empirically unmet.

4. **Funding as signal vs veto (funding_filter):** trader-blog sources claim extreme funding produces 12–25% annual returns / Sharpe 3–6 from contrarian directional trading; Presto Research finds ~0 R² for single-asset funding → price. Resolution: blog claims conflate decaying cash-and-carry arbitrage with directional reversal trading. Funding works as a veto layer, not as a primary signal.

5. **OBV alternatives ranking (obv_alternatives):** no published BTC 15m head-to-head exists; ranking MFI > CMF > Force Index > ADL is consensus-based, not data-backed for our specific timeframe.

## Top 5 actionable changes — ranked by evidence strength + ease of implementation

### 1. Replace bootstrap-based shelving verdict with PSR + MinTRL framework
- **Rationale:** At n=12 trades, every bootstrap method is known-undercovered (Hall/Horowitz; AJUR 2024). Lo (2002) asymptotic SE for SR=1, n=12 gives 95% CI ≈ ±0.69 — wider than the point estimate. Bailey & López de Prado (2012) PSR/MinTRL is the canonical small-n tool. Current shelving decision likely rests on noise, not evidence.
- **Evidence strength:** Strong (peer-reviewed methods, broad consensus across `arch` docs, Portfolio Optimizer write-up, Pav SharpeR vignette).
- **Build cost:** S (Python implementation is ~30 lines using `scipy.stats.norm` + skew/kurt estimates from `scipy.stats`).
- **Touch:** new tool in `tools/`, update divergence-v1 shelving memo.

### 2. Add 4H EMA200 regime gate to divergence-v1 entries
- **Rationale:** QuantPedia (2025-11-13) shows 4H-trend-filtered 1H MACD entries on BTC improved Sharpe 0.33 → 0.80 → 1.07. Stratbase (2021–2024 BTC) shows 4H divergence at 58% WR vs 15m at 44% WR. Same direction of evidence; expect WR lift ~44% → ~55–60% with ~70% signal-count drop.
- **Evidence strength:** Moderate (one solid backtest source, transfers across MACD → divergence is plausible but unproven).
- **Build cost:** S (single condition: `close_15m > ema_4h_200` for longs, inverse for shorts).
- **Touch:** `divergence-v1` strategy code if revived, `config/params.yaml`.

### 3. Replace cumulative OBV with MFI(14) as the divergence indicator
- **Rationale:** OBV is unbounded → poor for normalized divergence pivot detection. MFI is bounded 0–100, RSI-like, and is the most-cited divergence indicator in pandas-ta and ta-lib documentation. CMF(20) as a secondary zero-line gate. Avoid A/D Line (gap-blind, per StockCharts ChartSchool).
- **Evidence strength:** Moderate (strong qualitative consensus, no published BTC 15m quantitative head-to-head).
- **Build cost:** S (`pandas_ta.mfi` and `pandas_ta.cmf` are one-line replacements; existing pivot detection logic reusable).
- **Touch:** divergence-v1 indicator block.

### 4. Replace absolute funding thresholds with 30d Z-score per symbol
- **Rationale:** Q3 2025 BitMEX data shows 0.05%/8h is genuinely tail (fires <5% on BTC, more often on ETH/alts); absolute thresholds don't generalize. Presto Research 2024: single-asset funding has ~0 linear predictive power → use as veto, not signal. Z-score gate is direction-aware and adapts to the 2026 negative-funding regime.
- **Evidence strength:** Moderate (Presto methodology is sound; Z-score formulation is consensus across cross-asset research; exact threshold (|Z| ≥ 2) is convention not proven).
- **Build cost:** S (rolling 30d mean/std of funding per symbol; one comparison in entry logic).
- **Touch:** adx-dual-regime entry filter, requires funding-history schema in `data/state.db`.

### 5. Add ATR-compressed-regime skip (instead of any high-vol skip)
- **Rationale:** Nagel (2012, RFS) — reversal Sharpe increases with vol. Practitioner evidence (QuantStrategy.io) shows ~40% PF improvement from filtering OUT low-vol regimes via `ATR(14) > MA50(ATR)`. SetupAlpha + AlgoXpert flag top-percentile vol filters as classic overfit vectors.
- **Evidence strength:** Strong on direction (academic + practitioner aligned); Moderate on the specific parameterization.
- **Build cost:** S (single ATR comparison).
- **Touch:** adx-dual-regime entry filter, but ONLY after binning current trades by vol decile to confirm Sharpe-by-vol curve before committing.

## What this DOESN'T support

- **Do NOT add a "skip top-decile realized vol" filter.** Nagel 2012 and the broader liquidity-provision literature say the top decile is where reversal logic earns most of its premium. This is the most-likely-wrong intuition in the current research set.
- **Do NOT adopt A/D Line as an OBV replacement.** It inherits OBV's drift problem AND adds gap-blindness (StockCharts ChartSchool, authoritative source for these indicators).
- **Do NOT take vendor claims of "+25% alpha bull / neutral bear" for hidden divergence as evidence.** No primary backtest exists; appears to be AI-generated SEO copy.
- **Do NOT take "75% WR double-divergence" or "Sharpe 3–6 from funding contrarian" claims at face value.** Both run on N≤12 samples or conflate carry arbitrage with directional trading.
- **Do NOT build Volume Profile POC-bounce as a near-term priority.** Zero quantitative backing in the literature; too many free parameters (session, bin width); look-ahead trap is easy to introduce. If built later, requires regime gate + bin-width sensitivity sweep before any Sharpe number is meaningful.
- **Do NOT trust "X% false signal reduction" claims from any vendor source.** Trendrider's "80% false signal reduction" headline is unsubstantiated in body text.
- **Do NOT decide that divergence-v1 has "no edge" based on the current n=12 sample.** The honest verdict is "insufficient evidence." Rebuild the shelving memo around PSR/MinTRL.

## Open questions for further research

1. **Does any divergence-family strategy work on crypto perps?** Only one peer-reviewed paper exists (Chmielewski & Wójtowicz 2023, regular RSI divergence, negative); MTF, hidden, and indicator-swap variants are untested in peer-reviewed work. Our own 5-OOS-window backtest framework is the only way to resolve this.
2. **What is the Sharpe-by-realized-vol-decile curve for our own divergence/ADX trades?** Required before committing to any vol filter. Nagel's result is on US equities daily; crypto perp replication is missing.
3. **Does funding Z-score gate improve expectancy on our specific 2024–2026 sample?** No public backtest tests this hypothesis on our regime. Required before adding the gate.
4. **Does Donchian retest beat raw breakout on crypto 15m specifically?** The DailyForex 2017 head-to-head was FX H4. Crypto microstructure (funding-driven false breakouts, 24/7 trade) could flip the answer.
5. **Bin-width and session-definition sensitivity for Volume Profile POC.** If we pursue POC-bounce, this needs a proper sweep before any Sharpe number is meaningful.
6. **What is the autocorrelation structure of divergence-v1's 12 trades?** Needed to choose between IID percentile bootstrap on trade P&Ls vs stationary block bootstrap on bar returns.
7. **Does hidden divergence add anything over a vanilla pullback-to-MA entry?** If not, hidden divergence is an indicator overload not an edge.
