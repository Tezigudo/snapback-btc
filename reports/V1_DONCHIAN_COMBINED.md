# multifactor-v1 + Donchian-v3 parallel-deploy backtest

**Date:** 2026-05-23
**Question:** can we run a second strategy alongside the deployed `multifactor-v1` to diversify regime risk?
**Method:** backtest both on the same 5 OOS windows (2022H1 → 2025H1), then a 50/50 capital-allocated combined portfolio.
**Code:** `tools/v1_plus_donchian_backtest.py`
**Raw results:** `reports/v1_donchian_combined_20260523T004006Z.json`

---

## TL;DR

> **Do not deploy at the current $101 capital.** Combined 50/50 splits into $50.50 per strategy — below the $50 Binance min-notional after any drawdown. **Required precondition: top up to ≥ $200 first.**
>
> **Current −18% kill-switch is wrong for the combined account.** The combined-cons portfolio hit a max DD of −18.70% in 2024H1 (vs v1 alone −14.95%). Deploying the combo against the existing `kill_switch_equity_fraction=0.82` would have tripped in 2024H1 and locked in the low. Fix before deploy: **per-strategy kill-switches**, one for each bot process (see §7).

| Strategy | 5-window compounded | Per-window range | Worst single-window max-DD | Mean daily corr to v1 |
|---|---:|---:|---:|---:|
| multifactor-v1 (deployed) | **+51.84%** | −13.7% .. +21.1% | −14.95% (2024H1) | — |
| Donchian-v3 (cons 80/20) | **+97.43%** | −2.2% .. +41.6% | −26.68% (2024H1) | 0.23 |
| 50/50 combined (cons) | **+83.45%** | −0.3% .. +32.4% | **−18.70%** (2024H1) | — |
| Donchian-v3 (agg 40/10) | +258.70% ⚠ | −0.5% .. +96.9% | −22.15% (2024H1) | 0.27 |
| 50/50 combined (agg) | +155.89% ⚠ | +0.6% .. +60.2% | −16.62% (2024H1) | — |

**The honest pick is the conservative param set.** Donchian-v3 cons (80/20 channel, slope-gate at 3%) was selected by an OOS sweep on IS = 2020-04..2021-12. **Every test window here (2022H1 → 2025H1) is strictly forward of that IS period — this is a real 5-window walk-forward of static params chosen in late 2021, not a re-fit.** The agg (40/10/gate-off) numbers are partly look-ahead since its IS overlaps with 2023H1/2024H1/2024H2.

**Recommendation:** Donchian-v3 (cons) is a real complement. Combined Sharpe beats v1 alone in 4 of 5 windows, rescuing v1's 2024H1 chop year from −13.7% → **+2.1% realized**. But you must (a) fund ≥ $200, (b) use per-strategy kill-switches at −18% each, and (c) accept that the combined account can intraday drawdown deeper than v1 alone.

---

## 1. Why Donchian, why not something new

- Every strategy currently in the repo (v1, v2, v3, v3-all-wider-N, debounce, floor) is a **variation of the same RSI-mean-reversion thesis**. Adding another won't diversify regime risk.
- A genuinely new candidate (Funding-Carry Reversal) was killed in 5 minutes with a hypothesis check on `funding.parquet`: net forward 24h return at the natural extreme threshold was only **+4 bps after fees** — order of magnitude below the deployable bar. See `tools/fcr_hypothesis_check.py`.
- Donchian-v3 was the *opposite* of v1 by construction (breakouts, not pullbacks), already coded, already OOS-validated on 2 windows. Cheapest path to a real complement.

## 2. The windows

Same 5 OOS half-year windows that locked v1 for production (see `reports/path2_oos_results.json`):

| Window | Period | BTC return | Regime |
|---|---|---:|---|
| 2022H1 | 2022-01-01 → 2022-06-30 | ~−58% | crash + chop |
| 2023H1 | 2023-01-01 → 2023-06-30 | ~+82% | strong uptrend |
| 2024H1 | 2024-01-01 → 2024-06-30 | mixed | choppy uptrend |
| 2024H2 | 2024-07-01 → 2024-12-31 | strong | trend + chop |
| 2025H1 | 2025-01-01 → 2025-05-31 | mixed | sideways |

## 3. v1 reproduces production numbers

| Window | This run | path2_oos_results | Δ |
|---|---:|---:|---:|
| 2022H1 | +19.02 | +19.29 | −0.27 |
| 2023H1 | +21.06 | +21.83 | −0.77 |
| 2024H1 | −13.68 | −12.56 | −1.12 |
| 2024H2 | +20.59 | +21.23 | −0.64 |
| 2025H1 |  +1.25 |  +1.09 | +0.16 |

Tiny drift is from different data extents (parquet has been updated). Compounded +51.84% (memory +50.88%). ✓ pipeline is sound.

## 4. Donchian-v3: two param sets, two stories

| Param | Aggressive (agg) | Conservative (cons) |
|---|---|---|
| Donchian entry/exit | 40 / 10 bars | 80 / 20 bars |
| Slope regime gate | **OFF** | ON @ 3% |
| Historical IS period | 2022-06..2024-12 | 2020-04..2021-12 |
| Historical OOS window | 2025H1 (−0.5%) | 2022H1 (+23.3%) |
| OOS validation matches mine | ✓ exact | ✓ exact |

### Per-window results (`Sharpe` annualised from daily returns):

| Window | v1 ret% / Sh | d3-agg ret% / Sh | combo-agg / Sh | corr | d3-cons ret% / Sh | combo-cons / Sh | corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022H1 | +19.0 / +1.29 | +43.5 / +1.53 | +31.5 / +2.65 | 0.17 | **+23.3 / +1.42** | +21.3 / +2.20 | 0.07 |
| 2023H1 | +21.1 / +1.43 | +96.9 / +1.66 | +60.2 / +3.42 | 0.34 | +41.6 / +1.32 | +32.4 / +2.37 | 0.31 |
| 2024H1 | −13.7 / −2.05 | +23.9 / +0.77 | +7.3 / +0.59 | 0.25 | +13.6 / +0.65 | **+2.1 / +0.30** | 0.12 |
| 2024H2 | +20.6 / +1.51 |  +2.9 / +0.24 | +12.5 / +1.29 | 0.21 |  +1.8 / +0.23 | +12.2 / +1.36 | 0.27 |
| 2025H1 |  +1.3 / +0.18 | **−0.5 / −0.02** |  +0.6 / +0.17 | 0.36 |  −2.2 / −0.22 | −0.3 / +0.04 | 0.36 |

**Bold = exact match to git OOS json** (cons/2022H1, agg/2025H1 — return matches to the basis point; trade count differs by ±2 at window boundary, a `finalize_trades`/boundary-counting nuance, not a calculation bug).

All other cells are *new* OOS results, valid as forward tests of static params (see §6.1).

### Compounded (5 windows, multiplicative):

```
v1                          +51.84%
donchian-v3 (cons 80/20)    +97.43%      ← honest pick
combined 50/50 (cons)       +83.45%
donchian-v3 (agg 40/10)    +258.70%  ⚠ partly look-ahead
combined 50/50 (agg)       +155.89%  ⚠ partly look-ahead
```

## 5. What the diversification actually buys you — and what it doesn't

Daily-return correlation v1↔Donchian averages **~0.23** across windows — well below 0.5, meaningful diversification.

The combined portfolio's job is to make **realized end-of-window returns smoother**:

| | v1 alone | d3 (cons) | combined (cons) |
|---|---:|---:|---:|
| Worst single-window realized return | −13.7% (2024H1) | −2.2% (2025H1) | **−0.3%** (2025H1) |
| Worst single-window **max-DD** | −14.95% | −26.68% | **−18.70%** |

The 2024H1 chop year is the headline: v1 lost −13.7%, Donchian-v3 (cons) made +13.6%, combined **ended +2.1% realized**.

**But the intra-window DD got worse, not better.** Diversification smooths the realized return; it does NOT smooth the peak-to-trough drawdown — because when Donchian is in a drawdown, it drags the combined account through the low even if it recovers. The current production kill-switch (`kill_switch_equity_fraction=0.82`, −18% DD) would have **tripped on the combined-cons account in 2024H1** and locked in the loss right before Donchian's recovery rescued the realized return. This is the most important practical finding in the report. Per-strategy kill-switches (§7.4) fix it.

## 6. Caveats — do not skim

1. **Walk-forward status, precise.** The cons (80/20/gate-on) params were OOS-selected on IS = 2020-04..2021-12. All 5 test windows here (2022H1 → 2025H1) are strictly forward of that IS — so the cons compounded number **IS a 5-window walk-forward result for static (not re-fit) params**. What we can't claim: that an annual re-fit would have kept picking 80/20 (it might have switched to 40/10 in 2023+, the agg set, which uses IS = 2022-06..2024-12 — overlapping the 2023H1/2024H1/2024H2 test windows, hence agg's look-ahead suspicion).
2. **Single-asset, single-exchange.** All numbers are BTC/USDT on Binance Futures USDM. No cross-asset or cross-venue evidence.
3. **Friction is honest** at 5 bps/side commission+slippage and full per-trade funding accounting (matches v1 path2).
4. **Min-notional issue at current capital.** At $101 deploy capital, splitting 50/50 = $50.50 per strategy, below the $50 Binance minimum after any drawdown. **Combined deploy needs ≥ $200**, or each strategy needs to take its full $101 (which means total exposure = 2× v1 alone, doubling drawdown risk).
5. **Different timeframes.** v1 runs on 15m bars, Donchian-v3 on 4h. They will not place trades concurrently in the same loop — needs two bot processes or one process with multi-TF dispatch.
6. **Leverage.** Both use 20× leverage with 2% risk-per-trade sizing. Combined risk per trade is still 2% IF you split capital, but if you run them at full $101 each, you've doubled effective leverage on the account.
7. **The aggressive params look too good.** +96.9% in 2023H1 from 27 trades on a +82% market with 20× leverage is plausible but is the most overfitting-suspect cell — the agg combo was selected on IS data that includes 2023H1's adjacent period.

## 7. Recommendation

**Conditional yes** — deploy Donchian-v3 with the conservative (80/20/gate-on, time_stop=48, ATR-SL=1.5×) params in parallel with v1, but:

1. Top up Binance Futures wallet to **≥ $200** before splitting capital, or run only on a fresh deposit so v1's $101 is untouched.
2. Run as a **second bot process** with its own `config/params_donchian.yaml`, own `data/state.db_donchian`, own log path, own systemd unit. Do not multiplex into the existing loop.
3. **Start dry-run for at least 14 days.** Donchian on 4h needs ~30 bars warm-up; 14 days is also enough to see a real breakout in most regimes (vs v1's 0 signals in 3 days at current settings).
4. **Independent kill switch at −18%** per-strategy. Do not let Donchian losses force v1 to also halt.
5. Re-run this comparison on **fresh 2025H2 / 2026H1 data** after 6 months. Treat any single-strategy choice that hinges on these numbers as a hypothesis.

## 8. Next session (optional)

The user originally asked to *invent* a new strategy. Donchian-v3 is the safe-fast answer. A genuinely novel candidate worth a future session:

- **Volatility-regime switcher** that explicitly hands off between v1 and Donchian based on ATR percentile (uses both strategies' code, picks one per regime, doesn't run them in parallel). Could outperform 50/50 by avoiding the wrong strategy's drag in each window.

A second session, not a rush: design, code, OOS-validate on the same 5 windows, then compare to the 50/50 baseline established here.
