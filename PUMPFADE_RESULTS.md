# Pump-Fade Study — "Short the day's top gainer, TP at prior support"

**Date:** 2026-05-31 · **Verdict: NOT a tradeable edge.** Proof-of-edge research only
(not a deployable bot). Survivorship-safe, point-in-time, adversarially verified.

## The idea (user's)
Scan every Binance USDT-M perp daily. Find the day's top gainer(s) (+40–100%+).
Wait for the intraday rollover ("confirmation of the dropped zone"), **short** it,
and take profit at the **pre-pump support**. Stop above the peak.

## Why this study is trustworthy (the hard part)
- **Survivorship-safe universe.** The live fapi REST `/klines` returns garbage
  placeholder bars for delisted coins — and pumped microcaps are exactly the ones
  that later vanish. We enumerate the full **732-symbol** point-in-time universe via
  the `data.binance.vision` S3 listing and pull daily + 1h + funding for **delisted
  coins too**. `tools/pumpfade_data.py`.
- **Faithful to the intraday idea.** The headline uses **rolling-24h (intraday-high)
  selection** — a coin qualifies the moment its price first crosses +thresh over
  `close[D-1]`, exactly the live "+71.94% top-gainer leaderboard" read, and entry is
  only allowed **from that crossing bar** (point-in-time). This captures the pumps that
  spiked intraday then *faded back below thresh by close* — the best fade setups, which
  a daily-close proxy silently drops. We also ran daily-close selection (entries gated
  to D+1, since `close[D]` isn't known until 23:59) as a cross-check — same verdict.
  (Leaving same-day entries in under close-selection was an 18%-of-trades look-ahead
  that made results look *worse*; removing it was the one real bug fixed.)
- **Honest friction.** Taker fees both sides, illiquidity-scaled slippage, and a
  **heavy asymmetric stop slippage** (a short's stop fires *into* a continuation
  squeeze). **Real funding** accrued every interval. **Delisting force-settle** (you
  can't ride a winner to zero). Loss tail + equity path reported, not just EV.
- **Adversarially verified** by 5 independent agents (look-ahead/sign, friction
  realism, alternative mechanizations, data integrity, steelman-the-bull). All five
  upheld the negative verdict. The only bug found (the same-day look-ahead) hurt the
  strategy, not helped it.

## Headline — faithful intraday rolling-24h selection, dedup, $20M liq floor

| Config | Slice | n | Win% | EV/trade | Median | Worst |
|---|---|---|---|---|---|---|
| **intraday-high, thr 40%** | ALL | 1354 | 57% | **−1.62%** | +4.64% | −208% |
| | IS (2020–24) | 663 | 52% | −2.58% | +2.36% | −74% |
| | OOS (2025–26) | 691 | 62% | **−0.70%** | +6.53% | −208% |
| **intraday-high, thr 60%** | ALL | 697 | 59% | **−1.27%** | +7.42% | −208% |
| | OOS | 446 | 62% | −1.15% | +8.34% | −208% |
| daily-close (cross-check) | ALL | 786 | 55% | **−1.49%** | +5.25% | −110% |
| | OOS | 456 | 59% | −0.12% | +8.04% | −110% |

- **Every configuration and threshold agrees:** EV ≈ −1.3% to −1.6%/trade, OOS hovers
  at breakeven-to-negative, and **every calendar year is negative except a 19-trade
  2020 sample.** No robust positive edge.
- Pre-bugfix close-baseline (look-ahead in, no dedup) was −5.4%; the honest faithful
  number is ≈ −1.5%. Equity path (2%-risk sizing): $1000 → $245, max DD −83%.

## Why it fails (the mechanism)
- **High win rate (55–59%), positive median (+5–8%) — it *feels* like it works.**
  Most pumps really do mean-revert.
- **But expectancy is ~zero-to-negative because of the fat LEFT tail.** ~5–10% of
  pumps *continue* (a coin you shorted at 0.24 hits 1.19 = **5× against you**). One
  such blow-out erases dozens of small reversion wins. Worst trades −110% to −440%.
- **No stop policy escapes it.** Wide/peak stop → rare but catastrophic blow-outs.
  Tight stop → 53–67% whipsaw rate (a freshly-pumped coin is too volatile). No stop →
  unbounded tail (−440%, account wipe). There is **no point in the space that is
  simultaneously positive-EV and survivable.**
- **Friction is not the cause:** price-only EV (zero fees/slip/funding) = **−2.66%**.
  The raw signal has no edge. **Funding** is usually ~neutral (median +0.03%) but has
  a vicious negative tail (a short paid −69% funding into one squeeze).

## What the survivorship-safe data revealed (and why it still fails)
- **Delisted cohort EV +2.81%** vs **surviving −2.71%** (faithful intraday run; the
  daily-close run agrees: +3.07% vs −2.36%). The coins that pumped and *died* were
  profitable shorts — a survivor-only backtest (all you'd get from the live API) would
  have shown only the losers and **missed this entirely**. This validates the whole approach.
- It does **not** rescue the strategy: you can't know ex-ante which coins delist, the
  surviving majority (1087 vs 267) loses, and no calendar year except a 19-trade 2020
  is positive.

## What was tested and rejected (no grid-fishing — pre-registered structural levers)
top-1 gainer only (−5.5%, *worse*) · liquidity floors $100M/$300M (no help) ·
EMA-confirm entry (−3.65%) · 72h hold (worse) · stop-caps 12/20/30% (whipsaw, all
negative) · no-stop (OOS −0.33%, worst −440%) · high-magnitude ≥80% pumps (+5.5% IS →
**−3.6% OOS**, textbook overfit). Best *executable* OOS = −0.12% (dedup) — still not an edge.

## Can it be tuned? (SL + short 1.5-2 day hold) — investigated 2026-05-31
**Tunable for RISK, not for EDGE.** Tested the full hold × SL grid (`tools/pumpfade_tune.py`),
intraday selection + dedup, IS/OOS:
- **Diagnostic (why short-hold is only a partial lever):** stops fire FAST — 61% of
  stop-outs and ~half the big losers (net< −50%) happen **within 48h**, so a 1.5-2d
  time-cap can't dodge the dangerous continuations; and only 52% of TP wins arrive by
  48h, so a short cap also forfeits ~half the biggest wins.
- **Hold sweep (36/48/72/168h):** short hold helps at the *margin* (best ≈ −1.33%) but
  never flips positive.
- **SL sweep @ 48h** (peak / local-swing-high / ATR / fixed-cap / no-stop): **every design
  lands −1.1% to −1.8% EV, all OOS-negative.** Best cell anywhere = **ATR-4× @ 48h, −1.13%**
  (OOS −1.21%, worst −152%, DD −76%).
- **The SL only trades tail for whipsaw:** a cap/local/ATR stop *successfully bounds the
  worst trade to −42% to −152%* (survivable!) but EV stays negative; removing the stop
  cuts EV only marginally while the tail explodes to −214% to −526%. There is no setting
  that is both positive-EV and survivable.
- **Why:** the negative expectancy lives in the *signal* (price-only EV ≈ −2.7%), not the
  exit. Tuning SL/hold redistributes *where* the loss comes from (fat tail vs whipsaw);
  it cannot manufacture an edge that isn't there.

## Verdict
**Do not trade this.** It is the canonical "pennies in front of a steamroller":
attractive win rate and median, but negative/zero expectancy dominated by continuation
blow-outs, with a tail that liquidates a leveraged account on a single trade. Consistent
with the prior `snapback-xsec-momentum-failed` finding (fading/reversal is friction- and
tail-fragile). Don't re-pitch a stop-tuning variant — the tail-vs-whipsaw trade-off is
structural. The *only* untested lever that could bound the tail without whipsaw is a
defined-risk **options** structure (long put / put-spread), but the IV on a freshly-pumped
alt makes the premium prohibitive; prior is strongly negative.

## Files
- `tools/pumpfade_data.py` — survivorship-safe data layer
- `tools/pumpfade_backtest.py` — event detection + simulator + stats (`--study`, `--slice`)
- `tools/pumpfade_phase2.py` — IS/OOS variant driver
- `data/pumpfade/trades_intraday.parquet` — **faithful intraday-high canonical results**
- `data/pumpfade/trades_honest.parquet`, `report_honest.json` — daily-close cross-check
- Run faithful: `Params(select_mode="intraday_high", thresh=0.40, cooldown_days=7)` → `run_study`
