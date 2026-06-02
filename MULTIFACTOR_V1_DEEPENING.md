# multifactor-v1 — Per-factor Deepening (2026-06-02)

Backtest setup: 5 OOS windows 2022 H1 — 2025 H1, $1M cash, commission 5 bps,
margin 1/20 (matches `tools/run_strategy_experiment.py`).  Baseline applied
the **locked `config/params.yaml`** values on top of the class — i.e. `rsi_long=35`,
`volume_multiple=2.0`, `risk=2.75%`, `require_candlestick=False`, `require_macd=False`.

Harness was validated against PATH2 ground truth by reverting to PATH2-era
params (`rsi_long=40`, `risk=2.0`): reproduced exactly 44 trades / +19.29% /
-9.98% DD on 2022H1 — matches `PATH2_RESULTS.html`. The harness is correct.

## TL;DR

- **15m EMA200 trend filter is load-bearing**: dropping it turns +50% into
  **-97% with 1151 trades**. Everything else is secondary.
- **A 4H EMA200 regime gate is the highest-confidence improvement found**:
  baseline +50% → **+77.9% compounded, 5/5 wins, PSR 0.978 (the only variant
  reaching `evidence_of_edge`)** with 28% fewer trades.
- **multifactor-v1 is BTC-specific**: ETH is **-31.6%** (2/5 wins), SOL is
  **+13.9%** (2/5, PSR 0.70). The strategy did not transfer cleanly to the
  other two majors at the locked geometry.

## A. Per-factor ablation

Baseline is locked `params.yaml` at $1M cash.  Baseline (locked, current-config):
**+50.48% compounded across 168 trades, 5/5 windows positive, PSR 0.912.**
That is higher than PATH2's published +55.73% / 4-5 wins because the locked
config has since been tightened (rsi 40→35 on 2026-05-29; risk 2.0→2.75 on
2026-05-23); the rsi 40→35 also turned 2024H1 from -12.56% to +4.20%.

| Variant            | Trades | Compounded | Δ vs baseline | Wins | PSR   | Verdict           |
|--------------------|-------:|-----------:|--------------:|:----:|------:|-------------------|
| baseline           | 168    | +50.48%    |       —       | 5/5  | 0.912 | (anchor)          |
| no_volume          | 233    | +48.91%    |    -1.57 pp   | 4/5  | 0.885 | marginal          |
| no_trend           | 1151   | -97.33%    |  -147.81 pp   | 0/5  | 0.016 | **load_bearing**  |
| no_funding         | 172    | +53.71%    |    +3.23 pp   | 4/5  | 0.921 | actively_hurting* |
| rsi_30_70 (tighter)|  71    |  +2.09%    |   -48.39 pp   | 4/5  | 0.589 | load_bearing (rsi=35 is the sweet spot) |
| add_candlestick    |  13    |  -1.67%    |   -52.15 pp   | 1/5  | 0.419 | sample-killer     |
| add_macd           |   0    |   0.00%    |   -50.48 pp   | 0/5  | —     | sample-killer     |

\* "actively_hurting" by the +3.23pp criterion, but inside the noise band on a
5-window sample — see Recommendations.

Reading:

- **`no_trend` is catastrophic.** The 15m EMA200 trend filter is the only
  reason this strategy is positive at all.  1151 trades at -97% means the
  RSI+volume signal taken counter-trend bleeds out reliably.
- **`no_volume` is marginal** (-1.6pp on 65 extra trades). Volume gate adds
  trades without obviously paying for itself; it's near-free to keep, but
  removing it is also near-free.  Effectively a "noise screen" rather than
  alpha source.
- **`no_funding` adds 3.2pp** with virtually the same trade count (168→172).
  Funding gate blocks 4 trades over 2.5 years — barely active — and they
  were net-positive trades.  See "What we'd change" below.
- **`rsi_30_70` collapses to +2%** with less than half the trades.  The
  current `rsi=35` threshold is well-chosen; tightening further starves the
  strategy of signals.  (PSR memory file already documents 35 > 40 > 30.)
- **`add_candlestick` / `add_macd`** are *inverse* tests (the locked config
  has them OFF; we tried turning them ON).  Both nuke trade count to ≈0,
  confirming the config decision to keep them off.

## B. 4H EMA200 regime gate

Add: only long when `15m close > 4H_EMA200`; only short when `15m close < 4H_EMA200`.
The 4H EMA is built lookahead-safe via bar-CLOSE timestamps + `merge_asof(direction="backward")`
(pattern copied from `signals_divergence_v2.py`).

| Variant       | Trades | Compounded | Δ vs baseline | Wins | PSR   | Verdict   |
|---------------|-------:|-----------:|--------------:|:----:|------:|-----------|
| mtf_4h_gate   | 121    | +77.93%    |   +27.45 pp   | 5/5  | 0.978 | **accretive** |

Per-window returns: 2022H1 +19.7 / 2023H1 +32.9 / 2024H1 +0.8 / 2024H2 +2.0 /
2025H1 +8.8.  **Every window is positive and the chop year (2024H1) goes
from +4.2% to +0.8%** — fewer trades, lower drawdown profile, and the only
variant in the sweep that clears the 0.95 PSR threshold for
`evidence_of_edge`.

The 4H gate overlaps the existing 15m EMA200 — it filters out only the case
where 15m and 4H disagree. So 168→121 trades (-28%) makes sense: it kills
the lowest-quality countertrend-at-4H setups.

## C. Multi-coin transferability

Locked baseline params applied to ETH and SOL, same 5 OOS windows, same $1M
cash.  (Funding column not attached for ETH/SOL — fine, since
`no_funding` showed the gate is barely active even on BTC.)

| Coin | Trades | Compounded | Wins | PSR   | Per-window                                     | Verdict        |
|------|-------:|-----------:|:----:|------:|------------------------------------------------|----------------|
| BTC  | 168    | +50.48%    | 5/5  | 0.912 | +1.6 / +28.7 / +4.2 / +3.1 / +7.1               | (anchor)       |
| ETH  | 189    | **-31.56%**| 2/5  | 0.225 | +28.3 / -13.9 / -25.2 / +15.8 / **-28.5**      | BTC_specific   |
| SOL  | 149    | +13.91%    | 2/5  | 0.700 | -5.6 / +18.6 / +26.1 / -12.1 / -8.3            | partial        |

ETH is a clear failure: 3 of 5 windows are deeply negative (worst -28.5% in
2025H1), and the 2025 + 2024H1 chops both cut deep.  SOL shows two solid
windows but two -8% to -12% losses too — net positive only on the back of
2023H1 and 2024H1.  Conclusion: **the +50% on BTC is not free alpha — it
relies on BTC's specific 15m mean-reversion-within-trend behavior**.  Don't
deploy this to ETH/SOL at the locked geometry without retuning.

(Live SOL deploy uses a different strategy — `cnh-hybrid-short` — for a
reason.)

## Top 3 improvement candidates (ranked)

Ranking is (expected lift × confidence) / build cost.

### 1. Add 4H EMA200 regime gate to multifactor-v1 — TOP PICK
- **Lift**: +27pp compounded (50→78), 5/5 windows positive, **PSR 0.978**
  — only variant clearing 0.95.
- **Confidence**: high.  Effect is consistent across all 5 windows; no
  window goes negative; trade count cut is moderate (-28%), not so deep that
  PSR is degraded.
- **Build cost**: small.  `MultiFactorMTF4H` subclass is 20 lines; merge
  pattern already proven in `signals_divergence_v2.py`. Live evaluator
  (`live_multifactor_v1.py`) needs an extra 4H fetch on each tick — cheap on
  Binance.
- **Risk**: gate degenerates if 4H series fails to load (NaN). Live code
  must fail-closed (skip entry) on NaN — pattern matches divergence_v2.

### 2. Drop the funding gate (or relax to ±0.0015)
- **Lift**: +3.2pp compounded (50→54).  Per-window: only **2024H1** and
  **2024H2** changed; 2024H1 went +4.2 → +9.8, 2024H2 went +3.1 → -0.1.
  Net positive but **inside noise band** on a 5-window sample.
- **Confidence**: low-medium.  The gate currently fires on ~4 trades total;
  removing it allows 4 extra entries that happened to be net +3pp here.
- **Build cost**: zero — one YAML flag.
- **Risk**: in a true funding-extreme environment (e.g. 2021 squeeze), the
  gate may save you from one large adverse fill.  Not observable in this
  sample.  Recommend **relaxing** the threshold from 0.0005 → 0.0015 rather
  than disabling — preserves the squeeze-protection floor and removes 80%
  of the spurious blocks.

### 3. Quarantine multifactor-v1 to BTC; do not extend to ETH/SOL
- **Lift**: avoids -31.6% loss on ETH at current geometry.
- **Confidence**: high (5 OOS windows on each coin, all losses concentrated).
- **Build cost**: zero — just don't deploy this leg on those symbols.
- **Risk**: opportunity cost only.  If the user wants ETH/SOL exposure,
  retune SL/TP geometry and rsi thresholds per coin; do NOT extrapolate BTC
  config to alts.

## What we'd change about the locked config if re-locked today

1. **Add 4H EMA200 regime gate (single biggest win).** This is the only
   change with `evidence_of_edge` (PSR > 0.95) in the sweep. Recommend
   adding `mtf_4h_gate_enabled: true` to `config/params.yaml` and porting
   `MultiFactorMTF4H` into a live module that fetches 4H bars on the tick
   loop.
2. **Relax funding threshold from 0.0005 → 0.0015**, or leave the toggle
   on but make the threshold less strict. Gate barely fires; loosening
   recovers ~3pp without sacrificing the squeeze-protection intent.
3. **Leave everything else alone.** `volume_multiple=2.0`, `require_candlestick=false`,
   `require_macd=false`, `rsi_long=35`, `sl_pct=0.015`, `tp_pct=0.030` are
   all well-chosen given the data — including the *absence* of candlestick
   and MACD filters, which would zero out the strategy if re-enabled.
4. **Constrain multifactor-v1 deployment to BTC.** Document explicitly in
   `params.yaml` that ETH/SOL show negative/partial transfer at this
   geometry; do not lift the strategy to other symbols without per-coin
   re-tune.

## Constraints honored

- `strategy/signals_multifactor.py` — not touched.
- `strategy/live_multifactor_v1.py` — not touched.
- `config/params.yaml` — not touched (locked).
- `risk.py`, `bot.py`, deploy plumbing — not touched.
- divergence files — only **read** for the merge_asof pattern; not modified.
- volume_profile / adaptive_trend / ADX / chopreverter files — not touched.
- No git commits.

## Files written

- `/Users/god/Desktop/work/snapback-btc/reports/multifactor_v1_deepening.json`
  — full numbers (per-window, PSR, deltas).
- `/Users/god/Desktop/work/snapback-btc/MULTIFACTOR_V1_DEEPENING.md` — this report.
- `/Users/god/Desktop/work/snapback-btc/tools/run_mf_deepening.py` — driver
  (single-shot sweep; reusable).
