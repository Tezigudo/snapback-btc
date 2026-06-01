"""Phase 1 feasibility audit for chop-reverter-v1.

Question: at the current capital allocation, is there enough chop-time per year
where BOTH multifactor-v1 AND donchian-v3 are silent, to make a third leg
worthwhile?

Gate: ≥ 60 chop-trade-days/yr → proceed to Phase 2 (signal prototype).
      < 60                    → abandon, pivot to Plan B (ORB).

Run:
    uv run python tools/phase1_chop_feasibility.py

Reads:
    data/historical/BTC_USDT_USDT_15m.parquet   (for v1 signal silence)
    data/historical/BTC_USDT_USDT_4h.parquet    (for ER chop + Donchian breakout)
    data/historical/BTC_USDT_USDT_funding.parquet (for v1's funding gate)

Conservatism notes (deliberate — we'd rather under-count chop-trade-days than
inflate the case for a 3rd leg):
  - Donchian-silence = no FRESH 20-bar breakout that 4h close. Real v3-cons also
    requires an EMA-slope gate that kills breakouts in chop → real Donchian
    fires LESS than this baseline → we over-count Donchian signals → under-
    count chop-trade-days. Conservative.
  - v1 silence uses the live 4-condition form (RSI + EMA(200) + 2× vol +
    funding). Candle/MACD are disabled in production (params.yaml). Identical
    to live decisions.
  - Chop classifier = EMA(120) slope-strength on 4h (slope window=30 bars,
    threshold 0.05% per bar = project default "trending" cutoff from
    regime_summary). Slope is the recommended metric on BTC perp: see
    strategy/regime_classifier.py — "BTC's intraday volatility crushes the ER
    signal." ER kept as a side comparison only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Use the project's own indicators so v1-signal math is byte-identical to live.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy.indicators import ema, rsi, sma  # noqa: E402
from strategy.regime_classifier import efficiency_ratio, ema_slope_strength  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "historical"

# ---- Parameters (mirror live params.yaml where applicable) ----
# Slope-strength is the PRIMARY chop classifier. ER kept as a side comparison.
SLOPE_EMA_PERIOD = 120      # matches regime_summary's slope_30 convention
SLOPE_WINDOW = 30           # 30 4h bars = 5 days slope window
SLOPE_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]   # % per 4h bar — below = chop
PROJECT_SLOPE_THR = 0.05    # regime_summary's default "trending" cutoff
ER_PERIOD_4H = 180          # 30 days × 6 bars/day on 4h (side comparison only)
ER_THRESHOLDS = [0.20, 0.25, 0.30, 0.35]   # sensitivity sweep
DONCH_N = 20                # 4h Donchian breakout lookback
V1_RSI_PERIOD = 14
V1_RSI_LONG = 40.0
V1_RSI_SHORT = 70.0
V1_EMA_PERIOD = 200
V1_VOL_PERIOD = 20
V1_VOL_MULT = 2.0
V1_FUND_THR = 0.0005

# Backtest universe — match the 5 OOS windows: 2022 H1 → 2025 H1
START = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2025-06-30", tz="UTC")

GATE_TRADES_PER_YEAR = 60.0


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df_15m = pd.read_parquet(DATA / "BTC_USDT_USDT_15m.parquet")
    df_4h = pd.read_parquet(DATA / "BTC_USDT_USDT_4h.parquet")
    fund = pd.read_parquet(DATA / "BTC_USDT_USDT_funding.parquet")["funding_rate"]

    # Capitalise to match live `bars_15m["Close"]` convention used by indicators.
    for df in (df_15m, df_4h):
        df.columns = [c.capitalize() for c in df.columns]

    # Clip to the validation window (leave warm-up headroom by going back 60d).
    warm_start = START - pd.Timedelta(days=60)
    df_15m = df_15m.loc[warm_start:END].copy()
    df_4h = df_4h.loc[warm_start:END].copy()
    fund = fund.loc[warm_start:END].copy()
    return df_15m, df_4h, fund


def _v1_signal_by_day(df_15m: pd.DataFrame, fund: pd.Series) -> pd.Series:
    """Per-UTC-day boolean: did the live v1 signal fire at any 15m bar?"""
    close = df_15m["Close"]
    vol = df_15m["Volume"]

    rsi_v = rsi(close, V1_RSI_PERIOD)
    ema_v = ema(close, V1_EMA_PERIOD)
    vol_sma_v = sma(vol, V1_VOL_PERIOD)
    # Funding is recorded every 8h; forward-fill onto 15m bars (matches live
    # which polls the latest funding_rate at decision time).
    fund_15m = fund.reindex(df_15m.index, method="ffill").fillna(0.0)

    vol_ok = vol > V1_VOL_MULT * vol_sma_v
    trend_up = close > ema_v
    long_block = fund_15m > V1_FUND_THR
    short_block = fund_15m < -V1_FUND_THR

    long_sig = (rsi_v < V1_RSI_LONG) & vol_ok & trend_up & ~long_block
    short_sig = (rsi_v > V1_RSI_SHORT) & vol_ok & ~trend_up & ~short_block
    any_sig = long_sig | short_sig

    return any_sig.groupby(any_sig.index.floor("D")).any()


def _donchian_fresh_breakout_by_day(df_4h: pd.DataFrame) -> pd.Series:
    """Per-UTC-day boolean: was there a FRESH 20-bar Donchian breakout?"""
    close = df_4h["Close"]
    # Prior-bars only — shift(1) to exclude current bar from the channel.
    upper = close.shift(1).rolling(DONCH_N).max()
    lower = close.shift(1).rolling(DONCH_N).min()
    crossed_up = (close > upper) & ~(close.shift(1) > upper.shift(1))
    crossed_dn = (close < lower) & ~(close.shift(1) < lower.shift(1))
    fresh = crossed_up | crossed_dn
    return fresh.groupby(fresh.index.floor("D")).any()


def _chop_by_day_slope(df_4h: pd.DataFrame, threshold_pct: float) -> pd.Series:
    """Per-UTC-day boolean from EMA-slope-strength.

    chop = slope_strength < threshold_pct  (the EMA isn't moving fast enough
    to call it a trend). Reading taken at the FIRST 4h bar of the UTC day
    (00:00) to match the design's daily gate refresh and avoid look-ahead.
    """
    slope = ema_slope_strength(
        df_4h["Close"], ema_period=SLOPE_EMA_PERIOD, slope_window=SLOPE_WINDOW
    )
    is_chop_bar = slope < threshold_pct
    return is_chop_bar.groupby(is_chop_bar.index.floor("D")).first()


def _chop_by_day_er(df_4h: pd.DataFrame, threshold: float) -> pd.Series:
    """Per-UTC-day boolean from Kaufman ER (side comparison only)."""
    er = efficiency_ratio(df_4h["Close"], ER_PERIOD_4H)
    is_chop_bar = er < threshold
    return is_chop_bar.groupby(is_chop_bar.index.floor("D")).first()


def _summarise(matrix: pd.DataFrame, label: str) -> dict:
    total = len(matrix)
    years = total / 365.25
    chop = int(matrix["is_chop"].sum())
    v1_sil = int(matrix["v1_silent"].sum())
    dnc_sil = int(matrix["donch_silent"].sum())
    overlap = int(matrix["chop_trade_day"].sum())
    return {
        "label": label,
        "days": total,
        "years": years,
        "chop_pct": chop / total * 100,
        "v1_silent_pct": v1_sil / total * 100,
        "donch_silent_pct": dnc_sil / total * 100,
        "overlap_pct": overlap / total * 100,
        "overlap_per_year": overlap / years,
    }


def main() -> int:
    df_15m, df_4h, fund = _load()

    v1_by_day = _v1_signal_by_day(df_15m, fund)
    donch_by_day = _donchian_fresh_breakout_by_day(df_4h)

    print("=" * 76)
    print("Phase 1 — chop-reverter feasibility audit")
    print(f"Window: {START.date()} → {END.date()}")
    print("=" * 76)

    def _run_sweep(label_fmt, by_day_fn, thresholds):
        rows = []
        for thr in thresholds:
            chop_by_day = by_day_fn(df_4h, thr)
            all_days = (
                chop_by_day.index.union(donch_by_day.index).union(v1_by_day.index)
                .sort_values()
            )
            all_days = all_days[(all_days >= START) & (all_days <= END)]
            m = pd.DataFrame(index=all_days)
            m["is_chop"] = chop_by_day.reindex(all_days).fillna(False).astype(bool)
            m["donch_silent"] = (
                ~donch_by_day.reindex(all_days).fillna(False).astype(bool)
            )
            m["v1_silent"] = (
                ~v1_by_day.reindex(all_days).fillna(False).astype(bool)
            )
            m["chop_trade_day"] = m["is_chop"] & m["v1_silent"] & m["donch_silent"]
            rows.append((thr, m, _summarise(m, label_fmt.format(thr=thr))))
        return rows

    # PRIMARY: slope-strength
    slope_rows = _run_sweep("slope<{thr}", _chop_by_day_slope, SLOPE_THRESHOLDS)
    print(
        f"\n{'slope thr':<12}{'chop%':>8}{'v1-sil%':>10}{'donch-sil%':>14}"
        f"{'overlap%':>12}{'days/yr':>12}"
    )
    print("-" * 76)
    for _thr, _m, s in slope_rows:
        print(
            f"{s['label']:<12}{s['chop_pct']:>8.1f}{s['v1_silent_pct']:>10.1f}"
            f"{s['donch_silent_pct']:>14.1f}{s['overlap_pct']:>12.1f}"
            f"{s['overlap_per_year']:>12.1f}"
        )

    # COMPARISON: ER (known to be broken for BTC perp — for context only)
    er_rows = _run_sweep("ER<{thr}", _chop_by_day_er, ER_THRESHOLDS)
    print(
        f"\n{'ER thr':<12}{'chop%':>8}{'v1-sil%':>10}{'donch-sil%':>14}"
        f"{'overlap%':>12}{'days/yr':>12}"
    )
    print("-" * 76)
    for _thr, _m, s in er_rows:
        print(
            f"{s['label']:<12}{s['chop_pct']:>8.1f}{s['v1_silent_pct']:>10.1f}"
            f"{s['donch_silent_pct']:>14.1f}{s['overlap_pct']:>12.1f}"
            f"{s['overlap_per_year']:>12.1f}"
        )

    # Gate verdict — slope at project convention 0.05% per 4h bar
    primary = next(r for r in slope_rows if r[0] == PROJECT_SLOPE_THR)
    _thr, primary_matrix, primary_summary = primary
    annual = primary_summary["overlap_per_year"]

    print()
    print("=" * 76)
    print(
        f"Gate (project convention, slope<{PROJECT_SLOPE_THR}%): "
        f"{annual:.1f} chop-trade-days/yr"
    )
    print(f"Required:                                       {GATE_TRADES_PER_YEAR:.0f}/yr")
    verdict = "PASS — proceed to Phase 2" if annual >= GATE_TRADES_PER_YEAR else (
        "FAIL — abandon, pivot to Plan B (ORB)"
    )
    print(f"Verdict:                            {verdict}")
    print("=" * 76)

    # Per-year breakdown at the chosen threshold so we can see regime fairness
    primary_matrix["year"] = primary_matrix.index.year
    agg = primary_matrix.groupby("year").agg(
        days=("is_chop", "size"),
        chop=("is_chop", "sum"),
        v1_silent=("v1_silent", "sum"),
        donch_silent=("donch_silent", "sum"),
        chop_trade=("chop_trade_day", "sum"),
    )
    agg["chop_trade_pct"] = (agg["chop_trade"] / agg["days"] * 100).round(1)
    print(f"\nPer-year breakdown (slope<{PROJECT_SLOPE_THR}%):")
    print(agg.to_string())

    # Worst-window check — was there a year with too few opportunities?
    worst_yr = agg["chop_trade"].idxmin()
    print(
        f"\nWorst year: {worst_yr} with {int(agg.loc[worst_yr, 'chop_trade'])} "
        f"chop-trade-days. (Less than 30 → strategy is too thin in some regimes.)"
    )

    return 0 if annual >= GATE_TRADES_PER_YEAR else 1


if __name__ == "__main__":
    sys.exit(main())
