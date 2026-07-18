"""
Unit tests for AdaptiveTrendV1 (strategy/signals_adaptive_trend.py).

Three contracts:
  1. Trend regime: linearly rising price for ~100 H6 bars must fire a long.
  2. Sideways regime: random-walk-around-mean must NOT fire (much).
  3. Funding accounting: a 16h long position over 2 funding events is
     decremented by ~ 2 * (notional * funding_rate).

Style mirrors tests/test_adx_donchian.py — pytest, naive UTC index, deterministic
random seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from backtesting import Backtest

from backtest import funding_cost_for_trades
from strategy.signals_adaptive_trend import (
    AdaptiveTrendV1,
    _resample_h6,
    compute_h6_signal,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_15m_frame(n_bars: int, closes: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a 15m OHLC frame from a close series.  H=close+0.5, L=close-0.5, O=prev_close."""
    idx = pd.date_range(start=start, periods=n_bars, freq="15min")
    opens = np.empty_like(closes)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": closes + 0.5,
            "Low": closes - 0.5,
            "Close": closes,
            "Volume": np.ones(n_bars) * 1.0,  # backtesting.py tolerates either case
        },
        index=idx,
    )


def _trending_frame(n_15m_bars: int = 24 * 4 * 30, base: float = 30_000.0, slope: float = 30.0):
    """30 days at 15m = 2880 bars.  ~30 USD/15m up -> ~+8.6% over 30d, easily clears 2% theta."""
    closes = base + np.arange(n_15m_bars) * slope
    return _make_15m_frame(n_15m_bars, closes)


def _sideways_frame(n_15m_bars: int = 24 * 4 * 30, base: float = 30_000.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Mean-zero increments scaled small so total drift is negligible.
    increments = rng.normal(0.0, 5.0, size=n_15m_bars)
    closes = base + np.cumsum(increments)
    return _make_15m_frame(n_15m_bars, closes)


# ---------------------------------------------------------------------------
# Section 1: compute_h6_signal internal correctness
# ---------------------------------------------------------------------------

class TestH6Signal:
    def test_h6_resample_aligns_on_six_hour_boundaries(self):
        df = _trending_frame()
        h6 = _resample_h6(df)
        # All right-aligned timestamps must land on 00/06/12/18 UTC.
        bad = [ts for ts in h6.index if ts.hour % 6 != 0 or ts.minute != 0]
        assert not bad, f"non-H6 boundaries leaked: {bad[:5]}"

    def test_mom_is_causal_no_lookahead(self):
        # With a strict uptrend, the H6 MOM at any 15m bar must equal the
        # paper's Eq. 2 applied to PRIOR H6 closes — NOT the current one.
        df = _trending_frame()
        aligned = compute_h6_signal(df, momentum_lookback_h6=4, atr_period_h6=14)
        # Pick a mid-window 15m timestamp that is NOT itself on an H6 boundary.
        # At such bars, close_h6 (the most recently *closed* H6 bar) should be
        # strictly less than the live close of the bar itself in an uptrend.
        mid = len(df) // 2
        live_close = df["Close"].iloc[mid]
        h6_close = aligned["close_h6"].iloc[mid]
        # In an uptrend, the most recent CLOSED H6 close is BEHIND live.
        assert h6_close < live_close, (
            f"H6 lookup is not causal: h6_close={h6_close} >= live_close={live_close}"
        )

    def test_mom_positive_in_uptrend(self):
        df = _trending_frame()
        aligned = compute_h6_signal(df, momentum_lookback_h6=4, atr_period_h6=14)
        # Late-window MOM must be positive (price went up 4 H6 bars ago vs now).
        late_mom = aligned["mom_h6"].iloc[-100:].dropna()
        assert (late_mom > 0).all(), "MOM should be positive throughout an uptrend"


# ---------------------------------------------------------------------------
# Section 2: Strategy-level behaviour
# ---------------------------------------------------------------------------

class TestStrategyTrendSignal:
    """Trending fixture -> at least one long trade fires."""

    def test_uptrend_fires_long(self):
        df = _trending_frame(n_15m_bars=24 * 4 * 60)  # 60 days for warmup + signal
        bt = Backtest(
            df,
            AdaptiveTrendV1,
            cash=1_000_000.0,
            commission=0.0005,
            margin=1.0 / 20,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run(theta_entry=0.02, momentum_lookback_h6=4, alpha=2.5)
        trades = getattr(stats, "_trades", None)
        assert trades is not None and len(trades) >= 1, (
            f"Expected ≥1 trade in a sustained uptrend, got {0 if trades is None else len(trades)}"
        )
        # First fill must be a long (positive Size).
        assert trades["Size"].iloc[0] > 0, "First trade in uptrend should be long"


class TestStrategySidewaysSignal:
    """Sideways fixture -> few or no trades."""

    def test_sideways_fires_sparingly(self):
        df = _sideways_frame(n_15m_bars=24 * 4 * 60, seed=42)
        bt = Backtest(
            df,
            AdaptiveTrendV1,
            cash=1_000_000.0,
            commission=0.0005,
            margin=1.0 / 20,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run(theta_entry=0.02, momentum_lookback_h6=4, alpha=2.5)
        n_trades = int(stats.get("# Trades", 0))
        # 60 days = 240 H6 bars.  In a stationary random walk with sigma~5/15m,
        # 24h drift is tight; with theta_entry=2% on the 24h return we should
        # fire very rarely.  Hard ceiling: 10.
        assert n_trades <= 10, (
            f"Sideways fixture fired {n_trades} times — signal is too noisy"
        )


# ---------------------------------------------------------------------------
# Section 3: Funding accounting
# ---------------------------------------------------------------------------

class TestFundingAccounting:
    """Verify funding_cost_for_trades produces ~ 2 * notional * rate over 16h hold."""

    def test_16h_long_two_funding_events(self):
        # Construct a 15m OHLC frame that spans 24h starting at 00:00.
        # Binance funding fires at 00:00, 08:00, 16:00 UTC.
        start = pd.Timestamp("2024-01-01 00:00:00")
        idx = pd.date_range(start=start, periods=24 * 4, freq="15min")
        price = 30_000.0
        data = pd.DataFrame(
            {
                "Open": price,
                "High": price + 1,
                "Low": price - 1,
                "Close": price,
            },
            index=idx,
        )

        # Trade: open at 01:00 (after first 00:00 funding), close at 17:00.
        # Funding events in [01:00, 17:00] = {08:00, 16:00} -> 2 events.
        entry_time = pd.Timestamp("2024-01-01 01:00:00")
        exit_time = pd.Timestamp("2024-01-01 17:00:00")
        notional_size = 10.0  # 10 BTC long (positive Size)
        trades = pd.DataFrame(
            [
                {
                    "Size": notional_size,
                    "EntryTime": entry_time,
                    "ExitTime": exit_time,
                }
            ]
        )

        # Build a funding frame matching Binance's 8h cadence, rate = 0.01% (1bp).
        funding_ts = pd.date_range(start=start, end=start + pd.Timedelta(hours=24), freq="8h")
        funding_rate = 0.0001  # 1 bp per 8h
        funding = pd.DataFrame({"funding_rate": [funding_rate] * len(funding_ts)}, index=funding_ts)

        cost, events = funding_cost_for_trades(trades, data, funding)
        assert events == 2, f"Expected 2 funding events in (01:00, 17:00], got {events}"

        # Cost = sum_i (Size * Close_i * rate_i).  All prices and rates equal:
        expected = notional_size * price * funding_rate * events
        assert cost == pytest.approx(expected, rel=1e-9), (
            f"Funding cost {cost} != expected {expected}"
        )

    def test_short_receives_funding(self):
        """A short (negative Size) over a positive-funding window pays NEGATIVE cost
        (i.e. equity gains).  Mirrors the sign convention in backtest.py."""
        start = pd.Timestamp("2024-01-01 00:00:00")
        idx = pd.date_range(start=start, periods=24 * 4, freq="15min")
        price = 30_000.0
        data = pd.DataFrame(
            {"Open": price, "High": price + 1, "Low": price - 1, "Close": price},
            index=idx,
        )
        trades = pd.DataFrame(
            [
                {
                    "Size": -5.0,  # short
                    "EntryTime": pd.Timestamp("2024-01-01 01:00:00"),
                    "ExitTime": pd.Timestamp("2024-01-01 17:00:00"),
                }
            ]
        )
        funding_ts = pd.date_range(start=start, end=start + pd.Timedelta(hours=24), freq="8h")
        funding = pd.DataFrame({"funding_rate": [0.0001] * len(funding_ts)}, index=funding_ts)

        cost, events = funding_cost_for_trades(trades, data, funding)
        assert events == 2
        # Negative cost = strategy received funding.
        assert cost < 0, f"Short over positive-funding should RECEIVE funding (cost<0), got {cost}"
