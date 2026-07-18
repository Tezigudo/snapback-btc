"""
Unit tests for adx() and donchian_channel() in strategy/indicators.py.

Style mirrors tests/test_indicators.py: pytest, _series() helper, approx()
for floats. No external dependencies beyond numpy/pandas.

Key contracts tested:
  - ADX: analytic-limit cases, head NaNs during double-Wilder warm-up,
    causal (no future data leakage).
  - Donchian: trivial hand-checks (max/min per window), 2-tuple return,
    NaN before window full, causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import adx, donchian_channel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlc(n: int, base: float = 100.0, step: float = 1.0):
    """Build simple monotonic-up OHLC Series of length n."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
    close = pd.Series(base + np.arange(n) * step, index=idx, dtype=float)
    high  = close + 1.0
    low   = close - 1.0
    return high, low, close


def _series(*values, freq="15min"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC").tz_localize(None)
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------

class TestAdx:
    def test_returns_same_length_as_input(self):
        high, low, close = _ohlc(50)
        result = adx(high, low, close, period=14)
        assert len(result) == 50

    def test_head_is_nan_during_warmup(self):
        """ADX requires two Wilder passes; first valid bar is around 2×period."""
        high, low, close = _ohlc(60)
        result = adx(high, low, close, period=14)
        # First period-1 values must be NaN (pass-1 warmup alone)
        assert result.iloc[:14].isna().all(), "first 14 bars must be NaN"
        # Some valid (non-NaN) values must exist after full warmup
        assert result.iloc[28:].notna().any(), "values after 2×period must be valid"

    def test_monotonic_uptrend_approaches_100(self):
        """In a clean uptrend +DM always positive, -DM≡0 → DX→100 → ADX→100."""
        n = 80
        high, low, close = _ohlc(n, step=5.0)  # big step so +DM dominates
        result = adx(high, low, close, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        # ADX should converge toward 100 in a steady uptrend
        assert valid.iloc[-1] > 80.0, f"Expected ADX>80 in monotonic uptrend, got {valid.iloc[-1]:.2f}"

    def test_monotonic_downtrend_also_approaches_100(self):
        """Downtrend: -DM dominates, DX still →100, ADX→100 (trend strength is direction-agnostic)."""
        n = 80
        high, low, close = _ohlc(n, step=-5.0)
        result = adx(high, low, close, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 50.0, f"Expected ADX>50 in monotonic downtrend, got {valid.iloc[-1]:.2f}"

    def test_alternating_chop_gives_low_adx(self):
        """Alternating up/down bars → small DM values → ADX stays low."""
        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        # Zigzag close: alternates +2, -2
        closes = [100.0 + (2 if i % 2 == 0 else -2) for i in range(n)]
        close = pd.Series(closes, index=idx, dtype=float)
        high  = close + 0.5
        low   = close - 0.5
        result = adx(high, low, close, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] < 40.0, f"Expected ADX<40 in choppy market, got {valid.iloc[-1]:.2f}"

    def test_causal_slice(self):
        """adx computed on a prefix of data must match same bar index in full-series adx."""
        high, low, close = _ohlc(80)
        full_adx = adx(high, low, close, period=14)

        # Check several bars after warmup
        for j in [35, 50, 65]:
            prefix_val = adx(high.iloc[:j+1], low.iloc[:j+1], close.iloc[:j+1], 14).iloc[j]
            full_val   = full_adx.iloc[j]
            assert np.isfinite(prefix_val) and np.isfinite(full_val), \
                f"bar {j} should be valid post-warmup"
            assert prefix_val == pytest.approx(full_val, abs=1e-6), \
                f"causal mismatch at bar {j}: prefix={prefix_val}, full={full_val}"

    def test_invalid_period_raises(self):
        high, low, close = _ohlc(20)
        with pytest.raises(ValueError, match="period"):
            adx(high, low, close, period=0)

    def test_values_bounded_0_to_100(self):
        high, low, close = _ohlc(80, step=3.0)
        result = adx(high, low, close, period=14)
        valid = result.dropna()
        assert (valid >= 0.0).all() and (valid <= 100.0).all(), \
            "ADX must be in [0, 100]"


# ---------------------------------------------------------------------------
# Donchian channel tests
# ---------------------------------------------------------------------------

class TestDonchianChannel:
    def test_returns_two_series(self):
        high, low, _ = _ohlc(30)
        result = donchian_channel(high, low, period=20)
        assert isinstance(result, tuple) and len(result) == 2
        upper, lower = result
        assert isinstance(upper, pd.Series) and isinstance(lower, pd.Series)

    def test_same_length_as_input(self):
        high, low, _ = _ohlc(30)
        upper, lower = donchian_channel(high, low, period=20)
        assert len(upper) == 30
        assert len(lower) == 30

    def test_head_nan_before_window_full(self):
        """First period-1 bars must be NaN (rolling min_periods=period)."""
        high, low, _ = _ohlc(30, step=1.0)
        upper, lower = donchian_channel(high, low, period=20)
        assert upper.iloc[:19].isna().all(), "upper: first 19 bars must be NaN"
        assert lower.iloc[:19].isna().all(), "lower: first 19 bars must be NaN"
        # After warmup there should be valid values
        assert upper.iloc[19:].notna().all()
        assert lower.iloc[19:].notna().all()

    def test_upper_is_max_of_window(self):
        """Upper channel = rolling max(high, period): hand-check a specific bar."""
        n = 25
        period = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        highs = pd.Series(list(range(n)), index=idx, dtype=float)
        lows  = pd.Series([0.0] * n, index=idx, dtype=float)
        upper, _ = donchian_channel(highs, lows, period=period)
        # At bar 9 (0-indexed), high values are [5,6,7,8,9] → max = 9
        assert upper.iloc[9] == pytest.approx(9.0)
        # At bar 24 (last), high values are [20,21,22,23,24] → max = 24
        assert upper.iloc[24] == pytest.approx(24.0)

    def test_lower_is_min_of_window(self):
        """Lower channel = rolling min(low, period): hand-check a specific bar."""
        n = 25
        period = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        highs = pd.Series([100.0] * n, index=idx, dtype=float)
        lows  = pd.Series(list(range(n)), index=idx, dtype=float)
        _, lower = donchian_channel(highs, lows, period=period)
        # At bar 9 (0-indexed), low values are [5,6,7,8,9] → min = 5
        assert lower.iloc[9] == pytest.approx(5.0)
        # At bar 24 (last), low values are [20,21,22,23,24] → min = 20
        assert lower.iloc[24] == pytest.approx(20.0)

    def test_upper_gte_lower(self):
        """Upper must always be >= lower after warmup."""
        high, low, _ = _ohlc(50, step=1.0)
        upper, lower = donchian_channel(high, low, period=10)
        valid = upper.dropna()
        valid_lower = lower.dropna()
        assert (valid >= valid_lower).all()

    def test_not_shifted_internally(self):
        """Channel includes the current bar — shift is the caller's responsibility."""
        period = 3
        idx = pd.date_range("2024-01-01", periods=6, freq="15min").tz_localize(None)
        highs = pd.Series([10.0, 20.0, 30.0, 5.0, 5.0, 5.0], index=idx)
        lows  = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
        upper, _ = donchian_channel(highs, lows, period=period)
        # At bar 2 (0-indexed), window is bars [0,1,2] → max = 30
        assert upper.iloc[2] == pytest.approx(30.0)
        # At bar 3, window is bars [1,2,3] → max = 30
        assert upper.iloc[3] == pytest.approx(30.0)
        # At bar 4, window is bars [2,3,4] → max = 30
        assert upper.iloc[4] == pytest.approx(30.0)
        # At bar 5, window is bars [3,4,5] → max = 5
        assert upper.iloc[5] == pytest.approx(5.0)

    def test_causal_slice_donchian(self):
        """Donchian computed on prefix must match same bar in full-series result."""
        high, low, _ = _ohlc(60, step=2.0)
        full_upper, full_lower = donchian_channel(high, low, period=20)

        for j in [25, 35, 50]:
            pu, pl = donchian_channel(high.iloc[:j+1], low.iloc[:j+1], period=20)
            assert pu.iloc[j] == pytest.approx(full_upper.iloc[j], abs=1e-9)
            assert pl.iloc[j] == pytest.approx(full_lower.iloc[j], abs=1e-9)

    def test_invalid_period_raises(self):
        high, low, _ = _ohlc(20)
        with pytest.raises(ValueError, match="period"):
            donchian_channel(high, low, period=0)


# ---------------------------------------------------------------------------
# Donchian retest state-machine tests
#
# These tests exercise _advance_retest_state() directly by constructing a
# minimal ADXDualRegimeV1-like namespace (SimpleNamespace) that carries the
# same attributes the method reads.  No full Backtest fixture needed.
# ---------------------------------------------------------------------------

def _make_strategy_ns(
    *,
    retest_window_bars: int = 10,
    retest_proximity_pct: float = 0.005,
    retest_invalidation_pct: float = 0.005,
    pending_long: dict | None = None,
    pending_short: dict | None = None,
    n_bars: int = 20,
    base_close: float = 100.0,
    base_open: float = 99.0,
    base_high: float = 101.0,
    base_low: float = 98.0,
) -> "types.SimpleNamespace":
    """Create a minimal namespace that _advance_retest_state can operate on."""
    import types
    ns = types.SimpleNamespace()
    ns.retest_window_bars = retest_window_bars
    ns.retest_proximity_pct = retest_proximity_pct
    ns.retest_invalidation_pct = retest_invalidation_pct
    # Arrays: all bars have the same default OHLC unless overridden per test
    ns._close_arr = np.full(n_bars, base_close)
    ns._open_arr  = np.full(n_bars, base_open)
    ns._high_arr  = np.full(n_bars, base_high)
    ns._low_arr   = np.full(n_bars, base_low)
    ns._pending_long_break  = pending_long
    ns._pending_short_break = pending_short
    # Bind the method from ADXDualRegimeV1 to this namespace
    from strategy.signals_adx_dual_regime import ADXDualRegimeV1
    import types as _types
    ns._advance_retest_state = _types.MethodType(
        ADXDualRegimeV1._advance_retest_state, ns
    )
    return ns


class TestDonchianRetestStateMachine:
    """
    Tests for the break+retest+rejection state machine in ADXDualRegimeV1.

    Strategy:
    - Bind _advance_retest_state to a minimal SimpleNamespace.
    - Manually set pending state and OHLC arrays to reproduce each scenario.
    - Drive bar-by-bar and assert fire / no-fire / state cleared.
    """

    # -----------------------------------------------------------------
    # Test 1: Happy path
    # Break registered → retest bar with bullish rejection → fires once;
    # raw break bar alone (no pending setup) must NOT fire.
    # -----------------------------------------------------------------
    def test_happy_path_retest_fires_rejection_bar(self):
        BREAK_LEVEL = 100.0

        # --- Part A: raw break bar should NOT fire (no pending setup yet) ---
        ns = _make_strategy_ns(pending_long=None)
        long_fire, _ = ns._advance_retest_state(0)
        assert long_fire is False, "Break bar with no pending setup must not fire"

        # --- Part B: pending setup → retest bar with rejection → fires ---
        # Pending setup: break at BREAK_LEVEL, age=0
        ns2 = _make_strategy_ns(
            pending_long={"break_level": BREAK_LEVEL, "age_bars": 0},
        )
        # Bar i=0: low touches retest zone (just at break level), bullish close above bl
        ns2._low_arr[0]   = BREAK_LEVEL * 1.001  # within 0.5% above break level
        ns2._close_arr[0] = BREAK_LEVEL + 0.5     # close above break level
        ns2._open_arr[0]  = BREAK_LEVEL - 0.5     # open < close → bullish

        long_fire, _ = ns2._advance_retest_state(0)
        assert long_fire is True, "Retest + bullish rejection must fire"
        assert ns2._pending_long_break is None, "Pending must be cleared after fire"

        # --- Part C: raw break without prior pending never fires on that bar ---
        ns3 = _make_strategy_ns(pending_long=None)
        # Even if bar looks like a rejection candle, no pending → no fire
        ns3._low_arr[0]   = BREAK_LEVEL
        ns3._close_arr[0] = BREAK_LEVEL + 1.0
        ns3._open_arr[0]  = BREAK_LEVEL - 1.0
        long_fire3, _ = ns3._advance_retest_state(0)
        assert long_fire3 is False, "No pending setup → no retest fire"

    # -----------------------------------------------------------------
    # Test 2: Invalidation
    # After a break, price drives hard below the break level (low < bl×0.995) →
    # setup is dropped; a later bar that would have been a retest does NOT fire.
    # -----------------------------------------------------------------
    def test_invalidation_deep_pullback_drops_setup(self):
        BREAK_LEVEL = 100.0
        ns = _make_strategy_ns(
            pending_long={"break_level": BREAK_LEVEL, "age_bars": 0},
            n_bars=3,
        )
        # Bar 0: invalidation — low drops 1% below break level (> 0.5% threshold)
        ns._low_arr[0]   = BREAK_LEVEL * (1.0 - 0.01)   # 99.0 — clearly below 99.5
        ns._close_arr[0] = BREAK_LEVEL - 0.5
        ns._open_arr[0]  = BREAK_LEVEL + 0.1

        long_fire, _ = ns._advance_retest_state(0)
        assert long_fire is False, "Invalidation bar must not fire"
        assert ns._pending_long_break is None, "Setup must be cleared on invalidation"

        # Bar 1: even a perfect rejection candle must not fire — setup is dead
        ns._low_arr[1]   = BREAK_LEVEL * 1.001
        ns._close_arr[1] = BREAK_LEVEL + 0.5
        ns._open_arr[1]  = BREAK_LEVEL - 0.5

        long_fire2, _ = ns._advance_retest_state(1)
        assert long_fire2 is False, "No fire after invalidation cleared the setup"

    # -----------------------------------------------------------------
    # Test 3: Expiry
    # Break registered; no retest for retest_window_bars+1 bars → setup dropped.
    # -----------------------------------------------------------------
    def test_expiry_no_retest_within_window(self):
        BREAK_LEVEL = 100.0
        WINDOW = 5
        # Bars are far above the retest zone so no retest triggers
        n = WINDOW + 3
        ns = _make_strategy_ns(
            pending_long={"break_level": BREAK_LEVEL, "age_bars": 0},
            retest_window_bars=WINDOW,
            n_bars=n,
            base_low=BREAK_LEVEL * 1.05,    # low stays 5% above break level — not in zone
            base_close=BREAK_LEVEL * 1.06,
            base_open=BREAK_LEVEL * 1.04,
        )

        # Drive WINDOW bars without retest: each bar increments age by 1.
        # Bars 0..WINDOW-1 (WINDOW total): age goes 0→1→...→WINDOW.
        # None expire because age starts at 0, and check is age>WINDOW.
        for bar in range(WINDOW):
            long_fire, _ = ns._advance_retest_state(bar)
            assert long_fire is False, f"Must not fire at bar {bar} (not in retest zone)"
            assert ns._pending_long_break is not None, \
                f"Setup must still be alive at bar {bar}"

        # After the loop, age_bars == WINDOW (incremented WINDOW times from 0).
        # Bar WINDOW: age == WINDOW, check is age > WINDOW → False → age becomes WINDOW+1.
        # Bar WINDOW+1: age == WINDOW+1 > WINDOW → expired → cleared.
        long_fire_almost, _ = ns._advance_retest_state(WINDOW)
        assert long_fire_almost is False, "Should not fire one bar before expiry fires"
        assert ns._pending_long_break is not None, "Not yet expired at age==WINDOW"

        long_fire_exp, _ = ns._advance_retest_state(WINDOW + 1)
        assert long_fire_exp is False, "Expired setup must not fire"
        assert ns._pending_long_break is None, "Setup must be cleared after expiry"

        # Further bar: no setup, no fire
        long_fire_after, _ = ns._advance_retest_state(WINDOW + 2)
        assert long_fire_after is False, "No fire after expiry"
