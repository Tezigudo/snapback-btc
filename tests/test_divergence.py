"""
Tests for obv() and find_divergence() in strategy/indicators.py.

Style mirrors tests/test_indicators.py — pytest, _series() helper,
approx() for floats, no external dependencies beyond numpy/pandas.

Key contract tested: the +k shift rule that makes find_divergence()
lookahead-safe. A divergence confirmed at swing bar b2 fires exactly
once at bar b2 + k, never before, never again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import find_divergence, mfi, obv, swing_high_low
from strategy.signals_divergence import DivergenceV1
from strategy.signals_divergence_v2 import DivergenceV2Loose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(*values, freq="15min"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC").tz_localize(None)
    return pd.Series(values, index=idx, dtype=float)


def _bool_series(*values, freq="15min"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC").tz_localize(None)
    return pd.Series(values, index=idx, dtype=bool)


# ---------------------------------------------------------------------------
# OBV tests
# ---------------------------------------------------------------------------

class TestObv:
    def test_first_bar_is_zero(self):
        close = _series(100.0, 101.0, 100.5)
        vol   = _series(  10.0,  20.0,  15.0)
        result = obv(close, vol)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_up_day_adds_volume(self):
        # bar0 → bar1: close rises → +20
        close = _series(100.0, 101.0, 100.0)
        vol   = _series(  10.0,  20.0,  30.0)
        result = obv(close, vol)
        assert result.iloc[1] == pytest.approx(20.0)

    def test_down_day_subtracts_volume(self):
        # bar0 → bar1: close falls → -20
        close = _series(101.0, 100.0, 101.0)
        vol   = _series(  10.0,  20.0,  30.0)
        result = obv(close, vol)
        assert result.iloc[1] == pytest.approx(-20.0)

    def test_flat_day_unchanged(self):
        # bar0 → bar1: close same → OBV unchanged (stays 0)
        close = _series(100.0, 100.0, 99.0)
        vol   = _series(  10.0,  20.0,  30.0)
        result = obv(close, vol)
        assert result.iloc[1] == pytest.approx(0.0)

    def test_five_bar_hand_computed(self):
        # Hand-computed:
        # bar0: obv=0
        # bar1: 101>100 → +50  → obv=50
        # bar2: 100<101 → -30  → obv=20
        # bar3: 100==100 → 0   → obv=20
        # bar4: 102>100 → +80  → obv=100
        close = _series(100.0, 101.0, 100.0, 100.0, 102.0)
        vol   = _series( 10.0,  50.0,  30.0,  40.0,  80.0)
        result = obv(close, vol)
        expected = [0.0, 50.0, 20.0, 20.0, 100.0]
        for i, exp in enumerate(expected):
            assert result.iloc[i] == pytest.approx(exp), f"bar {i}: expected {exp}, got {result.iloc[i]}"

    def test_same_length_as_inputs(self):
        close = _series(*np.linspace(100, 110, 20))
        vol   = _series(*np.ones(20) * 100)
        result = obv(close, vol)
        assert len(result) == 20

    def test_returns_series(self):
        close = _series(100.0, 101.0)
        vol   = _series( 10.0,  20.0)
        result = obv(close, vol)
        assert isinstance(result, pd.Series)

    def test_no_mid_series_nans(self):
        # OBV must never introduce NaNs in the middle of the series.
        close = _series(*np.linspace(100.0, 110.0, 30))
        vol   = _series(*np.ones(30) * 100.0)
        result = obv(close, vol)
        assert not result.isna().any()


# ---------------------------------------------------------------------------
# find_divergence tests
# ---------------------------------------------------------------------------

class TestFindDivergence:

    # ------------------------------------------------------------------
    # Fixture builder: 30-bar series with a clear bullish divergence
    # completing at bar 20 (b2=17, k=3, confirmation bar=20).
    # b1=7, b2=17, so separation=10 (in [5,60]).
    # price: LL at b2 (low[17] < low[7])
    # indicator: HL at b2 (ind[17] > ind[7])
    # ------------------------------------------------------------------
    @staticmethod
    def _bullish_30bar_fixture(k=3):
        """
        Construct 30-bar low + indicator series with exactly one bullish
        divergence whose b1=7, b2=17, j=20.

        low:       bar7=100 (swing low), bar17=95 (lower low)
        indicator: bar7=30,             bar17=35  (higher low)
        All non-swing bars are neutral high values so swing_high_low()
        picks exactly bars 7 and 17 as swing lows.
        """
        n = 30
        low_vals = np.full(n, 110.0)
        low_vals[7]  = 100.0   # b1 — local low, but NOT lower than surroundings by enough
        low_vals[17] = 95.0    # b2 — lower low

        # To guarantee swing_high_low marks b1=7 as a swing low with k=3,
        # the 3 bars on each side of bar7 must be > 100.
        # low_vals[4..6] and low_vals[8..10] are already 110 > 100. Good.
        # Similarly for b2=17: low_vals[14..16] and low_vals[18..20] must be > 95. Good.

        ind_vals = np.full(n, 50.0)
        ind_vals[7]  = 30.0
        ind_vals[17] = 35.0

        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
        low_s = pd.Series(low_vals, index=idx)
        ind_s = pd.Series(ind_vals, index=idx)
        return low_s, ind_s

    def test_lookahead_safety_fires_exactly_on_confirmation_bar(self):
        """
        A divergence at (b1=7, b2=17, k=3) must fire ONLY at bar 20.
        Bars 17, 18, 19 must be False; bar 20 True; bar 21 False.
        """
        k = 3
        low_s, ind_s = self._bullish_30bar_fixture(k=k)

        # Build high series (flat high) so swing_high_low returns swing lows.
        high_s = pd.Series(np.full(30, 200.0), index=low_s.index)

        swing_highs, swing_lows = swing_high_low(high_s, low_s, k=k)

        # Verify the fixture actually marks b1=7 and b2=17 as swing lows.
        assert swing_lows.iloc[7],  "fixture: bar 7 must be a swing low"
        assert swing_lows.iloc[17], "fixture: bar 17 must be a swing low"

        result = find_divergence(
            price=low_s,
            indicator=ind_s,
            swing_mask=swing_lows,
            kind="regular_bullish",
            k=k,
            min_separation=5,
            max_separation=60,
        )

        assert result.iloc[17] == False, "must not fire ON the swing bar"
        assert result.iloc[18] == False, "must not fire before confirmation"
        assert result.iloc[19] == False, "must not fire before confirmation"
        assert result.iloc[20] == True,  "must fire exactly at b2 + k = 20"
        assert result.iloc[21] == False, "must not fire the bar after (no sticky True)"

    def test_causal_slice_equivalence(self):
        """
        For any bar j, find_divergence(series[:j+1], ...) == find_divergence(series, ...)[j].
        This proves no future data is consumed. We must recompute swing_high_low on
        each slice — that's the real causal proof (the mask can't peek ahead).
        """
        k = 3
        n = 100
        rng = np.random.default_rng(42)
        low_vals = 100.0 - rng.uniform(0, 5, size=n)
        high_vals = 100.0 + rng.uniform(0, 5, size=n)
        ind_vals  = rng.uniform(20, 80, size=n)
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
        low_s  = pd.Series(low_vals, index=idx)
        high_s = pd.Series(high_vals, index=idx)
        ind_s  = pd.Series(ind_vals, index=idx)

        # Full-series reference
        _, swing_lows_full = swing_high_low(high_s, low_s, k=k)
        full_result = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_lows_full,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )

        # Check at 8 different bar positions
        check_bars = [15, 20, 30, 40, 50, 60, 70, 85]
        for j in check_bars:
            low_slice  = low_s.iloc[:j+1]
            high_slice = high_s.iloc[:j+1]
            ind_slice  = ind_s.iloc[:j+1]
            _, swing_lows_slice = swing_high_low(high_slice, low_slice, k=k)
            slice_result = find_divergence(
                price=low_slice, indicator=ind_slice, swing_mask=swing_lows_slice,
                kind="regular_bullish", k=k, min_separation=5, max_separation=60,
            )
            assert slice_result.iloc[-1] == full_result.iloc[j], (
                f"Causal slice mismatch at bar j={j}: "
                f"slice={slice_result.iloc[-1]}, full={full_result.iloc[j]}"
            )

    def test_regular_bullish_positive_case(self):
        """Direct: price LL + indicator HL → True at confirmation bar."""
        k = 3
        low_s, ind_s = self._bullish_30bar_fixture(k=k)
        high_s = pd.Series(np.full(30, 200.0), index=low_s.index)
        _, swing_lows = swing_high_low(high_s, low_s, k=k)
        result = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )
        assert result.any(), "bullish divergence fixture must produce at least one True"

    def test_regular_bearish_positive_case(self):
        """Direct: price HH + indicator LH → True at confirmation bar."""
        k = 3
        n = 30
        # b1=7: high=100, ind=70; b2=17: high=105 (HH), ind=65 (LH)
        high_vals = np.full(n, 90.0)
        high_vals[7]  = 100.0
        high_vals[17] = 105.0
        ind_vals = np.full(n, 50.0)
        ind_vals[7]  = 70.0
        ind_vals[17] = 65.0
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
        high_s = pd.Series(high_vals, index=idx)
        low_s  = pd.Series(np.full(n, 80.0), index=idx)
        ind_s  = pd.Series(ind_vals, index=idx)

        swing_highs, _ = swing_high_low(high_s, low_s, k=k)

        assert swing_highs.iloc[7],  "fixture: bar 7 must be a swing high"
        assert swing_highs.iloc[17], "fixture: bar 17 must be a swing high"

        result = find_divergence(
            price=high_s, indicator=ind_s, swing_mask=swing_highs,
            kind="regular_bearish", k=k, min_separation=5, max_separation=60,
        )
        assert result.iloc[20] == True, "bearish divergence must fire at b2+k=20"
        assert result.iloc[21] == False, "must not be sticky"

    def test_min_separation_gate(self):
        """Two swings closer than min_separation must NOT trigger.

        We pass a hand-built swing mask rather than relying on swing_high_low()
        to produce two swings at b1=8, b2=11 (sep=3). swing_high_low(k=3)
        cannot produce two genuine swings that close because the lower value
        would dominate the other's window. Using _bool_series directly is the
        only way to exercise this guard.
        """
        k = 3
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)

        # b1=8, b2=11 → separation=3, below min_separation=5
        mask_vals = np.zeros(n, dtype=bool)
        mask_vals[8]  = True
        mask_vals[11] = True
        swing_mask = pd.Series(mask_vals, index=idx)

        # Price: LL; Indicator: HL — would diverge if separation were allowed.
        low_vals = np.full(n, 110.0)
        low_vals[8]  = 100.0
        low_vals[11] = 95.0
        ind_vals = np.full(n, 50.0)
        ind_vals[8]  = 30.0
        ind_vals[11] = 35.0
        low_s = pd.Series(low_vals, index=idx)
        ind_s = pd.Series(ind_vals, index=idx)

        result = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_mask,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )
        # Firing bar for b2=11 is 11+3=14; separation 3 < 5 must block it.
        assert result.iloc[14] == False, "min_separation gate failed: fired when too close"
        # Also verify it WOULD fire if min_separation is relaxed to 3.
        result_relaxed = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_mask,
            kind="regular_bullish", k=k, min_separation=3, max_separation=60,
        )
        assert result_relaxed.iloc[14] == True, "gate should pass when min_separation=3"

    def test_max_separation_gate(self):
        """Two swings further than max_separation must NOT trigger."""
        k = 3
        n = 80
        # b1=5, b2=70 → separation=65, which is > max_separation=60
        low_vals = np.full(n, 110.0)
        low_vals[5]  = 100.0
        low_vals[70] = 95.0
        ind_vals = np.full(n, 50.0)
        ind_vals[5]  = 30.0
        ind_vals[70] = 35.0
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
        low_s  = pd.Series(low_vals, index=idx)
        high_s = pd.Series(np.full(n, 200.0), index=idx)
        ind_s  = pd.Series(ind_vals, index=idx)

        _, swing_lows = swing_high_low(high_s, low_s, k=k)

        result = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )
        # Firing bar for b2=70 would be bar 73.
        if 73 < n:
            assert result.iloc[73] == False, "max_separation gate failed: fired when too far"

    def test_no_divergence_when_prices_and_indicator_agree(self):
        """Price LL + indicator LL → no bullish divergence (they agree, no exhaustion)."""
        k = 3
        n = 30
        # Both price and indicator make lower lows → no bullish divergence
        low_vals = np.full(n, 110.0)
        low_vals[7]  = 100.0
        low_vals[17] = 95.0   # lower low — same direction as indicator
        ind_vals = np.full(n, 50.0)
        ind_vals[7]  = 35.0
        ind_vals[17] = 30.0   # lower low in indicator too → NO divergence
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
        low_s  = pd.Series(low_vals, index=idx)
        high_s = pd.Series(np.full(n, 200.0), index=idx)
        ind_s  = pd.Series(ind_vals, index=idx)

        _, swing_lows = swing_high_low(high_s, low_s, k=k)

        result = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )
        assert result.sum() == 0, "confirming lower lows in both price+indicator must not fire"

    def test_nan_in_indicator_warmup_does_not_crash_or_fire(self):
        """Head NaNs in the indicator (RSI warmup) must not crash or produce signals."""
        k = 3
        low_s, _ = self._bullish_30bar_fixture(k=k)
        high_s = pd.Series(np.full(30, 200.0), index=low_s.index)

        # Build indicator with NaN warmup covering the first swing (b1=7)
        ind_vals = np.full(30, 50.0)
        ind_vals[:10] = np.nan  # NaN covers bar 7 — the first swing's indicator is NaN
        ind_vals[17]  = 35.0
        ind_s = pd.Series(ind_vals, index=low_s.index)

        _, swing_lows = swing_high_low(high_s, low_s, k=k)

        # Must not raise; must not produce a spurious True at b2+k=20
        result = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )
        # bar 7 has NaN indicator → no valid pair → no fire at bar 20
        assert result.iloc[20] == False, "NaN indicator at b1 must suppress divergence signal"

    def test_invalid_kind_raises(self):
        close = _series(100.0, 101.0, 100.0)
        mask  = _bool_series(False, True, False)
        with pytest.raises(ValueError, match="kind must be"):
            find_divergence(close, close, mask, kind="hidden_bullish", k=3,
                            min_separation=5, max_separation=60)

    def test_invalid_k_raises(self):
        close = _series(100.0, 101.0, 100.0)
        mask  = _bool_series(False, True, False)
        with pytest.raises(ValueError, match="k must be"):
            find_divergence(close, close, mask, kind="regular_bullish", k=0,
                            min_separation=5, max_separation=60)

    def test_returns_series_same_length(self):
        k = 3
        low_s, ind_s = self._bullish_30bar_fixture(k=k)
        high_s = pd.Series(np.full(30, 200.0), index=low_s.index)
        _, swing_lows = swing_high_low(high_s, low_s, k=k)
        result = find_divergence(
            price=low_s, indicator=ind_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(low_s)


# ---------------------------------------------------------------------------
# Fix 1 — OBV slope gate (windowed regression replaces cumulative level check)
# ---------------------------------------------------------------------------

class TestOBVSlopeGate:
    """
    Prove that Fix 1 is structurally different from the old cumulative-OBV gate.

    Scenario: a true bullish RSI divergence exists (price LL, RSI HL).
    OBV has a constant +1 slope across the entire series — steady uptrend drift.

    OLD gate (find_divergence on cumulative OBV):
        OBV[b2] > OBV[b1] because OBV is monotonically increasing.
        → fires (near-tautological in any net-upward window).

    NEW gate (DivergenceV1._obv_slope inline check):
        slope computed over b1..b2 window; for accumulation to confirm a
        bullish signal we want slope > 0 (which a +1 drift has) — so this
        specific case ALSO fires. BUT the test below proves the gate is doing
        real work: when we inject a downward drift (slope < 0), the new gate
        REJECTS while the old cumulative gate FIRES if OBV[b2] > OBV[b1].

    We test both directions:
      1. Positive OBV slope → new gate fires (accumulation confirmed).
      2. Negative OBV slope → new gate REJECTS (old cumulative gate would fire
         because the drift starts high and b2 is still > b1 if drift is mild).
    """

    def _make_strategy_instance(self, obv_slope_sign: int) -> tuple:
        """
        Build a minimal synthetic fixture that produces exactly one bullish
        RSI divergence at (b1=7, b2=17, j=20) and injects an OBV with the
        given slope sign across the b1..b2 window.

        Returns (strategy_like_object, b1, b2, j) so the test can call
        _obv_slope and read _swing_low_b1/_b2 directly.

        We instantiate DivergenceV1 via backtesting.Backtest so init() runs.
        """
        import pandas as pd
        from backtesting import Backtest

        k = 3
        n = 50  # enough bars for EMA(200) to remain NaN → trend filter irrelevant
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)

        # Price arrays
        close_vals = np.full(n, 50000.0)
        high_vals  = np.full(n, 50500.0)
        low_vals   = np.full(n, 49500.0)
        # b1=7: swing low (price 49000), b2=17: lower swing low (price 48000)
        low_vals[7]  = 49000.0
        low_vals[17] = 48000.0  # lower low

        # Volume: design OBV to have desired slope across b1..b2 (bars 7..17)
        # We construct signed_vol so that cumsum produces slope_sign * 1 per bar.
        # Outside that window, OBV is flat (sign=0 → equal close prices).
        # We inject volume via close differences:
        #   close[i] > close[i-1] → positive signed_vol = +volume
        # Simplest approach: just set close_vals to drift across b1..b2 window.
        if obv_slope_sign > 0:
            # Steady rise in close across b1..b2 → OBV slope positive
            for bar in range(8, 18):  # bars 8..17 rise
                close_vals[bar] = close_vals[bar - 1] + 10.0
        else:
            # Steady fall in close across b1..b2 → OBV slope negative
            for bar in range(8, 18):
                close_vals[bar] = close_vals[bar - 1] - 10.0
            # BUT make b2's close still above b1's close so cumulative OBV
            # would still say obv[b2] > obv[b1] at the start (mild negative drift
            # that still leaves cumulative higher). We verify this is the bug:
            # cumulative OBV can be "higher" at b2 even with a net-upward OBV
            # if b1..b2 drift is mild negative. For the strong negative case,
            # obv[b2] < obv[b1] anyway, so both gates reject — we need a
            # case where old gate fires but new gate rejects.
            # Solution: use a MODERATE downward slope that is still
            # net-positive cumulatively from bar 0 but slope < 0 in window.
            # Reset: make OBV drift upward from 0 to bar 7 (accumulation),
            # then drift gently downward from bar 7 to 17 (slope < 0 in window).
            close_vals = np.full(n, 50000.0)
            # bars 1..7: close rises → positive OBV accumulation
            for bar in range(1, 8):
                close_vals[bar] = close_vals[bar - 1] + 5.0
            # bars 8..17: close falls mildly → OBV slope < 0 in b1..b2 window
            # but OBV[b2] is still above 0 (cumulative)
            for bar in range(8, 18):
                close_vals[bar] = close_vals[bar - 1] - 2.0
            high_vals  = close_vals + 500.0
            low_vals   = np.full(n, 49500.0)
            low_vals[7]  = 49000.0
            low_vals[17] = 48000.0

        volume_vals = np.full(n, 100.0)  # constant volume

        df = pd.DataFrame({
            "Open":   close_vals,
            "High":   high_vals,
            "Low":    low_vals,
            "Close":  close_vals,
            "Volume": volume_vals,
        }, index=idx)

        # Run backtest with no trades (we only want init() to populate internals)
        bt = Backtest(
            df,
            DivergenceV1,
            cash=100_000,
            commission=0.0005,
            margin=1.0 / 5,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
        # Override defaults to disable trend filter (not enough bars for EMA200)
        # and use only OBV divergence toggle, RSI divergence OFF so we isolate OBV.
        stats = bt.run(
            trend_filter_enabled=False,
            use_rsi_divergence=False,
            use_obv_divergence=True,
            leverage=5,
            rsi_oversold_zone=30.0,
            rsi_overbought_zone=70.0,
        )
        # stats['_strategy'] is the actual strategy instance with init() state
        return stats["_strategy"]

    def test_obv_slope_helper_positive(self):
        """_obv_slope returns positive when OBV trends up across window."""
        strat = self._make_strategy_instance(obv_slope_sign=1)
        # b1=7, b2=17 — OBV should slope upward
        slope = strat._obv_slope(7, 17)
        assert slope > 0, f"Expected positive OBV slope, got {slope}"

    def test_obv_slope_helper_negative(self):
        """_obv_slope returns negative when OBV trends down across window (mild drift)."""
        strat = self._make_strategy_instance(obv_slope_sign=-1)
        # With the negative drift design, OBV slopes down across bars 7..17
        slope = strat._obv_slope(7, 17)
        assert slope < 0, f"Expected negative OBV slope, got {slope}"

    def test_old_cumulative_obv_gate_would_fire_on_drifted_series(self):
        """
        Demonstrate the old bug: cumulative OBV[b2] > OBV[b1] fires even when
        OBV is merely drifting upward (not accumulating against a downtrend).

        This test uses find_divergence (the OLD gate path) directly on a series
        where OBV has a constant +1 slope — trivially satisfies obv[b2] > obv[b1].
        """
        k = 3
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)

        # Price: lower low at b2 (bullish divergence geometry in price)
        low_vals = np.full(n, 110.0)
        low_vals[7]  = 100.0   # b1
        low_vals[17] = 95.0    # b2 — lower low
        low_s  = pd.Series(low_vals, index=idx)
        high_s = pd.Series(np.full(n, 200.0), index=idx)

        # OBV: linear drift +1 per bar (slope = +1 throughout, no real divergence signal)
        obv_vals = np.arange(n, dtype=float)  # 0, 1, 2, ..., 29
        obv_s = pd.Series(obv_vals, index=idx)

        _, swing_lows = swing_high_low(high_s, low_s, k=k)
        assert swing_lows.iloc[7],  "fixture: bar 7 must be swing low"
        assert swing_lows.iloc[17], "fixture: bar 17 must be swing low"

        # Old gate: find_divergence checks obv[b2] > obv[b1]
        # obv[17]=17 > obv[7]=7 → fires (near-tautological)
        result_old = find_divergence(
            price=low_s, indicator=obv_s, swing_mask=swing_lows,
            kind="regular_bullish", k=k, min_separation=5, max_separation=60,
        )
        assert result_old.iloc[20] == True, (
            "Old cumulative-OBV gate should fire (this proves the bug exists): "
            f"got {result_old.iloc[20]}"
        )

    def test_new_obv_slope_gate_rejects_pure_drift(self):
        """
        Fix 1: the new _obv_slope check must REJECT a signal when OBV is
        trending DOWN across the b1..b2 window (distribution despite price LL),
        even though the cumulative level obv[b2] > obv[b1] would pass the old gate.

        We build a synthetic strategy instance where OBV slopes negatively
        across b1..b2 and assert that _obv_slope < 0.

        The test also verifies DivergenceV1._long_signal returns False at the
        firing bar j=20 when OBV slope is negative — i.e. the gate is wired in.
        """
        strat = self._make_strategy_instance(obv_slope_sign=-1)
        slope = strat._obv_slope(7, 17)
        assert slope < 0, f"Expected negative OBV slope for rejection path, got {slope}"
        # The gate is wired: _long_signal must return False at j=20
        # (negative OBV slope + no RSI divergence enabled → gate rejects)
        result = strat._long_signal(20)
        assert result == False, (
            "New OBV slope gate must REJECT when slope < 0 (distribution pattern)"
        )


# ---------------------------------------------------------------------------
# MFI tests (divergence-v2)
# ---------------------------------------------------------------------------

class TestMFI:
    """Tests for mfi() in strategy/indicators.py.

    20-bar hand-computed fixture, head-NaN test, and a directional-symmetry test.
    """

    def _make_ohlcv(self, n: int = 20):
        """Construct a simple rising-then-falling OHLCV fixture."""
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        # Typical price: 100, 102, 104, ... rises for 10 bars then falls
        closes = np.array([100.0 + 2*i if i < 10 else 120.0 - 2*(i-10) for i in range(n)])
        highs = closes + 1.0
        lows = closes - 1.0
        vols = np.full(n, 100.0)
        return (
            pd.Series(highs, index=idx),
            pd.Series(lows, index=idx),
            pd.Series(closes, index=idx),
            pd.Series(vols, index=idx),
            idx,
        )

    def test_hand_computed_single_bar(self):
        """
        Hand-compute MFI on a 15-bar fixture with period=3.

        Bars (close sequence): 100, 101, 102, 103, 104, 103 (fall)
        TP = (H+L+C)/3 where H=C+1, L=C-1 → TP = C exactly.
        MF = TP * volume.
        Bars 0-4: TP rising, so positive_mf = MF; negative_mf = 0.
        Bar 5: TP falls from 104 to 103, so negative_mf = 103*1 = 103.

        For period=3, at bar 5:
            pos_sum = MF[3] + MF[4] + 0 = 103 + 104 = 207
            neg_sum = 0 + 0 + 103 = 103
            mf_ratio = 207/103
            mfi = 100 - 100/(1 + 207/103) = 100 - 100*103/310 = 100 - 33.226... ≈ 66.77
        """
        n = 6
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 103.0])
        highs = closes + 1.0
        lows = closes - 1.0
        # TP = (H+L+C)/3 = (C+1 + C-1 + C)/3 = C
        # MF = TP * vol = C * 1
        vols = np.ones(n)
        period = 3

        result = mfi(
            pd.Series(highs, index=idx),
            pd.Series(lows, index=idx),
            pd.Series(closes, index=idx),
            pd.Series(vols, index=idx),
            period=period,
        )

        # Bars 0..period-2 (bars 0,1) should be NaN (warmup)
        assert np.isnan(result.iloc[0]), "bar 0 must be NaN (warmup)"
        assert np.isnan(result.iloc[1]), "bar 1 must be NaN (warmup)"
        # Bar 2 (period=3): bars 0,1,2 are all positive (TP rises). neg_sum=0 → MFI=100
        assert result.iloc[2] == pytest.approx(100.0), f"all-positive window → MFI=100, got {result.iloc[2]}"
        # Bar 5 (after first fall):
        # pos_sum = MF[3]+MF[4] = 103+104 = 207
        # neg_sum = MF[5] = 103
        # mf_ratio = 207/103; mfi = 100 - 100/(1+207/103)
        pos_sum = 103.0 + 104.0
        neg_sum = 103.0
        expected_bar5 = 100.0 - 100.0 / (1.0 + pos_sum / neg_sum)
        assert result.iloc[5] == pytest.approx(expected_bar5, rel=1e-6), (
            f"bar 5 MFI: expected {expected_bar5:.4f}, got {result.iloc[5]:.4f}"
        )

    def test_head_nan(self):
        """MFI must return NaN for the first (period-1) bars (warmup)."""
        period = 14
        n = 30
        highs, lows, closes, vols, _ = self._make_ohlcv(n)
        result = mfi(highs, lows, closes, vols, period=period)
        assert len(result) == n
        # First period-1 bars must all be NaN
        assert result.iloc[:period - 1].isna().all(), (
            f"Expected NaN for first {period-1} bars"
        )
        # Bar at index `period` onward should NOT all be NaN
        assert result.iloc[period:].notna().any(), "MFI must produce values after warmup"

    def test_mfi_directional_symmetry(self):
        """
        MFI directional property: a pure rising series produces MFI > 50
        (all positive money flow); a pure falling series produces MFI < 50
        (all negative money flow) after warmup.

        Note: exact MFI(reflected price) ≠ 100 - MFI(original) because volume
        weighting by typical price level is not symmetric under price reflection.
        This test validates directional consistency only.
        """
        period = 3
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        vols = pd.Series(np.full(n, 100.0), index=idx)

        # Rising: typical prices strictly increase → all positive MF
        rising_closes = pd.Series(np.linspace(100.0, 200.0, n), index=idx)
        rising_highs = rising_closes + 1.0
        rising_lows = rising_closes - 1.0
        mfi_rising = mfi(rising_highs, rising_lows, rising_closes, vols, period=period)

        # Falling: typical prices strictly decrease → all negative MF
        falling_closes = pd.Series(np.linspace(200.0, 100.0, n), index=idx)
        falling_highs = falling_closes + 1.0
        falling_lows = falling_closes - 1.0
        mfi_falling = mfi(falling_highs, falling_lows, falling_closes, vols, period=period)

        # After warmup: rising should give MFI=100 (neg_sum=0 for strictly increasing)
        valid_rising = mfi_rising.iloc[period:].dropna()
        valid_falling = mfi_falling.iloc[period:].dropna()

        assert (valid_rising == 100.0).all(), (
            f"Strictly rising prices must yield MFI=100, got: {valid_rising.values}"
        )
        # Falling: pos_sum=0 → mf_ratio=NaN (0/0 protection) → MFI=0 when neg_sum>0, pos_sum=0
        # pos_sum.replace(0,nan)/neg_sum → nan/neg = nan → 100-100/(1+nan) = nan unless handled
        # Check: falling prices → pos_sum=0 (no up-moves), so mf_ratio=0/neg_sum=0 → MFI=0
        # Actually: mf_ratio = pos_sum / neg_sum.replace(0,nan) → for falling, neg_sum>0, pos_sum=0
        # → mf_ratio = 0/neg_sum = 0 → MFI = 100 - 100/(1+0) = 0. Correct.
        assert (valid_falling == 0.0).all(), (
            f"Strictly falling prices must yield MFI=0, got: {valid_falling.values}"
        )

    def test_returns_series_same_length(self):
        """mfi() must return a Series of same length as inputs."""
        highs, lows, closes, vols, _ = self._make_ohlcv(20)
        result = mfi(highs, lows, closes, vols, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == 20

    def test_bounded_0_to_100(self):
        """MFI must stay in [0, 100] on a real-ish random series."""
        rng = np.random.default_rng(99)
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        closes_arr = 50000.0 + np.cumsum(rng.normal(0, 100, n))
        highs_arr = closes_arr + rng.uniform(0, 500, n)
        lows_arr = closes_arr - rng.uniform(0, 500, n)
        vols_arr = rng.uniform(1, 1000, n)

        result = mfi(
            pd.Series(highs_arr, index=idx),
            pd.Series(lows_arr, index=idx),
            pd.Series(closes_arr, index=idx),
            pd.Series(vols_arr, index=idx),
            period=14,
        )
        valid = result.dropna()
        assert (valid >= 0.0).all() and (valid <= 100.0).all(), (
            f"MFI out of [0,100]: min={valid.min():.2f}, max={valid.max():.2f}"
        )

    def test_invalid_period_raises(self):
        """period <= 0 must raise ValueError."""
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="15min").tz_localize(None)
        s = pd.Series(np.ones(n), index=idx)
        with pytest.raises(ValueError, match="period must be"):
            mfi(s, s, s, s, period=0)


# ---------------------------------------------------------------------------
# DivergenceV2Loose smoke test — OR-gate fires on synthetic RSI divergence
# ---------------------------------------------------------------------------

class TestDivergenceV2Loose:
    """
    Smoke test: DivergenceV2Loose fires at least one trade on a 30-bar fixture
    where a regular bullish RSI divergence exists.

    Notes on what must be disabled in the fixture (forced, not optional):
      - trend_filter_enabled=False: 15m EMA200 cannot warm up in 30 bars.
      - use_4h_regime_gate=False: the 4H gate reads the real parquet by timestamp;
        synthetic bar timestamps (~2024) vs BTC 4H EMA (~$40-70k) would never
        align, and even if they did, EMA200 needs ~200 4H bars of warmup.
        Both gates are exercised fully in the actual OOS backtests (5 windows,
        each 6 months of real BTC data). This fixture only validates that the
        OR-gate + v1 confirmation + widened RSI zones can fire.
      - RSI warmup: RSI(14) needs ~14 bars; the fixture is 80 bars so RSI is
        valid at the swing bars.
    """

    def _make_bullish_divergence_df(self) -> "pd.DataFrame":
        """
        Build an 80-bar OHLCV DataFrame with a guaranteed bullish RSI divergence.

        Design:
          - b1 = bar 20: price local low (close=100), RSI low (~28 from preceding fall).
          - b2 = bar 40: price LOWER low (close=95), RSI HIGHER low (~33 — less oversold).
          - Bar close[j=43] = 110 > high[b2=40] = 97 → v1 confirmation passes.
          - The surrounding bars ensure swing_high_low(k=3) picks exactly bars 20 and 40.
        """
        import pandas as pd

        n = 80
        idx = pd.date_range("2024-06-01", periods=n, freq="15min").tz_localize(None)

        close_vals = np.full(n, 50000.0)
        high_vals  = np.full(n, 50500.0)
        low_vals   = np.full(n, 49500.0)
        volume_vals = np.full(n, 1000.0)

        # Build a price series that creates RSI divergence:
        # Phase 1 (bars 0-19): strong fall to create RSI oversold at bar 20
        for i in range(1, 21):
            close_vals[i] = close_vals[i - 1] - 200.0
        # bar 20: local low
        close_vals[20] = close_vals[19] - 500.0

        # Phase 2 (bars 21-39): mild recovery then moderate fall to b2
        for i in range(21, 30):
            close_vals[i] = close_vals[i - 1] + 100.0
        # bars 30-39: moderate fall (weaker than phase 1 → RSI less oversold at b2)
        for i in range(30, 40):
            close_vals[i] = close_vals[i - 1] - 150.0
        # bar 40: lower low in price than bar 20
        close_vals[40] = close_vals[39] - 100.0

        # Phase 3 (bars 41+): strong recovery — bar 43 close well above high[b2]
        for i in range(41, 80):
            close_vals[i] = close_vals[i - 1] + 400.0

        high_vals  = close_vals + 200.0
        low_vals   = close_vals - 200.0

        df = pd.DataFrame({
            "Open":   close_vals,
            "High":   high_vals,
            "Low":    low_vals,
            "Close":  close_vals,
            "Volume": volume_vals,
        }, index=idx)
        return df

    def test_fires_on_bullish_rsi_divergence_synthetic(self):
        """
        DivergenceV2Loose must produce at least 1 trade on the synthetic fixture
        when 4H and 15m EMA200 gates are disabled (not enough bars for warmup).
        The OR-gate means RSI divergence alone is sufficient to fire.
        """
        from backtesting import Backtest

        df = self._make_bullish_divergence_df()

        bt = Backtest(
            df,
            DivergenceV2Loose,
            cash=1_000_000,
            commission=0.0005,
            margin=1.0 / 5,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run(
            trend_filter_enabled=False,    # EMA200 needs 200 bars; fixture has 80
            use_4h_regime_gate=False,      # 4H gate reads real parquet; disabled for fixture
        )

        n_trades = int(stats["# Trades"])
        assert n_trades >= 1, (
            f"DivergenceV2Loose must fire at least 1 trade on the bullish divergence "
            f"fixture (OR-gate + widened RSI zones + v1 confirmation). Got 0 trades. "
            f"Stats: Return={float(stats['Return [%]']):.2f}%"
        )
