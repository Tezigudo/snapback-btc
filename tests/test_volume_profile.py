"""
Tests for Volume Profile POC indicator and VolumeProfilePOC strategy.

Tests cover:
  - session_poc(): 2-session fixture with known volume concentration.
  - Lookahead safety: bars in session 1 see NaN; bars in session 2 see session 1 POC.
  - Causal test: session_poc(full)[j] == session_poc(truncated_at_j)[j].
  - Strategy smoke: synthetic fixture where _long_signal fires at the right bar.

Style mirrors test_indicators.py and test_adx_donchian.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import session_poc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _15m_index(n: int, start: str = "2024-01-01"):
    """Return a tz-naive DatetimeIndex of n 15-minute bars starting at `start`."""
    return pd.date_range(start, periods=n, freq="15min", tz="UTC").tz_localize(None)


def _make_2session_data(
    poc_target_price: float = 100.0,
    n_bins: int = 50,
):
    """Build a 2-day fixture (192 bars × 15m = 2 × 96-bar days).

    Session 1 (day 0): bars 0–95.
      - Price range for session: 90..110 (20-unit spread).
      - All volume concentrated at poc_target_price ± 0 (single bar with high volume).
      - Each bar has low=90, high=110, close=poc_target_price, volume=1.0.
      - One bar (bar 48, midday) has volume=1_000_000 and close=poc_target_price.
        This makes poc_target_price the dominant bin by a massive margin.

    Session 2 (day 1): bars 96–191.
      - Price is 200.0, volume=1.0 (irrelevant — should not affect session 1 POC).
    """
    n_session = 96
    n_total = 2 * n_session
    idx = _15m_index(n_total, start="2024-01-01")

    high   = np.full(n_total, 110.0)
    low    = np.full(n_total, 90.0)
    close  = np.full(n_total, poc_target_price)
    open_  = np.full(n_total, poc_target_price)
    volume = np.ones(n_total)

    # Spike bar in session 1 — dominant volume at poc_target_price.
    high[48]   = poc_target_price + 0.5
    low[48]    = poc_target_price - 0.5
    close[48]  = poc_target_price
    volume[48] = 1_000_000.0

    # Session 2: different price so it's clearly distinct.
    high[n_session:]   = 210.0
    low[n_session:]    = 190.0
    close[n_session:]  = 200.0
    open_[n_session:]  = 200.0

    return (
        pd.Series(high,   index=idx),
        pd.Series(low,    index=idx),
        pd.Series(close,  index=idx),
        pd.Series(volume, index=idx),
        n_session,
        poc_target_price,
        n_bins,
    )


# ---------------------------------------------------------------------------
# session_poc tests
# ---------------------------------------------------------------------------

class TestSessionPoc:
    def test_session1_bars_are_nan(self):
        """Bars in the first session must be NaN — no prior session exists."""
        high, low, close, volume, n_session, _, n_bins = _make_2session_data()
        result = session_poc(high, low, close, volume, n_bins=n_bins)

        # All bars from session 1 must be NaN.
        assert result.iloc[:n_session].isna().all(), (
            "Session 1 bars must be NaN (no prior session POC)"
        )

    def test_session2_bars_see_session1_poc(self):
        """Bars in session 2 must see session 1's POC, not NaN."""
        high, low, close, volume, n_session, poc_target, n_bins = _make_2session_data()
        result = session_poc(high, low, close, volume, n_bins=n_bins)

        # Session 2 bars should all be non-NaN.
        assert result.iloc[n_session:].notna().all(), (
            "Session 2 bars must have a valid POC from session 1"
        )

    def test_session2_poc_matches_session1_concentration(self):
        """POC value seen by session 2 should be near the high-volume price from session 1."""
        poc_target = 100.0
        n_bins = 50
        high, low, close, volume, n_session, _, _ = _make_2session_data(
            poc_target_price=poc_target, n_bins=n_bins
        )
        result = session_poc(high, low, close, volume, n_bins=n_bins)

        # Bin width = (110 - 90) / 50 = 0.4 — POC midpoint within half a bin of target.
        bin_width = (110.0 - 90.0) / n_bins
        poc_seen = result.iloc[n_session]
        assert abs(poc_seen - poc_target) <= bin_width, (
            f"POC seen by session 2 ({poc_seen:.4f}) is more than one bin_width "
            f"({bin_width:.4f}) away from the volume concentration at {poc_target}"
        )

    def test_all_session2_bars_see_same_poc(self):
        """Within session 2, every bar should return the same prior-session POC (no drift)."""
        high, low, close, volume, n_session, _, n_bins = _make_2session_data()
        result = session_poc(high, low, close, volume, n_bins=n_bins)

        s2 = result.iloc[n_session:]
        assert s2.nunique() == 1, (
            f"All bars in session 2 should share the same POC value; "
            f"got {s2.nunique()} distinct values"
        )

    def test_returns_same_length_as_input(self):
        """Output length must equal input length."""
        high, low, close, volume, _, _, n_bins = _make_2session_data()
        result = session_poc(high, low, close, volume, n_bins=n_bins)
        assert len(result) == len(high)

    def test_invalid_n_bins_raises(self):
        """n_bins <= 0 must raise ValueError."""
        high, low, close, volume, _, _, _ = _make_2session_data()
        with pytest.raises(ValueError, match="n_bins"):
            session_poc(high, low, close, volume, n_bins=0)

    def test_invalid_session_raises(self):
        """Unknown session string must raise NotImplementedError."""
        high, low, close, volume, _, _, _ = _make_2session_data()
        with pytest.raises(NotImplementedError):
            session_poc(high, low, close, volume, session="Asian")


class TestSessionPocLookahead:
    """Causal (lookahead-free) property: session_poc(full)[j] == session_poc(truncated)[j]."""

    def test_causal_session_boundary(self):
        """POC for bar j should not change if we feed more bars after j.

        We test this at the first bar of session 2 (bar 96): feeding bars 0..96
        vs the full 192 bars should give the same POC at position 96.
        """
        high, low, close, volume, n_session, _, n_bins = _make_2session_data()

        j = n_session  # first bar of session 2

        # Full series POC at j
        poc_full = session_poc(high, low, close, volume, n_bins=n_bins).iloc[j]

        # Truncated at j (only bars 0..j inclusive)
        poc_trunc = session_poc(
            high.iloc[:j + 1],
            low.iloc[:j + 1],
            close.iloc[:j + 1],
            volume.iloc[:j + 1],
            n_bins=n_bins,
        ).iloc[j]

        if np.isnan(poc_full) and np.isnan(poc_trunc):
            return  # both NaN → causal (session 2 hadn't started yet — acceptable)

        assert poc_full == pytest.approx(poc_trunc, abs=1e-9), (
            f"Causal violation: full[{j}]={poc_full:.4f} != truncated[{j}]={poc_trunc:.4f}"
        )

    def test_causal_mid_session2(self):
        """POC for a mid-session-2 bar should match the truncated result."""
        high, low, close, volume, n_session, _, n_bins = _make_2session_data()

        j = n_session + 30  # mid-session-2

        poc_full = session_poc(high, low, close, volume, n_bins=n_bins).iloc[j]
        poc_trunc = session_poc(
            high.iloc[:j + 1],
            low.iloc[:j + 1],
            close.iloc[:j + 1],
            volume.iloc[:j + 1],
            n_bins=n_bins,
        ).iloc[j]

        assert poc_full == pytest.approx(poc_trunc, abs=1e-9), (
            f"Causal violation at mid-session-2 bar {j}"
        )


# ---------------------------------------------------------------------------
# Strategy smoke test
# ---------------------------------------------------------------------------

class TestVolumeProfilePOCStrategySignal:
    """Smoke test: inject a known POC and verify _long_signal fires correctly.

    Strategy: compute all indicators with "placeholder" data first (to discover
    the actual POC value the signal bar will see), then set the signal bar's
    OHLC to satisfy all four entry conditions using that actual POC value.
    This avoids the binning/midpoint guessing problem.
    """

    def _build_base_arrays(self):
        """Build 3-session synthetic arrays (before setting the signal bar).

        Returns arrays, index, and constants — signal bar (n_bars-1) is left
        as chop data so indicators can be computed with it as a placeholder.
        """
        n_bars = 288      # 3 × 96-bar sessions
        poc_level = 100.0
        n_session = 96
        n_bins = 50

        idx = _15m_index(n_bars, start="2024-01-01")

        # Session 1 (bars 0–95): heavy volume at poc_level.
        high   = np.full(n_bars, 110.0)
        low    = np.full(n_bars, 90.0)
        close  = np.full(n_bars, poc_level)
        open_  = np.full(n_bars, poc_level)
        volume = np.ones(n_bars)
        volume[48] = 1_000_000.0  # dominant volume spike at poc_level

        # Sessions 2+3 (bars 96–287): alternating chop to keep ADX low.
        for k in range(n_session, n_bars):
            if k % 2 == 0:
                high[k]   = poc_level + 2.0
                low[k]    = poc_level - 1.0
                close[k]  = poc_level + 1.5
                open_[k]  = poc_level - 0.5
            else:
                high[k]   = poc_level + 1.0
                low[k]    = poc_level - 2.0
                close[k]  = poc_level - 1.5
                open_[k]  = poc_level + 0.5

        return high, low, close, open_, volume, idx, n_bars, n_session, n_bins, poc_level

    def _make_strategy_fixture(self):
        """Build indicator arrays with a correctly-placed signal bar.

        Two-pass approach:
          Pass 1: compute POC over base arrays to discover the actual POC at the
                  signal bar position (determined by the preceding session's volume).
          Pass 2: overwrite the signal bar's OHLC so it satisfies all four entry
                  conditions (POC zone, green rejection candle) and recompute.
        """
        from strategy.indicators import adx as _adx, atr as _atr, session_poc as _session_poc

        high, low, close, open_, volume, idx, n_bars, n_session, n_bins, poc_level = (
            self._build_base_arrays()
        )

        signal_bar = n_bars - 1

        # --- Pass 1: discover actual POC at signal bar ---
        high_s   = pd.Series(high,   index=idx)
        low_s    = pd.Series(low,    index=idx)
        close_s  = pd.Series(close,  index=idx)
        volume_s = pd.Series(volume, index=idx)

        poc_pass1 = _session_poc(high_s, low_s, close_s, volume_s, n_bins=n_bins).values
        actual_poc = poc_pass1[signal_bar]

        # If POC is NaN or zero, can't build a valid signal bar.
        if not np.isfinite(actual_poc) or actual_poc <= 0:
            # Return as-is; test will skip via the NaN guard.
            poc_arr = poc_pass1
            adx_arr = _adx(high_s, low_s, close_s, period=14).values
            atr_arr = _atr(high_s, low_s, close_s, period=14).values
            return {
                "poc": poc_arr, "adx": adx_arr, "atr": atr_arr,
                "high": high, "low": low, "close": close, "open": open_,
                "signal_bar": signal_bar, "poc_level": actual_poc,
            }

        # --- Pass 2: place signal bar at actual POC (entry conditions satisfied) ---
        # long condition:
        #   low[i] in [poc*(1-0.005), poc*(1+0.005)]  → set low = actual_poc (inside zone)
        #   close[i] > open[i] (green)                → close = poc + 0.5, open = poc - 0.5
        #   close[i] >= poc[i]                        → poc+0.5 >= poc ✓
        #   high[i] >= close[i]                       → high = poc + 1.5
        high[signal_bar]   = actual_poc + 1.5
        low[signal_bar]    = actual_poc          # exactly at POC → inside 0.5% zone
        close[signal_bar]  = actual_poc + 0.5   # green and above POC
        open_[signal_bar]  = actual_poc - 0.5

        # Recompute indicators with the updated signal bar.
        high_s   = pd.Series(high,   index=idx)
        low_s    = pd.Series(low,    index=idx)
        close_s  = pd.Series(close,  index=idx)
        volume_s = pd.Series(volume, index=idx)
        open_s   = pd.Series(open_,  index=idx)

        poc_arr = _session_poc(high_s, low_s, close_s, volume_s, n_bins=n_bins).values
        adx_arr = _adx(high_s, low_s, close_s, period=14).values
        atr_arr = _atr(high_s, low_s, close_s, period=14).values

        return {
            "poc": poc_arr,
            "adx": adx_arr,
            "atr": atr_arr,
            "high": high,
            "low": low,
            "close": close,
            "open": open_,
            "signal_bar": signal_bar,
            "poc_level": actual_poc,
        }

    def _make_strategy_instance(self, data: dict):
        """Construct a VolumeProfilePOC with arrays injected directly.

        Bypasses __init__ (which needs self.data / self.equity from backtesting.py)
        so we can call _long_signal / _short_signal in unit-test context.
        """
        from strategy.signals_volume_profile import VolumeProfilePOC

        s = VolumeProfilePOC.__new__(VolumeProfilePOC)
        s._poc        = data["poc"]
        s._adx        = data["adx"]
        s._low_arr    = data["low"]
        s._high_arr   = data["high"]
        s._close_arr  = data["close"]
        s._open_arr   = data["open"]
        # Strategy defaults
        s.poc_proximity_pct       = 0.005
        s.adx_chop_threshold      = 20.0
        s.require_rejection_candle = True
        s.allow_shorts             = True
        return s

    def test_long_signal_fires_at_poc_retest(self):
        """_long_signal returns True when all four conditions are met."""
        data = self._make_strategy_fixture()
        i = data["signal_bar"]
        poc_v = data["poc"][i]
        adx_v = data["adx"][i]

        if not np.isfinite(poc_v):
            pytest.skip(f"POC at bar {i} is NaN — not enough session history in fixture")
        if not np.isfinite(adx_v):
            pytest.skip(f"ADX at bar {i} is NaN — not enough warmup bars in fixture")

        s = self._make_strategy_instance(data)
        assert s._long_signal(i) is True, (
            f"_long_signal must return True at bar {i}; "
            f"poc={poc_v:.4f}, adx={adx_v:.4f}, "
            f"low={data['low'][i]:.4f}, close={data['close'][i]:.4f}"
        )

    def test_signal_does_not_fire_in_trend_regime(self):
        """When ADX >= 20, _long_signal must return False."""
        data = self._make_strategy_fixture()
        i = data["signal_bar"]
        poc_v = data["poc"][i]
        adx_v = data["adx"][i]

        if not np.isfinite(poc_v) or not np.isfinite(adx_v):
            pytest.skip("Indicators not yet warmed up at signal bar — skip")

        # Inject ADX > threshold at the signal bar.
        injected_adx = data["adx"].copy()
        injected_adx[i] = 30.0   # > adx_chop_threshold (20.0)
        data_trend = {**data, "adx": injected_adx}

        s = self._make_strategy_instance(data_trend)
        assert s._long_signal(i) is False, (
            "_long_signal must return False when ADX >= adx_chop_threshold"
        )

    def test_signal_does_not_fire_outside_poc_zone(self):
        """When low is far from POC zone, _long_signal must return False."""
        data = self._make_strategy_fixture()
        i = data["signal_bar"]
        poc_v = data["poc"][i]
        adx_v = data["adx"][i]

        if not np.isfinite(poc_v) or not np.isfinite(adx_v):
            pytest.skip("Indicators not yet warmed up — skip")

        # Inject low far above POC (5% away → outside 0.5% zone).
        injected_low = data["low"].copy()
        injected_low[i] = poc_v * 1.05
        data_far = {**data, "low": injected_low}

        s = self._make_strategy_instance(data_far)
        assert s._long_signal(i) is False, (
            "_long_signal must return False when low is far from POC zone"
        )
