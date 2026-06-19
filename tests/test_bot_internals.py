"""Tests for the pure helpers in bot_internals.py.

These functions are extracted from bot.py and have no side effects, so
they're easy to unit-test without mocking the exchange or state.db.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot_internals import (
    SignalDecision,
    evaluate_for_strategy,
    limit_entry_price,
    resolve_strategy_name,
)


class TestResolveStrategyName:
    def test_default_when_missing(self) -> None:
        assert resolve_strategy_name({}) == "multifactor-v1"

    def test_default_when_none(self) -> None:
        assert resolve_strategy_name({"strategy_name": None}) == "multifactor-v1"

    def test_default_when_empty_string(self) -> None:
        # Falsy values fall through to the default — same behavior as the
        # pre-extraction inline expression `... or "multifactor-v1"`.
        assert resolve_strategy_name({"strategy_name": ""}) == "multifactor-v1"

    def test_passthrough_v1(self) -> None:
        assert resolve_strategy_name({"strategy_name": "multifactor-v1"}) == "multifactor-v1"

    def test_passthrough_v3(self) -> None:
        assert resolve_strategy_name({"strategy_name": "v3-all-wider-4"}) == "v3-all-wider-4"


class TestLimitEntryPrice:
    """LONG entries place limit BELOW close (try to fill at a better price);
    SHORT entries place limit ABOVE close. This is the maker-rebate
    direction — verify the sign doesn't flip."""

    def test_long_at_close(self) -> None:
        assert limit_entry_price("long", 100.0, 0.0) == pytest.approx(100.0)

    def test_short_at_close(self) -> None:
        assert limit_entry_price("short", 100.0, 0.0) == pytest.approx(100.0)

    def test_long_below_close_with_offset(self) -> None:
        # 1 bp = 0.01% → 100 × (1 - 0.0001) = 99.99
        assert limit_entry_price("long", 100.0, 1.0) == pytest.approx(99.99)

    def test_short_above_close_with_offset(self) -> None:
        assert limit_entry_price("short", 100.0, 1.0) == pytest.approx(100.01)

    def test_realistic_btc_price(self) -> None:
        # 1 bp offset on $65,000 BTC = $6.50
        long_p = limit_entry_price("long", 65000.0, 1.0)
        short_p = limit_entry_price("short", 65000.0, 1.0)
        assert long_p == pytest.approx(65000.0 - 6.5)
        assert short_p == pytest.approx(65000.0 + 6.5)


class TestSignalDecision:
    def test_sl_tp_long(self) -> None:
        d = SignalDecision(side="long", price=100.0, sl_distance=1.5,
                           tp_distance=3.0, debug={})
        assert d.sl_price == 98.5
        assert d.tp_price == 103.0

    def test_sl_tp_short(self) -> None:
        d = SignalDecision(side="short", price=100.0, sl_distance=1.5,
                           tp_distance=3.0, debug={})
        assert d.sl_price == 101.5
        assert d.tp_price == 97.0

    def test_frozen(self) -> None:
        # Decisions are immutable — accidental mutation should fail loudly.
        d = SignalDecision(side="long", price=100.0, sl_distance=1.0,
                           tp_distance=2.0, debug={})
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            d.price = 200.0  # type: ignore[misc]


def _synthetic_bars(n: int = 300, base: float = 65000.0) -> pd.DataFrame:
    """Build a deterministic OHLCV frame long enough to clear v1 warmup
    (200-EMA + buffer). No signal will actually fire — we only care about
    the dispatch shape, not the entry verdict."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
    rng = np.random.default_rng(seed=42)
    walk = base + np.cumsum(rng.standard_normal(n) * 5)
    return pd.DataFrame({
        "Open": walk - 1, "High": walk + 5, "Low": walk - 5,
        "Close": walk, "Volume": 100 + rng.random(n) * 10,
    }, index=idx)


class TestEvaluateForStrategy:
    """Verify the dispatch shape — both strategy paths must return a
    SignalDecision with the right distances. Whether side is non-None
    depends on the bars; we don't pin that here (varies with random walk)."""

    def _v1_params(self) -> dict:
        return {
            "strategy": {
                "rsi_period": 14, "rsi_long_threshold": 40, "rsi_short_threshold": 70,
                "ema_period": 200, "mf_trend_ema_period": 200,
                "volume_ma_period": 20, "volume_multiple": 2.0,
                "funding_extreme_threshold": 0.0005,
                "require_funding_not_extreme": True,
                "sl_pct": 0.015, "tp_pct": 0.030,
            },
        }

    def test_v1_returns_signal_decision(self) -> None:
        bars = _synthetic_bars()
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._v1_params())
        assert isinstance(d, SignalDecision)
        # v1 derives distances from sl_pct/tp_pct × price.
        assert d.sl_distance == pytest.approx(0.015 * d.price)
        assert d.tp_distance == pytest.approx(0.030 * d.price)

    def test_default_strategy_routes_to_v1(self) -> None:
        """An unknown strategy name should fall through to v1 (the default),
        matching pre-refactor behavior of the `else` branch."""
        bars = _synthetic_bars()
        d = evaluate_for_strategy("some-unknown-strategy", bars, 0.0, self._v1_params())
        # v1 path → distances derived from sl_pct/tp_pct
        assert d.sl_distance == pytest.approx(0.015 * d.price)

    def test_v1_returns_warmup_with_short_history(self) -> None:
        # Fewer than warm-up bars → side=None, debug has 'reason': 'warmup'.
        bars = _synthetic_bars(n=10)
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._v1_params())
        assert d.side is None
        assert d.debug.get("reason") == "warmup"

    def test_decision_price_falls_back_when_dbg_lacks_cur_close(self) -> None:
        # When debug doesn't carry cur_close (warmup), price falls back to the
        # last close. Verify that fallback is what the bar actually shows.
        bars = _synthetic_bars(n=10)
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._v1_params())
        assert d.price == pytest.approx(float(bars["Close"].iloc[-1]))


def _firing_long_bars(n: int = 300, base: float = 65000.0) -> pd.DataFrame:
    """Build a frame whose LAST bar satisfies every v1 long-entry gate:
      RSI(14) < 40, close > EMA(200), volume > 2×SMA(20), funding not extreme.

    Construction: a long slow uptrend (so close > EMA200) that dips on the
    final few bars (so RSI < 40), with a volume spike on the last bar (so the
    volume gate passes). This is the closed-bar state the backtest validated on.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC").tz_localize(None)
    # Slow uptrend keeps close above the 200-EMA.
    close = base + np.arange(n, dtype=float) * 8.0
    # Final 6-bar dip drives RSI down without crossing back under the EMA.
    close[-6:] = close[-7] - np.arange(1, 7) * 25.0
    vol = np.full(n, 100.0)
    vol[-1] = 100.0 * 5.0  # last bar: ~5× the SMA(20) baseline → vol gate passes
    return pd.DataFrame({
        "Open": close - 1, "High": close + 5, "Low": close - 30,
        "Close": close, "Volume": vol,
    }, index=idx)


class TestFormingBarParity:
    """Regression for the forming-bar bug (fix/forming-bar-closed-bar-eval).

    The live bot fetched a frame whose last row is the still-FORMING bar
    (~0.5% of eventual volume), so the volume gate could never fire. The fix
    drops that row (`df = df.iloc[:-1]`) in bot._maybe_enter before any
    evaluator/price/SL-TP logic reads iloc[-1]. These tests pin both halves:
    the closed bar DOES fire, and a forming bar appended on top does NOT —
    but only because the slice is applied. The existing validators only ever
    fed closed bars, so this gap was untested."""

    def _params(self) -> dict:
        return {
            "strategy": {
                "rsi_period": 14, "rsi_long_threshold": 40, "rsi_short_threshold": 70,
                "ema_period": 200, "mf_trend_ema_period": 200,
                "volume_ma_period": 20, "volume_multiple": 2.0,
                "funding_extreme_threshold": 0.0005,
                "require_funding_not_extreme": True,
                "sl_pct": 0.015, "tp_pct": 0.030,
            },
        }

    def test_closed_bar_fires_long(self) -> None:
        # Sanity: the engineered closed-bar frame produces a long signal.
        bars = _firing_long_bars()
        d = evaluate_for_strategy("multifactor-v1", bars, 0.0, self._params())
        assert d.side == "long", f"expected long on closed bar, got {d.debug}"
        assert d.debug["vol_ok"] is True
        assert d.debug["trend_up"] is True

    def test_forming_bar_alone_does_not_fire(self) -> None:
        # If the bot evaluated the RAW frame (forming bar last, near-zero
        # volume), the volume gate fails → no signal. This is the BUG state,
        # asserted here so the fix's necessity is documented.
        bars = _firing_long_bars()
        forming = bars.iloc[[-1]].copy()
        forming.index = forming.index + pd.Timedelta(minutes=15)
        forming["Close"] = float(bars["Close"].iloc[-1]) - 40.0  # partial-bar price
        forming["Volume"] = 0.5  # ~0.5% of a full bar, as Binance returns at open
        raw = pd.concat([bars, forming])
        d = evaluate_for_strategy("multifactor-v1", raw, 0.0, self._params())
        assert d.side is None, "forming bar must NOT satisfy the volume gate"
        assert d.debug["vol_ok"] is False

    def test_slice_restores_closed_bar_signal(self) -> None:
        # The fix: drop the forming bar (mirrors bot._maybe_enter `df.iloc[:-1]`)
        # and the underlying closed bar fires again — restoring live↔backtest
        # parity. Entry price must anchor to the CLOSED bar, not the partial.
        bars = _firing_long_bars()
        forming = bars.iloc[[-1]].copy()
        forming.index = forming.index + pd.Timedelta(minutes=15)
        forming["Close"] = float(bars["Close"].iloc[-1]) - 40.0
        forming["Volume"] = 0.5
        raw = pd.concat([bars, forming])

        deformed = raw.iloc[:-1]  # the surgical fix, applied at the test level
        d = evaluate_for_strategy("multifactor-v1", deformed, 0.0, self._params())
        assert d.side == "long"
        # Price/SL/TP must come from the CLOSED bar, never the forming one.
        assert d.price == pytest.approx(float(bars["Close"].iloc[-1]))
        assert d.price != pytest.approx(float(forming["Close"].iloc[-1]))
