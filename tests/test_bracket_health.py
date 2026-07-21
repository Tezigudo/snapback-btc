"""Pure-helper tests for the bracket-health-check + boot-flatten PnL fix.

order_avg_price and bracket_is_intact are side-effect-free (extracted to
bot_internals.py), so they're unit-tested directly without touching the
exchange or state.db — same strategy as test_bot_internals.py.
"""

from __future__ import annotations

from bot_internals import bracket_is_intact, order_avg_price


def _sl(reduce_only: str | bool = "true") -> dict:
    return {"info": {"type": "STOP_MARKET", "reduceOnly": reduce_only, "stopPrice": "63527"}}


def _tp(reduce_only: str | bool = "true") -> dict:
    return {"info": {"type": "TAKE_PROFIT_MARKET", "reduceOnly": reduce_only, "stopPrice": "66429"}}


def _limit_entry() -> dict:
    return {"info": {"type": "LIMIT", "reduceOnly": "false", "price": "64494"}}


class TestOrderAvgPrice:
    def test_from_average(self) -> None:
        assert order_avg_price({"average": 65912.5}) == 65912.5

    def test_from_info_avgprice_when_average_missing(self) -> None:
        assert order_avg_price({"average": None, "info": {"avgPrice": "65900"}}) == 65900.0

    def test_from_price_fallback(self) -> None:
        assert order_avg_price({"average": 0, "info": {}, "price": "64000"}) == 64000.0

    def test_none_when_empty(self) -> None:
        assert order_avg_price(None) is None
        assert order_avg_price({}) is None

    def test_none_when_all_zero(self) -> None:
        assert order_avg_price({"average": 0, "info": {"avgPrice": "0"}, "price": 0}) is None


class TestBracketIsIntact:
    def test_full_bracket_ok(self) -> None:
        assert bracket_is_intact([_sl(), _tp()], place_tp=True) is True

    def test_missing_tp_when_required(self) -> None:
        assert bracket_is_intact([_sl()], place_tp=True) is False

    def test_missing_sl(self) -> None:
        assert bracket_is_intact([_tp()], place_tp=True) is False

    def test_sl_only_ok_for_channel_exit_strategy(self) -> None:
        # donchian-v3 places SL only (place_tp=False) — the channel exit is its TP
        assert bracket_is_intact([_sl()], place_tp=False) is True

    def test_empty_is_never_intact(self) -> None:
        assert bracket_is_intact([], place_tp=True) is False
        assert bracket_is_intact([], place_tp=False) is False

    def test_ignores_non_reduce_only_entry(self) -> None:
        # an unfilled limit ENTRY order is not a bracket leg
        assert bracket_is_intact([_limit_entry(), _sl()], place_tp=False) is True
        assert bracket_is_intact([_limit_entry()], place_tp=False) is False

    def test_close_position_flag_counts_as_reduce_only(self) -> None:
        sl = {"info": {"type": "STOP_MARKET", "closePosition": "true", "stopPrice": "63527"}}
        assert bracket_is_intact([sl], place_tp=False) is True

    def test_boolean_reduce_only(self) -> None:
        # ccxt can surface a python bool rather than the raw "true"/"false" string
        assert bracket_is_intact([_sl(reduce_only=True)], place_tp=False) is True
        assert bracket_is_intact([_sl(reduce_only=False)], place_tp=False) is False
