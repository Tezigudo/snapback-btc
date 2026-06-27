"""Tests for orphan bracket cleanup (fix/orphan-bracket-cleanup).

Covers:
1. cancel_open_orders COID-prefix filtering — only matching orders cancelled
2. _detect_bracket_exit sweeps surviving bracket on open->flat transition
3. _detect_bracket_exit: cancel is called BEFORE PnL recording
4. _place_live_entry pre-clears stale orders (market + limit paths)
5. boot() flat+live -> orphan sweep; flat+dry_run -> no sweep
6. Cancel failure is non-fatal in both _detect_bracket_exit and _place_live_entry
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bot_internals import SignalDecision
from exchange.binance_client import BinanceClient, Position


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_PARAMS: dict = {
    "symbol": "BTC/USDT:USDT",
    "hedge": {"enabled": False, "client_order_id_prefix": "snap-v1-"},
    "execution": {
        "poll_interval_s": "5",
        "order_type": "market",
        "limit_offset_bps": "0",
        "limit_timeout_s": "20",
    },
    "timeframes": {"entry": "15m"},
    "sizing": {"leverage": 5, "risk_per_trade_pct": 1.0},
    "deploy": {"kill_switch_equity_fraction": 0.5},
    "strategy": {},
}


def _mock_client(coid_prefix: str = "snap-v1-") -> MagicMock:
    # Don't use spec=BinanceClient — `ex` is set in __init__, not the class body,
    # so spec would hide it.  Plain MagicMock gives us free attribute creation.
    mc = MagicMock()
    mc.ex.parse_timeframe.return_value = 900  # 15m = 900 s
    mc.coid_prefix = coid_prefix
    mc.env = "testnet"
    mc.cancel_open_orders.return_value = 0
    return mc


def _make_bot(dry_run: bool = False, order_type: str = "market"):
    from bot import Bot
    params = {
        **_MINIMAL_PARAMS,
        "execution": {**_MINIMAL_PARAMS["execution"], "order_type": order_type},
    }
    mc = _mock_client()
    with patch("bot.BinanceClient.from_env", return_value=mc):
        bot = Bot(params=params, dry_run=dry_run)
    return bot, mc


def _flat() -> Position:
    return Position(symbol="BTC/USDT:USDT", side="flat", qty=0.0,
                    entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)


def _open(side: str = "long") -> Position:
    return Position(symbol="BTC/USDT:USDT", side=side, qty=0.01,
                    entry_price=65000.0, unrealized_pnl=10.0, margin_used=100.0)


def _boot_patches():
    """Context manager that patches all side-effects in boot()."""
    return patch.multiple(
        "bot",
        check_symbol=MagicMock(),
        check_leverage=MagicMock(),
        send_alert=MagicMock(),
    )


# ---------------------------------------------------------------------------
# 1. cancel_open_orders COID-prefix filtering (BinanceClient unit tests)
# ---------------------------------------------------------------------------

class TestCancelOpenOrdersCoidPrefix:
    """BinanceClient.cancel_open_orders should filter by prefix when given one."""

    def _client_with_orders(self, orders: list) -> BinanceClient:
        mc_ex = MagicMock()
        mc_ex.fetch_open_orders.return_value = orders
        return BinanceClient(ex=mc_ex, env="testnet", coid_prefix="snap-v1-")

    def _order(self, order_id: str, coid: str | None = None,
                via_info: bool = False) -> dict:
        o: dict = {"id": order_id}
        if coid is not None:
            if via_info:
                o["info"] = {"clientOrderId": coid}
            else:
                o["clientOrderId"] = coid
        return o

    def test_prefix_cancels_only_matching_top_level_coid(self):
        orders = [
            self._order("1", "snap-v1-1234-s"),   # should cancel
            self._order("2", "snap-d3-1234-s"),   # different prefix — skip
            self._order("3", "web_manual"),        # manual — skip
            self._order("4"),                      # no coid — skip
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.cancel_order.assert_called_once_with("1", "BTC/USDT:USDT")

    def test_prefix_cancels_matching_coid_from_info_dict(self):
        """ccxt sometimes nests clientOrderId inside the `info` dict."""
        orders = [
            self._order("1", "snap-v1-1234-s", via_info=True),  # matches
            self._order("2", "snap-d3-9999-t", via_info=True),  # different prefix
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.cancel_order.assert_called_once_with("1", "BTC/USDT:USDT")

    def test_no_prefix_cancels_all_orders(self):
        orders = [
            self._order("1", "snap-v1-1234-s"),
            self._order("2", "snap-d3-9999-t"),
            self._order("3"),
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT")  # coid_prefix=None
        assert n == 3

    def test_fetch_failure_returns_zero_without_cancelling(self):
        mc_ex = MagicMock()
        mc_ex.fetch_open_orders.side_effect = Exception("network error")
        client = BinanceClient(ex=mc_ex, env="testnet", coid_prefix="snap-v1-")
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 0
        mc_ex.cancel_order.assert_not_called()

    def test_returns_count_of_cancelled_orders(self):
        orders = [
            self._order("1", "snap-v1-aaa-s"),
            self._order("2", "snap-v1-bbb-t"),
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 2

    def test_info_present_but_none_does_not_crash(self):
        """Regression: an order with info=None (key present, value None) must
        not raise AttributeError under prefix filtering. `o.get("info", {})`
        does NOT fall back to {} when the value is None, so the code guards with
        `(o.get("info") or {})`. The loop must skip the no-coid order and keep
        going."""
        orders = [
            {"id": "1", "info": None},                      # no coid, info None → skip
            {"id": "2", "clientOrderId": "snap-v1-aaa-s"},  # matches → cancel
        ]
        client = self._client_with_orders(orders)
        n = client.cancel_open_orders("BTC/USDT:USDT", coid_prefix="snap-v1-")
        assert n == 1
        client.ex.cancel_order.assert_called_once_with("2", "BTC/USDT:USDT")


# ---------------------------------------------------------------------------
# 2. _detect_bracket_exit: sweep on open->flat transition
# ---------------------------------------------------------------------------

class TestDetectBracketExit:

    def _setup_transition(self, bot, mc) -> None:
        """Configure mocks for a long->flat bracket-exit scenario."""
        bot._last_position_side = "long"
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [{"side": "sell", "price": 66000.0}]

    def _run(self, bot, mc, record_fill_spy=None) -> None:
        fill_patch = patch("bot.state.record_fill",
                           side_effect=record_fill_spy)
        with patch("bot.sqlite3.connect") as mock_db, \
             fill_patch, \
             patch("bot.state.enqueue_bot_event"), \
             patch("bot.send_alert"):
            (mock_db.return_value
             .__enter__.return_value
             .execute.return_value
             .fetchone.return_value) = ("entry", "long", 0.01, 65000.0, "1234567890123")
            bot._detect_bracket_exit(equity=10000.0)

    def test_cancel_called_once_with_prefix(self):
        bot, mc = _make_bot()
        self._setup_transition(bot, mc)
        self._run(bot, mc)
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_cancel_called_before_pnl_recording(self):
        """Cancel must precede state.record_fill so the orphan is swept even
        if the PnL lookup later errors out."""
        bot, mc = _make_bot()
        self._setup_transition(bot, mc)
        call_order: list[str] = []
        mc.cancel_open_orders.side_effect = (
            lambda *a, **kw: call_order.append("cancel") or 0
        )
        self._run(bot, mc,
                  record_fill_spy=lambda *a, **kw: call_order.append("record_fill"))
        assert "cancel" in call_order, "cancel_open_orders was not called"
        assert "record_fill" in call_order, "state.record_fill was not called"
        assert (call_order.index("cancel") < call_order.index("record_fill")), (
            f"cancel must precede record_fill; got order: {call_order}"
        )

    def test_no_cancel_when_position_was_already_flat(self):
        """No open->flat transition → no cancel."""
        bot, mc = _make_bot()
        bot._last_position_side = "flat"
        mc.fetch_position.return_value = _flat()
        bot._detect_bracket_exit(equity=10000.0)
        mc.cancel_open_orders.assert_not_called()

    def test_no_cancel_when_position_still_open(self):
        """Transition open->open → no cancel."""
        bot, mc = _make_bot()
        bot._last_position_side = "long"
        mc.fetch_position.return_value = _open()
        bot._detect_bracket_exit(equity=10000.0)
        mc.cancel_open_orders.assert_not_called()

    def test_cancel_failure_does_not_block_pnl_recording(self):
        """A cancel exception must not swallow the PnL recording that follows."""
        bot, mc = _make_bot()
        self._setup_transition(bot, mc)
        mc.cancel_open_orders.side_effect = Exception("exchange timeout")
        with patch("bot.sqlite3.connect") as mock_db, \
             patch("bot.state.record_fill") as mock_fill, \
             patch("bot.state.enqueue_bot_event"), \
             patch("bot.send_alert"):
            (mock_db.return_value
             .__enter__.return_value
             .execute.return_value
             .fetchone.return_value) = ("entry", "long", 0.01, 65000.0, "1234567890123")
            bot._detect_bracket_exit(equity=10000.0)
        mock_fill.assert_called_once()


# ---------------------------------------------------------------------------
# 3. _place_live_entry: pre-clear before entry placement
# ---------------------------------------------------------------------------

class TestPlaceLiveEntry:

    def _decision(self) -> SignalDecision:
        return SignalDecision(side="long", price=65000.0,
                              sl_distance=975.0, tp_distance=1950.0, debug={})

    def test_cancel_called_before_market_order(self):
        bot, mc = _make_bot(order_type="market")
        mc.reset_mock()  # clear __init__ call history
        mc.cancel_open_orders.return_value = 0
        mc.market_order_with_bracket.return_value = {"entry": {}, "sl": {}, "tp": {}}

        with patch("bot.trade_events.record_market_entry"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1234567890123", equity=10000.0)

        call_names = [c[0] for c in mc.method_calls]
        assert "cancel_open_orders" in call_names
        assert "market_order_with_bracket" in call_names
        assert (call_names.index("cancel_open_orders")
                < call_names.index("market_order_with_bracket")), (
            f"cancel must precede market_order_with_bracket; got: {call_names}"
        )
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_cancel_called_before_limit_order(self):
        bot, mc = _make_bot(order_type="limit")
        mc.reset_mock()
        mc.cancel_open_orders.return_value = 0
        mc.limit_order_with_bracket.return_value = {
            "entry": {}, "sl": {}, "tp": {},
            "filled_as": "limit", "fill_price": 64990.0, "filled_qty": 0.01,
        }

        with patch("bot.trade_events.record_limit_entry"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1234567890123", equity=10000.0)

        call_names = [c[0] for c in mc.method_calls]
        assert "cancel_open_orders" in call_names
        assert "limit_order_with_bracket" in call_names
        assert (call_names.index("cancel_open_orders")
                < call_names.index("limit_order_with_bracket")), (
            f"cancel must precede limit_order_with_bracket; got: {call_names}"
        )
        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_cancel_failure_does_not_block_entry_placement(self):
        """A cancel exception before the entry must not prevent the order."""
        bot, mc = _make_bot(order_type="market")
        mc.cancel_open_orders.side_effect = Exception("timeout")
        mc.market_order_with_bracket.return_value = {"entry": {}, "sl": {}, "tp": {}}

        with patch("bot.trade_events.record_market_entry"):
            bot._place_live_entry(self._decision(), qty=0.01,
                                  signal_id="1234567890123", equity=10000.0)

        mc.market_order_with_bracket.assert_called_once()


# ---------------------------------------------------------------------------
# 4. boot(): orphan sweep when flat at startup
# ---------------------------------------------------------------------------

class TestBootOrphanSweep:

    def _state_patches(self):
        return patch.multiple(
            "bot.state",
            init_db=MagicMock(),
            get_float=MagicMock(return_value=0.0),
            set_float=MagicMock(),
            set_meta=MagicMock(),
            enqueue_bot_event=MagicMock(),
            record_event=MagicMock(),
            latest_entry_coid_root=MagicMock(return_value=None),
        )

    def test_flat_live_sweeps_orphaned_orders(self):
        bot, mc = _make_bot(dry_run=False)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _flat()
        mc.cancel_open_orders.return_value = 1  # one orphan found

        with _boot_patches(), self._state_patches():
            bot.boot()

        mc.cancel_open_orders.assert_called_once_with(
            "BTC/USDT:USDT", coid_prefix="snap-v1-"
        )

    def test_flat_dry_run_does_not_sweep(self):
        bot, mc = _make_bot(dry_run=True)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _flat()

        with _boot_patches(), self._state_patches():
            bot.boot()

        mc.cancel_open_orders.assert_not_called()

    def test_open_position_does_not_trigger_flat_sweep(self):
        """With an open position at boot, the boot-flatten branch runs (calling
        close_position), NOT the new flat-path else-branch sweep.

        NOTE: in PRODUCTION close_position itself calls cancel_open_orders
        (binance_client.py) — but here the client is a MagicMock whose
        close_position is a no-op stub that does not call through. So this test
        only asserts the NEW else-branch sweep does not fire on the open-position
        path; it is NOT a claim that production cancels nothing here."""
        bot, mc = _make_bot(dry_run=False)
        mc.fetch_equity_usdt.return_value = 1000.0
        mc.fetch_position.return_value = _open()

        with _boot_patches(), self._state_patches():
            bot.boot()

        # The new flat-path else-branch must not run when a position is open.
        # (mc.close_position is a mock stub, so it doesn't cancel through here.)
        mc.cancel_open_orders.assert_not_called()
