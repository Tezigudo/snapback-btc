"""Boot-resume Phase B — the fixes that only matter once adoption is ARMED.

Plan: docs/BOOT_RESUME_PHASE_B_PLAN.md. These run against the REAL per-test
sqlite DB that tests/conftest.py provides, not patched `state.*` calls: B1 and
B2 are about what gets PERSISTED (a close row, a bar timestamp, a guard
counter), so patching the persistence away would test nothing.
"""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd

from exchange import state
from tests.test_boot_resume import (
    _ARMED_BRACKET,
    _MINIMAL_PARAMS,
    _boot_patches,
    _flat,
    _make_bot,
    _open,
    _principal_patches,
)

_ARMED_PARAMS = {**_MINIMAL_PARAMS,
                 "boot_resume": {"enabled": True, "observe_only": False}}


def _seed_db(bracket: dict | None = _ARMED_BRACKET, root: str = "sig-1",
             entry_ts: str | None = None) -> None:
    """An entry fill + its stashed bracket, as _place_live_entry leaves them."""
    state.record_fill(side="long", qty=0.01, price=65000.0, reason="entry",
                      client_order_id_root=root)
    if entry_ts is not None:
        with sqlite3.connect(state.DB_PATH) as c:
            c.execute("UPDATE fills SET ts=? WHERE reason='entry'", (entry_ts,))
    if bracket is not None:
        state.set_meta("active_bracket", json.dumps(bracket))


def _boot(bot, mc, position):
    mc.fetch_equity_usdt.return_value = 1000.0
    mc.fetch_position.return_value = position
    with _boot_patches(), _principal_patches():
        bot.boot()


def _fills() -> list[tuple]:
    with sqlite3.connect(state.DB_PATH) as c:
        return c.execute(
            "SELECT side, reason, price, client_order_id_root FROM fills ORDER BY id"
        ).fetchall()


def _outbox_kinds() -> list[str]:
    with sqlite3.connect(state.DB_PATH) as c:
        return [r[0] for r in c.execute("SELECT kind FROM outbox ORDER BY id")]


# ---------------------------------------------------------------------------
# B1 — an exit between the boot read and the first tick must be recorded
# ---------------------------------------------------------------------------

class TestB1TrackingSeed:

    def test_bracket_fill_before_first_tick_writes_the_close(self):
        """T1. Adopt, then the SL fills before the loop's first tick.

        Teeth: with `_seed_adopted_tracking`'s assignments removed, the first
        tick sees unknown→flat, returns without writing, and this fails (checked
        by hand while building Phase B)."""
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        mc.close_position.assert_not_called()
        assert "boot_adopt" in _outbox_kinds()

        # The bracket SL fills while boot() is still finishing.
        mc.fetch_position.return_value = _flat()
        mc.ex.fetch_my_trades.return_value = [
            {"side": "sell", "price": 64500.0, "amount": 0.01}]
        with patch("bot.send_alert"):
            bot._detect_bracket_exit(995.0)

        closes = [f for f in _fills() if f[0] == "close"]
        assert closes == [("close", "bracket_exit", 64500.0, "sig-1")]
        assert "exit" in _outbox_kinds()

    def test_seed_names_the_adopted_position(self):
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open(entry_price=65010.0, qty=0.01))
        assert bot._last_position_side == "long"
        assert bot._last_position_entry == 65010.0
        assert bot._last_position_qty == 0.01
        assert bot._last_entry_root == "sig-1"

    def test_an_unchanged_position_on_the_first_tick_is_not_an_exit(self):
        """The seed must not fabricate an exit when nothing happened."""
        _seed_db()
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        _boot(bot, mc, _open())
        bot._detect_bracket_exit(1000.0)
        assert [f for f in _fills() if f[0] == "close"] == []
        assert "exit" not in _outbox_kinds()

    def test_flatten_path_does_not_seed(self):
        """Refused → flatten must leave the tracking exactly as before."""
        _seed_db(bracket=None)          # no active_bracket → refuse
        bot, mc = _make_bot(params=dict(_ARMED_PARAMS))
        mc.close_position.return_value = {}
        _boot(bot, mc, _open())
        mc.close_position.assert_called_once()
        assert bot._last_position_side == "unknown"
        assert bot._last_entry_root is None
