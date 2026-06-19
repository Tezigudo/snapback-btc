"""Tests for the daily-loss circuit breaker.

The breaker (risk.check_daily_loss + bot._daily_loss_blocks_entry) blocks NEW
entries for the rest of the UTC day once intraday drawdown hits
MAX_DAILY_LOSS_PCT (2%). It is a tighter, daily-resetting sibling of the -18%
cumulative kill-switch (_check_kill_switch) and does NOT flatten or HALT.

Covers:
  1. risk.check_daily_loss: just-under threshold passes, at/over raises.
  2. The UTC-day anchor (state.meta) re-anchors on date rollover.
  3. The bot-level gate: under → entries allowed, at/over → blocked + event
     logged once, then a new UTC day re-allows.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------
# 1. Pure ceiling function
# --------------------------------------------------------------------------

def test_check_daily_loss_under_threshold_passes() -> None:
    from risk import CEILINGS, check_daily_loss
    # 1.99% loss with a 2.0% ceiling → allowed (no raise).
    day_start = 1000.0
    equity = day_start * (1 - (CEILINGS.MAX_DAILY_LOSS_PCT - 0.01) / 100.0)
    check_daily_loss(equity, day_start)  # must not raise


def test_check_daily_loss_at_threshold_raises() -> None:
    from risk import CEILINGS, RiskBreach, check_daily_loss
    day_start = 1000.0
    equity = day_start * (1 - CEILINGS.MAX_DAILY_LOSS_PCT / 100.0)  # exactly -2%
    with pytest.raises(RiskBreach):
        check_daily_loss(equity, day_start)


def test_check_daily_loss_over_threshold_raises() -> None:
    from risk import RiskBreach, check_daily_loss
    with pytest.raises(RiskBreach):
        check_daily_loss(950.0, 1000.0)  # -5%


def test_check_daily_loss_noop_on_bad_anchor() -> None:
    from risk import check_daily_loss
    # Zero/negative day-start equity is treated as "no anchor yet" → never trips.
    check_daily_loss(500.0, 0.0)
    check_daily_loss(500.0, -1.0)


# --------------------------------------------------------------------------
# Helpers: a minimal Bot stub exercising only the breaker plumbing, plus a
# temp state.db so we never touch the real one.
# --------------------------------------------------------------------------

def _with_temp_db(test_fn):
    def wrapped():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "state.db"
            with patch("exchange.state.DB_PATH", tmp):
                from exchange import state
                state.init_db()
                test_fn(tmp)
    return wrapped


class _StubLog:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def _make_breaker_bot():
    """Build a bare object exposing bot.Bot's breaker methods bound to it,
    without running Bot.__init__ (which needs an exchange client + config)."""
    import bot as bot_mod
    b = object.__new__(bot_mod.Bot)
    b.log = _StubLog()
    b._daily_loss_blocked = False
    return b, bot_mod


# --------------------------------------------------------------------------
# 2. UTC-day anchor + rollover
# --------------------------------------------------------------------------

@_with_temp_db
def test_anchor_set_then_stable_within_day(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None):
        # First call anchors to current equity.
        assert b._daily_anchor_equity(1000.0) == 1000.0
        assert state.get_meta("daily_anchor_date") is not None
        # Same day, different equity → anchor must NOT move.
        assert b._daily_anchor_equity(1234.0) == 1000.0


@_with_temp_db
def test_anchor_resets_on_utc_date_rollover(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None):
        b._daily_anchor_equity(1000.0)
        # Simulate a stale anchor from "yesterday".
        state.set_meta("daily_anchor_date", "2000-01-01")
        # New UTC day → re-anchor to today's equity.
        assert b._daily_anchor_equity(880.0) == 880.0
        assert state.get_meta("daily_anchor_date") != "2000-01-01"


# --------------------------------------------------------------------------
# 3. Bot-level gate behaviour
# --------------------------------------------------------------------------

@_with_temp_db
def test_under_threshold_allows_entry(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None):
        b._daily_anchor_equity(1000.0)
        # -1.5% loss, under the 2% ceiling → not blocked.
        assert b._daily_loss_blocks_entry(985.0) is False
        assert b._daily_loss_blocked is False


@_with_temp_db
def test_over_threshold_blocks_and_logs_event_once(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    alerts: list = []
    with patch.object(bot_mod, "send_alert", lambda *a, **k: alerts.append(a)):
        b._daily_anchor_equity(1000.0)
        # -3% loss → blocked.
        assert b._daily_loss_blocks_entry(970.0) is True
        assert b._daily_loss_blocked is True
        # Still blocked on subsequent polls the same day...
        assert b._daily_loss_blocks_entry(965.0) is True
        # ...but the event/alert fires only ONCE (latched).
        with sqlite3.connect(tmp) as c:
            n = c.execute(
                "SELECT COUNT(*) FROM events WHERE kind='daily_loss_breaker'"
            ).fetchone()[0]
        assert n == 1
        assert len(alerts) == 1


@_with_temp_db
def test_new_utc_day_reallows_after_block(tmp: Path) -> None:
    b, bot_mod = _make_breaker_bot()
    from exchange import state
    with patch.object(bot_mod, "send_alert", lambda *a, **k: None):
        b._daily_anchor_equity(1000.0)
        assert b._daily_loss_blocks_entry(970.0) is True  # -3% → blocked today
        # Roll the UTC day forward.
        state.set_meta("daily_anchor_date", "2000-01-01")
        # New day re-anchors to the (lower) current equity → fresh 0% loss → allowed.
        assert b._daily_loss_blocks_entry(970.0) is False
        assert b._daily_loss_blocked is False
