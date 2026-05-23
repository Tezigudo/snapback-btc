"""Tests for the outbox + consolidate push module.

The bot queues events in state.db `outbox` and pushes them to consolidate's
/bot-event/batch endpoint. These tests cover:
  - enqueue + drain happy path
  - dedup behavior: external_id = "snapback-btc:<outbox.id>" is stable
  - HTTP failure leaves rows queued, increments attempts, doesn't lose data
  - Missing config (CONSOLIDATE_API_URL/TOKEN unset) → drain is a no-op
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import URLError


def _with_temp_db_and_env(test_fn):
    """Patch DB_PATH + clear consolidate env vars so tests are isolated."""
    def wrapped():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "state.db"
            with patch("exchange.state.DB_PATH", tmp), \
                 patch.dict("os.environ",
                            {"CONSOLIDATE_API_URL": "",
                             "CONSOLIDATE_API_TOKEN": ""},
                            clear=False):
                test_fn(tmp)
    return wrapped


@_with_temp_db_and_env
def test_outbox_enqueue_inserts_row(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    row_id = state.enqueue_bot_event(
        "heartbeat", equity_usd=101.0, payload={"halt_present": False})
    assert row_id > 0
    pending = state.outbox_pending(10)
    assert len(pending) == 1
    assert pending[0][0] == row_id
    assert pending[0][1] == "heartbeat"
    body = json.loads(pending[0][2])
    assert body["equity_usd"] == 101.0
    assert body["payload"]["halt_present"] is False
    assert "bot_ts_ms" in body and body["bot_ts_ms"] > 0


@_with_temp_db_and_env
def test_outbox_delete_removes_rows(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    a = state.enqueue_bot_event("heartbeat")
    b = state.enqueue_bot_event("heartbeat")
    assert state.outbox_size() == 2
    state.outbox_delete([a, b])
    assert state.outbox_size() == 0


@_with_temp_db_and_env
def test_outbox_mark_failed_increments_attempts(tmp: Path) -> None:
    from exchange import state
    state.init_db()
    row_id = state.enqueue_bot_event("heartbeat")
    state.outbox_mark_failed([row_id], "net: timeout")
    with sqlite3.connect(tmp) as c:
        attempts, err = c.execute(
            "SELECT attempts, last_error FROM outbox WHERE id=?", (row_id,)
        ).fetchone()
    assert attempts == 1
    assert err == "net: timeout"
    state.outbox_mark_failed([row_id], "net: again")
    with sqlite3.connect(tmp) as c:
        attempts, err = c.execute(
            "SELECT attempts, last_error FROM outbox WHERE id=?", (row_id,)
        ).fetchone()
    assert attempts == 2
    assert err == "net: again"


@_with_temp_db_and_env
def test_drain_no_op_when_unconfigured(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat")
    result = consolidate_push.drain()
    assert result == {"skipped": "not_configured"}
    # Event stays queued.
    assert state.outbox_size() == 1


@_with_temp_db_and_env
def test_drain_no_op_when_outbox_empty(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}):
        result = consolidate_push.drain()
    assert result == {"sent": 0, "pending": 0}


@_with_temp_db_and_env
def test_drain_pushes_and_deletes_on_success(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event(
        "entry", signal_id="1716120000000", strategy="multifactor-v1",
        side="long", qty=0.001, price_usd=65000, equity_usd=101)
    state.enqueue_bot_event("heartbeat", equity_usd=101)

    # Fake HTTP response: {ok: true, inserted: 2, skipped: 0}
    fake_response = MagicMock()
    fake_response.read.return_value = b'{"ok":true,"inserted":2,"skipped":0,"errors":[]}'
    fake_response.__enter__ = lambda self: self
    fake_response.__exit__ = lambda self, *_a: None

    captured_request = {}
    def fake_urlopen(req, timeout=None):
        captured_request["url"] = req.full_url
        captured_request["headers"] = dict(req.headers)
        captured_request["body"] = json.loads(req.data.decode("utf-8"))
        return fake_response

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok123"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = consolidate_push.drain()

    assert result["sent"] == 2
    assert result["inserted"] == 2
    assert state.outbox_size() == 0  # both deleted
    # Request structure
    assert captured_request["url"] == "https://example.test/bot-event/batch"
    assert captured_request["headers"]["Authorization"] == "Bearer tok123"
    body = captured_request["body"]
    assert len(body["events"]) == 2
    e0 = body["events"][0]
    assert e0["source"] == "snapback-btc"
    assert e0["external_id"].startswith("snapback-btc:")
    assert e0["kind"] == "entry"
    assert e0["signal_id"] == "1716120000000"
    assert e0["side"] == "long"
    assert e0["price_usd"] == 65000


@_with_temp_db_and_env
def test_drain_keeps_rows_on_network_error(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat", equity_usd=101)

    def fake_urlopen(_req, timeout=None):
        raise URLError("connection refused")

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = consolidate_push.drain()

    assert "error" in result
    # Row stays queued.
    assert state.outbox_size() == 1
    # attempts incremented.
    with sqlite3.connect(tmp) as c:
        attempts, last_error = c.execute(
            "SELECT attempts, last_error FROM outbox LIMIT 1"
        ).fetchone()
    assert attempts == 1
    assert "net:" in last_error


@_with_temp_db_and_env
def test_drain_keeps_rows_on_4xx(tmp: Path) -> None:
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat", equity_usd=101)

    from urllib.error import HTTPError
    from io import BytesIO

    def fake_urlopen(_req, timeout=None):
        raise HTTPError(
            url="https://example.test/bot-event/batch", code=401,
            msg="Unauthorized", hdrs={}, fp=BytesIO(b'{"error":"unauthorized"}'),
        )

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "bad-token"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = consolidate_push.drain()

    assert "error" in result
    assert "HTTP 401" in result["error"]
    # Row stays queued for retry — operator must fix the token first.
    assert state.outbox_size() == 1


@_with_temp_db_and_env
def test_drain_handles_partial_failure_207(tmp: Path) -> None:
    """Server returned 207 (some events failed, others succeeded). Bot must
    delete the successful local outbox rows AND keep the failed ones queued."""
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat", equity_usd=101)
    state.enqueue_bot_event("entry", signal_id="bad", side="long",
                            qty=0.001, price_usd=65000)
    state.enqueue_bot_event("heartbeat", equity_usd=101)
    assert state.outbox_size() == 3

    fake_response = MagicMock()
    fake_response.status = 207
    # id2 (the entry) failed; id1 and id3 succeeded.
    fake_response.read.return_value = json.dumps({
        "ok": False,
        "inserted": 2,
        "skipped": 0,
        "errors": [
            {"external_id": f"snapback-btc:{id2}", "message": "constraint violation"},
        ],
    }).encode("utf-8")
    fake_response.__enter__ = lambda self: self
    fake_response.__exit__ = lambda self, *_a: None

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", return_value=fake_response):
        result = consolidate_push.drain()

    assert result["status"] == 207
    assert result["failed"] == 1
    assert result["deleted_locally"] == 2
    # Only the failed event remains; successful ones deleted.
    assert state.outbox_size() == 1
    with sqlite3.connect(tmp) as c:
        remaining = c.execute("SELECT id, attempts, last_error FROM outbox").fetchone()
    assert remaining[0] == id2
    assert remaining[1] == 1
    assert "207" in remaining[2]


@_with_temp_db_and_env
def test_external_id_is_stable_across_retries(tmp: Path) -> None:
    """If push fails and we retry, the same external_id must be sent so
    the server-side dedup kicks in. Tests this by simulating: enqueue,
    fail to push, retry, capture both requests, check external_ids match.
    """
    from exchange import state
    from tools import consolidate_push
    state.init_db()
    state.enqueue_bot_event("heartbeat")

    captured = []

    def fake_urlopen_capture(req, timeout=None):
        captured.append(json.loads(req.data.decode("utf-8")))
        # First call fails, second succeeds — but the SAME row should be sent.
        if len(captured) == 1:
            raise URLError("transient")
        fake = MagicMock()
        fake.read.return_value = b'{"ok":true,"inserted":1,"skipped":0,"errors":[]}'
        fake.__enter__ = lambda self: self
        fake.__exit__ = lambda self, *_a: None
        return fake

    with patch.dict("os.environ",
                    {"CONSOLIDATE_API_URL": "https://example.test",
                     "CONSOLIDATE_API_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen_capture):
        consolidate_push.drain()
        consolidate_push.drain()

    assert len(captured) == 2
    assert captured[0]["events"][0]["external_id"] == captured[1]["events"][0]["external_id"]
