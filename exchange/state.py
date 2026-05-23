"""SQLite state store for the bot. WAL mode. Survives restarts.

Schema:
  meta(key TEXT PRIMARY KEY, value TEXT)
    'deploy_start_equity' : float
    'deploy_start_ts'     : ISO ts
    'last_entry_bar_ts'   : ISO ts of bar bot last considered for entry
    'consecutive_losses'  : int

  fills(id INTEGER PRIMARY KEY, ts TEXT, side TEXT, qty REAL, price REAL,
        pnl_usd REAL, reason TEXT, equity_after REAL,
        client_order_id_root TEXT NULL)

  events(id INTEGER PRIMARY KEY, ts TEXT, level TEXT, kind TEXT, msg TEXT,
         signal_id TEXT NULL)

  client_order_id_root: the per-signal millisecond id that anchors all 3 legs
    of a bot trade (entry + SL + TP) on Binance. See exchange/binance_client.py
    `_coid()`. Lets us join state.db fills to Binance-reported trades by
    clientOrderId prefix `snap-v1-<root>-{e|s|t|x|bf|h|k}`.

  outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, payload TEXT,
         created_at TEXT, attempts INTEGER, last_error TEXT NULL)
    Pending events queued for push to investing-consolidate's /bot-event API.
    Rows are inserted by enqueue_bot_event() and removed by
    tools.consolidate_push.drain() once the API acknowledges. The bot's local
    state.db is the source of truth; consolidate is a downstream read-only
    view. If consolidate is unreachable, events queue here and replay
    automatically when the API comes back.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from .env import REPO_ROOT

# Env-var override so a second bot instance (Donchian leg) can point at
# data/state_donchian.db without code changes. Default matches the single-bot
# v1 deploy.
DB_PATH = Path(os.environ.get("SNAPBACK_STATE_DB", REPO_ROOT / "data" / "state.db"))


def set_db_path(path: str | Path) -> None:
    """Override DB_PATH from bot.main() after CLI args are parsed.

    Used by the second bot instance (Donchian leg) to point at
    data/state_donchian.db. Call BEFORE any other state.* function.
    """
    global DB_PATH
    DB_PATH = Path(path)


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10.0)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def _columns(c: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            pnl_usd REAL,
            reason TEXT,
            equity_after REAL,
            client_order_id_root TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            kind TEXT NOT NULL,
            msg TEXT NOT NULL,
            signal_id TEXT
        );
        CREATE TABLE IF NOT EXISTS outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_outbox_id ON outbox(id);
        """)
        # Additive migrations for pre-existing databases.
        if "client_order_id_root" not in _columns(c, "fills"):
            c.execute("ALTER TABLE fills ADD COLUMN client_order_id_root TEXT")
        if "signal_id" not in _columns(c, "events"):
            c.execute("ALTER TABLE events ADD COLUMN signal_id TEXT")


def get_meta(key: str, default: str | None = None) -> str | None:
    with _conn() as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(key: str, value: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO meta(key, value) VALUES (?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, value))


def get_float(key: str, default: float = 0.0) -> float:
    v = get_meta(key)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def set_float(key: str, value: float) -> None:
    set_meta(key, repr(float(value)))


def record_fill(side: str, qty: float, price: float, reason: str,
                pnl_usd: float | None = None, equity_after: float | None = None,
                client_order_id_root: str | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO fills(ts, side, qty, price, pnl_usd, reason, "
            "equity_after, client_order_id_root) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), side, qty, price,
             pnl_usd, reason, equity_after, client_order_id_root),
        )


def record_event(level: str, kind: str, msg: str | dict,
                 signal_id: str | None = None) -> None:
    if isinstance(msg, dict):
        msg = json.dumps(msg, default=str)
    with _conn() as c:
        c.execute(
            "INSERT INTO events(ts, level, kind, msg, signal_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), level, kind, msg, signal_id),
        )


def enqueue_bot_event(
    kind: str,
    *,
    signal_id: str | None = None,
    strategy: str | None = None,
    side: str | None = None,
    qty: float | None = None,
    price_usd: float | None = None,
    notional_usd: float | None = None,
    equity_usd: float | None = None,
    payload: dict | None = None,
) -> int:
    """Queue a bot event for push to consolidate's /bot-event API.

    The push happens out-of-band via tools.consolidate_push.drain(). This
    function just inserts to the local outbox table and returns the row id.

    bot_ts_ms is captured here (at enqueue time) and stored in the payload
    so a queued event keeps its original timestamp even if the push is
    delayed by an API outage.
    """
    import time as _time
    body = json.dumps({
        "bot_ts_ms": int(_time.time() * 1000),
        "kind": kind,
        "signal_id": signal_id,
        "strategy": strategy,
        "side": side,
        "qty": qty,
        "price_usd": price_usd,
        "notional_usd": notional_usd,
        "equity_usd": equity_usd,
        "payload": payload or {},
    }, default=str)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO outbox(kind, payload, created_at) VALUES (?, ?, ?)",
            (kind, body, datetime.utcnow().isoformat()),
        )
        return int(cur.lastrowid or 0)


def outbox_pending(limit: int = 50) -> list[tuple[int, str, str, int]]:
    """Return up to `limit` pending outbox rows oldest-first.

    Each row: (id, kind, payload_json, attempts).
    """
    with _conn() as c:
        return [
            (int(r[0]), str(r[1]), str(r[2]), int(r[3]))
            for r in c.execute(
                "SELECT id, kind, payload, attempts FROM outbox "
                "ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()
        ]


def outbox_delete(ids: list[int]) -> int:
    """Hard-delete outbox rows by id (after successful push)."""
    if not ids:
        return 0
    with _conn() as c:
        placeholders = ",".join("?" * len(ids))
        cur = c.execute(f"DELETE FROM outbox WHERE id IN ({placeholders})", ids)
        return cur.rowcount or 0


def outbox_mark_failed(ids: list[int], error: str) -> None:
    """Increment attempts + record last_error for the given outbox rows."""
    if not ids:
        return
    with _conn() as c:
        placeholders = ",".join("?" * len(ids))
        c.execute(
            f"UPDATE outbox SET attempts = attempts + 1, last_error = ? "
            f"WHERE id IN ({placeholders})",
            [error] + ids,
        )


def outbox_size() -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM outbox").fetchone()
    return int(row[0]) if row else 0


def latest_entry_coid_root() -> str | None:
    """The most recent entry fill's client_order_id_root, or None.

    Used by close paths (time-stop, boot-flatten, HALT, kill-switch) to tag
    closing orders with the same root as the entry — so consolidate's
    importer can join entry+exit fills via the shared root.

    Returns None if no entry has been recorded OR if the latest entry fill
    has no root (e.g. recorded by a pre-COID bot version). In that case the
    close is placed untagged, which is honest: better than mis-attributing
    to an older, already-closed position's root.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT client_order_id_root FROM fills "
            "WHERE reason='entry' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None
