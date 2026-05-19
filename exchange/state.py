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
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from .env import REPO_ROOT

DB_PATH = REPO_ROOT / "data" / "state.db"


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
