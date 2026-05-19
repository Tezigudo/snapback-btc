"""SQLite state store for the bot. WAL mode. Survives restarts.

Schema:
  meta(key TEXT PRIMARY KEY, value TEXT)
    'deploy_start_equity' : float
    'deploy_start_ts'     : ISO ts
    'last_entry_bar_ts'   : ISO ts of bar bot last considered for entry
    'consecutive_losses'  : int

  fills(id INTEGER PRIMARY KEY, ts TEXT, side TEXT, qty REAL, price REAL,
        pnl_usd REAL, reason TEXT, equity_after REAL)

  events(id INTEGER PRIMARY KEY, ts TEXT, level TEXT, kind TEXT, msg TEXT)
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
            equity_after REAL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            kind TEXT NOT NULL,
            msg TEXT NOT NULL
        );
        """)


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
                pnl_usd: float | None = None, equity_after: float | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO fills(ts, side, qty, price, pnl_usd, reason, equity_after) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), side, qty, price,
             pnl_usd, reason, equity_after),
        )


def record_event(level: str, kind: str, msg: str | dict) -> None:
    if isinstance(msg, dict):
        msg = json.dumps(msg, default=str)
    with _conn() as c:
        c.execute(
            "INSERT INTO events(ts, level, kind, msg) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), level, kind, msg),
        )
