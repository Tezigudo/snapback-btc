"""
Environment + mainnet lockfile gate.

Reads BINANCE_ENV from .env (default: testnet). If BINANCE_ENV=mainnet, also
requires a `confirm_mainnet.lock` file at the repo root — created manually by
the user, never committed.

This is the first defence against accidentally running real-money trades.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
except ImportError:  # P0: deps may not be installed yet
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

Env = Literal["testnet", "mainnet"]

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE = REPO_ROOT / "confirm_mainnet.lock"

# Load .env once at import time. Missing file is fine — env vars may be set elsewhere.
load_dotenv(REPO_ROOT / ".env", override=False)


class EnvError(RuntimeError):
    pass


def get_env() -> Env:
    """Return the active environment, enforcing the mainnet lockfile gate."""
    raw = os.environ.get("BINANCE_ENV", "testnet").strip().lower()
    if raw not in ("testnet", "mainnet"):
        raise EnvError(
            f"BINANCE_ENV must be 'testnet' or 'mainnet', got {raw!r}"
        )

    if raw == "mainnet" and not LOCKFILE.exists():
        raise EnvError(
            f"BINANCE_ENV=mainnet but lockfile missing: {LOCKFILE}\n"
            "Refusing to run on mainnet without explicit confirmation.\n"
            "If you really mean it, run `/promote-mainnet` and follow the checklist."
        )

    return raw  # type: ignore[return-value]


def get_api_credentials() -> tuple[str, str]:
    """Return (api_key, api_secret), erroring if missing."""
    key = os.environ.get("BINANCE_API_KEY", "").strip()
    secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not key or not secret:
        raise EnvError(
            "BINANCE_API_KEY / BINANCE_API_SECRET not set. "
            f"Edit {REPO_ROOT / '.env'} (copy from .env.example)."
        )
    return key, secret


def is_halted() -> bool:
    """True if data/HALT exists — bot must close all + exit clean."""
    return (REPO_ROOT / "data" / "HALT").exists()


if __name__ == "__main__":
    print(get_env())
