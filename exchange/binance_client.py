"""
Binance Futures client wrapper. Implemented in P4.

For P0 this is a placeholder that just records the env it would talk to. No
network calls yet, no orders, nothing real.
"""

from __future__ import annotations

from dataclasses import dataclass

from .env import get_api_credentials, get_env


@dataclass
class BinanceClient:
    env: str
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "BinanceClient":
        env = get_env()
        key, secret = get_api_credentials()
        return cls(env=env, api_key=key, api_secret=secret)

    def __repr__(self) -> str:
        # Never log secrets.
        return f"BinanceClient(env={self.env!r}, api_key='***')"
