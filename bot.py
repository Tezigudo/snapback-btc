"""
Bot daemon entrypoint. Implemented in P4.

P0 stub — just confirms env loads and the lockfile gate works.
"""

from __future__ import annotations

import sys

from exchange.env import get_env, is_halted


def main() -> int:
    env = get_env()
    print(f"[snapback-btc] env={env}")
    if is_halted():
        print("[snapback-btc] HALT file present — bot would close all and exit.")
        return 0
    print("[snapback-btc] P0 stub. Live loop is implemented in P4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
