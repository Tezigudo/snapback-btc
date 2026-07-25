"""
A non-v1 leg must refuse to boot without its own `.env.<instance>`.

Why this guard exists: `load_env_for_instance` returns None when the file is
absent, which is correct for v1 (the base `.env` IS its env) but a live-money
hazard for every other leg — without the overlay the leg inherits v1's
BINANCE_API_KEY and places orders for a DIFFERENT symbol inside v1's account,
pushes telemetry under v1's CONSOLIDATE_SOURCE, and emails under v1's ALERT_TAG.
exchange/env.py's docstring names this as the thing the overlay prevents, but
nothing enforced it until bot._main() did.

These tests exercise the predicate rather than booting a client, so they never
touch the network or a real key.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from exchange.env import load_env_for_instance  # noqa: E402


def _guard_blocks(instance: str) -> bool:
    """Mirror of the condition in bot._main()."""
    return instance != "v1" and load_env_for_instance(instance) is None


def test_v1_is_exempt_even_without_an_env_file():
    """v1 legitimately runs off the base .env — it must never be blocked."""
    assert _guard_blocks("v1") is False


@pytest.mark.parametrize("instance", ["donchian", "cnh_short", "sol_supertrend"])
def test_non_v1_leg_blocked_when_env_file_missing(instance):
    env_file = REPO / f".env.{instance}"
    if env_file.exists():
        pytest.skip(f"{env_file.name} exists in this checkout — guard not exercised")
    assert _guard_blocks(instance) is True


def test_non_v1_leg_allowed_when_env_file_present(tmp_path, monkeypatch):
    """With the overlay present the guard must let the leg through."""
    import exchange.env as envmod
    monkeypatch.setattr(envmod, "REPO_ROOT", tmp_path)
    (tmp_path / ".env.sol_supertrend").write_text(
        "BINANCE_API_KEY=dummy\nBINANCE_API_SECRET=dummy\n")
    assert envmod.load_env_for_instance("sol_supertrend") is not None


def test_bot_exits_2_for_unkeyed_instance():
    """End-to-end: the CLI must exit non-zero BEFORE constructing a client.

    Exit code 2 (not 1) so systemd's Restart=on-failure surfaces it as a config
    error rather than a transient crash-loop.
    """
    if (REPO / ".env.sol_supertrend").exists():
        pytest.skip(".env.sol_supertrend exists — guard not exercised")
    proc = subprocess.run(
        [sys.executable, "-m", "bot", "--instance", "sol_supertrend", "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert "has no .env.sol_supertrend" in proc.stderr
