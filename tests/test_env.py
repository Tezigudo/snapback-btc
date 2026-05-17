"""
Tests for the env + lockfile gate. These must pass before any other phase.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from exchange.env import EnvError, LOCKFILE, get_env


def test_default_is_testnet(monkeypatch):
    monkeypatch.delenv("BINANCE_ENV", raising=False)
    assert get_env() == "testnet"


def test_explicit_testnet(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "testnet")
    assert get_env() == "testnet"


def test_invalid_env_rejected(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "production")
    with pytest.raises(EnvError):
        get_env()


def test_mainnet_without_lockfile_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("BINANCE_ENV", "mainnet")
    # Ensure no lockfile.
    if LOCKFILE.exists():
        pytest.skip("lockfile already exists in checkout; manual cleanup needed")
    with pytest.raises(EnvError, match="lockfile missing"):
        get_env()


def test_mainnet_with_lockfile_allowed(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "mainnet")
    created = False
    if not LOCKFILE.exists():
        LOCKFILE.touch()
        created = True
    try:
        assert get_env() == "mainnet"
    finally:
        if created:
            LOCKFILE.unlink()
