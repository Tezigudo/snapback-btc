"""
Deterministic signal generation. Implemented in P2.

Strategy v1 = "RSI Extreme + EMA Confluence + Volume + Funding":
  LONG entry when ALL of:
    - RSI(2, 15m) close < rsi_long_threshold (default 10)
    - price > EMA(200, 1h)
    - volume(15m) > volume_multiple * SMA(20, vol)
    - funding_rate <= funding_long_max (default -0.0003 per 8h)

  SHORT is the mirror.

Exits = +1.5 * ATR(20, 1h) TP, -1.0 * ATR(20, 1h) SL, time-stop at 48 bars.
"""

from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


def generate_signal(*args, **kwargs) -> Side:
    """Stub. Implemented in P2."""
    raise NotImplementedError("Strategy signals are implemented in P2.")
