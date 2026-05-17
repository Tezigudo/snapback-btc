"""
Hard risk ceilings. DO NOT EDIT without git env RISK_REVIEW=1.

These are absolute, non-overridable limits enforced before every order placement
and after every fill. No YAML config can relax them. The bot crashes loudly if
any check fails.

Tunable strategy params live in config/params.yaml — those are NOT a safety
surface. This file is.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskCeilings:
    # Absolute notional cap per single order, in USDT
    MAX_NOTIONAL_USD: float = 500.0

    # Hard leverage cap. Bot may request less, never more.
    MAX_LEVERAGE: int = 3

    # If equity drawdown today exceeds this %, flatten + HALT for 24h.
    MAX_DAILY_LOSS_PCT: float = 2.0

    # Single-symbol simplicity for v1.
    MAX_OPEN_POSITIONS: int = 1

    # Rate-limit defence against runaway loops.
    MAX_ORDERS_PER_MINUTE: int = 6

    # Trip kill-switch after this many losses in a row.
    MAX_CONSECUTIVE_LOSSES: int = 4

    # Minimum seconds between trades, prevents rapid-fire on bad signals.
    MIN_TIME_BETWEEN_TRADES_S: int = 60

    # Hard symbol allowlist. v1 = BTC only.
    ALLOWED_SYMBOLS: tuple[str, ...] = ("BTC/USDT:USDT",)


CEILINGS = RiskCeilings()


class RiskBreach(Exception):
    """Raised when an action would violate a hard ceiling."""


def check_notional(notional_usd: float) -> None:
    if notional_usd > CEILINGS.MAX_NOTIONAL_USD:
        raise RiskBreach(
            f"notional {notional_usd:.2f} USDT exceeds MAX_NOTIONAL_USD={CEILINGS.MAX_NOTIONAL_USD}"
        )


def check_leverage(requested: int) -> None:
    if requested > CEILINGS.MAX_LEVERAGE:
        raise RiskBreach(
            f"leverage {requested}x exceeds MAX_LEVERAGE={CEILINGS.MAX_LEVERAGE}x"
        )


def check_symbol(symbol: str) -> None:
    if symbol not in CEILINGS.ALLOWED_SYMBOLS:
        raise RiskBreach(
            f"symbol {symbol!r} not in ALLOWED_SYMBOLS={CEILINGS.ALLOWED_SYMBOLS}"
        )


def check_daily_loss(equity_now: float, equity_day_start: float) -> None:
    if equity_day_start <= 0:
        return
    loss_pct = (equity_day_start - equity_now) / equity_day_start * 100.0
    if loss_pct >= CEILINGS.MAX_DAILY_LOSS_PCT:
        raise RiskBreach(
            f"daily loss {loss_pct:.2f}% reached MAX_DAILY_LOSS_PCT={CEILINGS.MAX_DAILY_LOSS_PCT}%"
        )
