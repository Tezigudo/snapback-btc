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
    # Raised 3→20 per explicit user override (P3.4+). At 20x liquidation
    # ≈ 5% adverse move; flash crashes can gap past SLs. This ceiling
    # is now load-bearing for tail-risk control — DO NOT raise further
    # without a separate RISK_REVIEW.
    MAX_LEVERAGE: int = 20

    # If equity drawdown since this UTC day's start reaches this %, block NEW
    # entries for the rest of the day (bot._daily_loss_blocks_entry). Does NOT
    # flatten or HALT — open positions keep their exchange-native brackets, and
    # the daily anchor resets at the next UTC midnight. The tighter,
    # daily-resetting sibling of the cumulative kill-switch.
    # Raised 2.0 → 3.5 → 4.5 on 2026-07-12 (RISK_REVIEW, user-approved): the
    # breaker must exceed one full stop-loss or a single normal SL freezes the
    # leg for the rest of the UTC day — a halt no backtest models. Moves in
    # lockstep with sizing risk_per_trade_pct (now 3.5% per the sizing sweep,
    # reports/sizing_sweep_2026.json): 4.5 tolerates exactly one full SL
    # (3.5% + slippage) and still halts before a second consecutive one.
    MAX_DAILY_LOSS_PCT: float = 4.5

    # Single-symbol simplicity for v1.
    MAX_OPEN_POSITIONS: int = 1

    # Rate-limit defence against runaway loops.
    MAX_ORDERS_PER_MINUTE: int = 6

    # Trip kill-switch after this many losses in a row.
    MAX_CONSECUTIVE_LOSSES: int = 4

    # Minimum seconds between trades, prevents rapid-fire on bad signals.
    MIN_TIME_BETWEEN_TRADES_S: int = 60

    # Hard symbol allowlist.
    # BTC: v1 (multifactor) + donchian-v3 legs.
    # SOL: added 2026-07-25 (RISK_REVIEW, user-approved) for the sol_supertrend
    #      leg — round-3 win-rate-blended walk-forward winner, 9/9 OOS folds
    #      positive, ~uncorrelated with both BTC legs (-0.02 vs multifactor-v1,
    #      +0.06 vs donchian-v3). Evidence in SOL_LEG_VERDICT.md; live↔backtest
    #      parity in tools/supertrend_parity.py (119/119 entries, 0 spurious).
    #      NOTE: this also un-blocks config/params_cnh_hybrid_short_sol.yaml,
    #      which measured CAGR 1.3% with 1,360 days underwater — that unit must
    #      NOT be enabled. The allowlist is no longer what prevents it.
    #      MAX_NOTIONAL_USD=500 stays the per-order backstop for both symbols.
    ALLOWED_SYMBOLS: tuple[str, ...] = ("BTC/USDT:USDT", "SOL/USDT:USDT")


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
