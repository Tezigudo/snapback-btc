"""Live exchange constraints for BTC/USDT:USDT perpetual futures.

Hard-coded fallbacks so a config glitch can't loosen them. The bot reads
the LIVE values from ccxt at boot and applies whichever is TIGHTER.

Source for fallbacks: ccxt market("BTC/USDT:USDT") fetched 2026-05-17.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExchangeConstraints:
    # Minimum order quantity in base asset (BTC)
    min_qty_btc: float = 0.001
    # Minimum notional cost per order in quote asset (USDT)
    min_notional_usdt: float = 50.0
    # Price tick size
    price_step: float = 0.1
    # Qty step size
    qty_step: float = 0.001
    # Standard taker fee (private; treat as upper bound)
    taker_fee: float = 0.0005


# Module-level constant used everywhere unless overridden at boot.
DEFAULT_CONSTRAINTS = ExchangeConstraints()


def merge_with_live(c: ExchangeConstraints, live_market: dict) -> ExchangeConstraints:
    """Return new constraints using the TIGHTER value between hard-coded
    fallback and the live ccxt market spec. Safe-by-default."""
    limits = live_market.get("limits", {}) or {}
    precision = live_market.get("precision", {}) or {}

    live_min_qty = (limits.get("amount", {}) or {}).get("min")
    live_min_cost = (limits.get("cost", {}) or {}).get("min")
    live_price_step = precision.get("price")
    live_qty_step = precision.get("amount")

    def tighter_min(a: float, b: float | None) -> float:
        return max(a, float(b)) if b is not None else a

    def tighter_step(a: float, b: float | None) -> float:
        return max(a, float(b)) if b is not None else a

    return ExchangeConstraints(
        min_qty_btc=tighter_min(c.min_qty_btc, live_min_qty),
        min_notional_usdt=tighter_min(c.min_notional_usdt, live_min_cost),
        price_step=tighter_step(c.price_step, live_price_step),
        qty_step=tighter_step(c.qty_step, live_qty_step),
        taker_fee=c.taker_fee,
    )


def round_qty_down(qty: float, step: float) -> float:
    """Floor-round to step. Never rounds UP — that could violate balance."""
    if step <= 0:
        return qty
    return (int(qty / step)) * step


def passes_minimums(qty: float, price: float, c: ExchangeConstraints) -> tuple[bool, str]:
    """Return (ok, reason) — reason describes the FIRST failing check."""
    notional = qty * price
    if qty < c.min_qty_btc:
        return False, f"qty {qty:.6f} below min {c.min_qty_btc}"
    if notional < c.min_notional_usdt:
        return False, f"notional ${notional:.2f} below min ${c.min_notional_usdt}"
    return True, "ok"
