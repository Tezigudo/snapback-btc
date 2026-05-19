"""Pure-function helpers used by bot.py — extracted to keep the trading
loop focused on orchestration rather than dispatch and arithmetic.

No side effects: nothing here touches the exchange, state.db, alerts, or
consolidate. The Bot class calls these to make decisions, then handles
the I/O itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy.live_multifactor_v1 import evaluate_signal
from strategy.live_v3all_wider4 import evaluate_signal_v3all_wider4


def resolve_strategy_name(params: dict) -> str:
    """Default to multifactor-v1 if `strategy_name` is missing/empty/None.

    Five callsites in bot.py used to spell this `self.params.get(...) or "..."` —
    one place to look now.
    """
    name = params.get("strategy_name")
    return str(name) if name else "multifactor-v1"


def limit_entry_price(side: str, close_price: float, offset_bps: float) -> float:
    """Maker-style limit price: place BELOW close for long buys, ABOVE for short
    sells, so the fill (if it happens) earns the maker rebate.

    offset_bps=0 means "at close". Positive offsets push further from close
    (lower fill probability, better price if filled).
    """
    offset = offset_bps / 10000.0
    return close_price * (1.0 - offset) if side == "long" else close_price * (1.0 + offset)


@dataclass(frozen=True)
class SignalDecision:
    """What a strategy evaluator returned: the side to trade (or None), the
    price it observed, and the SL/TP distances in absolute price units.

    sl_price / tp_price are computed on demand — only meaningful when
    `side` is non-None.
    """

    side: str | None
    price: float
    sl_distance: float
    tp_distance: float
    debug: dict

    @property
    def sl_price(self) -> float:
        if self.side == "long":
            return self.price - self.sl_distance
        return self.price + self.sl_distance

    @property
    def tp_price(self) -> float:
        if self.side == "long":
            return self.price + self.tp_distance
        return self.price - self.tp_distance


def evaluate_for_strategy(
    strategy_name: str,
    bars_15m: pd.DataFrame,
    funding_rate: float,
    params: dict,
) -> SignalDecision:
    """Dispatch to the live signal evaluator for the configured strategy.

    Two paths today:
      - "v3-all-wider-4": evaluator returns (side, sl_dist, tp_dist, dbg) — SL/TP
        are already in price units from ATR×k math.
      - default ("multifactor-v1"): evaluator returns (side, dbg); SL/TP come
        from fixed-pct multipliers in params (`sl_pct`, `tp_pct`) applied to
        the close price.

    The price fallback (`float(df["Close"].iloc[-1])`) handles the case where
    a strategy returns a non-dict debug (legacy path, kept for safety).
    """
    fallback_price = float(bars_15m["Close"].iloc[-1])

    if strategy_name == "v3-all-wider-4":
        side, sl_dist, tp_dist, dbg = evaluate_signal_v3all_wider4(
            bars_15m, funding_rate, params)
        price = (dbg.get("cur_close", fallback_price)
                 if isinstance(dbg, dict) else fallback_price)
        return SignalDecision(
            side=side, price=price,
            sl_distance=float(sl_dist), tp_distance=float(tp_dist),
            debug=dbg if isinstance(dbg, dict) else {},
        )

    # multifactor-v1 (and any future fixed-pct variant)
    side, dbg = evaluate_signal(bars_15m, funding_rate, params)
    price = (dbg.get("cur_close", fallback_price)
             if isinstance(dbg, dict) else fallback_price)
    sl_pct = float(params["strategy"]["sl_pct"])
    tp_pct = float(params["strategy"]["tp_pct"])
    return SignalDecision(
        side=side, price=price,
        sl_distance=sl_pct * price, tp_distance=tp_pct * price,
        debug=dbg if isinstance(dbg, dict) else {},
    )
