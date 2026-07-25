"""Pure-function helpers used by bot.py — extracted to keep the trading
loop focused on orchestration rather than dispatch and arithmetic.

No side effects: nothing here touches the exchange, state.db, alerts, or
consolidate. The Bot class calls these to make decisions, then handles
the I/O itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy.live_cnh_hybrid_short import evaluate_signal_cnh_hybrid_short

# channel_exit_signal is re-exported here on purpose: bot.py imports it from
# bot_internals (byte-identical between main and droplet) rather than adding a
# new import to the heavily-diverging exchange/env/risk import block — keeps the
# eventual droplet cherry-pick's bot.py diff inside non-diverging regions.
from strategy.live_donchian_v3 import (  # noqa: F401  (channel_exit_signal re-exported for bot.py)
    channel_exit_signal,
    evaluate_signal_donchian_v3,
)
from strategy.live_multifactor_v1 import evaluate_signal
from strategy.live_supertrend import evaluate_signal_supertrend, flip_exit_signal
from strategy.live_v3all_wider4 import evaluate_signal_v3all_wider4


def strategy_uses_channel_exit(strategy_name: str) -> bool:
    """True for strategies that omit the TP bracket leg because a live trend
    exit IS their profit-taking mechanism. Currently only donchian-v3.

    NOT the same question as "does this leg run a trend-exit check each tick" —
    that is `strategy_uses_trend_exit`. supertrend keeps its TP bracket AND runs
    a trend exit, so it is deliberately absent here.

    Every other strategy (v1/multifactor, cnh, v3all) is untouched: it keeps its
    TP bracket.
    """
    return strategy_name == "donchian-v3"


def strategy_uses_trend_exit(strategy_name: str) -> bool:
    """True for strategies whose loop must run a trend-exit check each tick.

    - donchian-v3: Donchian channel cross — its ONLY profit-taking mechanism.
    - supertrend: opposite STDir flip, which closes the position even though a
      TP bracket also exists. Both exits are live at once.

    Kept separate from `strategy_uses_channel_exit` so donchian's TP-omission
    behaviour is unchanged by adding a leg that needs the hook but keeps its TP.
    """
    return strategy_name in ("donchian-v3", "supertrend")


def trend_exit_signal(
    strategy_name: str,
    bars: pd.DataFrame,
    position_side: str,
    params: dict,
) -> tuple[bool, dict]:
    """Dispatch the per-strategy trend exit. Returns (should_exit, debug).

    Thin dispatcher so bot._maybe_trend_exit has one callsite, and the donchian
    path is reached by exactly the same call as before this leg was added.
    """
    if strategy_name == "supertrend":
        return flip_exit_signal(bars, position_side, params)
    return channel_exit_signal(bars, position_side, params)


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

    Three paths today:
      - "v3-all-wider-4": evaluator returns (side, sl_dist, tp_dist, dbg) — SL/TP
        are already in price units from ATR×k math.
      - "donchian-v3": Donchian-cons breakout on 4h bars. Returns (side, sl_dist,
        tp_dist, dbg). SL = 1.5×ATR. The entry places NO TP leg — the live
        Donchian channel cross (bot._maybe_channel_exit / channel_exit_signal)
        closes the trade; tp_dist here is advisory telemetry only.
      - default ("multifactor-v1"): evaluator returns (side, dbg); SL/TP come
        from fixed-pct multipliers in params (`sl_pct`, `tp_pct`) applied to
        the close price.

    The `bars_15m` argument name is historical — for donchian-v3 the bot passes
    4h bars in this slot (entry timeframe from config). Each strategy reads
    whatever its config says to.
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

    if strategy_name == "donchian-v3":
        side, sl_dist, tp_dist, dbg = evaluate_signal_donchian_v3(
            bars_15m, funding_rate, params)
        price = (dbg.get("cur_close", fallback_price)
                 if isinstance(dbg, dict) else fallback_price)
        return SignalDecision(
            side=side, price=price,
            sl_distance=float(sl_dist), tp_distance=float(tp_dist),
            debug=dbg if isinstance(dbg, dict) else {},
        )

    if strategy_name == "supertrend":
        # Supertrend flip on native 4h. SL and TP are both real bracket legs
        # (unlike donchian, whose tp_dist is advisory only), and the opposite
        # flip closes the position on top of them via trend_exit_signal.
        side, sl_dist, tp_dist, dbg = evaluate_signal_supertrend(
            bars_15m, funding_rate, params)
        price = (dbg.get("cur_close", fallback_price)
                 if isinstance(dbg, dict) else fallback_price)
        return SignalDecision(
            side=side, price=price,
            sl_distance=float(sl_dist), tp_distance=float(tp_dist),
            debug=dbg if isinstance(dbg, dict) else {},
        )

    if strategy_name == "cnh-hybrid-short-v1":
        # HYBRID short pattern detector on 4h. The bot passes 4h bars in the
        # bars_15m slot (entry timeframe = 4h per params YAML). Returns
        # (side, sl_dist, tp_dist, dbg) — SL = sl_atr_mult × ATR(14, 4h),
        # TP = distance from entry to the configured EMA (default EMA100).
        side, sl_dist, tp_dist, dbg = evaluate_signal_cnh_hybrid_short(
            bars_15m, funding_rate, params)
        price = (dbg.get("close", fallback_price)
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


def gate_status(strategy_name: str, decision: SignalDecision, params: dict) -> dict:
    """Build a structured 'what's true now, what are we waiting for' snapshot
    from a strategy evaluator's debug output. The bot logs this on every bar
    evaluation and includes it in heartbeat-event payloads pushed to consolidate,
    so the dashboard can answer 'why isn't this firing?' without you SSHing
    into the droplet.

    Returns a dict with stable JSON-serializable shape:
      {
        "strategy": "<name>",
        "would_fire": "long" | "short" | None,
        "values":  {<indicator name>: <numeric>},
        "thresholds": {<threshold name>: <numeric>},
        "gates_long":  {<gate name>: <bool>},
        "gates_short": {<gate name>: <bool>},
        "missing_long":  [<gate name>, ...],
        "missing_short": [<gate name>, ...],
        "waiting_for": "<human-readable summary>",
      }
    """
    dbg = decision.debug or {}
    s = params.get("strategy", {}) if isinstance(params, dict) else {}

    if strategy_name == "multifactor-v1":
        rsi = dbg.get("rsi")
        close = dbg.get("cur_close")
        ema = dbg.get("trend_ema")
        vol_sma = dbg.get("vol_sma")
        cur_vol = dbg.get("cur_vol")
        funding = dbg.get("funding_rate")

        rsi_lt_long = float(s.get("rsi_long_threshold", 40))
        rsi_gt_short = float(s.get("rsi_short_threshold", 70))
        vol_mult = float(s.get("volume_multiple", 2.0))
        funding_extreme = float(s.get("funding_extreme_threshold", 0.0005))
        require_funding = bool(s.get("require_funding_not_extreme", True))

        def _safe_lt(a, b):
            return a is not None and b is not None and a < b
        def _safe_gt(a, b):
            return a is not None and b is not None and a > b

        gates_long = {
            "rsi_oversold":  _safe_lt(rsi, rsi_lt_long),
            "trend_up":      _safe_gt(close, ema),
            "volume_spike":  _safe_gt(cur_vol, vol_mult * vol_sma) if vol_sma else False,
            "funding_ok":    (not require_funding) or (funding is not None and funding <= funding_extreme),
        }
        gates_short = {
            "rsi_overbought": _safe_gt(rsi, rsi_gt_short),
            "trend_down":     _safe_lt(close, ema),
            "volume_spike":   gates_long["volume_spike"],  # same volume rule
            "funding_ok":     (not require_funding) or (funding is not None and funding >= -funding_extreme),
        }
        missing_long  = [k for k, v in gates_long.items()  if not v]
        missing_short = [k for k, v in gates_short.items() if not v]
        vol_ratio = (cur_vol / vol_sma) if (cur_vol is not None and vol_sma) else None
        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "rsi":          float(rsi) if rsi is not None else None,
                "close":        float(close) if close is not None else None,
                "ema200":       float(ema) if ema is not None else None,
                "vol_ratio":    float(vol_ratio) if vol_ratio is not None else None,
                "funding_rate": float(funding) if funding is not None else None,
            },
            "thresholds": {
                "rsi_long_lt":   rsi_lt_long,
                "rsi_short_gt":  rsi_gt_short,
                "vol_multiple":  vol_mult,
                "funding_extreme": funding_extreme,
            },
            "gates_long":  gates_long,
            "gates_short": gates_short,
            "missing_long":  missing_long,
            "missing_short": missing_short,
            "waiting_for": _format_waiting(missing_long, missing_short, decision.side),
        }

    if strategy_name == "donchian-v3":
        close = dbg.get("cur_close")
        upper = dbg.get("upper")
        lower = dbg.get("lower")
        slope = dbg.get("slope")
        slope_thr = (dbg.get("slope_threshold")
                     if dbg.get("slope_threshold") is not None
                     else float(s.get("slope_trend_threshold_pct", 0.03)))
        gate_on = bool(dbg.get("gate_on", True))

        breakout_ok = close is not None and upper is not None and close > upper
        breakdown_ok = close is not None and lower is not None and close < lower
        slope_long_ok  = (not gate_on) or (slope is not None and slope >=  slope_thr)
        slope_short_ok = (not gate_on) or (slope is not None and slope <= -slope_thr)

        gates_long  = {"breakout_above_80bar": breakout_ok,  "slope_up":   slope_long_ok}
        gates_short = {"breakdown_below_80bar": breakdown_ok, "slope_down": slope_short_ok}
        missing_long  = [k for k, v in gates_long.items()  if not v]
        missing_short = [k for k, v in gates_short.items() if not v]
        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "close":      float(close) if close is not None else None,
                "upper_80bar": float(upper) if upper is not None else None,
                "lower_80bar": float(lower) if lower is not None else None,
                "slope":       float(slope) if slope is not None else None,
            },
            "thresholds": {
                "slope_threshold": float(slope_thr),
                "gate_on":         gate_on,
            },
            "gates_long":  gates_long,
            "gates_short": gates_short,
            "missing_long":  missing_long,
            "missing_short": missing_short,
            "waiting_for": _format_waiting(missing_long, missing_short, decision.side),
        }

    if strategy_name == "supertrend":
        close = dbg.get("cur_close")
        dir_now = dbg.get("st_dir")
        dir_prev = dbg.get("st_dir_prev")
        allow_shorts = bool(dbg.get("allow_shorts", True))

        flip_long = dir_prev == -1.0 and dir_now == 1.0
        flip_short = dir_prev == 1.0 and dir_now == -1.0
        gates_long = {"st_flip_up": bool(flip_long)}
        gates_short = {"st_flip_down": bool(flip_short),
                       "shorts_enabled": allow_shorts}
        missing_long = [k for k, v in gates_long.items() if not v]
        missing_short = [k for k, v in gates_short.items() if not v]
        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "close": float(close) if close is not None else None,
                "st_dir": float(dir_now) if dir_now is not None else None,
                "st_dir_prev": float(dir_prev) if dir_prev is not None else None,
                "atr": (float(dbg["atr"]) if dbg.get("atr") is not None else None),
            },
            "thresholds": {
                "st_period": float(dbg.get("st_period", s.get("st_period", 14))),
                "st_multiplier": float(dbg.get("st_multiplier",
                                               s.get("st_multiplier", 3.5))),
                "sl_atr": float(dbg.get("sl_atr", s.get("st_sl_atr", 2.0))),
                "tp_atr": float(dbg.get("tp_atr", s.get("st_tp_atr", 10.0))),
                "allow_shorts": allow_shorts,
            },
            "gates_long": gates_long,
            "gates_short": gates_short,
            "missing_long": missing_long,
            "missing_short": missing_short,
            "waiting_for": _format_waiting(missing_long, missing_short, decision.side),
        }

    if strategy_name == "cnh-hybrid-short-v1":
        close = dbg.get("close")
        ema24 = dbg.get("ema24")
        ema100 = dbg.get("ema100")
        atr_v = dbg.get("atr")
        last_admitted = dbg.get("last_admitted_pattern")
        pattern_fired = dbg.get("pattern")   # "DT" | "ICNH" | None
        cross_down = dbg.get("entry_ema_cross_down")

        # HYBRID is SHORT-only; "would_fire" is short or None.
        tp_slot_ok = (close is not None and ema100 is not None
                      and ema100 < close)
        # "Admission actionable this bar" — true for DT when admission happened
        # at the current bar (the only window DT can fire), and true for ICnH
        # whenever the evaluator chose to fire (admission is in the lookback +
        # cross-down just happened). The earlier `last_admitted.ts == dbg.ts`
        # check was always False for ICnH entries because ICnH admits at the
        # handle-end bar, not the cross-down bar — so the dashboard reported
        # "waiting on pattern_admitted_this_bar" while simultaneously firing.
        gates_short = {
            "pattern_admitted_this_bar": bool(
                last_admitted is not None
                and (
                    last_admitted.get("ts") == dbg.get("ts")
                    or pattern_fired == "ICNH"
                )
            ),
            "tp_slot_below_entry":  tp_slot_ok,
            "icnh_lookback_ema_xd": bool(cross_down) if cross_down is not None else False,
        }
        # The 3rd gate is only meaningful when ICnH is the pending pattern.
        missing_short = [k for k, v in gates_short.items() if not v]
        return {
            "strategy": strategy_name,
            "would_fire": decision.side,
            "values": {
                "close":  float(close) if close is not None else None,
                "ema24":  float(ema24) if ema24 is not None else None,
                "ema100": float(ema100) if ema100 is not None else None,
                "atr":    float(atr_v) if atr_v is not None else None,
                "last_admitted_pattern": last_admitted,
                "pattern_fired": pattern_fired,
            },
            "thresholds": {
                "sl_atr_mult": float(s.get("sl_atr_mult", 1.5)),
                "tp_ema":      (s.get("tp_emas", ["ema100"]) or ["ema100"])[0],
                "dedup_bars":  int(s.get("dedup_bars", 15)),
            },
            # HYBRID has no long side — keep gates_long empty for shape compat.
            "gates_long":  {},
            "gates_short": gates_short,
            "missing_long":  [],
            "missing_short": missing_short,
            "waiting_for": _format_waiting([], missing_short, decision.side),
        }

    # Unknown strategy — return minimal envelope so the dashboard doesn't blow up.
    return {
        "strategy": strategy_name,
        "would_fire": decision.side,
        "values": {},
        "thresholds": {},
        "gates_long": {},
        "gates_short": {},
        "missing_long": [],
        "missing_short": [],
        "waiting_for": "no gate introspection available for this strategy",
    }


def _format_waiting(missing_long: list, missing_short: list, fired: str | None) -> str:
    if fired:
        return f"signal fired: {fired}"
    long_part = "long ready" if not missing_long else "long waiting on " + ", ".join(missing_long)
    short_part = "short ready" if not missing_short else "short waiting on " + ", ".join(missing_short)
    return f"{long_part} | {short_part}"


def order_avg_price(order: dict | None) -> float | None:
    """Best-effort average fill price from a ccxt order dict (market fills).
    Returns None if no positive price is present."""
    if not order:
        return None
    info = order.get("info") or {}
    for v in (order.get("average"), info.get("avgPrice"), order.get("price")):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def reduce_only_bracket_leg(order: dict) -> str | None:
    """Classify a ccxt order as a reduce-only bracket leg: 'sl' (STOP*), 'tp'
    (TAKE_PROFIT*), or None (not a reduce-only stop/take-profit — e.g. an
    unfilled limit entry). Handles reduceOnly/closePosition as bool or string."""
    info = order.get("info") or {}
    reduce_only = (
        str(info.get("reduceOnly")).lower() == "true"
        or str(info.get("closePosition")).lower() == "true"
    )
    if not reduce_only:
        return None
    otype = str(info.get("type") or info.get("origType") or order.get("type") or "").upper()
    if "TAKE_PROFIT" in otype:
        return "tp"
    if "STOP" in otype:
        return "sl"
    return None


def bracket_is_intact(open_orders: list[dict], place_tp: bool) -> bool:
    """True when the resting reduce-only bracket is complete for an open
    position: a stop (SL) leg is present, plus a take-profit (TP) leg when the
    strategy places one (place_tp). Pure — unit-tested."""
    legs = {reduce_only_bracket_leg(o) for o in (open_orders or [])}
    return "sl" in legs and ("tp" in legs or not place_tp)


def has_bracket_leg(open_orders: list[dict]) -> bool:
    """True if ANY reduce-only bracket leg (SL or TP) is present. Used to verify
    a cancel actually cleared the bracket before re-placing (cancel_open_orders
    swallows per-order failures, so a partial cancel could otherwise leave a
    surviving leg alongside a freshly-placed pair)."""
    return any(reduce_only_bracket_leg(o) is not None for o in (open_orders or []))
