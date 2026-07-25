"""
Live evaluator for the `supertrend` leg (SOL/USDT perp, native 4h, long+short).

Live counterpart of `strategy/signals_supertrend.py::SupertrendBTC`, the winner
of the round-3 win-rate-blended walk-forward (see SOL_LEG_VERDICT.md). Same
relationship as `live_donchian_v3.py` ↔ `DonchianBreakoutBTCv3`: the backtest
class is unreachable from the trading loop, so its decision rule is ported here
and held to parity by `tools/supertrend_parity.py`.

Entry (mirrors SupertrendBTC.next()'s entry branch exactly):
    long  when STDir flips -1 → +1 on the last CLOSED bar
    short when STDir flips +1 → -1  (only if allow_shorts)
    SL = close ∓ st_sl_atr × ATR(st_atr_period)
    TP = close ± st_tp_atr × ATR(st_atr_period)

Exit: SL/TP are exchange-native bracket legs. The *third* exit — "opposite STDir
flip closes the position even if SL/TP haven't hit" — has no exchange
equivalent, so `flip_exit_signal()` reproduces it and the bot's loop hook fires
it. Unlike donchian-v3, this strategy keeps its TP bracket AND runs a trend
exit; both are live at once.

Parity notes
------------
* `supertrend()` in strategy/indicators.py is causal (bar i uses data ≤ i), and
  everything here reads `.iloc[-1]` / `.iloc[-2]` — the last CLOSED bar and the
  one before it. bot.py drops the still-forming bar before calling in, matching
  the backtest, which only ever sees closed bars.
* The backtest guards on `direction`/`atr` finite AND `prev_direction` finite,
  and needs `len(data) >= 2`. Reproduced below, including the ORDER of guards,
  so a NaN warm-up bar produces "no signal" in both places rather than a crash.
* The backtest reads STDir off a frame built by `attach_supertrend` over the
  WHOLE window. Supertrend is a recursive band, so its value at bar i depends on
  history; bot.py feeds 1500 4h bars (~250 days), far past any seeding
  sensitivity for period ≤ 20, so the last-bar value matches a full-history
  computation. `tools/supertrend_parity.py` verifies this empirically rather
  than trusting the argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import atr, supertrend

# Fallback defaults only — config/params_sol_supertrend.yaml is the source of
# truth. Values are the round-3 walk-forward modal pick.
ST_PERIOD = 14
ST_MULTIPLIER = 3.5
ST_ATR_PERIOD = 14
ST_SL_ATR = 2.0
ST_TP_ATR = 10.0
ALLOW_SHORTS = True


def _cfg(params: dict) -> dict:
    s = params.get("strategy", {}) if isinstance(params, dict) else {}
    return {
        "period": int(s.get("st_period", ST_PERIOD)),
        "multiplier": float(s.get("st_multiplier", ST_MULTIPLIER)),
        "atr_period": int(s.get("st_atr_period", ST_ATR_PERIOD)),
        "sl_atr": float(s.get("st_sl_atr", ST_SL_ATR)),
        "tp_atr": float(s.get("st_tp_atr", ST_TP_ATR)),
        "allow_shorts": bool(s.get("allow_shorts", ALLOW_SHORTS)),
    }


def _st_frame(bars: pd.DataFrame, c: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (STDir, ATR, STLine) computed the same way attach_supertrend does.

    STLine is the trailing band itself. It is not needed for the entry decision
    (the flip is read off STDir) but it IS the number a human needs: it is the
    price level that has to be crossed for the next flip, so the dashboard can
    say "needs +4.1%" instead of only "direction is -1".
    """
    st = supertrend(bars["High"], bars["Low"], bars["Close"],
                    period=c["period"], multiplier=c["multiplier"])
    return (st["direction"],
            atr(bars["High"], bars["Low"], bars["Close"], c["atr_period"]),
            st["supertrend"])


def _bars_since_flip(direction: pd.Series, max_lookback: int = 500) -> int | None:
    """How many CLOSED bars the current STDir has held.

    Answers "is this quiet spell normal or is something stuck?" — the leg
    averages ~12 days between flips, so 3 bars vs 60 bars is the difference
    between "just turned" and "a long trend". None when the series is too short
    or all-NaN.
    """
    if len(direction) < 2:
        return None
    cur = direction.iloc[-1]
    if not np.isfinite(cur):
        return None
    n = 0
    for v in reversed(direction.iloc[-max_lookback:].to_list()):
        if not np.isfinite(v) or v != cur:
            break
        n += 1
    return n


def _warmup_bars(c: dict) -> int:
    # attach_supertrend needs `period` bars for the band and `atr_period` for
    # ATR; +2 because the flip test reads bar -1 AND bar -2.
    return max(c["period"], c["atr_period"]) + 2


def evaluate_signal_supertrend(
    bars_4h: pd.DataFrame,
    funding_rate: float,              # unused; supertrend doesn't gate on funding
    params: dict,
) -> tuple[str | None, float, float, dict]:
    """Evaluate a Supertrend flip entry on the last CLOSED 4h bar.

    Returns (side, sl_distance, tp_distance, debug) — distances in price units,
    matching the donchian-v3 / v3all evaluator contract.
    """
    _ = funding_rate
    c = _cfg(params)

    need = _warmup_bars(c)
    if len(bars_4h) < need:
        return None, float("nan"), float("nan"), {
            "reason": "warmup", "have": len(bars_4h), "need": need}

    direction, atr_s, st_line_s = _st_frame(bars_4h, c)
    dir_now = float(direction.iloc[-1]) if np.isfinite(direction.iloc[-1]) else float("nan")
    dir_prev = float(direction.iloc[-2]) if np.isfinite(direction.iloc[-2]) else float("nan")
    atr_v = float(atr_s.iloc[-1])
    cur_close = float(bars_4h["Close"].iloc[-1])
    st_line = float(st_line_s.iloc[-1]) if np.isfinite(st_line_s.iloc[-1]) else float("nan")

    debug = {
        "cur_close": cur_close, "atr": atr_v,
        "st_dir": dir_now if np.isfinite(dir_now) else None,
        "st_dir_prev": dir_prev if np.isfinite(dir_prev) else None,
        # The band, and how far price must travel to cross it. This is what
        # turns "direction is -1" into an actionable "needs +4.1% for the flip".
        "st_line": st_line if np.isfinite(st_line) else None,
        "dist_to_flip_pct": (
            (st_line / cur_close - 1.0) * 100.0
            if np.isfinite(st_line) and cur_close > 0 else None
        ),
        # ATR in dollars is meaningless without scale; as a % of price it reads
        # directly as "how wide the stop will be".
        "atr_pct": (atr_v / cur_close * 100.0
                    if np.isfinite(atr_v) and cur_close > 0 else None),
        "bars_since_flip": _bars_since_flip(direction),
        "st_period": c["period"], "st_multiplier": c["multiplier"],
        "sl_atr": c["sl_atr"], "tp_atr": c["tp_atr"],
        "allow_shorts": c["allow_shorts"],
    }

    # Backtest guard order: direction + atr finite, then prev_direction finite.
    if not np.isfinite(dir_now) or not np.isfinite(atr_v) or atr_v <= 0:
        return None, float("nan"), float("nan"), {**debug, "reason": "nan_indicators"}
    if not np.isfinite(dir_prev):
        return None, float("nan"), float("nan"), {**debug, "reason": "nan_prev_direction"}

    sl_dist = c["sl_atr"] * atr_v
    tp_dist = c["tp_atr"] * atr_v
    debug["sl_dist"] = sl_dist
    debug["tp_dist"] = tp_dist
    # Absolute bracket prices a LONG firing on THIS bar would get. Present even
    # when no signal fires, so the dashboard can answer "what would the trade
    # look like if it triggered now?" without the reader doing ATR arithmetic.
    # (Short side mirrors: close + sl_dist / close - tp_dist.)
    debug["would_sl_price"] = cur_close - sl_dist
    debug["would_tp_price"] = cur_close + tp_dist

    flipped_long = dir_prev == -1.0 and dir_now == 1.0
    flipped_short = dir_prev == 1.0 and dir_now == -1.0

    if flipped_long:
        # Backtest also requires sl < close, which holds whenever sl_dist > 0.
        if sl_dist <= 0 or cur_close - sl_dist >= cur_close:
            return None, float("nan"), float("nan"), {**debug, "reason": "bad_sl"}
        return "long", sl_dist, tp_dist, {**debug, "reason": "flip_long"}

    if flipped_short:
        if not c["allow_shorts"]:
            return None, float("nan"), float("nan"), {
                **debug, "reason": "short_blocked_allow_shorts_false"}
        if sl_dist <= 0 or cur_close + sl_dist <= cur_close:
            return None, float("nan"), float("nan"), {**debug, "reason": "bad_sl"}
        # A 10×ATR short target below a cheap asset can go negative; the
        # backtest hits an assertion there, live we must simply not trade it.
        if cur_close - tp_dist <= 0:
            return None, float("nan"), float("nan"), {
                **debug, "reason": "short_tp_below_zero"}
        return "short", sl_dist, tp_dist, {**debug, "reason": "flip_short"}

    return None, float("nan"), float("nan"), {**debug, "reason": "no_flip"}


def flip_exit_signal(
    bars_4h: pd.DataFrame,
    position_side: str,
    params: dict,
) -> tuple[bool, dict]:
    """Should an OPEN supertrend position close on an opposite STDir flip?

    Byte-for-byte with SupertrendBTC.next()'s in-position branch:

        long  closes when direction == -1.0
        short closes when direction == +1.0

    Note this is a *level* test, not a transition test — the backtest closes on
    any bar where direction opposes the position, not only the flip bar. If the
    bot missed the flip bar (restart, outage), it still exits on the next tick.
    """
    c = _cfg(params)
    need = _warmup_bars(c)
    if len(bars_4h) < need:
        return False, {"reason": "warmup", "have": len(bars_4h), "need": need}

    direction, atr_s = _st_frame(bars_4h, c)
    dir_now = direction.iloc[-1]
    cur_close = float(bars_4h["Close"].iloc[-1])
    dbg = {
        "cur_close": cur_close,
        "st_dir": float(dir_now) if np.isfinite(dir_now) else None,
        "atr": float(atr_s.iloc[-1]) if np.isfinite(atr_s.iloc[-1]) else None,
        "position_side": position_side,
    }
    if not np.isfinite(dir_now):
        return False, {**dbg, "reason": "nan_direction"}

    if position_side == "long" and float(dir_now) == -1.0:
        return True, {**dbg, "reason": "flip_exit_long"}
    if position_side == "short" and float(dir_now) == 1.0:
        return True, {**dbg, "reason": "flip_exit_short"}
    return False, {**dbg, "reason": "no_flip_exit"}
