"""
TurnOfCandle15m — Caporale, Plastun & Oliinyk (Heliyon 2023, 9(3) e14077)
"Turn-of-the-candle effect in bitcoin returns".

Signal (paper Table 6, momentum-continuation variant)
-----------------------------------------------------
At minute 0 of each hour (the "turn of the 15m candle" on the most-watched
intraday clock), check the sign of the JUST-CLOSED 15m candle:

    direction = sign(close_prev_15m - open_prev_15m)

If direction == +1 → enter long.
If direction == -1 → enter short.
Hold EXACTLY 1 15m candle. Exit market at the next 15m close (minute 15).

That's it. No vol filter. No regime gate. No tuning. The whole point of v1
is to test whether the published time-of-candle pattern still exists on
2023-2026 BTC perp data — not to make a deployable strategy.

Headline (paper): Sharpe 4.96 vs B&H 0.77, 74.18% p.a. net of 2022 retail
fees, on a sample ending 2022-08. Three years post-publication + 4x higher
realistic costs (~15-20bps RT on Binance Futures 2026) means the gate-4
PSR-at-15bps test is the load-bearing kill switch — if costs eat the edge,
no parameter rescue is possible.

Why this file is intentionally minimal
--------------------------------------
The paper tested 48 combos (4 turn minutes × 4 direction rules × 3 holds);
in-window optimisation on a published+crowded signal would overfit a dead
edge. We pin to the headline configuration (turn=0 of hour, prior-candle
sign, hold=1 bar). Alt-aggressive variant (turn at all 4 of {0,15,30,45})
is gated behind `trigger_minutes` for the gate-4 cost-stress decision.

Sizing
------
Repo convention: risk-per-trade-pct + leverage cap, integer units. Under
the harness's PRICE_SCALE pattern, 1 unit == 0.001 BTC (Binance USDT-M
qty_step). No stop-loss in v1 — exit is purely time-based (hold 1 15m
candle). The `stop_atr_mult` / `take_profit_atr_mult` fields exist on the
scope only for the gate-4 stress sweep; v1 ignores them.

Authority: research-only. Not wired to bot.py, not in live deploy, not in
backtest.py STRATEGIES dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


class TurnOfCandle15m(Strategy):
    """Caporale/Plastun/Oliinyk 2023 turn-of-the-candle, hour-only variant."""

    # --- Signal parameters (paper headline, NOT tunable in v1) ---
    # Minute of the hour at which entries are evaluated. Paper headline:
    # minute 0 of each hour (one of the four 15m boundaries). Aggressive
    # variant (all four 15m boundaries) would set this to (0, 15, 30, 45),
    # but per spec note (2) we START with hour-only; gate-4 cost stress
    # decides whether the 4x-turnover version is even worth trying.
    trigger_minutes: tuple[int, ...] = (0,)

    # Hold period in 15m bars. Paper headline = 1 (exit at next 15m close).
    hold_bars_15m: int = 1

    # --- Direction control ---
    allow_shorts: bool = True

    # --- Sizing (repo convention; risk-based, leverage-capped) ---
    # Tiny per-trade alpha → tiny per-trade risk. Spec calls for 0.25%
    # risk and 5x leverage; we mirror those.
    risk_per_trade_pct: float = 0.25
    leverage: int = 5

    # ------------------------------------------------------------------ init

    def init(self) -> None:
        # Pre-compute the index once so next() doesn't do per-bar pandas
        # work. Bar-local minute lookups go through self._minutes /
        # self._hours int arrays.
        idx = self.data.index
        if isinstance(idx, pd.DatetimeIndex):
            self._minutes = idx.minute.values.astype(np.int32)
            self._hours = idx.hour.values.astype(np.int32)
        else:
            # Fallback for non-DatetimeIndex inputs (shouldn't happen with
            # the standard 15m parquet loader, but defensive).
            arr = pd.DatetimeIndex(idx)
            self._minutes = arr.minute.values.astype(np.int32)
            self._hours = arr.hour.values.astype(np.int32)

        # Trigger minute set for O(1) membership test in next().
        self._trigger_set = set(int(m) for m in self.trigger_minutes)

        # Bar index of the most recent entry. Used to enforce hold_bars_15m.
        self._entry_bar: int | None = None

    # ------------------------------------------------------------------ sizing

    def _position_units(self, price: float) -> int:
        """Integer position size in scaled-BTC units.

        Under PRICE_SCALE pattern, 1 returned unit == 0.001 BTC, matching
        Binance USDT-M perp qty_step. No stop-loss in v1 so we size off
        the leverage cap directly — the time-stop is the only exit.
        """
        if price <= 0 or not np.isfinite(price):
            return 0
        # Cap by leverage * 0.95 safety margin.
        max_btc = (self.equity * self.leverage * 0.95) / price
        # Tiny edge → tiny size. Use risk_pct as a notional cap as well
        # (risk_pct of equity as notional, bounded above by leverage cap).
        risk_notional = self.equity * (self.risk_per_trade_pct / 100.0) * self.leverage
        notional_btc = risk_notional / price
        # Use the larger of the two so we still get a non-zero position
        # even when risk_pct is small; capped by leverage.
        size_btc = min(max(notional_btc, 1.0), max_btc)
        return max(int(size_btc), 0)

    # ------------------------------------------------------------------ loop

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = float(self.data.Close[-1])

        # --- Exit-first: enforce time-based hold. ---
        if self.position and self._entry_bar is not None:
            bars_held = i - self._entry_bar
            if bars_held >= self.hold_bars_15m:
                self.position.close()
                self._entry_bar = None
                # Don't re-enter on the SAME bar we exit. Paper exits at
                # minute 15 and (in the hour-only variant) re-evaluates
                # at minute 0 of the next hour — so no same-bar flip.
                return
            # Still inside the hold window — let it ride.
            return

        # --- Entry: only at configured trigger minute(s). ---
        # Need at least 1 prior bar to read the just-closed candle's
        # open/close. i == 0 has no prior bar.
        if i < 1:
            return
        minute = int(self._minutes[i])
        if minute not in self._trigger_set:
            return

        # Direction = sign of the JUST-CLOSED 15m candle (paper rule).
        # Bar at i is the bar OPENING at this minute. We want the bar
        # that just closed — that's bar i-1.
        prev_open = float(self.data.Open[-2])
        prev_close = float(self.data.Close[-2])
        if not (np.isfinite(prev_open) and np.isfinite(prev_close)):
            return
        delta = prev_close - prev_open
        if delta == 0.0:
            # Doji — paper specifies sign; flat candle has no signal.
            return

        units = self._position_units(close_v)
        if units <= 0:
            return

        if delta > 0:
            self.buy(size=units)
            self._entry_bar = i
        elif self.allow_shorts and delta < 0:
            self.sell(size=units)
            self._entry_bar = i
