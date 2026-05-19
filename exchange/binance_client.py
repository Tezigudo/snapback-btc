"""Binance USDM Futures client wrapper (ccxt-based, testnet-aware).

Wraps ccxt.binanceusdm with sandbox_mode auto-set when BINANCE_ENV=testnet.
Provides the small surface area the bot loop needs:
  - fetch_ohlcv(symbol, tf, limit)
  - fetch_funding_rate(symbol)
  - fetch_equity_usdt()
  - fetch_position(symbol)
  - cancel_open_orders(symbol)
  - market_order_with_bracket(symbol, side, qty, sl, tp)
  - close_position(symbol)

NEVER call from Claude Code sessions — this is the trading runtime.
Bot uses this; Claude Code is read-only per CLAUDE.md.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import ccxt
import pandas as pd

from .env import get_api_credentials, get_env

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 15_000


@dataclass
class Position:
    symbol: str
    side: str        # "long" | "short" | "flat"
    qty: float       # absolute BTC qty (>= 0)
    entry_price: float
    unrealized_pnl: float
    margin_used: float


class BinanceClient:
    """Thin facade over ccxt.binanceusdm with bot-friendly methods."""

    def __init__(self, ex: ccxt.Exchange, env: str) -> None:
        self.ex = ex
        self.env = env

    @classmethod
    def from_env(cls) -> BinanceClient:
        env = get_env()
        api_key, api_secret = get_api_credentials()
        ex = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": DEFAULT_TIMEOUT_MS,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        })
        if env == "testnet":
            ex.set_sandbox_mode(True)
            log.info("BinanceClient: SANDBOX/TESTNET mode")
        else:
            log.info("BinanceClient: MAINNET mode (real money)")
        ex.load_markets()
        return cls(ex=ex, env=env)

    # --- market data --------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        raw = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.set_index("ts")
        return df

    def fetch_funding_rate(self, symbol: str) -> float:
        """Most-recent funding rate (per 8h, decimal). 0.0 on failure."""
        try:
            r = self.ex.fetch_funding_rate(symbol)
            rate = r.get("fundingRate")
            return float(rate) if rate is not None else 0.0
        except Exception as e:
            log.warning("fetch_funding_rate failed: %s", e)
            return 0.0

    # --- account ------------------------------------------------------------
    def fetch_equity_usdt(self) -> float:
        """Total USDT-denominated equity (wallet + unrealized PnL)."""
        bal = self.ex.fetch_balance({"type": "future"})
        info = bal.get("info", {}) or {}
        try:
            wallet = float(info.get("totalWalletBalance", 0.0))
            unreal = float(info.get("totalUnrealizedProfit", 0.0))
            return wallet + unreal
        except (TypeError, ValueError):
            usdt = bal.get("USDT", {})
            return float(usdt.get("total", 0.0) or 0.0)

    def fetch_position(self, symbol: str) -> Position:
        positions = self.ex.fetch_positions([symbol])
        for p in positions:
            if p.get("symbol") != symbol:
                continue
            qty = float(p.get("contracts") or 0.0)
            if qty == 0:
                continue
            side = "long" if (p.get("side") or "").lower() == "long" else "short"
            return Position(
                symbol=symbol, side=side, qty=abs(qty),
                entry_price=float(p.get("entryPrice") or 0.0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
                margin_used=float(p.get("initialMargin") or 0.0),
            )
        return Position(symbol=symbol, side="flat", qty=0.0,
                        entry_price=0.0, unrealized_pnl=0.0, margin_used=0.0)

    # --- order management ---------------------------------------------------
    def cancel_open_orders(self, symbol: str) -> int:
        try:
            orders = self.ex.fetch_open_orders(symbol=symbol)
        except Exception as e:
            log.warning("cancel_open_orders fetch failed: %s", e)
            return 0
        n = 0
        for o in orders:
            try:
                self.ex.cancel_order(o["id"], symbol)
                n += 1
            except Exception as e:
                log.warning("cancel_order %s failed: %s", o.get("id"), e)
        return n

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.ex.set_leverage(int(leverage), symbol)
        except Exception as e:
            # Binance returns 400 if leverage is already set; safe to swallow.
            log.info("set_leverage(%dx, %s): %s", leverage, symbol, e)

    def _round_qty(self, symbol: str, qty: float) -> float:
        m = self.ex.market(symbol)
        step = m.get("precision", {}).get("amount", 3)
        if isinstance(step, int):
            return round(qty, step)
        try:
            return math.floor(qty / float(step)) * float(step)
        except Exception:
            return round(qty, 3)

    def market_order_with_bracket(
        self, symbol: str, side: str, qty: float,
        sl_price: float, tp_price: float,
    ) -> dict[str, Any]:
        """Place market entry + stop-market SL + take-profit-market TP brackets.

        Returns {"entry": order, "sl": order, "tp": order}.
        Brackets are reduce-only and trigger off mark price.
        """
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            raise ValueError(f"qty rounded to <= 0 for {symbol}")
        ccxt_side = "buy" if side == "long" else "sell"
        bracket_side = "sell" if side == "long" else "buy"

        entry = self.ex.create_order(symbol, "market", ccxt_side, qty, None,
                                     params={"reduceOnly": False})

        sl_params = {"stopPrice": float(sl_price), "reduceOnly": True,
                     "workingType": "MARK_PRICE"}
        sl = self.ex.create_order(symbol, "STOP_MARKET", bracket_side, qty,
                                  None, params=sl_params)

        tp_params = {"stopPrice": float(tp_price), "reduceOnly": True,
                     "workingType": "MARK_PRICE"}
        tp = self.ex.create_order(symbol, "TAKE_PROFIT_MARKET", bracket_side, qty,
                                  None, params=tp_params)
        return {"entry": entry, "sl": sl, "tp": tp}

    def _place_brackets(
        self, symbol: str, side: str, qty: float,
        fill_price: float, sl_distance: float, tp_distance: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bracket_side = "sell" if side == "long" else "buy"
        if side == "long":
            sl_price = fill_price - sl_distance
            tp_price = fill_price + tp_distance
        else:
            sl_price = fill_price + sl_distance
            tp_price = fill_price - tp_distance
        sl = self.ex.create_order(
            symbol, "STOP_MARKET", bracket_side, qty, None,
            params={"stopPrice": float(sl_price), "reduceOnly": True,
                    "workingType": "MARK_PRICE"})
        tp = self.ex.create_order(
            symbol, "TAKE_PROFIT_MARKET", bracket_side, qty, None,
            params={"stopPrice": float(tp_price), "reduceOnly": True,
                    "workingType": "MARK_PRICE"})
        return sl, tp

    def limit_order_with_bracket(
        self, symbol: str, side: str, qty: float,
        limit_price: float, sl_distance: float, tp_distance: float,
        timeout_s: float = 20.0, poll_s: float = 2.0,
    ) -> dict[str, Any]:
        """Place a maker-style limit entry; fall back to market if not filled.

        sl_distance and tp_distance are PRICE units. Brackets are computed AFTER
        the entry fills (based on actual fill price), so a slow limit fill that
        gets a better price than `limit_price` propagates correctly into the
        SL/TP geometry.

        Returns {"entry", "sl", "tp", "filled_as", "fill_price", "filled_qty"}.
        filled_as is one of: "limit", "limit_partial" (some filled, market
        fallback skipped to avoid doubling position), or "market_fallback".
        """
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            raise ValueError(f"qty rounded to <= 0 for {symbol}")
        ccxt_side = "buy" if side == "long" else "sell"

        entry = self.ex.create_order(
            symbol, "limit", ccxt_side, qty, float(limit_price),
            params={"reduceOnly": False, "timeInForce": "GTC"})
        order_id = entry["id"]
        log.info("limit %s placed id=%s qty=%.6f @ %.2f (timeout=%.0fs)",
                 ccxt_side, order_id, qty, limit_price, timeout_s)

        filled_qty = 0.0
        avg_price = float(limit_price)
        elapsed = 0.0
        last_status: str | None = None
        while elapsed < timeout_s:
            time.sleep(poll_s)
            elapsed += poll_s
            try:
                o = self.ex.fetch_order(order_id, symbol)
            except Exception as e:
                log.warning("fetch_order(%s) failed during limit poll: %s", order_id, e)
                continue
            last_status = o.get("status")
            filled_qty = float(o.get("filled") or 0.0)
            avg = o.get("average")
            if avg:
                avg_price = float(avg)
            if last_status in ("closed", "filled") and filled_qty >= qty * 0.999:
                log.info("limit filled fully @ %.2f after %.0fs", avg_price, elapsed)
                sl, tp = self._place_brackets(symbol, side, filled_qty,
                                              avg_price, sl_distance, tp_distance)
                return {"entry": o, "sl": sl, "tp": tp, "filled_as": "limit",
                        "fill_price": avg_price, "filled_qty": filled_qty}
            if last_status in ("canceled", "rejected", "expired"):
                log.warning("limit ended with status=%s, filled_qty=%.6f",
                            last_status, filled_qty)
                break

        # Timed out OR cancelled exchange-side.
        if last_status not in ("canceled", "rejected", "expired"):
            try:
                self.ex.cancel_order(order_id, symbol)
                log.info("cancelled limit after %.0fs (filled=%.6f / %.6f)",
                         elapsed, filled_qty, qty)
            except Exception as e:
                # If the order filled between our last poll and cancel, ccxt
                # raises. Re-fetch to find out what really happened.
                log.info("cancel race: %s — re-fetching order", e)
                try:
                    o = self.ex.fetch_order(order_id, symbol)
                    filled_qty = float(o.get("filled") or 0.0)
                    avg = o.get("average")
                    if avg:
                        avg_price = float(avg)
                except Exception as e2:
                    log.warning("post-cancel fetch_order failed: %s", e2)

        if filled_qty >= qty * 0.999:
            # Filled fully right at the cancel boundary.
            sl, tp = self._place_brackets(symbol, side, filled_qty,
                                          avg_price, sl_distance, tp_distance)
            return {"entry": entry, "sl": sl, "tp": tp, "filled_as": "limit",
                    "fill_price": avg_price, "filled_qty": filled_qty}

        if filled_qty > 0:
            # Partial fill — bracket what we got, do NOT fall back to market
            # (would double the position size). Caller can log the size mismatch.
            partial_qty = self._round_qty(symbol, filled_qty)
            log.warning("limit partially filled %.6f / %.6f — bracketing partial, "
                        "skipping market fallback to avoid position doubling",
                        partial_qty, qty)
            sl, tp = self._place_brackets(symbol, side, partial_qty,
                                          avg_price, sl_distance, tp_distance)
            return {"entry": entry, "sl": sl, "tp": tp, "filled_as": "limit_partial",
                    "fill_price": avg_price, "filled_qty": partial_qty}

        # Zero fill — fall back to market for full qty.
        log.info("limit unfilled after %.0fs — falling back to market", elapsed)
        market_entry = self.ex.create_order(
            symbol, "market", ccxt_side, qty, None,
            params={"reduceOnly": False})
        try:
            m = self.ex.fetch_order(market_entry["id"], symbol)
            m_avg = m.get("average") or m.get("price") or limit_price
            market_fill = float(m_avg)
        except Exception:
            market_fill = float(limit_price)
        sl, tp = self._place_brackets(symbol, side, qty,
                                      market_fill, sl_distance, tp_distance)
        return {"entry": market_entry, "sl": sl, "tp": tp,
                "filled_as": "market_fallback",
                "fill_price": market_fill, "filled_qty": qty}

    def close_position(self, symbol: str) -> dict[str, Any] | None:
        """Flatten via reduce-only market order. No-op if already flat."""
        p = self.fetch_position(symbol)
        if p.side == "flat" or p.qty == 0:
            return None
        ccxt_side = "sell" if p.side == "long" else "buy"
        self.cancel_open_orders(symbol)
        return self.ex.create_order(
            symbol, "market", ccxt_side, p.qty, None,
            params={"reduceOnly": True},
        )

    def __repr__(self) -> str:
        return f"BinanceClient(env={self.env!r}, api_key='***')"
