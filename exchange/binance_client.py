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
