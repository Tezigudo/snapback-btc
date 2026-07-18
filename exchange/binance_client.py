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
import re
import time
from dataclasses import dataclass
from typing import Any

import ccxt
import pandas as pd

from .env import get_api_credentials, get_env

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 15_000

# clientOrderId scheme that lets investing-consolidate attribute trades back
# to this bot. Format: snap-<version>-<signal_id>-<leg>
#   version  : strategy version, e.g. "v1"
#   signal_id: millisecond epoch when the bot decided on the signal
#              (passed in by bot.py — anchors all 3 legs of a trade)
#   leg      : "e" entry, "s" stop-loss, "t" take-profit,
#              "x" time-stop close, "ce" channel-exit close,
#              "bf" boot-flatten, "h" HALT close, "k" kill-switch close
# Binance allows alnum + ._-, max 36 chars. snap-v1-<13 digit ms>-tx = 24 chars.
# Donchian leg uses prefix "snap-d3-" to distinguish in fills/logs.
_COID_PREFIX_DEFAULT = "snap-v1-"
_COID_VALID = re.compile(r"^[A-Za-z0-9._\-]{1,36}$")


def _coid(root: str | None, leg: str, prefix: str = _COID_PREFIX_DEFAULT) -> str | None:
    """Build a Binance-safe clientOrderId. Returns None if root is None.

    Returns None (not a raised error) on invalid inputs so callers can fall
    back to an untagged order placement gracefully.
    """
    if root is None:
        return None
    coid = f"{prefix}{root}-{leg}"
    return coid if _COID_VALID.fullmatch(coid) else None


@dataclass
class Position:
    symbol: str
    side: str        # "long" | "short" | "flat"
    qty: float       # absolute BTC qty (>= 0)
    entry_price: float
    unrealized_pnl: float
    margin_used: float


class BinanceClient:
    """Thin facade over ccxt.binanceusdm with bot-friendly methods.

    Hedge mode + position_side support: when running two bots on one account,
    each instance MUST pass its `position_side` ("LONG" or "SHORT") to the
    exchange so Binance tracks both legs independently. The bot reads
    `hedge.enabled` from params.yaml and constructs the client accordingly.
    """

    def __init__(self, ex: ccxt.Exchange, env: str,
                 hedge_mode: bool = False,
                 coid_prefix: str = _COID_PREFIX_DEFAULT) -> None:
        self.ex = ex
        self.env = env
        self.hedge_mode = hedge_mode
        self.coid_prefix = coid_prefix

    @classmethod
    def from_env(cls, hedge_mode: bool = False,
                 coid_prefix: str = _COID_PREFIX_DEFAULT) -> BinanceClient:
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
        if hedge_mode:
            log.info("BinanceClient: HEDGE MODE on (positionSide will be sent on every order)")
        return cls(ex=ex, env=env, hedge_mode=hedge_mode, coid_prefix=coid_prefix)

    def _position_side(self, side: str) -> str | None:
        """Return positionSide for Binance Futures hedge mode.

        side: "long" | "short" — direction of the trade we're opening.
        Returns "LONG" / "SHORT" when hedge mode is on; None otherwise (one-way
        mode — Binance rejects positionSide when account is in one-way mode).
        """
        if not self.hedge_mode:
            return None
        return "LONG" if side == "long" else "SHORT"

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
    def cancel_open_orders(self, symbol: str,
                           coid_prefix: str | None = None) -> int:
        """Cancel open orders for `symbol`.

        When `coid_prefix` is given, only orders whose clientOrderId starts
        with that prefix are cancelled — manual orders and orders from other
        bot legs are left untouched.  When `coid_prefix` is None, all open
        orders for the symbol are cancelled (backward-compatible behaviour).

        Returns the count of orders actually cancelled.
        """
        try:
            orders = self.ex.fetch_open_orders(symbol=symbol)
        except Exception as e:
            log.warning("cancel_open_orders fetch failed: %s", e)
            return 0
        n = 0
        for o in orders:
            if coid_prefix is not None:
                # ccxt may surface clientOrderId in the top-level field or nested
                # inside the raw `info` dict — check both. External adapters can
                # also return a missing or non-string id; treat those as
                # non-matching rather than letting `.startswith` raise and abort
                # the whole sweep.
                coid = o.get("clientOrderId") or (o.get("info") or {}).get("clientOrderId")
                if not isinstance(coid, str) or not coid.startswith(coid_prefix):
                    continue
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

    def _create_order_with_coid_retry(
        self, symbol: str, order_type: str, side: str, qty: float,
        price: float | None, params: dict[str, Any],
    ) -> dict[str, Any]:
        """create_order wrapper that retries without newClientOrderId if the
        exchange rejects the COID specifically.

        Causes of COID rejection in practice: duplicate (effectively impossible
        with ms-precision signal_id), invalid chars, length > 36, or Binance
        silently tightening their regex. We never want a tagging failure to
        prevent a real trade from being placed.
        """
        coid = params.get("newClientOrderId")
        try:
            return self.ex.create_order(symbol, order_type, side, qty, price, params=params)
        except (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.OperationFailed) as e:
            msg = str(e).lower()
            if coid and ("client" in msg or "newclientorderid" in msg or "duplicate" in msg):
                log.warning("clientOrderId %r rejected (%s) — retrying untagged", coid, e)
                params_clean = {k: v for k, v in params.items() if k != "newClientOrderId"}
                return self.ex.create_order(symbol, order_type, side, qty, price, params=params_clean)
            raise

    def market_order_with_bracket(
        self, symbol: str, side: str, qty: float,
        sl_price: float, tp_price: float,
        client_order_id_root: str | None = None,
        place_tp: bool = True,
    ) -> dict[str, Any]:
        """Place market entry + stop-market SL (+ optional take-profit-market TP).

        Returns {"entry": order, "sl": order, "tp": order}.
        Brackets are reduce-only and trigger off mark price.

        place_tp=False places entry + SL ONLY and returns "tp": None — for
        strategies that manage their own exit (donchian-v3 closes on the live
        Donchian channel cross, so it has no TP leg). `tp_price` is ignored when
        place_tp is False. Default True preserves the v1/multifactor behaviour
        byte-for-byte.

        Hedge mode: when self.hedge_mode is True, every leg includes the
        position_side matching the entry (LONG for long entries, SHORT for
        shorts) — even reduce-only legs. Binance uses positionSide to identify
        which position SLOT the order touches, not the order direction.

        If `client_order_id_root` is provided, each leg gets a Binance
        clientOrderId of form {prefix}<root>-{e|s|t} so consolidate-investment
        can attribute the trade. See `_coid` for the scheme.
        """
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            raise ValueError(f"qty rounded to <= 0 for {symbol}")
        ccxt_side = "buy" if side == "long" else "sell"
        bracket_side = "sell" if side == "long" else "buy"
        pos_side = self._position_side(side)   # LONG/SHORT/None

        entry_params: dict[str, Any] = {"reduceOnly": False}
        if (coid := _coid(client_order_id_root, "e", self.coid_prefix)):
            entry_params["newClientOrderId"] = coid
        if pos_side is not None:
            entry_params["positionSide"] = pos_side
        entry = self._create_order_with_coid_retry(
            symbol, "market", ccxt_side, qty, None, entry_params)

        sl_params: dict[str, Any] = {"stopPrice": float(sl_price), "reduceOnly": True,
                                      "workingType": "MARK_PRICE"}
        if (coid := _coid(client_order_id_root, "s", self.coid_prefix)):
            sl_params["newClientOrderId"] = coid
        if pos_side is not None:
            sl_params["positionSide"] = pos_side
        sl = self._create_order_with_coid_retry(
            symbol, "STOP_MARKET", bracket_side, qty, None, sl_params)

        tp = None
        if place_tp:
            tp_params: dict[str, Any] = {"stopPrice": float(tp_price), "reduceOnly": True,
                                          "workingType": "MARK_PRICE"}
            if (coid := _coid(client_order_id_root, "t", self.coid_prefix)):
                tp_params["newClientOrderId"] = coid
            if pos_side is not None:
                tp_params["positionSide"] = pos_side
            tp = self._create_order_with_coid_retry(
                symbol, "TAKE_PROFIT_MARKET", bracket_side, qty, None, tp_params)
        return {"entry": entry, "sl": sl, "tp": tp}

    def _place_brackets(
        self, symbol: str, side: str, qty: float,
        fill_price: float, sl_distance: float, tp_distance: float,
        client_order_id_root: str | None = None,
        place_tp: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Place SL (+ optional TP) brackets after a fill. place_tp=False
        returns (sl, None) and skips the TP leg — see market_order_with_bracket."""
        bracket_side = "sell" if side == "long" else "buy"
        pos_side = self._position_side(side)
        if side == "long":
            sl_price = fill_price - sl_distance
            tp_price = fill_price + tp_distance
        else:
            sl_price = fill_price + sl_distance
            tp_price = fill_price - tp_distance
        sl_params: dict[str, Any] = {"stopPrice": float(sl_price), "reduceOnly": True,
                                      "workingType": "MARK_PRICE"}
        if (coid := _coid(client_order_id_root, "s", self.coid_prefix)):
            sl_params["newClientOrderId"] = coid
        if pos_side is not None:
            sl_params["positionSide"] = pos_side
        sl = self._create_order_with_coid_retry(
            symbol, "STOP_MARKET", bracket_side, qty, None, sl_params)
        tp = None
        if place_tp:
            tp_params: dict[str, Any] = {"stopPrice": float(tp_price), "reduceOnly": True,
                                          "workingType": "MARK_PRICE"}
            if (coid := _coid(client_order_id_root, "t", self.coid_prefix)):
                tp_params["newClientOrderId"] = coid
            if pos_side is not None:
                tp_params["positionSide"] = pos_side
            tp = self._create_order_with_coid_retry(
                symbol, "TAKE_PROFIT_MARKET", bracket_side, qty, None, tp_params)
        return sl, tp

    def limit_order_with_bracket(
        self, symbol: str, side: str, qty: float,
        limit_price: float, sl_distance: float, tp_distance: float,
        timeout_s: float = 20.0, poll_s: float = 2.0,
        client_order_id_root: str | None = None,
        place_tp: bool = True,
    ) -> dict[str, Any]:
        """Place a maker-style limit entry; fall back to market if not filled.

        place_tp=False places entry + SL ONLY (returned "tp" is None) for
        channel-exit strategies; `tp_distance` is ignored then. Default True
        preserves the v1/multifactor behaviour byte-for-byte.

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
        pos_side = self._position_side(side)

        entry_params: dict[str, Any] = {"reduceOnly": False, "timeInForce": "GTC"}
        if (coid := _coid(client_order_id_root, "e", self.coid_prefix)):
            entry_params["newClientOrderId"] = coid
        if pos_side is not None:
            entry_params["positionSide"] = pos_side
        entry = self._create_order_with_coid_retry(
            symbol, "limit", ccxt_side, qty, float(limit_price), entry_params)
        order_id = entry["id"]
        log.info("limit %s placed id=%s qty=%.6f @ %.2f (timeout=%.0fs, coid=%s)",
                 ccxt_side, order_id, qty, limit_price, timeout_s,
                 entry_params.get("newClientOrderId", "—"))

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
                sl, tp = self._place_brackets(
                    symbol, side, filled_qty, avg_price, sl_distance, tp_distance,
                    client_order_id_root=client_order_id_root, place_tp=place_tp)
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
            sl, tp = self._place_brackets(
                symbol, side, filled_qty, avg_price, sl_distance, tp_distance,
                client_order_id_root=client_order_id_root, place_tp=place_tp)
            return {"entry": entry, "sl": sl, "tp": tp, "filled_as": "limit",
                    "fill_price": avg_price, "filled_qty": filled_qty}

        if filled_qty > 0:
            # Partial fill — bracket what we got, do NOT fall back to market
            # (would double the position size). Caller can log the size mismatch.
            partial_qty = self._round_qty(symbol, filled_qty)
            log.warning("limit partially filled %.6f / %.6f — bracketing partial, "
                        "skipping market fallback to avoid position doubling",
                        partial_qty, qty)
            sl, tp = self._place_brackets(
                symbol, side, partial_qty, avg_price, sl_distance, tp_distance,
                client_order_id_root=client_order_id_root, place_tp=place_tp)
            return {"entry": entry, "sl": sl, "tp": tp, "filled_as": "limit_partial",
                    "fill_price": avg_price, "filled_qty": partial_qty}

        # Zero fill — fall back to market for full qty. Reuse the same
        # client_order_id_root with leg "e" — there's only ever ONE entry
        # per signal_id, so this is unambiguous from the importer's view.
        log.info("limit unfilled after %.0fs — falling back to market", elapsed)
        mkt_params: dict[str, Any] = {"reduceOnly": False}
        if (coid := _coid(client_order_id_root, "e", self.coid_prefix)):
            mkt_params["newClientOrderId"] = coid
        if pos_side is not None:
            mkt_params["positionSide"] = pos_side
        market_entry = self._create_order_with_coid_retry(
            symbol, "market", ccxt_side, qty, None, mkt_params)
        try:
            m = self.ex.fetch_order(market_entry["id"], symbol)
            m_avg = m.get("average") or m.get("price") or limit_price
            market_fill = float(m_avg)
        except Exception:
            market_fill = float(limit_price)
        sl, tp = self._place_brackets(
            symbol, side, qty, market_fill, sl_distance, tp_distance,
            client_order_id_root=client_order_id_root, place_tp=place_tp)
        return {"entry": market_entry, "sl": sl, "tp": tp,
                "filled_as": "market_fallback",
                "fill_price": market_fill, "filled_qty": qty}

    def close_position(
        self, symbol: str,
        client_order_id_root: str | None = None, close_leg: str = "c",
    ) -> dict[str, Any] | None:
        """Flatten via reduce-only market order. No-op if already flat.

        close_leg names the exit reason for tagging:
          - "x"  time-stop
          - "ce" Donchian channel-exit close
          - "bf" boot-flatten (recovering from a stale position)
          - "h"  HALT-triggered close
          - "k"  kill-switch close
          - "c"  generic (fallback)
        If client_order_id_root is None, places the close untagged.
        """
        p = self.fetch_position(symbol)
        if p.side == "flat" or p.qty == 0:
            return None
        ccxt_side = "sell" if p.side == "long" else "buy"
        pos_side = self._position_side(p.side)
        self.cancel_open_orders(symbol, coid_prefix=self.coid_prefix)
        params: dict[str, Any] = {"reduceOnly": True}
        if (coid := _coid(client_order_id_root, close_leg, self.coid_prefix)):
            params["newClientOrderId"] = coid
        if pos_side is not None:
            params["positionSide"] = pos_side
        return self._create_order_with_coid_retry(
            symbol, "market", ccxt_side, p.qty, None, params)

    def __repr__(self) -> str:
        return f"BinanceClient(env={self.env!r}, api_key='***')"
