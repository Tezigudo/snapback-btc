"""Snapback-BTC live trading daemon.

Strategy: multifactor-v1 (see config/params.yaml).
Modes:
  - LIVE   (default): real orders against the configured exchange env.
  - DRY-RUN (--dry-run or DRY_RUN=1): fetches real market+balance data,
    evaluates signals, logs everything it WOULD do, but never calls create_order.

Binance Futures testnet/sandbox is no longer supported by ccxt for private
endpoints (deprecated late 2025). Run dry-run first against real account to
verify wiring before flipping to live orders.

Loop (every poll_interval_s):
  1. heartbeat: touch data/heartbeat
  2. check data/HALT → if present, flatten + exit
  3. fetch equity → if equity / deploy_start < kill_switch_fraction → HALT
  4. fetch latest 15m bars + funding
  5. if no open position AND signal fires on last closed bar → place bracket
  6. time-stop: close positions older than max_hold_bars
  7. log to logs/bot.jsonl + data/state.db

DOES NOT:
  - run any LLM call
  - place orders unless all gates pass
  - touch mainnet without confirm_mainnet.lock present
  - place orders that fail exchange minimums (min qty / min notional)

USAGE:
  uv run python -m bot --dry-run     # SAFE: no orders placed
  uv run python -m bot                # live (real orders)
  # Ctrl+C to stop cleanly. Or: touch data/HALT in another terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from alerts import send_alert
from exchange import state
from exchange.binance_client import BinanceClient
from exchange.constraints import (
    DEFAULT_CONSTRAINTS,
    ExchangeConstraints,
    merge_with_live,
    passes_minimums,
    round_qty_down,
)
from exchange.env import REPO_ROOT, get_env, is_halted
from risk import (
    RiskBreach,
    check_leverage,
    check_notional,
    check_symbol,
)
from strategy.indicators import ema, rsi, sma
from strategy.live_v3all_wider4 import evaluate_signal_v3all_wider4

LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "bot.jsonl"
HEARTBEAT = REPO_ROOT / "data" / "heartbeat"

# Console logs display in Bangkok time (GMT+7) for human readability.
# JSONL `ts` field and state.db remain UTC for alignment with Binance candles.
LOCAL_TZ = ZoneInfo("Asia/Bangkok")


class _LocalTimeFormatter(logging.Formatter):
    """Formats `%(asctime)s` in Asia/Bangkok local time."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=LOCAL_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S %z")


def _setup_logging(level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("snapback")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_LocalTimeFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(sh)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, default=str)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    return logger


def load_params() -> dict:
    path = REPO_ROOT / "config" / "params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# --- Signal evaluation (pure function over bars) -----------------------------
def evaluate_signal(bars_15m: pd.DataFrame, funding_rate: float, params: dict
                    ) -> tuple[str | None, dict]:
    """Return ('long'/'short'/None, debug_dict) for the last CLOSED 15m bar.

    Pure-function port of DayTradeMultiFactorBTC._long_signal/_short_signal.
    """
    s = params["strategy"]
    warm = max(s["mf_trend_ema_period"], s["volume_ma_period"], s["rsi_period"]) + 5
    if len(bars_15m) < warm:
        return None, {"reason": "warmup"}

    close = bars_15m["Close"]
    vol = bars_15m["Volume"]

    rsi_v = rsi(close, s["rsi_period"]).iloc[-1]
    vol_sma_v = sma(vol, s["volume_ma_period"]).iloc[-1]
    trend_ema_v = ema(close, s["mf_trend_ema_period"]).iloc[-1]
    cur_vol = vol.iloc[-1]
    cur_close = close.iloc[-1]

    if not all(np.isfinite([rsi_v, vol_sma_v, trend_ema_v, cur_vol, cur_close])):
        return None, {"reason": "nan_indicators"}

    vol_ok = cur_vol > s["volume_multiple"] * vol_sma_v
    trend_up = cur_close > trend_ema_v
    funding_long_blocked = (s["require_funding_not_extreme"]
                            and funding_rate > s["funding_extreme_threshold"])
    funding_short_blocked = (s["require_funding_not_extreme"]
                             and funding_rate < -s["funding_extreme_threshold"])

    debug = {
        "ts": bars_15m.index[-1].isoformat(),
        "rsi": float(rsi_v), "vol_sma": float(vol_sma_v), "trend_ema": float(trend_ema_v),
        "cur_vol": float(cur_vol), "cur_close": float(cur_close),
        "vol_ok": bool(vol_ok), "trend_up": bool(trend_up),
        "funding_rate": funding_rate,
    }

    if (rsi_v < s["rsi_long_threshold"] and vol_ok and trend_up
            and not funding_long_blocked):
        return "long", debug
    if (rsi_v > s["rsi_short_threshold"] and vol_ok and not trend_up
            and not funding_short_blocked):
        return "short", debug
    return None, debug


def compute_qty(equity: float, price: float, sl_pct: float,
                risk_pct: float, leverage: int) -> float:
    """Risk-based sizing matching backtest: target_btc = risk / sl_distance.

    Kept for backward-compat with multifactor-v1 (fixed-pct SL).
    """
    if price <= 0 or sl_pct <= 0:
        return 0.0
    return compute_qty_from_distance(equity, price, sl_pct * price, risk_pct, leverage)


def compute_qty_from_distance(equity: float, price: float, sl_distance: float,
                              risk_pct: float, leverage: int) -> float:
    """Sizing with sl_distance in PRICE units (works for ATR-based stops)."""
    if price <= 0 or sl_distance <= 0:
        return 0.0
    risk_amount = equity * (risk_pct / 100.0)
    target = risk_amount / sl_distance
    cap = (equity * leverage * 0.95) / price
    return max(min(target, cap), 0.0)


# --- Main loop ---------------------------------------------------------------
class Bot:
    def __init__(self, params: dict, dry_run: bool = False) -> None:
        self.params = params
        self.symbol = params["symbol"]
        self.dry_run = dry_run
        self.log = logging.getLogger("snapback.bot")
        self.client = BinanceClient.from_env()
        self.poll_s = float(params["execution"]["poll_interval_s"])
        self.leverage = int(params["sizing"]["leverage"])
        self.kill_fraction = float(params["deploy"]["kill_switch_equity_fraction"])
        # Min-capital warning threshold (read from params, default $100).
        self.min_capital_warn = float(params.get("deploy", {}).get("min_capital_warn_usdt", 100.0))
        self.constraints: ExchangeConstraints = DEFAULT_CONSTRAINTS
        self._stopped = False
        self._last_signal_ts: pd.Timestamp | None = None
        if self.dry_run:
            self.log.warning("DRY-RUN MODE: no real orders will be placed")

    def boot(self) -> None:
        check_symbol(self.symbol)
        check_leverage(self.leverage)

        # Load live exchange constraints (min qty, min notional). Tighter
        # of hard-coded fallbacks vs live values wins.
        try:
            live_market = self.client.ex.market(self.symbol)
            self.constraints = merge_with_live(DEFAULT_CONSTRAINTS, live_market)
            self.log.info("Exchange constraints: min_qty=%s BTC, min_notional=$%s",
                          self.constraints.min_qty_btc, self.constraints.min_notional_usdt)
        except Exception as e:
            self.log.warning("Could not fetch live market constraints, using defaults: %s", e)

        if not self.dry_run:
            self.client.set_leverage(self.symbol, self.leverage)

        state.init_db()

        equity = self.client.fetch_equity_usdt()
        if equity < self.min_capital_warn:
            self.log.warning("LOW CAPITAL: equity $%.2f < $%.2f recommended. "
                             "Some signals may be skipped due to exchange minimums.",
                             equity, self.min_capital_warn)

        start_eq = state.get_float("deploy_start_equity", 0.0)
        if start_eq <= 0:
            state.set_float("deploy_start_equity", equity)
            state.set_meta("deploy_start_ts", datetime.now(timezone.utc).isoformat())
            self.log.info("Recorded deploy_start_equity=%.2f USDT", equity)
            mode = "DRY-RUN" if self.dry_run else "LIVE"
            send_alert(f"Bot deploy start [{mode}]",
                       f"snapback-btc started ({mode}).\nenv={self.client.env}\n"
                       f"deploy_start_equity={equity:.2f} USDT\n"
                       f"kill_switch at {(self.kill_fraction*100):.0f}% = "
                       f"{equity * self.kill_fraction:.2f} USDT")
        else:
            self.log.info("Resuming deploy. start=%.2f current=%.2f (%+.2f%%)",
                          start_eq, equity, (equity/start_eq - 1) * 100)

        pos = self.client.fetch_position(self.symbol)
        if pos.side != "flat":
            if self.dry_run:
                self.log.warning("Boot found open position %s qty=%.4f @ %.2f. "
                                 "DRY-RUN: leaving it alone.",
                                 pos.side, pos.qty, pos.entry_price)
            else:
                self.log.warning("Boot found open position %s qty=%.4f @ %.2f. Flattening.",
                                 pos.side, pos.qty, pos.entry_price)
                self.client.close_position(self.symbol)
                state.record_event("WARN", "boot_flatten",
                                   {"side": pos.side, "qty": pos.qty,
                                    "entry": pos.entry_price})

    def stop(self, *_args) -> None:
        self._stopped = True
        self.log.info("Stop requested.")

    def _heartbeat(self) -> None:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.touch()

    def _check_kill_switch(self, equity: float) -> bool:
        start = state.get_float("deploy_start_equity", 0.0)
        if start <= 0:
            return False
        if equity < start * self.kill_fraction:
            self.log.error("KILL SWITCH: equity %.2f < %.2f (start %.2f * %.2f)",
                           equity, start * self.kill_fraction, start, self.kill_fraction)
            (REPO_ROOT / "data" / "HALT").touch()
            send_alert(
                "BOT KILL SWITCH FIRED",
                f"Equity drawdown breached -{(1-self.kill_fraction)*100:.0f}%.\n"
                f"deploy_start={start:.2f}  current={equity:.2f}  "
                f"drawdown={(equity/start - 1)*100:+.2f}%\n"
                f"Position will be flattened, HALT file created, bot will exit.",
            )
            return True
        return False

    def _maybe_enter(self, equity: float) -> None:
        # Skip entry evaluation if already in a position — bracket SL/TP
        # manage the existing trade. Matches backtest's exclusive_orders=True.
        pos = self.client.fetch_position(self.symbol)
        if pos.side != "flat":
            return

        # 1500 bars (~15 days) covers v1 warmup (200-EMA + 50). v3 needs more
        # for the 30-day daily ATR percentile gate; vol-regime will report
        # NaN until ~30 days of accumulated data on the exchange side.
        df = self.client.fetch_ohlcv(self.symbol, "15m", limit=1500)
        if len(df) < 250:
            return
        last_ts = df.index[-1]
        if self._last_signal_ts is not None and last_ts <= self._last_signal_ts:
            return

        funding = self.client.fetch_funding_rate(self.symbol)
        strategy_name = self.params.get("strategy_name", "multifactor-v1")

        if strategy_name == "v3-all-wider-4":
            side, sl_dist, tp_dist, dbg = evaluate_signal_v3all_wider4(
                df, funding, self.params)
            price = dbg.get("cur_close", float(df["Close"].iloc[-1]))
        else:
            side, dbg = evaluate_signal(df, funding, self.params)
            price = dbg.get("cur_close", float(df["Close"].iloc[-1])) if isinstance(dbg, dict) else float(df["Close"].iloc[-1])
            sl_pct = float(self.params["strategy"]["sl_pct"])
            tp_pct = float(self.params["strategy"]["tp_pct"])
            sl_dist = sl_pct * price
            tp_dist = tp_pct * price

        self._last_signal_ts = last_ts

        if side is None:
            return

        risk_pct = float(self.params["sizing"]["risk_per_trade_pct"])
        raw_qty = compute_qty_from_distance(equity, price, sl_dist, risk_pct, self.leverage)
        qty = round_qty_down(raw_qty, self.constraints.qty_step)
        notional = qty * price

        # Exchange minimums (min qty + min notional). These come from the live
        # market spec — tighter than our own caps. If a signal can't be filled,
        # SKIP it. Do NOT scale up to meet the minimum; that would violate the
        # risk budget.
        ok, reason = passes_minimums(qty, price, self.constraints)
        if not ok:
            self.log.warning("Skipping %s signal: %s (equity=$%.2f). "
                             "Need more capital or accept smaller positions.",
                             side, reason, equity)
            state.record_event("WARN", "signal_skipped_minimum",
                               {**dbg, "side": side, "qty": qty, "reason": reason})
            return

        try:
            check_notional(notional)
        except RiskBreach as e:
            self.log.warning("Skipping signal: %s", e)
            return

        sl_price = price - sl_dist if side == "long" else price + sl_dist
        tp_price = price + tp_dist if side == "long" else price - tp_dist

        if self.dry_run:
            self.log.info("DRY-RUN would %s qty=%.4f price=%.2f sl=%.2f tp=%.2f notional=$%.2f",
                          side, qty, price, sl_price, tp_price, notional)
            state.record_event("INFO", "dry_run_signal", {**dbg, "side": side, "qty": qty,
                                                          "sl": sl_price, "tp": tp_price,
                                                          "notional": notional})
            send_alert(f"DRY-RUN: would {side.upper()}",
                       f"DRY-RUN: would have entered {side.upper()} "
                       f"{qty:.4f} BTC @ {price:.2f}\n"
                       f"SL: {sl_price:.2f}  TP: {tp_price:.2f}\n"
                       f"Notional: ${notional:.2f}\n"
                       f"Equity: {equity:.2f} USDT\n"
                       f"(No real order placed.)")
            return

        self.log.info("Signal %s qty=%.4f price=%.2f sl=%.2f tp=%.2f notional=%.2f",
                      side, qty, price, sl_price, tp_price, notional)
        try:
            orders = self.client.market_order_with_bracket(
                self.symbol, side, qty, sl_price, tp_price)
            state.record_fill(side=side, qty=qty, price=price,
                              reason="entry", equity_after=equity)
            state.record_event("INFO", "entry", {**dbg, "side": side, "qty": qty,
                                                  "sl": sl_price, "tp": tp_price,
                                                  "order_ids": {k: v.get("id") for k, v in orders.items()}})
            send_alert(f"Bot {side.upper()} entry",
                       f"{side.upper()} {qty:.4f} BTC @ {price:.2f}\n"
                       f"SL: {sl_price:.2f}  TP: {tp_price:.2f}\n"
                       f"Equity: {equity:.2f} USDT")
        except Exception as e:
            self.log.exception("order placement failed: %s", e)
            state.record_event("ERROR", "order_failed", str(e))
            send_alert("Bot order failed", f"{side} entry failed: {e}")

    def _maybe_time_stop(self, equity: float) -> None:
        pos = self.client.fetch_position(self.symbol)
        if pos.side == "flat":
            return
        max_hold = int(self.params["strategy"]["max_hold_bars"])
        max_hold_s = max_hold * 15 * 60
        try:
            with sqlite3.connect(state.DB_PATH) as c:
                row = c.execute(
                    "SELECT ts FROM fills WHERE reason='entry' "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if not row:
                return
            entry_ts = datetime.fromisoformat(row[0])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
            age_s = (datetime.now(timezone.utc) - entry_ts).total_seconds()
            if age_s >= max_hold_s:
                if self.dry_run:
                    self.log.info("DRY-RUN: would time-stop close after %.1f hours", age_s / 3600)
                    return
                self.log.info("Time-stop firing after %.1f hours", age_s / 3600)
                self.client.close_position(self.symbol)
                state.record_fill(side="close", qty=pos.qty, price=0.0,
                                  reason="time_stop", equity_after=equity)
                send_alert("Bot time-stop close",
                           f"Closed {pos.side} {pos.qty:.4f} BTC after "
                           f"{age_s/3600:.1f}h hold. Equity: {equity:.2f}")
        except Exception as e:
            self.log.warning("time-stop check failed: %s", e)

    def loop(self) -> int:
        self.log.info("Bot loop started. poll=%.1fs symbol=%s", self.poll_s, self.symbol)
        backoff_s = self.poll_s
        while not self._stopped:
            try:
                self._heartbeat()
                if is_halted():
                    self.log.warning("HALT file present — %s.",
                                     "exiting (DRY-RUN, no flatten)" if self.dry_run
                                     else "flattening and exiting")
                    if not self.dry_run:
                        self.client.close_position(self.symbol)
                    send_alert("Bot HALTED",
                               f"data/HALT detected. "
                               f"{'DRY-RUN exit, no flatten.' if self.dry_run else 'Position flattened.'} "
                               f"Bot exiting.")
                    return 0

                equity = self.client.fetch_equity_usdt()
                if self._check_kill_switch(equity):
                    if not self.dry_run:
                        self.client.close_position(self.symbol)
                    return 0

                self._maybe_time_stop(equity)
                self._maybe_enter(equity)
                backoff_s = self.poll_s
                time.sleep(self.poll_s)
            except KeyboardInterrupt:
                self.log.info("KeyboardInterrupt — clean stop.")
                return 0
            except Exception as e:
                self.log.exception("loop error: %s", e)
                state.record_event("ERROR", "loop_error", str(e))
                backoff_s = min(backoff_s * 2, 300)
                time.sleep(backoff_s)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Observe-only: fetch real data, evaluate signals, log what WOULD happen, "
                         "but never place real orders. Also honored via DRY_RUN=1 env var.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    dry_run = args.dry_run or os.environ.get("DRY_RUN") in ("1", "true", "yes")

    params = load_params()
    log = _setup_logging(args.log_level)
    env = get_env()
    log.info("snapback-btc booting env=%s dry_run=%s", env, dry_run)
    if env == "mainnet" and not dry_run:
        log.warning("=" * 60)
        log.warning("MAINNET LIVE MODE. Real money at risk.")
        log.warning("=" * 60)

    bot = Bot(params, dry_run=dry_run)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.boot()
    return bot.loop()


if __name__ == "__main__":
    sys.exit(main())
