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
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import ccxt
import pandas as pd
import yaml

from alerts import send_alert
from bot_internals import (
    SignalDecision,
    evaluate_for_strategy,
    gate_status,
    limit_entry_price,
    resolve_strategy_name,
)
from exchange import principal, state, trade_events
from exchange.binance_client import BinanceClient
from exchange.constraints import (
    DEFAULT_CONSTRAINTS,
    ExchangeConstraints,
    merge_with_live,
    passes_minimums,
    round_qty_down,
)
from exchange.env import (
    REPO_ROOT,
    get_env,
    halt_source,
    is_halted,
    leg_halt_path,
    load_env_for_instance,
)
from risk import (
    CEILINGS,
    RiskBreach,
    check_daily_loss,
    check_leverage,
    check_notional,
    check_symbol,
)

# Re-export for tools/preflight_live.py and any downstream importers.
from strategy.live_multifactor_v1 import evaluate_signal  # noqa: F401
from tools import consolidate_push

LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "bot.jsonl"
HEARTBEAT = REPO_ROOT / "data" / "heartbeat"
# CONFIG_PATH default — overridable via --config CLI flag (or --instance).
CONFIG_PATH = REPO_ROOT / "config" / "params.yaml"

# Named instance profiles for the multi-leg deploy. Picking `--instance donchian`
# derives all four paths (config / state.db / log / heartbeat) at once so the
# systemd unit and tmux commands don't have to spell out four flags each.
# Individual --config / --state-db / --log-file / --heartbeat overrides still
# work and take precedence — useful for one-off experiments.
INSTANCE_PROFILES: dict[str, dict[str, Path]] = {
    "v1": {
        "config":    REPO_ROOT / "config" / "params.yaml",
        "state_db":  REPO_ROOT / "data" / "state.db",
        "log_file":  REPO_ROOT / "logs" / "bot.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat",
    },
    "donchian": {
        "config":    REPO_ROOT / "config" / "params_donchian.yaml",
        "state_db":  REPO_ROOT / "data" / "state_donchian.db",
        "log_file":  REPO_ROOT / "logs" / "donchian.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_donchian",
    },
    "cnh_short": {
        "config":    REPO_ROOT / "config" / "params_cnh_hybrid_short.yaml",
        "state_db":  REPO_ROOT / "data" / "state_cnh_short.db",
        "log_file":  REPO_ROOT / "logs" / "cnh_short.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_cnh_short",
    },
    # NEW LEG (2026-05-30): SOL cnh-hybrid-short. Validated Phase 1-5 on SOL
    # (memory: snapback-sol-hybrid-short-phase2to5). BLOCKED until risk.py
    # ALLOWED_SYMBOLS includes "SOL/USDT:USDT" — check_symbol() runs at startup
    # so this gate blocks even --dry-run. Tier-3 edit (RISK_REVIEW=1 + user OK).
    "cnh_short_sol": {
        "config":    REPO_ROOT / "config" / "params_cnh_hybrid_short_sol.yaml",
        "state_db":  REPO_ROOT / "data" / "state_cnh_short_sol.db",
        "log_file":  REPO_ROOT / "logs" / "cnh_short_sol.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_cnh_short_sol",
    },
}

# Derive each leg's SELF-halt flag (data/HALT_<instance>) the same way as the
# other per-leg paths, from a single source of truth (env.leg_halt_path). The
# shared data/HALT stays a GLOBAL manual stop-all and is NOT in the profile —
# no leg's kill switch may write it (that shared write is what caused the 07-01
# cascade). See exchange/env.py :: is_halted / leg_halt_path.
for _name, _profile in INSTANCE_PROFILES.items():
    _profile["halt"] = leg_halt_path(_name)

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
                "ts": datetime.now(UTC).isoformat(),
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


def load_params(path: str | None = None) -> dict:
    """Load YAML params. Default path = config/params.yaml. Override via --config."""
    p = path or CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


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
    def __init__(self, params: dict, dry_run: bool = False,
                 instance: str = "v1") -> None:
        self.params = params
        self.symbol = params["symbol"]
        self.strategy_name = resolve_strategy_name(params)
        self.dry_run = dry_run
        self.instance = instance
        # This leg's SELF-halt flag. The kill switch touches ONLY this file, so
        # a leg self-halting can never take a sibling down.
        self.halt_path = leg_halt_path(instance)
        self.log = logging.getLogger("snapback.bot")
        # Hedge mode + clientOrderId prefix come from `hedge` block in params YAML.
        # When unset, behaves exactly like the legacy single-bot deploy.
        hedge_cfg = params.get("hedge", {}) or {}
        self.hedge_enabled = bool(hedge_cfg.get("enabled", False))
        self.coid_prefix = str(hedge_cfg.get("client_order_id_prefix", "snap-v1-"))
        self.client = BinanceClient.from_env(
            hedge_mode=self.hedge_enabled,
            coid_prefix=self.coid_prefix,
        )
        exec_cfg = params.get("execution", {})
        self.poll_s = float(exec_cfg["poll_interval_s"])
        self.order_type = str(exec_cfg.get("order_type", "market")).lower()
        self.limit_offset_bps = float(exec_cfg.get("limit_offset_bps", 0.0))
        self.limit_timeout_s = float(exec_cfg.get("limit_timeout_s", 20.0))
        # Entry timeframe from the params YAML — drives both fetch_ohlcv and
        # the time-stop window. v1=15m, donchian=4h, cnh_short=4h. Defaulting
        # to 15m preserves legacy behavior for any config that omits the block.
        self.entry_tf = str((params.get("timeframes") or {}).get("entry", "15m"))
        # parse_timeframe returns seconds — ccxt's standard converter.
        self.bar_seconds = int(self.client.ex.parse_timeframe(self.entry_tf))
        self.leverage = int(params["sizing"]["leverage"])
        self.risk_pct = float(params["sizing"]["risk_per_trade_pct"])
        self.kill_fraction = float(params["deploy"]["kill_switch_equity_fraction"])
        # Min-capital warning threshold (read from params, default $100).
        self.min_capital_warn = float(params.get("deploy", {}).get("min_capital_warn_usdt", 100.0))
        self.constraints: ExchangeConstraints = DEFAULT_CONSTRAINTS
        self._stopped = False
        self._last_signal_ts: pd.Timestamp | None = None
        # Push to consolidate every PUSH_INTERVAL_S; also send a heartbeat
        # event at the same cadence so the dashboard's "alive" check works.
        # 30s is well under consolidate's 60s healthy-threshold.
        self._push_interval_s = 30.0
        self._last_push_ts: float = 0.0
        # Most recent gate-status snapshot from the live evaluator. Updated on
        # every bar-close evaluation. Pushed to consolidate as part of the
        # heartbeat payload so the dashboard can show "what's true now / waiting
        # on what" without you SSHing in.
        self._latest_gates: dict | None = None
        self._last_waiting_for: str | None = None
        # Last observed position side, used by _detect_bracket_exit to spot
        # open→flat transitions. "unknown" until the first loop call so we
        # don't emit a spurious exit alert for a historical entry in state.db.
        self._last_position_side: str = "unknown"
        # Latch so the daily-loss breaker logs/alerts once per trip, not every
        # poll. Cleared on drawdown recovery or UTC-day rollover.
        self._daily_loss_blocked: bool = False
        # Principal-anchor reconcile cadence. The kill switch compares equity to
        # NET DEPOSITED PRINCIPAL (exchange/principal.py), refreshed from the
        # Binance income ledger. DCA is monthly so hourly is ample once the
        # anchor exists; retry faster while it is still uninitialised.
        self._principal_reconcile_interval_s: float = 3600.0
        self._principal_retry_interval_s: float = 60.0
        self._last_principal_reconcile_ts: float = 0.0
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

        # Establish the principal anchor P (kill-switch denominator = NET
        # DEPOSITED PRINCIPAL, not a balance snapshot). Best-effort: if the
        # income backfill fails here, the loop retries and the kill switch
        # stays fail-safe (disabled) until P is known. NEVER snapshots balance.
        try:
            P = principal.initialize(self.client, self.params, log=self.log)
            self._last_principal_reconcile_ts = time.time()
            if P is not None:
                self.log.info(
                    "Principal anchor P=%.2f USDT (kill floor=%.2f at fraction %.3f)",
                    P, P * self.kill_fraction, self.kill_fraction)
            else:
                self.log.warning(
                    "Principal anchor NOT established at boot — kill switch "
                    "DISABLED until income reconcile succeeds.")
        except Exception as e:
            self.log.warning("Principal init failed at boot (will retry in loop): %s", e)

        equity = self.client.fetch_equity_usdt()
        if equity < self.min_capital_warn:
            self.log.warning("LOW CAPITAL: equity $%.2f < $%.2f recommended. "
                             "Some signals may be skipped due to exchange minimums.",
                             equity, self.min_capital_warn)

        start_eq = state.get_float("deploy_start_equity", 0.0)
        if start_eq <= 0:
            state.set_float("deploy_start_equity", equity)
            state.set_meta("deploy_start_ts", datetime.now(UTC).isoformat())
            self.log.info("Recorded deploy_start_equity=%.2f USDT", equity)
            mode = "DRY-RUN" if self.dry_run else "LIVE"
            send_alert(
                f"Bot deploy start [{mode}]",
                self._format_deploy_summary(equity, mode),
            )
            start_eq = equity
        else:
            self.log.info("Resuming deploy. start=%.2f current=%.2f (%+.2f%%)",
                          start_eq, equity, (equity/start_eq - 1) * 100)

        # Push a boot event to consolidate so the dashboard knows the bot
        # is alive and which strategy/env it's running. The deploy-start
        # equity and kill-switch fraction let consolidate compute the
        # kill-switch level without needing a separate config endpoint.
        state.enqueue_bot_event(
            "boot",
            strategy=self.strategy_name,
            equity_usd=float(equity),
            payload={
                "env": self.client.env,
                "dry_run": bool(self.dry_run),
                "strategy_name": self.strategy_name,
                "deploy_start_equity": float(start_eq),
                "principal_anchor": principal.get_principal(),
                "kill_switch_fraction": float(self.kill_fraction),
                "leverage": int(self.leverage),
                "order_type": self.order_type,
            },
        )

        pos = self.client.fetch_position(self.symbol)
        if pos.side != "flat":
            if self.dry_run:
                self.log.warning("Boot found open position %s qty=%.4f @ %.2f. "
                                 "DRY-RUN: leaving it alone.",
                                 pos.side, pos.qty, pos.entry_price)
            else:
                root = state.latest_entry_coid_root()
                self.log.warning("Boot found open position %s qty=%.4f @ %.2f. "
                                 "Flattening (root=%s).",
                                 pos.side, pos.qty, pos.entry_price, root or "—")
                self.client.close_position(self.symbol,
                                           client_order_id_root=root, close_leg="bf")
                state.record_event("WARN", "boot_flatten",
                                   {"side": pos.side, "qty": pos.qty,
                                    "entry": pos.entry_price,
                                    "signal_id": root},
                                   signal_id=root)
                state.enqueue_bot_event(
                    "boot_flatten",
                    signal_id=root,
                    side=pos.side,
                    qty=float(pos.qty),
                    price_usd=float(pos.entry_price),
                    payload={"reason": "stale_position_at_boot"},
                )

    def stop(self, *_args) -> None:
        self._stopped = True
        self.log.info("Stop requested.")

    def _heartbeat(self) -> None:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.touch()

    def _format_deploy_summary(self, equity: float, mode: str) -> str:
        kill_pct = self.kill_fraction * 100.0
        # Kill floor is anchored on DEPOSITED PRINCIPAL, not this boot's equity.
        # Fall back to the equity figure only if the anchor isn't established yet
        # (income backfill pending) — marked provisional so it isn't mistaken
        # for the real floor.
        P = principal.get_principal()
        kill_anchor = P if (P is not None and P > 0) else equity
        kill_usd = kill_anchor * self.kill_fraction
        kill_anchor_label = ("principal" if (P is not None and P > 0)
                             else "equity (PROVISIONAL — principal pending)")
        max_hold = int(
            (self.params.get("strategy") or {}).get("max_hold_bars", 0)
        )
        max_hold_days = (max_hold * self.bar_seconds) / 86400.0 if max_hold else 0.0
        dedup_bars = int(
            (self.params.get("strategy") or {}).get("dedup_bars", 0)
        )
        consolidate_source = (os.environ.get("CONSOLIDATE_SOURCE") or "").strip() or "(default snapback-btc)"
        return (
            f"snapback-btc deploy start\n"
            f"=========================\n"
            f"Instance      : {self.instance}\n"
            f"Strategy      : {self.strategy_name}\n"
            f"Symbol        : {self.symbol}\n"
            f"Config        : {CONFIG_PATH.relative_to(REPO_ROOT)}\n"
            f"Mode          : {mode}\n"
            f"Environment   : {self.client.env}\n"
            f"PID           : {os.getpid()}\n"
            f"Boot ts (UTC) : {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"\n"
            f"Capital\n"
            f"  Start equity   : ${equity:,.2f} USDT\n"
            f"  Min recommended: ${self.min_capital_warn:,.2f} USDT"
            + ("  ⚠ BELOW FLOOR" if equity < self.min_capital_warn else "")
            + f"\n"
            f"  Kill switch    : {kill_pct:.1f}% of {kill_anchor_label}\n"
            f"                   → exit at ${kill_usd:,.2f} USDT "
            f"(anchor ${kill_anchor:,.2f})\n"
            f"\n"
            f"Risk\n"
            f"  Per trade      : {self.risk_pct:.2f}% equity\n"
            f"  Leverage       : {self.leverage}x\n"
            f"  Hedge mode     : {'on' if self.hedge_enabled else 'off'}\n"
            + (f"  Dedup          : {dedup_bars} bars\n" if dedup_bars else "")
            + f"\n"
            f"Timing\n"
            f"  Entry timeframe: {self.entry_tf} ({self.bar_seconds}s/bar)\n"
            f"  Poll interval  : {self.poll_s:.0f}s\n"
            + (f"  Max hold       : {max_hold} bars ({max_hold_days:.1f} days)\n"
               if max_hold else "")
            + f"\n"
            f"Routing\n"
            f"  Consolidate src: {consolidate_source}\n"
            f"  Alert tag      : {os.environ.get('ALERT_TAG', '(default snapback-btc)')}\n"
            f"\n"
            f"Paths\n"
            f"  State DB       : {state.DB_PATH.relative_to(REPO_ROOT)}\n"
            f"  Log file       : {LOG_FILE.relative_to(REPO_ROOT)}\n"
            f"  Heartbeat      : {HEARTBEAT.relative_to(REPO_ROOT)}\n"
        )

    def _maybe_push_consolidate(self, equity: float | None) -> None:
        """If enough time elapsed since last push: enqueue a heartbeat event
        and drain the outbox. Never blocks for more than the HTTP timeout
        (3s by default). Failures are logged and retried on the next tick.
        """
        now = time.time()
        if now - self._last_push_ts < self._push_interval_s:
            return
        self._last_push_ts = now
        if not consolidate_push.is_configured():
            return  # don't even enqueue heartbeats if no consumer
        try:
            payload: dict = {"halt_present": is_halted(self.instance),
                             "outbox_size_before_drain": state.outbox_size()}
            # Include the most recent gate-status snapshot so the dashboard
            # can show "what's true now / waiting on what" without operator
            # intervention. Falls back to absent key when no bar has been
            # evaluated yet (first 5-10s of boot).
            if self._latest_gates is not None:
                payload["gates"] = self._latest_gates
            state.enqueue_bot_event(
                "heartbeat",
                equity_usd=float(equity) if equity is not None else None,
                payload=payload,
            )
            result = consolidate_push.drain()
            if result.get("error"):
                self.log.debug("consolidate push deferred: %s", result.get("error"))
        except Exception as e:
            # Push must NEVER affect the trading loop. Log + move on.
            self.log.warning("consolidate push raised (continuing): %s", e)

    def _check_kill_switch(self, equity: float) -> bool:
        # Anchor = NET DEPOSITED PRINCIPAL (God's rule), NOT a balance snapshot
        # and NOT a high-water mark. principal.breached() is fail-safe: it never
        # trips on an unknown/degenerate anchor, so a stale/wrong equity read or
        # a not-yet-initialised P cannot cause a false-positive kill (the
        # donchian $114.75 class of incident).
        P = principal.get_principal()
        if not principal.breached(equity, P, self.kill_fraction):
            return False
        # breached() only returns True for a positive float anchor. Make that
        # invariant explicit with a real guard — an `assert` here is stripped
        # under `python -O`, which would let `P * fraction` raise TypeError (or
        # touch the HALT on a None anchor) in the hot trading path.
        if P is None:
            raise RuntimeError(
                "kill-switch invariant violated: breached() true but principal "
                "is None — refusing to act on an unknown anchor.")
        floor = P * self.kill_fraction
        self.log.error("KILL SWITCH: equity %.2f < %.2f (principal %.2f * %.2f)",
                       equity, floor, P, self.kill_fraction)
        # Touch ONLY this leg's self-halt flag — never the shared data/HALT.
        self.halt_path.touch()
        state.enqueue_bot_event(
            "kill_switch", equity_usd=float(equity),
            payload={"principal_anchor": float(P),
                     "kill_switch_fraction": float(self.kill_fraction),
                     "kill_floor": float(floor),
                     "drawdown_vs_principal_pct": (equity / P - 1) * 100},
        )
        # Force a push so the dashboard sees this before the bot exits.
        try:
            consolidate_push.drain()
        except Exception as e:
            self.log.warning("kill-switch push failed: %s", e)
        send_alert(
            "BOT KILL SWITCH FIRED",
            f"Equity fell below -{(1-self.kill_fraction)*100:.0f}% of DEPOSITED "
            f"PRINCIPAL.\n"
            f"principal={P:.2f}  floor={floor:.2f}  current={equity:.2f}  "
            f"({(equity / P - 1)*100:+.2f}% vs principal)\n"
            f"Position will be flattened, data/HALT_{self.instance} created, "
            f"bot will exit.",
        )
        return True

    def _maybe_reconcile_principal(self) -> None:
        """Periodically refresh the principal anchor from the Binance income
        ledger. Idempotent (tranId-keyed) so overlapping windows never
        double-count. Never blocks or crashes the loop — failures are logged."""
        now = time.time()
        initialized = principal.is_initialized()
        interval = (self._principal_reconcile_interval_s if initialized
                    else self._principal_retry_interval_s)
        if now - self._last_principal_reconcile_ts < interval:
            return
        self._last_principal_reconcile_ts = now
        try:
            if not initialized:
                principal.initialize(self.client, self.params, log=self.log)
            else:
                principal.reconcile_recent(self.client, log=self.log)
        except Exception as e:
            self.log.warning("principal reconcile failed (continuing): %s", e)

    def _daily_anchor_equity(self, equity: float) -> float:
        """Return the UTC-day's starting equity, re-anchoring on date rollover.

        Persisted in state.meta so it survives restarts within a day. Unlike
        deploy_start_equity (set once at first deploy, drives the -18% cumulative
        kill-switch), this anchor resets every UTC midnight and drives the tighter
        2% daily-loss breaker. The two are intentionally separate ceilings.
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        anchor_date = state.get_meta("daily_anchor_date")
        anchor_eq = state.get_float("daily_anchor_equity", 0.0)
        if anchor_date != today or anchor_eq <= 0:
            state.set_meta("daily_anchor_date", today)
            state.set_float("daily_anchor_equity", equity)
            self.log.info("Daily anchor set: date=%s equity=%.2f USDT", today, equity)
            return equity
        return anchor_eq

    def _daily_loss_blocks_entry(self, equity: float) -> bool:
        """True if today's drawdown has hit MAX_DAILY_LOSS_PCT — block new
        entries for the rest of the UTC day. Does NOT flatten or HALT: existing
        brackets keep managing any open position, and the anchor resets at the
        next UTC midnight (see _daily_anchor_equity). This is the tighter,
        daily-resetting sibling of the -18% cumulative kill-switch.
        """
        day_start = self._daily_anchor_equity(equity)
        try:
            check_daily_loss(equity, day_start)
        except RiskBreach as e:
            loss_pct = (day_start - equity) / day_start * 100.0 if day_start > 0 else 0.0
            # Dedupe: emit the outbox event + alert at most once per UTC day.
            # self._daily_loss_blocked is an in-memory fast path; the persisted
            # meta key 'daily_loss_breaker_date' ensures the latch survives a
            # bot restart within the same UTC day (prevents the dashboard from
            # receiving a duplicate event on every restart after a breach).
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            if not self._daily_loss_blocked:
                if state.get_meta("daily_loss_breaker_date") == today:
                    self._daily_loss_blocked = True  # sync in-memory with persisted latch
                else:
                    # Log + event once per day, on the bar the breaker trips.
                    self.log.warning("DAILY LOSS BREAKER: %s — blocking new entries "
                                     "until next UTC day.", e)
                    state.record_event("WARN", "daily_loss_breaker",
                                       {"equity": equity, "day_start": day_start,
                                        "loss_pct": loss_pct,
                                        "threshold_pct": CEILINGS.MAX_DAILY_LOSS_PCT})
                    state.enqueue_bot_event(
                        "daily_loss_breaker", equity_usd=float(equity),
                        payload={"day_start_equity": float(day_start),
                                 "loss_pct": loss_pct,
                                 "threshold_pct": CEILINGS.MAX_DAILY_LOSS_PCT},
                    )
                    send_alert(
                        "Bot daily-loss breaker tripped",
                        f"Daily drawdown breached -{CEILINGS.MAX_DAILY_LOSS_PCT:.0f}%.\n"
                        f"day_start={day_start:.2f}  current={equity:.2f}  "
                        f"loss={loss_pct:.2f}%\n"
                        f"New entries blocked until next UTC day. "
                        f"Open position (if any) left to its brackets; no flatten.",
                    )
                    state.set_meta("daily_loss_breaker_date", today)
                    self._daily_loss_blocked = True
            return True
        # Drawdown recovered or new day anchored — clear the in-memory latch.
        # (The persisted key expires naturally: next breach day uses a new date.)
        self._daily_loss_blocked = False
        return False

    def _maybe_enter(self, equity: float) -> None:
        # Skip entry evaluation if already in a position — bracket SL/TP
        # manage the existing trade. Matches backtest's exclusive_orders=True.
        if self.client.fetch_position(self.symbol).side != "flat":
            return

        # Daily-loss breaker: tighter (2%) daily-resetting guard, separate from
        # the -18% cumulative kill-switch. Blocks NEW entries only.
        if self._daily_loss_blocks_entry(equity):
            return

        # 1500 bars covers warmup on every supported timeframe:
        #   15m → ~15 days (v1: 200-EMA + 50 warmup, vol-regime needs ~30d
        #        of daily ATR — NaN until accumulated)
        #   4h  → ~250 days (donchian-v3: 80-bar channel, cnh-hybrid: 200-bar
        #        admission walk)
        # Binance Futures klines cap at 1500/call so 1500 is also the ceiling.
        df = self.client.fetch_ohlcv(self.symbol, self.entry_tf, limit=1500)
        # Binance returns the still-FORMING current bar as the last row (~5s
        # after it opens it holds ~0.5% of its eventual volume). The backtest
        # only ever sees CLOSED bars (backtesting.py never shows a forming bar;
        # signals_multifactor.next() reads Close[-1] of a closed bar). Drop the
        # forming bar HERE, before anything downstream touches the frame: every
        # evaluator, the entry price, SL/TP, the 4H-gate alignment timestamp,
        # the warmup count, and the dedup ts all read iloc[-1]/index[-1], so a
        # single slice at the source keeps live↔backtest parity intact. Without
        # this, the volume gate (cur_vol > 2×SMA20) can never be satisfied →
        # 0 trades, and entry/SL/TP would anchor to an incomplete bar.
        df = df.iloc[:-1]
        if len(df) < 250:
            return
        last_ts = df.index[-1]
        if self._last_signal_ts is not None and last_ts <= self._last_signal_ts:
            return

        funding = self.client.fetch_funding_rate(self.symbol)
        decision = evaluate_for_strategy(self.strategy_name, df, funding, self.params)
        self._last_signal_ts = last_ts

        # Snapshot the gate state for heartbeat-payload + log. Computed on every
        # bar evaluation so the dashboard's "current state" panel can always
        # answer "why isn't this firing?". Log only when the waiting-for
        # summary *changes*, to avoid spamming the JSONL with identical lines
        # bar after bar in long quiet stretches.
        self._latest_gates = gate_status(self.strategy_name, decision, self.params)
        cur_waiting = self._latest_gates.get("waiting_for", "")
        if cur_waiting and cur_waiting != self._last_waiting_for:
            self.log.info("gates: %s", cur_waiting)
            self._last_waiting_for = cur_waiting

        if decision.side is None:
            return

        qty = round_qty_down(
            compute_qty_from_distance(
                equity, decision.price, decision.sl_distance,
                self.risk_pct, self.leverage),
            self.constraints.qty_step,
        )
        notional = qty * decision.price

        # Exchange minimums (min qty + min notional) come from the live market
        # spec — tighter than our own caps. If a signal can't be filled, SKIP
        # it. Do NOT scale up to meet the minimum; that would violate the
        # risk budget.
        ok, reason = passes_minimums(qty, decision.price, self.constraints)
        if not ok:
            self.log.warning("Skipping %s signal: %s (equity=$%.2f). "
                             "Need more capital or accept smaller positions.",
                             decision.side, reason, equity)
            state.record_event("WARN", "signal_skipped_minimum",
                               {**decision.debug, "side": decision.side,
                                "qty": qty, "reason": reason})
            return

        try:
            check_notional(notional)
        except RiskBreach as e:
            self.log.warning("Skipping signal: %s", e)
            return

        # signal_id anchors all 3 legs of this trade on Binance via clientOrderId.
        # Format: snap-v1-<signal_id>-{e|s|t|x|bf|h|k}. Investing-consolidate's
        # importer joins entry+exit fills via this root. ms precision avoids
        # collisions across bot restarts.
        signal_id = str(int(time.time() * 1000))

        self.log.info(
            "%s %s [%s] sid=%s qty=%.4f price=%.2f sl=%.2f tp=%.2f notional=$%.2f",
            "DRY-RUN would" if self.dry_run else "Signal",
            decision.side, self.order_type, signal_id, qty,
            decision.price, decision.sl_price, decision.tp_price, notional,
        )

        if self.dry_run:
            trade_events.record_dry_run_entry(
                side=decision.side, qty=qty, price=decision.price,
                sl_price=decision.sl_price, tp_price=decision.tp_price,
                notional=notional, equity=equity,
                signal_id=signal_id, strategy_name=self.strategy_name,
                order_type=self.order_type, dbg=decision.debug,
            )
            return

        try:
            self._place_live_entry(decision, qty, signal_id, equity)
        except Exception as e:
            self.log.exception("order placement failed: %s", e)
            state.record_event("ERROR", "order_failed", str(e), signal_id=signal_id)
            send_alert("Bot order failed", f"{decision.side} entry failed: {e}")

    def _place_live_entry(
        self, decision: SignalDecision, qty: float, signal_id: str, equity: float,
    ) -> None:
        """Place a live entry (market or limit per config) + brackets, then
        record the fill via trade_events. Raises on order-placement failure
        so the caller can log + alert."""
        if self.order_type == "limit":
            limit_price = limit_entry_price(
                decision.side, decision.price, self.limit_offset_bps)
            orders = self.client.limit_order_with_bracket(
                self.symbol, decision.side, qty, limit_price,
                sl_distance=decision.sl_distance, tp_distance=decision.tp_distance,
                timeout_s=self.limit_timeout_s,
                client_order_id_root=signal_id,
            )
            trade_events.record_limit_entry(
                side=decision.side,
                filled_qty=float(orders.get("filled_qty", qty)),
                fill_price=float(orders.get("fill_price", decision.price)),
                sl_distance=decision.sl_distance,
                tp_distance=decision.tp_distance,
                limit_price=limit_price,
                signal_price=decision.price,
                limit_offset_bps=self.limit_offset_bps,
                equity=equity, signal_id=signal_id,
                strategy_name=self.strategy_name,
                filled_as=str(orders.get("filled_as", "limit")),
                orders=orders, dbg=decision.debug,
            )
            return

        orders = self.client.market_order_with_bracket(
            self.symbol, decision.side, qty,
            decision.sl_price, decision.tp_price,
            client_order_id_root=signal_id,
        )
        trade_events.record_market_entry(
            side=decision.side, qty=qty, price=decision.price,
            sl_price=decision.sl_price, tp_price=decision.tp_price,
            equity=equity, signal_id=signal_id,
            strategy_name=self.strategy_name,
            orders=orders, dbg=decision.debug,
        )

    def _detect_bracket_exit(self, equity: float) -> None:
        """Bracket SL/TP fills close the position on Binance's side without
        a bot-initiated close. Detect open→flat transitions and emit an
        exit alert with PnL.

        Conditions for an alert:
          - Not dry-run (no real brackets in dry-run mode).
          - Current position is flat.
          - Latest fill row is `reason='entry'` — i.e., no bot-initiated
            close (time_stop, kill, halt) has been recorded since the entry.
          - We can find the matching opposite-side trade on Binance.

        Skipped silently (with warning log) on any ccxt error so an outage
        in fetch_my_trades doesn't crash the loop.
        """
        if self.dry_run:
            return
        try:
            pos = self.client.fetch_position(self.symbol)
            current = pos.side
            # First observation since boot — initialise state without
            # emitting. A historical "entry" row in state.db whose position
            # already closed on the exchange must not trigger an alert.
            if self._last_position_side == "unknown":
                self._last_position_side = current
                return
            had_open = self._last_position_side != "flat"
            self._last_position_side = current
            if not had_open or current != "flat":
                return
            with sqlite3.connect(state.DB_PATH) as c:
                row = c.execute(
                    "SELECT reason, side, qty, price, client_order_id_root "
                    "FROM fills ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if not row or row[0] != "entry":
                return
            _, entry_side, entry_qty, entry_price, signal_id = row
            entry_qty = float(entry_qty)
            entry_price = float(entry_price)

            opposite = "sell" if entry_side == "long" else "buy"
            trades = self.client.ex.fetch_my_trades(self.symbol, limit=10)
            exit_trade = next(
                (t for t in reversed(trades) if t.get("side") == opposite),
                None,
            )
            if exit_trade is None:
                self.log.warning(
                    "bracket-exit: no opposite-side trade found for signal_id=%s",
                    signal_id,
                )
                return
            exit_price = float(exit_trade.get("price") or 0.0)
            if exit_price <= 0:
                return

            if entry_side == "long":
                pnl = (exit_price - entry_price) * entry_qty
            else:
                pnl = (entry_price - exit_price) * entry_qty
            notional = entry_price * entry_qty
            pnl_pct = (pnl / notional) * 100.0 if notional > 0 else 0.0

            state.record_fill(
                side="close", qty=entry_qty, price=exit_price,
                pnl_usd=pnl, reason="bracket_exit",
                equity_after=equity, client_order_id_root=signal_id,
            )
            state.enqueue_bot_event(
                "exit", signal_id=signal_id, side=entry_side,
                qty=float(entry_qty), price_usd=float(exit_price),
                equity_usd=float(equity),
                payload={"reason": "bracket_exit",
                         "entry_price": float(entry_price),
                         "exit_price": float(exit_price),
                         "pnl_usd": float(pnl),
                         "pnl_pct": float(pnl_pct)},
            )
            send_alert(
                f"Bot {entry_side.upper()} exit",
                f"{entry_side.upper()} {entry_qty:.4f} BTC closed "
                f"by bracket (SL or TP)\n"
                f"Entry: {entry_price:,.2f}  Exit: {exit_price:,.2f}\n"
                f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                f"Equity now: ${equity:,.2f}\n"
                f"signal_id: {signal_id or '(untagged)'}",
            )
        except Exception as e:
            self.log.warning("bracket-exit detection failed: %s", e)

    def _maybe_time_stop(self, equity: float) -> None:
        pos = self.client.fetch_position(self.symbol)
        if pos.side == "flat":
            return
        max_hold = int(self.params["strategy"]["max_hold_bars"])
        # max_hold_bars is in entry-TF bars, not 15m bars. Multiply by the
        # entry timeframe's bar duration so 4h strategies don't time-stop 16×
        # too early.
        max_hold_s = max_hold * self.bar_seconds
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
                entry_ts = entry_ts.replace(tzinfo=UTC)
            age_s = (datetime.now(UTC) - entry_ts).total_seconds()
            if age_s >= max_hold_s:
                if self.dry_run:
                    self.log.info("DRY-RUN: would time-stop close after %.1f hours", age_s / 3600)
                    return
                root = state.latest_entry_coid_root()
                self.log.info("Time-stop firing after %.1f hours (root=%s)",
                              age_s / 3600, root or "—")
                self.client.close_position(self.symbol,
                                           client_order_id_root=root, close_leg="x")
                state.record_fill(side="close", qty=pos.qty, price=0.0,
                                  reason="time_stop", equity_after=equity,
                                  client_order_id_root=root)
                state.enqueue_bot_event(
                    "exit", signal_id=root, side=pos.side, qty=float(pos.qty),
                    equity_usd=float(equity),
                    payload={"reason": "time_stop", "age_h": age_s / 3600},
                )
                send_alert("Bot time-stop close",
                           f"Closed {pos.side} {pos.qty:.4f} BTC after "
                           f"{age_s/3600:.1f}h hold. Equity: {equity:.2f}\n"
                           f"signal_id: {root or '(untagged)'}")
        except Exception as e:
            self.log.warning("time-stop check failed: %s", e)

    def loop(self) -> int:
        self.log.info("Bot loop started. poll=%.1fs symbol=%s", self.poll_s, self.symbol)
        backoff_s = self.poll_s
        transient_backoff_s = self.poll_s
        while not self._stopped:
            try:
                self._heartbeat()
                halt_by = halt_source(self.instance)
                if halt_by is not None:
                    which = ("GLOBAL data/HALT (manual stop-all)"
                             if halt_by == "global"
                             else f"self data/HALT_{self.instance}")
                    self.log.warning("HALT flag present [%s] — %s.", which,
                                     "exiting (DRY-RUN, no flatten)" if self.dry_run
                                     else "flattening and exiting")
                    if not self.dry_run:
                        root = state.latest_entry_coid_root()
                        self.client.close_position(
                            self.symbol,
                            client_order_id_root=root, close_leg="h")
                    state.enqueue_bot_event(
                        "halt",
                        payload={"dry_run": bool(self.dry_run),
                                 "halt_source": halt_by},
                    )
                    try:
                        consolidate_push.drain()
                    except Exception as e:
                        self.log.warning("halt push failed: %s", e)
                    send_alert("Bot HALTED",
                               f"{which} detected. "
                               f"{'DRY-RUN exit, no flatten.' if self.dry_run else 'Position flattened.'} "
                               f"Bot exiting.")
                    return 0

                self._maybe_reconcile_principal()
                equity = self.client.fetch_equity_usdt()
                if self._check_kill_switch(equity):
                    if not self.dry_run:
                        root = state.latest_entry_coid_root()
                        self.client.close_position(
                            self.symbol,
                            client_order_id_root=root, close_leg="k")
                    return 0

                self._detect_bracket_exit(equity)
                self._maybe_time_stop(equity)
                self._maybe_enter(equity)
                self._maybe_push_consolidate(equity)
                backoff_s = self.poll_s
                transient_backoff_s = self.poll_s
                time.sleep(self.poll_s)
            except KeyboardInterrupt:
                self.log.info("KeyboardInterrupt — clean stop.")
                return 0
            except ccxt.InvalidNonce as e:
                # Binance error -1021: request timestamp outside recvWindow.
                # Caused by droplet clock drift. ccxt's adjustForTimeDifference
                # only re-syncs on the next fetch_time call, so force one now
                # before retrying. Short capped backoff — drift normally clears
                # within a few seconds after the resync.
                self.log.warning("InvalidNonce (clock drift) — forcing time re-sync: %s", e)
                try:
                    self.client.ex.load_time_difference()
                except Exception as sync_err:
                    self.log.warning("load_time_difference() failed: %s", sync_err)
                transient_backoff_s = min(transient_backoff_s * 2, 30)
                time.sleep(transient_backoff_s)
            except ccxt.NetworkError as e:
                # Connection drop, timeout, DDoSProtection, ExchangeNotAvailable.
                # All transient — retry on a short capped backoff so a 10-second
                # Binance hiccup doesn't escalate to the 300s catch-all backoff.
                self.log.warning("NetworkError (transient) — retrying: %s", e)
                transient_backoff_s = min(transient_backoff_s * 2, 30)
                time.sleep(transient_backoff_s)
            except Exception as e:
                self.log.exception("loop error: %s", e)
                state.record_event("ERROR", "loop_error", str(e))
                backoff_s = min(backoff_s * 2, 300)
                time.sleep(backoff_s)
        return 0


def main() -> int:
    global LOG_FILE, HEARTBEAT, CONFIG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Observe-only: fetch real data, evaluate signals, log what WOULD happen, "
                         "but never place real orders. Also honored via DRY_RUN=1 env var.")
    ap.add_argument("--log-level", default="INFO")
    # Canonical multi-instance selector. `--instance donchian` derives config,
    # state.db, log, heartbeat paths in one shot. Individual --config etc.
    # below override single paths if you need to mix and match.
    ap.add_argument("--instance", default="v1", choices=list(INSTANCE_PROFILES.keys()),
                    help="Named instance profile (v1 | donchian). Derives all four paths "
                         "from INSTANCE_PROFILES. Individual --config / --state-db / "
                         "--log-file / --heartbeat take precedence if also set.")
    ap.add_argument("--config", default=None,
                    help="Path to params YAML (overrides --instance default).")
    ap.add_argument("--state-db", default=None,
                    help="Path to SQLite state.db (overrides --instance default).")
    ap.add_argument("--log-file", default=None,
                    help="Path to JSONL log file (overrides --instance default).")
    ap.add_argument("--heartbeat", default=None,
                    help="Path to heartbeat file (overrides --instance default).")
    args = ap.parse_args()

    # Resolve effective paths: profile defaults, then per-flag overrides.
    profile = INSTANCE_PROFILES[args.instance]
    config_path = Path(args.config) if args.config else profile["config"]
    state_db_path = Path(args.state_db) if args.state_db else profile["state_db"]
    log_file_path = Path(args.log_file) if args.log_file else profile["log_file"]
    heartbeat_path = Path(args.heartbeat) if args.heartbeat else profile["heartbeat"]

    # Apply path overrides BEFORE the rest of main runs — state, LOG_FILE,
    # and HEARTBEAT are module-level constants that the rest of the bot
    # reads, so write them before _setup_logging() / bot.boot() / bot.loop().
    CONFIG_PATH = config_path
    state.set_db_path(state_db_path)
    LOG_FILE = log_file_path
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT = heartbeat_path
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)

    # Overlay .env.{instance} (donchian's sub-account API key, cnh_short's
    # ALERT_TAG, etc.) on top of the base .env loaded at exchange.env import.
    instance_env_path = load_env_for_instance(args.instance)
    # Make the instance name discoverable by anything that emits alerts
    # later (alerts.send_alert prepends it to the subject so every leg's
    # mail threads cleanly in Gmail).
    os.environ["SNAPBACK_INSTANCE"] = args.instance

    dry_run = args.dry_run or os.environ.get("DRY_RUN") in ("1", "true", "yes")

    params = load_params(config_path)
    log = _setup_logging(args.log_level)
    log.info("snapback-btc booting instance=%s strategy=%s config=%s state=%s log=%s",
             args.instance, resolve_strategy_name(params),
             config_path, state.DB_PATH, LOG_FILE)
    if instance_env_path is not None:
        log.info("snapback-btc loaded per-instance env: %s",
                 instance_env_path.relative_to(REPO_ROOT))
    env = get_env()
    log.info("snapback-btc booting env=%s dry_run=%s", env, dry_run)
    if env == "mainnet" and not dry_run:
        log.warning("=" * 60)
        log.warning("MAINNET LIVE MODE. Real money at risk.")
        log.warning("=" * 60)

    bot = Bot(params, dry_run=dry_run, instance=args.instance)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.boot()
    return bot.loop()


if __name__ == "__main__":
    sys.exit(main())
