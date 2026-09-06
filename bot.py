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
    bracket_state,
    channel_exit_signal,  # noqa: F401  (re-exported; tools import it from here)
    evaluate_for_strategy,
    gate_status,
    limit_entry_price,
    order_avg_price,
    resolve_strategy_name,
    strategy_uses_channel_exit,
    strategy_uses_trend_exit,
    time_stop_due,
    trend_exit_fill_reason,
    trend_exit_signal,
)
from exchange import principal, state, trade_events
from exchange.binance_client import BinanceClient
from exchange.constraints import (
    DEFAULT_CONSTRAINTS,
    ExchangeConstraints,
    fallbacks_for_symbol,
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
    # NEW LEG (2026-07-25): SOL supertrend — replaces the never-funded
    # cnh_short slot per God's decision. Winner of the round-3 win-rate-blended
    # walk-forward (9/9 OOS folds positive); see SOL_LEG_VERDICT.md.
    # risk.py ALLOWED_SYMBOLS now includes SOL/USDT:USDT (RISK_REVIEW 2026-07-25).
    # Needs .env.sol_supertrend with its OWN sub-account key — there is no
    # cnh_short key to inherit (that leg was never keyed), and booting without
    # the file is refused outright, see _main()'s per-instance env guard.
    "sol_supertrend": {
        "config":    REPO_ROOT / "config" / "params_sol_supertrend.yaml",
        "state_db":  REPO_ROOT / "data" / "state_sol_supertrend.db",
        "log_file":  REPO_ROOT / "logs" / "sol_supertrend.jsonl",
        "heartbeat": REPO_ROOT / "data" / "heartbeat_sol_supertrend",
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


def _exit_pnl_usd(side: str, entry_price: float | None,
                  exit_price: float | None, qty: float) -> float | None:
    """GROSS realised PnL for a closed position, or None if it cannot be known.

    Deliberately gross (no fees/funding), because that is the convention the
    bracket-exit path at _detect_bracket_exit already writes into the same
    `fills.pnl_usd` column, and one column must mean one thing. Expect it to
    read a few tenths of a percent above the equity ledger: donchian's
    2026-08-23 trade is +$10.09 here versus +$9.95 by equity_after delta, the
    difference being round-trip fees. For true net PnL, difference consecutive
    `equity_after` values instead.
    """
    if not entry_price or not exit_price or not qty:
        return None
    direction = 1.0 if side == "long" else -1.0
    return round((float(exit_price) - float(entry_price)) * float(qty) * direction, 4)


def _rel_to_repo(p: Path) -> str:
    """Repo-relative form of `p` for the startup banner, absolute if outside.

    The state DB, log file and heartbeat are all env-overridable
    (SNAPBACK_STATE_DB / LOG_DIR / …) and those overrides accept ABSOLUTE
    paths pointing anywhere on the box. `Path.relative_to` raises ValueError in
    that case, which would take down boot() over a cosmetic banner line.

    Found 2026-08-23 by the new tests/conftest.py isolation fixture, which
    points the state DB at a pytest tmp dir — the first time anything had ever
    run this code with a DB outside the repo. Latent in production only because
    both live legs happen to use data/*.db.
    """
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


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
        # "SOL/USDT:USDT" -> "SOL". Alerts used to hardcode "BTC", so the
        # sol_supertrend leg emailed "Closed long 1.2000 BTC" for a SOL trade.
        self.base_asset = str(self.symbol).split("/")[0]
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
        # ...and the rest of that position's IDENTITY, because the side alone
        # cannot see a same-poll replacement. Live on 2026-08-23: v1's bracket
        # SL filled at 05:14:30.039 and _maybe_enter re-entered at 05:14:31.290
        # — 1.25 s later, inside one 5 s poll. The loop therefore never observed
        # `flat`, the open→flat edge never fired, and the exit was dropped on
        # the floor: no `close` row in fills, no `exit` bot_event, no alert, and
        # −$5.37 missing from the ledger until it was backfilled by hand weeks
        # later. `entry_price` is the discriminator — side and qty are usually
        # identical across a re-entry, and it can ONLY move when the position is
        # replaced, because _maybe_enter returns early unless flat (there is no
        # scale-in or averaging anywhere in this bot).
        self._last_position_entry: float | None = None
        self._last_position_qty: float | None = None
        # The coid root of the entry that opened the position we are tracking.
        # Captured when we first observe it open, because by the next tick
        # state.latest_entry_coid_root() already names the REPLACEMENT entry.
        self._last_entry_root: str | None = None
        # Consecutive ticks a detected exit has failed to be RECORDED (see
        # _hold_unrecorded_exit). Reset on success and on giving up.
        self._exit_retry_n: int = 0
        # Throttle bracket re-placement (see _maybe_reprotect) so a persistent
        # placement failure can't spam orders/alerts every poll.
        self._last_reprotect_ts: float = 0.0
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
        # One alert per process when the per-position re-place cap is hit — the
        # cap itself is persisted in active_bracket; this only de-dupes the mail.
        self._reprotect_capped_alerted: bool = False
        if self.dry_run:
            self.log.warning("DRY-RUN MODE: no real orders will be placed")

    def boot(self) -> None:
        check_symbol(self.symbol)
        check_leverage(self.leverage)

        # Load live exchange constraints (min qty, min notional). Tighter
        # of hard-coded fallbacks vs live values wins.
        try:
            live_market = self.client.ex.market(self.symbol)
            # Per-symbol fallbacks: BTC's $50 min-notional / $0.10 tick are not
            # a sane default for SOL. merge_with_live still takes the TIGHTER of
            # fallback vs live, so this cannot loosen anything.
            self.constraints = merge_with_live(
                fallbacks_for_symbol(self.symbol), live_market)
            self.log.info("Exchange constraints for %s: min_qty=%s, min_notional=$%s, "
                          "price_step=%s, qty_step=%s",
                          self.symbol, self.constraints.min_qty_base,
                          self.constraints.min_notional_usdt,
                          self.constraints.price_step, self.constraints.qty_step)
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
                close_order = self.client.close_position(
                    self.symbol, client_order_id_root=root, close_leg="bf")
                # Emit the REAL flatten fill + post-close equity so the dashboard
                # attributes the actual PnL. Previously this sent the ENTRY price
                # and no equity, so a boot-flattened WIN showed as $0/loss on the
                # dashboard (makeTrade fell back to (entry-entry)*qty = 0, which
                # deriveLegStats then counts as a loss).
                # Refetch: the create response carries avgPrice "0.00", so the
                # old `or pos.entry_price` fallback always fired and made every
                # boot-flatten report PnL of exactly 0.
                close_price = (self._resolve_fill_price(close_order)
                               or pos.entry_price)
                try:
                    equity_after = float(self.client.fetch_equity_usdt())
                except Exception:
                    equity_after = None
                pnl_usd = _exit_pnl_usd(pos.side, pos.entry_price,
                                        close_price, pos.qty)
                state.record_event("WARN", "boot_flatten",
                                   {"side": pos.side, "qty": pos.qty,
                                    "entry": pos.entry_price, "exit": close_price,
                                    "pnl_usd": pnl_usd, "signal_id": root},
                                   signal_id=root)
                state.enqueue_bot_event(
                    "boot_flatten",
                    signal_id=root,
                    side=pos.side,
                    qty=float(pos.qty),
                    price_usd=float(close_price),
                    equity_usd=equity_after,
                    payload={"reason": "stale_position_at_boot",
                             "exit_price": float(close_price),
                             "entry_price": float(pos.entry_price),
                             "pnl_usd": pnl_usd},
                )
        else:
            # Position is already flat at boot.  Sweep any orphaned reduce-only
            # orders left over from a prior run (e.g. a surviving bracket sibling
            # after an exchange-driven TP/SL fill).  Only our own COID prefix.
            if not self.dry_run:
                try:
                    n = self.client.cancel_open_orders(
                        self.symbol, coid_prefix=self.coid_prefix)
                    if n:
                        self.log.warning(
                            "boot: swept %d orphaned order(s) while flat", n)
                except Exception:
                    self.log.exception("boot: cancel_open_orders failed")

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
            f"  State DB       : {_rel_to_repo(state.DB_PATH)}\n"
            f"  Log file       : {_rel_to_repo(LOG_FILE)}\n"
            f"  Heartbeat      : {_rel_to_repo(HEARTBEAT)}\n"
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
        """Return the UTC-day's RAW starting equity, re-anchoring on date rollover.

        Persisted in state.meta so it survives restarts within a day. Unlike
        deploy_start_equity (set once at first deploy), this anchor resets every
        UTC midnight and drives the tighter 2% daily-loss breaker. On each new UTC
        day we ALSO snapshot the cumulative principal-ledger sum
        (daily_anchor_principal_sum) so the breaker can neutralise intraday
        transfers — see _daily_book_anchor. The two ceilings stay separate.
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        anchor_date = state.get_meta("daily_anchor_date")
        anchor_eq = state.get_float("daily_anchor_equity", 0.0)
        if anchor_date != today or anchor_eq <= 0:
            state.set_meta("daily_anchor_date", today)
            state.set_float("daily_anchor_equity", equity)
            # Baseline for intraday-transfer neutralisation: net deposited
            # principal (Part C ledger) as of the anchor. Intraday transfers move
            # the ledger sum; _daily_book_anchor adds the delta back so only
            # trading P&L counts toward the threshold.
            state.set_float("daily_anchor_principal_sum",
                            state.principal_ledger_sum(principal.PRINCIPAL_ASSET))
            self.log.info("Daily anchor set: date=%s equity=%.2f USDT", today, equity)
            return equity
        return anchor_eq

    def _daily_book_anchor(self, equity: float) -> float:
        """Transfer-immune daily baseline (BOOK equity at day start).

        A raw equity anchor misreads an intraday deposit/withdrawal as P&L: a
        deposit inflates equity (can MASK a real trading loss), a withdrawal
        deflates it (can FALSELY trip the breaker). Neutralise both by shifting
        the baseline by the net principal moved since the anchor was set:

            book_anchor = raw_anchor + (ledger_sum_now − ledger_sum_at_anchor)

        A transfer then moves equity and the baseline by the same amount, so only
        true trading P&L counts toward the 2% threshold — consistent with the
        principal-derived kill switch. Reuses the Part C principal ledger.

        Fail-safe to legacy behaviour: an empty/uninitialised ledger gives a 0
        delta (== raw anchor). A legacy anchor set before this migration (no
        baseline key) is seeded to the current sum, so PRE-existing principal is
        never mistaken for an intraday transfer (which would inflate the baseline
        and stop the breaker from ever tripping).
        """
        raw_anchor = self._daily_anchor_equity(equity)
        now_sum = state.principal_ledger_sum(principal.PRINCIPAL_ASSET)
        baseline = state.get_meta("daily_anchor_principal_sum")
        if baseline is None:
            state.set_float("daily_anchor_principal_sum", now_sum)
            return raw_anchor
        try:
            anchor_sum = float(baseline)
        except ValueError:
            anchor_sum = now_sum
        return raw_anchor + (now_sum - anchor_sum)

    def _daily_loss_blocks_entry(self, equity: float) -> bool:
        """True if today's TRADING drawdown has hit MAX_DAILY_LOSS_PCT — block new
        entries for the rest of the UTC day. Does NOT flatten or HALT: existing
        brackets keep managing any open position, and the anchor resets at the
        next UTC midnight (see _daily_anchor_equity). The baseline is transfer-
        immune book equity (_daily_book_anchor), so an intraday deposit/withdrawal
        neither trips nor clears the breaker. Tighter, daily-resetting sibling of
        the principal-anchored kill switch.
        """
        # Fail-safe (mirrors the kill switch): the transfer-immune baseline is
        # derived from Part C's principal ledger. Until that ledger is
        # initialised, principal_ledger_sum() is 0 and a later full-history
        # backfill would look like one huge intraday deposit — inflating the book
        # anchor and falsely blocking entries for the rest of the UTC day. So the
        # breaker stays INACTIVE until P is ready (the same window the kill switch
        # is fail-safe/disabled), and the daily anchor is not snapshotted yet.
        if not principal.is_initialized():
            self._daily_loss_blocked = False
            return False
        day_start = self._daily_book_anchor(equity)
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
        # Pre-clear any stale bot orders before placing a new entry so we
        # don't leave a stale bracket from a prior trade orphaned alongside
        # the new one.  Only cancels orders with our own COID prefix.
        try:
            self.client.cancel_open_orders(self.symbol, coid_prefix=self.coid_prefix)
        except Exception:
            self.log.exception("pre-entry cancel_open_orders failed (continuing)")

        # Channel-exit strategies (donchian-v3) place entry + SL ONLY — the
        # live Donchian channel cross (see _maybe_channel_exit) closes the
        # trade, so there is no TP leg. Every other strategy keeps its TP.
        place_tp = not strategy_uses_channel_exit(self.strategy_name)

        # Remember this trade's bracket params so _maybe_reprotect can restore
        # the SL/TP if they later go missing while the position is still open
        # (external cancel, or a leverage change → Binance auto-cancels orders).
        state.set_meta("active_bracket", json.dumps({
            "signal_id": signal_id, "side": decision.side,
            "entry_price": float(decision.price),
            "sl_distance": float(decision.sl_distance),
            "tp_distance": float(decision.tp_distance),
            "place_tp": bool(place_tp),
        }))

        if self.order_type == "limit":
            limit_price = limit_entry_price(
                decision.side, decision.price, self.limit_offset_bps)
            orders = self.client.limit_order_with_bracket(
                self.symbol, decision.side, qty, limit_price,
                sl_distance=decision.sl_distance, tp_distance=decision.tp_distance,
                timeout_s=self.limit_timeout_s,
                client_order_id_root=signal_id,
                place_tp=place_tp,
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
            place_tp=place_tp,
        )
        trade_events.record_market_entry(
            side=decision.side, qty=qty, price=decision.price,
            sl_price=decision.sl_price, tp_price=decision.tp_price,
            equity=equity, signal_id=signal_id,
            strategy_name=self.strategy_name,
            orders=orders, dbg=decision.debug,
        )

    def _maybe_reprotect(self, equity: float) -> None:
        """Restore a missing SL/TP bracket while a position is still open.

        The bot places a reduce-only bracket at entry, but an EXTERNAL event can
        remove it without the bot knowing — a manual cancel on Binance, or a
        leverage change (Binance auto-cancels ALL open orders on a leverage
        change). _detect_bracket_exit only reacts to a bracket FILL (whether it
        leaves the book flat or is replaced inside one poll by a re-entry),
        never a cancel, so a cancelled bracket would leave the position silently
        unprotected until the time stop — which, measured 2026-08-11, fires 0
        times in 4.6 years. This guard notices the gap and, from the params
        stashed at entry, re-places the bracket (cancel-then-replace so it never
        leaves duplicates), then alerts.

        RE-ENABLED 2026-08-11 with algo-aware detection. It was disabled
        2026-07-22 because `bracket_is_intact(fetch_open_orders(...))` cannot see
        the conditional/algo orders `_place_brackets` actually creates, so it
        read a healthy bracket as missing and re-placed every cooldown until
        `-4045 Reach max stop order limit`. Detection now merges BOTH order books
        via `bracket_state`, and `algo_bracket_leg` classifies the algo shape by
        client-order-id suffix because those rows carry no type field at all
        (see its docstring for the live payload).

        Three independent brakes, because the failure mode is a detector that is
        wrong in the unsafe direction:
          - `reprotect.observe_only` — log the decision, place nothing. The
            rollout runs here until a full live position passes with zero
            "would re-place" lines.
          - `reprotect.max_replaces_per_position` — hard cap per signal_id,
            persisted in active_bracket so a restart cannot reset it. The 60s
            throttle did NOT stop July; it only paced the spam.
          - an empty algo read is treated as UNKNOWN, not as "no bracket" — a
            transient endpoint failure looks identical to a cancelled bracket,
            and acting on it is precisely the July bug.
        """
        if self.dry_run:
            return
        rp = (self.params.get("reprotect") or {})
        if not rp.get("enabled", False):
            return
        pos = self.client.fetch_position(self.symbol)
        if pos.side == "flat" or pos.qty == 0:
            # No position → drop any stale bracket record so it can never be
            # applied to a later, unrelated position (the side + entry-price
            # guards below are the second line of defence).
            state.set_meta("active_bracket", "")
            return
        raw = state.get_meta("active_bracket")
        if not raw:
            return
        try:
            ab = json.loads(raw)
        except (ValueError, TypeError):
            return
        # Only act on stashed params that clearly belong to THIS position —
        # matching side + entry price (2%) — so a stale record never fabricates
        # a bracket at the wrong levels for a different/manual position.
        if ab.get("side") != pos.side:
            return
        ep = float(ab.get("entry_price") or 0.0)
        if ep <= 0 or pos.entry_price <= 0 or abs(ep - pos.entry_price) / pos.entry_price > 0.02:
            return
        place_tp = bool(ab.get("place_tp", True))
        try:
            open_orders = self.client.ex.fetch_open_orders(self.symbol)
        except Exception:
            self.log.exception("reprotect: fetch_open_orders failed")
            return
        # Read BOTH books. fetch_algo_orders never raises, so it reports success
        # separately: `ok=False` means missing endpoint, failed call, or a bad
        # payload. Acting on a read we could not make is the July bug wearing a
        # different hat — a 5-second network blip would re-place a live bracket.
        algo_rows, algo_ok = self.client.fetch_algo_orders(self.symbol)
        if not algo_ok:
            self.log.warning("reprotect: algo book unreadable — skipping "
                             "(cannot distinguish 'no bracket' from 'no answer')")
            return
        state_ = bracket_state(open_orders, algo_rows, self.coid_prefix, place_tp)
        if state_.intact:
            return
        now = time.time()
        if now - self._last_reprotect_ts < 60:
            return

        # Hard cap per position, persisted so a restart cannot reset it. A
        # detector bug must not be able to walk to -4045 again.
        cap = int(rp.get("max_replaces_per_position", 3))
        done = int(ab.get("reprotect_count", 0))
        if done >= cap:
            if not self._reprotect_capped_alerted:
                self._reprotect_capped_alerted = True
                self.log.error("reprotect: hit cap %d for signal_id=%s — STOPPING. %s",
                               cap, ab.get("signal_id"), state_.describe())
                send_alert(
                    "Bracket re-place CAP reached — position may be unprotected",
                    f"{pos.side} {pos.qty:.4f} @ {pos.entry_price:.2f}\n"
                    f"{state_.describe()}\n"
                    f"Re-placed {done} time(s); cap is {cap}. No further attempts "
                    f"will be made for this position — check the bracket manually.")
            return

        self._last_reprotect_ts = now  # throttle (relaxed to a 15s retry on failure below)

        if rp.get("observe_only", True):
            # Rollout phase 1: prove the detector against a real resting bracket
            # without placing anything. Any line here during a healthy position
            # is the July false-negative reappearing.
            self.log.warning(
                "reprotect OBSERVE: WOULD re-place for %s %.4f @ %.2f — %s "
                "(plain=%d algo=%d)",
                pos.side, pos.qty, pos.entry_price, state_.describe(),
                len(open_orders or []), len(algo_rows or []))
            return

        self.log.warning(
            "Bracket MISSING while holding %s %.4f @ %.2f (equity=$%.2f) — re-placing SL%s.",
            pos.side, pos.qty, pos.entry_price, equity, "/TP" if place_tp else "")
        try:
            # Clean slate: cancel our reduce-only orders, then CONFIRM none
            # survive before placing — cancel_open_orders swallows per-order
            # failures, so a partial cancel + blind re-place could leave a stale
            # leg alongside the new pair.
            self.client.cancel_open_orders(self.symbol, coid_prefix=self.coid_prefix)
            # Re-check BOTH books: a plain-only check here would miss a surviving
            # ALGO leg and happily place a duplicate pair beside it — the same
            # blind spot as the detector, on the more dangerous side.
            post_algo, post_ok = self.client.fetch_algo_orders(self.symbol)
            if not post_ok:
                raise RuntimeError(
                    "algo book unreadable after cancel — cannot prove the old "
                    "bracket is gone, so skipping re-place to avoid duplicates")
            survivors = bracket_state(
                self.client.ex.fetch_open_orders(self.symbol),
                post_algo, self.coid_prefix, place_tp)
            if survivors.any_leg:
                raise RuntimeError(
                    "a bracket leg survived cancel — skipping re-place to avoid "
                    f"duplicate orders ({survivors.describe()})")
            self.client._place_brackets(
                self.symbol, pos.side, pos.qty, pos.entry_price,
                float(ab["sl_distance"]), float(ab["tp_distance"]),
                client_order_id_root=ab.get("signal_id"), place_tp=place_tp)
        except Exception as e:
            self.log.exception("reprotect: re-place failed: %s", e)
            # An unprotected live position is urgent — retry ~15s (not the 60s
            # success cooldown, and not every 5s poll).
            self._last_reprotect_ts = now - 45
            send_alert(
                "Bracket re-place FAILED",
                f"{pos.side} {pos.qty:.4f} @ {pos.entry_price:.2f} — automatic SL/TP "
                f"re-placement failed (position may be unprotected): {e}")
            return
        # Persist the attempt count INSIDE active_bracket so the cap survives a
        # restart — an in-memory counter would reset on every boot and let a
        # detector bug resume walking toward -4045.
        ab["reprotect_count"] = done + 1
        state.set_meta("active_bracket", json.dumps(ab))
        state.record_event("WARN", "bracket_reprotect",
                           {"side": pos.side, "qty": pos.qty,
                            "entry": pos.entry_price, "equity": equity,
                            "attempt": done + 1, "cap": cap,
                            "signal_id": ab.get("signal_id")},
                           signal_id=ab.get("signal_id"))
        send_alert(
            "Bracket re-placed",
            f"SL/TP for {pos.side} {pos.qty:.4f} @ {pos.entry_price:.2f} went missing "
            f"(external cancel?) and was automatically restored.")

    # ~5 min at a 5 s poll. Bounded on purpose: fetch_my_trades only returns the
    # last 10 trades, so a retry that ran for hours would eventually be matching
    # against a window that no longer contains the close at all.
    EXIT_RETRY_LIMIT = 60

    def _hold_unrecorded_exit(self, side: str, entry: float | None,
                              qty: float | None, root: str | None) -> None:
        """Keep a DETECTED but UNRECORDED exit alive for a retry — either path.

        If the exit is detected but cannot be written (fetch_my_trades down, the
        closing trade not yet visible on the account, the `fills` read raising),
        advancing the snapshot past that failure loses the exit *permanently*,
        which is precisely the bug this detector was written to end. Putting the
        OLD identity back makes the next tick re-detect the same edge and try
        again.

        Why the snapshot and not the ledger, on each path:
          - REPLACED: the snapshot is the ONLY record the closed position ever
            existed, because `fills` already names the replacement entry.
          - FLAT EDGE: `fills` still names the closed position, so the restored
            snapshot only has to be non-flat — it re-creates the open→flat edge,
            and the retry re-reads `fills` for pricing.

        Known limitation, left deliberately: if the REPLACEMENT position also
        closes while a retry is outstanding, the next tick takes the flat path
        and prices the exit off the replacement's `fills` row. That mis-prices a
        double failure rather than losing it, which is the better of the two.
        """
        self._exit_retry_n += 1
        if self._exit_retry_n > self.EXIT_RETRY_LIMIT:
            attempts = self._exit_retry_n - 1
            self._exit_retry_n = 0
            self.log.error(
                "bracket-exit: GAVE UP recording the exit of %s %.4f @ %.2f "
                "(root=%s) after %d attempts — the ledger is missing this close",
                side, qty or 0.0, entry or 0.0, root, attempts)
            send_alert(
                "Bot exit NOT recorded",
                f"A {side.upper()} {qty or 0.0:.4f} {self.base_asset} position "
                f"closed and its exit could not be recorded after {attempts} "
                f"attempts.\n"
                f"Entry: {entry or 0.0:,.2f}  signal_id: {root or '(untagged)'}\n"
                f"The ledger is missing this close — it needs a manual backfill.",
            )
            return
        self._last_position_side = side
        self._last_position_entry = entry
        self._last_position_qty = qty
        self._last_entry_root = root

    def _detect_bracket_exit(self, equity: float) -> None:
        """Bracket SL/TP fills close the position on Binance's side without
        a bot-initiated close. Detect that the tracked position has ENDED and
        emit an exit alert with PnL.

        Two ways it can end, and they are priced from different sources:

          - FLAT EDGE (open→flat). Sweeps the surviving bracket sibling, then
            prices the exit from the latest `fills` row, which must be
            `reason='entry'` — i.e. no bot-initiated close (time_stop, kill,
            halt) has been recorded since the entry.

          - REPLACED (open→open, different position). The bracket filled and
            _maybe_enter re-entered inside one poll, so the loop never sees
            `flat`. Detected by a change in `entry_price`, priced from the
            in-memory snapshot, and it must NOT sweep — see the branch comment.
            Before this existed the exit was dropped entirely: no `close` row,
            no `exit` event, no alert (live 2026-08-23, −$5.37).

        Also required either way: not dry-run (no real brackets in dry-run
        mode), and a matching opposite-side trade on Binance.

        BOTH paths are retried. Once either has decided the position ended, the
        exit is owed, and every failure before it is written — a raising sqlite
        read, an unmatched or empty trade list, a zero fill price, any ccxt
        error — re-arms `retain` so the next tick re-detects instead of losing
        the close. A branch that decides no exit is owed (a bot-initiated close
        already in the ledger) disarms explicitly. Bounded by EXIT_RETRY_LIMIT
        ticks, then it alerts rather than giving up silently.
        """
        if self.dry_run:
            return
        # Set once an exit is DETECTED on EITHER path — replacement since #26,
        # flat edge since the 2026-09-04 donchian loss — and cleared the moment
        # it is written, or when a branch decides no exit is owed at all.
        # Anything still holding it at `finally` is an exit we found and failed
        # to record — see _hold_unrecorded_exit.
        retain: tuple[str, float | None, float | None, str | None] | None = None
        try:
            pos = self.client.fetch_position(self.symbol)
            current = pos.side

            prev_side = self._last_position_side
            prev_entry = self._last_position_entry
            prev_qty = self._last_position_qty
            prev_root = self._last_entry_root
            had_open = prev_side not in ("flat", "unknown")

            # A position we were tracking can disappear two ways:
            #   flat      — the classic open→flat edge.
            #   REPLACED  — still open, but it is not the same position: the
            #               bracket filled and _maybe_enter re-entered inside a
            #               single poll (live 2026-08-23, 1.25 s apart).
            # entry_price is the discriminator, compared at float-noise
            # tolerance: Binance echoes the identical number for an untouched
            # position, while a re-entry seconds later can be only a few dollars
            # away — so a percentage threshold would miss exactly the case this
            # branch exists to catch. Both prices must be non-zero, or a missing
            # `entryPrice` field would fabricate an exit out of nothing.
            replaced = (
                had_open
                and current != "flat"
                and prev_entry is not None
                and prev_entry > 0.0
                and pos.entry_price > 0.0
                and (current != prev_side
                     or abs(pos.entry_price - prev_entry)
                     > 1e-9 * max(prev_entry, pos.entry_price))
            )

            # Refresh the snapshot BEFORE any early return, so every path leaves
            # it agreeing with what the exchange just said.
            self._last_position_side = current
            if current == "flat":
                self._last_position_entry = None
                self._last_position_qty = None
                self._last_entry_root = None
            else:
                self._last_position_entry = pos.entry_price
                self._last_position_qty = pos.qty
                if replaced or not had_open:
                    # Only correct at the moment the position opens; one tick
                    # later this already names the replacement entry.
                    self._last_entry_root = state.latest_entry_coid_root()

            # First observation since boot — initialise state without emitting.
            # A historical "entry" row in state.db whose position already closed
            # on the exchange must not trigger an alert.
            if prev_side == "unknown" or not had_open:
                return
            if current != "flat" and not replaced:
                return

            if replaced:
                # ⚠️ NO SWEEP ON THIS PATH, AND DO NOT "UNIFY" IT WITH THE ONE
                # BELOW. _maybe_enter → _place_live_entry has already placed a
                # fresh bracket for the position that is open RIGHT NOW, and
                # cancel_open_orders is scoped to the whole leg by coid prefix —
                # calling it here would strip the SL and TP off a live
                # real-money position, which is strictly worse than the dropped
                # exit this branch fixes.
                #
                # The fills table cannot price this exit either: _maybe_enter
                # runs last in the tick, so its `entry` row is already the most
                # recent one by the time we get here. The in-memory snapshot is
                # the only record of the position that just closed.
                entry_side = prev_side
                entry_qty = float(prev_qty or 0.0)
                entry_price = float(prev_entry)
                signal_id = prev_root
                if entry_qty <= 0.0:
                    self.log.warning(
                        "bracket-exit: replacement %s→%s but no tracked qty; "
                        "cannot price the exit", prev_side, current)
                    return
                # ⚠️ A BOT-INITIATED close can share a tick with a re-entry too,
                # and without this guard that would be written TWICE. The tick
                # order is _detect_bracket_exit → _maybe_time_stop →
                # _maybe_channel_exit → _maybe_enter, so on one tick the channel
                # or trend exit can record its own close and _maybe_enter can
                # then open at a new price (it never advanced _last_signal_ts
                # while the position was open, so the bar guard lets it through).
                # The next tick sees a changed entry_price and would emit a
                # SECOND exit for a position already closed in the ledger —
                # corrupting it in the opposite direction from the dropped exit
                # this branch fixes. The classic path gets this for free from its
                # `reason='entry'` check on the newest row; the replacement path
                # has to ask explicitly, because the newest row is the re-entry.
                with sqlite3.connect(state.DB_PATH) as c:
                    anchor = c.execute(
                        "SELECT id FROM fills WHERE reason = 'entry' "
                        "AND client_order_id_root IS ? ORDER BY id DESC LIMIT 1",
                        (signal_id,),
                    ).fetchone()
                    if anchor:
                        already_closed = c.execute(
                            "SELECT 1 FROM fills WHERE id > ? AND reason <> 'entry' "
                            "LIMIT 1", (anchor[0],),
                        ).fetchone() is not None
                    else:
                        # Untagged entry, so there is nothing to anchor on. Fall
                        # back to the row just BEFORE the newest one (the newest
                        # being the re-entry): if that is a close, it is ours.
                        prior = c.execute(
                            "SELECT reason FROM fills WHERE id < "
                            "(SELECT MAX(id) FROM fills) ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                        already_closed = bool(prior) and prior[0] != "entry"
                if already_closed:
                    self.log.info(
                        "bracket-exit: replacement %s→%s, but this position "
                        "already has a recorded close — not double-writing",
                        prev_side, current)
                    return
                retain = (prev_side, prev_entry, prev_qty, prev_root)
            else:
                # Arm the retry BEFORE anything that can fail. The flat edge is
                # already confirmed, so the position HAS ended and an exit is
                # owed; from here every early exit — a raising sqlite read, the
                # trade lookup below, a zero fill price — is a failure to record
                # something real. The snapshot advanced to flat at the top of
                # this method, so returning without retaining loses it for good.
                # The replacement path has been retried since #26; the flat edge
                # — the older and far more common one — never was.
                # Live 2026-09-04: donchian's algo SL filled at 12:31:49 and
                # fetch_my_trades still did not list the closing trade at
                # 12:32:04, so the single lookup missed, this branch returned,
                # and the ledger lost a −$4.78 close (no fills row, no exit
                # event, no alert) until it was backfilled by hand.
                #
                # prev_* (the exchange snapshot of the position that just
                # closed) rather than the `fills` row, precisely so the arming
                # happens before the read that can raise. It is also all the
                # next tick needs: restoring a non-flat side re-creates this
                # same edge, and the retry re-reads `fills` for pricing.
                retain = (prev_side, prev_entry, prev_qty, prev_root)

                # Sweep the surviving bracket leg (the sibling that Binance did NOT
                # fill).  Do this before the PnL lookup so even if the fill-lookup
                # errors out, the orphan is still cancelled.
                try:
                    self.client.cancel_open_orders(self.symbol, coid_prefix=self.coid_prefix)
                except Exception:
                    self.log.exception("bracket-exit: cancel_open_orders failed")

                with sqlite3.connect(state.DB_PATH) as c:
                    row = c.execute(
                        "SELECT reason, side, qty, price, client_order_id_root "
                        "FROM fills ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                if not row or row[0] != "entry":
                    # DELIBERATE skip, not a failure: a bot-initiated close
                    # (time_stop, kill, halt) is already in the ledger, so no
                    # exit is owed. Disarm, or this holds the snapshot open for
                    # EXIT_RETRY_LIMIT ticks over nothing and then alerts.
                    retain = None
                    return
                _, entry_side, entry_qty, entry_price, signal_id = row
                entry_qty = float(entry_qty)
                entry_price = float(entry_price)

            opposite = "sell" if entry_side == "long" else "buy"
            trades = self.client.ex.fetch_my_trades(self.symbol, limit=10)
            if replaced and current != prev_side:
                # A flip closes a long with a SELL and opens the short with a
                # SELL too, so the newest opposite-side trade may be the new
                # ENTRY rather than our close. Both land inside one poll, so any
                # mispricing is bounded by a second or two of movement — worth
                # logging loudly and recording, not worth guessing at.
                self.log.warning(
                    "bracket-exit: side-flip replacement %s→%s; exit price may "
                    "name the new entry (prev %.4f @ %.2f, now %.4f @ %.2f)",
                    prev_side, current, entry_qty, entry_price,
                    pos.qty, pos.entry_price)
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
            # Durable now — stop holding the old snapshot open for a retry.
            retain = None
            self._exit_retry_n = 0
            state.enqueue_bot_event(
                "exit", signal_id=signal_id, side=entry_side,
                qty=float(entry_qty), price_usd=float(exit_price),
                equity_usd=float(equity),
                payload={"reason": "bracket_exit",
                         "detected_by": "replacement" if replaced else "flat_edge",
                         "entry_price": float(entry_price),
                         "exit_price": float(exit_price),
                         "pnl_usd": float(pnl),
                         "pnl_pct": float(pnl_pct)},
            )
            send_alert(
                f"Bot {entry_side.upper()} exit",
                f"{entry_side.upper()} {entry_qty:.4f} {self.base_asset} closed "
                f"by bracket (SL or TP)\n"
                f"Entry: {entry_price:,.2f}  Exit: {exit_price:,.2f}\n"
                f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                f"Equity now: ${equity:,.2f}\n"
                f"signal_id: {signal_id or '(untagged)'}",
            )
        except Exception as e:
            self.log.warning("bracket-exit detection failed: %s", e)
        finally:
            if retain is not None:
                self._hold_unrecorded_exit(*retain)

    def _resolve_fill_price(self, order: dict | None,
                            attempts: int = 3,
                            delay_s: float = 0.4) -> float | None:
        """True average fill price of an order the bot just placed.

        Binance's CREATE response for a reduce-only MARKET order carries
        avgPrice "0.00" -- the fill is not attributed to the order yet -- so
        `order_avg_price()` returns None at EVERY bot-initiated close. Measured
        2026-08-23: every `close` row in all three legs' fills tables since
        launch has price=0.0, and the boot-flatten path's
        `order_avg_price(o) or pos.entry_price` fallback therefore always
        reported PnL of exactly 0.

        Re-fetching the SAME order by clientOrderId returns the real average --
        verified against both real closes that day (v1
        snap-v1-1787462070917-ce -> 76003.10, donchian
        snap-d3-1787256026107-ce -> 76010.00). Keying on the order is
        authoritative; scanning fetch_my_trades and side-matching (what
        _detect_bracket_exit does, correctly, a tick later) can pick up an
        unrelated fill when run immediately after placing.

        Returns None if the price cannot be established, leaving the decision
        to the caller. NEVER raises: the exchange has already executed the
        close, and bookkeeping must not turn that into an exception.
        """
        px = order_avg_price(order)
        if px:
            return px
        o = order or {}
        coid = str(o.get("clientOrderId") or "")
        oid = str(o.get("id") or "")
        if not coid and not oid:
            return None
        params = {"origClientOrderId": coid} if coid else {}
        for i in range(max(1, attempts)):
            if i:
                time.sleep(delay_s)
            try:
                fetched = self.client.ex.fetch_order(oid or None, self.symbol,
                                                     params=params)
            except Exception:
                # Do NOT retry a failed CALL. ccxt's timeout is 15s, so three
                # attempts would block the tick loop ~46s -- and this path is
                # pure bookkeeping running AFTER the exchange has already closed
                # the position. A stalled loop is worse than a missing price,
                # which is recoverable from order history later. Retries exist
                # only for the "fetched, but the fill is not attributed yet"
                # case handled below.
                self.log.warning("exit-price: fetch_order failed for %s -- "
                                 "not retrying (bookkeeping must not stall the "
                                 "loop)", coid or oid)
                return None
            px = order_avg_price(fetched)
            if px:
                return px
        self.log.warning("exit-price: could not resolve fill price for %s -- "
                         "recording without it", coid or oid)
        return None

    def _open_entry_fill(self) -> tuple[str, float, float] | None:
        """(side, qty, price) of the entry for the position that is OPEN NOW.

        Reads the most recent fill of ANY kind and requires it to be an entry,
        exactly as _detect_bracket_exit does. Filtering on `reason='entry'`
        instead would skip BACKWARDS over intervening closes and hand back an
        entry that is already closed -- attributing a stale entry price to the
        current position and writing a wrong number into the trade record.

        That is reachable: a position adopted at boot (stale_position_at_boot)
        or opened manually has no entry row of its own, so the newest fill is
        the PREVIOUS trade's close. Returning None there is correct -- the
        caller records the exit without a PnL rather than with a false one.
        """
        try:
            with sqlite3.connect(state.DB_PATH) as c:
                row = c.execute(
                    "SELECT reason, side, qty, price FROM fills "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
        except Exception:
            self.log.exception("exit-pnl: entry-fill lookup failed")
            return None
        if not row or row[0] != "entry":
            return None
        return str(row[1]), float(row[2]), float(row[3])

    def _maybe_time_stop(self, equity: float) -> None:
        # Read the config BEFORE fetch_position so a leg with no time stop
        # costs zero API calls per tick instead of one. That is supertrend
        # (max_hold_bars: 0); donchian-v3 sets 48 and DOES pay the fetch.
        max_hold = int((self.params.get("strategy") or {}).get("max_hold_bars", 0))
        if max_hold <= 0:
            return
        pos = self.client.fetch_position(self.symbol)
        if pos.side == "flat":
            return
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
            if time_stop_due(max_hold, self.bar_seconds, age_s):
                if self.dry_run:
                    self.log.info("DRY-RUN: would time-stop close after %.1f hours", age_s / 3600)
                    return
                root = state.latest_entry_coid_root()
                self.log.info("Time-stop firing after %.1f hours (root=%s)",
                              age_s / 3600, root or "—")
                order = self.client.close_position(
                    self.symbol, client_order_id_root=root, close_leg="x")
                exit_price = self._resolve_fill_price(order)
                _entry = self._open_entry_fill()
                state.record_fill(side="close", qty=pos.qty,
                                  price=float(exit_price or 0.0),
                                  pnl_usd=_exit_pnl_usd(
                                      pos.side, _entry[2] if _entry else None,
                                      exit_price, pos.qty),
                                  reason="time_stop", equity_after=equity,
                                  client_order_id_root=root)
                state.enqueue_bot_event(
                    "exit", signal_id=root, side=pos.side, qty=float(pos.qty),
                    price_usd=exit_price,
                    equity_usd=float(equity),
                    payload={"reason": "time_stop", "age_h": age_s / 3600},
                )
                send_alert("Bot time-stop close",
                           f"Closed {pos.side} {pos.qty:.4f} {self.base_asset} after "
                           f"{age_s/3600:.1f}h hold"
                           + (f" @ {exit_price:,.4f}" if exit_price else "")
                           + f". Equity: {equity:.2f}\n"
                           f"signal_id: {root or '(untagged)'}")
        except Exception as e:
            self.log.warning("time-stop check failed: %s", e)

    def _maybe_channel_exit(self, equity: float) -> None:
        """The generic live trend-exit hook. Name is historical — donchian-v3
        was the first leg to need it; three legs use it now, each with its own
        rule (see bot_internals.trend_exit_signal):

          - donchian-v3:   opposite Donchian exit-channel cross on 4h. Its ONLY
                           profit-taking mechanism — the entry places no TP leg.
          - supertrend:    opposite STDir flip on 4h, alongside its TP bracket.
          - multifactor-v1: adverse 15m EMA(200) cross, alongside its TP bracket
                           (added 2026-08-10 — see MULTIFACTOR_V1_LIVE_EXIT_VERDICT.md).

        Strategy-gated: a no-op for every leg not in strategy_uses_trend_exit
        (cnh, v3all), which keep their brackets and never reach this code.

        Structurally mirrors _maybe_time_stop: fetch position, evaluate, and on
        a trigger close reduce-only (COID-tagged, close_leg='ce'), record the
        fill, enqueue the exit event, alert. The surviving bracket sibling is
        swept by close_position's own COID-scoped cancel_open_orders before the
        reduce-only close is placed.

        Bars are fetched on self.entry_tf, so each leg is evaluated on the
        timeframe its own rule is defined on (4h for donchian/supertrend, 15m
        for v1) without any per-strategy branching here.
        """
        if not strategy_uses_trend_exit(self.strategy_name):
            return
        try:
            pos = self.client.fetch_position(self.symbol)
            if pos.side == "flat":
                return
            df = self.client.fetch_ohlcv(self.symbol, self.entry_tf, limit=1500)
            # Evaluate CLOSED bars only. Binance returns the still-forming
            # current bar as the last row; the backtest only ever sees closed
            # bars, so drop it here to keep the channel-exit decision in parity.
            # No extra length precheck: channel_exit_signal carries the
            # backtest's own warmup/NaN guard (max(entry, exit, atr)+1 bars),
            # and suppressing an EXIT on a short fetch is worse than checking
            # (Sourcery, PR #7).
            df = df.iloc[:-1]
            should_exit, dbg = trend_exit_signal(
                self.strategy_name, df, pos.side, self.params)
            if not should_exit:
                return
            if self.dry_run:
                self.log.info("DRY-RUN: would trend-exit close %s %.4f (%s)",
                              pos.side, pos.qty, dbg.get("reason"))
                return
            root = state.latest_entry_coid_root()
            fill_reason = trend_exit_fill_reason(self.strategy_name)
            self.log.info("Trend-exit firing (%s): closing %s (root=%s) rule=%s "
                          "close=%.2f dbg=%s",
                          fill_reason, pos.side, root or "—", dbg.get("reason"),
                          dbg.get("cur_close", float("nan")), dbg)
            order = self.client.close_position(
                self.symbol, client_order_id_root=root, close_leg="ce")
            exit_price = self._resolve_fill_price(order)
            _entry = self._open_entry_fill()
            state.record_fill(side="close", qty=pos.qty,
                              price=float(exit_price or 0.0),
                              pnl_usd=_exit_pnl_usd(
                                  pos.side, _entry[2] if _entry else None,
                                  exit_price, pos.qty),
                              reason=fill_reason, equity_after=equity,
                              client_order_id_root=root)
            state.enqueue_bot_event(
                "exit", signal_id=root, side=pos.side, qty=float(pos.qty),
                price_usd=exit_price,
                equity_usd=float(equity),
                # Rule-specific keys are absent per strategy (donchian has no
                # trend_ema, v1 has no channel) — .get() yields None, which the
                # dashboard renders as blank rather than a wrong number.
                payload={"reason": fill_reason,
                         "rule": dbg.get("reason"),
                         "cur_close": dbg.get("cur_close"),
                         "exit_lower": dbg.get("exit_lower"),
                         "exit_upper": dbg.get("exit_upper"),
                         "trend_ema": dbg.get("trend_ema")},
            )
            send_alert("Bot trend-exit close",
                       f"Closed {pos.side} {pos.qty:.4f} {self.base_asset} on "
                       f"{dbg.get('reason', 'trend exit')}"
                       + (f" @ {exit_price:,.4f}" if exit_price else "")
                       + f". Equity: {equity:.2f}\n"
                       f"signal_id: {root or '(untagged)'}")
        except Exception as e:
            self.log.warning("channel-exit check failed: %s", e)

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
                self._maybe_channel_exit(equity)
                # RE-ENABLED 2026-08-11 with algo-aware detection (bracket_state
                # merges /fapi/v1/openOrders AND /fapi/v1/openAlgoOrders). It was
                # disabled 2026-07-22 because the plain-only check read a healthy
                # conditional bracket as missing and re-placed to -4045. Gated on
                # `reprotect.enabled`, and ships `observe_only: true` — it logs
                # the decision and places nothing until a full live position
                # passes with zero "WOULD re-place" lines.
                self._maybe_reprotect(equity)
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
    # Fail loudly if a NON-v1 leg has no per-instance env file.
    #
    # load_env_for_instance returns None when `.env.<instance>` is absent, which
    # is correct for v1 (the base .env IS its env) but a live-money hazard for
    # every other leg: without the overlay the leg silently inherits v1's
    # BINANCE_API_KEY and trades a different symbol inside v1's account, pushes
    # telemetry under v1's CONSOLIDATE_SOURCE, and emails under v1's ALERT_TAG.
    # env.py's own docstring names this as the thing the overlay exists to
    # prevent, but nothing enforced it. Enforced here, before any client is
    # constructed.
    #
    # Not exempted for --dry-run on purpose: a dry run against the wrong account
    # reads the wrong equity (so sizing and the min-notional check are both
    # meaningless) and still emits events under the wrong source.
    if args.instance != "v1" and instance_env_path is None:
        print(
            f"FATAL: instance {args.instance!r} has no .env.{args.instance}.\n"
            f"  Without it this leg would use the BASE .env — i.e. v1's Binance\n"
            f"  API key and account — and place {args.instance} orders there.\n"
            f"  Create {REPO_ROOT / f'.env.{args.instance}'} holding that leg's own\n"
            f"  sub-account BINANCE_API_KEY / BINANCE_API_SECRET, plus\n"
            f"  CONSOLIDATE_SOURCE=... and ALERT_TAG=... so its dashboard events\n"
            f"  and alert emails do not collide with v1's.",
            file=sys.stderr,
        )
        return 2
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
