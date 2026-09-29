#!/usr/bin/env python3
"""Read-only boot-resume probe — "if this leg restarted RIGHT NOW, would it adopt?"

    ./.venv/bin/python tools/boot_resume_probe.py <instance>
        instance: v1 | donchian | sol_supertrend     (the INSTANCE, not the unit)

This is the counted evidence for arming boot-resume
(docs/BOOT_RESUME_PHASE_B_PLAN.md §2). It evaluates the SAME pure gate that
`Bot._can_adopt` calls (`bot_internals.can_adopt`) against live exchange state
and the leg's state DB, and prints the verdict, the reason, and every input.

READ-ONLY, ENFORCED STRUCTURALLY — not by care:
  - The ccxt exchange is wrapped in `ReadOnlyExchange`, a WHITELIST proxy.
    Every BinanceClient method goes through `self.ex`, so any order, cancel,
    leverage or other non-whitelisted call raises `ReadOnlyViolation` BEFORE it
    reaches ccxt. The violation derives from BaseException so a mutator's own
    broad `except Exception` cannot swallow it.
  - The client itself is wrapped in `ReadOnlyClient` (whitelist again).
  - The state DB is opened `file:...?mode=ro` with `PRAGMA query_only=ON`, so
    SQLite itself refuses writes, and a missing DB is an error (never created).
  - It never constructs `Bot`, never calls `bot.main()` or `_setup_logging()`.
    It prints to stdout only — otherwise its lines would land in the leg's
    bot.jsonl and be mistaken for organic `boot-resume OBSERVE` evidence.

Verdicts:
  ADOPT    gate says adopt AND the armed-path checks (dedup-bar seed, D5 guard)
           would pass.
  REFUSE   gate or an armed-path check refuses → a restart flattens.
  FLATTEN  boot_resume is disabled/absent for this leg → a restart flattens
           without consulting the gate at all.
  FLAT     no open position; a restart just sweeps orphans.
The `gate:` line prints the exact reason string `_can_adopt` returns, so it is
byte-comparable with a boot's `boot-resume OBSERVE: WOULD ... — <reason>` line.

Exit codes: 0 evaluated; 2 usage / env error; 3 read failure.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROBE_INSTANCES = ("v1", "donchian", "sol_supertrend")


class ReadOnlyViolation(BaseException):
    """A write was attempted through the probe's client.

    BaseException, NOT Exception and NOT AttributeError: BinanceClient mutators
    wrap their exchange calls in `except Exception` (set_leverage,
    cancel_algo_by_coid, ...), and `getattr(ex, name, None)` swallows
    AttributeError. Either would turn a blocked write into a silent no-op
    instead of a loud failure.
    """


class ReadOnlyExchange:
    """Whitelist proxy over a ccxt exchange. Anything not named here raises."""

    ALLOWED = frozenset({
        "fetch_positions",               # BinanceClient.fetch_position
        "fetch_open_orders",             # plain order book
        "fapiPrivateGetOpenAlgoOrders",  # algo order book (GET)
        "market", "markets",             # local market metadata
        "parse_timeframe",               # local helper
        "id",
    })

    def __init__(self, ex: Any) -> None:
        object.__setattr__(self, "_ex", ex)
        object.__setattr__(self, "violations", [])

    def __getattr__(self, name: str) -> Any:
        if name in ReadOnlyExchange.ALLOWED:
            return getattr(self._ex, name)
        self.violations.append(name)
        raise ReadOnlyViolation(f"read-only probe: exchange.{name} is not permitted")

    def __setattr__(self, name: str, value: Any) -> None:
        raise ReadOnlyViolation(f"read-only probe: cannot set exchange.{name}")


class ReadOnlyClient:
    """Whitelist proxy over BinanceClient: only the reads the gate needs."""

    ALLOWED = frozenset({"fetch_position", "fetch_algo_orders", "ex", "env",
                         "coid_prefix", "hedge_mode"})

    def __init__(self, client: Any) -> None:
        if not isinstance(client.ex, ReadOnlyExchange):
            raise ReadOnlyViolation("ReadOnlyClient requires a ReadOnlyExchange")
        object.__setattr__(self, "_c", client)

    def __getattr__(self, name: str) -> Any:
        if name in ReadOnlyClient.ALLOWED:
            return getattr(self._c, name)
        raise ReadOnlyViolation(f"read-only probe: client.{name} is not permitted")

    def __setattr__(self, name: str, value: Any) -> None:
        raise ReadOnlyViolation(f"read-only probe: cannot set client.{name}")


def open_state_ro(path: Path) -> sqlite3.Connection:
    """The leg's state DB, read-only at the SQLite level. Never creates it."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _one(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> Any:
    row = conn.execute(sql, args).fetchone()
    return row[0] if row else None


@dataclass
class ProbeResult:
    verdict: str
    reason: str
    gate: tuple[bool, str] | None = None
    inputs: dict[str, Any] = field(default_factory=dict)


def evaluate(params: dict, client: Any, conn: sqlite3.Connection,
             now_s: float | None = None) -> ProbeResult:
    """Run the boot's adopt decision, read-only. `client` must be a ReadOnlyClient."""
    from bot_internals import adopt_loop_guard, bracket_state, can_adopt, signal_bar_seed
    from exchange import state
    from exchange.constraints import fallbacks_for_symbol, merge_with_live

    if not isinstance(client, ReadOnlyClient):
        raise ReadOnlyViolation("evaluate() only accepts a ReadOnlyClient")
    symbol = str(params["symbol"])
    hedge_cfg = params.get("hedge") or {}
    coid_prefix = str(hedge_cfg.get("client_order_id_prefix", "snap-v1-"))
    br = params.get("boot_resume") or {}
    entry_tf = str((params.get("timeframes") or {}).get("entry", "15m"))

    inputs: dict[str, Any] = {
        "symbol": symbol, "env": client.env, "coid_prefix": coid_prefix,
        "boot_resume": {"enabled": bool(br.get("enabled", False)),
                        "observe_only": bool(br.get("observe_only", True))},
        "reprotect": params.get("reprotect") or {},
    }

    pos = client.fetch_position(symbol)
    inputs["position"] = {"side": pos.side, "qty": pos.qty,
                          "entry_price": pos.entry_price}
    if pos.side == "flat":
        return ProbeResult("FLAT", "no open position — a restart sweeps orphans only",
                           None, inputs)

    # Same constraint resolution as Bot.boot(): tighter of fallback vs live.
    try:
        qty_step = merge_with_live(fallbacks_for_symbol(symbol),
                                   client.ex.market(symbol)).qty_step
    except Exception:
        qty_step = fallbacks_for_symbol(symbol).qty_step
    inputs["qty_step"] = qty_step

    ab_raw = _one(conn, state.META_GET_SQL, ("active_bracket",))
    latest_root = _one(conn, state.LATEST_ENTRY_ROOT_SQL)
    entry_ts = _one(conn, state.LATEST_ENTRY_TS_SQL)
    last_bar = _one(conn, state.META_GET_SQL, ("last_entry_bar_ts",))
    adopt_log = _one(conn, state.META_GET_SQL, ("boot_adopt_log",))
    inputs.update({"active_bracket": ab_raw, "latest_entry_root": latest_root,
                   "latest_entry_fill_ts": entry_ts,
                   "last_entry_bar_ts": last_bar, "boot_adopt_log": adopt_log})
    try:
        ab_sid = (json.loads(ab_raw) or {}).get("signal_id") if ab_raw else None
    except (ValueError, TypeError, AttributeError):
        ab_sid = None
    # Checklist §8.5(c) — informational; the gate does NOT check it (C5 not built).
    inputs["active_bracket_sid_matches_latest_root"] = (
        ab_sid is not None and ab_sid == latest_root)

    # Books are read unconditionally (the boot skips them when the precheck
    # already refused); can_adopt re-runs the precheck, so the verdict is the same.
    try:
        open_orders = client.ex.fetch_open_orders(symbol)
    except ReadOnlyViolation:
        raise
    except Exception as e:
        inputs["plain_book_error"] = f"{type(e).__name__}"
        open_orders = None
    algo_rows, algo_ok = client.fetch_algo_orders(symbol)
    inputs["algo_ok"] = algo_ok
    ours = [r for r in (algo_rows or [])
            if str(r.get("clientAlgoId") or "").startswith(coid_prefix)]
    inputs["our_algo_rows"] = [
        {k: r.get(k) for k in ("clientAlgoId", "side", "triggerPrice",
                               "reduceOnly", "algoStatus")} for r in ours]
    inputs["plain_rows"] = len(open_orders or [])
    inputs["bracket"] = bracket_state(open_orders, algo_rows, coid_prefix, True).describe()

    gate = can_adopt(params, pos, ab_raw, open_orders, algo_rows, algo_ok,
                     qty_step, coid_prefix)

    # What the ARMED path would additionally check (bot._prepare_adopt).
    bar_seconds = int(client.ex.parse_timeframe(entry_tf))
    seed = signal_bar_seed(last_bar, entry_ts, bar_seconds)
    inputs["dedup_seed"] = seed.isoformat() if seed is not None else None
    allowed, gwhy, _ = adopt_loop_guard(adopt_log, latest_root,
                                        time.time() if now_s is None else now_s)
    inputs["loop_guard"] = gwhy

    if not inputs["boot_resume"]["enabled"]:
        return ProbeResult("FLATTEN", "boot_resume disabled", gate, inputs)
    if not gate[0]:
        return ProbeResult("REFUSE", gate[1], gate, inputs)
    if seed is None:
        return ProbeResult("REFUSE", "no entry-dedup bar to seed (no "
                           "last_entry_bar_ts and no entry fill)", gate, inputs)
    if not allowed:
        return ProbeResult("REFUSE", gwhy, gate, inputs)
    return ProbeResult("ADOPT", gate[1], gate, inputs)


def render(instance: str, res: ProbeResult) -> str:
    lines = [f"boot-resume probe — instance={instance} (read-only)",
             f"VERDICT: {res.verdict} — {res.reason}"]
    if res.gate is not None:
        lines.append(f"gate: {'ADOPT' if res.gate[0] else 'REFUSE'} — {res.gate[1]}")
    br = res.inputs.get("boot_resume") or {}
    if res.verdict != "FLAT":
        if not br.get("enabled"):
            now = "FLATTEN (boot_resume disabled)"
        elif br.get("observe_only", True):
            now = "FLATTEN (observe_only: verdict is logged, then the position is closed)"
        else:
            now = "ADOPT" if res.verdict == "ADOPT" else "FLATTEN"
        lines.append(f"a restart right now would: {now}")
    lines.append("inputs:")
    for k, v in res.inputs.items():
        lines.append(f"  {k}: {json.dumps(v, default=str)}")
    return "\n".join(lines)


def build_readonly_client(params: dict) -> ReadOnlyClient:
    """A BinanceClient whose exchange cannot write, then wrapped again."""
    from exchange.binance_client import BinanceClient

    hedge_cfg = params.get("hedge") or {}
    hedge = bool(hedge_cfg.get("enabled", False))
    prefix = str(hedge_cfg.get("client_order_id_prefix", "snap-v1-"))
    raw = BinanceClient.from_env(hedge_mode=hedge, coid_prefix=prefix)
    guarded = BinanceClient(ex=ReadOnlyExchange(raw.ex), env=raw.env,
                            hedge_mode=hedge, coid_prefix=prefix)
    return ReadOnlyClient(guarded)


def resolve_instance(instance: str) -> tuple[dict, Path]:
    """(params, state_db) through the SAME INSTANCE_PROFILES main() uses,
    without calling main(). Also applies main()'s per-instance env rule."""
    import bot as botmod  # module import only: no Bot(), no main(), no logging setup
    from exchange.env import load_env_for_instance

    profile = botmod.INSTANCE_PROFILES[instance]
    env_path = load_env_for_instance(instance)   # raises for a keyless sub-account leg
    if instance != "v1" and env_path is None:
        raise SystemExit(2)
    return botmod.load_params(profile["config"]), Path(profile["state_db"])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in PROBE_INSTANCES:
        print(f"usage: boot_resume_probe.py <{'|'.join(PROBE_INSTANCES)}>\n"
              "  takes the INSTANCE name (v1), not the systemd unit (snapback-btc)",
              file=sys.stderr)
        return 2
    instance = argv[0]
    try:
        params, db = resolve_instance(instance)
    except SystemExit as e:
        print(f"FATAL: {instance} has no .env.{instance}; refusing to read "
              "another account's exchange state against this leg's DB.",
              file=sys.stderr)
        return int(e.code or 2)
    except Exception as e:
        print(f"FATAL: could not resolve instance {instance}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    try:
        conn = open_state_ro(db)
    except sqlite3.Error as e:
        print(f"FATAL: cannot open {db} read-only: {e}", file=sys.stderr)
        return 3
    try:
        client = build_readonly_client(params)
        res = evaluate(params, client, conn)
    except ReadOnlyViolation:
        raise
    except Exception as e:
        # Type only: a ccxt error message can echo the signed request.
        print(f"FATAL: read failed: {type(e).__name__}", file=sys.stderr)
        return 3
    finally:
        conn.close()
    print(render(instance, res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
