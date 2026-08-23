#!/usr/bin/env python3
"""Restart ONE snapback leg, but only while it is provably flat and clean.

    ./.venv/bin/python /root/leg_safe_restart.py <instance>

WHY THIS EXISTS. boot() FLATTENS any open position in live mode, so restarting
a leg mid-trade silently closes it at market. /root/v1_deferred_restart.py
enforced that check for v1 only -- donchian and sol_supertrend had no sanctioned
path at all, so they got raw `systemctl restart`, which performs no check
whatsoever. This generalises the v1 guard to every leg.

FLAT means all three, checked against the EXCHANGE, not the local DB:
  - position side == flat
  - zero plain open orders
  - zero resting ALGO orders  (SL/TP live on /fapi/v1/openAlgoOrders, which
    fetch_open_orders does NOT return -- the blind spot behind the July -4045
    spam. Checking only plain orders would call "flat" with a bracket resting.)
An UNREADABLE algo book is treated as "unknown", never as "clean".

Also fixes a real footgun in the v1 script: it logged a HARDCODED target commit
("restarting onto 8bbe584") that had gone stale, so its output claimed a deploy
that was not what was on disk. Here the target is read from git at run time.

On anything unexpected: do nothing and leave the bot alone.

EXIT CODES are deliberately distinct, because "I safely did nothing" and "I may
have just broken a real-money leg" must never share a code:
    0  restarted and healthy   (or, with --check, the leg IS restartable)
    1  RESTARTED BUT NOT ACTIVE -- needs a human NOW
    2  usage error
    3  refused; the bot was NOT touched (in position, orders resting, or the
       algo book could not be read).  This is the SAFE outcome.

RACE (accepted, same as the v1 script it replaces): a leg could open a position
in the moment between the flat check and `systemctl restart`, and boot() would
then flatten it. The window is a couple of seconds and entry additionally
requires a fresh signal, so this is left unhandled rather than papered over with
a lock -- but check the leg is not mid-signal before running it.

Usage:
    leg_safe_restart.py <instance>            restart if flat
    leg_safe_restart.py <instance> --check    report only, never restart
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/root/snapback-btc")
sys.path.insert(0, str(REPO))

# instance -> (systemd unit, symbol)
LEGS = {
    "v1":             ("snapback-btc",            "BTC/USDT:USDT"),
    "donchian":       ("snapback-btc-donchian",   "BTC/USDT:USDT"),
    "sol_supertrend": ("snapback-sol-supertrend", "SOL/USDT:USDT"),
}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


def _unit_prop(unit: str, prop: str) -> str:
    return subprocess.run(["systemctl", "show", unit, "-p", prop, "--value"],
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    args = sys.argv[1:]
    check_only = "--check" in args
    args = [a for a in args if a != "--check"]
    if len(args) != 1 or args[0] not in LEGS:
        log(f"usage: leg_safe_restart.py <{'|'.join(LEGS)}> [--check]")
        return 2
    inst = args[0]
    unit, symbol = LEGS[inst]

    # Real deployed commit, not a hardcoded string that can rot.
    target = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"

    from exchange.env import load_env_for_instance
    load_env_for_instance(inst)
    from exchange.binance_client import BinanceClient

    c = BinanceClient.from_env()
    pos = c.fetch_position(symbol)
    if pos.side != "flat" or pos.qty:
        log(f"[{inst}] IN POSITION: {pos.side} qty={pos.qty} @ {pos.entry_price} "
            f"-- refusing (boot() would flatten it)")
        return 3

    plain = c.ex.fetch_open_orders(symbol)
    algo, algo_ok = c.fetch_algo_orders(symbol)
    if not algo_ok:
        log(f"[{inst}] algo book UNREADABLE -- refusing "
            f"(cannot prove the book is clean)")
        return 3
    if plain or algo:
        log(f"[{inst}] flat but {len(plain)} plain + {len(algo)} algo orders "
            f"resting -- refusing")
        return 3

    equity = c.fetch_equity_usdt()
    if check_only:
        log(f"[{inst}] FLAT and clean (equity=${equity:.2f}) -- WOULD restart "
            f"{unit} onto {target}. --check given, doing nothing.")
        return 0
    log(f"[{inst}] FLAT and clean (equity=${equity:.2f}) -- restarting {unit} "
        f"onto {target}")

    old_pid = _unit_prop(unit, "MainPID")
    subprocess.run(["systemctl", "restart", unit], check=True)
    time.sleep(15)
    new_pid = _unit_prop(unit, "MainPID")
    active = subprocess.run(["systemctl", "is-active", unit],
                            capture_output=True, text=True).stdout.strip()
    log(f"[{inst}] restarted: pid {old_pid} -> {new_pid}, is-active={active}")

    stamp = REPO / "data" / f"{inst}_safe_restart.done"
    stamp.write_text(json.dumps({
        "instance": inst, "unit": unit,
        "fired_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": target, "old_pid": old_pid, "new_pid": new_pid,
        "is_active": active, "equity_at_restart": equity,
    }, indent=2))
    log(f"[{inst}] wrote {stamp}")
    if active != "active":
        log(f"[{inst}] *** RESTARTED BUT is-active={active} -- NEEDS ATTENTION ***")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        log(f"ERROR (leaving bot untouched): {e}")
        raise SystemExit(1) from e
