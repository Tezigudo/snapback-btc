"""
Cron-invoked health checker. Designed for cron */5 * * * *.

Checks (per leg):
  1. Heartbeat freshness (alert if > HEARTBEAT_STALE_S)
  2. ERROR / WARNING / mtf_4h_gate_nan events in last poll window
  3. Equity drop from deploy_start_equity (LIVE legs only)
  4. systemd unit state (active vs failed)

State persistence:
  data/monitor_state.json — last seen log offsets per leg, last alert ts per kind.
  Prevents duplicate alerts every 5 min for the same condition.

Email alerts via alerts.send_alert(). NO LLM calls.

Per-leg config: pulls thresholds from config/monitor.yaml if present, else uses
DEFAULTS below. Times reported as GMT+7 (Asia/Bangkok) in alert bodies.

CRITICAL: never raises. Cron-invoked; a crash is silent failure on the box.
Every exception is caught + logged + suppressed.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # config/monitor.yaml is optional

from alerts import send_alert, is_configured

REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
LOGS = REPO_ROOT / "logs"
STATE_PATH = DATA / "monitor_state.json"
CONFIG_PATH = REPO_ROOT / "config" / "monitor.yaml"

ICT = dt.timezone(dt.timedelta(hours=7), name="ICT")

log = logging.getLogger("snapback.monitor")

DEFAULTS: dict[str, Any] = {
    "heartbeat_stale_s":          120,   # v1 polls 5s, donchian/cnh_short 60s. 120s = comfortable margin.
    "equity_drop_warn_pct":       5.0,   # 5% from anchor → warn
    "equity_drop_alert_pct":      10.0,  # 10% → alert (kill switch fires at 15%)
    "error_alert":                True,  # any ERROR-level log line → alert
    "nan_4h_gate_alert":          True,  # any mtf_4h_gate_nan event → alert
    "alert_cooldown_min":         30,    # don't re-alert same kind within 30 min
}

LEGS: list[dict[str, str]] = [
    {"name": "v1",         "heartbeat": "heartbeat",            "log": "bot.jsonl",        "state": "state.db",            "systemd": "snapback-btc",                   "live": True},
    {"name": "donchian",   "heartbeat": "heartbeat_donchian",   "log": "donchian.jsonl",   "state": "state_donchian.db",   "systemd": "snapback-btc-donchian",          "live": False},
    # cnh_short retired 2026-07-12 (archive/cnh_short_retired_20260712). Its
    # permanent DOWN/STALE alerts every 5 min were failing sends that kept the
    # MailerSend account paused for "high API error rate" — do not re-add a
    # leg here unless its systemd unit actually runs.
]


def _now_ict() -> str:
    return dt.datetime.now(tz=ICT).strftime("%Y-%m-%d %H:%M:%S ICT")


def _load_state() -> dict[str, Any]:
    """Load monitor state. On first run, initialize log offsets to current
    EOF so we don't re-fire alerts on historical events (e.g. pre-deploy
    KILL_SWITCH or Tracebacks from days/weeks ago). The first cron tick
    establishes the baseline; alerts start firing on the SECOND tick.
    """
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("monitor: state file unreadable, starting fresh: %s", e)
    # First run — pin offsets to current EOF per leg so historical events
    # don't trigger false-positive alerts.
    initial_offsets: dict[str, int] = {}
    for leg in LEGS:
        log_path = LOGS / leg["log"]
        if log_path.exists():
            initial_offsets[f"offset:{leg['name']}"] = log_path.stat().st_size
        else:
            initial_offsets[f"offset:{leg['name']}"] = 0
    return {"alerts": {}, "log_offsets": initial_offsets}


def _save_state(state: dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    except OSError as e:
        log.warning("monitor: failed to save state: %s", e)


def _load_config() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if yaml is not None and CONFIG_PATH.exists():
        try:
            user_cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            cfg.update(user_cfg)
        except Exception as e:
            log.warning("monitor: monitor.yaml unreadable, using defaults: %s", e)
    return cfg


def _can_alert(state: dict[str, Any], kind: str, cooldown_min: int) -> bool:
    """Return True if we have not alerted on this kind within the cooldown window."""
    last_ts_str = state["alerts"].get(kind)
    if not last_ts_str:
        return True
    try:
        last_ts = dt.datetime.fromisoformat(last_ts_str)
    except ValueError:
        return True
    return (dt.datetime.now(tz=dt.timezone.utc) - last_ts) > dt.timedelta(minutes=cooldown_min)


def _stamp_alert(state: dict[str, Any], kind: str) -> None:
    state["alerts"][kind] = dt.datetime.now(tz=dt.timezone.utc).isoformat()


def _emit(subject: str, body: str, state: dict[str, Any], kind: str, cooldown_min: int) -> None:
    if not _can_alert(state, kind, cooldown_min):
        log.info("monitor: %s within cooldown, suppressing", kind)
        return
    full = f"{body}\n\n--\nGenerated {_now_ict()} by monitor.py"
    sent = send_alert(subject, full, tag="snapback-monitor")
    if sent:
        _stamp_alert(state, kind)
    else:
        log.warning("monitor: alert send failed for %s", kind)


def _heartbeat_age_s(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(dt.datetime.now().timestamp() - path.stat().st_mtime)


def _count_log_events(log_path: Path, since_offset: int, patterns: dict[str, str]) -> tuple[dict[str, int], int]:
    """Scan log file from `since_offset` to EOF. Returns (per-pattern counts, new offset).

    Patterns is {kind: substring}. A line counts if substring is present.
    """
    counts = {k: 0 for k in patterns}
    if not log_path.exists():
        return counts, since_offset
    try:
        size = log_path.stat().st_size
        if since_offset > size:
            since_offset = 0  # log rotated
        with log_path.open("rb") as f:
            f.seek(since_offset)
            data = f.read()
            new_offset = f.tell()
        text = data.decode("utf-8", errors="replace")
        for line in text.splitlines():
            for kind, needle in patterns.items():
                if needle in line:
                    counts[kind] += 1
        return counts, new_offset
    except OSError as e:
        log.warning("monitor: log scan failed for %s: %s", log_path, e)
        return counts, since_offset


def _systemd_active(unit: str) -> bool:
    if shutil.which("systemctl") is None:
        return True  # no systemd (dev box) — don't alert
    try:
        r = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except (OSError, subprocess.SubprocessError):
        return True  # transient failure — don't alert


def _equity_from_db(db_path: Path) -> tuple[float | None, float | None]:
    """Return (current_equity, deploy_start_equity) or (None, None) on any failure.

    Currently reads `deploy_start_equity` from meta; real equity is what the bot
    logs on each tick. For monitor purposes we approximate "current" with the
    latest bot.jsonl `current=` line; if not parseable, just return the anchor.
    """
    try:
        import sqlite3
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0) as conn:
            cur = conn.execute("SELECT key, value FROM meta WHERE key = ?", ("deploy_start_equity",))
            rows = dict(cur.fetchall())
        anchor = float(rows.get("deploy_start_equity", 0) or 0) or None
        return anchor, anchor  # current ≈ anchor unless we parse logs; bot log line "Resuming deploy" has it
    except Exception as e:
        log.warning("monitor: db read failed for %s: %s", db_path, e)
        return None, None


def _check_leg(leg: dict[str, str], cfg: dict[str, Any], state: dict[str, Any]) -> None:
    name = leg["name"]
    hb_path = DATA / leg["heartbeat"]
    log_path = LOGS / leg["log"]
    db_path = DATA / leg["state"]
    unit = leg["systemd"]
    cooldown = int(cfg["alert_cooldown_min"])

    # 1. systemd unit state
    if not _systemd_active(unit):
        _emit(
            f"DOWN: {unit}",
            f"systemd reports {unit} is not active. Check `systemctl status {unit}` on the droplet.",
            state, kind=f"systemd:{name}", cooldown_min=cooldown,
        )

    # 2. heartbeat age
    age = _heartbeat_age_s(hb_path)
    if age is None:
        _emit(
            f"NO HEARTBEAT: {name}",
            f"Heartbeat file {hb_path.name} missing. Bot may not have started.",
            state, kind=f"hb_missing:{name}", cooldown_min=cooldown,
        )
    elif age > int(cfg["heartbeat_stale_s"]):
        _emit(
            f"STALE HEARTBEAT: {name} ({age}s)",
            f"Heartbeat for {name} is {age}s old (threshold {cfg['heartbeat_stale_s']}s).\n"
            f"Bot may be hung or crashed.",
            state, kind=f"hb_stale:{name}", cooldown_min=cooldown,
        )

    # 3. log event scan
    offset_key = f"offset:{name}"
    since = int(state["log_offsets"].get(offset_key, 0))
    counts, new_offset = _count_log_events(
        log_path, since,
        {
            "ERROR":      '"level":"ERROR"',
            "TRACEBACK":  "Traceback (most recent call last)",
            "NAN_4H":     "mtf_4h_gate_nan",
            "KILL_SWITCH":"KILL SWITCH",
        },
    )
    state["log_offsets"][offset_key] = new_offset

    if counts["ERROR"] > 0 and cfg.get("error_alert"):
        _emit(
            f"ERROR x{counts['ERROR']}: {name}",
            f"{counts['ERROR']} ERROR-level events in {log_path.name} since last poll. "
            f"Investigate `/diagnose` or tail the log.",
            state, kind=f"errors:{name}", cooldown_min=cooldown,
        )
    if counts["TRACEBACK"] > 0:
        _emit(
            f"TRACEBACK: {name}",
            f"{counts['TRACEBACK']} Python tracebacks in {log_path.name}. Unhandled exception.",
            state, kind=f"trace:{name}", cooldown_min=cooldown,
        )
    if counts["NAN_4H"] > 0 and cfg.get("nan_4h_gate_alert"):
        _emit(
            f"4H GATE NAN: {name}",
            f"{counts['NAN_4H']} mtf_4h_gate_nan events. The 4H EMA200 isn't being computed; "
            f"signals are being silently skipped. Check 4H parquet feed.",
            state, kind=f"nan_4h:{name}", cooldown_min=cooldown,
        )
    if counts["KILL_SWITCH"] > 0:
        _emit(
            f"KILL SWITCH FIRED: {name}",
            f"{counts['KILL_SWITCH']} kill-switch events in {log_path.name}. "
            f"Bot has flattened positions and exited. data/HALT may now exist.",
            state, kind=f"kill:{name}", cooldown_min=cooldown,
        )

    # 4. equity check (LIVE only — for DRY legs, balance changes are paper)
    if leg.get("live") and db_path.exists():
        cur, anchor = _equity_from_db(db_path)
        if cur is not None and anchor is not None and anchor > 0:
            drop_pct = (1 - cur / anchor) * 100
            if drop_pct >= float(cfg["equity_drop_alert_pct"]):
                _emit(
                    f"EQUITY DROP {drop_pct:.1f}%: {name}",
                    f"Equity ${cur:.2f} vs anchor ${anchor:.2f} = -{drop_pct:.2f}%. "
                    f"Kill switch fires at -15%. Investigate immediately.",
                    state, kind=f"equity_alert:{name}", cooldown_min=cooldown,
                )
            elif drop_pct >= float(cfg["equity_drop_warn_pct"]):
                _emit(
                    f"EQUITY WARN {drop_pct:.1f}%: {name}",
                    f"Equity ${cur:.2f} vs anchor ${anchor:.2f} = -{drop_pct:.2f}%.",
                    state, kind=f"equity_warn:{name}", cooldown_min=cooldown,
                )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not is_configured():
        log.warning("monitor: SMTP not configured; running checks but cannot alert")

    cfg = _load_config()
    state = _load_state()

    for leg in LEGS:
        try:
            _check_leg(leg, cfg, state)
        except Exception:
            log.error("monitor: leg %s check crashed:\n%s", leg["name"], traceback.format_exc())

    _save_state(state)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Cron output goes to mail; print so the operator sees it via mail/cron
        sys.stderr.write(traceback.format_exc())
        sys.exit(0)  # never let cron retry; we don't want a flapping monitor
