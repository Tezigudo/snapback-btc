"""
Cron-invoked health checker. Designed for cron */5 * * * *.

Checks (per leg):
  1. Heartbeat freshness (alert if > HEARTBEAT_STALE_S)
  2. ERROR / WARNING / mtf_4h_gate_nan events in last poll window
  3. Equity drop from deploy_start_equity (LIVE legs only)
  4. systemd unit state (active vs failed)

State persistence:
  data/monitor_state.json — last seen log offsets per leg, last alert ts per
  kind, and the last equity band (ok/warn/alert) per leg.
  Prevents duplicate alerts every 5 min for the same condition. Event-style
  checks are rate-limited by cooldown; the equity check fires only when its
  band changes (see _check_equity).

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
import math
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
    # Equity alerts fire on BAND TRANSITIONS, not on a cooldown (see
    # _check_equity). The severe band still repeats this often so a sustained
    # drawdown can't go silent.
    "equity_alert_reminder_hours": 24,
    # Minutes after UTC midnight during which a stale daily anchor means "the
    # bot has not re-anchored yet", not "here is a reading worth alerting on".
    # See the ROLLOVER GRACE section of _equity_from_db. Must comfortably clear
    # the slowest leg's poll interval (60s for donchian/sol_supertrend).
    "equity_anchor_grace_min":    10,
}

LEGS: list[dict[str, str]] = [
    {"name": "v1",         "heartbeat": "heartbeat",            "log": "bot.jsonl",        "state": "state.db",            "systemd": "snapback-btc",                   "live": True},
    {"name": "donchian",   "heartbeat": "heartbeat_donchian",   "log": "donchian.jsonl",   "state": "state_donchian.db",   "systemd": "snapback-btc-donchian",          "live": True},   # real-money since 2026-07-02
    {"name": "sol_supertrend", "heartbeat": "heartbeat_sol_supertrend", "log": "sol_supertrend.jsonl", "state": "state_sol_supertrend.db", "systemd": "snapback-sol-supertrend", "live": True},  # real-money since 2026-07-27 (God's go)
    # cnh_short retired 2026-07-12 (archive/cnh_short_retired_20260712). Its
    # permanent DOWN/STALE alerts every 5 min were failing sends that kept the
    # MailerSend account paused for "high API error rate" — do not re-add a
    # leg here unless its systemd unit actually runs.
    #
    # sol_supertrend flipped to real money 2026-07-27 (DRY_RUN=0 in
    # .env.sol_supertrend on the droplet); `live: True` keeps the equity-drop
    # and error checks armed. Keep both flags in sync if it ever goes back to
    # dry-run.
]


def _now_ict() -> str:
    return dt.datetime.now(tz=ICT).strftime("%Y-%m-%d %H:%M:%S ICT")


def _mins_since_utc_midnight(now: dt.datetime) -> float:
    """Minutes elapsed since the most recent UTC midnight.

    How long the bot has HAD to re-anchor `daily_anchor_date` for the new UTC
    day. Deliberately measured from the rollover rather than from the anchor's
    own date: a leg that has held a position for three days has an anchor three
    days old, but right after midnight it is no more overdue than any other leg.
    """
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (now - midnight).total_seconds() / 60.0


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


def _emit(subject: str, body: str, state: dict[str, Any], kind: str,
          cooldown_min: int, force: bool = False) -> bool:
    """Send an alert unless it is within its cooldown. Returns True if sent.

    `force` bypasses the cooldown — used by the equity band checks, which are
    already rate-limited by only firing on state transitions.
    """
    if not force and not _can_alert(state, kind, cooldown_min):
        log.info("monitor: %s within cooldown, suppressing", kind)
        return False
    full = f"{body}\n\n--\nGenerated {_now_ict()} by monitor.py"
    sent = send_alert(subject, full, tag="snapback-monitor")
    if sent:
        _stamp_alert(state, kind)
    else:
        log.warning("monitor: alert send failed for %s", kind)
    return bool(sent)


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


def _equity_from_db(
    db_path: Path,
    grace_min: float = DEFAULTS["equity_anchor_grace_min"],
    now: dt.datetime | None = None,
) -> tuple[float | None, float | None]:
    """Return (current_equity, anchor_equity) or (None, None) on any failure.

    Anchor: `principal_anchor` (net deposited principal, what the bot's kill
    switch actually uses) when the leg has one, else `deploy_start_equity`.
    The old code returned deploy_start_equity for BOTH values, so drop_pct
    was always 0 and the equity check could never fire.

    Current: freshest of the bot's per-UTC-day breaker anchor
    (`daily_anchor_equity`, refreshed on the first tick of each UTC day) and
    the last fill's `equity_after`. Granularity is day/trade — this catches
    multi-day bleeds and post-stop-out drops; intraday protection lives
    in-process (daily-loss breaker + kill switch).

    TRANSFER NEUTRALISATION (2026-08-28). `daily_anchor_equity` is a RAW
    snapshot of equity at UTC midnight and never moves again that day, but
    `principal_anchor` is refreshed hourly from the income ledger. An intraday
    deposit therefore raises the DENOMINATOR while the numerator stays at its
    pre-deposit value, and the leg reads as a drawdown it never took. This
    fired for real: all three legs were funded at 02:28 UTC on 2026-08-28
    (+21 / +22 / +20 USDT) and within 32 minutes donchian crossed into `warn`
    at -6.36% (it was actually UP 6.4%) and sol_supertrend into `alert` at
    -25.5% (actually -0.5%). A withdrawal produces the mirror-image failure —
    a false all-clear — which is the more dangerous direction.

    The bot already solves this for its own daily-loss breaker in
    `bot._daily_book_anchor`: shift the raw anchor by the net principal moved
    since that anchor was set. We mirror the same computation at READ time —

        current = raw_anchor + (ledger_sum_now - daily_anchor_principal_sum)

    — and deliberately write nothing back. The stored anchor must stay RAW,
    because `_daily_book_anchor` applies this delta itself on every tick;
    pre-shifting the stored value would apply it twice and corrupt the live
    daily-loss breaker.

    Both sides come from ONE read of ONE table on ONE connection. The
    denominator is `principal_base + Σledger` — P exactly as
    `principal.get_principal()` derives it for the kill switch — rather than
    the cached `principal_anchor` copy, so the numerator and denominator can
    never be observed a reconcile apart from each other.

    Neither the shift nor the derived P is applied until `principal_source` is
    set, mirroring `principal.is_initialized()` and the bot's matching guard in
    `_daily_loss_blocks_entry`. During a first-run income backfill the ledger
    sum climbs from 0 to the entire deposit history while the seed is still
    unrecorded; treating that as an intraday transfer would shift the numerator
    by the whole principal.

    Fail-safe, matching the bot: a missing, unparseable or non-finite baseline,
    or a pre-Part-C DB with no `principal_ledger` table, yields delta 0 —
    exactly the previous behaviour. The shift applies only to the daily-anchor
    readings. `fills.equity_after` is already post-transfer whenever the fill
    is newer than the deposit; a fill OLDER than the deposit stays uncorrected.

    ROLLOVER GRACE (2026-09-03). The paragraph above used to end "...which is
    acceptable because that branch is only reached when today's daily anchor is
    missing, and it self-heals at the next UTC rollover." Both halves were
    wrong, and they cost a daily false alarm for six days.

    The fill branch is reached AT every rollover, not only when the anchor is
    absent: at 00:00 UTC `today` becomes the new date while `daily_anchor_date`
    still holds the old one, because only the bot writes that key
    (`bot._daily_anchor_equity`, reached via `_maybe_enter` on a 60s poll for
    donchian/sol_supertrend). The monitor's `*/5` cron fires inside that gap,
    reads a stale pre-deposit fill, and alerts. The bot re-anchors seconds later
    and the next tick "recovers" — so the rollover is the CAUSE, not the cure.

    Observed 2026-08-29..09-03: donchian warned at -7.00% ($160.46 vs $172.53)
    at 00:00 and recovered at +6.39% ($183.56) at 00:05, every day, with
    byte-identical numbers; sol_supertrend did the same at -25.2% -> -0.52% on
    the days its own poll happened to lose the race. Each leg's error equalled
    ITS OWN 2026-08-28 deposit — donchian 23.10 (+22), sol 19.71 (+20), v1
    21.00 (+21) — which is the signature of this branch specifically, the only
    one that skips the transfer shift. `daily_digest.py` imports this reader and
    ran an hour later on the same data: it saw the correct figures throughout.

    So for `grace_min` after midnight, a stale anchor yields (None, anchor) and
    `_check_equity` returns without alerting. Past that window the fill is used
    exactly as before — a leg holding a position for days never re-anchors
    (`_maybe_enter` short-circuits on a non-flat position), and a stale fill is
    then the only reading there is, which is precisely when a real bleed matters
    most. The gate is on how long the anchor has been overdue, not on the
    branch: deleting the fallback would blind the check to the case it exists
    for. One tick of silence costs nothing — this is a day/trade-granularity
    check, and intraday protection lives in-process.
    """
    try:
        import sqlite3
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0) as conn:
            cur = conn.execute(
                "SELECT key, value FROM meta WHERE key IN "
                "('principal_anchor', 'deploy_start_equity', "
                "'daily_anchor_equity', 'daily_anchor_date', "
                "'daily_anchor_principal_sum', 'principal_base', "
                "'principal_source')")
            rows = dict(cur.fetchall())
            fill = conn.execute(
                "SELECT equity_after FROM fills WHERE equity_after IS NOT NULL "
                "ORDER BY id DESC LIMIT 1").fetchone()
            try:
                # Must match bot's state.principal_ledger_sum(PRINCIPAL_ASSET):
                # USDT rows ONLY. Non-USDT transfers are excluded from P as
                # well, so excluding them here keeps numerator and denominator
                # measured in the same units.
                ledger_sum: float | None = conn.execute(
                    "SELECT COALESCE(SUM(income_usd), 0.0) FROM principal_ledger "
                    "WHERE asset = 'USDT'").fetchone()[0]
            except sqlite3.Error:
                ledger_sum = None   # pre-Part-C DB → no neutralisation possible
        # Mirror `principal.is_initialized()` / the bot's own guard in
        # `_daily_loss_blocks_entry`: until the ledger has been marked
        # initialised, a full-history backfill is still in flight and the sum
        # climbs from 0 to the whole deposit history. Treating that as an
        # intraday transfer would shift the numerator by the ENTIRE principal.
        # No neutralisation and no derived P until the seed is recorded.
        ledger_ready = rows.get("principal_source") is not None

        def _pos_float(key: str) -> float | None:
            """Parse one meta key defensively — a malformed value in one key
            must not invalidate the whole equity read (Sourcery, PR #6).

            `inf` must be rejected as well as unparseable text: it survives the
            `> 0` test, and an infinite reading propagates into `drop_pct` as
            NaN, which `_equity_band` classifies as `ok` — silently suppressing
            an alert rather than failing loudly (Sourcery, PR #25).
            """
            try:
                v = float(rows.get(key))
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) and v > 0 else None

        def _adjusted(raw: float | None) -> float | None:
            """Raw daily anchor shifted by principal moved since it was set.

            Returns `raw` unchanged whenever the delta cannot be established,
            so this can only ever restore the pre-fix reading, never invent a
            correction from an unknown baseline.
            """
            if raw is None or ledger_sum is None or not ledger_ready:
                return raw
            try:
                baseline = float(rows.get("daily_anchor_principal_sum"))
            except (TypeError, ValueError):
                # Anchor predates the baseline key; the bot seeds it lazily on
                # its next tick. No baseline → no defensible delta → don't guess.
                return raw
            if not math.isfinite(baseline) or not math.isfinite(ledger_sum):
                # A non-finite delta yields NaN equity, and _equity_band scores
                # NaN as `ok`. Refuse to compute rather than go quiet.
                return raw
            return raw + (ledger_sum - baseline)

        # P as the KILL SWITCH computes it. `bot._check_kill_switch` calls
        # `principal.get_principal()`, which derives `principal_base + Σledger`
        # LIVE; the cached `principal_anchor` meta key is a convenience copy for
        # dashboards. Deriving it here from the SAME connection and the SAME
        # ledger read as the numerator means the two can never be measured a
        # reconcile apart from each other (Sourcery, PR #25), and makes the
        # monitor agree with the ceiling it is reporting against. `principal_base`
        # is legitimately 0.0 in income_backfill mode, so it cannot go through
        # `_pos_float`, which rejects non-positive values.
        derived_principal: float | None = None
        if ledger_ready and ledger_sum is not None and math.isfinite(ledger_sum):
            try:
                base = float(rows.get("principal_base", 0.0) or 0.0)
            except (TypeError, ValueError):
                base = None
            if base is not None and math.isfinite(base):
                candidate = base + ledger_sum
                if candidate > 0:
                    derived_principal = candidate

        anchor = (derived_principal or _pos_float("principal_anchor")
                  or _pos_float("deploy_start_equity"))
        if now is None:
            now = dt.datetime.now(dt.timezone.utc)
        today = now.strftime("%Y-%m-%d")
        daily = _pos_float("daily_anchor_equity")
        if rows.get("daily_anchor_date") == today and daily is not None:
            current: float | None = _adjusted(daily)
        elif _mins_since_utc_midnight(now) < grace_min:
            # ROLLOVER GRACE — see docstring. The UTC day just turned over and
            # the bot has not re-anchored yet. Every fallback below is stale by
            # at least a day here, and the fill is not transfer-adjusted, so
            # report nothing rather than something wrong.
            current = None
        elif fill is not None and fill[0] is not None:
            # equity_after of 0.0 is a VALID (and alarming) reading — do not
            # truthiness-filter it away. Not transfer-adjusted: see docstring.
            try:
                current = float(fill[0])
            except (TypeError, ValueError):
                current = _adjusted(daily)
        else:
            current = _adjusted(daily)
        return current, anchor
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
        _check_equity(name, db_path, cfg, state)


def _equity_band(drop_pct: float, cfg: dict[str, Any]) -> str:
    """Classify a drawdown-from-anchor into ok / warn / alert."""
    if drop_pct >= float(cfg["equity_drop_alert_pct"]):
        return "alert"
    if drop_pct >= float(cfg["equity_drop_warn_pct"]):
        return "warn"
    return "ok"


def _check_equity(name: str, db_path: Path, cfg: dict[str, Any],
                  state: dict[str, Any]) -> None:
    """Alert on equity-band TRANSITIONS rather than every tick the band holds.

    Equity-vs-anchor is a state, not an event. The previous version re-emitted
    on the 30-min cooldown for as long as the condition persisted — v1 sat 6.3%
    below its anchor for 30 hours and sent 52 identical warns, which is how a
    real alert gets buried. Behaviour now:

      - crossing into warn/alert emits once,
      - holding that band is silent, except `alert` repeats every
        `equity_alert_reminder_hours` so a sustained drawdown can't go quiet,
      - returning to ok emits a recovery notice,
      - a non-ok band seen with no prior history still emits, so a problem that
        already exists at deploy time is never silently swallowed.

    The band is committed only after a successful send, so an SMTP failure is
    retried on the next tick instead of being lost.
    """
    cur, anchor = _equity_from_db(
        db_path,
        float(cfg.get("equity_anchor_grace_min",
                      DEFAULTS["equity_anchor_grace_min"])),
    )
    if cur is None or anchor is None or anchor <= 0:
        return
    drop_pct = (1 - cur / anchor) * 100
    band = _equity_band(drop_pct, cfg)
    bands = state.setdefault("equity_bands", {})
    prev = bands.get(name)
    # Signed DELTA, not the drop. A hardcoded "-" printed the recovery notice as
    # "= --6.39%" whenever equity was above anchor — a double minus reporting a
    # 6.4% GAIN, on the very mail meant to reassure. Matches daily_digest's
    # "{delta_pct:+.2f}%".
    position = f"Equity ${cur:.2f} vs anchor ${anchor:.2f} = {-drop_pct:+.2f}%."

    if band == prev:
        if band == "alert":
            _emit(
                f"EQUITY DROP {drop_pct:.1f}%: {name} (ongoing)",
                f"{position} Still past the {cfg['equity_drop_alert_pct']}% alert line. "
                f"Bot kill switch flattens at -35.5% from principal.",
                state, kind=f"equity_alert:{name}",
                cooldown_min=int(float(cfg["equity_alert_reminder_hours"]) * 60),
            )
        return

    if band == "ok":
        if prev is None:
            bands[name] = band  # healthy and nothing to compare against
            return
        sent = _emit(
            f"EQUITY RECOVERED: {name}",
            f"{position} Back inside the {cfg['equity_drop_warn_pct']}% warn threshold.",
            state, kind=f"equity_ok:{name}", cooldown_min=0, force=True,
        )
    elif band == "alert":
        sent = _emit(
            f"EQUITY DROP {drop_pct:.1f}%: {name}",
            f"{position} Bot kill switch flattens at -35.5% from principal. "
            f"Investigate immediately.",
            state, kind=f"equity_alert:{name}", cooldown_min=0, force=True,
        )
    else:
        origin = ("improving — was past the alert line" if prev == "alert"
                  else "newly below the warn line")
        sent = _emit(
            f"EQUITY WARN {drop_pct:.1f}%: {name}",
            f"{position} ({origin}) No further warns unless this changes band.",
            state, kind=f"equity_warn:{name}", cooldown_min=0, force=True,
        )
    if sent:
        bands[name] = band


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
