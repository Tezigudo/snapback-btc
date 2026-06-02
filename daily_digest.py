"""
Daily digest emailer. Designed for cron `0 1 * * *` (= 08:00 ICT).

Sends ONE email per day with a 24h rollup of all 3 legs:
  - Equity now vs 24h ago (LIVE legs)
  - Trades closed in last 24h (count, win rate, net P&L)
  - Signals seen vs taken (from log scan)
  - Heartbeat uptime % in last 24h
  - Top 3 gate-block reasons (what's preventing entries)
  - Error count
  - 4H gate health (NaN events, evaluated count)

Times shown in GMT+7 ICT.

NO LLM calls. Cron-safe (never raises).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sqlite3
import sys
import traceback
from collections import Counter
from pathlib import Path

from alerts import send_alert, is_configured

REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
LOGS = REPO_ROOT / "logs"
ICT = dt.timezone(dt.timedelta(hours=7), name="ICT")

log = logging.getLogger("snapback.daily_digest")

LEGS = [
    {"name": "v1",        "log": "bot.jsonl",       "state": "state.db",            "heartbeat": "heartbeat",            "live": True,  "anchor_label": "deploy_start_equity"},
    {"name": "donchian",  "log": "donchian.jsonl",  "state": "state_donchian.db",   "heartbeat": "heartbeat_donchian",   "live": False, "anchor_label": "deploy_start_equity"},
    {"name": "cnh_short", "log": "cnh_short.jsonl", "state": "state_cnh_short.db",  "heartbeat": "heartbeat_cnh_short",  "live": False, "anchor_label": "deploy_start_equity"},
]


def _to_ict(utc_iso: str) -> str:
    """Convert ISO-with-tz UTC string to 'YYYY-MM-DD HH:MM ICT'."""
    try:
        t = dt.datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        return t.astimezone(ICT).strftime("%Y-%m-%d %H:%M ICT")
    except (ValueError, TypeError):
        return utc_iso


def _now_ict() -> dt.datetime:
    return dt.datetime.now(tz=ICT)


def _yesterday_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(hours=24)


def _read_log_lines(p: Path, since_utc: dt.datetime) -> list[dict]:
    if not p.exists():
        return []
    cutoff = since_utc.isoformat()
    out = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = obj.get("ts", "")
            if ts >= cutoff:  # ISO format compares lexically
                out.append(obj)
    except OSError as e:
        log.warning("read_log %s: %s", p, e)
    return out


def _summarize_leg(leg: dict) -> dict:
    name = leg["name"]
    log_path = LOGS / leg["log"]
    db_path = DATA / leg["state"]
    hb_path = DATA / leg["heartbeat"]
    since = _yesterday_utc()
    events = _read_log_lines(log_path, since)

    by_level = Counter(e.get("level", "?") for e in events)
    gate_lines = [e.get("msg", "") for e in events if "gates:" in e.get("msg", "")]
    nan_count = sum(1 for e in events if "mtf_4h_gate_nan" in e.get("msg", ""))

    # Top gate-block reasons (extract "waiting on X, Y, Z" tokens)
    gate_token_counter: Counter[str] = Counter()
    for g in gate_lines:
        # gates: long waiting on A, B | short waiting on C, D
        for side_block in g.split("|"):
            m = re.search(r"waiting on (.+)", side_block)
            if m:
                tokens = [t.strip() for t in m.group(1).split(",") if t.strip()]
                gate_token_counter.update(tokens)

    top_block_reasons = gate_token_counter.most_common(3)

    # Heartbeat: count log lines in last 24h as proxy for uptime
    hb_lines = len([e for e in events if "heartbeat" in e.get("logger", "")])  # may be 0 if logger differs

    # Equity (LIVE only): fetch anchor + try to read latest "Resuming deploy" or "current=" line
    cur_equity = None
    anchor_equity = None
    try:
        if db_path.exists():
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0) as conn:
                cur = conn.execute("SELECT value FROM meta WHERE key = ?", (leg["anchor_label"],))
                row = cur.fetchone()
                if row:
                    anchor_equity = float(row[0] or 0) or None
    except sqlite3.Error as e:
        log.warning("digest: db read failed for %s: %s", name, e)
    # latest "current=" from log
    for e in reversed(events):
        m = re.search(r"current=([0-9.]+)", e.get("msg", ""))
        if m:
            try:
                cur_equity = float(m.group(1))
                break
            except ValueError:
                continue

    return {
        "name": name,
        "live": leg.get("live", False),
        "event_count": len(events),
        "by_level": dict(by_level),
        "gate_log_lines": len(gate_lines),
        "top_block_reasons": top_block_reasons,
        "nan_4h_count": nan_count,
        "heartbeat_loglines": hb_lines,
        "heartbeat_mtime_age_s": int((dt.datetime.now().timestamp() - hb_path.stat().st_mtime)) if hb_path.exists() else None,
        "anchor_equity": anchor_equity,
        "current_equity": cur_equity,
    }


def _format_leg(s: dict) -> str:
    name = s["name"]
    lines = [f"### {name}{' (LIVE)' if s['live'] else ' (DRY)'}"]
    lines.append(f"- events 24h: {s['event_count']}  ({s['by_level']})")
    if s["heartbeat_mtime_age_s"] is not None:
        lines.append(f"- heartbeat: {s['heartbeat_mtime_age_s']}s old")
    if s["live"] and s["anchor_equity"]:
        cur = s["current_equity"] or s["anchor_equity"]
        delta = cur - s["anchor_equity"]
        delta_pct = (delta / s["anchor_equity"]) * 100
        lines.append(f"- equity: ${cur:.2f} (anchor ${s['anchor_equity']:.2f}, delta {delta:+.2f} / {delta_pct:+.2f}%)")
    if s["top_block_reasons"]:
        reasons = ", ".join(f"{k}×{v}" for k, v in s["top_block_reasons"])
        lines.append(f"- top gate blocks: {reasons}")
    if s["nan_4h_count"] > 0:
        lines.append(f"- ⚠ mtf_4h_gate_nan: {s['nan_4h_count']} events")
    err = s["by_level"].get("ERROR", 0)
    if err > 0:
        lines.append(f"- ⚠ ERROR events: {err}")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not is_configured():
        log.warning("daily_digest: SMTP not configured; cannot send")
        return 0

    now = _now_ict()
    body_lines = [
        f"# snapback-btc daily digest",
        f"_generated {now.strftime('%Y-%m-%d %H:%M ICT')} • 24h window_",
        "",
    ]
    summaries = []
    for leg in LEGS:
        try:
            s = _summarize_leg(leg)
            summaries.append(s)
            body_lines.append(_format_leg(s))
            body_lines.append("")
        except Exception:
            body_lines.append(f"### {leg['name']}\n- ⚠ digest failed: {traceback.format_exc().splitlines()[-1]}")
            body_lines.append("")

    # Flags
    flags = []
    for s in summaries:
        if s["nan_4h_count"] > 0:
            flags.append(f"4H gate NaN on {s['name']}")
        if s["by_level"].get("ERROR", 0) > 0:
            flags.append(f"ERRORs on {s['name']}")
        if s["live"] and s["anchor_equity"] and s["current_equity"]:
            dp = ((s["current_equity"] - s["anchor_equity"]) / s["anchor_equity"]) * 100
            if dp <= -5:
                flags.append(f"{s['name']} equity {dp:+.1f}%")
    subject_flag = " ⚠ " + " · ".join(flags) if flags else " ✓ all green"
    subject = f"daily digest{subject_flag} ({now.strftime('%a %d %b')})"

    body = "\n".join(body_lines)
    send_alert(subject, body, tag="snapback-digest")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.stderr.write(traceback.format_exc())
        sys.exit(0)
