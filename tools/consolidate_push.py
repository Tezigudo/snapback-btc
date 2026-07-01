"""Drain state.db outbox → POST /bot-event/batch to investing-consolidate.

Push-based, fire-and-forget (with retry). The bot's own state.db is the
source of truth; this module ships a downstream copy to consolidate so the
dashboard can render bot status.

Failure mode: if consolidate is unreachable (network, Fly.io down, auth
mismatch), events stay in the outbox and replay on the next drain. The
bot keeps trading regardless — this is best-effort, never blocking.

Configuration via .env:
  CONSOLIDATE_API_URL    e.g. https://investment-consolidation.fly.dev
  CONSOLIDATE_API_TOKEN  the same Bearer token the web app uses

If either is unset, drain() is a no-op (returns {"skipped": "not_configured"}).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from exchange import state

log = logging.getLogger(__name__)

# Source identifier in the consolidate /bot-event payload. Each running bot
# instance needs a DISTINCT source so the dashboard can show them as separate
# bot cards. Default is "snapback-btc" (the original v1 leg, unchanged).
# The Donchian leg sets CONSOLIDATE_SOURCE=snapback-btc-donchian in its env.
#
# Read at CALL time, NOT import time. bot.py imports this module at top level,
# but load_env_for_instance() overlays the per-instance .env INSIDE main() —
# AFTER this import. A module-level constant would freeze to the default, so
# every leg (donchian, cnh_short, …) would push under "snapback-btc" and
# collide on one dashboard card. Reading os.environ live fixes that.
def _source() -> str:
    return (os.environ.get("CONSOLIDATE_SOURCE") or "snapback-btc").strip()


DEFAULT_TIMEOUT_S = 3.0
DEFAULT_BATCH_LIMIT = 50

# After this many failed attempts with a 4xx response, a row is moved from the
# outbox to the dead_letter table so it can no longer block subsequent rows.
# Transient 5xx / network errors do NOT count toward this limit — those stay
# in the outbox and retry indefinitely. Configurable via env var for ops tuning.
DEAD_LETTER_AFTER_ATTEMPTS: int = int(
    os.environ.get("CONSOLIDATE_DEAD_LETTER_ATTEMPTS", "5")
)


def _config() -> tuple[str | None, str | None]:
    url = (os.environ.get("CONSOLIDATE_API_URL") or "").strip().rstrip("/")
    token = (os.environ.get("CONSOLIDATE_API_TOKEN") or "").strip()
    return (url or None), (token or None)


def is_configured() -> bool:
    url, token = _config()
    return url is not None and token is not None


def _build_event(row_id: int, kind: str, payload_str: str, source: str) -> dict[str, Any]:
    """Translate an outbox row into the /bot-event JSON shape."""
    body = json.loads(payload_str)
    return {
        "source": source,
        "external_id": f"{source}:{row_id}",
        "bot_ts_ms": int(body.get("bot_ts_ms", 0)),
        "kind": kind,
        "signal_id": body.get("signal_id"),
        "strategy": body.get("strategy"),
        "side": body.get("side"),
        "qty": body.get("qty"),
        "price_usd": body.get("price_usd"),
        "notional_usd": body.get("notional_usd"),
        "equity_usd": body.get("equity_usd"),
        "payload": body.get("payload") or {},
    }


def drain(limit: int = DEFAULT_BATCH_LIMIT, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Send up to `limit` queued events to consolidate. Returns status dict.

    On success: outbox rows for sent events are DELETED.
    On failure: rows stay; attempts incremented; last_error stored.
    """
    url, token = _config()
    if not url or not token:
        return {"skipped": "not_configured"}

    rows = state.outbox_pending(limit)
    if not rows:
        return {"sent": 0, "pending": 0}

    source = _source()
    events: list[dict[str, Any]] = []
    ids: list[int] = []
    for row_id, kind, payload_str, _attempts in rows:
        try:
            events.append(_build_event(row_id, kind, payload_str, source))
            ids.append(row_id)
        except Exception as e:
            # Malformed payload — should never happen, but tolerate. Mark this
            # one as failed and skip; the others still go.
            log.warning("outbox row %s payload parse failed: %s", row_id, e)
            state.outbox_mark_failed([row_id], f"parse: {e}")

    if not events:
        return {"sent": 0, "pending": state.outbox_size()}

    body = json.dumps({"events": events}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/bot-event/batch",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "snapback-btc/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp_status = int(resp.status)
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
        if 400 <= e.code < 500:
            # Client error: the batch payload itself was rejected (unknown kind,
            # auth mismatch, schema violation). Increment attempts for all rows in
            # the batch, then dead-letter any that have exceeded the retry budget.
            # Without dead-lettering, a single bad row (e.g. unknown kind) causes
            # every subsequent batch to fail — head-of-line blocking.
            log.warning("consolidate push 4xx: %s", err_msg)
            state.outbox_mark_failed(ids, err_msg)
            dead = state.outbox_dead_letter_over_limit(ids, err_msg, DEAD_LETTER_AFTER_ATTEMPTS)
            for row_id, kind, payload in dead:
                log.error(
                    "OUTBOX DEAD LETTER: row %s (kind=%s reached %d attempts) "
                    "moved to dead_letter table — payload: %.500s",
                    row_id, kind, DEAD_LETTER_AFTER_ATTEMPTS, payload,
                )
            return {
                "error": err_msg,
                "queued": len(ids) - len(dead),
                "dead_lettered": len(dead),
            }
        else:
            # 5xx — transient server error (Fly cold start, DB hiccup, etc.).
            # Keep all rows in the outbox and retry on the next drain.
            log.warning("consolidate push 5xx (transient): %s", err_msg)
            state.outbox_mark_failed(ids, err_msg)
            return {"error": err_msg, "queued": len(ids)}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # Transient network failure — keep in outbox, retry next drain.
        err_msg = f"net: {e}"
        log.info("consolidate push deferred (%s) — %d event(s) queued", e, len(ids))
        state.outbox_mark_failed(ids, err_msg)
        return {"error": str(e), "queued": len(ids)}
    except Exception as e:
        err_msg = f"unexpected: {e}"
        log.exception("consolidate push exception")
        state.outbox_mark_failed(ids, err_msg)
        return {"error": err_msg, "queued": len(ids)}

    # 2xx response. Server returns 200 when every event was accepted (inserted
    # OR deduped), and 207 (Multi-Status) when some failed but others made it
    # through. In the 207 case we must delete only the successful ids — the
    # failed ones stay queued for retry.
    error_external_ids: set[str] = set()
    for err in (resp_data.get("errors") or []):
        if isinstance(err, dict) and err.get("external_id"):
            error_external_ids.add(str(err["external_id"]))

    success_ids: list[int] = []
    failed_ids: list[int] = []
    for row_id in ids:
        ext_id = f"{source}:{row_id}"
        if ext_id in error_external_ids:
            failed_ids.append(row_id)
        else:
            success_ids.append(row_id)

    deleted = state.outbox_delete(success_ids) if success_ids else 0
    if failed_ids:
        state.outbox_mark_failed(
            failed_ids, f"server status {resp_status}: per-event failure")
        log.warning("consolidate push: %d/%d events failed server-side",
                    len(failed_ids), len(ids))

    return {
        "sent": len(events),
        "inserted": int(resp_data.get("inserted") or 0),
        "skipped": int(resp_data.get("skipped") or 0),
        "failed": len(failed_ids),
        "deleted_locally": deleted,
        "status": resp_status,
    }


if __name__ == "__main__":
    # Manual one-shot drain for ops/debugging.
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(drain(), indent=2))
