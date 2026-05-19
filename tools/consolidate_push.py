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

SOURCE = "snapback-btc"
DEFAULT_TIMEOUT_S = 3.0
DEFAULT_BATCH_LIMIT = 50


def _config() -> tuple[str | None, str | None]:
    url = (os.environ.get("CONSOLIDATE_API_URL") or "").strip().rstrip("/")
    token = (os.environ.get("CONSOLIDATE_API_TOKEN") or "").strip()
    return (url or None), (token or None)


def is_configured() -> bool:
    url, token = _config()
    return url is not None and token is not None


def _build_event(row_id: int, kind: str, payload_str: str) -> dict[str, Any]:
    """Translate an outbox row into the /bot-event JSON shape."""
    body = json.loads(payload_str)
    return {
        "source": SOURCE,
        "external_id": f"{SOURCE}:{row_id}",
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

    events: list[dict[str, Any]] = []
    ids: list[int] = []
    for row_id, kind, payload_str, _attempts in rows:
        try:
            events.append(_build_event(row_id, kind, payload_str))
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
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Distinguish 4xx (don't retry — payload bug) from 5xx (retry).
        # 401 (bad token) is a config problem, not transient — mark and skip
        # so they don't pile up forever. Operator must fix .env then.
        err_msg = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
        log.warning("consolidate push failed: %s", err_msg)
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

    # API ack'd. The response has {ok, inserted, skipped, errors}. We delete
    # ALL sent ids regardless of inserted-vs-skipped — dedup on the server
    # side just means a retry succeeded earlier; either way, we're done.
    deleted = state.outbox_delete(ids)
    return {
        "sent": len(events),
        "inserted": int(resp_data.get("inserted") or 0),
        "skipped": int(resp_data.get("skipped") or 0),
        "deleted_locally": deleted,
    }


if __name__ == "__main__":
    # Manual one-shot drain for ops/debugging.
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(drain(), indent=2))
