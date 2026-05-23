"""
Out-of-band alerting via SMTP email.

Used by monitor.py (cron) and bot.py (fill/halt events) to notify the operator
without paying for any service.

⚠ DigitalOcean droplets block ports 25/465/587 since 2025-03-06. Use a
transactional provider that exposes port 2525 (MailerSend, Mailgun,
SendGrid, Brevo). Gmail SMTP no longer works from DO droplets.

Config (in .env):
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    ALERT_EMAIL_FROM, ALERT_EMAIL_TO

If any required setting is missing, `send_alert` returns False and logs a
warning instead of raising — alerting failure must NEVER crash the bot.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

# Import order matters: exchange.env calls load_dotenv() at import time, which
# populates os.environ from .env before we read SMTP_* below.
from exchange import env as _env_module  # noqa: F401  (side-effect import)

log = logging.getLogger(__name__)

SMTP_TIMEOUT_S = 15


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def is_configured() -> bool:
    """True if all required SMTP env vars are populated."""
    return all(
        _env(k)
        for k in (
            "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
            "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO",
        )
    )


def _default_tag() -> str:
    """Subject prefix for alerts. Each bot instance overrides via ALERT_TAG env
    so the Donchian leg's emails thread separately from v1's in Gmail."""
    return (os.environ.get("ALERT_TAG") or "snapback-btc").strip()


def send_alert(subject: str, body: str, *, tag: str | None = None) -> bool:
    """
    Send an email alert. Returns True on success, False on any failure.

    Never raises — the bot must keep running even if alerting is broken.
    """
    if not is_configured():
        log.warning("alerts: SMTP not configured; skipping send (subject=%r)", subject)
        return False

    effective_tag = tag if tag is not None else _default_tag()
    msg = EmailMessage()
    msg["Subject"] = f"[{effective_tag}] {subject}"
    msg["From"] = _env("ALERT_EMAIL_FROM")
    msg["To"] = _env("ALERT_EMAIL_TO")
    msg.set_content(body)

    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT", "587"))
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")

    try:
        ctx = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_S, context=ctx) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_S) as s:
                s.ehlo()
                s.starttls(context=ctx)
                s.ehlo()
                s.login(user, password)
                s.send_message(msg)
        log.info("alerts: sent %r to %s", subject, msg["To"])
        return True
    except (smtplib.SMTPException, OSError) as e:
        log.warning("alerts: failed to send %r: %s", subject, e)
        return False


def _main() -> int:
    """CLI test: `python alerts.py "subject" "body"` — useful to verify .env wiring."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    subject = sys.argv[1] if len(sys.argv) > 1 else "test alert from snapback-btc"
    body = sys.argv[2] if len(sys.argv) > 2 else (
        "If you're reading this, SMTP is wired up correctly.\n"
        "Sent from alerts.py CLI."
    )
    ok = send_alert(subject, body, tag="snapback-btc-test")
    print("ok" if ok else "FAILED — check SMTP_* env vars and Gmail app password")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
