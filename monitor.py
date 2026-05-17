"""
Cron-invoked health checker (every 5 min). Implemented in P5.

Checks:
  - heartbeat file mtime (alert if > 90s stale)
  - equity drawdown vs MAX_DAILY_LOSS_PCT
  - new ERROR/Traceback lines in today's log
  - position reconciliation vs state.db

Sends email via alerts.send_alert(). NO LLM calls — per-minute cost is $0.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Monitor is implemented in P5.")


if __name__ == "__main__":
    main()
