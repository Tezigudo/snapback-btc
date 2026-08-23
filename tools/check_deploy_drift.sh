#!/usr/bin/env bash
# Guard against main<->droplet drift silently stranding fixes.
#
# Why this exists: the donchian slope-formula fix (10adaef, 2026-06-03) was
# merged to main but never cherry-picked to the droplet deploy branch; the leg
# went real-money 2026-07-02 with a gate ~3.18x stricter than the backtest and
# traded ZERO times for two weeks. This script makes that class of drift loud.
#
# Usage: tools/check_deploy_drift.sh [deploy-ref]   (default: origin/droplet)
# Run it before every droplet deploy (see DEPLOY.md checklist).
#
# HARD-FAIL: the LIVE import chain — strategy/live_*.py, strategy/indicators.py
#            (bot_internals imports live_*; live_* import indicators), plus
#            risk.py, bot_internals.py, monitor.py. Must be byte-identical.
#            ALSO (added 2026-08-23, all byte-identical at the time so this
#            costs nothing today and catches the NEXT silent patch):
#              alerts.py, daily_digest.py   — how you find out something broke.
#                daily_digest was hand-scp'd onto the droplet once already
#                (c876aaf "was scp-only"), i.e. precisely the drift this script
#                exists to make loud, and it was not covered.
#              exchange/constraints.py      — exchange minimums ARE trading
#                semantics (a wrong min_notional silently skips signals).
#              exchange/trade_events.py     — what gets written to the trade
#                record; drift here corrupts the audit trail, not the trading.
# WARN-ONLY: backtest-side strategy modules (signals_*, regime_classifier),
#            config/*.yaml, bot.py, backtest.py — these legitimately diverge
#            (research evolves on main; deploy-only features live on droplet),
#            but every diff here should be a KNOWN one.
#            exchange/{binance_client,data,env,state}.py are warn-only for the
#            same reason: droplet carries the leg-isolation work (per-instance
#            env + state paths). WARN and not silence, because binance_client
#            is the order-placement layer — a diff there should be one you can
#            name out loud.
set -u
cd "$(git rev-parse --show-toplevel)"
REF="${1:-origin/droplet}"
git rev-parse --verify -q "$REF" >/dev/null || { echo "no such ref: $REF (git fetch origin droplet?)"; exit 2; }

common_files() {
    comm -12 \
        <(git ls-tree -r --name-only main -- "$@" | sort) \
        <(git ls-tree -r --name-only "$REF" -- "$@" | sort)
}

fail=0
echo "== deploy-drift check: main vs $REF =="
for f in $(common_files strategy/ | grep -E '^strategy/(live_[^/]+|indicators)\.py$') \
         risk.py bot_internals.py monitor.py alerts.py daily_digest.py \
         $(common_files exchange/ | grep -E '^exchange/(constraints|trade_events)\.py$'); do
    if ! git diff --quiet main "$REF" -- "$f"; then
        echo "DRIFT (HARD FAIL): $f"
        fail=1
    fi
done
for f in $(common_files strategy/ | grep -vE '^strategy/(live_[^/]+|indicators)\.py$') \
         $(common_files config/) bot.py backtest.py \
         $(common_files exchange/ | grep -vE '^exchange/(constraints|trade_events)\.py$'); do
    if ! git diff --quiet main "$REF" -- "$f"; then
        echo "drift (warn-only): $f"
    fi
done
if [ "$fail" -eq 0 ]; then
    echo "OK: no hard drift (live signal path, risk, monitor, alerting and trade-record files identical on both branches)"
else
    echo "FAIL: trading-semantics files differ between main and $REF — port before deploying."
fi
exit "$fail"
