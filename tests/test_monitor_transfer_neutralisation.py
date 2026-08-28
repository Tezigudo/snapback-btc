"""The monitor must not read a deposit as a drawdown.

Regression cover for 2026-08-28. `daily_anchor_equity` is a RAW snapshot of
equity taken at UTC midnight and frozen for the rest of the day, while
`principal_anchor` is refreshed hourly from the income ledger. Funding a leg
mid-day therefore raised the DENOMINATOR and left the numerator behind, and the
5-minute monitor reported a drawdown that never happened:

    leg              raw       P      reported   reality
    donchian      161.56  172.53      -6.36%     +6.4%   -> crossed into warn
    sol_supertrend 59.58   80.00     -25.52%     -0.5%   -> crossed into alert
    v1            137.68  183.93     -25.15%    -13.7%   -> alert, inflated

The fix mirrors `bot._daily_book_anchor` at read time: shift the raw anchor by
the net principal moved since it was set. `daily_digest.py` imports the same
reader, so the daily mail carried the same phantom numbers.
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import monitor  # noqa: E402

CFG = {
    "equity_drop_warn_pct": 5.0,
    "equity_drop_alert_pct": 10.0,
    "equity_alert_reminder_hours": 24,
}


@pytest.fixture
def make_db(tmp_path: Path):
    """Build a leg DB with the Part C principal ledger the live legs carry."""
    counter = iter(range(1000))

    def _make(
        *,
        raw_equity: float,
        principal: float,
        baseline: float | None,
        ledger: list[tuple[float, str]] | None,
        anchor_date: str | None = None,
        with_ledger_table: bool = True,
        fill_equity: float | None = None,
        principal_source: str | None = "income_backfill",
        principal_base: float | None = 0.0,
    ) -> Path:
        path = tmp_path / f"state_{next(counter)}.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE fills (id INTEGER PRIMARY KEY, equity_after REAL)")
        if anchor_date is None:
            anchor_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        meta = [
            ("principal_anchor", str(principal)),
            ("deploy_start_equity", str(principal)),
            ("daily_anchor_equity", str(raw_equity)),
            ("daily_anchor_date", anchor_date),
        ]
        if baseline is not None:
            meta.append(("daily_anchor_principal_sum", str(baseline)))
        if principal_source is not None:
            meta.append(("principal_source", principal_source))
        if principal_base is not None:
            meta.append(("principal_base", str(principal_base)))
        conn.executemany("INSERT INTO meta VALUES (?, ?)", meta)
        if with_ledger_table:
            conn.execute(
                "CREATE TABLE principal_ledger (tran_id INTEGER PRIMARY KEY, "
                "income_type TEXT, income_usd REAL, asset TEXT, ts_ms INTEGER)")
            for i, (amount, asset) in enumerate(ledger or []):
                conn.execute(
                    "INSERT INTO principal_ledger VALUES (?, 'TRANSFER', ?, ?, ?)",
                    (i + 1, amount, asset, 1_700_000_000_000 + i))
        if fill_equity is not None:
            conn.execute("INSERT INTO fills (equity_after) VALUES (?)", (fill_equity,))
        conn.commit()
        conn.close()
        return path
    return _make


def _drop_pct(db_path: Path) -> float:
    cur, anchor = monitor._equity_from_db(db_path)
    assert cur is not None and anchor is not None
    return (1 - cur / anchor) * 100


# --- the bug itself ----------------------------------------------------------

def test_intraday_deposit_is_not_a_drawdown(make_db):
    """Deposit +20 after the anchor was set: numerator moves with denominator."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT")])
    cur, anchor = monitor._equity_from_db(db)
    assert cur == pytest.approx(120.0)
    assert anchor == pytest.approx(120.0)
    assert monitor._equity_band(_drop_pct(db), CFG) == "ok"


def test_intraday_withdrawal_is_not_a_gain(make_db):
    """The mirror image — a withdrawal must not manufacture headroom."""
    db = make_db(raw_equity=100.0, principal=80.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (-20.0, "USDT")])
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(80.0)
    assert _drop_pct(db) == pytest.approx(0.0)


def test_real_trading_loss_still_reported_through_a_deposit(make_db):
    """Funding a losing leg must not paper over the loss it already carries.

    v1's real 2026-08-28 numbers: down 13.7% against deposited principal both
    before and after the +21 deposit. The deposit changes neither the loss nor
    the band — it only stopped the monitor from doubling it.
    """
    db = make_db(raw_equity=137.67903059, principal=183.93261979,
                 baseline=162.93261979,
                 ledger=[(162.93261979, "USDT"), (21.0, "USDT")])
    assert _drop_pct(db) == pytest.approx(13.73, abs=0.01)
    assert monitor._equity_band(_drop_pct(db), CFG) == "alert"


@pytest.mark.parametrize("leg, raw, principal, baseline, deposit, before, after", [
    ("donchian",       161.55832398, 172.53125, 150.53125, 22.0, "warn",  "ok"),
    ("sol_supertrend",  59.58291866,  80.0,      60.0,     20.0, "alert", "ok"),
])
def test_2026_08_28_false_alarms_are_gone(make_db, leg, raw, principal,
                                          baseline, deposit, before, after):
    """The two legs that actually crossed a band on the funding, by their real
    numbers. `before` is what the pre-fix reader produced, `after` the truth."""
    db = make_db(raw_equity=raw, principal=principal, baseline=baseline,
                 ledger=[(baseline, "USDT"), (deposit, "USDT")])
    assert monitor._equity_band((1 - raw / principal) * 100, CFG) == before
    assert monitor._equity_band(_drop_pct(db), CFG) == after


def test_only_usdt_rows_count_toward_the_delta(make_db):
    """P is USDT-denominated, so the shift must be too — every live leg carries
    a small BNB fee-transfer row that principal.py deliberately excludes."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT"), (0.016, "BNB")])
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(120.0)


# --- fail-safes: an unknown delta must restore the old reading, never guess ---

def test_missing_baseline_key_falls_back_to_raw(make_db):
    """A daily anchor set before the baseline key existed. The bot seeds it on
    its next tick; until then we must not invent a correction."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=[(120.0, "USDT")])
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)


def test_unparseable_baseline_falls_back_to_raw(make_db):
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=[(120.0, "USDT")])
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO meta VALUES ('daily_anchor_principal_sum', 'oops')")
    conn.commit()
    conn.close()
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)


def test_pre_part_c_db_without_ledger_table_still_reads(make_db):
    """Older DBs have no principal_ledger. The read must not raise — that would
    return (None, None) and silence the equity check entirely."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=None, with_ledger_table=False)
    cur, anchor = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)
    assert anchor == pytest.approx(120.0)


def test_fill_branch_is_not_shifted(make_db):
    """`fills.equity_after` is already post-transfer when the fill is newer than
    the deposit; shifting it would double-count. Reached only when today's daily
    anchor is absent (here: a stale anchor date)."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT")],
                 anchor_date="2020-01-01", fill_equity=118.0)
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(118.0)


# --- denominator provenance (Sourcery PR #25, comment 1) ---------------------

def test_anchor_is_derived_from_the_same_ledger_read(make_db):
    """P must be `principal_base + Sum(ledger)` — what the kill switch actually
    computes — not the cached `principal_anchor` copy. Here the cache is stale
    on purpose: numerator and denominator must still agree."""
    db = make_db(raw_equity=100.0, principal=100.0, baseline=100.0,
                 ledger=[(100.0, "USDT"), (20.0, "USDT")])
    cur, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(120.0), "stale principal_anchor was used"
    assert cur == pytest.approx(120.0)
    assert monitor._equity_band(_drop_pct(db), CFG) == "ok"


def test_manual_principal_base_is_included(make_db):
    """`manual_principal_usdt` mode seeds a non-zero base; P is base + ledger."""
    db = make_db(raw_equity=150.0, principal=1.0, baseline=50.0,
                 ledger=[(50.0, "USDT")], principal_base=100.0)
    _, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(150.0)


def test_zero_principal_base_is_not_treated_as_missing(make_db):
    """income_backfill mode stores base 0.0. A `> 0` filter would discard it and
    silently fall back to the cached anchor."""
    db = make_db(raw_equity=100.0, principal=999.0, baseline=100.0,
                 ledger=[(120.0, "USDT")], principal_base=0.0)
    _, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(120.0)


def test_uninitialised_ledger_disables_both_shift_and_derived_p(make_db):
    """Mirrors `principal.is_initialized()`. Mid-backfill the ledger holds the
    whole deposit history while the seed is unrecorded — shifting by that would
    move the numerator by the ENTIRE principal."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=0.0,
                 ledger=[(120.0, "USDT")], principal_source=None)
    cur, anchor = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0), "shifted while backfill was in flight"
    assert anchor == pytest.approx(120.0), "derived P used before initialisation"


# --- non-finite values must never go quiet (Sourcery PR #25, comment 2) ------

@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_baseline_falls_back_to_raw(make_db, bad):
    """A NaN/inf delta yields NaN equity, and `_equity_band` scores NaN as `ok`
    — a real drawdown would be silently suppressed."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=None,
                 ledger=[(120.0, "USDT")])
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO meta VALUES ('daily_anchor_principal_sum', ?)", (bad,))
    conn.commit()
    conn.close()
    cur, _ = monitor._equity_from_db(db)
    assert cur == pytest.approx(100.0)
    assert math.isfinite(cur)


def test_infinite_meta_value_is_rejected_by_pos_float(make_db):
    """`inf` survives a bare `> 0` test. Left in, it propagates to a NaN band."""
    db = make_db(raw_equity=100.0, principal=120.0, baseline=100.0,
                 ledger=None, with_ledger_table=False, principal_source=None)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value = 'inf' WHERE key = 'principal_anchor'")
    conn.commit()
    conn.close()
    _, anchor = monitor._equity_from_db(db)
    assert anchor == pytest.approx(120.0), "inf anchor leaked past _pos_float"


def test_equity_band_scores_nan_as_ok_which_is_why_we_guard():
    """Pins the reason the guards above exist."""
    assert monitor._equity_band(float("nan"), CFG) == "ok"
    assert monitor._equity_band(float("-inf"), CFG) == "ok"
