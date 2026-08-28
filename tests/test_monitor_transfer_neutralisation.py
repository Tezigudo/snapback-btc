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
