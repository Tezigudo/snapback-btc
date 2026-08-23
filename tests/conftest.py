"""Suite-wide isolation for the sqlite state layer.

WHY THIS EXISTS
---------------
`exchange/state.py` resolves `DB_PATH` at import time to `data/state.db` (or
`$SNAPBACK_STATE_DB`). Nothing in the suite ever set that, so any test that
reached a `state.*` call — or `bot.py`'s direct `sqlite3.connect(state.DB_PATH)`
— operated on a REAL database file inside the repo, and created `data/state.db`
as a side effect of merely running the tests.

That is the hazard behind the droplet branch's red suite (fixed in `3c30737`):
six tests drove `bot._place_live_entry`, which persists `active_bracket` via
`state.set_meta`, against whatever `data/state.db` happened to exist on that
machine — and failed with

    sqlite3.OperationalError: no such table: meta

The tests that passed did so by patching `state.set_meta` one call site at a
time. That is a defence every future test author has to remember, and forgetting
it produces a failure whose message points at sqlite rather than at the missing
patch. Worse, on a developer machine where a populated `data/state.db` already
exists, the same test would silently PASS while writing to a real database.

This fixture removes the hazard rather than defending against it: every test
gets its own freshly-initialised database in its own tmp dir.

`set_db_path()` rebinds the module-level `DB_PATH` global (state.py:48-56), so
both `state._conn()` and `bot.py`'s direct read of `state.DB_PATH` follow it —
the direct read is why an accessor-only fix would not have been enough.

NOTE: this is isolation, not a behaviour change. Tests that patch `state.*` keep
working untouched; they simply no longer NEED to.
"""

from __future__ import annotations

import pytest

from exchange import state


@pytest.fixture(autouse=True)
def isolated_state_db(tmp_path, monkeypatch):
    """Point the state layer at a per-test sqlite file, then restore it.

    The env var is set too so that any code re-reading `SNAPBACK_STATE_DB`
    (or a subprocess) agrees with the rebound global.
    """
    original = state.DB_PATH
    db = tmp_path / "state.db"
    monkeypatch.setenv("SNAPBACK_STATE_DB", str(db))
    state.set_db_path(db)
    state.init_db()
    try:
        yield db
    finally:
        state.set_db_path(original)
