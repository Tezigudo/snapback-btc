"""Equivalence guard for the canonical-PSR dedup (methodology debt #1).

Proves the merge of tools/portfolio_psr.py into tools/aggregate.py is
behavior-preserving, and that the unified `aggregate_psr` dispatcher delegates
literally to the two underlying public functions.

Reconciling "additive Lo key" (task 2) with "bit-for-bit equivalence" (task 3):

  - PORTFOLIO path: `aggregate_portfolio_psr` reads only specific scalar keys
    out of `compute_psr` and builds its own dict, so its full output is FROZEN
    across the Lo change -> full-dict `==` against the pre-merge golden holds
    bit-for-bit.

  - WINDOWS path: `aggregate_windows` / `build_canonical_block` /
    `legacy_stitched_psr` embed the FULL `compute_psr` dict (inside
    psr_walkforward / psr_per_window / legacy_psr_stitched), which after the Lo
    patch gains the additive `psr_lo_adjusted` (+ sr_lo_adjusted, lo_eta) keys.
    We therefore strip those purely-additive keys recursively before comparing
    to the pre-merge golden. Everything ELSE must be byte-identical (this is the
    honest form of "bit-for-bit on the frozen surface").

The golden was generated from the BASE-COMMIT (10adaef) modules — see
tools-history / tests/fixtures/psr_golden_premerge.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.aggregate as agg  # noqa: E402
import tools.portfolio_psr as pp  # noqa: E402

GOLDEN = json.loads(
    (ROOT / "tests" / "fixtures" / "psr_golden_premerge.json").read_text()
)

# Keys added purely by the additive Lo (2002) correction. They did not exist
# pre-merge; stripping them yields the frozen pre-Lo surface for comparison.
_ADDITIVE_LO_KEYS = {"psr_lo_adjusted", "sr_lo_adjusted", "lo_eta"}


def _strip_lo(obj):
    """Recursively drop the additive Lo keys from a result dict/list."""
    if isinstance(obj, dict):
        return {k: _strip_lo(v) for k, v in obj.items() if k not in _ADDITIVE_LO_KEYS}
    if isinstance(obj, list):
        return [_strip_lo(v) for v in obj]
    return obj


# ----- Fixture reconstruction (identical inputs to the golden generator) -----


def _rebuild_per_window():
    return [dict(d) for d in GOLDEN["_fixture_per_window"]]


def _iid_equity_curve(seed, n=500, mu=0.001, sigma=0.01, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=mu, scale=sigma, size=n)
    equity = 1_000_000.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(equity, index=idx, name="Equity")


def _rebuild_portfolio_eq():
    s = GOLDEN["_fixture_portfolio_seeds"]
    btc = _iid_equity_curve(seed=s["btc"], n=s["n"])
    sol = _iid_equity_curve(seed=s["sol"], n=s["n"])
    port_eq = agg.build_portfolio_equity_curve(btc, sol, s["w_btc"], s["w_sol"])
    return {"w1": port_eq}


# ----- (a) shim identity: no divergent copy -----


def test_shim_reexports_are_same_objects():
    assert pp.aggregate_portfolio_psr is agg.aggregate_portfolio_psr
    assert pp.build_portfolio_equity_curve is agg.build_portfolio_equity_curve
    assert pp.equity_to_period_returns is agg.equity_to_period_returns


# ----- (b) portfolio path: full-dict bit-for-bit == golden (frozen surface) ---


def test_portfolio_path_bitforbit_equals_golden():
    per_window_eq = _rebuild_portfolio_eq()
    out = agg.aggregate_portfolio_psr(per_window_eq, resample_period="1D")
    # Portfolio output never surfaced the Lo key -> exact equality.
    assert out == GOLDEN["portfolio"]
    # And the shim produces the identical dict (same object underneath).
    out_shim = pp.aggregate_portfolio_psr(per_window_eq, resample_period="1D")
    assert out_shim == GOLDEN["portfolio"]


# ----- (c) windows path: equals golden after stripping additive Lo keys ------


def test_windows_path_equals_golden_after_stripping_lo():
    per_window = _rebuild_per_window()
    windows = agg.aggregate_windows(per_window)
    canonical = agg.build_canonical_block(per_window)
    legacy = agg.legacy_stitched_psr(per_window)

    assert _strip_lo(windows) == GOLDEN["windows"]
    assert _strip_lo(canonical) == GOLDEN["canonical_block"]
    assert _strip_lo(legacy) == GOLDEN["legacy_stitched"]

    # The additive keys ARE present post-merge (proves the Lo patch landed,
    # so the strip is meaningful and not a tautology).
    assert "psr_lo_adjusted" in windows["psr_walkforward"]
    assert "psr_lo_adjusted" in windows["psr_per_window"][0]


# ----- (d) dispatcher delegates literally to the two public functions --------


def test_dispatcher_windows_branch_equals_aggregate_windows():
    per_window = _rebuild_per_window()
    assert agg.aggregate_psr(per_window) == agg.aggregate_windows(per_window)


def test_dispatcher_portfolio_branch_equals_aggregate_portfolio_psr():
    per_window_eq = _rebuild_portfolio_eq()
    out = agg.aggregate_psr(
        per_window_eq=per_window_eq,
        portfolio_weights={"btc": 0.5, "sol": 0.5},
    )
    assert out == agg.aggregate_portfolio_psr(per_window_eq, resample_period="1D")


def test_dispatcher_validates_branch_selectors():
    per_window = _rebuild_per_window()
    # Windows branch with no per_window -> error.
    with pytest.raises(ValueError):
        agg.aggregate_psr(None)
    # Portfolio branch with no per_window_eq -> error.
    with pytest.raises(ValueError):
        agg.aggregate_psr(portfolio_weights={"btc": 0.5, "sol": 0.5})
    # Portfolio weights must be 2 and sum to ~1.0.
    with pytest.raises(ValueError):
        agg.aggregate_psr(
            per_window_eq=_rebuild_portfolio_eq(),
            portfolio_weights={"btc": 1.0},
        )
    with pytest.raises(ValueError):
        agg.aggregate_psr(
            per_window_eq=_rebuild_portfolio_eq(),
            portfolio_weights={"btc": 0.7, "sol": 0.7},
        )


# ----- (e) psr_vs_hurdle byte-identical pre/post Lo (locked-ref preservation) -


def test_psr_vs_hurdle_unchanged_vs_golden():
    """The uncorrected PSR (locked v1 0.978 reference family) must be byte-
    identical to the pre-merge golden in BOTH paths after the Lo patch."""
    per_window = _rebuild_per_window()
    windows = agg.aggregate_windows(per_window)
    assert (
        windows["psr_walkforward"]["psr_vs_hurdle"]
        == GOLDEN["windows"]["psr_walkforward"]["psr_vs_hurdle"]
    )
    for got, ref in zip(
        windows["psr_per_window"], GOLDEN["windows"]["psr_per_window"]
    ):
        assert got.get("psr_vs_hurdle") == ref.get("psr_vs_hurdle")

    per_window_eq = _rebuild_portfolio_eq()
    out = agg.aggregate_portfolio_psr(per_window_eq, resample_period="1D")
    assert out["psr_equity_curve"] == GOLDEN["portfolio"]["psr_equity_curve"]
