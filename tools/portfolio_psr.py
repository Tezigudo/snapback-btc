"""MOVED to tools.aggregate — thin re-export shim (methodology debt #1 dedup).

The true weighted-equity-curve portfolio PSR (methodology debt #2) now lives in
``tools/aggregate.py`` alongside the 5-OOS / walk-forward aggregation family, so
there is a single canonical PSR home. This module is kept ONLY so the existing
importers (tools/_postfrac_mf_4h_btc_sol_portfolio.py,
tools/_postfrac_wf_mf_4h_btc_sol_portfolio.py, tools/tests/test_portfolio_psr.py)
keep working unchanged. New code should import from ``tools.aggregate`` directly.

The re-exported objects ARE the same objects defined in tools.aggregate (no
divergent copy) — see tests/test_unified_psr_equivalence.py which asserts the
identity.
"""

from __future__ import annotations

from tools.aggregate import (  # noqa: F401
    build_portfolio_equity_curve,
    equity_to_period_returns,
    aggregate_portfolio_psr,
)

# _normalized_equity is module-private; re-export only if a test/runner needs it
# (currently none do).

__all__ = [
    "build_portfolio_equity_curve",
    "equity_to_period_returns",
    "aggregate_portfolio_psr",
]
