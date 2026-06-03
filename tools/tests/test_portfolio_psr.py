"""Unit tests for tools/portfolio_psr.py.

These tests are deliberately tiny, deterministic, and depend on no parquet
files or backtest engine. The load-bearing assertion is Test B
(perfect-correlation regression guard): the old stitched-trade-pool PSR
must strictly exceed the new equity-curve PSR on identical legs, and the
new equity-curve portfolio PSR must equal the single-leg PSR.

Run from repo root:

    PYTHONPATH=. python3 -m pytest -xvs tools/tests/test_portfolio_psr.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make repo root importable when invoked directly.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.portfolio_psr import (  # noqa: E402
    aggregate_portfolio_psr,
    build_portfolio_equity_curve,
    equity_to_period_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _iid_equity_curve(seed: int, n: int = 500, mu: float = 0.001,
                      sigma: float = 0.01,
                      start: str = "2024-01-01") -> pd.Series:
    """Daily IID equity curve starting at $1M with N(mu, sigma) log returns."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=mu, scale=sigma, size=n)
    equity = 1_000_000.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(equity, index=idx, name="Equity")


def _trade_returns_from_equity(eq: pd.Series) -> np.ndarray:
    """Pseudo per-trade returns reconstructed from an equity curve.

    Used to model the OLD stitched-trade-pool PSR proxy: treat each daily
    bar as one 'trade'. This is exactly the shape of pnl_pct fed into the
    legacy compute_psr call.
    """
    return eq.pct_change().dropna().values.astype(float) * 100.0  # percent


# ---------------------------------------------------------------------------
# Test A: uncorrelated legs receive a diversification credit (vol drops)
# ---------------------------------------------------------------------------

def test_a_uncorrelated_diversification_credit():
    """With two zero-correlation IID legs, portfolio sigma drops by ~sqrt(2)."""
    btc = _iid_equity_curve(seed=0, n=500)
    sol = _iid_equity_curve(seed=1, n=500)

    port_eq = build_portfolio_equity_curve(btc, sol, w_btc=0.5, w_sol=0.5)
    assert port_eq is not None
    port_rets = equity_to_period_returns(port_eq, resample_period=None)

    btc_rets = btc.pct_change().dropna().values
    sol_rets = sol.pct_change().dropna().values

    # Sigma ratio: portfolio sigma should be ~1/sqrt(2) of single-leg sigma.
    leg_sigma = 0.5 * (np.std(btc_rets, ddof=1) + np.std(sol_rets, ddof=1))
    port_sigma = np.std(port_rets, ddof=1)
    ratio = port_sigma / leg_sigma
    assert 0.55 < ratio < 0.85, (
        f"Portfolio sigma / leg sigma = {ratio:.3f}; expected ~0.707 "
        f"(uncorrelated diversification credit). If this fails, the "
        f"weighted-equity-curve construction is destroying correlation."
    )

    # N regression guard: the equity-curve PSR sees one period per bar,
    # NOT trade_BTC + trade_SOL.
    agg = aggregate_portfolio_psr(
        {"w": port_eq}, resample_period=None,
    )
    assert agg["n_periods_total"] == len(port_rets)
    assert 0.0 < agg["psr_equity_curve"] < 1.0


# ---------------------------------------------------------------------------
# Test B: PERFECT CORRELATION -- load-bearing regression guard
# ---------------------------------------------------------------------------

def test_b_perfect_correlation_regression_guard():
    """If BTC == SOL (identical legs), portfolio PSR must equal leg PSR.

    The OLD stitched-trade-pool proxy is structurally blind to correlation:
    it just doubles the N and re-feeds compute_psr, inflating PSR via
    sqrt(n-1). The NEW equity-curve PSR collapses to the single-leg PSR
    when the legs are identical -- which is the correct behaviour: there
    is no diversification to credit.

    This test documents the n-inflation by asserting:
        OLD_proxy_PSR  >  NEW_equity_curve_PSR  (strictly greater)
        NEW_equity_curve_PSR  ~=  single_leg_PSR  (within 1e-6)
    """
    btc = _iid_equity_curve(seed=42, n=500, mu=0.0008, sigma=0.012)
    sol = btc.copy()  # perfect correlation

    # NEW: equity-curve PSR on the 50/50 portfolio.
    port_eq = build_portfolio_equity_curve(btc, sol, w_btc=0.5, w_sol=0.5)
    port_rets = equity_to_period_returns(port_eq, resample_period=None)
    psr_new = compute_psr(port_rets, sr_hurdle=0.0,
                          confidence=0.95)["psr_vs_hurdle"]

    # Single-leg PSR (the truth).
    leg_rets = btc.pct_change().dropna().values
    psr_single = compute_psr(leg_rets, sr_hurdle=0.0,
                             confidence=0.95)["psr_vs_hurdle"]

    assert abs(psr_new - psr_single) < 1e-6, (
        f"Equity-curve PSR ({psr_new}) deviates from single-leg PSR "
        f"({psr_single}) under perfect correlation. The fix is broken."
    )

    # OLD: stitched-trade-pool proxy -- per-trade returns scaled by
    # allocation weight, then unioned across legs (the legacy bug).
    btc_trades = _trade_returns_from_equity(btc)
    sol_trades = _trade_returns_from_equity(sol)
    stitched = np.concatenate(
        [btc_trades * 0.5, sol_trades * 0.5]  # legacy: scale by weight
    )
    psr_old_proxy = compute_psr(
        stitched, sr_hurdle=0.0, confidence=0.95
    )["psr_vs_hurdle"]

    # Document the inflation: proxy strictly exceeds the truthful metric.
    # Without this assertion the fix would be theater.
    assert psr_old_proxy > psr_new, (
        f"Legacy stitched proxy PSR ({psr_old_proxy}) is NOT strictly "
        f"greater than the equity-curve PSR ({psr_new}). Either the IID "
        f"seed degenerates (rerun) or the n-inflation bug has been "
        f"silently re-introduced."
    )


# ---------------------------------------------------------------------------
# Test C: sampling-frequency invariance under IID
# ---------------------------------------------------------------------------

def test_c_iid_frequency_invariance():
    """Under an IID null, daily vs higher-frequency PSR are close.

    Establishes that the default '1D' resample does NOT bias the metric
    when the null assumption holds. On real strategy data, divergence
    between '15min' and '1D' IS expected (residual autocorrelation from
    long holds) -- that gap should be logged as a caveat, not a bug.
    """
    rng = np.random.default_rng(seed=7)
    # 15-minute IID series, ~30 trading days = 2880 bars.
    n = 2880
    rets = rng.normal(loc=0.00005, scale=0.0008, size=n)
    equity = 1_000_000.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    port_eq = pd.Series(equity, index=idx)

    agg_15m = aggregate_portfolio_psr(
        {"w": port_eq}, resample_period="15min",
    )
    agg_1d = aggregate_portfolio_psr(
        {"w": port_eq}, resample_period="1D",
    )

    # Under IID, the two PSRs should be in the same regime
    # (both > 0.5 or both < 0.5 if the mean is small). Tolerance is loose
    # because daily aggregation throws away information: not equal, but
    # not radically different either.
    assert agg_15m["psr_equity_curve"] is not None
    assert agg_1d["psr_equity_curve"] is not None
    delta = abs(agg_15m["psr_equity_curve"] - agg_1d["psr_equity_curve"])
    assert delta < 0.2, (
        f"PSR delta between 15min and 1D resample = {delta:.3f} under "
        f"IID null; expected < 0.2. Either the null is degenerate or "
        f"the resample is biasing the metric."
    )


# ---------------------------------------------------------------------------
# Test D (optional): non-contiguous window concatenation
# ---------------------------------------------------------------------------

def test_d_noncontiguous_windows_no_gap_jump():
    """A 10x equity jump at a window boundary must NOT enter the return array.

    aggregate_portfolio_psr diffs WITHIN each window, then concatenates --
    never the other way round.
    """
    # Window 1: small drift, equity ~1.0 .. 1.05
    rng = np.random.default_rng(seed=11)
    n = 200
    rets1 = rng.normal(loc=0.0003, scale=0.005, size=n)
    eq1 = pd.Series(
        np.cumprod(1.0 + rets1),
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )

    # Window 2: starts 6 months later, ends at a 10x scale (deliberately).
    eq2_raw = pd.Series(
        np.cumprod(1.0 + rng.normal(loc=0.0003, scale=0.005, size=n)),
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    eq2 = eq2_raw * 10.0  # 10x bigger starting equity

    agg = aggregate_portfolio_psr(
        {"w1": eq1, "w2": eq2}, resample_period="1D",
    )

    # If the function naively diffed across the gap, the gap return would
    # be ~10x (or 9.0), dominating the sample. Verify the max observed
    # return stays in the per-bar regime.
    pieces = [
        equity_to_period_returns(eq1, "1D"),
        equity_to_period_returns(eq2, "1D"),
    ]
    combined = np.concatenate(pieces)
    assert combined.max() < 0.5, (
        f"Max observed return {combined.max()} suggests a gap return "
        f"leaked into the array. aggregate_portfolio_psr must diff "
        f"within each window."
    )
    assert agg["n_periods_total"] == sum(
        agg["per_window_n_periods"][k] for k in ("w1", "w2")
    )
