"""Lo (2002) serial-correlation correction tests for tools/psr_eval.compute_psr.

The correction is ADDITIVE: compute_psr emits BOTH the uncorrected
`psr_vs_hurdle` (the locked v1 0.978 reference family — never touched) AND a new
`psr_lo_adjusted`. Positive serial correlation in an ordered series inflates the
naive Sharpe's confidence; Lo deflates the psr_z denominator via a
variance-inflation factor, so:

  - IID / near-zero autocorrelation  -> psr_lo_adjusted ~= psr_vs_hurdle
  - positive AR(1) autocorrelation   -> psr_lo_adjusted  <  psr_vs_hurdle

Gating (Trap 2): the correction is a no-op when contiguous=False (disjoint
windows) or n < lo_min_n.

Anchored by SIGN, not a recalled coefficient — the exact triangular weight /
truncation lag is a defensible canonical choice, not pinned by any assertion.
Fixtures are tuned so the uncorrected psr stays well below 1.0 (room for the
deflation to show) and n is comfortably above the gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.psr_eval import compute_psr  # noqa: E402


def _ar1(phi: float, mu: float, sigma: float, n: int, seed: int) -> np.ndarray:
    """Generate a positive-mean AR(1) return series x_t = phi*x_{t-1} + e_t + mu."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + mu


# ----- additivity: the uncorrected key is always present and untouched -------


def test_psr_lo_adjusted_key_present_in_all_shapes():
    # Main return shape.
    rng = np.random.default_rng(0)
    main = compute_psr(rng.normal(0.5, 1.0, size=50), contiguous=True)
    assert "psr_lo_adjusted" in main
    assert "psr_vs_hurdle" in main  # uncorrected NEVER dropped

    # std==0 / n<2 early-return stub.
    stub = compute_psr(np.array([3.0, 3.0, 3.0]))  # zero variance
    assert stub["psr_lo_adjusted"] == 0.0
    assert stub["psr_vs_hurdle"] == 0.0

    tiny = compute_psr(np.array([1.0]))  # n < 2
    assert tiny["psr_lo_adjusted"] == 0.0


# ----- IID: Lo correction ~no-op (no serial correlation to deflate) ----------


def test_iid_lo_adjusted_approx_equals_psr():
    # Modest mu/sigma + n=400 (seed chosen so the realized sample autocorr is
    # genuinely near zero) so the UNCORRECTED psr lands in the discriminating
    # band (~0.90, not saturated at 1.0), leaving headroom for the comparison
    # to be meaningful.
    rng = np.random.default_rng(0)
    iid = rng.normal(loc=0.10, scale=1.0, size=400)
    r = compute_psr(iid, contiguous=True)
    # Sanity: psr is in the discriminating band, not saturated.
    assert 0.55 < r["psr_vs_hurdle"] < 0.999
    # No serial correlation -> correction is approximately a no-op (tolerance,
    # not exact: finite-sample autocorrelations are not exactly zero).
    assert r["psr_lo_adjusted"] == pytest.approx(r["psr_vs_hurdle"], abs=0.05)


# ----- positive AR(1): Lo STRICTLY deflates ----------------------------------


def test_positive_ar1_lo_adjusted_strictly_less_than_psr():
    x = _ar1(phi=0.6, mu=0.10, sigma=1.0, n=200, seed=11)
    r = compute_psr(x, contiguous=True)
    # Uncorrected psr must be in the discriminating band (not saturated at 1.0,
    # or the strict-less comparison would break on rounding).
    assert 0.55 < r["psr_vs_hurdle"] < 0.999
    # Positive serial correlation inflates the naive Sharpe -> Lo deflates it.
    assert r["psr_lo_adjusted"] < r["psr_vs_hurdle"]
    # The deflation factor (lo_eta = 1/sqrt(VIF)) is < 1 under positive AR.
    assert r["lo_eta"] < 1.0
    # The uncorrected value is preserved exactly (additive contract).
    bare = compute_psr(x, contiguous=False)  # Lo off
    assert r["psr_vs_hurdle"] == bare["psr_vs_hurdle"]


# ----- gate (Trap 2): no-op on disjoint windows ------------------------------


def test_gate_contiguous_false_is_noop():
    x = _ar1(phi=0.6, mu=0.10, sigma=1.0, n=200, seed=11)
    r = compute_psr(x, contiguous=False)
    assert r["psr_lo_adjusted"] == r["psr_vs_hurdle"]
    assert r["lo_eta"] == 1.0


def test_gate_disjoint_walkforward_series_is_noop():
    # n=5 disjoint OOS window returns (the psr_walkforward shape). Even with
    # apparent autocorrelation, Lo must be a no-op (spurious across boundaries).
    wf = np.array([5.0, -3.0, 8.0, 1.0, 4.0])
    r = compute_psr(wf, contiguous=False)
    assert r["psr_lo_adjusted"] == r["psr_vs_hurdle"]


def test_gate_below_min_n_is_noop():
    # n=10 < lo_min_n default (20) -> no-op even when contiguous=True.
    x = _ar1(phi=0.6, mu=0.10, sigma=1.0, n=10, seed=3)
    r = compute_psr(x, contiguous=True)
    assert r["psr_lo_adjusted"] == r["psr_vs_hurdle"]


# ----- fires when above min_n AND contiguous ---------------------------------


def test_lo_fires_above_min_n_when_contiguous():
    x = _ar1(phi=0.6, mu=0.10, sigma=1.0, n=200, seed=11)
    fired = compute_psr(x, contiguous=True)
    noop = compute_psr(x, contiguous=False)
    # The two must differ under positive AR -> the gate actually fired.
    assert fired["psr_lo_adjusted"] != noop["psr_lo_adjusted"]
