"""
PSR + MinTRL evaluator for trade-level P&L series.

CLI:
    python tools/psr_eval.py --trades trades.csv --sr-hurdle 0.0 --confidence 0.95

Where trades.csv has column `pnl_pct` (one row per trade, percent return).

Implements Bailey & López de Prado (2012) PSR and MinTRL, with Lo (2002) asymptotic SE.
No scipy dependency — uses math.erf for the normal CDF.

Output: JSON to stdout.

============================================================================
VALID INPUT-SERIES SEMANTICS (methodology debt #1)
============================================================================
`compute_psr` is unit-agnostic but assumes the input series is drawn from a
single contiguous return-generating process.  Two legitimate constructions:

  (a) Equity-impact returns within a SINGLE contiguous window — that is,
      `PnL_i / equity_at_entry_i * 100`.  This is sizing-aware and matches
      backtesting.py's `Return [%]` headline.  Use
      `tools.aggregate.equity_impact_returns(stats, cash)` to build it.

  (b) Window-level returns across a walk-forward quarterly sequence —
      one observation per quarter.  Use this for the multi-window
      "psr_walkforward" emitted by `tools.aggregate.aggregate_windows`.

STITCHED per-trade `ReturnPct` across DISJOINT OOS windows (the historic
default for several runners) is N-INFLATED and sizing-blind: it treats
trades from 2022_H1 and 2025_H1 as if drawn from the same process, which
artificially shrinks the standard error and overstates PSR.

If you need the stitched form for backcompat diff only, use
`tools.aggregate.legacy_stitched_psr()` which adds a `deprecation` flag
to the returned dict.  Never quote it as the canonical PSR.
============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Statistical helpers (no scipy)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf. Accurate to ~1e-15."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF) — Newton's method on erf.

    Starts from a rational approximation, then refines with one Newton step.
    Accurate to ~1e-10 for p in (1e-10, 1-1e-10).
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0,1), got {p}")
    # Rational approximation (Abramowitz & Stegun 26.2.17 variant)
    # for p in (0.5, 1); mirror for p < 0.5
    sign = 1.0 if p >= 0.5 else -1.0
    q = p if p >= 0.5 else 1.0 - p
    t = math.sqrt(-2.0 * math.log(1.0 - q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x0 = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
    # One Newton step to improve accuracy
    x0 = x0 - (_norm_cdf(sign * x0) - q) / (math.exp(-0.5 * x0**2) / math.sqrt(2 * math.pi))
    return sign * x0


def _skew_kurt(returns: np.ndarray) -> tuple[float, float]:
    """Compute skewness and RAW kurtosis (normal=3) from returns array.

    Uses the standard unbiased formulas (Fisher-corrected for skew, raw kurtosis).
    No scipy needed.
    """
    n = len(returns)
    if n < 4:
        return 0.0, 3.0  # degenerate: assume normal
    mu = returns.mean()
    diffs = returns - mu
    m2 = np.mean(diffs**2)
    m3 = np.mean(diffs**3)
    m4 = np.mean(diffs**4)
    if m2 == 0.0:
        return 0.0, 3.0
    # Sample skewness (Fisher unbiased correction)
    skew = (math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / (m2 ** 1.5))
    # Raw kurtosis (not excess; normal=3).
    # Use the sample formula with bias correction similar to scipy's fisher=False.
    raw_kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / m2**2) - 3 * (n - 1)) + 3.0
    return float(skew), float(raw_kurt)


# ---------------------------------------------------------------------------
# PSR / MinTRL (Bailey & López de Prado 2012)
# ---------------------------------------------------------------------------

# Lag-k autocorrelation truncation default and minimum n for the Lo correction.
_LO_DEFAULT_MIN_N = 20


def _lag_autocorr(returns: np.ndarray, k: int) -> float:
    """Sample lag-k autocorrelation rho_k of an ORDERED return series.

    Uses the standard biased (N-denominator) estimator on mean-centered
    returns — the form Lo (2002) assumes for the variance-inflation factor.
    Returns 0.0 on degenerate (zero-variance / too-short) input.
    """
    n = len(returns)
    if n <= k or k < 1:
        return 0.0
    x = returns - returns.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    num = float(np.dot(x[:-k], x[k:]))
    return num / denom


def compute_psr(
    returns: np.ndarray,
    sr_hurdle: float = 0.0,
    confidence: float = 0.95,
    *,
    contiguous: bool = True,
    lo_min_n: int = _LO_DEFAULT_MIN_N,
) -> dict:
    """Compute PSR, MinTRL, and related statistics on an array of trade-level returns.

    Parameters
    ----------
    returns:
        1-D array of per-trade P&L percentages (or any unit — Sharpe is unit-neutral).
    sr_hurdle:
        Sharpe hurdle to test against (default 0.0 = "any edge vs coin flip").
    confidence:
        Confidence level for MinTRL / PSR interpretation (default 0.95).
    contiguous:
        Whether `returns` is a SINGLE ordered, contiguous return series for
        which serial correlation is meaningful (True) or a stitched/disjoint
        series across window boundaries where autocorrelation is spurious
        (False). Gates the Lo (2002) serial-correlation correction below.
    lo_min_n:
        Minimum n for the Lo correction to fire (default 20). Below this the
        sample autocorrelations are too noisy to trust.

    Returns
    -------
    dict with keys:
        n_trades, point_sharpe, sr_se_lo, psr_vs_hurdle, psr_lo_adjusted,
        min_trl, skew, kurt, interpretation.

    Lo (2002) serial-correlation correction — ADDITIVE, never replaces PSR:
        `psr_vs_hurdle` above is the IID-ordering PSR and is the locked,
        byte-identical reference (deployed v1 PSR 0.978). Lo (2002, "The
        Statistics of Sharpe Ratios," Financial Analysts Journal) shows that
        positive serial correlation in the ORDERED return series inflates the
        naive Sharpe's confidence. We DEFLATE the psr_z denominator (i.e.
        inflate the SR standard error) by a variance-inflation factor
            VIF = 1 + 2 * sum_{k=1..K} (1 - k/(K+1)) * rho_k
        (Bartlett/Newey-West-style triangular weights; K = min(K_max, n//4)).
        Then psr_z_lo = psr_z / sqrt(max(VIF, eps)) and
        psr_lo_adjusted = Phi(psr_z_lo).
        GATE (Trap 2): the correction is a NO-OP — psr_lo_adjusted ==
        psr_vs_hurdle — when `contiguous=False` (disjoint windows) OR
        n < lo_min_n. Positive autocorrelation strictly lowers
        psr_lo_adjusted vs psr_vs_hurdle; near-zero autocorrelation leaves it
        approximately unchanged.

    Notes on units:
        point_sharpe is computed as mean/std WITHOUT annualization (trade-level,
        not bar-level). The PSR framework is scale-invariant: PSR depends on
        the RATIO (SR - hurdle) / SR_se, which is dimensionally consistent when
        both SR and hurdle are expressed in the same units. For interpretation,
        treat point_sharpe as "mean return per unit of std risk per trade" —
        not directly comparable to an annualized bar-level Sharpe.

    PSR formula (canonical Bailey & López de Prado 2012):
        varterm = 1 - gamma3 * SR + ((gamma4 - 1) / 4) * SR^2
            where gamma3 = skewness, gamma4 = raw kurtosis (normal=3)
        SR_se = sqrt(varterm / N)   [Lo 2002 asymptotic SE]
        PSR   = Phi((SR - hurdle) * sqrt(N-1) / sqrt(varterm))

    Note: PSR = Phi((SR - hurdle) * sqrt(N-1) / sqrt(varterm)), NOT divided by
    SR_se (which would double-count N). The formula divides the signal by sqrt(varterm)
    directly — SR_se is reported separately as a diagnostic.

    MinTRL formula:
        MinTRL = 1 + varterm * (z_conf / (SR - hurdle))^2
    """
    n = len(returns)
    mu = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if n > 1 else 0.0

    if std == 0.0 or n < 2:
        return {
            "n_trades": n,
            "point_sharpe": 0.0,
            "sr_se_lo": 0.0,
            "psr_vs_hurdle": 0.0,
            "psr_lo_adjusted": 0.0,
            "min_trl": int(1e9),
            "skew": 0.0,
            "kurt": 3.0,
            "interpretation": "insufficient_evidence",
        }

    sr = mu / std

    gamma3, gamma4 = _skew_kurt(returns)

    # varterm for non-IID skewed/fat-tailed correction (Bailey & López de Prado 2012 eq. 4)
    varterm = 1.0 - gamma3 * sr + ((gamma4 - 1.0) / 4.0) * sr**2
    # varterm should be positive; guard against numerical issues
    if varterm <= 0.0:
        varterm = 1.0  # fallback to IID normal assumption

    sr_se_lo = math.sqrt(varterm / n)

    # PSR: use sqrt(N-1) / sqrt(varterm) NOT SR_se in denominator
    # (avoids double-counting N — see module docstring and advisor note)
    if n > 1:
        psr_z = (sr - sr_hurdle) * math.sqrt(n - 1) / math.sqrt(varterm)
    else:
        psr_z = 0.0
    psr = _norm_cdf(psr_z)

    # ---------------------------------------------------------------------
    # Lo (2002) serial-correlation correction — ADDITIVE.  Computed into NEW
    # variables only; `sr`, `varterm`, `psr_z`, `psr` are NEVER reassigned, so
    # `psr_vs_hurdle` stays byte-identical (locked v1 0.978 reference).
    #
    # GATE (Trap 2): no-op on disjoint/short series.  When skipped,
    # psr_lo_adjusted == psr (the IID value) so downstream code that reads it
    # degrades gracefully to the uncorrected PSR.
    # ---------------------------------------------------------------------
    if contiguous and n >= lo_min_n:
        # Triangular (Bartlett/Newey-West) truncation lag.
        K = min(int(math.sqrt(n)), n // 4)
        if K < 1:
            K = 1
        vif = 1.0
        for k in range(1, K + 1):
            rho_k = _lag_autocorr(returns, k)
            weight = 1.0 - k / (K + 1.0)
            vif += 2.0 * weight * rho_k
        # VIF < 1 (net-negative autocorrelation) would INFLATE PSR; Lo's
        # correction targets positive serial correlation, so floor at 1.0 to
        # keep the adjustment a one-sided (conservative) deflation and avoid
        # sqrt of a non-positive number.
        vif_eff = max(vif, 1.0)
        psr_z_lo = psr_z / math.sqrt(vif_eff)
        psr_lo_adjusted = _norm_cdf(psr_z_lo)
        sr_lo_adjusted = sr / math.sqrt(vif_eff)
        lo_eta = 1.0 / math.sqrt(vif_eff)
    else:
        # No-op: disjoint windows (spurious autocorr) or too few obs.
        psr_lo_adjusted = psr
        sr_lo_adjusted = sr
        lo_eta = 1.0

    # MinTRL (minimum trades needed to reject SR <= hurdle at `confidence` level)
    z_conf = _norm_ppf(confidence)
    if sr > sr_hurdle:
        min_trl = 1.0 + varterm * (z_conf / (sr - sr_hurdle)) ** 2
        min_trl_int = max(int(math.ceil(min_trl)), 1)
    else:
        # SR <= hurdle: MinTRL is undefined / infinite (no evidence of edge)
        min_trl_int = int(1e9)

    # Interpretation
    if psr > confidence:
        interpretation = "evidence_of_edge"
    elif psr < (1.0 - confidence) and n >= min_trl_int:
        # PSR < 0.05 AND we have enough data to say no edge
        interpretation = "evidence_of_no_edge"
    else:
        interpretation = "insufficient_evidence"

    return {
        "n_trades": n,
        "point_sharpe": round(sr, 6),
        "sr_se_lo": round(sr_se_lo, 6),
        "psr_vs_hurdle": round(psr, 6),
        "psr_lo_adjusted": round(psr_lo_adjusted, 6),
        "sr_lo_adjusted": round(sr_lo_adjusted, 6),
        "lo_eta": round(lo_eta, 6),
        "min_trl": min_trl_int,
        "skew": round(gamma3, 6),
        "kurt": round(gamma4, 6),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute PSR + MinTRL from a trade-level P&L CSV."
    )
    parser.add_argument(
        "--trades", required=True,
        help="CSV file with column 'pnl_pct' (one row per trade).",
    )
    parser.add_argument(
        "--sr-hurdle", type=float, default=0.0,
        help="Sharpe hurdle to test against (default 0.0).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level for PSR interpretation (default 0.95).",
    )
    args = parser.parse_args(argv)

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"ERROR: trades file not found: {trades_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(trades_path)
    except Exception as exc:
        print(f"ERROR: cannot read CSV: {exc}", file=sys.stderr)
        return 1

    if "pnl_pct" not in df.columns:
        print(f"ERROR: CSV must have a 'pnl_pct' column. Got: {list(df.columns)}", file=sys.stderr)
        return 1

    returns = df["pnl_pct"].dropna().values.astype(float)
    if len(returns) == 0:
        print("ERROR: no valid pnl_pct values after dropping NaN.", file=sys.stderr)
        return 1

    result = compute_psr(returns, sr_hurdle=args.sr_hurdle, confidence=args.confidence)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
