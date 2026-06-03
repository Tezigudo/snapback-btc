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

def compute_psr(
    returns: np.ndarray,
    sr_hurdle: float = 0.0,
    confidence: float = 0.95,
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

    Returns
    -------
    dict with keys:
        n_trades, point_sharpe, sr_se_lo, psr_vs_hurdle, min_trl, skew, kurt,
        interpretation.

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
