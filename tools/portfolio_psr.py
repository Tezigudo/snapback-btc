"""True weighted-equity-curve portfolio PSR.

Methodology debt #2 fix. Replaces the stitched-trade-pool PSR proxy used in
``tools/_postfrac_mf_4h_btc_sol_portfolio.py`` (and its walk-forward sibling)
with a correctly-constructed period-return series.

Sum-then-diff pipeline:
    1. Superimpose normalized leg-equity curves on a union DatetimeIndex
       (ffill gaps; default 1.0 before the first tick on a leg = capital
       sits in cash before the first fill).
    2. Weight-sum into one portfolio equity series per window.
    3. Resample to '1D' (mitigates intraday autocorrelation -- documented
       caveat, not eliminated), pct_change, drop leading NaN.
    4. Concatenate the per-window return arrays AFTER differencing
       (NEVER diff across a window boundary -- that would inject a
       spurious return at the gap).
    5. Feed the concatenated array to ``compute_psr``.

Why this absorbs correlation: Var(w_b*r_b + w_s*r_s) carries
2*w_b*w_s*Cov(r_b, r_s); the stitched per-trade union destroys that cross
term and inflates N by ~sqrt(2) via sqrt(n-1) in psr_eval.compute_psr.

Sharpe units: per-period (default daily), NOT per-trade. Reviewers must NOT
compare the new ``point_sharpe_period`` to historical trade-level Sharpes
from other strategies in this repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.psr_eval import compute_psr


def _normalized_equity(eq: "pd.Series | None") -> "pd.Series | None":
    """Normalize a leg equity series to start=1.0; return None on bad input."""
    if eq is None or len(eq) == 0:
        return None
    e0 = float(eq.iloc[0])
    if e0 <= 0 or not np.isfinite(e0):
        return None
    return eq / e0


def build_portfolio_equity_curve(
    btc_eq: "pd.Series | None",
    sol_eq: "pd.Series | None",
    w_btc: float,
    w_sol: float,
) -> "pd.Series | None":
    """Time-aligned weighted-sum portfolio equity for ONE window.

    Union DatetimeIndex, ffill gaps, default 1.0 before the first tick
    on a leg (capital sits in cash before the first fill). The resulting
    series carries the BTC<->SOL co-movement structure into its own vol.
    """
    btc_n = _normalized_equity(btc_eq)
    sol_n = _normalized_equity(sol_eq)
    if btc_n is None and sol_n is None:
        return None
    if btc_n is not None and sol_n is not None:
        idx = btc_n.index.union(sol_n.index).sort_values()
        b = btc_n.reindex(idx).ffill().fillna(1.0)
        s = sol_n.reindex(idx).ffill().fillna(1.0)
    elif btc_n is not None:
        b = btc_n
        s = pd.Series(1.0, index=btc_n.index)
    else:
        s = sol_n  # type: ignore[assignment]
        b = pd.Series(1.0, index=sol_n.index)  # type: ignore[union-attr]
    return w_btc * b + w_sol * s


def equity_to_period_returns(
    port_eq: "pd.Series | None",
    resample_period: "str | None" = "1D",
) -> np.ndarray:
    """Per-window equity -> period returns (default daily).

    NEVER diff across a window boundary. Caller concatenates the arrays.
    Pass ``None`` for ``resample_period`` to skip resampling (returns at
    native frequency; will inflate N and bias PSR upward under any residual
    autocorrelation).
    """
    if port_eq is None or len(port_eq) < 2:
        return np.asarray([], dtype=float)
    if resample_period:
        eq = port_eq.resample(resample_period).last().dropna()
    else:
        eq = port_eq.dropna()
    if len(eq) < 2:
        return np.asarray([], dtype=float)
    return eq.pct_change().dropna().values.astype(float)


def aggregate_portfolio_psr(
    per_window_eq: "dict[str, pd.Series | None]",
    resample_period: str = "1D",
    sr_hurdle: float = 0.0,
    confidence: float = 0.95,
) -> dict:
    """Headline portfolio PSR across non-contiguous windows.

    N becomes the count of PERIOD RETURNS (e.g. ~1080 daily obs across 6
    H1 windows), NOT the trade count. Per-window: equity -> resample ->
    pct_change. Cross-window: concatenate post-differencing -- the gap
    between windows never produces a return observation.
    """
    pieces: list[np.ndarray] = []
    per_window_n: dict[str, int] = {}
    for label, eq in per_window_eq.items():
        r = equity_to_period_returns(eq, resample_period=resample_period)
        per_window_n[label] = int(len(r))
        if len(r) > 0:
            pieces.append(r)
    if not pieces:
        return {
            "psr_equity_curve":     None,
            "psr_interpretation":   "insufficient_evidence",
            "point_sharpe_period":  None,
            "n_periods_total":      0,
            "per_window_n_periods": per_window_n,
            "resample_period":      resample_period,
            "sharpe_units":         "per_period (NOT per-trade)",
        }
    combined = np.concatenate(pieces)
    psr = compute_psr(combined, sr_hurdle=sr_hurdle, confidence=confidence)
    return {
        "psr_equity_curve":     psr.get("psr_vs_hurdle"),
        "psr_interpretation":   psr.get("interpretation"),
        "point_sharpe_period":  psr.get("point_sharpe"),
        "n_periods_total":      int(len(combined)),
        "per_window_n_periods": per_window_n,
        "resample_period":      resample_period,
        "sharpe_units":         (
            "per_period (NOT per-trade -- do not compare to historical "
            "trade-level Sharpe)"
        ),
    }
