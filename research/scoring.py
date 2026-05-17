"""
Scoring helpers for walk-forward ranking. Pure numpy/pandas — no new deps.

The core problem walk-forward solves is multiple-testing inflation: if you
sweep 200 parameter combos on the same data, the best one *will* show a
positive Sharpe by luck, even when no real edge exists. We mitigate two ways:

  1. Deflated Sharpe — subtract a penalty that grows with the number of
     combos tested. Pattern borrowed from AgentQuant's scoring; this is the
     simple sqrt-log version of López de Prado's DSR, sufficient for ranking.

  2. Bootstrap p5 — resample trade returns with replacement, recompute
     Sharpe per resample, report the 5th percentile. A combo whose
     "lucky-streak" Sharpe collapses in bootstrap is identified as fragile.

  3. Fold-stability — fraction of OOS folds where test_sharpe > 0. A real
     edge holds across regimes; a fitted curve only works on one window.
"""

from __future__ import annotations

import numpy as np


def deflated_sharpe(sharpe: float, num_trials: int, periods_per_year: int = 252) -> float:
    """Sharpe minus a multiple-testing penalty.

    Penalty grows with sqrt(2 * log(N)) — the expected max of N independent
    standard normals. Always <= raw Sharpe; equal when num_trials <= 1.
    """
    if num_trials <= 1 or not np.isfinite(sharpe):
        return float(sharpe)
    penalty = float(np.sqrt(2.0 * np.log(num_trials)) / np.sqrt(periods_per_year))
    return float(sharpe - penalty)


def bootstrap_sharpe_p5(
    trade_returns: np.ndarray | list[float],
    n_iter: int = 1000,
    seed: int = 42,
) -> float:
    """5th-percentile Sharpe from bootstrap resamples of trade returns.

    If a strategy's Sharpe is driven by a few lucky trades, bootstrap
    resampling without those trades produces a much lower Sharpe — the p5
    captures that fragility. Fixed seed = deterministic output.
    """
    arr = np.asarray(list(trade_returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    n = arr.size
    sharpes = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample = rng.choice(arr, size=n, replace=True)
        std = sample.std(ddof=1)
        sharpes[i] = 0.0 if std == 0 else sample.mean() / std
    return float(np.percentile(sharpes, 5))


def fold_stability_score(test_sharpes: list[float]) -> float:
    """Fraction of folds with test_sharpe > 0. Returns 0.0 if empty."""
    finite = [s for s in test_sharpes if np.isfinite(s)]
    if not finite:
        return 0.0
    return sum(1 for s in finite if s > 0) / len(finite)


def tail_aware_score(
    after_funding_pct: float,
    max_drawdown_pct: float,
    sharpe: float,
    num_trials: int,
    periods_per_year: int = 252,
    dd_floor_pct: float = 5.0,
) -> float:
    """Calmar-like ranking score that penalises blow-up combos.

    Replaces `deflated_sharpe` as the train-side combo selector. The P3.5
    experiment showed Sharpe-ranked selection picked combos that scored
    high on train (smooth, many small trades) but blew up OOS — because
    Sharpe doesn't see tail risk. This metric rewards return-per-DD and
    still applies the multi-trial deflation penalty.

    Formula:
        score = (after_funding_pct / max(|max_dd_pct|, dd_floor_pct))
              - deflation_penalty

    `dd_floor_pct` prevents division-by-tiny when a combo had no drawdown
    on the train window (a combo with 0.1% return and 0.01% DD shouldn't
    score 10x).
    """
    if not np.isfinite(after_funding_pct) or not np.isfinite(max_drawdown_pct):
        return -1e18
    dd = max(abs(float(max_drawdown_pct)), float(dd_floor_pct))
    base = float(after_funding_pct) / dd
    penalty = 0.0
    if num_trials > 1 and np.isfinite(sharpe):
        penalty = float(np.sqrt(2.0 * np.log(num_trials)) / np.sqrt(periods_per_year))
    return base - penalty
