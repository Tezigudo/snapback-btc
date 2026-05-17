"""
DeterministicResearcher — the zero-API-cost default. Pure stats.

Honest about what it sees: median/range OOS Sharpe, fold stability, most
common winning param values. No invented reasoning, no narrative. When
an LLM-backed researcher arrives later, it can call this as a fallback or
augment its prompt with these stats.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from .base import FoldResult


class DeterministicResearcher:
    """No LLM. No network. No paid API. Just descriptive stats over folds."""

    def commentary(self, folds: list[FoldResult]) -> str:
        if not folds:
            return "No folds to comment on."

        test_sharpes = [f.test_sharpe for f in folds]
        test_returns = [f.test_after_funding_pct for f in folds]
        train_sharpes = [f.train_sharpe for f in folds]
        wins = sum(1 for s in test_sharpes if s > 0)

        try:
            median_test = statistics.median(test_sharpes)
        except statistics.StatisticsError:
            median_test = 0.0
        try:
            median_ret = statistics.median(test_returns)
        except statistics.StatisticsError:
            median_ret = 0.0

        drift = []
        for tr, te in zip(train_sharpes, test_sharpes):
            if abs(tr) > 1e-6:
                drift.append((tr - te) / abs(tr) * 100.0)
        median_drift = statistics.median(drift) if drift else float("nan")

        lines = [
            f"Folds evaluated: {len(folds)}",
            f"Stability (test_sharpe > 0): {wins}/{len(folds)} = "
            f"{100.0 * wins / len(folds):.0f}%",
            f"Median OOS Sharpe:   {median_test:+.2f}",
            f"OOS Sharpe range:    [{min(test_sharpes):+.2f}, {max(test_sharpes):+.2f}]",
            f"Median OOS return:   {median_ret:+.2f}%",
            f"Median train→test drift: {median_drift:+.0f}%  "
            f"(positive = test worse than train; large drift = overfit)",
            "",
            "Winning param values across folds:",
        ]
        param_keys = sorted(folds[0].chosen_params.keys()) if folds else []
        for key in param_keys:
            counts = Counter(f.chosen_params.get(key) for f in folds)
            top = ", ".join(f"{v}×{c}" for v, c in counts.most_common(3))
            lines.append(f"  {key}: {top}")

        return "\n".join(lines)

    def next_sweep_ranges(self, folds: list[FoldResult]) -> dict[str, list[Any]]:
        """Suggest tighter sweep ranges = top-2 most common winning values per param."""
        if not folds:
            return {}
        out: dict[str, list[Any]] = {}
        for key in folds[0].chosen_params:
            counts = Counter(f.chosen_params.get(key) for f in folds)
            out[key] = [v for v, _ in counts.most_common(2)]
        return out
