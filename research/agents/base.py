"""
Researcher-agent interface — interface only, no implementation here.

The trading loop NEVER calls a Researcher. Researchers consume *fold
results* after walk-forward runs and produce human-facing commentary or
suggested next-sweep ranges. This isolation is the whole point of the
seam: an LLM-backed researcher could be plugged in later without
touching trading code or violating the "no LLM in trading loop" rule.

See AGENT_ROLES.md (sibling file) for the role taxonomy this protocol
is shaped after (mirrored from TradingAgents and AgentQuant patterns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class FoldResult:
    """One walk-forward fold: train window + winning params + OOS metrics."""

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_params: dict[str, Any] = field(default_factory=dict)
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    test_return_pct: float = 0.0
    test_after_funding_pct: float = 0.0
    trades: int = 0
    max_drawdown_pct: float = 0.0


@runtime_checkable
class Researcher(Protocol):
    """Produces post-hoc commentary + tighter sweep ranges from fold results."""

    def commentary(self, folds: list[FoldResult]) -> str: ...

    def next_sweep_ranges(self, folds: list[FoldResult]) -> dict[str, list[Any]]: ...
