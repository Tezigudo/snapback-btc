"""
Thin shim — DivergenceV1 single-window backtest.

The generalised runner is now ``tools/run_strategy_experiment.py``.
This shim preserves back-compat: callers that directly invoke
``run_divergence_experiment.py`` continue to work unchanged.

Usage (unchanged from before):
    python tools/run_divergence_experiment.py \\
        --config-json '{"swing_k": 5, "rsi_oversold_zone": 30}' \\
        --start 2022-01-01 --end 2022-06-30

Output shape and stdout/stderr conventions are identical to the old file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_strategy_experiment import main  # noqa: E402

if __name__ == "__main__":
    # Inject --strategy-class divergence default if not already given,
    # so the shim is byte-identical in output to the old implementation.
    argv = sys.argv[1:]
    if "--strategy-class" not in argv:
        argv = ["--strategy-class", "strategy.signals_divergence:DivergenceV1"] + argv
    raise SystemExit(main(argv))
