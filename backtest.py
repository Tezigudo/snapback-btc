"""
Backtest harness (backtesting.py). Implemented in P1+P2.

MUST model: 0.04% taker fee * 2 sides, slippage (>=1 tick), funding every 8h,
sequential bars (no shuffling), realistic fills (no bar-close fills).

Walk-forward (P3) wraps this with sliding train/test folds + held-out OOS.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Backtester is implemented in P1+P2.")


if __name__ == "__main__":
    main()
