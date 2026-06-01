"""Download historical OHLCV for ETH, SOL, ADA, WLD futures.

Uses the existing `exchange.data.load_klines` cache + fetcher, which writes
to `data/historical/{SYMBOL_normalized}_{tf}.parquet` — same naming convention
the cnh-hybrid-short tooling already expects.

Idempotent: subsequent runs only top up missing tails. Safe to interrupt.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exchange.data import load_klines

# Days back. WLD started trading ~Jul 2023 so its full history is ~1000 days.
# Asking for 2200 just returns what exists (no error).
DAYS = 2200

SYMBOLS = [
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "ADA/USDT:USDT",
    "WLD/USDT:USDT",
]

TIMEFRAMES = ["4h", "15m"]


def main() -> int:
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            t0 = time.monotonic()
            print(f"[{sym} {tf}] downloading…", flush=True)
            try:
                df = load_klines(symbol=sym, timeframe=tf, days_back=DAYS)
                elapsed = time.monotonic() - t0
                if df.empty:
                    print(f"[{sym} {tf}] EMPTY (likely not listed) in {elapsed:.1f}s",
                          flush=True)
                else:
                    print(f"[{sym} {tf}] {len(df):,} bars "
                          f"{df.index.min().date()} → {df.index.max().date()} "
                          f"({elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"[{sym} {tf}] FAILED: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
