"""
Timeframe-agnostic data prep for non-snapback strategies (carry, donchian).

snapback's `prepare_strategy_data` hardcodes 15m entry + 1h trend filter
because the strategy thesis specifically requires that multi-timeframe
shape. Carry and Donchian don't need a separate trend TF:

  - carry-v2 only reads Funding + Close
  - donchian-v2 reads DonchianUpper/Lower (computed from the same TF as
    entry, or a higher one — but ATTACHED to entry-TF bars)

So this prep is just: load klines at entry_tf, attach the ffilled funding
column, normalise tz and column names. Donchian's `attach_donchian`
layers channels on top with whatever klines you pass it (we use the same
entry_tf klines — single-TF Donchian, simpler and more direct).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_simple_data(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
) -> pd.DataFrame:
    """Capitalise OHLCV cols, drop tz, ffill funding onto entry bars.

    No indicators are computed — strategy classes that need ATR/EMA
    should compute them via attach_donchian or similar helpers."""
    if klines.empty:
        raise ValueError("klines is empty")

    df = klines.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    if funding is None or funding.empty:
        df["Funding"] = np.nan
    else:
        f = funding.copy()
        if f.index.tz is not None:
            f.index = f.index.tz_convert("UTC").tz_localize(None)
        df["Funding"] = f["funding_rate"].reindex(df.index, method="ffill")

    return df
