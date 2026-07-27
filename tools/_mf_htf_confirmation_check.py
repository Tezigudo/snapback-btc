"""Does multifactor-v1 need MORE higher-timeframe confirmation? (God, 2026-07-27)

Question raised after the 07-25 short (-$4.83, SL): "1h/4h looked like an
uptrend while 15m was down — should the bot check the higher-TF trend before
firing, or is it not necessary?"

v1 already gates on the 4H EMA200 LEVEL (use_mtf_4h_gate, live since
2026-06-02). This experiment layers the two natural strengthenings on top of
the DEPLOYED config and asks whether either would have earned its keep:

  baseline_deployed   MultiFactorMTF4H (4H EMA200 level gate) — what is live
  plus_1h_level       + short only when close < 1H EMA200 (long mirror)
  plus_4h_slope       + short only when 4H EMA200 is FALLING (long: rising)
  plus_both           both of the above

Windows: the 5 canonical OOS windows (2022H1..2025H1, comparable to
MULTIFACTOR_V1_DEEPENING.md) + 2025H2 + 2026YTD (fresh OOS — parquet runs
through 2026-07-25, so the losing trade's regime is in-sample for the test).

Read-only research: no strategy/, config/, or live files touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import ema  # noqa: E402
from tools.run_mf_deepening import (  # noqa: E402
    PARQ,
    MultiFactorMTF4H,
    _load_4h_ema200_aligned,
    _load_slice,
    run_one,
    summarize,
)

WINDOWS = [
    ("2022H1", "2022-01-01", "2022-06-30"),
    ("2023H1", "2023-01-01", "2023-06-30"),
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2025H2", "2025-07-01", "2025-12-31"),
    ("2026YTD", "2026-01-01", "2026-07-24"),
]

PARQ_4H = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
PARQ_1H = ROOT / "data" / "historical" / "BTC_USDT_USDT_1h.parquet"


def _aligned_ema_and_slope(dates_15m: pd.DatetimeIndex, parquet: Path,
                           bar_hours: int, period: int = 200
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Lookahead-safe EMA(period) + its per-bar slope from a higher-TF parquet,
    aligned to 15m bar timestamps (same close-time merge_asof pattern as
    _load_4h_ema200_aligned)."""
    df = pd.read_parquet(parquet)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    e = ema(df["close"], period)
    slope = e.diff()
    close_times = pd.DatetimeIndex(
        (df.index + pd.Timedelta(hours=bar_hours)).astype("datetime64[us]"))
    left = pd.DataFrame(index=pd.DatetimeIndex(dates_15m.astype("datetime64[us]")))
    right = pd.DataFrame({"ema": e.values, "slope": slope.values},
                         index=close_times).sort_index()
    merged = pd.merge_asof(left, right, left_index=True, right_index=True,
                           direction="backward")
    return merged["ema"].values, merged["slope"].values


class MF4HPlus1HLevel(MultiFactorMTF4H):
    """Deployed 4H gate + 1H EMA200 level confirmation."""

    def init(self) -> None:
        super().init()
        self._ema1h = np.asarray(self.data.Ema1h)

    def _long_signal(self, i: int) -> bool:
        v = self._ema1h[i]
        if not (np.isfinite(v) and self.data.Close[-1] > v):
            return False
        return super()._long_signal(i)

    def _short_signal(self, i: int) -> bool:
        v = self._ema1h[i]
        if not (np.isfinite(v) and self.data.Close[-1] < v):
            return False
        return super()._short_signal(i)


class MF4HPlusSlope(MultiFactorMTF4H):
    """Deployed 4H gate + 4H EMA200 slope direction confirmation."""

    def init(self) -> None:
        super().init()
        self._slope4h = np.asarray(self.data.Slope4h)

    def _long_signal(self, i: int) -> bool:
        s = self._slope4h[i]
        if not (np.isfinite(s) and s > 0):
            return False
        return super()._long_signal(i)

    def _short_signal(self, i: int) -> bool:
        s = self._slope4h[i]
        if not (np.isfinite(s) and s < 0):
            return False
        return super()._short_signal(i)


class MF4HPlusBoth(MF4HPlus1HLevel):
    """Deployed 4H gate + 1H level + 4H slope."""

    def init(self) -> None:
        super().init()
        self._slope4h = np.asarray(self.data.Slope4h)

    def _long_signal(self, i: int) -> bool:
        s = self._slope4h[i]
        if not (np.isfinite(s) and s > 0):
            return False
        return super()._long_signal(i)

    def _short_signal(self, i: int) -> bool:
        s = self._slope4h[i]
        if not (np.isfinite(s) and s < 0):
            return False
        return super()._short_signal(i)


VARIANTS = {
    "baseline_deployed": MultiFactorMTF4H,
    "plus_1h_level":     MF4HPlus1HLevel,
    "plus_4h_slope":     MF4HPlusSlope,
    "plus_both":         MF4HPlusBoth,
}


def main() -> int:
    slices: dict[str, pd.DataFrame] = {}
    print("[htf_check] loading slices ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        df = _load_slice(PARQ["BTC"], start, end, attach_funding=True)
        df["Ema4h"] = _load_4h_ema200_aligned(df.index, PARQ_4H)
        ema1h, _ = _aligned_ema_and_slope(df.index, PARQ_1H, bar_hours=1)
        _, slope4h = _aligned_ema_and_slope(df.index, PARQ_4H, bar_hours=4)
        df["Ema1h"] = ema1h
        df["Slope4h"] = slope4h
        slices[label] = df
        print(f"  {label} bars={len(df)}", file=sys.stderr)

    out: dict = {"windows": [w[0] for w in WINDOWS], "variants": {}}
    for vname, cls in VARIANTS.items():
        print(f"[htf_check] variant={vname}", file=sys.stderr)
        per_window = {}
        for label, _s, _e in WINDOWS:
            per_window[label] = run_one(slices[label], {}, strategy_class=cls)
        out["variants"][vname] = summarize(vname, per_window)
        s = out["variants"][vname]["summary"]
        print(f"  -> trades={s['n_trades']} compounded={s['compounded_pct']}% "
              f"wins={s['windows_positive']} "
              f"per_window={s['per_window_return_pct']}", file=sys.stderr)

    base = out["variants"]["baseline_deployed"]["summary"]["compounded_pct"]
    for v in out["variants"].values():
        v["delta_compounded_pp"] = round(v["summary"]["compounded_pct"] - base, 4)

    out_path = ROOT / "reports" / "mf_htf_confirmation_check.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[htf_check] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
