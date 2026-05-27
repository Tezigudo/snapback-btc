"""Pattern detectors for the C&H hybrid family (production-safe).

Copied verbatim (with light refactor) from `tools/icnh_mega_sweep.py` so the
live bot does not depend on the research tree (`tools/` is not in the wheel —
see pyproject.toml `[tool.hatch.build.targets.wheel] packages`).

Two detectors:

  - `detect_distribution_top` (DT): uptrend → sideways chop → breakdown bar.
    Matches the user's image-16 visual rule. SHORT direction only here
    (the LONG mirror lives in tools and is not deployed).

  - `detect_inverse_cnh` (ICnH): inverted parabolic cup with a small handle.
    SHORT direction only here.

Both detectors are pure functions over a dataframe at a specific bar index.
They do NOT scan; they answer "does the pattern END at this bar?". The live
evaluator does the per-bar lookup.

Regression test `tests/test_cnh_hybrid_short.py::test_detectors_match_tools`
guards against divergence from the in-house grid sweep that produced the
locked Phase 1 numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.indicators import atr, ema


@dataclass
class HybridConfig:
    """Knobs for the HYBRID short detectors.

    Defaults match the Phase 1-locked winner:
        dedup_bars=15, sl_atr_mult=1.5, tp_emas=("ema100",),
        entry_emas=("ema24",), tf="4h".
    """
    # DT params
    uptrend_bars: int = 16
    chop_bars: int = 8
    min_rise_pct: float = 2.5
    max_chop_ratio: float = 0.55
    require_chop_at_top: bool = True
    breakdown_mode: str = "chop_low_or_ema24"
    # ICnH params
    cup_len: int = 20
    handle_len: int = 4
    min_r2: float = 0.50
    min_cup_depth_atr: float = 1.0
    handle_max_depth_frac: float = 0.70
    peak_tolerance: int = 6
    entry_max_bars_after_handle: int = 8
    # Entry / SL / TP
    entry_emas: tuple = ("ema24",)
    sl_atr_mult: float = 1.5
    tp_emas: tuple = ("ema100",)
    # Indicator periods
    atr_period: int = 14
    # Pattern-level dedup (bars). Phase 1 winner = 15.
    dedup_bars: int = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def attach_indicators(df: pd.DataFrame, cfg: HybridConfig | None = None) -> pd.DataFrame:
    """Return a copy of `df` (lowercase OHLCV) with the indicator columns
    the detectors expect: ema7, ema24, ema50, ema100, ema200, atr14 (or the
    period specified in cfg)."""
    cfg = cfg or HybridConfig()
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"missing column: {col}")
    out["ema7"] = ema(out["close"], 7)
    out["ema24"] = ema(out["close"], 24)
    out["ema50"] = ema(out["close"], 50)
    out["ema100"] = ema(out["close"], 100)
    out["ema200"] = ema(out["close"], 200)
    out[f"atr{cfg.atr_period}"] = atr(
        out["high"], out["low"], out["close"], cfg.atr_period
    )
    # Keep `atr14` as a stable alias regardless of cfg.atr_period.
    if "atr14" not in out.columns:
        out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    return out


def _fit_parabola_r2(y: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(y), dtype=float)
    coeffs = np.polyfit(x, y, 2)
    a = coeffs[0]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(r2)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_distribution_top(
    df: pd.DataFrame, end_idx: int, cfg: HybridConfig
) -> dict | None:
    """Distribution-top SHORT pattern ending at `end_idx`.

    chop slice EXCLUDES `end_idx` — current bar is the breakdown trigger.
    """
    total = cfg.uptrend_bars + cfg.chop_bars + 1
    start = end_idx - total + 1
    if start < 0:
        return None
    up = df.iloc[start : start + cfg.uptrend_bars]
    chop = df.iloc[start + cfg.uptrend_bars : end_idx]
    if len(up) < 3 or len(chop) < 3:
        return None
    up_start = float(up["close"].iloc[0])
    up_end = float(up["close"].iloc[-1])
    up_pct = (up_end - up_start) / up_start * 100
    if up_pct < cfg.min_rise_pct:
        return None
    up_range = float(up["high"].max() - up["low"].min())
    if up_range <= 0:
        return None
    chop_range = float(chop["high"].max() - chop["low"].min())
    chop_high = float(chop["high"].max())
    chop_low = float(chop["low"].min())
    if chop_range > up_range * cfg.max_chop_ratio:
        return None
    if cfg.require_chop_at_top:
        chop_mid = (chop_high + chop_low) / 2
        up_top = float(up["high"].max())
        up_bot = float(up["low"].min())
        if chop_mid < (up_top + up_bot) / 2:
            return None
    bar_close = float(df["close"].iloc[end_idx])
    bar_open = float(df["open"].iloc[end_idx])
    ema24 = float(df["ema24"].iloc[end_idx])
    triggered = False
    trigger_kind = ""
    if cfg.breakdown_mode in ("chop_low", "either", "chop_low_or_ema24"):
        if bar_close < chop_low and bar_open >= chop_low:
            triggered = True
            trigger_kind = "chop_low"
    if not triggered and cfg.breakdown_mode in ("ema24", "either", "chop_low_or_ema24"):
        prev_close = float(df["close"].iloc[end_idx - 1])
        prev_ema24 = float(df["ema24"].iloc[end_idx - 1])
        if prev_close >= prev_ema24 and bar_close < ema24:
            triggered = True
            trigger_kind = "ema24"
    if not triggered:
        return None
    return {
        "chop_low": chop_low, "chop_high": chop_high, "up_pct": up_pct,
        "trigger": trigger_kind,
    }


def detect_inverse_cnh(
    df: pd.DataFrame, end_idx: int, cfg: HybridConfig
) -> dict | None:
    """Inverse cup-and-handle SHORT pattern with handle ending at `end_idx`.

    Pure pattern detection — does NOT include the EMA24 breakdown entry
    trigger. Caller does that next, see `find_inverse_cnh_entry`.
    """
    handle_start = end_idx - cfg.handle_len + 1
    cup_end = handle_start - 1
    cup_start = cup_end - cfg.cup_len + 1
    if cup_start < 0 or end_idx >= len(df):
        return None
    cup = df.iloc[cup_start : cup_end + 1]
    handle = df.iloc[handle_start : end_idx + 1]
    closes = cup["close"].to_numpy()
    if np.any(np.isnan(closes)):
        return None
    a, r2 = _fit_parabola_r2(closes)
    # SHORT inverse cup: parabola coefficient negative (concave-down).
    if a >= 0:
        return None
    if r2 < cfg.min_r2:
        return None
    peak_pos = int(cup["high"].values.argmax())
    if abs(peak_pos - cfg.cup_len // 2) > cfg.peak_tolerance:
        return None
    atr_val = float(df["atr14"].iloc[cup_end])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    peak_price = float(cup["high"].iloc[peak_pos])
    left_lip = float(cup["low"].iloc[: peak_pos + 1].min())
    right_lip = float(cup["low"].iloc[peak_pos:].min())
    base = min(left_lip, right_lip)
    cup_depth = peak_price - base
    if cup_depth < cfg.min_cup_depth_atr * atr_val:
        return None
    midpoint = (peak_price + base) / 2.0
    if handle["low"].min() < midpoint:
        return None
    handle_range = float(handle["high"].max() - handle["low"].min())
    if handle_range > cfg.handle_max_depth_frac * cup_depth:
        return None
    return {"r2": r2, "peak": peak_price, "cup_depth": cup_depth}


def is_ema_breakdown(
    df: pd.DataFrame, idx: int, ema_name: str = "ema24"
) -> bool:
    """True iff the bar at `idx` closed BELOW `ema_name` after being at-or-above
    the prior bar's `ema_name`. Mirrors simulate_trades' SHORT entry trigger
    for ICnH patterns.
    """
    if idx <= 0:
        return False
    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    close = float(row["close"])
    prev_close = float(prev["close"])
    ema_now = float(row[ema_name])
    ema_prev = float(prev[ema_name])
    if not all(np.isfinite([close, prev_close, ema_now, ema_prev])):
        return False
    return prev_close >= ema_prev and close < ema_now
