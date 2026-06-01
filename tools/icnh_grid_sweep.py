"""Comprehensive grid sweep of ICnH + alternative patterns.

Tests ~60 configs across:
  - Parameter axes around S10 (the best survivor from initial sweep)
  - Alternative patterns: double top, bearish engulfing at high, N-bar breakdown
  - Multiple TFs: 15m, 1h, 4h

Parallelized via ProcessPoolExecutor. Results saved to grid_results.json.

Usage: uv run python tools/icnh_grid_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strategy.indicators import atr, ema, bearish_engulfing  # noqa: E402

DATA = ROOT / "data" / "historical"
RESULTS = ROOT / "data" / "grid_results.json"
FRICTION_BPS = 10.0
ATR_LEN = 14


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    name: str
    pattern_type: str = "inverse_cnh"   # inverse_cnh | classic_cnh | double_top | engulfing_at_high | breakdown_n
    direction: str = "short"
    tf: str = "4h"
    # Pattern params
    cup_len: int = 20
    handle_len: int = 5
    peak_tolerance: int = 3
    min_cup_depth_atr: float = 2.5
    min_r2: float = 0.70
    handle_max_depth_frac: float = 0.45
    entry_max_bars_after_handle: int = 8
    # Alt pattern params
    breakdown_lookback: int = 50      # for breakdown_n pattern
    resistance_proximity_bars: int = 5  # for engulfing_at_high
    double_top_max_pct: float = 1.5   # peaks within X%
    double_top_window: int = 60       # peaks within X bars
    double_top_min_gap: int = 10
    # Entry
    entry_emas: tuple = ("ema24",)
    # SL
    sl_atr_mult: float = 1.5
    regime_sl_mode: str = "off"
    # TP
    tp_emas: tuple = ("ema100", "ema200")
    # Filters
    require_uptrend_for_short: bool = True
    uptrend_min_pct: float = 5.0
    require_downtrend_for_long: bool = False
    note: str = ""


# ============================================================
# DATA
# ============================================================

def load_tf(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"BTC_USDT_USDT_{tf}.parquet").sort_index()
    df["ema7"] = ema(df["close"], 7)
    df["ema24"] = ema(df["close"], 24)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 200)
    df["atr14"] = atr(df["high"], df["low"], df["close"], ATR_LEN)
    df["bear_eng"] = bearish_engulfing(df["open"], df["high"], df["low"], df["close"])
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    return df


# ============================================================
# PATTERN DETECTORS
# ============================================================

def _fit_parabola_r2(y: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(y), dtype=float)
    coeffs = np.polyfit(x, y, 2)
    a = coeffs[0]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(r2)


def _detect_cnh(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    """Inverse C&H (short) or classic C&H (long)."""
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
    if cfg.direction == "short" and a >= 0:
        return None
    if cfg.direction == "long" and a <= 0:
        return None
    if r2 < cfg.min_r2:
        return None
    if cfg.direction == "short":
        peak_pos = int(cup["high"].values.argmax())
    else:
        peak_pos = int(cup["low"].values.argmin())
    if abs(peak_pos - cfg.cup_len // 2) > cfg.peak_tolerance:
        return None
    atr_val = float(df["atr14"].iloc[cup_end])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    if cfg.direction == "short":
        peak_price = float(cup["high"].iloc[peak_pos])
        left_lip = float(cup["low"].iloc[:peak_pos + 1].min())
        right_lip = float(cup["low"].iloc[peak_pos:].min())
        base = min(left_lip, right_lip)
        cup_depth = peak_price - base
    else:
        peak_price = float(cup["low"].iloc[peak_pos])
        left_lip = float(cup["high"].iloc[:peak_pos + 1].max())
        right_lip = float(cup["high"].iloc[peak_pos:].max())
        base = max(left_lip, right_lip)
        cup_depth = base - peak_price
    if cup_depth < cfg.min_cup_depth_atr * atr_val:
        return None
    midpoint = (peak_price + base) / 2.0
    if cfg.direction == "short" and handle["low"].min() < midpoint:
        return None
    if cfg.direction == "long" and handle["high"].max() > midpoint:
        return None
    handle_range = float(handle["high"].max() - handle["low"].min())
    if handle_range > cfg.handle_max_depth_frac * cup_depth:
        return None
    if cfg.direction == "short" and cfg.require_uptrend_for_short:
        e200 = float(df["ema200"].iloc[cup_start])
        c0 = float(df["close"].iloc[cup_start])
        if not np.isfinite(e200) or (c0 / e200 - 1.0) < cfg.uptrend_min_pct / 100.0:
            return None
    if cfg.direction == "long" and cfg.require_downtrend_for_long:
        e200 = float(df["ema200"].iloc[cup_start])
        c0 = float(df["close"].iloc[cup_start])
        if not np.isfinite(e200) or (c0 / e200 - 1.0) > -cfg.uptrend_min_pct / 100.0:
            return None
    return {"r2": r2, "depth": cup_depth, "peak": peak_price}


def _detect_double_top(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    """Two highs within X% of each other, separated by >= min_gap bars,
    within last `double_top_window` bars. Most recent bar must be in pullback
    between them OR just after."""
    win = cfg.double_top_window
    start = end_idx - win + 1
    if start < 0:
        return None
    block = df.iloc[start : end_idx + 1]
    if len(block) < cfg.double_top_min_gap * 2:
        return None
    highs = block["high"].values
    # Find local maxima (3-bar)
    local_max_idx = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            local_max_idx.append(i)
    if len(local_max_idx) < 2:
        return None
    # Find two largest, separated by gap
    sorted_by_h = sorted(local_max_idx, key=lambda i: -highs[i])
    for i, hi1 in enumerate(sorted_by_h):
        for hi2 in sorted_by_h[i + 1:]:
            if abs(hi1 - hi2) < cfg.double_top_min_gap:
                continue
            diff_pct = abs(highs[hi1] - highs[hi2]) / max(highs[hi1], highs[hi2]) * 100
            if diff_pct > cfg.double_top_max_pct:
                continue
            # Pattern present
            atr_val = float(df["atr14"].iloc[end_idx])
            if not np.isfinite(atr_val) or atr_val <= 0:
                return None
            if cfg.require_uptrend_for_short:
                e200 = float(df["ema200"].iloc[start])
                c0 = float(df["close"].iloc[start])
                if not np.isfinite(e200) or (c0 / e200 - 1.0) < cfg.uptrend_min_pct / 100.0:
                    return None
            peak_price = float(max(highs[hi1], highs[hi2]))
            return {"peak": peak_price}
    return None


def _detect_engulfing_at_high(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    """Bearish engulfing within `resistance_proximity_bars` of a 50-bar high."""
    if end_idx < 50:
        return None
    if not bool(df["bear_eng"].iloc[end_idx]):
        return None
    block = df.iloc[end_idx - 49 : end_idx + 1]
    high50 = float(block["high"].max())
    if df["high"].iloc[end_idx] < high50 * 0.99:
        return None  # bar not at resistance
    if cfg.require_uptrend_for_short:
        e200 = float(df["ema200"].iloc[end_idx])
        c0 = float(df["close"].iloc[end_idx])
        if not np.isfinite(e200) or (c0 / e200 - 1.0) < cfg.uptrend_min_pct / 100.0:
            return None
    return {"high50": high50}


def _detect_breakdown_n(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    """Close breaks below N-bar low. Simple control."""
    if end_idx < cfg.breakdown_lookback + 1:
        return None
    block_low = float(df["low"].iloc[end_idx - cfg.breakdown_lookback : end_idx].min())
    if float(df["close"].iloc[end_idx]) >= block_low:
        return None
    if cfg.require_uptrend_for_short:
        e200 = float(df["ema200"].iloc[end_idx - cfg.breakdown_lookback])
        c0 = float(df["close"].iloc[end_idx - cfg.breakdown_lookback])
        if not np.isfinite(e200) or (c0 / e200 - 1.0) < cfg.uptrend_min_pct / 100.0:
            return None
    return {"break_level": block_low}


def detect_pattern(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    if cfg.pattern_type in ("inverse_cnh", "classic_cnh"):
        return _detect_cnh(df, end_idx, cfg)
    if cfg.pattern_type == "double_top":
        return _detect_double_top(df, end_idx, cfg)
    if cfg.pattern_type == "engulfing_at_high":
        return _detect_engulfing_at_high(df, end_idx, cfg)
    if cfg.pattern_type == "breakdown_n":
        return _detect_breakdown_n(df, end_idx, cfg)
    return None


def find_all_patterns(df: pd.DataFrame, cfg: Config) -> list[int]:
    hits: list[int] = []
    start = max(cfg.cup_len + cfg.handle_len, cfg.breakdown_lookback + 1, 50)
    for i in range(start, len(df)):
        if detect_pattern(df, i, cfg) is not None:
            if hits and (i - hits[-1]) < 10:
                continue
            hits.append(i)
    return hits


# ============================================================
# SIMULATION
# ============================================================

def simulate_trades(df: pd.DataFrame, pattern_bars: list[int], cfg: Config,
                    window_label: str) -> list[dict]:
    trades: list[dict] = []
    direction = cfg.direction
    for pidx in pattern_bars:
        entry_idx: int | None = None
        broken_ema_name: str | None = None
        max_j = min(pidx + 1 + cfg.entry_max_bars_after_handle, len(df))

        # For pattern types where the pattern bar IS the trigger (breakdown_n, engulfing),
        # enter at next bar open
        if cfg.pattern_type in ("breakdown_n", "engulfing_at_high"):
            if pidx + 1 >= len(df):
                continue
            entry_idx = pidx + 1
            broken_ema_name = "ema24"  # placeholder for regime SL logic
        else:
            for j in range(pidx + 1, max_j):
                row = df.iloc[j]
                prev = df.iloc[j - 1]
                close = float(row["close"])
                prev_close = float(prev["close"])
                for ema_name in cfg.entry_emas:
                    ema_now = float(row[ema_name])
                    ema_prev = float(prev[ema_name])
                    if direction == "short" and prev_close >= ema_prev and close < ema_now:
                        broken_ema_name = ema_name
                        entry_idx = j
                        break
                    if direction == "long" and prev_close <= ema_prev and close > ema_now:
                        broken_ema_name = ema_name
                        entry_idx = j
                        break
                if entry_idx is not None:
                    break

        if entry_idx is None or broken_ema_name is None:
            continue
        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        atr_at_entry = float(entry_row["atr14"])
        if not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
            continue
        if direction == "short":
            hard_sl = entry_price + cfg.sl_atr_mult * atr_at_entry
        else:
            hard_sl = entry_price - cfg.sl_atr_mult * atr_at_entry

        regime_ema_name = None
        if cfg.regime_sl_mode == "broken_ema":
            regime_ema_name = broken_ema_name
        elif cfg.regime_sl_mode == "slower_ema":
            order = ["ema7", "ema24", "ema50", "ema100", "ema200"]
            try:
                idx = order.index(broken_ema_name)
                if idx + 1 < len(order):
                    regime_ema_name = order[idx + 1]
            except ValueError:
                pass
        elif cfg.regime_sl_mode == "ema200":
            regime_ema_name = "ema200"

        tp_candidates: list[tuple[str, float]] = []
        for nm in cfg.tp_emas:
            v = float(entry_row[nm])
            if direction == "short" and v < entry_price:
                tp_candidates.append((nm, v))
            elif direction == "long" and v > entry_price:
                tp_candidates.append((nm, v))
        if not tp_candidates:
            continue
        tp_candidates.sort(key=lambda kv: (-kv[1]) if direction == "short" else kv[1])
        tp_name, _ = tp_candidates[0]

        exit_idx, exit_price, exit_reason = None, None, ""
        for k in range(entry_idx + 1, len(df)):
            row = df.iloc[k]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            if regime_ema_name is not None:
                er = float(row[regime_ema_name])
                if direction == "short" and close > er:
                    exit_idx, exit_price = k, close
                    exit_reason = f"regime_sl_{regime_ema_name}"
                    break
                if direction == "long" and close < er:
                    exit_idx, exit_price = k, close
                    exit_reason = f"regime_sl_{regime_ema_name}"
                    break
            if direction == "short" and high >= hard_sl:
                exit_idx, exit_price, exit_reason = k, hard_sl, "atr_sl"
                break
            if direction == "long" and low <= hard_sl:
                exit_idx, exit_price, exit_reason = k, hard_sl, "atr_sl"
                break
            ema_tp = float(row[tp_name])
            if direction == "short" and low <= ema_tp:
                exit_idx, exit_price, exit_reason = k, ema_tp, f"tp_{tp_name}"
                break
            if direction == "long" and high >= ema_tp:
                exit_idx, exit_price, exit_reason = k, ema_tp, f"tp_{tp_name}"
                break
        if exit_idx is None:
            exit_idx = len(df) - 1
            exit_price = float(df.iloc[exit_idx]["close"])
            exit_reason = "eod"

        if direction == "short":
            gross = (entry_price - exit_price) / entry_price
        else:
            gross = (exit_price - entry_price) / entry_price
        net = gross - FRICTION_BPS / 10000.0
        trades.append({
            "window": window_label,
            "entry_ts": str(df.index[entry_idx]),
            "entry_price": entry_price,
            "exit_ts": str(df.index[exit_idx]),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "gross_pct": gross,
            "net_pct": net,
            "bars_held": exit_idx - entry_idx,
        })
    return trades


WINDOWS = [
    ("2020-H2", "2020-07-01", "2020-12-31"),
    ("2021-H1-bull", "2021-01-01", "2021-06-30"),
    ("2021-H2-bull", "2021-07-01", "2021-12-31"),
    ("2022-H1-bear", "2022-01-01", "2022-06-30"),
    ("2022-H2-bear", "2022-07-01", "2022-12-31"),
    ("2023-H1-rec", "2023-01-01", "2023-06-30"),
    ("2023-H2-rec", "2023-07-01", "2023-12-31"),
    ("2024-H1-bull", "2024-01-01", "2024-06-30"),
    ("2024-H2-bull", "2024-07-01", "2024-12-31"),
    ("2025-H1-mix", "2025-01-01", "2025-06-30"),
    ("2025-H2-mix", "2025-07-01", "2025-12-31"),
    ("2026-H1-bull", "2026-01-01", "2026-05-23"),
]


def run_config(cfg: Config) -> dict:
    df_full = load_tf(cfg.tf)
    all_trades: list[dict] = []
    per_window: list[dict] = []
    for label, start, end in WINDOWS:
        sub = df_full.loc[start:end]
        if len(sub) < 100:
            continue
        pats = find_all_patterns(sub, cfg)
        trades = simulate_trades(sub, pats, cfg, label)
        if trades:
            nets = np.array([t["net_pct"] for t in trades])
            per_window.append({
                "window": label,
                "trades": len(trades),
                "win_rate": float((nets > 0).mean()),
                "cum": float(np.prod(1.0 + nets) - 1.0),
                "sharpe": float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0,
            })
        all_trades.extend(trades)
    if not all_trades:
        return {"config": asdict(cfg), "trades": 0, "win_rate": 0, "cum": 0,
                "sharpe": 0, "per_window": []}
    nets = np.array([t["net_pct"] for t in all_trades])
    return {
        "config": asdict(cfg),
        "trades": len(all_trades),
        "win_rate": float((nets > 0).mean()),
        "cum": float(np.prod(1.0 + nets) - 1.0),
        "sharpe": float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0,
        "per_window": per_window,
        "sample_trades": all_trades[:5],  # first 5 for inspection
    }


# ============================================================
# CONFIGS TO RUN
# ============================================================

def build_configs() -> list[Config]:
    cfgs: list[Config] = []

    # BASELINE (best survivor S10)
    base = dict(pattern_type="inverse_cnh", direction="short", tf="4h",
                cup_len=20, handle_len=5, min_r2=0.70, min_cup_depth_atr=2.5,
                entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
                tp_emas=("ema100", "ema200"), require_uptrend_for_short=True,
                uptrend_min_pct=5.0)

    cfgs.append(Config(name="BASELINE_S10", **base, note="S10 reproduction"))

    # ATR multiplier axis
    for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        kwargs = dict(base); kwargs["sl_atr_mult"] = mult
        cfgs.append(Config(name=f"ATR_{mult}x", **kwargs,
                           note=f"ATR-SL = {mult}× ATR(14)"))

    # Cup length axis
    for cl in [10, 15, 20, 25, 30, 40]:
        kwargs = dict(base); kwargs["cup_len"] = cl
        cfgs.append(Config(name=f"CUP_{cl}b", **kwargs, note=f"Cup window = {cl} bars"))

    # R² axis
    for r2 in [0.50, 0.60, 0.70, 0.80, 0.90]:
        kwargs = dict(base); kwargs["min_r2"] = r2
        cfgs.append(Config(name=f"R2_{r2}", **kwargs, note=f"Parabola R² ≥ {r2}"))

    # Cup depth axis
    for d in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        kwargs = dict(base); kwargs["min_cup_depth_atr"] = d
        cfgs.append(Config(name=f"DEPTH_{d}atr", **kwargs,
                           note=f"Cup depth ≥ {d} × ATR"))

    # Entry EMA axis
    for emas in [("ema7",), ("ema24",), ("ema50",), ("ema7", "ema24"), ("ema24", "ema50")]:
        kwargs = dict(base); kwargs["entry_emas"] = emas
        nm = "_".join(emas).replace("ema", "e")
        cfgs.append(Config(name=f"ENT_{nm}", **kwargs,
                           note=f"Entry trigger: break {emas}"))

    # TP EMA axis
    for tps in [("ema50",), ("ema100",), ("ema200",), ("ema50", "ema100"), ("ema100", "ema200")]:
        kwargs = dict(base); kwargs["tp_emas"] = tps
        nm = "_".join(tps).replace("ema", "e")
        cfgs.append(Config(name=f"TP_{nm}", **kwargs, note=f"TP candidates: {tps}"))

    # Uptrend filter axis
    for pct in [0.0, 3.0, 5.0, 8.0, 12.0]:
        kwargs = dict(base); kwargs["uptrend_min_pct"] = pct
        cfgs.append(Config(name=f"UP_{pct}pct", **kwargs,
                           note=f"Require close ≥ EMA200 × (1 + {pct}%)"))

    # Handle length axis
    for hl in [3, 5, 7, 10]:
        kwargs = dict(base); kwargs["handle_len"] = hl
        cfgs.append(Config(name=f"HANDLE_{hl}b", **kwargs, note=f"Handle = {hl} bars"))

    # Regime SL modes
    for mode in ["off", "broken_ema", "slower_ema", "ema200"]:
        kwargs = dict(base); kwargs["regime_sl_mode"] = mode
        cfgs.append(Config(name=f"REGIME_{mode}", **kwargs,
                           note=f"Regime SL mode: {mode}"))

    # Different timeframes (only for the best config)
    for tf in ["1h", "4h"]:
        kwargs = dict(base); kwargs["tf"] = tf
        cfgs.append(Config(name=f"TF_{tf}", **kwargs, note=f"Best config on {tf}"))

    # ALTERNATIVE PATTERNS
    # Double top
    cfgs.append(Config(name="ALT_double_top_4h", pattern_type="double_top", direction="short",
                       tf="4h", entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
                       require_uptrend_for_short=True, uptrend_min_pct=5.0,
                       note="Double top (2 highs within 1.5%, gap ≥ 10 bars)"))
    cfgs.append(Config(name="ALT_double_top_1h", pattern_type="double_top", direction="short",
                       tf="1h", entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
                       require_uptrend_for_short=True, uptrend_min_pct=5.0,
                       note="Double top on 1h"))
    # Engulfing at high
    cfgs.append(Config(name="ALT_engulfing_4h", pattern_type="engulfing_at_high", direction="short",
                       tf="4h", entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
                       require_uptrend_for_short=True, uptrend_min_pct=5.0,
                       note="Bearish engulfing within 1% of 50-bar high"))
    cfgs.append(Config(name="ALT_engulfing_1h", pattern_type="engulfing_at_high", direction="short",
                       tf="1h", entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
                       require_uptrend_for_short=True, uptrend_min_pct=5.0,
                       note="Bearish engulfing 1h"))
    # N-bar breakdown control
    for n in [20, 50, 80]:
        cfgs.append(Config(name=f"ALT_breakdown_{n}_4h", pattern_type="breakdown_n",
                           direction="short", tf="4h", breakdown_lookback=n,
                           sl_atr_mult=1.5, regime_sl_mode="off",
                           require_uptrend_for_short=True, uptrend_min_pct=5.0,
                           note=f"Close < {n}-bar low (control — no pattern)"))

    # CLASSIC C&H LONG (focused sweep — this is the winner pattern)
    long_base = dict(base)
    long_base["direction"] = "long"
    long_base["pattern_type"] = "classic_cnh"
    long_base["require_uptrend_for_short"] = False
    cfgs.append(Config(name="LONG_baseline", **long_base,
                       note="Classic C&H long, S10-like params"))

    for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        kwargs = dict(long_base); kwargs["sl_atr_mult"] = mult
        cfgs.append(Config(name=f"LONG_ATR_{mult}x", **kwargs,
                           note=f"Long C&H, ATR={mult}×"))

    for r2 in [0.50, 0.60, 0.70, 0.80]:
        kwargs = dict(long_base); kwargs["min_r2"] = r2
        cfgs.append(Config(name=f"LONG_R2_{r2}", **kwargs,
                           note=f"Long, R²≥{r2}"))

    for cl in [15, 20, 25, 30]:
        kwargs = dict(long_base); kwargs["cup_len"] = cl
        cfgs.append(Config(name=f"LONG_CUP_{cl}b", **kwargs,
                           note=f"Long, cup={cl}b"))

    for emas in [("ema7",), ("ema24",), ("ema50",), ("ema7", "ema24")]:
        kwargs = dict(long_base); kwargs["entry_emas"] = emas
        nm = "_".join(emas).replace("ema", "e")
        cfgs.append(Config(name=f"LONG_ENT_{nm}", **kwargs,
                           note=f"Long, entry={emas}"))

    for tps in [("ema50",), ("ema100",), ("ema200",), ("ema50", "ema100")]:
        kwargs = dict(long_base); kwargs["tp_emas"] = tps
        nm = "_".join(tps).replace("ema", "e")
        cfgs.append(Config(name=f"LONG_TP_{nm}", **kwargs,
                           note=f"Long, TP={tps}"))

    for mode in ["off", "slower_ema", "ema200"]:
        kwargs = dict(long_base); kwargs["regime_sl_mode"] = mode
        cfgs.append(Config(name=f"LONG_REGIME_{mode}", **kwargs,
                           note=f"Long, regime SL = {mode}"))

    for d in [1.5, 2.0, 2.5, 3.0, 4.0]:
        kwargs = dict(long_base); kwargs["min_cup_depth_atr"] = d
        cfgs.append(Config(name=f"LONG_DEPTH_{d}atr", **kwargs,
                           note=f"Long, depth≥{d}×ATR"))

    # LONG on 1h
    for tf_ in ["1h"]:
        kwargs = dict(long_base); kwargs["tf"] = tf_
        cfgs.append(Config(name=f"LONG_TF_{tf_}", **kwargs,
                           note=f"Long on {tf_}"))

    # LONG with downtrend filter (buy bottoms only)
    for pct in [3.0, 5.0, 8.0]:
        kwargs = dict(long_base)
        kwargs["require_downtrend_for_long"] = True
        kwargs["uptrend_min_pct"] = pct
        cfgs.append(Config(name=f"LONG_DT_{pct}pct", **kwargs,
                           note=f"Long, require downtrend ≥ -{pct}% below EMA200"))

    # OPTIMAL LONG combinations (combining best params from individual axes)
    optimal_base = dict(long_base)
    optimal_base["sl_atr_mult"] = 2.0
    optimal_base["tp_emas"] = ("ema200",)
    optimal_base["entry_emas"] = ("ema24",)
    optimal_base["regime_sl_mode"] = "off"
    cfgs.append(Config(name="LONG_OPTIMAL", **optimal_base,
                       note="OPTIMAL: ATR=2.0, TP=EMA200, EMA24 entry, no regime SL"))

    opt_dt = dict(optimal_base)
    opt_dt["require_downtrend_for_long"] = True
    opt_dt["uptrend_min_pct"] = 3.0
    cfgs.append(Config(name="LONG_OPTIMAL_DT", **opt_dt,
                       note="OPTIMAL + downtrend filter (-3% from EMA200)"))

    opt_dt2 = dict(opt_dt); opt_dt2["uptrend_min_pct"] = 5.0
    cfgs.append(Config(name="LONG_OPTIMAL_DT5", **opt_dt2,
                       note="OPTIMAL + downtrend filter (-5% from EMA200)"))

    # OPTIMAL with R²=0.8 (stricter pattern)
    opt_strict = dict(opt_dt); opt_strict["min_r2"] = 0.80
    cfgs.append(Config(name="LONG_OPTIMAL_STRICT", **opt_strict,
                       note="OPTIMAL + DT + R²≥0.80"))

    return cfgs


def _run_one(cfg: Config) -> dict:
    return run_config(cfg)


def main() -> int:
    cfgs = build_configs()
    print(f"Running {len(cfgs)} configs in parallel...")
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_one, c): c for c in cfgs}
        done = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                cfg = futures[fut]
                r = {"config": asdict(cfg), "trades": 0, "win_rate": 0,
                     "cum": 0, "sharpe": 0, "per_window": [], "error": str(e)}
            results.append(r)
            done += 1
            cfg = futures[fut]
            print(f"  [{done}/{len(cfgs)}] {cfg.name:<30} "
                  f"trades={r['trades']:>4}  "
                  f"WR={r['win_rate']*100:>5.1f}%  "
                  f"cum={r['cum']*100:>+7.2f}%  "
                  f"sharpe={r['sharpe']:>+6.2f}")
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    RESULTS.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved → {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
