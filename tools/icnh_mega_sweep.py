"""MEGA sweep — much more permissive pattern detection + multi-TF.

Adds a new "distribution_top" pattern matching the user's actual visual rule
(uptrend → chop → breakdown, no parabola requirement). Loosens ICnH params
dramatically. Tests 15m, 1h, 4h.

Target: 2-4 signals/month frequency (= 24-48/year).

Usage: uv run python tools/icnh_mega_sweep.py
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
RESULTS = ROOT / "data" / "mega_sweep_results.json"
FRICTION_BPS = 10.0
ATR_LEN = 14


@dataclass
class Config:
    name: str
    pattern_type: str = "distribution_top"   # distribution_top | inverse_cnh | classic_cnh | accumulation_bottom
    direction: str = "short"
    tf: str = "4h"
    # Distribution-top / accumulation-bottom params
    uptrend_bars: int = 8
    chop_bars: int = 6
    min_rise_pct: float = 2.5      # min rise during uptrend phase (% of start price)
    max_chop_ratio: float = 0.55   # chop range / uptrend range max
    require_chop_at_top: bool = True
    breakdown_mode: str = "chop_low"  # chop_low | ema24 | either | chop_low_or_ema24
    # ICnH params (looser defaults than before)
    cup_len: int = 20
    handle_len: int = 5
    peak_tolerance: int = 5         # was 3 — looser
    min_cup_depth_atr: float = 1.0  # was 2.5
    min_r2: float = 0.45            # was 0.70
    handle_max_depth_frac: float = 0.60
    entry_max_bars_after_handle: int = 8
    # Entry trigger
    entry_emas: tuple = ("ema24",)
    # SL
    sl_atr_mult: float = 1.5
    regime_sl_mode: str = "off"
    # TP
    tp_emas: tuple = ("ema100", "ema200")
    # Filters
    require_uptrend_for_short: bool = False
    uptrend_min_pct: float = 5.0
    require_downtrend_for_long: bool = False
    # Dedup
    dedup_bars: int = 10
    note: str = ""


def load_tf(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"BTC_USDT_USDT_{tf}.parquet").sort_index()
    df["ema7"] = ema(df["close"], 7)
    df["ema24"] = ema(df["close"], 24)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 200)
    df["atr14"] = atr(df["high"], df["low"], df["close"], ATR_LEN)
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


def _detect_distribution_top(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    """Loose: uptrend phase → sideways chop → breakdown at end_idx.
    Matches the user's image-16 visual rule.

    NOTE: chop slice EXCLUDES current bar — the current bar is the breakdown
    trigger, so chop_low must be measured from prior bars only. Otherwise
    close-below-chop-low can never trigger (chop_low keeps tracking current).
    """
    total = cfg.uptrend_bars + cfg.chop_bars + 1   # +1 for the breakdown bar
    start = end_idx - total + 1
    if start < 0:
        return None
    # Uptrend phase
    up = df.iloc[start : start + cfg.uptrend_bars]
    chop = df.iloc[start + cfg.uptrend_bars : end_idx]   # EXCLUDES current breakdown bar
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
    # Chop phase
    chop_range = float(chop["high"].max() - chop["low"].min())
    chop_high = float(chop["high"].max())
    chop_low = float(chop["low"].min())
    if chop_range > up_range * cfg.max_chop_ratio:
        return None
    # Optional: chop must be at the TOP of the uptrend range (not pulling back lower)
    if cfg.require_chop_at_top:
        chop_mid = (chop_high + chop_low) / 2
        up_top = float(up["high"].max())
        up_bot = float(up["low"].min())
        if chop_mid < (up_top + up_bot) / 2:
            return None
    # Breakdown trigger
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
    return {"chop_low": chop_low, "chop_high": chop_high, "up_pct": up_pct,
            "trigger": trigger_kind}


def _detect_accumulation_bottom(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    """Mirror of distribution-top: downtrend → chop → breakout up. (LONG side.)"""
    total = cfg.uptrend_bars + cfg.chop_bars + 1
    start = end_idx - total + 1
    if start < 0:
        return None
    down = df.iloc[start : start + cfg.uptrend_bars]
    chop = df.iloc[start + cfg.uptrend_bars : end_idx]
    if len(down) < 3 or len(chop) < 3:
        return None
    d_start = float(down["close"].iloc[0])
    d_end = float(down["close"].iloc[-1])
    drop_pct = (d_start - d_end) / d_start * 100
    if drop_pct < cfg.min_rise_pct:
        return None
    down_range = float(down["high"].max() - down["low"].min())
    if down_range <= 0:
        return None
    chop_range = float(chop["high"].max() - chop["low"].min())
    chop_high = float(chop["high"].max())
    chop_low = float(chop["low"].min())
    if chop_range > down_range * cfg.max_chop_ratio:
        return None
    if cfg.require_chop_at_top:
        chop_mid = (chop_high + chop_low) / 2
        d_top = float(down["high"].max())
        d_bot = float(down["low"].min())
        if chop_mid > (d_top + d_bot) / 2:
            return None
    bar_close = float(df["close"].iloc[end_idx])
    bar_open = float(df["open"].iloc[end_idx])
    ema24 = float(df["ema24"].iloc[end_idx])
    triggered = False
    if cfg.breakdown_mode in ("chop_low", "either", "chop_low_or_ema24"):
        if bar_close > chop_high and bar_open <= chop_high:
            triggered = True
    if not triggered and cfg.breakdown_mode in ("ema24", "either", "chop_low_or_ema24"):
        prev_close = float(df["close"].iloc[end_idx - 1])
        prev_ema24 = float(df["ema24"].iloc[end_idx - 1])
        if prev_close <= prev_ema24 and bar_close > ema24:
            triggered = True
    if not triggered:
        return None
    return {"chop_low": chop_low, "chop_high": chop_high, "drop_pct": drop_pct}


def _detect_cnh(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
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
    return {"r2": r2, "peak": peak_price}


def detect_pattern(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    if cfg.pattern_type == "distribution_top":
        return _detect_distribution_top(df, end_idx, cfg)
    if cfg.pattern_type == "accumulation_bottom":
        return _detect_accumulation_bottom(df, end_idx, cfg)
    if cfg.pattern_type in ("inverse_cnh", "classic_cnh"):
        return _detect_cnh(df, end_idx, cfg)
    return None


def find_all_patterns(df: pd.DataFrame, cfg: Config) -> list[int]:
    hits: list[int] = []
    if cfg.pattern_type in ("distribution_top", "accumulation_bottom"):
        start = cfg.uptrend_bars + cfg.chop_bars + 1
    else:
        start = cfg.cup_len + cfg.handle_len
    start = max(start, 200)  # need EMA(200) warmup
    for i in range(start, len(df)):
        if detect_pattern(df, i, cfg) is not None:
            if hits and (i - hits[-1]) < cfg.dedup_bars:
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
        # For distribution_top / accumulation_bottom, the pattern bar IS the trigger
        if cfg.pattern_type in ("distribution_top", "accumulation_bottom"):
            entry_idx = pidx  # enter at close of the trigger bar
        else:
            # ICnH: wait for EMA breakdown after pattern
            entry_idx = None
            for j in range(pidx + 1, min(pidx + 1 + cfg.entry_max_bars_after_handle, len(df))):
                row = df.iloc[j]
                prev = df.iloc[j - 1]
                close = float(row["close"])
                prev_close = float(prev["close"])
                for ema_name in cfg.entry_emas:
                    ema_now = float(row[ema_name])
                    ema_prev = float(prev[ema_name])
                    if direction == "short" and prev_close >= ema_prev and close < ema_now:
                        entry_idx = j
                        break
                    if direction == "long" and prev_close <= ema_prev and close > ema_now:
                        entry_idx = j
                        break
                if entry_idx is not None:
                    break
        if entry_idx is None or entry_idx >= len(df):
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
                "sharpe": 0, "per_window": [], "trades_per_year": 0}
    nets = np.array([t["net_pct"] for t in all_trades])
    n_years = 5.9  # approximate
    return {
        "config": asdict(cfg),
        "trades": len(all_trades),
        "trades_per_year": len(all_trades) / n_years,
        "trades_per_month": len(all_trades) / (n_years * 12),
        "win_rate": float((nets > 0).mean()),
        "cum": float(np.prod(1.0 + nets) - 1.0),
        "sharpe": float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0,
        "per_window": per_window,
    }


# ============================================================
# CONFIGS
# ============================================================

def build_configs() -> list[Config]:
    cfgs: list[Config] = []

    # =====================================================
    # DISTRIBUTION TOP (SHORT) — new loose pattern matching user's image
    # =====================================================
    dt_base = dict(
        pattern_type="distribution_top",
        direction="short",
        sl_atr_mult=1.5,
        regime_sl_mode="off",
        tp_emas=("ema100",),
        require_uptrend_for_short=False,
    )

    # TF axis on baseline
    for tf in ["4h", "1h", "15m"]:
        c = dict(dt_base); c["tf"] = tf
        cfgs.append(Config(name=f"DT_baseline_{tf}", **c, note=f"DistTop baseline on {tf}"))

    # Uptrend bars
    for ub in [4, 6, 8, 12, 16]:
        c = dict(dt_base); c["tf"] = "4h"; c["uptrend_bars"] = ub
        cfgs.append(Config(name=f"DT_up{ub}b_4h", **c, note=f"DistTop uptrend={ub}b"))

    # Chop bars
    for cb in [3, 5, 8, 12, 16]:
        c = dict(dt_base); c["tf"] = "4h"; c["chop_bars"] = cb
        cfgs.append(Config(name=f"DT_chop{cb}b_4h", **c, note=f"DistTop chop={cb}b"))

    # Min rise percentage
    for r in [1.0, 2.0, 3.0, 5.0, 8.0]:
        c = dict(dt_base); c["tf"] = "4h"; c["min_rise_pct"] = r
        cfgs.append(Config(name=f"DT_rise{r}pct_4h", **c, note=f"DistTop min rise={r}%"))

    # Max chop ratio (tighter = stricter chop)
    for mc in [0.30, 0.45, 0.55, 0.70, 0.90]:
        c = dict(dt_base); c["tf"] = "4h"; c["max_chop_ratio"] = mc
        cfgs.append(Config(name=f"DT_chopR{mc}_4h", **c, note=f"DistTop chop ratio≤{mc}"))

    # ATR multiplier
    for atr_m in [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        c = dict(dt_base); c["tf"] = "4h"; c["sl_atr_mult"] = atr_m
        cfgs.append(Config(name=f"DT_atr{atr_m}_4h", **c, note=f"DistTop ATR-SL={atr_m}×"))

    # TP target
    for tps in [("ema50",), ("ema100",), ("ema200",), ("ema50", "ema100"), ("ema100", "ema200")]:
        c = dict(dt_base); c["tf"] = "4h"; c["tp_emas"] = tps
        nm = "_".join(tps).replace("ema", "e")
        cfgs.append(Config(name=f"DT_TP_{nm}_4h", **c, note=f"DistTop TP={tps}"))

    # Breakdown mode
    for bm in ["chop_low", "ema24", "either", "chop_low_or_ema24"]:
        c = dict(dt_base); c["tf"] = "4h"; c["breakdown_mode"] = bm
        cfgs.append(Config(name=f"DT_brk_{bm}_4h", **c, note=f"DistTop trigger={bm}"))

    # 1h-specific tuning
    for ub in [6, 10, 16, 24]:
        c = dict(dt_base); c["tf"] = "1h"; c["uptrend_bars"] = ub; c["chop_bars"] = 8
        cfgs.append(Config(name=f"DT_1h_up{ub}", **c, note=f"DistTop 1h, up={ub}b"))

    # Best-combo guesses
    cfgs.append(Config(name="DT_optimal_4h", pattern_type="distribution_top", direction="short",
                       tf="4h", uptrend_bars=8, chop_bars=6, min_rise_pct=3.0,
                       max_chop_ratio=0.55, sl_atr_mult=1.5, regime_sl_mode="off",
                       tp_emas=("ema100",), note="DT 4h optimal guess"))
    cfgs.append(Config(name="DT_optimal_1h", pattern_type="distribution_top", direction="short",
                       tf="1h", uptrend_bars=12, chop_bars=8, min_rise_pct=2.5,
                       max_chop_ratio=0.55, sl_atr_mult=1.5, regime_sl_mode="off",
                       tp_emas=("ema100",), note="DT 1h optimal guess"))
    cfgs.append(Config(name="DT_optimal_15m", pattern_type="distribution_top", direction="short",
                       tf="15m", uptrend_bars=16, chop_bars=10, min_rise_pct=1.5,
                       max_chop_ratio=0.55, sl_atr_mult=1.5, regime_sl_mode="off",
                       tp_emas=("ema100",), note="DT 15m optimal guess"))

    # WIDE-OPEN: very loose to maximize signals
    cfgs.append(Config(name="DT_loose_4h", pattern_type="distribution_top", direction="short",
                       tf="4h", uptrend_bars=6, chop_bars=4, min_rise_pct=1.5,
                       max_chop_ratio=0.80, require_chop_at_top=False, sl_atr_mult=1.5,
                       regime_sl_mode="off", tp_emas=("ema50",),
                       note="DT very loose"))
    cfgs.append(Config(name="DT_loose_1h", pattern_type="distribution_top", direction="short",
                       tf="1h", uptrend_bars=10, chop_bars=6, min_rise_pct=1.0,
                       max_chop_ratio=0.80, require_chop_at_top=False, sl_atr_mult=1.5,
                       regime_sl_mode="off", tp_emas=("ema50",),
                       note="DT very loose 1h"))

    # =====================================================
    # ACCUMULATION BOTTOM (LONG) — mirror
    # =====================================================
    ab_base = dict(dt_base); ab_base["direction"] = "long"
    ab_base["pattern_type"] = "accumulation_bottom"
    ab_base["require_uptrend_for_short"] = False

    for tf in ["4h", "1h", "15m"]:
        c = dict(ab_base); c["tf"] = tf
        cfgs.append(Config(name=f"AB_baseline_{tf}", **c, note=f"AccBot baseline {tf}"))

    cfgs.append(Config(name="AB_optimal_4h", pattern_type="accumulation_bottom", direction="long",
                       tf="4h", uptrend_bars=8, chop_bars=6, min_rise_pct=3.0,
                       max_chop_ratio=0.55, sl_atr_mult=2.0, regime_sl_mode="off",
                       tp_emas=("ema200",),
                       note="AccBot 4h optimal guess"))
    cfgs.append(Config(name="AB_optimal_1h", pattern_type="accumulation_bottom", direction="long",
                       tf="1h", uptrend_bars=12, chop_bars=8, min_rise_pct=2.5,
                       max_chop_ratio=0.55, sl_atr_mult=2.0, regime_sl_mode="off",
                       tp_emas=("ema200",),
                       note="AccBot 1h optimal guess"))

    # =====================================================
    # LOOSE ICnH (much more permissive than before)
    # =====================================================
    loose_icnh = dict(
        pattern_type="inverse_cnh", direction="short", tf="4h",
        cup_len=15, handle_len=4, peak_tolerance=5,
        min_cup_depth_atr=1.0, min_r2=0.40, handle_max_depth_frac=0.70,
        entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
        tp_emas=("ema100",), require_uptrend_for_short=False,
    )

    for r2 in [0.20, 0.30, 0.40, 0.50, 0.60]:
        c = dict(loose_icnh); c["min_r2"] = r2
        cfgs.append(Config(name=f"ICNH_loose_R2_{r2}", **c, note=f"Loose ICnH R²≥{r2}"))

    for cl in [8, 12, 15, 20]:
        c = dict(loose_icnh); c["cup_len"] = cl
        cfgs.append(Config(name=f"ICNH_loose_cup{cl}", **c, note=f"Loose ICnH cup={cl}"))

    for d in [0.5, 1.0, 1.5, 2.0]:
        c = dict(loose_icnh); c["min_cup_depth_atr"] = d
        cfgs.append(Config(name=f"ICNH_loose_depth{d}", **c, note=f"Loose ICnH depth≥{d}×ATR"))

    for tf in ["4h", "1h", "15m"]:
        c = dict(loose_icnh); c["tf"] = tf
        cfgs.append(Config(name=f"ICNH_loose_{tf}", **c, note=f"Loose ICnH {tf}"))

    # Multi-TF loose ICnH for LONG too
    long_loose = dict(loose_icnh)
    long_loose["pattern_type"] = "classic_cnh"
    long_loose["direction"] = "long"
    long_loose["tp_emas"] = ("ema200",)
    long_loose["sl_atr_mult"] = 2.0
    for tf in ["4h", "1h", "15m"]:
        c = dict(long_loose); c["tf"] = tf
        cfgs.append(Config(name=f"LONG_loose_{tf}", **c, note=f"Loose classic C&H LONG {tf}"))

    # =====================================================
    # PUSH FREQUENCY: tighter dedup + tighter parameters
    # =====================================================
    # Best loose-ICnH winner was cup20 + R²=0.5. Vary dedup + ATR + TP.
    push_base = dict(loose_icnh)
    push_base["cup_len"] = 20
    push_base["min_r2"] = 0.50
    for dedup in [3, 5, 10, 15]:
        c = dict(push_base); c["dedup_bars"] = dedup
        cfgs.append(Config(name=f"PUSH_dedup{dedup}_4h", **c, note=f"Loose ICnH cup20 R²0.5 dedup={dedup}b"))

    for atr_m in [0.8, 1.0, 1.5, 2.0, 2.5]:
        c = dict(push_base); c["sl_atr_mult"] = atr_m; c["dedup_bars"] = 5
        cfgs.append(Config(name=f"PUSH_atr{atr_m}_4h", **c, note=f"Loose ICnH ATR={atr_m} dedup=5"))

    for tps in [("ema50",), ("ema100",), ("ema200",), ("ema50", "ema100"), ("ema100", "ema200")]:
        c = dict(push_base); c["tp_emas"] = tps; c["dedup_bars"] = 5
        nm = "_".join(tps).replace("ema", "e")
        cfgs.append(Config(name=f"PUSH_TP_{nm}_4h", **c, note=f"Loose ICnH TP={tps}"))

    # Different entry triggers on the loose ICnH
    for emas in [("ema7",), ("ema24",), ("ema50",), ("ema7", "ema24"),
                 ("ema24", "ema50")]:
        c = dict(push_base); c["entry_emas"] = emas; c["dedup_bars"] = 5
        nm = "_".join(emas).replace("ema", "e")
        cfgs.append(Config(name=f"PUSH_ENT_{nm}_4h", **c, note=f"Loose ICnH entry={emas}"))

    # Combined "MAXIMUM SIGNAL" config
    cfgs.append(Config(name="PUSH_MAX_4h", pattern_type="inverse_cnh", direction="short",
                       tf="4h", cup_len=20, handle_len=4, min_r2=0.50,
                       min_cup_depth_atr=1.0, handle_max_depth_frac=0.70,
                       peak_tolerance=6,
                       entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
                       tp_emas=("ema100",), dedup_bars=3,
                       note="MAX-frequency loose ICnH 4h"))

    # LONG counterpart
    cfgs.append(Config(name="PUSH_MAX_LONG_4h", pattern_type="classic_cnh", direction="long",
                       tf="4h", cup_len=20, handle_len=4, min_r2=0.50,
                       min_cup_depth_atr=1.0, handle_max_depth_frac=0.70,
                       peak_tolerance=6,
                       entry_emas=("ema24",), sl_atr_mult=2.0, regime_sl_mode="off",
                       tp_emas=("ema200",), dedup_bars=3,
                       note="MAX-frequency loose LONG C&H 4h"))

    return cfgs


def _run_one(cfg: Config) -> dict:
    try:
        return run_config(cfg)
    except Exception as e:
        return {"config": asdict(cfg), "trades": 0, "trades_per_year": 0,
                "trades_per_month": 0,
                "win_rate": 0, "cum": 0, "sharpe": 0,
                "per_window": [], "error": str(e)}


def main() -> int:
    cfgs = build_configs()
    print(f"Running {len(cfgs)} configs in parallel...")
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_one, c): c for c in cfgs}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            cfg = futures[fut]
            tpm = r.get("trades_per_month", 0)
            print(f"  [{done}/{len(cfgs)}] {cfg.name:<28} "
                  f"tr={r['trades']:>4} ({tpm:.1f}/mo)  "
                  f"WR={r['win_rate']*100:>5.1f}%  "
                  f"cum={r['cum']*100:>+7.2f}%  "
                  f"sh={r['sharpe']:>+5.2f}")
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    RESULTS.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved → {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
