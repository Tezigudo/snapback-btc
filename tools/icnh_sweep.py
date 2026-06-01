"""Multi-variant sweep for Inverse Cup-and-Handle (short) and classic
Cup-and-Handle (long). Tests across multiple BTC regime windows so we don't
get fooled by a single bull-only period.

Usage:
    uv run python tools/icnh_sweep.py            # run all, save results to JSON
    uv run python tools/icnh_sweep.py --build    # build HTML report from JSON

Designed to be called by the HTML-report builder.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strategy.indicators import atr, ema  # noqa: E402

DATA = ROOT / "data" / "historical"
RESULTS = ROOT / "data" / "icnh_sweep_results.json"


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    name: str
    direction: str = "short"          # "short" = inverse C&H; "long" = classic C&H
    cup_len: int = 20
    handle_len: int = 5
    peak_tolerance: int = 3
    min_cup_depth_atr: float = 1.5
    min_r2: float = 0.55
    handle_max_depth_frac: float = 0.45
    entry_max_bars_after_handle: int = 8
    # Entry trigger: list of EMA names to break for entry
    entry_emas: tuple = ("ema7", "ema24")
    # SL
    sl_atr_mult: float = 1.5
    regime_sl_mode: str = "slower_ema"  # "off" | "broken_ema" | "slower_ema" | "ema200"
    # TP
    tp_emas: tuple = ("ema100", "ema200")
    # Filters
    require_uptrend_for_short: bool = False  # close > ema200 by X% before pattern
    uptrend_min_pct: float = 5.0
    require_downtrend_for_long: bool = False
    # Dual-TF (only meaningful for the dual_tf runner)
    note: str = ""


@dataclass
class WindowResult:
    window: str          # "2024-H1" etc
    tf: str              # "4h" or "1h"
    trades: int
    win_rate: float
    mean_pct: float
    cum_return: float
    sharpe_rough: float
    avg_bars: float
    exit_reasons: dict


@dataclass
class SweepRow:
    config: Config
    by_window: list[WindowResult] = field(default_factory=list)
    pooled_cum_return: float = 0.0
    pooled_win_rate: float = 0.0
    pooled_trades: int = 0
    pooled_sharpe: float = 0.0


# ============================================================
# DATA + INDICATORS
# ============================================================

ATR_LEN = 14

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
# PATTERN DETECTION
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


def detect_pattern(df: pd.DataFrame, end_idx: int, cfg: Config) -> dict | None:
    """Detect inverse C&H (concave-down dome) for shorts,
    or classic C&H (concave-up bowl) for longs."""
    handle_start = end_idx - cfg.handle_len + 1
    handle_end = end_idx
    cup_end = handle_start - 1
    cup_start = cup_end - cfg.cup_len + 1
    if cup_start < 0 or handle_end >= len(df):
        return None

    cup = df.iloc[cup_start : cup_end + 1]
    handle = df.iloc[handle_start : handle_end + 1]

    closes = cup["close"].to_numpy()
    if np.any(np.isnan(closes)):
        return None

    a, r2 = _fit_parabola_r2(closes)
    if cfg.direction == "short":
        if a >= 0:
            return None  # need concave-down (inverted bowl)
    else:
        if a <= 0:
            return None  # need concave-up (bowl)

    if r2 < cfg.min_r2:
        return None

    if cfg.direction == "short":
        peak_pos = int(cup["high"].values.argmax())
    else:
        peak_pos = int(cup["low"].values.argmin())
    center = cfg.cup_len // 2
    if abs(peak_pos - center) > cfg.peak_tolerance:
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
        peak_price = float(cup["low"].iloc[peak_pos])  # actually the trough
        left_lip = float(cup["high"].iloc[:peak_pos + 1].max())
        right_lip = float(cup["high"].iloc[peak_pos:].max())
        base = max(left_lip, right_lip)
        cup_depth = base - peak_price

    if cup_depth < cfg.min_cup_depth_atr * atr_val:
        return None

    # Handle requirements
    if cfg.direction == "short":
        cup_midpoint = (peak_price + base) / 2.0
        if handle["low"].min() < cup_midpoint:
            return None
    else:
        cup_midpoint = (peak_price + base) / 2.0
        if handle["high"].max() > cup_midpoint:
            return None

    handle_range = float(handle["high"].max() - handle["low"].min())
    if handle_range > cfg.handle_max_depth_frac * cup_depth:
        return None

    # Optional trend filter
    if cfg.direction == "short" and cfg.require_uptrend_for_short:
        ema200_at_cup_start = float(df["ema200"].iloc[cup_start])
        close_at_cup_start = float(df["close"].iloc[cup_start])
        if not np.isfinite(ema200_at_cup_start):
            return None
        if (close_at_cup_start / ema200_at_cup_start - 1.0) < cfg.uptrend_min_pct / 100.0:
            return None
    if cfg.direction == "long" and cfg.require_downtrend_for_long:
        ema200_at_cup_start = float(df["ema200"].iloc[cup_start])
        close_at_cup_start = float(df["close"].iloc[cup_start])
        if not np.isfinite(ema200_at_cup_start):
            return None
        if (close_at_cup_start / ema200_at_cup_start - 1.0) > -cfg.uptrend_min_pct / 100.0:
            return None

    return {
        "peak_price": peak_price,
        "cup_depth": cup_depth,
        "handle_low": float(handle["low"].min()),
        "handle_high": float(handle["high"].max()),
        "atr_at_handle": atr_val,
        "r2": r2,
    }


def find_all_patterns(df: pd.DataFrame, cfg: Config) -> list[int]:
    hits: list[int] = []
    start = cfg.cup_len + cfg.handle_len
    for i in range(start, len(df)):
        if detect_pattern(df, i, cfg) is not None:
            if hits and (i - hits[-1]) < 10:
                continue
            hits.append(i)
    return hits


# ============================================================
# SIMULATION
# ============================================================

FRICTION_BPS = 10.0  # round-trip


def simulate_trades(df: pd.DataFrame, pattern_bars: list[int], cfg: Config,
                    tf_label: str, window_label: str) -> list[dict]:
    trades: list[dict] = []
    direction = cfg.direction

    for pidx in pattern_bars:
        # Entry trigger
        entry_idx: int | None = None
        broken_ema_name: str | None = None

        max_j = min(pidx + 1 + cfg.entry_max_bars_after_handle, len(df))
        for j in range(pidx + 1, max_j):
            row = df.iloc[j]
            prev = df.iloc[j - 1]
            close = float(row["close"])
            prev_close = float(prev["close"])

            triggered = False
            for ema_name in cfg.entry_emas:
                ema_now = float(row[ema_name])
                ema_prev = float(prev[ema_name])
                if direction == "short":
                    # Short trigger: was above EMA, now below
                    if prev_close >= ema_prev and close < ema_now:
                        broken_ema_name = ema_name
                        triggered = True
                        break
                else:
                    # Long trigger: was below EMA, now above
                    if prev_close <= ema_prev and close > ema_now:
                        broken_ema_name = ema_name
                        triggered = True
                        break
            if triggered:
                entry_idx = j
                break

        if entry_idx is None or broken_ema_name is None:
            continue

        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        atr_at_entry = float(entry_row["atr14"])
        if not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
            continue

        # Hard SL
        if direction == "short":
            hard_sl = entry_price + cfg.sl_atr_mult * atr_at_entry
        else:
            hard_sl = entry_price - cfg.sl_atr_mult * atr_at_entry

        # Regime SL EMA selection
        regime_ema_name = None
        if cfg.regime_sl_mode == "broken_ema":
            regime_ema_name = broken_ema_name
        elif cfg.regime_sl_mode == "slower_ema":
            order = ["ema7", "ema24", "ema50", "ema100", "ema200"]
            idx = order.index(broken_ema_name)
            if idx + 1 < len(order):
                regime_ema_name = order[idx + 1]
        elif cfg.regime_sl_mode == "ema200":
            regime_ema_name = "ema200"
        # "off" → regime_ema_name stays None

        # TP target = first EMA in cfg.tp_emas that sits beyond entry
        # (below entry for shorts, above for longs)
        tp_candidates: list[tuple[str, float]] = []
        for nm in cfg.tp_emas:
            v = float(entry_row[nm])
            if direction == "short" and v < entry_price:
                tp_candidates.append((nm, v))
            elif direction == "long" and v > entry_price:
                tp_candidates.append((nm, v))
        if not tp_candidates:
            continue
        # nearest first
        tp_candidates.sort(key=lambda kv: (-kv[1]) if direction == "short" else kv[1])
        tp_name, _ = tp_candidates[0]

        # Walk forward
        exit_idx: int | None = None
        exit_price: float | None = None
        exit_reason: str = ""

        for k in range(entry_idx + 1, len(df)):
            row = df.iloc[k]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            if regime_ema_name is not None:
                ema_regime = float(row[regime_ema_name])
                if direction == "short" and close > ema_regime:
                    exit_idx, exit_price = k, close
                    exit_reason = f"regime_sl_{regime_ema_name}_reclaim"
                    break
                if direction == "long" and close < ema_regime:
                    exit_idx, exit_price = k, close
                    exit_reason = f"regime_sl_{regime_ema_name}_lost"
                    break

            if direction == "short" and high >= hard_sl:
                exit_idx, exit_price = k, hard_sl
                exit_reason = "atr_sl"
                break
            if direction == "long" and low <= hard_sl:
                exit_idx, exit_price = k, hard_sl
                exit_reason = "atr_sl"
                break

            ema_tp = float(row[tp_name])
            if direction == "short" and low <= ema_tp:
                exit_idx, exit_price = k, ema_tp
                exit_reason = f"tp_{tp_name}"
                break
            if direction == "long" and high >= ema_tp:
                exit_idx, exit_price = k, ema_tp
                exit_reason = f"tp_{tp_name}"
                break

        if exit_idx is None:
            exit_idx = len(df) - 1
            exit_price = float(df.iloc[exit_idx]["close"])
            exit_reason = "eod"

        if direction == "short":
            gross_pct = (entry_price - exit_price) / entry_price
        else:
            gross_pct = (exit_price - entry_price) / entry_price
        net_pct = gross_pct - FRICTION_BPS / 10000.0

        trades.append({
            "window": window_label,
            "tf": tf_label,
            "direction": direction,
            "entry_ts": str(df.index[entry_idx]),
            "entry_price": entry_price,
            "broken_ema": broken_ema_name,
            "atr_sl": hard_sl,
            "tp_target": tp_name,
            "exit_ts": str(df.index[exit_idx]),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "gross_pct": gross_pct,
            "net_pct": net_pct,
            "bars_held": exit_idx - entry_idx,
        })

    return trades


def stats_from_trades(trades: list[dict], tf: str, window: str) -> WindowResult:
    if not trades:
        return WindowResult(window=window, tf=tf, trades=0, win_rate=0,
                            mean_pct=0, cum_return=0, sharpe_rough=0,
                            avg_bars=0, exit_reasons={})
    nets = np.array([t["net_pct"] for t in trades])
    bars = np.array([t["bars_held"] for t in trades])
    cum = float(np.prod(1.0 + nets) - 1.0)
    sharpe = float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0
    exit_reasons: dict = {}
    for t in trades:
        exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1
    return WindowResult(
        window=window, tf=tf, trades=len(trades),
        win_rate=float((nets > 0).mean()),
        mean_pct=float(nets.mean()),
        cum_return=cum,
        sharpe_rough=sharpe,
        avg_bars=float(bars.mean()),
        exit_reasons=exit_reasons,
    )


# ============================================================
# RUNNER
# ============================================================

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


def run_config(cfg: Config) -> SweepRow:
    df_4h = load_tf("4h")
    df_1h = load_tf("1h")
    row = SweepRow(config=cfg)
    all_nets: list[float] = []
    all_trades = 0
    wins = 0

    for label, start, end in WINDOWS:
        sub4 = df_4h.loc[start:end]
        sub1 = df_1h.loc[start:end]
        if len(sub4) < 50 or len(sub1) < 50:
            continue
        pat4 = find_all_patterns(sub4, cfg)
        pat1 = find_all_patterns(sub1, cfg)
        t4 = simulate_trades(sub4, pat4, cfg, "4h", label)
        t1 = simulate_trades(sub1, pat1, cfg, "1h", label)
        # Combined: dedup near-time trades (within 12h)
        # Apply dual-TF rule
        combined = _dual_tf(t4, t1)
        if combined:
            res = stats_from_trades(combined, "combined", label)
            row.by_window.append(res)
            for t in combined:
                all_nets.append(t["net_pct"])
                all_trades += 1
                if t["net_pct"] > 0:
                    wins += 1

    if all_trades > 0:
        nets = np.array(all_nets)
        row.pooled_cum_return = float(np.prod(1.0 + nets) - 1.0)
        row.pooled_win_rate = wins / all_trades
        row.pooled_trades = all_trades
        row.pooled_sharpe = float(nets.mean() / nets.std() * np.sqrt(250)) if nets.std() > 0 else 0.0
    return row


def _dual_tf(trades_4h: list[dict], trades_1h: list[dict]) -> list[dict]:
    """4h confirmed + 1h-alone (no 4h within ±12h)"""
    kept: list[dict] = []
    for t in trades_4h:
        kept.append(dict(t, src="4h"))
    for t in trades_1h:
        t_ts = pd.Timestamp(t["entry_ts"])
        nearby = any(abs((pd.Timestamp(t4["entry_ts"]) - t_ts).total_seconds()) <= 12 * 3600
                     for t4 in trades_4h)
        if not nearby:
            kept.append(dict(t, src="1h_alone"))
    kept.sort(key=lambda t: t["entry_ts"])
    return kept


# ============================================================
# CONFIGS TO SWEEP
# ============================================================

CONFIGS: list[Config] = [
    # ---- Best survivor (rerun first for HTML stability) ----
    Config(name="S8_combo", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           entry_emas=("ema24",), min_r2=0.70, min_cup_depth_atr=2.5,
           require_uptrend_for_short=True, uptrend_min_pct=5.0,
           note="Combine: strict pattern + EMA24-only + uptrend filter + wide SL"),

    # ---- Tighter / looser variations on S8 ----
    Config(name="S9_extreme_strict", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           entry_emas=("ema24",), min_r2=0.80, min_cup_depth_atr=3.5,
           require_uptrend_for_short=True, uptrend_min_pct=5.0,
           note="EXTREME pattern: R²>0.80, depth ≥ 3.5 ATR"),
    Config(name="S10_tighter_atr", direction="short", regime_sl_mode="off", sl_atr_mult=1.5,
           entry_emas=("ema24",), min_r2=0.70, min_cup_depth_atr=2.5,
           require_uptrend_for_short=True, uptrend_min_pct=5.0,
           note="S8 but tighter ATR-SL=1.5"),
    Config(name="S11_strong_uptrend", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           entry_emas=("ema24",), min_r2=0.70, min_cup_depth_atr=2.5,
           require_uptrend_for_short=True, uptrend_min_pct=10.0,
           note="S8 but require STRONGER uptrend (10% above EMA200)"),

    # ---- Original short variants ----
    Config(name="S1_baseline", direction="short", regime_sl_mode="slower_ema",
           note="Original: EMA7/EMA24 entry, slower-EMA regime SL, ATR=1.5"),
    Config(name="S2_no_regime", direction="short", regime_sl_mode="off",
           note="Drop regime SL — only ATR-SL + EMA TP"),
    Config(name="S3_ema200_regime", direction="short", regime_sl_mode="ema200",
           note="Regime SL only on EMA200 reclaim (real trend reversal)"),
    Config(name="S4_wider_atr", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           note="Drop regime, widen ATR to 2.5×"),
    Config(name="S5_uptrend_filter", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           require_uptrend_for_short=True, uptrend_min_pct=5.0,
           note="Wider ATR + require strong uptrend first (top of run, not chop)"),
    Config(name="S6_strict_pattern", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           min_r2=0.70, min_cup_depth_atr=2.5,
           note="Stricter pattern (R²>0.70, depth ≥ 2.5 ATR)"),
    Config(name="S7_ema24_only", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           entry_emas=("ema24",),
           note="Drop EMA7 entry (too noisy); only EMA24 break"),
    Config(name="S8_combo", direction="short", regime_sl_mode="off", sl_atr_mult=2.5,
           entry_emas=("ema24",), min_r2=0.70, min_cup_depth_atr=2.5,
           require_uptrend_for_short=True, uptrend_min_pct=5.0,
           note="Combine: strict pattern + EMA24-only + uptrend filter + wide SL"),

    # ---- Long (classic C&H) variants ----
    Config(name="L1_baseline", direction="long", regime_sl_mode="slower_ema",
           note="Long C&H: cup + handle, EMA7/24 breakout up"),
    Config(name="L2_no_regime", direction="long", regime_sl_mode="off",
           note="Long, no regime SL"),
    Config(name="L3_wider_atr", direction="long", regime_sl_mode="off", sl_atr_mult=2.5,
           note="Long, wider ATR SL"),
    Config(name="L4_downtrend_filter", direction="long", regime_sl_mode="off", sl_atr_mult=2.5,
           require_downtrend_for_long=True, uptrend_min_pct=5.0,
           note="Long, require downtrend first (bottom of dip, not chasing)"),
    Config(name="L5_strict_pattern", direction="long", regime_sl_mode="off", sl_atr_mult=2.5,
           min_r2=0.70, min_cup_depth_atr=2.5,
           note="Long, stricter pattern"),
    Config(name="L6_combo", direction="long", regime_sl_mode="off", sl_atr_mult=2.5,
           entry_emas=("ema24",), min_r2=0.70, min_cup_depth_atr=2.5,
           note="Long combo: strict + EMA24-only + wide SL"),
]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only a single config by name")
    args = parser.parse_args(argv)

    results = []
    for cfg in CONFIGS:
        if args.only and cfg.name != args.only:
            continue
        print(f"\n>>> Running {cfg.name} ({cfg.direction}): {cfg.note}")
        row = run_config(cfg)
        print(f"    pooled: trades={row.pooled_trades}  "
              f"WR={row.pooled_win_rate*100:.1f}%  "
              f"cum={row.pooled_cum_return*100:+.2f}%  "
              f"sharpe={row.pooled_sharpe:+.2f}")
        results.append({
            "config": asdict(cfg),
            "pooled_trades": row.pooled_trades,
            "pooled_win_rate": row.pooled_win_rate,
            "pooled_cum_return": row.pooled_cum_return,
            "pooled_sharpe": row.pooled_sharpe,
            "by_window": [asdict(w) for w in row.by_window],
        })

    RESULTS.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved → {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
