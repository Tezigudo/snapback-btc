"""Live signal evaluator for cnh-hybrid-short-v1.

Pure-function port of the HYBRID short detector that produced the Phase 1
walk-forward result (OOS cum +18.8%, Sharpe 8.44, dedup=15). Uses the
pattern detectors from `strategy/cnh_detectors.py`.

Returns:
  side ∈ {'short', None}      — HYBRID is short-only by design
  sl_distance, tp_distance    — price units (NaN if side is None)
  debug                       — dict for logging

Stateful pattern dedup (Phase 3b, 2026-05-26):
  The evaluator reconstructs `find_hybrid_patterns`' admission state by
  scanning the visible bars window each call. Pattern admission is
  tracked at the detection bar (NOT the entry bar) — matching backtest.
  A pattern is "admitted" if and only if it's the first detection in its
  dedup_bars window. Admitted patterns block subsequent detections,
  even if they themselves never produce a trade (e.g., ICnH with no
  EMA24 cross-down within entry_max_bars_after_handle).

  Phase 4 portfolio sim verifies: with stateful dedup, live matches the
  "ideal" backtest behaviour (+0.31 Sharpe lift on the 3-leg portfolio).
  Without it, live over-fires by ~40% and drags the portfolio (-0.10
  Sharpe lift).

Other notes:

  - TP is sized to the EMA(100) distance below entry. If EMA(100) is
    at-or-above current close (no valid SHORT TP target), the trade is
    SKIPPED, matching `simulate_trades`' tp_candidates logic.
  - Pattern detection happens on the LAST CLOSED 4h bar. ICnH entries
    also consider patterns admitted in the previous
    `entry_max_bars_after_handle` bars (default 8) — if any of those was
    an ICnH pattern AND the current bar is an EMA(24) cross-down, fire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.cnh_detectors import (
    HybridConfig,
    attach_indicators,
    detect_distribution_top,
    detect_inverse_cnh,
    is_ema_breakdown,
)

# Defaults match the Phase 1 winner (dedup=15, sl=1.5×ATR, tp=EMA100, ent=EMA24).
SL_ATR_MULT = 1.5
TP_EMA = "ema100"
ENTRY_EMA = "ema24"
ATR_PERIOD = 14
ENTRY_MAX_BARS_AFTER_HANDLE = 8
CUP_LEN = 20
HANDLE_LEN = 4
DEDUP_BARS = 15


def _admitted_patterns(
    df: pd.DataFrame, cfg: HybridConfig, up_to_idx: int, dedup_bars: int
) -> list[tuple[int, str]]:
    """Replay `tools.icnh_final_tune.find_hybrid_patterns`' admission logic
    over the bars window. Returns list of (bar_idx, kind) for patterns that
    survived dedup.

    Admission rule mirrors backtest exactly:
      - Walk bars forward from `start` to `up_to_idx` (inclusive).
      - At each bar, check DT first; if hit AND >= dedup_bars since last
        admitted pattern, admit as ("idx", "DT").
      - Else check ICnH; same dedup rule.
      - A pattern that is admitted blocks subsequent patterns for
        dedup_bars-1 bars, regardless of whether it ever produces a
        live entry (e.g. ICnH with no EMA24 cross-down).
    """
    start = max(
        cfg.cup_len + cfg.handle_len,
        cfg.uptrend_bars + cfg.chop_bars + 1,
        200,
    )
    admitted: list[tuple[int, str]] = []
    last_idx: int | None = None
    for j in range(start, up_to_idx + 1):
        dt_hit = detect_distribution_top(df, j, cfg)
        icnh_hit = detect_inverse_cnh(df, j, cfg)
        if dt_hit is not None:
            if last_idx is None or (j - last_idx) >= dedup_bars:
                admitted.append((j, "DT"))
                last_idx = j
        elif icnh_hit is not None:
            if last_idx is None or (j - last_idx) >= dedup_bars:
                admitted.append((j, "ICNH"))
                last_idx = j
    return admitted


def _cfg_from_params(params: dict) -> HybridConfig:
    """Build a HybridConfig from the bot's strategy params dict.

    Falls back to module-level defaults when keys are missing — those
    defaults are the Phase 1-locked winner config.
    """
    s = params.get("strategy", {}) if params else {}
    return HybridConfig(
        uptrend_bars=int(s.get("uptrend_bars", 16)),
        chop_bars=int(s.get("chop_bars", 8)),
        min_rise_pct=float(s.get("min_rise_pct", 2.5)),
        max_chop_ratio=float(s.get("max_chop_ratio", 0.55)),
        require_chop_at_top=bool(s.get("require_chop_at_top", True)),
        breakdown_mode=str(s.get("breakdown_mode", "chop_low_or_ema24")),
        cup_len=int(s.get("cup_len", CUP_LEN)),
        handle_len=int(s.get("handle_len", HANDLE_LEN)),
        min_r2=float(s.get("min_r2", 0.50)),
        min_cup_depth_atr=float(s.get("min_cup_depth_atr", 1.0)),
        handle_max_depth_frac=float(s.get("handle_max_depth_frac", 0.70)),
        peak_tolerance=int(s.get("peak_tolerance", 6)),
        entry_max_bars_after_handle=int(
            s.get("entry_max_bars_after_handle", ENTRY_MAX_BARS_AFTER_HANDLE)
        ),
        entry_emas=tuple(s.get("entry_emas", (ENTRY_EMA,))),
        sl_atr_mult=float(s.get("sl_atr_mult", SL_ATR_MULT)),
        tp_emas=tuple(s.get("tp_emas", (TP_EMA,))),
        atr_period=int(s.get("atr_period", ATR_PERIOD)),
        dedup_bars=int(s.get("dedup_bars", DEDUP_BARS)),
    )


def _resolve_tp_distance(
    df: pd.DataFrame, entry_idx: int, entry_price: float, cfg: HybridConfig
) -> tuple[float, str]:
    """Pick the first configured TP EMA that sits BELOW entry_price.
    Returns (tp_distance, tp_name). tp_distance is NaN if no valid TP."""
    candidates: list[tuple[str, float]] = []
    for nm in cfg.tp_emas:
        v = float(df[nm].iloc[entry_idx])
        if np.isfinite(v) and v < entry_price:
            candidates.append((nm, v))
    if not candidates:
        return float("nan"), ""
    # Prefer the CLOSEST TP below entry (highest of the below-entry EMAs).
    candidates.sort(key=lambda kv: -kv[1])
    name, level = candidates[0]
    return float(entry_price - level), name


def evaluate_signal_cnh_hybrid_short(
    bars_4h: pd.DataFrame,
    funding_rate: float,           # unused; signature compat
    params: dict,
) -> tuple[str | None, float, float, dict]:
    """Evaluate HYBRID short entry on the last closed 4h bar.

    bars_4h: Capitalised OHLCV columns (Open, High, Low, Close, Volume),
    DateTimeIndex (tz-naive or UTC). Needs at least 250 bars (EMA200 warmup
    + DT pattern + ICnH cup + buffer).
    """
    _ = funding_rate
    cfg = _cfg_from_params(params)

    warmup = max(
        cfg.cup_len + cfg.handle_len + 4,
        cfg.uptrend_bars + cfg.chop_bars + 2,
        200 + 5,  # EMA200 warmup
    )
    if len(bars_4h) < warmup:
        return None, float("nan"), float("nan"), {
            "reason": "warmup", "have": len(bars_4h), "need": warmup,
        }

    df = attach_indicators(bars_4h, cfg)   # lowercase columns + EMAs/ATR
    i = len(df) - 1   # the last closed bar

    atr_col = f"atr{cfg.atr_period}"
    atr_v = float(df[atr_col].iloc[i])
    if not (np.isfinite(atr_v) and atr_v > 0):
        return None, float("nan"), float("nan"), {"reason": "atr_nan"}

    entry_price = float(df["close"].iloc[i])
    sl_distance = float(cfg.sl_atr_mult * atr_v)

    base_debug = {
        "ts": str(df.index[i]),
        "close": entry_price,
        "atr": atr_v,
        "ema24": float(df["ema24"].iloc[i]),
        "ema100": float(df["ema100"].iloc[i]),
    }

    # ---- Reconstruct backtest's pattern-level admission state ----
    admitted = _admitted_patterns(df, cfg, i, cfg.dedup_bars)
    base_debug["last_admitted_pattern"] = (
        {"idx": admitted[-1][0], "kind": admitted[-1][1],
         "ts": str(df.index[admitted[-1][0]])}
        if admitted else None
    )

    # ---- 1) Distribution-top trigger ON THIS BAR (and admitted by dedup) ----
    if admitted and admitted[-1][0] == i and admitted[-1][1] == "DT":
        tp_distance, tp_name = _resolve_tp_distance(df, i, entry_price, cfg)
        if not np.isfinite(tp_distance):
            return None, float("nan"), float("nan"), {
                **base_debug, "reason": "dt_admitted_but_no_tp",
                "tp_emas": list(cfg.tp_emas),
            }
        return "short", sl_distance, tp_distance, {
            **base_debug, "pattern": "DT",
            "sl_distance": sl_distance,
            "tp_distance": tp_distance, "tp_name": tp_name,
        }

    # If a DT or ICnH pattern was detected at bar i but BLOCKED by dedup,
    # surface that in debug for observability.
    raw_dt_at_i = detect_distribution_top(df, i, cfg)
    raw_icnh_at_i = detect_inverse_cnh(df, i, cfg)
    if raw_dt_at_i is not None and (not admitted or admitted[-1][0] != i):
        base_debug["dt_blocked_by_dedup"] = True
    if raw_icnh_at_i is not None and (not admitted or admitted[-1][0] != i
                                       or admitted[-1][1] != "ICNH"):
        base_debug["icnh_blocked_by_dedup"] = True

    # ---- 2) ICnH path: look for an ADMITTED ICnH within the lookback ----
    cross_down = is_ema_breakdown(df, i, ENTRY_EMA)
    base_debug["ema24_cross_down"] = cross_down
    if cross_down:
        lookback_start = i - cfg.entry_max_bars_after_handle
        # Walk admitted patterns backward — they're sorted ascending by index.
        for j, kind in reversed(admitted):
            if j >= i:
                continue  # the admitted-at-current-bar case is handled above
            if j < lookback_start:
                break
            if kind != "ICNH":
                # The most recent admitted pattern in the lookback isn't an
                # ICnH (it's DT, which has already been handled or expired).
                # ICnH-trigger semantics in backtest require the *admitted*
                # pattern in the window to be ICnH — DT doesn't wait for
                # cross-down. Match by stopping the scan.
                break
            tp_distance, tp_name = _resolve_tp_distance(df, i, entry_price, cfg)
            if not np.isfinite(tp_distance):
                return None, float("nan"), float("nan"), {
                    **base_debug, "reason": "icnh_admitted_but_no_tp",
                    "tp_emas": list(cfg.tp_emas),
                    "pattern_bar": str(df.index[j]),
                }
            return "short", sl_distance, tp_distance, {
                **base_debug, "pattern": "ICNH",
                "pattern_bar": str(df.index[j]),
                "sl_distance": sl_distance,
                "tp_distance": tp_distance, "tp_name": tp_name,
            }

    return None, float("nan"), float("nan"), {
        **base_debug, "reason": "no_signal",
    }
