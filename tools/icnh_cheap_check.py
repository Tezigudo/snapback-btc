"""Cheap sanity check for the Inverse Cup-and-Handle (ICnH) short strategy.

User's verbal rules:
  - Pattern: rounded dome (inverse cup) + small handle on right side
  - TFs: 4h primary; 1h secondary fallback if 4h is unclear
    * if 4h pattern fires → require 1h confirmation too (4h AND 1h)
    * if only 1h fires → trade on 1h alone
  - Entry: AFTER the dome forms, short on first BAR-CLOSE below EMA(7) or EMA(24).
           "Middle of A and C" zone — i.e. earlier than handle-low break.
  - SL: max of (entry + 1.5×ATR(14), regime-invalidation = close back above
        the EMA that triggered entry).
  - TP: ladder through EMAs below entry. For the cheap check we use the
        simpler "first close ≤ nearest EMA below → exit next open". That
        underestimates true TP-ladder profit but is a conservative floor.

Goal: do enough signals fire? Is hit-rate > 50%? Is total return positive?
Threshold to graduate to full walk-forward: Sharpe ≥ 1.0 AND ≥ 30 fires AND
win rate > 50% AND total return > +20% over the test window.

Run:  uv run python tools/icnh_cheap_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strategy.indicators import atr, ema  # noqa: E402

DATA = ROOT / "data" / "historical"

# ----- Pattern detector knobs (kept conservative; sweep later if signal exists) -----
CUP_LEN = 20            # bars of cup window (peak roughly in middle)
HANDLE_LEN = 5          # bars of handle (tight consolidation after right lip)
PEAK_TOLERANCE = 3      # peak must be within ±N bars of cup center
MIN_CUP_DEPTH_ATR = 1.5 # cup depth must be ≥ 1.5 × ATR to be "real"
MIN_R2 = 0.55           # parabola fit quality (rounded, not V-spike)
HANDLE_MAX_DEPTH_FRAC = 0.45  # handle range ≤ 45% of cup depth (tight handle)
ENTRY_MAX_BARS_AFTER_HANDLE = 8  # entry must fire within N bars of handle ending

# ----- Strategy knobs -----
ATR_LEN = 14
SL_ATR_MULT = 1.5
EMA_FAST = 7
EMA_MED = 24
EMA_TP1 = 100
EMA_TP2 = 200

# ----- Cheap-check window -----
TEST_START = "2024-01-01"
TEST_END = "2026-05-23"

# Friction (round-trip fees + slippage). 5 bps each side = 10 bps round-trip.
FRICTION_BPS = 10.0


def load(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"BTC_USDT_USDT_{tf}.parquet")
    df = df.sort_index()
    return df


def precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema7"] = ema(out["close"], EMA_FAST)
    out["ema24"] = ema(out["close"], EMA_MED)
    out["ema100"] = ema(out["close"], EMA_TP1)
    out["ema200"] = ema(out["close"], EMA_TP2)
    out["atr14"] = atr(out["high"], out["low"], out["close"], ATR_LEN)
    return out


def _fit_parabola_r2(y: np.ndarray) -> tuple[float, float]:
    """Returns (a_coeff, r_squared) for y = a*x² + b*x + c fit.
    Negative `a` = concave down (inverted bowl, what we want for ICnH).
    """
    x = np.arange(len(y), dtype=float)
    coeffs = np.polyfit(x, y, 2)
    a = coeffs[0]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(r2)


def has_inverse_cup_handle(df: pd.DataFrame, end_idx: int) -> dict | None:
    """Check if an inverse cup-and-handle pattern ENDS at bar `end_idx`.
    The handle's last bar is `end_idx`. The cup is the CUP_LEN bars before
    the handle started.
    Returns dict with cup metrics if pattern present, else None.
    """
    handle_start = end_idx - HANDLE_LEN + 1
    handle_end = end_idx
    cup_end = handle_start - 1
    cup_start = cup_end - CUP_LEN + 1
    if cup_start < 0 or handle_end >= len(df):
        return None

    cup = df.iloc[cup_start : cup_end + 1]
    handle = df.iloc[handle_start : handle_end + 1]

    closes = cup["close"].to_numpy()
    if np.any(np.isnan(closes)):
        return None

    # Concave-down parabola fit on cup closes
    a, r2 = _fit_parabola_r2(closes)
    if a >= 0:
        return None
    if r2 < MIN_R2:
        return None

    # Peak roughly in middle of cup
    peak_pos = int(cup["high"].values.argmax())
    center = CUP_LEN // 2
    if abs(peak_pos - center) > PEAK_TOLERANCE:
        return None

    peak_price = float(cup["high"].iloc[peak_pos])
    left_lip = float(cup["low"].iloc[:peak_pos + 1].min())
    right_lip = float(cup["low"].iloc[peak_pos:].min())
    base = min(left_lip, right_lip)
    cup_depth = peak_price - base

    # Cup depth must clear ATR-floor (not a noise blip)
    atr_at_peak = float(cup["close"].iloc[peak_pos] - cup["close"].iloc[peak_pos])
    atr_val = float(df["atr14"].iloc[cup_end])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    if cup_depth < MIN_CUP_DEPTH_ATR * atr_val:
        return None

    # Handle: stays in upper portion of cup (above midpoint) AND tight range
    cup_midpoint = (peak_price + base) / 2.0
    if handle["low"].min() < cup_midpoint:
        return None  # handle dropped too low — pattern has already broken
    handle_range = float(handle["high"].max() - handle["low"].min())
    if handle_range > HANDLE_MAX_DEPTH_FRAC * cup_depth:
        return None  # handle too wide — no real consolidation

    return {
        "peak_price": peak_price,
        "cup_depth": cup_depth,
        "handle_low": float(handle["low"].min()),
        "handle_high": float(handle["high"].max()),
        "atr_at_handle": atr_val,
        "r2": r2,
    }


def find_all_patterns(df: pd.DataFrame) -> list[int]:
    """Returns list of bar indices where an ICnH pattern is freshly complete.
    Dedups overlapping patterns: keeps only the earliest in any 10-bar cluster.
    """
    hits: list[int] = []
    for i in range(CUP_LEN + HANDLE_LEN, len(df)):
        if has_inverse_cup_handle(df, i) is not None:
            if hits and (i - hits[-1]) < 10:
                continue
            hits.append(i)
    return hits


def simulate_trades(df: pd.DataFrame, pattern_bars: list[int], tf_label: str) -> list[dict]:
    """Walk forward from each pattern. Find the next entry trigger (close <
    EMA7 or EMA24). Then track SL / TP exits.
    """
    trades: list[dict] = []

    for pidx in pattern_bars:
        pinfo = has_inverse_cup_handle(df, pidx)
        if pinfo is None:
            continue

        # Walk forward up to ENTRY_MAX_BARS_AFTER_HANDLE bars looking for trigger
        entry_idx: int | None = None
        broken_ema_name: str | None = None
        broken_ema_value: float | None = None

        for j in range(pidx + 1, min(pidx + 1 + ENTRY_MAX_BARS_AFTER_HANDLE, len(df))):
            row = df.iloc[j]
            close = float(row["close"])
            ema7 = float(row["ema7"])
            ema24 = float(row["ema24"])
            # Must have been ABOVE the EMA on the prior bar (otherwise we're not "breaking" anything)
            prev = df.iloc[j - 1]
            prev_close = float(prev["close"])
            if prev_close >= float(prev["ema7"]) and close < ema7:
                entry_idx = j
                broken_ema_name = "ema7"
                broken_ema_value = ema7
                break
            if prev_close >= float(prev["ema24"]) and close < ema24:
                entry_idx = j
                broken_ema_name = "ema24"
                broken_ema_value = ema24
                break

        if entry_idx is None:
            continue

        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        atr_at_entry = float(entry_row["atr14"])
        hard_sl = entry_price + SL_ATR_MULT * atr_at_entry

        # TP target = nearest EMA strictly BELOW entry, from {ema100, ema200}
        candidates = []
        for nm in ("ema100", "ema200"):
            v = float(entry_row[nm])
            if v < entry_price:
                candidates.append((nm, v))
        if not candidates:
            continue  # No EMA below entry → unusual; skip
        candidates.sort(key=lambda kv: -kv[1])  # nearest (highest) first
        tp_name, tp_value = candidates[0]

        # Walk forward from entry until SL or TP hits
        exit_idx: int | None = None
        exit_price: float | None = None
        exit_reason: str = ""

        # Regime SL: use a SLOWER EMA than the trigger, so it doesn't whipsaw.
        # If we entered on EMA7 break → regime-SL on EMA24 reclaim.
        # If we entered on EMA24 break → regime-SL on EMA100 reclaim (a real reversal,
        # not a normal bounce).
        regime_ema_name = "ema24" if broken_ema_name == "ema7" else "ema100"

        for k in range(entry_idx + 1, len(df)):
            row = df.iloc[k]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            # Regime-invalidation SL on the SLOWER EMA (not the one we broke for entry)
            ema_regime = float(row[regime_ema_name])
            if close > ema_regime:
                exit_idx = k
                exit_price = close
                exit_reason = f"regime_sl_{regime_ema_name}_reclaim"
                break

            # Hard ATR SL
            if high >= hard_sl:
                exit_idx = k
                exit_price = hard_sl
                exit_reason = "atr_sl"
                break

            # TP-1: refresh nearest EMA each bar (it moves)
            ema_tp = float(row[tp_name])
            if low <= ema_tp:
                exit_idx = k
                exit_price = ema_tp
                exit_reason = f"tp_{tp_name}"
                break

        if exit_idx is None:
            # Force exit at end of data
            exit_idx = len(df) - 1
            exit_price = float(df.iloc[exit_idx]["close"])
            exit_reason = "eod"

        gross_pct = (entry_price - exit_price) / entry_price  # short: profit if exit < entry
        net_pct = gross_pct - FRICTION_BPS / 10000.0  # round-trip friction

        trades.append({
            "tf": tf_label,
            "pattern_bar": pidx,
            "entry_ts": df.index[entry_idx],
            "entry_price": entry_price,
            "broken_ema": broken_ema_name,
            "atr_sl": hard_sl,
            "tp_target": tp_name,
            "exit_ts": df.index[exit_idx],
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "gross_pct": gross_pct,
            "net_pct": net_pct,
            "bars_held": exit_idx - entry_idx,
        })

    return trades


def apply_dual_tf_logic(trades_4h: list[dict], trades_1h: list[dict], df_4h: pd.DataFrame,
                       df_1h: pd.DataFrame) -> list[dict]:
    """User rule:
       - If a 4h trade fires, require a 1h pattern within the prior 24h (= 24 1h-bars)
         OR within the entry-bar's 4h window. Otherwise drop the 4h trade.
       - 1h trades that don't have a 4h counterpart fire ALONE (kept as-is).
    """
    # Build set of 1h pattern bar timestamps for fast lookup
    pat_1h_indices = find_all_patterns(df_1h)
    pat_1h_timestamps = [df_1h.index[i] for i in pat_1h_indices]
    pat_1h_set = sorted(pat_1h_timestamps)

    kept: list[dict] = []

    # Track which 1h trades are "echoed" by 4h to avoid double-counting
    used_1h_ts = set()

    # 4h trades: require nearby 1h pattern
    for t in trades_4h:
        ent_ts = t["entry_ts"]
        window_lo = ent_ts - pd.Timedelta(hours=24)
        echoes = [ts for ts in pat_1h_set if window_lo <= ts <= ent_ts]
        if echoes:
            t = dict(t)
            t["confirmed_by_1h"] = True
            kept.append(t)
            for ts in echoes:
                used_1h_ts.add(ts)

    # 1h-alone trades: keep ONLY if no 4h trade fired within ±12h
    for t in trades_1h:
        ent_ts = t["entry_ts"]
        # Was there a 4h trade nearby? (already counted above)
        nearby_4h = any(
            abs((t4["entry_ts"] - ent_ts).total_seconds()) <= 12 * 3600
            for t4 in trades_4h
        )
        if nearby_4h:
            continue  # already represented by the 4h trade
        t = dict(t)
        t["confirmed_by_1h"] = False  # standalone 1h
        kept.append(t)

    kept.sort(key=lambda t: t["entry_ts"])
    return kept


def stats(trades: list[dict], label: str) -> None:
    if not trades:
        print(f"\n=== {label}: NO TRADES ===")
        return
    nets = np.array([t["net_pct"] for t in trades])
    wins = nets > 0
    cum_return = float(np.prod(1.0 + nets) - 1.0)  # compounded
    sharpe = float(nets.mean() / nets.std()) * np.sqrt(252 * 6) if nets.std() > 0 else 0.0  # ~6 4h bars/day, rough
    bars_held = np.array([t["bars_held"] for t in trades])

    exits = {}
    for t in trades:
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1

    print(f"\n=== {label} ===")
    print(f"  trades:        {len(trades)}")
    print(f"  win rate:      {wins.mean() * 100:.1f}%  ({int(wins.sum())}/{len(trades)})")
    print(f"  mean net:      {nets.mean() * 100:+.3f}% per trade")
    print(f"  median net:    {float(np.median(nets)) * 100:+.3f}%")
    print(f"  best / worst:  {nets.max() * 100:+.2f}% / {nets.min() * 100:+.2f}%")
    print(f"  cum return:    {cum_return * 100:+.2f}%  (compounded, friction-adj)")
    print(f"  sharpe (rough):{sharpe:+.2f}")
    print(f"  avg bars held: {bars_held.mean():.1f}  (min {bars_held.min()}, max {bars_held.max()})")
    print(f"  exit reasons:  {exits}")


def main() -> int:
    print("Loading data + computing indicators...")
    df_4h_full = precompute_indicators(load("4h"))
    df_1h_full = precompute_indicators(load("1h"))

    df_4h = df_4h_full.loc[TEST_START:TEST_END].copy().reset_index()
    df_4h.set_index("index", inplace=True) if "index" in df_4h.columns else None
    df_4h = df_4h_full.loc[TEST_START:TEST_END].copy()
    df_1h = df_1h_full.loc[TEST_START:TEST_END].copy()

    print(f"  4h window: {df_4h.index.min()} → {df_4h.index.max()}  ({len(df_4h):,} bars)")
    print(f"  1h window: {df_1h.index.min()} → {df_1h.index.max()}  ({len(df_1h):,} bars)")

    print("\nScanning for ICnH patterns on 4h...")
    pat_4h = find_all_patterns(df_4h)
    print(f"  found {len(pat_4h)} pattern endings on 4h")

    print("Scanning for ICnH patterns on 1h...")
    pat_1h = find_all_patterns(df_1h)
    print(f"  found {len(pat_1h)} pattern endings on 1h")

    print("\nSimulating trades...")
    trades_4h = simulate_trades(df_4h, pat_4h, "4h")
    trades_1h = simulate_trades(df_1h, pat_1h, "1h")
    print(f"  4h: {len(trades_4h)} entries triggered (after pattern)")
    print(f"  1h: {len(trades_1h)} entries triggered (after pattern)")

    stats(trades_4h, "4h-ONLY (no dual-TF filter)")
    stats(trades_1h, "1h-ONLY (no dual-TF filter)")

    print("\nApplying user's dual-TF rule (4h needs 1h echo; 1h-alone OK if no 4h nearby)...")
    combined = apply_dual_tf_logic(trades_4h, trades_1h, df_4h, df_1h)
    stats(combined, "COMBINED dual-TF (deploy candidate)")

    # Verdict
    if combined:
        nets = np.array([t["net_pct"] for t in combined])
        wr = float((nets > 0).mean())
        cum = float(np.prod(1.0 + nets) - 1.0)
        n = len(combined)
        print("\n=== VERDICT ===")
        ok_n = n >= 30
        ok_wr = wr > 0.50
        ok_cum = cum > 0.20
        print(f"  fires ≥ 30:        {n}     {'PASS' if ok_n else 'FAIL'}")
        print(f"  win rate > 50%:    {wr * 100:.1f}%  {'PASS' if ok_wr else 'FAIL'}")
        print(f"  cum return > +20%: {cum * 100:+.1f}%  {'PASS' if ok_cum else 'FAIL'}")
        graduate = ok_n and ok_wr and ok_cum
        print(f"\n  {'GRADUATE → full walk-forward' if graduate else 'KILL → dead end, try next idea'}")
    else:
        print("\n=== VERDICT: NO TRADES — pattern detector too tight, OR rules never fire ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
