"""OOS validation of KC-BB Squeeze Breakout (todo-leg candidate) post fractional sizing.

Mirrors tools/_postfrac_mf_baseline.py:
  - 5 OOS windows (2022 H1, 2023 H1, 2024 H1, 2024 H2, 2025 H1)
  - $1M cash, 0.0005 commission, 1/20 margin
  - PRICE_SCALE = 0.001 applied to OHLC at slice time (volume / funding unscaled)
  - aggregates trade-level pnl_pct, computes PSR via tools/psr_eval.compute_psr
  - writes per-window CSV + aggregate JSON

This is the FIRST OOS validation of any TODO_LEG candidate, so the harness is
intentionally identical to multifactor-v1's baseline runner — no exotic
overrides, single config, no in-sample tuning. Base rate of prior strategies:
4 of 9 SHELVED. Expect this to fail unless there's clear evidence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_kc_squeeze_breakout import KCSqueezeBreakoutBTC  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20
PRICE_SCALE = 0.001

CONFIG = {
    # BB / KC
    "bb_period":          20,
    "bb_n_std":           2.0,
    "kc_ema_period":      20,
    "kc_atr_period":      20,
    "kc_mult":            1.5,
    # Squeeze gate
    "squeeze_min_bars":   10,
    # Breakout direction
    "donchian_period":    20,
    # Volume confirm
    "volume_ma_period":   20,
    "volume_multiple":    1.5,
    # ATR-based stops/targets
    "atr_period":         14,
    "stop_atr_mult":      1.5,
    "target_atr_mult":    2.5,
    # Sizing
    "risk_per_trade_pct": 2.0,
    "leverage":           20,
    "allow_shorts":       True,
    "max_hold_bars":      1344,
    "enabled":            True,
}

WINDOWS = [
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2023_H1", "2023-01-01", "2023-06-30"),
    ("2024_H1", "2024-01-01", "2024-06-30"),
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
]


def _load_slice_scaled(start: str, end: str) -> pd.DataFrame:
    """Load 15m BTC parquet, slice [start, end] inclusive, scale OHLC by PRICE_SCALE.

    Identical to tools/_postfrac_mf_baseline.py._load_slice_scaled minus funding
    (this strategy does not use funding).
    """
    df = pd.read_parquet(PARQUET)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sl = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sl) == 0:
        raise ValueError(f"Empty slice {start}..{end}")

    # Scale OHLC (volume stays un-scaled — it's a count, not a price)
    for col in ("Open", "High", "Low", "Close"):
        if col in sl.columns:
            sl[col] = sl[col] * PRICE_SCALE
    return sl


def run_window(label: str, start: str, end: str) -> dict:
    df = _load_slice_scaled(start, end)
    bt = Backtest(
        df, KCSqueezeBreakoutBTC,
        cash=CASH, commission=COMMISSION, margin=MARGIN,
        trade_on_close=False, exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run(**CONFIG)

    n_trades = int(stats.get("# Trades", 0) or 0)
    ret_pct = float(stats.get("Return [%]", 0.0) or 0.0)
    max_dd = float(stats.get("Max. Drawdown [%]", 0.0) or 0.0)
    win_rate = float(stats.get("Win Rate [%]") or 0.0)
    equity_final = float(stats.get("Equity Final [$]", CASH) or CASH)

    trades_df = getattr(stats, "_trades", None)
    pnl_pct_list = []
    if trades_df is not None and len(trades_df) > 0 and "ReturnPct" in trades_df.columns:
        pnl_pct_list = (trades_df["ReturnPct"].values * 100.0).tolist()
        out_csv = ROOT / "reports" / f"_postfrac_kc_squeeze_{label}.csv"
        out = pd.DataFrame({
            "pnl_pct": pnl_pct_list,
            "window_start": start,
            "window_end": end,
        })
        out.to_csv(out_csv, index=False)
        print(f"  [{label}] saved {len(out)} trades -> {out_csv.name}",
              file=sys.stderr)

    return {
        "label":        label,
        "start":        start,
        "end":          end,
        "trades":       n_trades,
        "return_pct":   round(ret_pct, 4),
        "max_dd_pct":   round(max_dd, 4),
        "win_rate_pct": round(win_rate, 4),
        "equity_final": round(equity_final, 4),
        "pnl_pct":      pnl_pct_list,
    }


def aggregate(per_window: list[dict]) -> dict:
    all_pnl: list[float] = []
    n_trades = 0
    n_pos = 0
    compounded = 1.0
    per_win_ret = []
    for r in per_window:
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        per_win_ret.append(r["return_pct"])
        if r["return_pct"] > 0:
            n_pos += 1
    return {
        "n_trades":              n_trades,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{len(per_window)}",
        "per_window_return_pct": per_win_ret,
        "all_pnl_pct":           all_pnl,
    }


def verdict(psr_value: float, compounded_pct: float, n_trades: int) -> dict:
    """Apply TODO_LEG validation gates (from candidate file).

    Gates (per task brief):
      - compounded < 0%      -> SHELF (regardless of PSR)
      - PSR > 0.95           -> CANDIDATE_PROMOTE (walk-forward next)
      - PSR > 0.80           -> ITERATE
      - else                 -> SHELF

    Also surface practical sanity checks:
      - trade_count too low (< 30 across 5 windows) -> insufficient_evidence
    """
    if n_trades < 30:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "reason":  f"only {n_trades} trades across 5 OOS windows (<30 min)",
        }
    if compounded_pct < 0.0:
        return {
            "verdict": "SHELF",
            "reason":  f"compounded {compounded_pct:.2f}% < 0% — auto-shelf regardless of PSR",
        }
    if psr_value > 0.95:
        return {
            "verdict": "CANDIDATE_PROMOTE",
            "reason":  f"PSR {psr_value:.4f} > 0.95 and compounded {compounded_pct:.2f}% > 0 — proceed to walk-forward",
        }
    if psr_value > 0.80:
        return {
            "verdict": "ITERATE",
            "reason":  f"PSR {psr_value:.4f} between 0.80 and 0.95 — refine signal or try variants",
        }
    return {
        "verdict": "SHELF",
        "reason":  f"PSR {psr_value:.4f} < 0.80 — insufficient edge",
    }


def main() -> int:
    t0 = time.time()
    per_window = []
    print("[postfrac_kc_squeeze] running 5 OOS windows ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        tw = time.time()
        r = run_window(label, start, end)
        print(
            f"  {label}  trades={r['trades']:4d}  ret={r['return_pct']:+8.2f}%  "
            f"dd={r['max_dd_pct']:+7.2f}%  win={r['win_rate_pct']:5.2f}%  "
            f"({time.time()-tw:.1f}s)",
            file=sys.stderr,
        )
        per_window.append(r)

    agg = aggregate(per_window)
    all_pnl = agg.pop("all_pnl_pct")

    pnl_arr = np.asarray(all_pnl, dtype=float)
    psr = compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95) if len(pnl_arr) >= 2 else {
        "n_trades": int(len(pnl_arr)),
        "psr_vs_hurdle": 0.0,
        "interpretation": "insufficient_evidence",
    }

    agg_csv = ROOT / "reports" / "_postfrac_kc_squeeze_aggregated.csv"
    pd.DataFrame({"pnl_pct": all_pnl}).to_csv(agg_csv, index=False)
    print(f"[postfrac_kc_squeeze] aggregated CSV -> {agg_csv.name}", file=sys.stderr)

    v = verdict(
        psr_value=float(psr.get("psr_vs_hurdle", 0.0)),
        compounded_pct=float(agg["compounded_pct"]),
        n_trades=int(agg["n_trades"]),
    )

    result = {
        "strategy_id":     "kc_squeeze_breakout_v0",
        "strategy_class":  "strategy.signals_kc_squeeze_breakout:KCSqueezeBreakoutBTC",
        "cash":            CASH,
        "commission":      COMMISSION,
        "margin":          MARGIN,
        "price_scale":     PRICE_SCALE,
        "config":          CONFIG,
        "windows":         [w[0] for w in WINDOWS],
        "per_window":      [{k: vv for k, vv in r.items() if k != "pnl_pct"} for r in per_window],
        "summary":         agg,
        "psr":             psr,
        "verdict":         v,
        "elapsed_sec":     round(time.time() - t0, 2),
    }

    out_path = ROOT / "reports" / "postfrac_kc_squeeze.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[postfrac_kc_squeeze] wrote {out_path}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    print(f"[postfrac_kc_squeeze] verdict: {v['verdict']} — {v['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
