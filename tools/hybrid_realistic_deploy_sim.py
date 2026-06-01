"""Realistic deploy simulation for cnh-hybrid-short-v1.

Models what the LIVE bot would actually do at a given starting equity:
  - Risk-based sizing: notional = equity × risk_pct / sl_distance_pct
  - Skip when notional < $50 (Binance min-notional) OR
                qty < 0.001 BTC (Binance min-qty)
  - Compound: each kept trade's return updates running equity
  - Kill switch at -35.5% of START equity
  - Output: equity curve, kept count, skip reason breakdown

Run a sweep across starting equity ∈ {50, 80, 100, 150, 200} and
risk ∈ {1.5%, 2.0%, 2.75%, 3.5%} to map the deploy decision space.

Run:
    uv run python tools/hybrid_realistic_deploy_sim.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.cnh_detectors import (  # noqa: E402
    HybridConfig,
    attach_indicators,
    is_ema_breakdown,
)
from strategy.live_cnh_hybrid_short import _admitted_patterns  # noqa: E402
from tools.icnh_mega_sweep import load_tf  # noqa: E402

RESULTS_PATH = ROOT / "data" / "hybrid_realistic_deploy_results.json"

SIM_START = pd.Timestamp("2020-01-01", tz="UTC")
SIM_END = pd.Timestamp("2026-05-23", tz="UTC")
TIME_STOP_BARS = 96
FRICTION_BPS_RT = 13.0
DEDUP_BARS = 15

MIN_NOTIONAL_USDT = 50.0
MIN_QTY_BTC = 0.001

KILL_SWITCH_FRACTION = 0.645   # -35.5%

CAPITAL_SCENARIOS = [
    {"start_equity": 50.0,  "risk_pct": 1.5},   # current plan (memory: phase2 floor)
    {"start_equity": 50.0,  "risk_pct": 2.75},  # raise risk
    {"start_equity": 80.0,  "risk_pct": 2.75},
    {"start_equity": 100.0, "risk_pct": 1.5},
    {"start_equity": 100.0, "risk_pct": 2.0},
    {"start_equity": 100.0, "risk_pct": 2.75},  # config sizing
    {"start_equity": 150.0, "risk_pct": 2.75},
    {"start_equity": 200.0, "risk_pct": 2.75},
]


def _build_candidates(df: pd.DataFrame, cfg: HybridConfig) -> list[dict]:
    """Compute the full candidate list (signal_idx, entry_price, atr) for the
    HYBRID short strategy. Sizing-agnostic: we apply (equity, risk_pct,
    min_qty, min_notional) filters per scenario in the simulator."""
    admitted = _admitted_patterns(df, cfg, len(df) - 1, DEDUP_BARS)
    cands: list[dict] = []
    for idx, kind in admitted:
        if kind == "DT":
            signal_idx = idx
        else:
            signal_idx = None
            limit = min(idx + 1 + cfg.entry_max_bars_after_handle, len(df))
            for j in range(idx + 1, limit):
                if is_ema_breakdown(df, j, "ema24"):
                    signal_idx = j
                    break
            if signal_idx is None:
                continue
        atr_v = float(df["atr14"].iloc[signal_idx])
        entry_price = float(df["close"].iloc[signal_idx])
        ema100 = float(df["ema100"].iloc[signal_idx])
        if not (np.isfinite(atr_v) and atr_v > 0 and np.isfinite(ema100)
                and ema100 < entry_price):
            continue
        cands.append({
            "signal_idx": signal_idx, "pattern": kind,
            "entry_price": entry_price,
            "atr": atr_v,
            "stop": entry_price + cfg.sl_atr_mult * atr_v,
            "tp": ema100,
            "sl_pct": cfg.sl_atr_mult * atr_v / entry_price,
            "tp_pct": (entry_price - ema100) / entry_price,
        })
    return cands


def _simulate_one_scenario(
    df: pd.DataFrame,
    cands: list[dict],
    start_equity: float,
    risk_pct: float,
) -> dict:
    """Walk forward through the candidate list, sizing each by current equity.
    Apply min-qty/notional skips. Compound returns into equity. Apply
    kill-switch."""
    risk_frac = risk_pct / 100.0
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    ts = df.index

    equity = start_equity
    kill_floor = start_equity * KILL_SWITCH_FRACTION
    trades: list[dict] = []
    skipped: list[dict] = []
    open_trade: dict | None = None
    killed = False
    equity_curve: list[tuple[pd.Timestamp, float]] = [(ts[0], equity)]

    cand_iter = iter(cands)
    next_cand = next(cand_iter, None)

    for i in range(250, len(df)):
        if killed:
            break
        # Resolve any open trade.
        if open_trade is not None:
            o = open_trade
            hit = None
            if high[i] >= o["stop"]:
                hit = ("sl", o["stop"])
            elif low[i] <= o["tp"]:
                hit = ("tp", o["tp"])
            elif (i - o["entry_idx"]) >= TIME_STOP_BARS:
                hit = ("time", float(close[i]))
            if hit is not None:
                reason, exit_price = hit
                gross = (o["entry_price"] - exit_price) / o["entry_price"]
                net = gross - FRICTION_BPS_RT / 10_000.0
                # Apply the net return to the AMOUNT THAT WAS AT RISK, not
                # to whole equity — risk-based sizing implies the dollar
                # P&L is risk_amount × R-multiple.
                #
                # Easier: compute pnl_usd = notional × net (the perp's
                # contract math). notional was set at entry.
                pnl_usd = o["notional"] * net
                equity += pnl_usd
                trades.append({
                    "entry_ts": str(o["entry_ts"]), "exit_ts": str(ts[i]),
                    "pattern": o["pattern"], "exit_reason": reason,
                    "notional": o["notional"], "net_pct": net,
                    "pnl_usd": pnl_usd, "equity_after": equity,
                })
                open_trade = None
                equity_curve.append((ts[i], equity))
                if equity <= kill_floor:
                    killed = True
                    break

        # Fast-forward candidates that already passed.
        while next_cand is not None and next_cand["signal_idx"] < i:
            next_cand = next(cand_iter, None)
        if (next_cand is None or next_cand["signal_idx"] != i
                or open_trade is not None):
            continue

        c = next_cand
        entry_price = c["entry_price"]
        sl_pct = c["sl_pct"]
        notional = equity * risk_frac / sl_pct
        qty = notional / entry_price
        skip_reason = None
        if notional < MIN_NOTIONAL_USDT:
            skip_reason = "min_notional"
        elif qty < MIN_QTY_BTC:
            skip_reason = "min_qty"
        if skip_reason:
            skipped.append({
                "ts": str(ts[i]), "pattern": c["pattern"],
                "notional": notional, "qty": qty, "reason": skip_reason,
            })
        else:
            open_trade = {
                "entry_ts": ts[i], "entry_idx": i,
                "entry_price": entry_price,
                "stop": c["stop"], "tp": c["tp"],
                "pattern": c["pattern"], "notional": notional,
            }
        next_cand = next(cand_iter, None)

    # Force-close any open trade at EOD.
    if open_trade is not None and not killed:
        last_idx = len(df) - 1
        o = open_trade
        exit_price = float(close[last_idx])
        gross = (o["entry_price"] - exit_price) / o["entry_price"]
        net = gross - FRICTION_BPS_RT / 10_000.0
        pnl_usd = o["notional"] * net
        equity += pnl_usd
        trades.append({
            "entry_ts": str(o["entry_ts"]), "exit_ts": str(ts[last_idx]),
            "pattern": o["pattern"], "exit_reason": "eod",
            "notional": o["notional"], "net_pct": net,
            "pnl_usd": pnl_usd, "equity_after": equity,
        })

    final_equity = equity
    cum_return = (final_equity - start_equity) / start_equity
    return {
        "start_equity": start_equity, "risk_pct": risk_pct,
        "final_equity": final_equity, "cum_pct": cum_return,
        "trades_kept": len(trades),
        "trades_skipped": len(skipped),
        "skip_reasons": {
            "min_notional": sum(1 for s in skipped if s["reason"] == "min_notional"),
            "min_qty": sum(1 for s in skipped if s["reason"] == "min_qty"),
        },
        "killed": killed,
        "wins": sum(1 for t in trades if t["net_pct"] > 0),
        "win_rate": (sum(1 for t in trades if t["net_pct"] > 0) / len(trades)
                     if trades else 0.0),
        "by_pattern_kept": {
            "DT": sum(1 for t in trades if t["pattern"] == "DT"),
            "ICNH": sum(1 for t in trades if t["pattern"] == "ICNH"),
        },
    }


def main() -> int:
    print("=" * 78)
    print("HYBRID SHORT — realistic deploy simulation")
    print(f"Window: {SIM_START.date()} → {SIM_END.date()}")
    print(f"Friction: {FRICTION_BPS_RT} bps  Time-stop: {TIME_STOP_BARS} bars")
    print(f"Kill switch: equity drops below {KILL_SWITCH_FRACTION:.3f} of start "
          f"(= -{(1 - KILL_SWITCH_FRACTION) * 100:.1f}%)")
    print(f"Exchange: min-notional ${MIN_NOTIONAL_USDT:.0f}, min-qty {MIN_QTY_BTC} BTC")
    print("=" * 78)

    t_ind = time.time()
    df_raw = load_tf("4h").loc[SIM_START:SIM_END]
    cfg = HybridConfig(dedup_bars=DEDUP_BARS)
    df = attach_indicators(df_raw, cfg)
    cands = _build_candidates(df, cfg)
    print(f"\nCandidate generation: {time.time() - t_ind:.1f}s "
          f"({len(cands)} HYBRID candidates over the window)")

    results = []
    for sc in CAPITAL_SCENARIOS:
        r = _simulate_one_scenario(df, cands, sc["start_equity"], sc["risk_pct"])
        results.append(r)

    print(
        f"\n{'$start':>8}{'risk%':>7}{'$final':>10}{'cum':>10}"
        f"{'kept':>7}{'skip':>7}{'WR':>7}{'killed':>8}"
        f"{'DT/ICNH':>10}"
    )
    print("-" * 78)
    for r in results:
        skip_breakdown = (
            f"({r['skip_reasons']['min_notional']}n,"
            f"{r['skip_reasons']['min_qty']}q)"
        )
        print(
            f"{r['start_equity']:>8.0f}{r['risk_pct']:>7.2f}"
            f"{r['final_equity']:>10.2f}{r['cum_pct'] * 100:>+9.1f}%"
            f"{r['trades_kept']:>7}{r['trades_skipped']:>4}{skip_breakdown:<7}"
            f"{r['win_rate'] * 100:>6.1f}%{('Y' if r['killed'] else 'n'):>8}"
            f"  {r['by_pattern_kept']['DT']:>2}/{r['by_pattern_kept']['ICNH']:>2}"
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved → {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
