"""multifactor-v1 + 4H EMA200 gate — live/backtest parity validation.

Pattern adapted from tools/hybrid_phase3_validate.py.

Two-stage validation:

(stage 1) IMPL CROSS-CHECK — fast.
    Compare the new in-class gate (`DayTradeMultiFactorBTC.use_mtf_4h_gate=True`)
    against the existing `MultiFactorMTF4H` subclass in tools/run_mf_deepening.py
    on a single OOS window. They should produce IDENTICAL trade lists. If they
    don't, the in-class implementation is wrong — investigate before going on.

(stage 2) LIVE PARITY — the actual gate.
    For each 15m bar in a 90-day window, evaluate BOTH paths and compare:
      (a) Backtest: re-run DayTradeMultiFactorBTC over the window, record the
          set of bar timestamps where _long_signal/_short_signal would return
          True at that bar.
      (b) Live: invoke evaluate_signal(bars_15m=window_up_to_bar_i,
                                       funding_rate=bar_i_funding,
                                       params=params,
                                       bars_4h=<injected slice>) bar by bar.
    Compute per-bar agreement % and dump mismatches.

Gate:
    Stage 1: 100% trade-list equality.
    Stage 2: ≥99.5% signal-bar parity (stricter than 99% used elsewhere since
             this is a LOCKED LIVE strategy).

Output: reports/multifactor_4h_parity_validation.json

Run:
    uv run python tools/multifactor_validate.py
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

from strategy.live_multifactor_v1 import evaluate_signal  # noqa: E402
from strategy.signals_multifactor import (  # noqa: E402
    DayTradeMultiFactorBTC,
    _build_4h_ema_aligned,
)
from tools.run_mf_deepening import (  # noqa: E402
    CASH,
    COMMISSION,
    LOCKED,
    MARGIN,
    MultiFactorMTF4H,
    _load_slice,
)

PARITY_GATE = 99.5

# Locked LIVE params (config/params.yaml) PLUS the 4H gate toggle.
LIVE_PARAMS: dict = {
    "symbol": "BTC/USDT:USDT",
    "strategy": {
        **LOCKED,
        "use_mtf_4h_gate": True,
        "mtf_4h_ema_period": 200,
    },
}

# 90-day stage-2 window. Use the most-recent slice that's fully cached.
STAGE2_START = "2025-04-01"
STAGE2_END = "2025-06-30"

# Stage-1 cross-check window (shortest OOS slice with reasonable trade count).
STAGE1_START = "2024-01-01"
STAGE1_END = "2024-06-30"

PARQ_15M = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
PARQ_4H = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
OUT = ROOT / "reports" / "multifactor_4h_parity_validation.json"


# ---------------------------------------------------------------------------
# Stage 1 — impl cross-check (new in-class gate vs old subclass)
# ---------------------------------------------------------------------------

def _trade_signatures(stats) -> list[tuple]:
    """Build a canonical, hashable trade signature for diffing.

    Uses (EntryTime, ExitTime, Size sign, EntryPrice rounded) — robust across
    columns that vary by backtesting.py version. Two strategies are
    behaviourally identical iff trade signatures match.
    """
    df = getattr(stats, "_trades", None)
    if df is None or len(df) == 0:
        return []
    out = []
    for _, r in df.iterrows():
        out.append((
            str(r["EntryTime"]),
            str(r["ExitTime"]),
            int(np.sign(float(r["Size"]))),
            round(float(r["EntryPrice"]), 2),
        ))
    return out


def stage1_impl_cross_check() -> dict:
    print(f"[validate stage 1] window={STAGE1_START}..{STAGE1_END}", file=sys.stderr)

    df = _load_slice(PARQ_15M, STAGE1_START, STAGE1_END, attach_funding=True)

    # Path A: new in-class gate via DayTradeMultiFactorBTC with use_mtf_4h_gate=True.
    config_a = {**LOCKED, "use_mtf_4h_gate": True, "mtf_4h_ema_period": 200,
                "mtf_4h_parquet_path": str(PARQ_4H)}
    bt_a = Backtest(df, DayTradeMultiFactorBTC, cash=CASH, commission=COMMISSION,
                     margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                     finalize_trades=True)
    stats_a = bt_a.run(**config_a)
    sigs_a = _trade_signatures(stats_a)

    # Path B: existing MultiFactorMTF4H subclass from the deepening tool. It
    # reads `self.data.Ema4h` (attached by the runner), so we replicate that.
    df_b = df.copy()
    df_b["Ema4h"] = _build_4h_ema_aligned(
        df_b.index, PARQ_4H, ema_period=200,
    )
    bt_b = Backtest(df_b, MultiFactorMTF4H, cash=CASH, commission=COMMISSION,
                     margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                     finalize_trades=True)
    stats_b = bt_b.run(**LOCKED)
    sigs_b = _trade_signatures(stats_b)

    only_a = sorted(set(sigs_a) - set(sigs_b))
    only_b = sorted(set(sigs_b) - set(sigs_a))
    common = sorted(set(sigs_a) & set(sigs_b))

    result = {
        "stage": 1,
        "window": [STAGE1_START, STAGE1_END],
        "trades_path_a_in_class": len(sigs_a),
        "trades_path_b_subclass": len(sigs_b),
        "trades_common": len(common),
        "trades_only_in_path_a": only_a[:50],   # cap diff dump
        "trades_only_in_path_b": only_b[:50],
        "stats_a_return_pct": float(stats_a.get("Return [%]", 0.0) or 0.0),
        "stats_b_return_pct": float(stats_b.get("Return [%]", 0.0) or 0.0),
        "verdict": "PASS" if (not only_a and not only_b) else "FAIL",
    }
    print(
        f"  trades A={len(sigs_a)} B={len(sigs_b)} common={len(common)}  "
        f"verdict={result['verdict']}",
        file=sys.stderr,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 2 — live evaluator vs backtest per-bar parity
# ---------------------------------------------------------------------------

class _BarRecorder(DayTradeMultiFactorBTC):
    """Subclass that records, for EVERY bar in the window, whether
    _long_signal / _short_signal would return True if asked.

    We don't piggyback on the parent's next() because the parent only calls
    _long_signal/_short_signal when there's no active position — so trades
    that are open during a bar would suppress the recording, biasing the
    parity-check sample. We probe both signal functions unconditionally
    here at every bar.
    """

    def init(self) -> None:
        super().init()
        self._fired_long: list[int] = []
        self._fired_short: list[int] = []

    def next(self) -> None:
        i = len(self.data) - 1
        # Probe both directions unconditionally. The parent's signal methods
        # have no side effects beyond returning a bool.
        try:
            if super()._long_signal(i):
                self._fired_long.append(i)
        except Exception:  # noqa: BLE001
            pass
        try:
            if super()._short_signal(i):
                self._fired_short.append(i)
        except Exception:  # noqa: BLE001
            pass


def stage2_live_vs_backtest() -> dict:
    print(f"[validate stage 2] window={STAGE2_START}..{STAGE2_END}", file=sys.stderr)
    df = _load_slice(PARQ_15M, STAGE2_START, STAGE2_END, attach_funding=True)
    print(f"  bars={len(df)}", file=sys.stderr)

    # --- Backtest path: collect signal bars ---
    config_bt = {**LOCKED, "use_mtf_4h_gate": True, "mtf_4h_ema_period": 200,
                  "mtf_4h_parquet_path": str(PARQ_4H)}
    bt = Backtest(df, _BarRecorder, cash=CASH, commission=COMMISSION,
                   margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                   finalize_trades=True)
    stats = bt.run(**config_bt)
    strat = stats._strategy  # backtesting.py exposes the strategy instance
    bt_long = set(strat._fired_long)
    bt_short = set(strat._fired_short)
    print(f"  backtest: long_fires={len(bt_long)} short_fires={len(bt_short)}",
          file=sys.stderr)

    # --- Live path: per-bar evaluate_signal with injected bars_4h ---
    # Pre-load 4H bars covering the window plus 180-day warmup.
    bars_4h_full = pd.read_parquet(PARQ_4H)
    if bars_4h_full.index.tz is not None:
        bars_4h_full.index = bars_4h_full.index.tz_localize(None)
    cutoff_lo = pd.Timestamp(STAGE2_START) - pd.Timedelta(days=180)
    bars_4h_window = bars_4h_full.loc[cutoff_lo:].copy()

    warm = (max(LOCKED["mf_trend_ema_period"], LOCKED["volume_ma_period"],
                LOCKED["rsi_period"]) + 5)
    live_long: set[int] = set()
    live_short: set[int] = set()

    t0 = time.time()
    for i in range(warm, len(df)):
        bars_slice = df.iloc[: i + 1]
        funding_v = float(bars_slice["Funding"].iloc[-1]) if "Funding" in bars_slice.columns else 0.0
        # Only pass 4H bars whose CLOSE time is <= current 15m bar's timestamp.
        # The live evaluator's _compute_4h_ema_at_15m_close enforces this too,
        # but slicing here keeps the live path's EMA from accidentally peeking
        # at a bar that closed after the 15m bar's wall clock.
        cur_ts = bars_slice.index[-1]
        cur_4h = bars_4h_window.loc[: cur_ts]
        side, _dbg = evaluate_signal(bars_slice, funding_v, LIVE_PARAMS,
                                      bars_4h=cur_4h)
        if side == "long":
            live_long.add(i)
        elif side == "short":
            live_short.add(i)
        if i % 500 == 0:
            print(f"    bar {i}/{len(df)} t={time.time()-t0:.1f}s", file=sys.stderr)

    print(f"  live:     long_fires={len(live_long)} short_fires={len(live_short)} "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)

    # Per-bar comparison: each bar i is "agree" iff bt-decision matches live-decision.
    # 4-state per bar: (bt_long, bt_short, live_long, live_short)
    n_total = len(df) - warm
    n_agree = 0
    mismatches: list[dict] = []
    for i in range(warm, len(df)):
        bt_dec = "long" if i in bt_long else ("short" if i in bt_short else None)
        lv_dec = "long" if i in live_long else ("short" if i in live_short else None)
        if bt_dec == lv_dec:
            n_agree += 1
        else:
            if len(mismatches) < 50:  # cap dump
                mismatches.append({
                    "bar_idx": i,
                    "ts": str(df.index[i]),
                    "backtest": bt_dec,
                    "live": lv_dec,
                })

    agreement_pct = 100.0 * n_agree / max(n_total, 1)
    verdict = "PASS" if agreement_pct >= PARITY_GATE else "FAIL"

    return {
        "stage": 2,
        "window": [STAGE2_START, STAGE2_END],
        "warmup_bars": warm,
        "evaluable_bars": n_total,
        "bars_in_agreement": n_agree,
        "agreement_pct": round(agreement_pct, 4),
        "parity_gate_pct": PARITY_GATE,
        "backtest_long_fires": len(bt_long),
        "backtest_short_fires": len(bt_short),
        "live_long_fires": len(live_long),
        "live_short_fires": len(live_short),
        "mismatches_sample": mismatches,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("multifactor-v1 + 4H EMA200 gate — parity validation")
    print("=" * 78)

    out: dict = {"strategy": "multifactor-v1+4h-gate"}

    out["stage1_impl_cross_check"] = stage1_impl_cross_check()
    out["stage2_live_vs_backtest"] = stage2_live_vs_backtest()

    overall = (
        out["stage1_impl_cross_check"]["verdict"] == "PASS"
        and out["stage2_live_vs_backtest"]["verdict"] == "PASS"
    )
    out["overall_verdict"] = "PASS" if overall else "FAIL"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print()
    print(f"stage 1: {out['stage1_impl_cross_check']['verdict']}")
    print(f"stage 2: {out['stage2_live_vs_backtest']['verdict']}  "
          f"({out['stage2_live_vs_backtest']['agreement_pct']}%)")
    print(f"overall: {out['overall_verdict']}")
    print(f"saved   → {OUT}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
