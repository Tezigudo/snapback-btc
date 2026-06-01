"""SOL x cnh-hybrid-short — Phase 2-5 promotion checks.

Reuses the EXACT detector + simulator + window split from the cross-coin gate
(no parallel harness). Adds, for SOL specifically:

  Phase 2  friction stress   — does OOS edge survive 10/15/20/25/30 bps?
  Phase 3  sizing floor      — skip rate vs SOL's real exchange minimums
                               (min-notional $5, min-qty 0.01 SOL) at several
                               per-leg equities. BTC needed $100/leg; SOL?
  Phase 4  diversification    — correlation of SOL hybrid-short per-window
                               returns vs BTC hybrid-short (the existing leg).
  Phase 5  dedup choice       — OOS gate at dedup_bars 5/10/15.

Run: uv run python tools/sol_hybrid_phase2to5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.icnh_mega_sweep as megasweep  # noqa: E402  (mutate FRICTION_BPS)
from tools.icnh_mega_sweep import Config, simulate_trades, WINDOWS  # noqa: E402
from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
from tools.cross_coin_backtest import (  # noqa: E402
    IS_LABELS,
    OOS_LABELS,
    _aggregate,
    _load_coin_4h,
)

# SOL real Binance USDT-M perp minimums (fetched live 2026-05-30).
SOL_MIN_NOTIONAL = 5.0
SOL_MIN_QTY = 0.01
SOL_QTY_STEP = 0.01
RISK_PCT = 2.75          # the leg's deploy risk (params_cnh_hybrid_short.yaml)
SL_ATR_MULT = 1.5        # hybrid-short stop distance


def _cfgs(dedup: int) -> tuple[Config, Config]:
    dt = Config(
        name="hybrid_dt", pattern_type="distribution_top", direction="short",
        tf="4h", uptrend_bars=16, chop_bars=8, min_rise_pct=2.5,
        max_chop_ratio=0.55, require_chop_at_top=True,
        breakdown_mode="chop_low_or_ema24",
        sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
        entry_emas=("ema24",), dedup_bars=dedup,
    )
    icnh = Config(
        name="hybrid_icnh", pattern_type="inverse_cnh", direction="short",
        tf="4h", cup_len=20, handle_len=4, min_r2=0.50,
        min_cup_depth_atr=1.0, handle_max_depth_frac=0.70, peak_tolerance=6,
        entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
        tp_emas=("ema100",), dedup_bars=dedup,
    )
    return dt, icnh


def run_windows(df: pd.DataFrame, dedup: int) -> list[dict]:
    """Per-window stats for SOL at a given dedup. Mirrors the gate runner."""
    dt_cfg, icnh_cfg = _cfgs(dedup)
    out: list[dict] = []
    for label, start, end in WINDOWS:
        sub = df.loc[start:end]
        if len(sub) < 100:
            out.append({"window": label, "no_data": True, "ret_pct": 0.0,
                        "trades": 0})
            continue
        hits = find_hybrid_patterns(sub, dt_cfg, icnh_cfg)
        dt_idxs = [h for h, src in hits if src == "DT"]
        icnh_idxs = [h for h, src in hits if src == "ICNH"]
        trades = (simulate_trades(sub, dt_idxs, dt_cfg, label)
                  + simulate_trades(sub, icnh_idxs, icnh_cfg, label))
        if not trades:
            out.append({"window": label, "ret_pct": 0.0, "trades": 0,
                        "sharpe": 0.0, "_trades": []})
            continue
        nets = np.array([t["net_pct"] for t in trades])
        cum = float(np.prod(1.0 + nets) - 1.0)
        out.append({
            "window": label, "trades": len(trades),
            "win_rate_pct": float((nets > 0).mean() * 100.0),
            "cum": cum, "ret_pct": cum * 100.0,
            "sharpe": float(nets.mean() / nets.std() * np.sqrt(250))
                      if nets.std() > 0 else 0.0,
            "_trades": trades,
        })
    return out


def gate(oos: dict) -> str:
    n = int(oos.get("n_windows", 0))
    if n == 0:
        return "n/a"
    ok = (float(oos.get("cum_ret_pct", 0)) > 0
          and int(oos.get("positive_windows", 0)) >= n / 2
          and float(oos.get("worst_window_pct", 0)) > -15.0)
    return "PASS" if ok else "FAIL"


def main() -> int:
    df = _load_coin_4h("SOL")
    if df.empty:
        print("no SOL data")
        return 1

    # ---------- Phase 2: friction stress ----------
    print("=" * 64)
    print("PHASE 2 — friction stress (SOL, dedup=15, OOS gate)")
    print("=" * 64)
    base_bps = megasweep.FRICTION_BPS
    for bps in (10, 15, 20, 25, 30):
        megasweep.FRICTION_BPS = float(bps)
        pw = run_windows(df, dedup=15)
        oos = _aggregate(pw, OOS_LABELS)
        nets = [t["net_pct"] for w in pw for t in w.get("_trades", [])
                if w["window"] in OOS_LABELS]
        ev = float(np.mean(nets) * 100) if nets else 0.0
        print(f"  {bps:>2}bps: OOS cum={oos['cum_ret_pct']:+7.2f}%  "
              f"{oos['positive_windows']}/{oos['n_windows']} pos  "
              f"worst={oos['worst_window_pct']:+6.2f}%  "
              f"EV/trade={ev:+.2f}%  -> {gate(oos)}")
    megasweep.FRICTION_BPS = base_bps

    # ---------- Phase 5: dedup choice ----------
    print("\n" + "=" * 64)
    print("PHASE 5 — dedup choice (SOL, 10bps, OOS gate)")
    print("=" * 64)
    dedup_runs = {}
    for dedup in (5, 10, 15):
        pw = run_windows(df, dedup=dedup)
        dedup_runs[dedup] = pw
        oos = _aggregate(pw, OOS_LABELS)
        is_a = _aggregate(pw, IS_LABELS)
        print(f"  dedup={dedup:>2}: OOS cum={oos['cum_ret_pct']:+7.2f}%  "
              f"{oos['positive_windows']}/{oos['n_windows']} pos  "
              f"worst={oos['worst_window_pct']:+6.2f}%  "
              f"trades={oos['trades']:>2}  medSh={oos['median_sharpe']:+.1f}  "
              f"(IS cum={is_a['cum_ret_pct']:+.1f}%) -> {gate(oos)}")

    # ---------- Phase 3: sizing floor vs SOL exchange minimums ----------
    print("\n" + "=" * 64)
    print("PHASE 3 — sizing floor (SOL min-notional $5, min-qty 0.01 SOL)")
    print("  risk 2.75%, SL=1.5xATR; notional = (equity*risk)/(1.5*atr_pct)")
    print("=" * 64)
    # Use dedup=15 OOS trades; look up atr% at each entry from the 4h frame.
    atr_pct = (df["atr14"] / df["close"])
    pw = dedup_runs[15]
    oos_trades = [t for w in pw for t in w.get("_trades", [])
                  if w["window"] in OOS_LABELS]
    for equity in (25.0, 50.0, 100.0, 200.0):
        kept = skipped_notional = skipped_qty = 0
        for t in oos_trades:
            entry_ts = pd.Timestamp(t["entry_ts"])
            price = float(t["entry_price"])
            ap = float(atr_pct.get(entry_ts, np.nan))
            if not np.isfinite(ap) or ap <= 0:
                continue
            notional = (equity * RISK_PCT / 100.0) / (SL_ATR_MULT * ap)
            qty = notional / price
            qty = np.floor(qty / SOL_QTY_STEP) * SOL_QTY_STEP
            if qty < SOL_MIN_QTY:
                skipped_qty += 1
            elif notional < SOL_MIN_NOTIONAL:
                skipped_notional += 1
            else:
                kept += 1
        n = len(oos_trades)
        pct = 100.0 * kept / n if n else 0.0
        print(f"  ${equity:>5.0f}/leg: kept {kept:>2}/{n}  ({pct:4.0f}%)  "
              f"skip[min-notional={skipped_notional}, min-qty={skipped_qty}]")

    # ---------- Phase 4: diversification vs BTC hybrid-short ----------
    print("\n" + "=" * 64)
    print("PHASE 4 — diversification: SOL vs BTC hybrid-short window returns")
    print("=" * 64)
    btc = _load_coin_4h("BTC")
    pw_btc = run_windows(btc, dedup=15)
    sol_by_w = {w["window"]: w.get("ret_pct", 0.0) for w in dedup_runs[15]
                if not w.get("no_data")}
    btc_by_w = {w["window"]: w.get("ret_pct", 0.0) for w in pw_btc
                if not w.get("no_data")}
    common = [w for _, w in [(0, x[0]) for x in WINDOWS]
              if w in sol_by_w and w in btc_by_w]
    sol_r = np.array([sol_by_w[w] for w in common])
    btc_r = np.array([btc_by_w[w] for w in common])
    if len(common) >= 3 and sol_r.std() > 0 and btc_r.std() > 0:
        corr = float(np.corrcoef(sol_r, btc_r)[0, 1])
    else:
        corr = float("nan")
    print(f"  windows compared: {len(common)}")
    print(f"  SOL vs BTC hybrid-short per-window return corr = {corr:+.2f}")
    print("  (both are the SAME signal on different coins; high corr expected.")
    print("   real diversification vs the LONG book — v1/donchian — is the")
    print("   point. This corr just bounds how independent the SOL leg's P&L is")
    print("   from the BTC short leg already in the book.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
