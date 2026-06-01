"""Phase 2 friction + sizing audit for cnh-hybrid-short-v1.

Working config (locked by Phase 1):
    dedup=15, sl=1.5×ATR, tp=EMA(100), entry=EMA(24), tf=4h

Two questions:

1.  FRICTION STRESS — existing backtest used FRICTION_BPS=10 round-trip
    (mega_sweep). Live taker on Binance Futures USDM is ~8 bps round-trip;
    a +5 bps slippage stress brings effective friction to 15 bps. Does the
    OOS edge survive?

2.  SIZING REALITY — at $50/leg with 1.5% risk-per-trade and 1.5×ATR stop,
    position notional = $50 × 0.015 / (1.5 × atr_pct). With BTC's ATR(14, 4h)
    typically 1-3% of close, this often falls below Binance's $50
    min-notional AND/OR 0.001 BTC min-qty (= $60-$100 at current price).
    What % of OOS signals get skipped? Does raising risk to 2.5% or equity to
    $80 fix it?

Gate (per HYBRID_SHORT_PLAN.md Phase 2):
    after-fee edge per trade ≥ 30 bps  AND  min-size skip rate ≤ 30%
    (or have a documented, accepted mitigation).

Run:
    uv run python tools/hybrid_friction_sizing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
from tools.icnh_mega_sweep import (  # noqa: E402
    Config,
    FRICTION_BPS,
    WINDOWS,
    load_tf,
    simulate_trades,
)

OOS_LABELS = {w[0] for w in WINDOWS[8:]}
RESULTS_PATH = ROOT / "data" / "hybrid_friction_sizing_results.json"

# Locked from Phase 1.
TF = "4h"
DEDUP = 15
SL_ATR = 1.5
TP_EMAS = ("ema100",)
ENTRY_EMAS = ("ema24",)

# Exchange constraints (from exchange/constraints.py).
MIN_NOTIONAL_USDT = 50.0
MIN_QTY_BTC = 0.001

# Sizing scenarios to probe.
SIZING_SCENARIOS = [
    {"equity": 50.0,  "risk_pct": 1.5},   # current capital plan
    {"equity": 50.0,  "risk_pct": 2.5},   # raise risk
    {"equity": 80.0,  "risk_pct": 1.5},   # top up capital
    {"equity": 100.0, "risk_pct": 1.5},   # whole-account scenario
]

# Friction scenarios (bps round-trip, replacing the existing 10 bps in
# net_pct). Existing live taker is ~8 bps; +5 slippage = 13; conservative
# stress at 15.
FRICTION_SCENARIOS = [8.0, 10.0, 13.0, 15.0]

GATE_MIN_EDGE_BPS = 30.0
GATE_MAX_SKIP_PCT = 30.0


def _collect_oos_trades_with_atr() -> list[dict]:
    """Run the locked HYBRID config on OOS windows and return one dict per
    trade with the extras needed for sizing analysis (atr_pct, entry_price)."""
    df = load_tf(TF)
    dt_cfg = Config(
        name="hybrid_dt", pattern_type="distribution_top", direction="short", tf=TF,
        uptrend_bars=16, chop_bars=8, min_rise_pct=2.5, max_chop_ratio=0.55,
        require_chop_at_top=True, breakdown_mode="chop_low_or_ema24",
        sl_atr_mult=SL_ATR, regime_sl_mode="off", tp_emas=TP_EMAS,
        entry_emas=ENTRY_EMAS, dedup_bars=DEDUP,
    )
    icnh_cfg = Config(
        name="hybrid_icnh", pattern_type="inverse_cnh", direction="short", tf=TF,
        cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
        handle_max_depth_frac=0.70, peak_tolerance=6,
        entry_emas=ENTRY_EMAS, sl_atr_mult=SL_ATR, regime_sl_mode="off",
        tp_emas=TP_EMAS, dedup_bars=DEDUP,
    )

    out: list[dict] = []
    for label, start, end in WINDOWS:
        if label not in OOS_LABELS:
            continue
        sub = df.loc[start:end]
        if len(sub) < 100:
            continue
        hits = find_hybrid_patterns(sub, dt_cfg, icnh_cfg)
        dt_idxs = [h for h, src in hits if src == "DT"]
        icnh_idxs = [h for h, src in hits if src == "ICNH"]
        # We need the same entry_idx logic that simulate_trades uses to look up
        # atr14 at entry. Easiest: call simulate_trades and then look up each
        # trade by entry_ts.
        for t in simulate_trades(sub, dt_idxs, dt_cfg, label):
            t["pattern"] = "DT"
            out.append(t)
        for t in simulate_trades(sub, icnh_idxs, icnh_cfg, label):
            t["pattern"] = "ICNH"
            out.append(t)

    # Now annotate each trade with atr_pct at entry. simulate_trades uses
    # `df.iloc[entry_idx]["atr14"]`; we replay that lookup by entry_ts.
    oos_starts = [s for label, s, _e in WINDOWS if label in OOS_LABELS]
    oos_ends = [e for label, _s, e in WINDOWS if label in OOS_LABELS]
    df_oos = df.loc[min(oos_starts):max(oos_ends)]
    by_ts = df_oos[["close", "atr14"]]
    for t in out:
        ts = pd.Timestamp(t["entry_ts"])
        if ts in by_ts.index:
            row = by_ts.loc[ts]
            atr = float(row["atr14"])
            close = float(row["close"])
            t["atr_at_entry"] = atr
            t["atr_pct_at_entry"] = atr / close if close > 0 else float("nan")
    return out


def _net_pct_with_friction(t: dict, friction_bps: float) -> float:
    """Recompute net_pct from gross with replacement friction."""
    gross = t["gross_pct"]
    return gross - friction_bps / 10_000.0


def _skip_reason(notional: float, qty_btc: float) -> str:
    if notional < MIN_NOTIONAL_USDT:
        return "min_notional"
    if qty_btc < MIN_QTY_BTC:
        return "min_qty"
    return ""


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+6.2f}%"


def _fmt_bps(x: float) -> str:
    return f"{x * 10_000:+6.1f} bps"


def main() -> int:
    print("=" * 78)
    print("HYBRID SHORT — Phase 2 friction + sizing")
    print(f"Locked config: dedup={DEDUP}, sl={SL_ATR}xATR, tp={TP_EMAS[0]}, "
          f"entry={ENTRY_EMAS[0]}")
    print(f"OOS windows:   {sorted(OOS_LABELS)}")
    print("=" * 78)

    trades = _collect_oos_trades_with_atr()
    if not trades:
        print("No OOS trades collected — abort.")
        return 1

    n = len(trades)
    atr_pcts = np.array([t.get("atr_pct_at_entry", np.nan) for t in trades])
    atr_pcts = atr_pcts[np.isfinite(atr_pcts)]
    print(f"\nOOS trades: {n}")
    print(f"ATR%(14, 4h) at entry — distribution:")
    print(f"  count   {len(atr_pcts)}")
    if len(atr_pcts):
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            print(f"  q={q:.2f}  {atr_pcts.quantile(q) if hasattr(atr_pcts,'quantile') else np.quantile(atr_pcts, q):.4f}  "
                  f"({np.quantile(atr_pcts, q) * 100:.2f}%)")

    # ---- 1) Friction stress ----
    print("\n" + "-" * 78)
    print("Friction stress (existing backtest = 10 bps; live taker ≈ 8 bps;")
    print("13 bps = live + 5 bps slippage; 15 bps = stress)")
    print("-" * 78)
    print(f"{'friction':<12}{'cum':>10}{'mean/trade':>14}{'WR':>8}{'gate':>8}")
    friction_rows = []
    for fr in FRICTION_SCENARIOS:
        nets = np.array([_net_pct_with_friction(t, fr) for t in trades])
        cum = float(np.prod(1.0 + nets) - 1.0)
        mean = float(nets.mean())
        wr = float((nets > 0).mean())
        edge_pass = (mean * 10_000) >= GATE_MIN_EDGE_BPS
        row = {
            "friction_bps": fr, "trades": n, "cum": cum,
            "mean_pct": mean, "win_rate": wr,
            "edge_bps": mean * 10_000, "gate_edge_pass": edge_pass,
        }
        friction_rows.append(row)
        print(
            f"{fr:>6.1f} bps  {_fmt_pct(cum):>10}  {_fmt_bps(mean):>14}  "
            f"{wr*100:>5.1f}%  {('PASS' if edge_pass else 'FAIL'):>8}"
        )

    # ---- 2) Sizing reality — skip-rate per (equity, risk_pct) ----
    print("\n" + "-" * 78)
    print("Sizing reality — skip rate by (equity, risk%) at min-notional $50 / "
          "min-qty 0.001 BTC")
    print("-" * 78)
    print(f"{'equity':>8}{'risk%':>8}{'trades':>8}{'skipped':>10}"
          f"{'skip%':>8}{'min_not':>10}{'min_qty':>10}")
    sizing_rows = []
    for sc in SIZING_SCENARIOS:
        equity = sc["equity"]
        risk_pct = sc["risk_pct"]
        skipped = 0
        skip_min_not = 0
        skip_min_qty = 0
        for t in trades:
            atr_pct = t.get("atr_pct_at_entry")
            entry_price = t.get("entry_price")
            if not (atr_pct and entry_price and np.isfinite(atr_pct) and atr_pct > 0):
                # No ATR → can't size → treat as skip.
                skipped += 1
                continue
            sl_distance_pct = SL_ATR * atr_pct
            notional = equity * (risk_pct / 100.0) / sl_distance_pct
            qty_btc = notional / entry_price
            reason = _skip_reason(notional, qty_btc)
            if reason:
                skipped += 1
                if reason == "min_notional":
                    skip_min_not += 1
                else:
                    skip_min_qty += 1
        skip_pct = skipped / n * 100
        sizing_rows.append({
            "equity": equity, "risk_pct": risk_pct, "skipped": skipped,
            "skip_pct": skip_pct, "skip_min_notional": skip_min_not,
            "skip_min_qty": skip_min_qty,
        })
        print(
            f"{equity:>8.1f}{risk_pct:>8.2f}{n:>8}{skipped:>10}{skip_pct:>7.1f}%"
            f"{skip_min_not:>10}{skip_min_qty:>10}"
        )

    # ---- 3) Combined edge AFTER min-size skips ----
    print("\n" + "-" * 78)
    print("Combined: OOS cum AFTER min-size skips, at 13 bps friction (live+5slip)")
    print("-" * 78)
    print(f"{'equity':>8}{'risk%':>8}{'kept':>8}{'cum_kept':>12}{'edge/trade':>14}")
    combined_rows = []
    fr_target = 13.0
    for sc in SIZING_SCENARIOS:
        equity = sc["equity"]
        risk_pct = sc["risk_pct"]
        kept_nets: list[float] = []
        for t in trades:
            atr_pct = t.get("atr_pct_at_entry")
            entry_price = t.get("entry_price")
            if not (atr_pct and entry_price and np.isfinite(atr_pct) and atr_pct > 0):
                continue
            sl_distance_pct = SL_ATR * atr_pct
            notional = equity * (risk_pct / 100.0) / sl_distance_pct
            qty_btc = notional / entry_price
            if _skip_reason(notional, qty_btc):
                continue
            kept_nets.append(_net_pct_with_friction(t, fr_target))
        if not kept_nets:
            print(f"{equity:>8.1f}{risk_pct:>8.2f}{0:>8}  no kept trades")
            continue
        kept_nets_a = np.array(kept_nets)
        cum = float(np.prod(1.0 + kept_nets_a) - 1.0)
        mean = float(kept_nets_a.mean())
        combined_rows.append({
            "equity": equity, "risk_pct": risk_pct,
            "kept_trades": len(kept_nets),
            "kept_cum": cum, "kept_edge_bps": mean * 10_000,
        })
        print(
            f"{equity:>8.1f}{risk_pct:>8.2f}{len(kept_nets):>8}  "
            f"{_fmt_pct(cum):>10}  {_fmt_bps(mean):>14}"
        )

    # ---- Verdict ----
    edge_pass_at_15 = next(r["gate_edge_pass"] for r in friction_rows if r["friction_bps"] == 15.0)
    # Find any sizing scenario where skip% ≤ 30 AND kept-edge ≥ 30 bps.
    sizing_pass = []
    for s, c in zip(sizing_rows, combined_rows):
        if (s["skip_pct"] <= GATE_MAX_SKIP_PCT and
                c["kept_edge_bps"] >= GATE_MIN_EDGE_BPS):
            sizing_pass.append((s["equity"], s["risk_pct"]))
    print("\n" + "=" * 78)
    print("PHASE 2 VERDICT")
    print("=" * 78)
    print(
        f"  Friction stress at 15 bps: "
        f"{'PASS' if edge_pass_at_15 else 'FAIL'} "
        f"(edge ≥ {GATE_MIN_EDGE_BPS} bps required)"
    )
    if sizing_pass:
        print(
            f"  Sizing scenarios that survive (skip ≤ {GATE_MAX_SKIP_PCT}% AND "
            f"kept edge ≥ {GATE_MIN_EDGE_BPS} bps): {sizing_pass}"
        )
        overall = "PASS"
    else:
        print(
            f"  NO sizing scenario survives both gates "
            f"(skip ≤ {GATE_MAX_SKIP_PCT}% AND edge ≥ {GATE_MIN_EDGE_BPS} bps)"
        )
        overall = "FAIL" if not edge_pass_at_15 else "MITIGATION REQUIRED"
    print(f"  Overall: {overall}")
    print("=" * 78)

    RESULTS_PATH.write_text(json.dumps({
        "trades": n,
        "atr_pct_quantiles": (
            {f"q{int(q*100)}": float(np.quantile(atr_pcts, q))
             for q in (0.10, 0.25, 0.50, 0.75, 0.90)}
            if len(atr_pcts) else {}
        ),
        "friction": friction_rows,
        "sizing": sizing_rows,
        "combined_13bps": combined_rows,
        "verdict": overall,
    }, indent=2, default=str))
    print(f"Saved → {RESULTS_PATH}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
