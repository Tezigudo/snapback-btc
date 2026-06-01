"""SOL hybrid-short risk-sweep — prove the 'crank risk to chase 4-5%/mo' failure.

Re-runs the SAME 1-position SOL backtest at risk_per_trade in {2.75,5,8,11}% and
shows: final equity, CAGR, max DD (from peak), whether the -35.5% kill switch
trips (and when), mean monthly return, % months >=+4%, % flat (no-signal)
months, worst single trade, effective leverage. Builds an HTML equity overlay.

Key mechanic: with risk-based sizing, a stopped-out trade loses ~= risk% of
equity. So DD scales ~linearly with risk; no-signal months stay flat at ANY risk.

Run: uv run --with plotly python tools/sol_risk_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.build_sol_report import (load_trades, enforce_one_position,  # noqa: E402
                                    START_EQUITY, SL_ATR_MULT, LEVERAGE,
                                    MIN_NOTIONAL, MIN_QTY, KILL_FRAC, IS_END)


def sim(df: pd.DataFrame, risk: float):
    eq = START_EQUITY
    kill_floor = START_EQUITY * KILL_FRAC
    curve = [(df["EntryTime"].iloc[0] - pd.Timedelta(days=1), eq)]
    killed = None
    notionals, worst = [], 0.0
    for _, t in df.iterrows():
        if killed:
            break
        sl_frac = SL_ATR_MULT * float(t["atr_pct"])
        if not np.isfinite(sl_frac) or sl_frac <= 0:
            continue
        notional = min(risk / sl_frac * eq, eq * LEVERAGE * 0.95)
        price = float(t["EntryPrice"]); qty = notional / price
        if qty < MIN_QTY or notional < MIN_NOTIONAL:
            continue
        notionals.append(notional / eq)
        pnl = notional * float(t["net_pct"])
        worst = min(worst, pnl / eq * 100)
        eq += pnl
        curve.append((t["ExitTime"], eq))
        if eq <= kill_floor:
            killed = str(t["ExitTime"])[:10]
    s = pd.Series(dict(curve)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    peak = s.cummax(); max_dd = float((s / peak - 1).min() * 100)
    days = (s.index[-1] - s.index[0]).days
    cagr = (s.iloc[-1] / START_EQUITY) ** (365.25 / days) * 100 - 100 if days > 0 else 0
    daily = s.resample("1D").last().ffill()
    mret = daily.resample("ME").last().pct_change().dropna() * 100
    oos = mret[mret.index > IS_END]
    return {
        "risk": risk * 100, "final": s.iloc[-1], "total": (s.iloc[-1] / START_EQUITY - 1) * 100,
        "cagr": cagr, "max_dd": max_dd, "killed": killed,
        "eff_lev": float(np.median(notionals)) if notionals else 0,
        "worst_trade": worst,
        "mo_mean": float(mret.mean()), "mo_oos_mean": float(oos.mean()) if len(oos) else 0,
        "mo_ge4": float((mret >= 4).mean() * 100), "mo_flat": float((mret.abs() < 0.01).mean() * 100),
        "curve": s,
    }


def main() -> int:
    raw = load_trades(); clean, _ = enforce_one_position(raw)
    rows = [sim(clean, r) for r in (0.0275, 0.05, 0.08, 0.11)]

    print(f"SOL hybrid-short risk sweep — $100 start, 1-position, kill -35.5%\n")
    h = (f"{'risk':>5}{'final$':>9}{'CAGR':>8}{'maxDD':>8}{'kill':>12}"
         f"{'effLev':>8}{'worstTr':>9}{'mo_mean':>9}{'mo_OOS':>8}{'>=4%mo':>8}{'flat':>7}")
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['risk']:>4.1f}%{r['final']:>9.0f}{r['cagr']:>7.1f}%{r['max_dd']:>7.1f}%"
              f"{(r['killed'] or 'no'):>12}{r['eff_lev']:>7.2f}x{r['worst_trade']:>8.1f}%"
              f"{r['mo_mean']:>8.2f}%{r['mo_oos_mean']:>7.2f}%{r['mo_ge4']:>7.0f}%{r['mo_flat']:>6.0f}%")

    print("\nReading:")
    print(" - worstTr = biggest single-trade equity hit. It scales ~1:1 with risk%")
    print("   (risk-based sizing: a stopped trade loses ~= risk% of equity).")
    print(" - effLev stays ~0.4-1.7x even at 11% risk -> liquidation is NOT the binding")
    print("   risk here; the KILL SWITCH / drawdown is. (Correcting the earlier claim.)")
    print(" - flat months stay ~35% at EVERY risk level -> cranking risk does NOTHING")
    print("   for the third of months with no signal. You can't size your way into a")
    print("   month the pattern didn't fire.")
    print(f" - +4.5%/mo target = +70%/yr. Even the highest risk that survives gets")
    print(f"   nowhere near it without tripping the kill switch.")

    # equity overlay
    fig = go.Figure()
    for r in rows:
        tag = f"risk {r['risk']:.1f}% (DD {r['max_dd']:.0f}%" + (f", KILLED {r['killed']})" if r['killed'] else ")")
        fig.add_trace(go.Scatter(x=r["curve"].index, y=r["curve"].values, name=tag))
    fig.add_hline(y=START_EQUITY * KILL_FRAC, line_dash="dash", line_color="red",
                  annotation_text="kill switch -35.5%")
    fig.update_layout(title="SOL hybrid-short — equity by risk per trade ($100 start)",
                      template="plotly_white", height=560,
                      yaxis=dict(title="equity $", range=[40, 200]))
    out = ROOT / "reports" / "SOL_RISK_SWEEP.html"
    fig.write_html(out)
    print(f"\nchart -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
