"""Experiment (b): stronger rollover confirmation to avoid catching CONTINUATIONS.

Pre-registered single change vs default:
  Entry still requires close < prior-K-bar low (the existing rollover trigger),
  AND ADDITIONALLY the trigger bar must CLOSE BELOW a trailing EMA of intraday
  closes (EMA span = CONFIRM_EMA bars). Rationale: a single-bar undercut of the
  prior-K low during an ongoing pump is often noise inside a continuation; demanding
  the price also be below its own short trailing mean is a genuine trend-break
  confirmation. Everything else identical to Params() default (+6% peak stop, etc).

We monkeypatch simulate_trade's entry loop by re-implementing it here against the
same cached bars/funding, so friction/exit logic is byte-identical to the audited
core (we import and call the same stop/TP/funding/PnL block).

ONE pre-registered EMA span (24 bars = ~1 day on 1h). No sweep. Screen OOS first.
"""
from __future__ import annotations
import sys
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_data as pfd          # noqa: E402
import pumpfade_backtest as pf       # noqa: E402

OOS = pd.Timestamp("2025-01-01", tz="UTC")
CONFIRM_EMA = 24   # pre-registered: trigger bar must close below 24-bar EMA


def simulate_trade_confirmed(ev, intr, funding, p, confirm_ema=CONFIRM_EMA):
    """Identical to pf.simulate_trade except the entry trigger additionally
    requires close[i] < EMA(close, confirm_ema)[i]. All exit/friction/PnL code
    is copied verbatim from pf.simulate_trade (kept in sync by inspection)."""
    t = pf.Trade(symbol=ev.symbol, day=ev.day, peak=np.nan, tp=ev.base_level,
                 qvol=ev.qvol, day_ret=ev.day_ret, is_delisted=ev.is_delisted)
    if intr.empty:
        t.reason = "SKIP_NODATA"; return t
    bars = intr[intr["volume"] > 0].sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    if len(bars) < p.break_k + 3:
        t.reason = "SKIP_NODATA"; return t
    day_start = ev.day.normalize()
    walk = bars.loc[day_start:]
    if len(walk) < p.break_k + 3:
        t.reason = "SKIP_NODATA"; return t

    idx = walk.index
    highs = walk["high"].values; lows = walk["low"].values
    closes = walk["close"].values; opens = walk["open"].values
    if confirm_ema is None:        # fidelity mode: disable the gate (close<+inf always True)
        ema = np.full(len(closes), np.inf)
    else:
        ema = pd.Series(closes).ewm(span=confirm_ema, adjust=False).mean().values  # trailing, causal

    peak_so_far = -np.inf; peak_time = idx[0]; entry_i = None
    for i in range(len(walk)):
        if highs[i] > peak_so_far:
            peak_so_far = highs[i]; peak_time = idx[i]
        if i >= p.break_k and i > 0:
            prior_low = lows[i - p.break_k:i].min()
            hours_since_peak = (idx[i] - peak_time) / pd.Timedelta(hours=1)
            # ---- THE ONLY CHANGE: add EMA confirmation ----
            confirmed = (closes[i] < prior_low) and (closes[i] < ema[i])
            if confirmed and idx[i] > peak_time:
                if hours_since_peak <= p.entry_wait_h and i + 1 < len(walk):
                    entry_i = i + 1; break
                if hours_since_peak > p.entry_wait_h:
                    break
    t.peak = float(peak_so_far)
    if entry_i is None:
        t.reason = "SKIP_NOROLL"; return t

    entry_open = opens[entry_i]
    illiq = pf.illiq_mult(ev.qvol, p)
    entry_fill = entry_open * (1 - p.slip_entry * illiq)
    stop_price = peak_so_far * (1 + p.sl_buf)
    if p.max_stop_pct > 0:
        stop_price = min(stop_price, entry_fill * (1 + p.max_stop_pct))
    tp_price = ev.base_level
    t.entry_time = idx[entry_i]; t.entry = float(entry_fill); t.stop = float(stop_price)
    if not (tp_price < entry_fill):
        t.reason = "SKIP_NOEDGE"; return t

    max_adverse = entry_fill; prev_time = idx[entry_i]
    exit_fill = exit_time = None; reason = None
    for j in range(entry_i, len(walk)):
        ts = idx[j]
        gap_d = (ts - prev_time) / pd.Timedelta(days=1)
        if gap_d > p.gap_days and j > entry_i:
            exit_fill = closes[j - 1] * (1 + p.slip_exit * illiq); exit_time = idx[j - 1]
            reason = "SETTLE"; break
        prev_time = ts
        max_adverse = max(max_adverse, highs[j])
        held_h = (ts - t.entry_time) / pd.Timedelta(hours=1)
        if highs[j] >= stop_price:
            exit_fill = stop_price * (1 + p.slip_stop * illiq); exit_time = ts
            reason = "STOP"; break
        if lows[j] <= tp_price:
            exit_fill = tp_price; exit_time = ts; reason = "TP"; break
        if held_h >= p.max_hold_h:
            exit_fill = closes[j] * (1 + p.slip_exit * illiq); exit_time = ts
            reason = "TIME"; break
    if exit_fill is None:
        exit_fill = closes[-1] * (1 + p.slip_exit * illiq); exit_time = idx[-1]; reason = "SETTLE"

    t.exit = float(exit_fill); t.exit_time = exit_time; t.reason = reason
    t.hold_h = float((exit_time - t.entry_time) / pd.Timedelta(hours=1))
    t.mae = float(max_adverse / entry_fill - 1.0)
    f_ret = 0.0
    if not funding.empty:
        fwin = funding.loc[t.entry_time:exit_time]
        f_ret = float(fwin["funding_rate"].sum())
    t.funding_ret = f_ret
    gross = (entry_fill - exit_fill) / entry_fill
    t.gross_ret = float(gross); t.net_ret = float(gross - p.fee * 2 + f_ret)
    return t


def run(p, confirm_ema=CONFIRM_EMA, also_short_hold=False):
    events = pf.collect_events(p)
    by_sym = {}
    for e in events:
        by_sym.setdefault(e.symbol, []).append(e)
    rows = []
    for sym, evs in by_sym.items():
        for ev in evs:
            months = pfd.months_spanning(
                (ev.day - timedelta(days=2)).to_pydatetime(),
                (ev.day + timedelta(days=p.max_hold_h / 24 + 4)).to_pydatetime())
            try:
                intr = pfd.load_intraday(ev.symbol, p.entry_tf, months)
                fund = pfd.load_funding(ev.symbol, months)
            except Exception:
                rows.append(pf.Trade(symbol=ev.symbol, day=ev.day, reason="SKIP_NODATA",
                                     qvol=ev.qvol, day_ret=ev.day_ret, is_delisted=ev.is_delisted)); continue
            rows.append(simulate_trade_confirmed(ev, intr, fund, p, confirm_ema))
    return pf.trades_to_df(rows)


def report(df, label):
    taken = df[df.reason.isin(["TP","STOP","TIME","SETTLE"])].copy()
    taken["et"] = pd.to_datetime(taken["entry_time"], utc=True)
    def s(sub, tag):
        if len(sub)==0: print(f"  {tag:<5} n=0"); return
        nr=sub.net_ret
        print(f"  {tag:<5} n={len(sub):<4} win {100*(nr>0).mean():5.1f}%  EV {100*nr.mean():+7.2f}%  "
              f"med {100*nr.median():+7.2f}%  worst {100*nr.min():+7.1f}%  stop% {100*(sub.reason=='STOP').mean():5.1f}  "
              f"tp% {100*(sub.reason=='TP').mean():5.1f}")
    print(f"\n### {label}")
    s(taken, "ALL"); s(taken[taken.et<OOS], "IS"); s(taken[taken.et>=OOS], "OOS")
    cen = df.reason.value_counts().to_dict()
    print("  skips:", {k:cen[k] for k in cen})


if __name__ == "__main__":
    p = pf.Params()
    df = run(p, CONFIRM_EMA)
    df.to_parquet(ROOT / "data" / "pumpfade" / "trades_expb_ema24.parquet")
    report(df, f"(b) rollover + close<EMA{CONFIRM_EMA}  (default +6% stop)")
    # Also report the combo (b)+(c) shorter hold 72h, and (b)+top1, using the SAME parquet's events?
    # No: short-hold changes exit, must re-run. Do 72h here too (cheap, cached).
    p2 = pf.Params(max_hold_h=72)
    df2 = run(p2, CONFIRM_EMA)
    df2.to_parquet(ROOT / "data" / "pumpfade" / "trades_expb_ema24_hold72.parquet")
    report(df2, f"(b)+(c) rollover + close<EMA{CONFIRM_EMA} + max_hold=72h")
