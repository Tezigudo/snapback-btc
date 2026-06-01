"""Pump-fade backtest: short the day's top gainer after it rolls over, TP at
the pre-pump support, stop above the peak.

Survivorship-safe (data via tools/pumpfade_data, incl. delisted coins).

PRE-REGISTERED RULES (one defensible default each — not a grid to fish in):

  SELECTION (daily, across the whole USDT-perp universe, point-in-time)
    - day_ret = close[D]/close[D-1]-1 >= THRESH               (default 0.40)
    - day-D quote volume >= QVOL_MIN                          (default $20M)
    - keep the top-N gainers that day by day_ret              (default 3)

  ENTRY ("wait for the dropped zone" = rollover confirmation; intraday TF)
    - track the running peak from day D onward
    - trigger: first bar that CLOSES below the prior-K-bar low (default K=6)
    - FILL at the NEXT bar's open  (no look-ahead)
    - if no trigger within ENTRY_WAIT hours of the peak -> SKIP (don't short strength)

  STOP  = peak_high_at_entry * (1 + SL_BUF)                   (default +6%)
          filled with HEAVY adverse slippage (stop fires into a continuation)
  TP    = pre-pump base = close BASE_DAYS before D            (default 3 days)
          require TP < entry (room to fall) else SKIP
  TIME  = exit at market after MAX_HOLD hours                 (default 168h = 7d)
  SETTLE= intraday bars run out / multi-day gap -> exit at last close (delisting)

  FRICTION (the part that decides the verdict)
    - taker FEE both sides                                    (default 5bps)
    - entry/exit slippage scaled by illiquidity
    - STOP slippage is large and asymmetric                   (default +1.5%, scaled)
    - real funding accrued every 8h while short (pumped coin: short usually RECEIVES)

Reports EV/trade AND win%, the LOSS TAIL (worst trade), the equity path (max DD)
under fixed-fractional risk sizing, and splits delisted vs still-trading cohorts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_data as pfd  # noqa: E402

CACHE = ROOT / "data" / "pumpfade"


@dataclass
class Params:
    thresh: float = 0.40          # min daily close-to-close return to qualify
    qvol_min: float = 20e6        # min day-D quote volume (USD), liquidity floor
    top_n: int = 3                # keep top-N gainers per day
    entry_tf: str = "1h"          # intraday timeframe for entry/exit walk
    break_k: int = 6              # rollover trigger: close < low of prior K bars
    select_mode: str = "close"    # "close" = day close-to-close ret; "intraday_high" =
                                  # high[D]/close[D-1] (rolling-24h leaderboard; captures
                                  # intraday spikes that FADE back below thresh by close —
                                  # the user's actual idea, "every 4-8hr")
    entry_wait_h: int = 48        # give up waiting for rollover after this many h post-peak
    require_vol_drying: bool = False  # user's rule: only short if volume is DRYING into entry
    vol_dry_max: float = 0.8      # skip if entry-vol / peak-bar-vol >= this (volume rising = continuation)
    sl_buf: float = 0.06          # stop = peak*(1+sl_buf)
    stop_mode: str = "peak"       # "peak" = above pump peak; "local" = above local swing-high
                                  # (the user's chart: stop above the lower-high); "atr" = entry+mult*ATR
    local_k: int = 12             # bars back for the local swing high (stop_mode="local")
    atr_mult: float = 2.5         # stop = entry + atr_mult*ATR (stop_mode="atr")
    max_stop_pct: float = 0.0     # if >0, cap stop at entry*(1+max_stop_pct) — "never risk more than this"
    base_days: int = 3            # TP = close this many days before the pump day
    max_hold_h: int = 168         # time-stop
    cooldown_days: int = 0        # dedup: skip a symbol's event within N days of a prior kept event
    entry_after_close: bool = True  # POINT-IN-TIME: selection uses close[D], so entry can only
                                    # fire from D+1 00:00 (else 18% of entries peek at close[D])
    gap_days: float = 2.0         # post-entry bar gap > this => treat as delisted/settle
    # friction
    fee: float = 0.0005           # taker, per side
    slip_entry: float = 0.0010    # base entry slippage
    slip_exit: float = 0.0010     # base time/settle exit slippage
    slip_stop: float = 0.0150     # STOP slippage (adverse, into a spike)
    illiq_ref: float = 50e6       # qvol at/above which slippage is at base; below scales up
    illiq_max: float = 4.0        # cap on the illiquidity slippage multiplier
    # sizing (for the equity path only; per-trade returns are leverage-agnostic)
    risk_pct: float = 0.02        # fraction of equity risked to the stop per trade
    start_equity: float = 1000.0


def illiq_mult(qvol: float, p: Params) -> float:
    """Slippage multiplier: 1x for liquid names, up to illiq_max for thin ones."""
    if qvol <= 0:
        return p.illiq_max
    return float(min(p.illiq_max, max(1.0, p.illiq_ref / qvol)))


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    symbol: str
    day: pd.Timestamp        # UTC day D of the pump (daily bar open_time)
    day_ret: float
    qvol: float
    base_level: float        # TP target (pre-pump support)
    prev_close: float        # close[D-1] — reference for the intraday threshold crossing
    last_active: pd.Timestamp  # last daily bar with volume>0 (delisting proxy)
    is_delisted: bool


def find_symbol_events(symbol: str, daily: pd.DataFrame, p: Params) -> list[Event]:
    if daily.empty or len(daily) < p.base_days + 2:
        return []
    df = daily.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    prev_close = df["close"].shift(1)
    close_ret = df["close"] / prev_close - 1.0
    hi_ret = df["high"] / prev_close - 1.0           # intraday spike (rolling-24h leaderboard)
    sel_ret = hi_ret if p.select_mode == "intraday_high" else close_ret
    base = df["close"].shift(p.base_days)  # close BASE_DAYS before D = prior support
    active = df.index[df["volume"] > 0]
    last_active = active.max() if len(active) else df.index.max()
    out: list[Event] = []
    qualify = (sel_ret >= p.thresh) & (df["quote_volume"] >= p.qvol_min) & base.notna() & prev_close.notna()
    for ts in df.index[qualify.fillna(False)]:
        out.append(Event(
            symbol=symbol, day=ts, day_ret=float(sel_ret.loc[ts]),
            qvol=float(df.loc[ts, "quote_volume"]), base_level=float(base.loc[ts]),
            prev_close=float(prev_close.loc[ts]),
            last_active=last_active, is_delisted=False,  # set in collect_events (needs panel end)
        ))
    return out


def rank_top_n_per_day(events: list[Event], p: Params) -> list[Event]:
    """Across the universe, keep only the top-N gainers each UTC day."""
    by_day: dict[pd.Timestamp, list[Event]] = {}
    for e in events:
        by_day.setdefault(e.day.normalize(), []).append(e)
    kept: list[Event] = []
    for _day, evs in by_day.items():
        evs.sort(key=lambda e: e.day_ret, reverse=True)
        kept.extend(evs[: p.top_n])
    kept.sort(key=lambda e: e.day)
    return kept


# --------------------------------------------------------------------------- #
# Trade simulation (one event) — pure given the bars
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    symbol: str
    day: pd.Timestamp
    entry_time: pd.Timestamp | None = None
    entry: float = np.nan
    exit_time: pd.Timestamp | None = None
    exit: float = np.nan
    reason: str = "SKIP"          # ENTRY reasons: TP / STOP / TIME / SETTLE / SKIP_*
    peak: float = np.nan
    stop: float = np.nan
    tp: float = np.nan
    gross_ret: float = 0.0        # short return on notional, before fees/funding
    funding_ret: float = 0.0      # + = received
    net_ret: float = 0.0
    mae: float = 0.0              # max adverse excursion (price up, against short)
    hold_h: float = 0.0
    qvol: float = np.nan
    day_ret: float = np.nan
    vol_ratio: float = np.nan     # entry-volume / peak-bar-volume (low = drying/exhaustion)
    is_delisted: bool = False


def simulate_trade(ev: Event, intr: pd.DataFrame, funding: pd.DataFrame, p: Params) -> Trade:
    t = Trade(symbol=ev.symbol, day=ev.day, peak=np.nan, tp=ev.base_level,
              qvol=ev.qvol, day_ret=ev.day_ret, is_delisted=ev.is_delisted)
    if intr.empty:
        t.reason = "SKIP_NODATA"
        return t
    bars = intr[intr["volume"] > 0].sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    if len(bars) < p.break_k + 3:
        t.reason = "SKIP_NODATA"
        return t

    day_start = ev.day.normalize()
    # Search starts at day D; the peak is the running max high from D onward.
    walk = bars.loc[day_start:]
    if len(walk) < p.break_k + 3:
        t.reason = "SKIP_NODATA"
        return t

    tf_h = {"1h": 1, "15m": 0.25, "5m": 1 / 12, "4h": 4}.get(p.entry_tf, 1)
    idx = walk.index
    highs = walk["high"].values
    lows = walk["low"].values
    closes = walk["close"].values
    opens = walk["open"].values
    vols = walk["volume"].values

    # --- find rollover entry ---
    # Point-in-time entry gate — the trade only becomes knowable once selection fires:
    #  - close mode: selection uses close[D] (known at D 23:59 UTC) => entry from D+1.
    #    (Leaving it on day D is an 18%-of-trades look-ahead that makes results look
    #    WORSE, not better.)
    #  - intraday_high mode (the user's rolling-24h idea): the coin becomes a "top
    #    gainer" the moment its price first crosses thresh above close[D-1]; you can
    #    act from THAT bar — no need to wait for the daily close.
    if p.select_mode == "intraday_high":
        thr_price = ev.prev_close * (1 + p.thresh)
        crossed = walk.index[walk["high"].values >= thr_price]
        entry_min = crossed.min() if len(crossed) else (day_start + pd.Timedelta(days=1))
    else:
        entry_min = (day_start + pd.Timedelta(days=1)) if p.entry_after_close else day_start
    peak_so_far = -np.inf
    peak_time = idx[0]
    peak_i = 0
    entry_i = None
    for i in range(len(walk)):
        if highs[i] > peak_so_far:
            peak_so_far = highs[i]
            peak_time = idx[i]
            peak_i = i
        # only consider entry once we have K prior bars, past the peak, and >= D+1
        if i >= p.break_k and i > 0 and idx[i] >= entry_min:
            prior_low = lows[i - p.break_k:i].min()
            hours_since_peak = (idx[i] - peak_time) / pd.Timedelta(hours=1)
            if closes[i] < prior_low and idx[i] > peak_time:
                if hours_since_peak <= p.entry_wait_h and i + 1 < len(walk):
                    entry_i = i + 1   # fill next bar open
                    break
                if hours_since_peak > p.entry_wait_h:
                    break             # rolled over too late; stop looking
    t.peak = float(peak_so_far)
    if entry_i is None:
        t.reason = "SKIP_NOROLL"
        return t

    entry_open = opens[entry_i]
    illiq = illiq_mult(ev.qvol, p)
    entry_fill = entry_open * (1 - p.slip_entry * illiq)   # shorting: filled slightly lower

    # USER'S RULE (money-validated): only short if volume is DRYING into entry. Volume
    # rising into the rollover = the market is still actively buying = continuation/
    # squeeze risk (the KAT failure + the -208% backtest tail). Skip those.
    peak_vol = float(vols[peak_i]) if peak_i < len(vols) else 0.0
    lo3 = max(0, entry_i - 3)
    entry_vol = float(vols[lo3:entry_i].mean()) if entry_i > lo3 else float(vols[entry_i])
    vol_ratio = (entry_vol / peak_vol) if peak_vol > 0 else 1.0
    t.vol_ratio = float(vol_ratio)
    if p.require_vol_drying and vol_ratio >= p.vol_dry_max:
        t.reason = "SKIP_VOLRISING"
        return t
    if p.stop_mode == "local":
        # Stop above the LOCAL swing high (the lower-high in the user's chart), not
        # the far pump peak — a structure-aware, tighter stop.
        lo = max(0, entry_i - p.local_k)
        local_high = highs[lo:entry_i].max() if entry_i > lo else peak_so_far
        stop_price = float(local_high) * (1 + p.sl_buf)
    elif p.stop_mode == "atr":
        # Classic volatility stop: entry + atr_mult * ATR(14) at entry.
        tr = (walk["high"].values - walk["low"].values)[:entry_i]
        atr = float(tr[-14:].mean()) if len(tr) else 0.0
        stop_price = entry_fill + p.atr_mult * atr
    else:  # "peak"
        stop_price = peak_so_far * (1 + p.sl_buf)
    if p.max_stop_pct > 0:
        # Cap the stop distance: never risk more than max_stop_pct above entry. A
        # stale peak far above a deep-rollover entry would otherwise make a stop-out
        # a >100%-of-notional loss (the fat tail that sinks the naive version).
        stop_price = min(stop_price, entry_fill * (1 + p.max_stop_pct))
    tp_price = ev.base_level
    t.entry_time = idx[entry_i]
    t.entry = float(entry_fill)
    t.stop = float(stop_price)

    if not (tp_price < entry_fill):           # no room to fall to support
        t.reason = "SKIP_NOEDGE"
        return t

    # --- walk forward from entry to resolution ---
    max_adverse = entry_fill
    prev_time = idx[entry_i]
    exit_fill = exit_time = None
    reason = None
    for j in range(entry_i, len(walk)):
        ts = idx[j]
        gap_d = (ts - prev_time) / pd.Timedelta(days=1)
        if gap_d > p.gap_days and j > entry_i:           # data gap => delisting/halt
            exit_fill = closes[j - 1] * (1 + p.slip_exit * illiq)
            exit_time = idx[j - 1]
            reason = "SETTLE"
            break
        prev_time = ts
        max_adverse = max(max_adverse, highs[j])
        held_h = (ts - t.entry_time) / pd.Timedelta(hours=1)
        # stop checked first (intrabar: a bar that hits both is treated as stop = conservative)
        if highs[j] >= stop_price:
            exit_fill = stop_price * (1 + p.slip_stop * illiq)   # buy-to-cover into a spike
            exit_time = ts
            reason = "STOP"
            break
        if lows[j] <= tp_price:
            exit_fill = tp_price                                  # resting limit cover
            exit_time = ts
            reason = "TP"
            break
        if held_h >= p.max_hold_h:
            exit_fill = closes[j] * (1 + p.slip_exit * illiq)
            exit_time = ts
            reason = "TIME"
            break
    if exit_fill is None:                                         # ran out of bars
        exit_fill = closes[-1] * (1 + p.slip_exit * illiq)
        exit_time = idx[-1]
        reason = "SETTLE"

    t.exit = float(exit_fill)
    t.exit_time = exit_time
    t.reason = reason
    t.hold_h = float((exit_time - t.entry_time) / pd.Timedelta(hours=1))
    t.mae = float(max_adverse / entry_fill - 1.0)               # +% = went against the short

    # --- funding while short (positive rate => short RECEIVES) ---
    f_ret = 0.0
    if not funding.empty:
        fwin = funding.loc[t.entry_time:exit_time]
        f_ret = float(fwin["funding_rate"].sum())               # fraction of notional
    t.funding_ret = f_ret

    # --- PnL (short): profit when exit < entry ---
    gross = (entry_fill - exit_fill) / entry_fill
    fees = p.fee * 2
    t.gross_ret = float(gross)
    t.net_ret = float(gross - fees + f_ret)
    return t


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def equity_path(trades: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Sequential fixed-fractional-risk equity curve (risk_pct to the stop)."""
    eq = p.start_equity
    rows = []
    for _, tr in trades.sort_values("entry_time").iterrows():
        stop_dist = (tr["stop"] - tr["entry"]) / tr["entry"]    # >0, short stop is above
        if stop_dist <= 0 or not np.isfinite(stop_dist):
            continue
        notional = (eq * p.risk_pct) / stop_dist
        pnl = notional * tr["net_ret"]
        eq += pnl
        rows.append({"entry_time": tr["entry_time"], "equity": eq, "pnl": pnl})
        if eq <= 0:
            break
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, p: Params, label: str = "") -> dict:
    taken = trades[trades["reason"].isin(["TP", "STOP", "TIME", "SETTLE"])].copy()
    n = len(taken)
    if n == 0:
        return {"label": label, "n_events": len(trades), "n_taken": 0}
    nr = taken["net_ret"]
    wins = taken[nr > 0]
    ep = equity_path(taken, p)
    if not ep.empty:
        eqs = ep["equity"].values
        peak = np.maximum.accumulate(eqs)
        max_dd = float(((eqs - peak) / peak).min())
        final_eq = float(eqs[-1])
    else:
        max_dd, final_eq = 0.0, p.start_equity
    return {
        "label": label,
        "n_events": int(len(trades)),
        "n_taken": int(n),
        "win_pct": round(100 * len(wins) / n, 1),
        "ev_net_pct": round(100 * nr.mean(), 3),
        "median_net_pct": round(100 * nr.median(), 3),
        "sum_net_pct": round(100 * nr.sum(), 1),
        "worst_trade_pct": round(100 * nr.min(), 1),
        "best_trade_pct": round(100 * nr.max(), 1),
        "mean_funding_pct": round(100 * taken["funding_ret"].mean(), 3),
        "stop_rate_pct": round(100 * (taken["reason"] == "STOP").mean(), 1),
        "tp_rate_pct": round(100 * (taken["reason"] == "TP").mean(), 1),
        "mean_mae_pct": round(100 * taken["mae"].mean(), 1),
        "final_equity": round(final_eq, 0),
        "max_dd_pct": round(100 * max_dd, 1),
    }


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in trades])


# --------------------------------------------------------------------------- #
# Full universe study
# --------------------------------------------------------------------------- #
def collect_events(p: Params, universe: list[str] | None = None) -> list[Event]:
    if universe is None:
        uf = CACHE / "universe.json"
        universe = json.loads(uf.read_text()) if uf.exists() else pfd.enumerate_universe()
    allev: list[Event] = []
    for sym in universe:
        path = pfd.KLINES_DIR / f"{sym}_1d.parquet"
        if not path.exists():
            continue
        try:
            daily = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        allev.extend(find_symbol_events(sym, daily, p))
    # A coin counts as "delisted/inactive" if its last active day is well before
    # the freshest coin in the panel (i.e., it stopped trading). Computed here,
    # not per-symbol, because flat zero-volume placeholder tails push a delisted
    # coin's raw data end all the way to the present.
    if allev:
        panel_end = max(e.last_active for e in allev)
        cutoff = panel_end - pd.Timedelta(days=21)
        for e in allev:
            e.is_delisted = bool(e.last_active < cutoff)
    kept = rank_top_n_per_day(allev, p)
    if p.cooldown_days > 0:
        # One sustained pump must not count as N separate trades (a coin can stay
        # a top-gainer for several days running). Keep the first event per symbol,
        # then skip any within cooldown_days of the last kept event for that symbol.
        kept.sort(key=lambda e: e.day)
        last_kept: dict[str, pd.Timestamp] = {}
        deduped: list[Event] = []
        for e in kept:
            prev = last_kept.get(e.symbol)
            if prev is not None and (e.day - prev) <= pd.Timedelta(days=p.cooldown_days):
                continue
            deduped.append(e)
            last_kept[e.symbol] = e.day
        kept = deduped
    return kept


def simulate_event(ev: Event, p: Params) -> Trade:
    months = pfd.months_spanning(
        (ev.day - timedelta(days=2)).to_pydatetime(),
        (ev.day + timedelta(days=p.max_hold_h / 24 + 4)).to_pydatetime(),
    )
    try:
        intr = pfd.load_intraday(ev.symbol, p.entry_tf, months)
        fund = pfd.load_funding(ev.symbol, months)
    except Exception:  # noqa: BLE001
        return Trade(symbol=ev.symbol, day=ev.day, reason="SKIP_NODATA",
                     qvol=ev.qvol, day_ret=ev.day_ret, is_delisted=ev.is_delisted)
    return simulate_trade(ev, intr, fund, p)


def run_study(p: Params, universe: list[str] | None = None, workers: int = 12,
              date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    events = collect_events(p, universe)
    if date_from:
        lo = pd.Timestamp(date_from, tz="UTC")
        events = [e for e in events if e.day >= lo]
    if date_to:
        hi = pd.Timestamp(date_to, tz="UTC")
        events = [e for e in events if e.day <= hi]
    print(f"events after rank/top-{p.top_n}: {len(events)} (thresh {p.thresh:.0%}, qvol>=${p.qvol_min/1e6:.0f}M)",
          flush=True)
    # Group by symbol so each symbol's intraday parquet cache is read-modify-written
    # by exactly one thread (no same-file races); parallelise ACROSS symbols.
    by_sym: dict[str, list[Event]] = {}
    for e in events:
        by_sym.setdefault(e.symbol, []).append(e)

    def do_symbol(evs: list[Event]) -> list[Trade]:
        return [simulate_event(e, p) for e in evs]

    trades: list[Trade] = []
    done_syms = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_symbol, evs) for evs in by_sym.values()]
        for fut in as_completed(futs):
            trades.extend(fut.result())
            done_syms += 1
            if done_syms % 50 == 0:
                print(f"  simulated {done_syms}/{len(by_sym)} symbols, {len(trades)} trades", flush=True)
    return trades_to_df(trades)


def full_report(trades: pd.DataFrame, p: Params) -> dict:
    rep = {"overall": summarize(trades, p, "ALL")}
    taken = trades[trades["reason"].isin(["TP", "STOP", "TIME", "SETTLE"])].copy()
    # skip-reason census
    rep["skips"] = trades["reason"].value_counts().to_dict()
    # cohort: delisted vs still-trading
    rep["delisted"] = summarize(trades[trades["is_delisted"]], p, "delisted")
    rep["surviving"] = summarize(trades[~trades["is_delisted"]], p, "surviving")
    # by year
    if not taken.empty:
        taken["year"] = pd.to_datetime(taken["entry_time"]).dt.year
        rep["by_year"] = {}
        for y, grp in taken.groupby("year"):
            sub = trades[trades.index.isin(grp.index)]
            rep["by_year"][int(y)] = summarize(sub, p, str(y))
    return rep


def print_report(rep: dict) -> None:
    def line(d: dict) -> str:
        if not d or d.get("n_taken", 0) == 0:
            return f"{d.get('label',''):<10} n_taken=0 (events={d.get('n_events','?')})"
        return (f"{d['label']:<10} n={d['n_taken']:<4} win {d['win_pct']:>5}%  "
                f"EV {d['ev_net_pct']:>+7}%  med {d['median_net_pct']:>+7}%  "
                f"sum {d['sum_net_pct']:>+8}%  worst {d['worst_trade_pct']:>+7}%  "
                f"stop {d['stop_rate_pct']:>5}%  tp {d['tp_rate_pct']:>5}%  "
                f"fund {d['mean_funding_pct']:>+6}%  finEq {d['final_equity']}  DD {d['max_dd_pct']}%")
    print("\n================ PUMP-FADE STUDY ================")
    print(line(rep["overall"]))
    print("\n-- cohort (survivorship check) --")
    print(line(rep["delisted"]))
    print(line(rep["surviving"]))
    print("\n-- by year --")
    for y in sorted(rep.get("by_year", {})):
        print(line(rep["by_year"][y]))
    print("\n-- event/skip census --")
    for k, v in sorted(rep["skips"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:<14} {v}")


# --------------------------------------------------------------------------- #
# Vertical slice CLI (hand-picked events) — validate before scaling
# --------------------------------------------------------------------------- #
def _run_one(symbol: str, day: str, p: Params) -> Trade:
    daily = pfd.load_daily(symbol)
    evs = find_symbol_events(symbol, daily, p)
    target = pd.Timestamp(day, tz="UTC").normalize()
    ev = next((e for e in evs if e.day.normalize() == target), None)
    if ev is None:
        # build it directly even if it didn't pass the filter, for inspection
        df = daily.sort_index()
        if target not in df.index:
            return Trade(symbol=symbol, day=target, reason="SKIP_NODAY")
        i = df.index.get_loc(target)
        base = df["close"].iloc[i - p.base_days] if i >= p.base_days else np.nan
        ev = Event(symbol, target, float(df["close"].iloc[i] / df["close"].iloc[i - 1] - 1),
                   float(df["quote_volume"].iloc[i]), float(base),
                   float(df["close"].iloc[i - 1]),
                   df.index[df["volume"] > 0].max(), False)
    months = pfd.months_spanning((ev.day - timedelta(days=2)).to_pydatetime(),
                                 (ev.day + timedelta(days=p.max_hold_h / 24 + 4)).to_pydatetime())
    intr = pfd.load_intraday(symbol, p.entry_tf, months)
    fund = pfd.load_funding(symbol, months)
    return simulate_trade(ev, intr, fund, p)


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="store_true", help="run the hand-picked vertical slice")
    ap.add_argument("--study", action="store_true", help="run the full universe study")
    ap.add_argument("--symbol")
    ap.add_argument("--day")
    ap.add_argument("--thresh", type=float)
    ap.add_argument("--qvol-min", type=float)
    ap.add_argument("--top-n", type=int)
    ap.add_argument("--date-from")
    ap.add_argument("--date-to")
    ap.add_argument("--tag", default="study")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    p = Params()
    if args.thresh is not None:
        p.thresh = args.thresh
    if args.qvol_min is not None:
        p.qvol_min = args.qvol_min
    if args.top_n is not None:
        p.top_n = args.top_n

    if args.study:
        df = run_study(p, workers=args.workers, date_from=args.date_from, date_to=args.date_to)
        out = CACHE / f"trades_{args.tag}.parquet"
        df.to_parquet(out)
        rep = full_report(df, p)
        print_report(rep)
        (CACHE / f"report_{args.tag}.json").write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nsaved trades -> {out}\nsaved report -> {CACHE / f'report_{args.tag}.json'}")
        return 0

    if args.symbol and args.day:
        t = _run_one(args.symbol, args.day, p)
        for k, v in t.__dict__.items():
            print(f"  {k}: {v}")
        return 0

    if args.slice:
        # PORTAL (user's example) + FTT (delisted/FTX collapse) + a few known pumps
        picks = [
            ("PORTALUSDT", "2025-07-11"),
            ("FTTUSDT", "2022-11-10"),
        ]
        rows = []
        for sym, day in picks:
            t = _run_one(sym, day, p)
            rows.append(t)
            print(f"\n=== {sym} {day} ===")
            print(f"  day_ret {t.day_ret*100:+.1f}%  qvol ${t.qvol/1e6:.0f}M  peak {t.peak:.6g}")
            print(f"  reason={t.reason}")
            if t.entry_time is not None:
                print(f"  entry {t.entry:.6g} @ {t.entry_time}  stop {t.stop:.6g}  tp {t.tp:.6g}")
                print(f"  exit  {t.exit:.6g} @ {t.exit_time}  hold {t.hold_h:.0f}h  MAE {t.mae*100:+.1f}%")
                print(f"  gross {t.gross_ret*100:+.2f}%  funding {t.funding_ret*100:+.2f}%  NET {t.net_ret*100:+.2f}%")
        df = trades_to_df(rows)
        print("\n", summarize(df, p, "slice"))
        return 0

    print("use --slice or --symbol X --day YYYY-MM-DD")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
