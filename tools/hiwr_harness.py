"""High-win-rate research harness (goal 2026-05-30).

Goal: find a low-risk strategy that fires ~1-3 signals/day with a high win
rate on BTC / ETH / PAXG. This module is the engine + a signal library +
a CLI for: validate (against backtest.py), probe (one config), grid, and
walk-forward.

Design rails (from advisor):
  - ONE position at a time (matches the live bot; overlapping entries inflate
    WR + expectancy — the project's phase3-overfire bug).
  - Next-bar-OPEN entry (signal evaluated on a CLOSED bar t, fill at open[t+1]).
  - Intrabar exit: SL checked BEFORE TP (conservative tie-break) using the
    bar's high/low; time-stop on max-hold bars; exit at close on time-stop.
  - Fees+slippage = 5 bps/side (10 bps round-trip), the project baseline.
  - SELECT on OOS expectancy + worst-window robustness, NOT win rate.
    Win rate is a *reported, filtered* constraint, never the objective.

No look-ahead: every indicator is computed on the full series then the entry
decision at bar t uses only values known at the close of bar t; the fill is at
t+1 open. Exits scan forward bar-by-bar.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr, ema, rsi, sma  # noqa: E402

# Project friction baseline: backtest.py COMMISSION_PER_SIDE = 0.0005.
COMMISSION_PER_SIDE = 0.0005

DATA = ROOT / "data" / "historical"

# PATH2 5-window OOS protocol used across the project.
PATH2_WINDOWS = [
    ("2022H1", "2022-01-01", "2022-07-01"),
    ("2023H1", "2023-01-01", "2023-07-01"),
    ("2024H1", "2024-01-01", "2024-07-01"),
    ("2024H2", "2024-07-01", "2025-01-01"),
    ("2025H1", "2025-01-01", "2025-07-01"),
]


def load(symbol: str, tf: str) -> pd.DataFrame:
    """symbol like 'BTC', 'ETH', 'PAXG'; tf like '15m','1h','4h'."""
    p = DATA / f"{symbol}_USDT_USDT_{tf}.parquet"
    df = pd.read_parquet(p)
    return df[["open", "high", "low", "close", "volume"]].copy()


def bars_per_day(tf: str) -> float:
    return {"15m": 96.0, "1h": 24.0, "4h": 6.0}[tf]


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    side: str
    entry_i: int
    entry_t: pd.Timestamp
    entry_px: float
    exit_t: pd.Timestamp
    exit_px: float
    bars_held: int
    reason: str  # 'sl' | 'tp' | 'time' | 'trail'
    ret_gross: float  # signed fractional move in price direction
    ret_net: float    # after round-trip commission
    risk_frac: float = 0.0  # initial-stop distance as fraction of entry price (1R)


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)
    n_signals: int = 0  # raw signals before 1-position suppression

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.ret_net > 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")

    @property
    def expectancy(self) -> float:
        """Mean NET fractional return per trade (on price, pre-leverage)."""
        return float(np.mean([t.ret_net for t in self.trades])) if self.n else float("nan")

    @property
    def profit_factor(self) -> float:
        gains = sum(t.ret_net for t in self.trades if t.ret_net > 0)
        losses = -sum(t.ret_net for t in self.trades if t.ret_net < 0)
        return gains / losses if losses > 0 else float("inf")

    @property
    def avg_win(self) -> float:
        w = [t.ret_net for t in self.trades if t.ret_net > 0]
        return float(np.mean(w)) if w else 0.0

    @property
    def avg_loss(self) -> float:
        ls = [t.ret_net for t in self.trades if t.ret_net < 0]
        return float(np.mean(ls)) if ls else 0.0

    @property
    def max_consec_losses(self) -> int:
        m = c = 0
        for t in self.trades:
            if t.ret_net <= 0:
                c += 1
                m = max(m, c)
            else:
                c = 0
        return m

    def signals_per_day(self, tf: str, span_days: float) -> float:
        return self.n / span_days if span_days > 0 else float("nan")

    def equity_curve_R(self, risk_frac: float) -> np.ndarray:
        """Compounded equity if each trade risks `risk_frac` of equity at the SL
        distance, scaled to the actual realised net return.  Approximates the
        live fixed-fractional sizing. Returns array starting at 1.0."""
        eq = [1.0]
        for t in self.trades:
            # position sized so that a move to SL = risk_frac loss; realised
            # pnl frac = (ret_net / sl_dist) * risk_frac, but we don't carry
            # sl_dist here, so caller uses max_dd_simple instead. Kept for ref.
            eq.append(eq[-1] * (1 + t.ret_net))
        return np.array(eq)

    def max_dd(self) -> float:
        """Max drawdown of the simple (unleveraged, 1x notional) equity curve."""
        eq = self.equity_curve_R(0.0)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return float(dd.min()) if len(dd) else 0.0


def run_engine(
    df: pd.DataFrame,
    long_sig: pd.Series,
    short_sig: pd.Series,
    *,
    sl_frac: float | None = None,
    tp_frac: float | None = None,
    sl_atr: float | None = None,
    tp_atr: float | None = None,
    atr_period: int = 14,
    max_hold: int = 24,
    commission_side: float = COMMISSION_PER_SIDE,
    tp_mean: pd.Series | None = None,
    trail_atr: float | None = None,
    breakeven_atr: float | None = None,
) -> Result:
    """Event-step the engine. Exactly one of (sl_frac/tp_frac) or (sl_atr/tp_atr)
    defines the stop geometry. `tp_mean` optionally overrides the fixed TP with a
    dynamic target series (e.g. mean-reversion to a moving average) evaluated at
    entry — used for "exit at the mean" strategies.

    long_sig/short_sig are booleans indexed like df, TRUE on the bar at whose
    close the condition holds. Entry fills at the NEXT bar's open.
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    n = len(df)

    use_atr = sl_atr is not None
    if use_atr:
        atr_s = atr(df["high"], df["low"], df["close"], atr_period).to_numpy()

    ls = long_sig.to_numpy()
    ss = short_sig.to_numpy()
    mean_arr = tp_mean.to_numpy() if tp_mean is not None else None

    res = Result()
    res.n_signals = int(np.nansum(ls) + np.nansum(ss))

    i = 0
    # we look at signal on bar t, enter at t+1 open, so iterate to n-2
    while i < n - 1:
        go_long = bool(ls[i]) if not np.isnan(ls[i]) else False
        go_short = bool(ss[i]) if not np.isnan(ss[i]) else False
        if not (go_long or go_short):
            i += 1
            continue
        side = "long" if go_long else "short"
        entry_i = i + 1
        entry_px = o[entry_i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            i += 1
            continue

        # stop geometry
        if use_atr:
            a = atr_arr_val = atr_s[i]
            if not np.isfinite(a) or a <= 0:
                i += 1
                continue
            sl_dist = sl_atr * a
            tp_dist = (tp_atr * a) if tp_atr is not None else None
        else:
            sl_dist = sl_frac * entry_px
            tp_dist = (tp_frac * entry_px) if tp_frac is not None else None

        if side == "long":
            sl_px = entry_px - sl_dist
            if mean_arr is not None and np.isfinite(mean_arr[i]):
                tp_px = max(mean_arr[i], entry_px)  # target the mean (>= entry)
            elif tp_dist is not None:
                tp_px = entry_px + tp_dist
            else:
                tp_px = np.inf
        else:
            sl_px = entry_px + sl_dist
            if mean_arr is not None and np.isfinite(mean_arr[i]):
                tp_px = min(mean_arr[i], entry_px)
            elif tp_dist is not None:
                tp_px = entry_px - tp_dist
            else:
                tp_px = -np.inf

        # trailing-stop geometry: trail distance = trail_atr * ATR(at entry).
        # Conservative no-look-ahead: the trailing stop for bar j uses the
        # favorable extreme as of the END of bar j-1; only AFTER checking the
        # stop on bar j do we ratchet the peak with bar j's extreme.
        a_entry = atr_s[i] if use_atr else atr(df["high"], df["low"], df["close"], atr_period).to_numpy()[i]
        trail_dist = (trail_atr * a_entry) if trail_atr is not None else None
        be_dist = (breakeven_atr * a_entry) if breakeven_atr is not None else None
        peak = entry_px  # favorable extreme since entry (high for long, low for short)

        # scan forward for exit
        exit_i = None
        reason = None
        exit_px = None
        last = min(entry_i + max_hold, n - 1)
        j = entry_i
        while j <= last:
            hi = h[j]
            lo = low[j]
            # effective stop = tightest of hard SL, breakeven (once triggered),
            # trailing stop. Breakeven: once price has run be_dist in favor,
            # raise the stop to entry (scratch instead of full loss).
            if side == "long":
                eff_stop = sl_px
                if be_dist is not None and (peak - entry_px) >= be_dist:
                    eff_stop = max(eff_stop, entry_px)
                if trail_dist is not None:
                    eff_stop = max(eff_stop, peak - trail_dist)
                hit_sl = lo <= eff_stop
                hit_tp = hi >= tp_px
            else:
                eff_stop = sl_px
                if be_dist is not None and (entry_px - peak) >= be_dist:
                    eff_stop = min(eff_stop, entry_px)
                if trail_dist is not None:
                    eff_stop = min(eff_stop, peak + trail_dist)
                hit_sl = hi >= eff_stop
                hit_tp = lo <= tp_px
            # SL/trail-first conservative tie-break
            if hit_sl:
                exit_i, reason, exit_px = j, ("trail" if (trail_dist is not None and eff_stop != sl_px) else "sl"), eff_stop
                break
            if hit_tp:
                exit_i, reason, exit_px = j, "tp", tp_px
                break
            # ratchet peak AFTER the stop check (no look-ahead)
            if side == "long":
                peak = max(peak, hi)
            else:
                peak = min(peak, lo)
            j += 1
        if exit_i is None:
            exit_i = last
            reason = "time"
            exit_px = c[last]

        if side == "long":
            ret_gross = (exit_px - entry_px) / entry_px
        else:
            ret_gross = (entry_px - exit_px) / entry_px
        ret_net = ret_gross - 2 * commission_side

        res.trades.append(
            Trade(
                side=side,
                entry_i=entry_i,
                entry_t=idx[entry_i],
                entry_px=float(entry_px),
                exit_t=idx[exit_i],
                exit_px=float(exit_px),
                bars_held=exit_i - entry_i,
                reason=reason,
                ret_gross=float(ret_gross),
                ret_net=float(ret_net),
                risk_frac=float(sl_dist / entry_px),
            )
        )
        # one-position-at-a-time: resume scanning AFTER the exit bar
        i = exit_i + 1
    return res


def summarize(res: Result, tf: str, span_days: float, label: str = "") -> dict:
    return {
        "label": label,
        "trades": res.n,
        "signals_raw": res.n_signals,
        "win_rate": round(res.win_rate * 100, 1) if res.n else None,
        "sig_per_day": round(res.signals_per_day(tf, span_days), 2) if res.n else 0.0,
        "expectancy_pct": round(res.expectancy * 100, 3) if res.n else None,
        "avg_win_pct": round(res.avg_win * 100, 3),
        "avg_loss_pct": round(res.avg_loss * 100, 3),
        "profit_factor": round(res.profit_factor, 2) if res.n else None,
        "max_consec_losses": res.max_consec_losses,
        "max_dd_1x_pct": round(res.max_dd() * 100, 2),
        "span_days": round(span_days, 0),
    }


def span_days_of(df: pd.DataFrame) -> float:
    return (df.index.max() - df.index.min()).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# Signal library
# ---------------------------------------------------------------------------
def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX — trend-strength. Low ADX (<~20) = ranging (mean-reversion
    edge); high ADX = trending (fade gets run over)."""
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bbands(close: pd.Series, period: int, k: float):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def build_meanrev(
    df: pd.DataFrame,
    *,
    bb_period: int = 20,
    bb_k: float = 2.0,
    rsi_period: int = 2,
    rsi_os: float = 10.0,
    rsi_ob: float = 90.0,
    trend_ema: int = 200,
    mode: str = "bb",          # 'bb' = band touch, 'rsi' = RSI extreme, 'both'
    allow_short: bool = True,
    require_trend: bool = True,
):
    """Mean-reversion fade. LONG = oversold in an uptrend; SHORT = overbought in
    a downtrend. tp_mean targets the band middle (the mean to revert to)."""
    close = df["close"]
    lo, mid, hi = bbands(close, bb_period, bb_k)
    r = rsi(close, rsi_period)
    e = ema(close, trend_ema)
    up = close > e if require_trend else pd.Series(True, index=df.index)
    dn = close < e if require_trend else pd.Series(True, index=df.index)

    bb_long = close < lo
    bb_short = close > hi
    rsi_long = r < rsi_os
    rsi_short = r > rsi_ob
    if mode == "bb":
        raw_long, raw_short = bb_long, bb_short
    elif mode == "rsi":
        raw_long, raw_short = rsi_long, rsi_short
    else:  # both
        raw_long, raw_short = (bb_long & rsi_long), (bb_short & rsi_short)

    long_sig = (raw_long & up).fillna(False)
    short_sig = (raw_short & dn).fillna(False) if allow_short else pd.Series(False, index=df.index)
    return long_sig, short_sig, mid


def build_meanrev_regime(
    df, *, bb_period=20, bb_k=2.0, rsi_period=2, rsi_os=10.0, rsi_ob=90.0,
    adx_max=20.0, ema_flat=200, slope_max_pct=0.05, mode="both", allow_short=True,
):
    """Range-regime-gated fade: only fade when the market is RANGING
    (ADX<adx_max AND the EMA is roughly flat). In ranges mean-reversion has
    genuine edge; in trends the fade gets run over. This is the one
    theoretically-sound lever for pushing WR up at positive expectancy."""
    close = df["close"]
    lo, mid, hi = bbands(close, bb_period, bb_k)
    r = rsi(close, rsi_period)
    adx_s = adx(df, 14)
    e = ema(close, ema_flat)
    slope = (e - e.shift(10)) / e * 100.0  # % slope over 10 bars
    ranging = (adx_s < adx_max) & (slope.abs() < slope_max_pct)

    bb_long, bb_short = close < lo, close > hi
    rsi_long, rsi_short = r < rsi_os, r > rsi_ob
    if mode == "bb":
        raw_long, raw_short = bb_long, bb_short
    elif mode == "rsi":
        raw_long, raw_short = rsi_long, rsi_short
    else:
        raw_long, raw_short = (bb_long & rsi_long), (bb_short & rsi_short)
    long_sig = (raw_long & ranging).fillna(False)
    short_sig = (raw_short & ranging).fillna(False) if allow_short else pd.Series(False, index=df.index)
    return long_sig, short_sig, mid


def _hiwr_run(df, cfg) -> Result:
    long_sig, short_sig, mid = build_meanrev_regime(
        df, bb_k=cfg["bb_k"], rsi_os=cfg["rsi_os"], rsi_ob=cfg["rsi_ob"],
        adx_max=cfg["adx_max"], slope_max_pct=cfg["slope_max"], mode=cfg["mode"])
    tp_mean = mid if cfg["tp"] == "mean" else None
    tp_atr = None if cfg["tp"] == "mean" else cfg["tp"]
    return run_engine(df, long_sig, short_sig, sl_atr=cfg["sl_atr"], tp_atr=tp_atr,
                      max_hold=cfg["max_hold"], tp_mean=tp_mean,
                      breakeven_atr=cfg["be_atr"])


def _hiwr_label(cfg):
    return (f"{cfg['mode']}|k{cfg['bb_k']}|rsi{cfg['rsi_os']}/{cfg['rsi_ob']}"
            f"|adx<{cfg['adx_max']}|slope<{cfg['slope_max']}|sl{cfg['sl_atr']}"
            f"|tp{cfg['tp']}|be{cfg['be_atr']}|hold{cfg['max_hold']}")


def _hiwr_configs():
    grid = {
        "mode": ["bb", "both"],
        "bb_k": [2.0, 2.5],
        "rsi_pair": [(5, 95), (10, 90)],
        "adx_max": [15.0, 20.0, 25.0],
        "slope_max": [0.03, 0.08, 1e9],   # 1e9 = slope filter off
        "sl_atr": [1.0, 1.5, 2.0],
        "tp": ["mean", 0.75, 1.0],
        "be_atr": [None, 0.5, 1.0],        # breakeven stop trigger
        "max_hold": [16, 32],
    }
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, combo))
        os_, ob_ = d.pop("rsi_pair")
        yield {"mode": d["mode"], "bb_k": d["bb_k"], "rsi_os": os_, "rsi_ob": ob_,
               "adx_max": d["adx_max"], "slope_max": d["slope_max"],
               "sl_atr": d["sl_atr"], "tp": d["tp"], "be_atr": d["be_atr"],
               "max_hold": d["max_hold"]}


def hiwr_grid(symbol, tf, *, is_start=None, is_end=None, min_n=25):
    """Range-regime fade + breakeven: hunt the HIGHEST win rate that still has
    positive expectancy and 0.5-3/day. This is the last honest attempt at the
    80% target."""
    is_start = is_start or IS_START
    is_end = is_end or IS_END
    df_is = _slice(load(symbol, tf), is_start, is_end)
    span = span_days_of(df_is)
    configs = list(_hiwr_configs())
    print(f"=== HI-WR REGIME-FADE {symbol} {tf} | IS {is_start}..{is_end} "
          f"({span:.0f}d) | {len(configs)} configs ===")
    rows = []
    for cfg in configs:
        res = _hiwr_run(df_is, cfg)
        if res.n < min_n:
            continue
        rows.append(summarize(res, tf, span, _hiwr_label(cfg)))
    pos = [s for s in rows if s["expectancy_pct"] and s["expectancy_pct"] > 0]
    print(f"  n>={min_n}: {len(rows)} | positive-exp: {len(pos)}")
    print("  -- TOP 12 by WIN RATE among positive-expectancy --")
    for s in sorted(pos, key=lambda x: x["win_rate"] or 0, reverse=True)[:12]:
        print("   ", _fmt_row(s))
    n70 = [s for s in pos if (s["win_rate"] or 0) >= 70 and 0.5 <= s["sig_per_day"] <= 3.5]
    print(f"  -- WR>=70 AND +exp AND 0.5-3/day: {len(n70)} --")
    for s in sorted(n70, key=lambda x: x["win_rate"], reverse=True)[:8]:
        print("   ", _fmt_row(s))
    n80 = [s for s in pos if (s["win_rate"] or 0) >= 80]
    print(f"  -- WR>=80 AND +exp (ANY cadence): {len(n80)} --")
    for s in sorted(n80, key=lambda x: x["expectancy_pct"], reverse=True)[:8]:
        print("   ", _fmt_row(s))


def probe() -> None:
    """One theory-motivated config on BTC 1h, full history — viability check."""
    df = load("BTC", "1h")
    long_sig, short_sig, mid = build_meanrev(
        df, bb_period=20, bb_k=2.0, mode="bb", rsi_os=10, trend_ema=200,
        allow_short=True, require_trend=True,
    )
    # Exit: target the mean (mid band), SL at 2.5*ATR, time-stop 24 bars (1 day on 1h).
    res = run_engine(
        df, long_sig, short_sig,
        sl_atr=2.5, tp_atr=None, atr_period=14, max_hold=24,
        tp_mean=mid,
    )
    print("=== PROBE: BTC 1h BB(20,2) fade, trend-filtered, TP=mean, SL=2.5ATR, hold24 ===")
    print(summarize(res, "1h", span_days_of(df), "probe-full-history"))
    print("  reasons:", pd.Series([t.reason for t in res.trades]).value_counts().to_dict())


# ---------------------------------------------------------------------------
# Grid search (IS) + OOS validation
# ---------------------------------------------------------------------------
import itertools  # noqa: E402

# Clean temporal split: search only on IS, hold OOS untouched until the survivor
# is chosen. (BTC/ETH have ~6yr; this keeps the recent regime out of search.)
IS_START, IS_END = "2020-01-01", "2023-01-01"
OOS_START, OOS_END = "2023-01-01", "2026-06-01"


def _slice(df, start, end):
    return df[(df.index >= start) & (df.index < end)]


def _run_config(df, cfg) -> Result:
    long_sig, short_sig, mid = build_meanrev(
        df,
        bb_period=cfg["bb_period"], bb_k=cfg["bb_k"], mode=cfg["mode"],
        rsi_period=cfg.get("rsi_period", 2),
        rsi_os=cfg["rsi_os"], rsi_ob=cfg["rsi_ob"],
        trend_ema=cfg["trend_ema"], allow_short=cfg["allow_short"],
        require_trend=cfg["require_trend"],
    )
    tp_mean = mid if cfg["tp"] == "mean" else None
    tp_atr = None if cfg["tp"] == "mean" else cfg["tp"]
    return run_engine(
        df, long_sig, short_sig,
        sl_atr=cfg["sl_atr"], tp_atr=tp_atr, atr_period=14,
        max_hold=cfg["max_hold"], tp_mean=tp_mean,
    )


def _grid_configs():
    grid = {
        "mode": ["bb", "rsi", "both"],
        "bb_k": [1.5, 2.0, 2.5],
        "rsi_pair": [(5, 95), (10, 90), (15, 85)],
        "sl_atr": [1.0, 1.5, 2.0],
        "max_hold": [8, 16, 32],
        "tp": ["mean", 0.75, 1.0],
        "trend": [True, False],
    }
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, combo))
        os_, ob_ = d.pop("rsi_pair")
        yield {
            "mode": d["mode"], "bb_period": 20, "bb_k": d["bb_k"],
            "rsi_period": 2, "rsi_os": os_, "rsi_ob": ob_,
            "sl_atr": d["sl_atr"], "max_hold": d["max_hold"], "tp": d["tp"],
            "trend_ema": 200, "require_trend": d["trend"],
            "allow_short": True,
        }


def cfg_label(cfg) -> str:
    return (f"{cfg['mode']}|k{cfg['bb_k']}|rsi{cfg['rsi_os']}/{cfg['rsi_ob']}"
            f"|sl{cfg['sl_atr']}|tp{cfg['tp']}|hold{cfg['max_hold']}"
            f"|trend{int(cfg['require_trend'])}")


def _fmt_row(s) -> str:
    return (f"exp{s['expectancy_pct']:+.3f}% WR{s['win_rate']:.1f} "
            f"PF{s['profit_factor']:.2f} sig/d{s['sig_per_day']:.2f} "
            f"avgW{s['avg_win_pct']:+.2f} avgL{s['avg_loss_pct']:+.2f} "
            f"n{s['trades']} dd{s['max_dd_1x_pct']:.1f} mcl{s['max_consec_losses']} "
            f":: {s['label']}")


def grid(symbol: str, tf: str, *, is_start=None, is_end=None, min_n=25) -> None:
    """Frontier report (no hard pass/fail): map the achievable WR/expectancy
    tradeoff. SELECT later on OOS expectancy, never on WR."""
    is_start = is_start or IS_START
    is_end = is_end or IS_END
    df_full = load(symbol, tf)
    df_is = _slice(df_full, is_start, is_end)
    span = span_days_of(df_is)
    configs = list(_grid_configs())
    print(f"=== GRID {symbol} {tf} | IS {is_start}..{is_end} ({span:.0f}d) | "
          f"{len(configs)} configs (multiple-comparisons: treat single hits skeptically) ===")
    rows = []
    for cfg in configs:
        res = _run_config(df_is, cfg)
        if res.n < min_n:
            continue
        rows.append(summarize(res, tf, span, cfg_label(cfg)))
    pos = [s for s in rows if s["expectancy_pct"] and s["expectancy_pct"] > 0]
    print(f"  configs with n>={min_n}: {len(rows)} | positive-expectancy: {len(pos)}")

    print("  -- TOP 10 by EXPECTANCY (any WR) --")
    for s in sorted(rows, key=lambda x: x["expectancy_pct"] or -9, reverse=True)[:10]:
        print("   ", _fmt_row(s))

    print("  -- TOP 10 by WIN RATE among positive-expectancy --")
    for s in sorted(pos, key=lambda x: x["win_rate"] or -9, reverse=True)[:10]:
        print("   ", _fmt_row(s))

    # cadence-aware: 1-3 signals/day AND positive expectancy
    cad = [s for s in pos if 0.8 <= s["sig_per_day"] <= 3.5]
    print(f"  -- positive-exp AND 1-3 sig/day: {len(cad)} (top 8 by exp) --")
    for s in sorted(cad, key=lambda x: x["expectancy_pct"], reverse=True)[:8]:
        print("   ", _fmt_row(s))

    n75 = [s for s in pos if (s["win_rate"] or 0) >= 75]
    print(f"  -- WR>=75 AND positive-exp: {len(n75)} (top 8 by exp) --")
    for s in sorted(n75, key=lambda x: x["expectancy_pct"], reverse=True)[:8]:
        print("   ", _fmt_row(s))


def build_breakout(
    df: pd.DataFrame, *, donchian_n: int = 20, trend_ema: int = 200,
    allow_short: bool = True, require_trend: bool = True,
):
    """Donchian-style momentum breakout. LONG = close breaks above the highest
    high of the prior `donchian_n` bars while in an uptrend. The prior-bar shift
    avoids comparing a bar's close to a channel that includes that same bar."""
    close = df["close"]
    hh = df["high"].rolling(donchian_n).max().shift(1)
    ll = df["low"].rolling(donchian_n).min().shift(1)
    e = ema(close, trend_ema)
    up = close > e if require_trend else pd.Series(True, index=df.index)
    dn = close < e if require_trend else pd.Series(True, index=df.index)
    long_sig = ((close > hh) & up).fillna(False)
    short_sig = ((close < ll) & dn).fillna(False) if allow_short else pd.Series(False, index=df.index)
    return long_sig, short_sig


def _tr_run_config(df, cfg) -> Result:
    long_sig, short_sig = build_breakout(
        df, donchian_n=cfg["donchian_n"], trend_ema=cfg["trend_ema"],
        allow_short=cfg["allow_short"], require_trend=cfg["require_trend"])
    return run_engine(
        df, long_sig, short_sig, sl_atr=cfg["sl_atr"], tp_atr=cfg["tp_atr"],
        atr_period=14, max_hold=cfg["max_hold"], trail_atr=cfg["trail_atr"])


def _tr_label(cfg) -> str:
    return (f"don{cfg['donchian_n']}|sl{cfg['sl_atr']}|tp{cfg['tp_atr']}"
            f"|trail{cfg['trail_atr']}|hold{cfg['max_hold']}|trend{int(cfg['require_trend'])}"
            f"|{'LS' if cfg['allow_short'] else 'L'}")


def _tr_grid_configs():
    grid = {
        "donchian_n": [20, 40, 55],
        "sl_atr": [0.75, 1.0, 1.5],          # SMALL stop
        "tp_atr": [4.0, 6.0, 8.0],            # BIG target
        "trail_atr": [2.0, 3.0, None],        # trailing stop (None = no trail)
        "max_hold": [48, 96, 200],
        "trend": [True, False],
        "short": [True, False],
    }
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, combo))
        yield {"donchian_n": d["donchian_n"], "sl_atr": d["sl_atr"],
               "tp_atr": d["tp_atr"], "trail_atr": d["trail_atr"],
               "max_hold": d["max_hold"], "trend_ema": 200,
               "require_trend": d["trend"], "allow_short": d["short"]}


def tr_grid(symbol: str, tf: str, *, is_start=None, is_end=None, min_n=25) -> None:
    """Trend-rider frontier: small SL, big TP, trailing stop. Select on
    expectancy (R:R does the work; WR is naturally low ~35-50%)."""
    is_start = is_start or IS_START
    is_end = is_end or IS_END
    df_is = _slice(load(symbol, tf), is_start, is_end)
    span = span_days_of(df_is)
    configs = list(_tr_grid_configs())
    print(f"=== TREND-RIDER GRID {symbol} {tf} | IS {is_start}..{is_end} "
          f"({span:.0f}d) | {len(configs)} configs ===")
    rows = []
    for cfg in configs:
        res = _tr_run_config(df_is, cfg)
        if res.n < min_n:
            continue
        rows.append(summarize(res, tf, span, _tr_label(cfg)))
    pos = [s for s in rows if s["expectancy_pct"] and s["expectancy_pct"] > 0]
    print(f"  configs n>={min_n}: {len(rows)} | positive-expectancy: {len(pos)}")
    print("  -- TOP 12 by EXPECTANCY --")
    for s in sorted(rows, key=lambda x: x["expectancy_pct"] or -9, reverse=True)[:12]:
        print("   ", _fmt_row(s))
    cad = [s for s in pos if 0.8 <= s["sig_per_day"] <= 3.5]
    print(f"  -- positive-exp AND 1-3 sig/day: {len(cad)} (top 8 by exp) --")
    for s in sorted(cad, key=lambda x: x["expectancy_pct"], reverse=True)[:8]:
        print("   ", _fmt_row(s))


OOS_WINDOWS = [
    ("2023H1", "2023-01-01", "2023-07-01"),
    ("2023H2", "2023-07-01", "2024-01-01"),
    ("2024H1", "2024-01-01", "2024-07-01"),
    ("2024H2", "2024-07-01", "2025-01-01"),
    ("2025H1", "2025-01-01", "2025-07-01"),
    ("2025H2+", "2025-07-01", "2026-06-01"),
]

# Robust trend-rider neighborhood (dominates IS top across BTC+ETH, 1h+4h).
TR_BASE = dict(donchian_n=55, sl_atr=1.0, tp_atr=8.0, trail_atr=None,
               max_hold=200, trend_ema=200, require_trend=True, allow_short=False)


def _tr_variant(**over):
    c = dict(TR_BASE)
    c.update(over)
    return c


# coins with 1h history for multi-coin pooling
POOL_COINS_1H = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOGE", "LTC",
                 "LINK", "AVAX", "DOT", "ATOM", "BCH", "NEAR", "INJ", "ARB"]


def oos_validate() -> None:
    """OOS (2023-2026, untouched during search) validation of the robust
    trend-rider neighborhood + trailing variants. Worst-window analysis."""
    configs = {
        "base(sl1,tp8,noTrail)": _tr_variant(),
        "sl1.5,tp6,noTrail": _tr_variant(sl_atr=1.5, tp_atr=6.0),
        "don40,sl1,tp8": _tr_variant(donchian_n=40),
        "trail3ATR": _tr_variant(trail_atr=3.0),
        "chandelier5ATR": _tr_variant(trail_atr=5.0),
        "long+short": _tr_variant(allow_short=True),
    }
    for sym, tf in [("BTC", "4h"), ("ETH", "4h"), ("BTC", "1h"), ("ETH", "1h")]:
        df_full = load(sym, tf)
        print(f"\n=== OOS {sym} {tf} (2023-01..2026-06) ===")
        for name, cfg in configs.items():
            df = _slice(df_full, OOS_START, OOS_END)
            res = _tr_run_config(df, cfg)
            if res.n < 5:
                print(f"  {name:24s} n<5 skip")
                continue
            s = summarize(res, tf, span_days_of(df), name)
            # worst window expectancy
            we = []
            for wn, ws, wend in OOS_WINDOWS:
                wdf = _slice(df_full, ws, wend)
                wr = _tr_run_config(wdf, cfg)
                if wr.n >= 3:
                    we.append((wn, round(wr.expectancy * 100, 2), wr.n))
            worst = min((e for _, e, _ in we), default=float("nan"))
            print(f"  {name:24s} exp{s['expectancy_pct']:+.3f}% WR{s['win_rate']:.0f} "
                  f"PF{s['profit_factor']:.2f} n{s['trades']} sig/d{s['sig_per_day']:.2f} "
                  f"avgW{s['avg_win_pct']:+.1f} avgL{s['avg_loss_pct']:+.1f} "
                  f"dd{s['max_dd_1x_pct']:.0f} mcl{s['max_consec_losses']} "
                  f"worstWin{worst:+.2f}%")


def portfolio(tf: str = "4h") -> None:
    """Multi-coin pool: does aggregating the selective 4h trend-rider across
    ~16 coins reach the target cadence at bounded risk? Pools all trades by entry
    time, sizes each at risk_frac of equity at the SL distance, caps concurrent
    positions, reports aggregate cadence + pooled metrics + portfolio DD."""
    cfg = _tr_variant()  # base sl1/tp8/noTrail long-only
    risk_frac = 0.01          # risk 1% of equity per trade at the stop
    max_concurrent = 6
    all_trades = []
    per_coin = []
    for sym in POOL_COINS_1H:
        try:
            df = _slice(load(sym, tf), OOS_START, OOS_END)
        except FileNotFoundError:
            continue
        if len(df) < 500:
            continue
        res = _tr_run_config(df, cfg)
        if res.n == 0:
            continue
        span = span_days_of(df)
        per_coin.append((sym, res.n, round(res.win_rate * 100, 0),
                         round(res.expectancy * 100, 3), round(res.profit_factor, 2),
                         round(res.signals_per_day(tf, span), 2)))
        for t in res.trades:
            all_trades.append((sym, t))
    # sort by entry time
    all_trades.sort(key=lambda x: x[1].entry_t)
    span_all = (OOS_WINDOWS[0][1], OOS_WINDOWS[-1][2])
    total_days = (pd.Timestamp(OOS_END, tz="UTC") - pd.Timestamp(OOS_START, tz="UTC")).days
    n = len(all_trades)
    wins = sum(1 for _, t in all_trades if t.ret_net > 0)
    exp = float(np.mean([t.ret_net for _, t in all_trades])) if n else 0
    print("=== MULTI-COIN POOL (1h, OOS 2023-2026, base sl1/tp8 long-only) ===")
    print(f"  coins: {len(per_coin)}  pooled trades: {n}  over {total_days}d")
    print(f"  AGGREGATE sig/day: {n/total_days:.2f}  pooled WR: {wins/n*100:.1f}%  "
          f"pooled exp/trade: {exp*100:+.3f}%")
    print("  per-coin: sym n WR exp% PF sig/d")
    for r in sorted(per_coin, key=lambda x: -x[3]):
        print(f"    {r[0]:5s} n{r[1]:<4} WR{r[2]:<4.0f} exp{r[3]:+.3f} PF{r[4]:<4.2f} sig/d{r[5]}")

    # Portfolio equity, R-multiple sizing, concurrency cap. Each taken trade
    # risks `risk_frac` of equity at its initial stop (1R). Realised account
    # return = risk_frac * R, R = ret_net / risk_frac_price. We free slots as
    # positions close (exit_t <= candidate entry_t), then take if a slot is open.
    cand = sorted(all_trades, key=lambda x: x[1].entry_t)
    open_pos = []  # list of (exit_t, trade)
    taken_trades = []
    taken = skipped = 0
    for sym, t in cand:
        # free any positions that have closed by this entry time
        open_pos = [(xt, tt) for (xt, tt) in open_pos if xt > t.entry_t]
        if t.risk_frac <= 0:
            skipped += 1
            continue
        if len(open_pos) >= max_concurrent:
            skipped += 1
            continue
        open_pos.append((t.exit_t, t))
        taken_trades.append(t)
        taken += 1
    # compound equity in EXIT order (fixed-fractional, sizing order ~irrelevant
    # at risk 1%); track DD on the realised curve.
    taken_trades.sort(key=lambda t: t.exit_t)
    equity = 1.0
    eq_curve = [1.0]
    for t in taken_trades:
        R = t.ret_net / t.risk_frac
        equity *= (1.0 + risk_frac * R)
        eq_curve.append(equity)
    eqc = np.array(eq_curve)
    peak = np.maximum.accumulate(eqc)
    maxdd = float(((eqc - peak) / peak).min()) * 100
    avg_hold_days = np.mean([t.bars_held for t in taken_trades]) * (
        {"4h": 4, "1h": 1}[tf] / 24.0) if taken_trades else 0
    print(f"  concurrency cap {max_concurrent}, risk {risk_frac*100:.1f}%/trade: "
          f"taken {taken}, skipped {skipped} "
          f"({skipped/(taken+skipped)*100:.0f}% at cap)")
    print(f"  taken-trade cadence: {taken/total_days:.2f}/day "
          f"(vs {n/total_days:.2f}/day raw signals); avg hold {avg_hold_days:.1f} days")
    print(f"  PORTFOLIO equity: {equity:.2f}x  ({(equity-1)*100:+.0f}% over {total_days}d, "
          f"~{((equity**(365/total_days))-1)*100:+.0f}%/yr)  maxDD {maxdd:.1f}%")
    Rs = np.array([t.ret_net / t.risk_frac for t in taken_trades if t.risk_frac > 0])
    if len(Rs):
        print(f"  R-multiples (taken): mean{Rs.mean():+.2f}R median{np.median(Rs):+.2f}R "
              f"best{Rs.max():+.1f}R worst{Rs.min():+.1f}R  "
              f"(avg trade {risk_frac*Rs.mean()*100:+.2f}% acct, typical loss ~{risk_frac*100:.0f}%)")


def _cfg(mode, bb_k, os_, ob_, sl_atr, tp, max_hold, trend=True):
    return {"mode": mode, "bb_period": 20, "bb_k": bb_k, "rsi_period": 2,
            "rsi_os": os_, "rsi_ob": ob_, "sl_atr": sl_atr, "max_hold": max_hold,
            "tp": tp, "trend_ema": 200, "require_trend": trend, "allow_short": True}


def fee_test() -> None:
    """Re-run the top IS survivors at taker (10bps RT) vs maker (4bps RT).
    Maker @ 4bps is an OPTIMISTIC bound: maker fills have adverse selection for
    mean-reversion (limit fills when price keeps going against you, misses the
    instant-bounce winners). If still marginal here, the fade family is dead."""
    # representative top survivors from the IS grids
    specs = [
        ("BTC", "1h", _cfg("both", 1.5, 10, 90, 2.0, "mean", 32)),   # top exp
        ("BTC", "1h", _cfg("both", 1.5, 10, 90, 2.0, 1.0, 16)),      # exp+highWR
        ("BTC", "1h", _cfg("both", 1.5, 10, 90, 2.0, 0.75, 32)),     # WR76.5
        ("BTC", "1h", _cfg("both", 1.5, 15, 85, 2.0, 0.75, 32)),     # WR75.8
        ("ETH", "1h", _cfg("both", 1.5, 5, 95, 1.5, "mean", 16)),    # ETH top exp
        ("ETH", "1h", _cfg("rsi", 1.5, 10, 90, 2.0, 1.0, 32)),       # ETH 67WR 1-3/day
    ]
    print("=== FEE SENSITIVITY (IS 2020-2023): taker 10bps RT  vs  maker 4bps RT ===")
    print("    (maker @4bps = optimistic upper bound; adverse selection ignored)")
    for sym, tf, cfg in specs:
        df = _slice(load(sym, tf), IS_START, IS_END)
        span = span_days_of(df)
        out = []
        for side_fee, tag in [(0.0005, "taker10"), (0.0002, "maker04")]:
            long_sig, short_sig, mid = build_meanrev(
                df, bb_period=cfg["bb_period"], bb_k=cfg["bb_k"], mode=cfg["mode"],
                rsi_os=cfg["rsi_os"], rsi_ob=cfg["rsi_ob"], trend_ema=cfg["trend_ema"],
                allow_short=cfg["allow_short"], require_trend=cfg["require_trend"])
            tp_mean = mid if cfg["tp"] == "mean" else None
            tp_atr = None if cfg["tp"] == "mean" else cfg["tp"]
            res = run_engine(df, long_sig, short_sig, sl_atr=cfg["sl_atr"],
                             tp_atr=tp_atr, max_hold=cfg["max_hold"], tp_mean=tp_mean,
                             commission_side=side_fee)
            s = summarize(res, tf, span, tag)
            out.append((tag, s))
        a = out[0][1]; b = out[1][1]
        print(f"  {sym} {tf} {cfg_label(cfg)}")
        print(f"     taker10: exp{a['expectancy_pct']:+.3f}% WR{a['win_rate']:.1f} "
              f"PF{a['profit_factor']:.2f} n{a['trades']} sig/d{a['sig_per_day']:.2f} "
              f"dd{a['max_dd_1x_pct']:.1f}")
        print(f"     maker04: exp{b['expectancy_pct']:+.3f}% WR{b['win_rate']:.1f} "
              f"PF{b['profit_factor']:.2f} n{b['trades']} sig/d{b['sig_per_day']:.2f} "
              f"dd{b['max_dd_1x_pct']:.1f}")


# ---------------------------------------------------------------------------
# Validation against backtest.py
# ---------------------------------------------------------------------------
def validate_engine() -> None:
    """Reproduce multifactor-v1 entry rules and compare trade count + direction
    to the project's run_backtest on the same window. Not bit-exact (different
    exit modelling) but must be in the same ballpark — a look-ahead bug would
    show up as wildly more trades or impossible WR."""
    from datetime import UTC, datetime

    from backtest import run_backtest

    sym, tf = "BTC/USDT:USDT", "15m"
    start, end = "2024-01-01", "2024-07-01"
    ref = run_backtest(
        "multifactor-v1", sym, tf,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 7, 1, tzinfo=UTC),
        quiet=True,
    )
    print("=== backtest.py reference (multifactor-v1 BTC 15m 2024H1) ===")
    print(f"  trades={ref.get('trades')} after_fees_pct={ref.get('backtest_return_pct')}")

    df = load("BTC", "15m")
    df = df[(df.index >= start) & (df.index < end)]
    close = df["close"]
    r = rsi(close, 14)
    e200 = ema(close, 200)
    vol_sma = sma(df["volume"], 20)
    long_sig = (r < 35) & (close > e200) & (df["volume"] > 2 * vol_sma)
    short_sig = pd.Series(False, index=df.index)
    res = run_engine(
        df, long_sig.fillna(False), short_sig,
        sl_frac=0.015, tp_frac=0.03, max_hold=96,
    )
    print("=== hiwr_harness reproduction (RSI<35 & >EMA200 & vol>2x, SL1.5/TP3) ===")
    print(summarize(res, tf, span_days_of(df), "repro"))
    print(f"  reasons: " + str(pd.Series([t.reason for t in res.trades]).value_counts().to_dict()))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "probe", "grid", "trgrid",
                                     "feetest", "oos", "portfolio", "hiwr"])
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--is-start", default=None)
    ap.add_argument("--is-end", default=None)
    args = ap.parse_args()
    if args.mode == "validate":
        validate_engine()
    elif args.mode == "probe":
        probe()
    elif args.mode == "grid":
        grid(args.symbol, args.tf, is_start=args.is_start, is_end=args.is_end)
    elif args.mode == "trgrid":
        tr_grid(args.symbol, args.tf, is_start=args.is_start, is_end=args.is_end)
    elif args.mode == "feetest":
        fee_test()
    elif args.mode == "oos":
        oos_validate()
    elif args.mode == "portfolio":
        portfolio(tf=args.tf if args.tf in ("4h", "1h") else "4h")
    elif args.mode == "hiwr":
        hiwr_grid(args.symbol, args.tf, is_start=args.is_start, is_end=args.is_end)
