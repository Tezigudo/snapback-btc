"""Should any snapback leg run a trailing stop? — one harness, all three legs.

Research only. Touches no config, no bot code, no droplet.

Question (God, 2026-09-23): "should we have trailing stop? and if so, based on
backtest, what % of trailing stop is best, or what strategy of it, for all legs".

Method
------
Each leg's DEPLOYED backtest class is subclassed with one mixin (`TrailMixin`)
that, on every bar CLOSE, ratchets the trade's resting stop toward a new level
and never loosens it. backtesting.py applies a stop set in `next()` from the
NEXT bar, filled intrabar on that bar's High/Low (or at the open on a gap) —
i.e. exactly what a live bot would get by cancel/replacing its exchange
STOP_MARKET once per closed bar. Binance's native TRAILING_STOP_MARKET moves
intrabar and is NOT what this models.

Trail families (all expressed in R = the leg's own initial stop distance, or
in ATR, and translated to % of price only in the report):
  be      — move stop to entry (+fee buffer) once max-favourable-excursion >= X R
  R       — trail at (extreme − D·R) once MFE >= activation
  atr     — chandelier: (extreme − K·ATR) once MFE >= activation
  pct     — (extreme × (1 − p)) once MFE >= activation
  line    — resting stop AT the leg's existing trailing exit line (donchian
            exit-channel, SOL Supertrend line), so that exit fills intrabar
            instead of waiting for a close through it
Plus TP removal (`no_tp`) for the two legs that carry a TP.

Parity controls — the harness must reproduce each leg's published baseline
before any arm is trusted (checked in `main`, hard stop on a miss):
  v1        +373.2% (2020-01-01 → 2026-08-11), 370 trades, PF 1.392
            (reports/multifactor_v1_live_exit_revalidation.json, as_validated)
  donchian  +1399.02%, 219 trades, PF 1.607 (reports/donchian_exit_period_sweep.json exit_10)
  SOL       +515.0% after funding, 119 trades, PF 1.63 (config/params_sol_supertrend.yaml header)

Donchian live/backtest gap handled here: `DonchianBreakoutBTCv3.next()` never
reads `time_stop_bars`, but the live bot (`_maybe_time_stop`, max_hold_bars 48)
closes every position 48 bars after the fill. The trail arms are compared
against an `as_live` arm that runs that time stop, not against the published
number.

Adoption rule — PRE-REGISTERED, written before any result was seen. A trail
config is recommended for a leg only if ALL hold:
  1. Walk-forward: choosing among {baseline + every arm} on each 18-month
     train window by MAR (return / |maxDD|) and applying it to the next 6
     months beats always-baseline on stitched OOS return.
  2. The fixed arm beats baseline on full-period return NET of funding AND on
     MAR, and wins at least 60% of calendar years.
  3. Plateau: its nearest neighbour(s) in the same family also beat baseline
     net return — a lone peak does not count.
  4. Kill switch: worst start-anchored drawdown no more than 2pp deeper than
     baseline and 0% of deploy dates breaching -35.5%.
  5. Cost stress: still beats baseline at 2x commission.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtesting import Backtest  # noqa: E402

OUT = ROOT / "reports" / "trailing_stop_study.json"
KILL_SWITCH_DD_PCT = -35.5
BE_BUFFER = 0.001   # breakeven = entry ± 0.1% so a scratch covers 2 × 5bps fees

# --------------------------------------------------------------------------
# The mixin
# --------------------------------------------------------------------------


class TrailMixin:
    """Bar-close ratchet on the open trade's stop. Must precede the leg class in the MRO."""

    trail_mode: str = "none"      # none | R | atr | pct
    trail_dist: float = 0.0       # R multiples | ATR multiples | fraction of price
    trail_act_r: float = 0.0      # MFE (in R) before the trail arms; 0 = from entry
    be_at_r: float = 0.0          # 0 = off
    line_stop: bool = False       # resting stop at the leg's own exit line
    live_time_stop: int = 0       # bars after the fill; 0 = off (donchian as-live = 48)
    _atr_col: str = ""            # per-leg ATR column
    _line_long: str = ""          # column holding the long-side exit line
    _line_short: str = ""

    no_tp: bool = False           # drop the leg's take-profit from the entry order

    def buy(self, *a, **k):
        if self.no_tp:
            k["tp"] = None
        return super().buy(*a, **k)

    def sell(self, *a, **k):
        # A short TP N·ATR below price goes NEGATIVE when ATR is a large share
        # of price (SOL 2020-21 at 10 ATR) and backtesting.py asserts. Treat it
        # as "no TP". Never triggers on the published spans — parity unchanged.
        if self.no_tp or (k.get("tp") is not None and k["tp"] <= 0):
            k["tp"] = None
        return super().sell(*a, **k)

    def init(self) -> None:
        super().init()
        self._tkey = None
        self._ext = 0.0
        self._risk = 0.0

    def _line(self, long: bool) -> float | None:
        col = self._line_long if long else self._line_short
        if not col:
            return None
        v = getattr(self.data, col)[-1]
        return float(v) if np.isfinite(v) else None

    def _trail_update(self) -> bool:
        """Returns True if it closed the position (time stop)."""
        if not self.position or not self.trades:
            return False
        t = self.trades[-1]
        i = len(self.data) - 1
        if self._tkey != t.entry_bar:
            self._tkey = t.entry_bar
            self._ext = t.entry_price
            self._risk = abs(t.entry_price - t.sl) if t.sl else 0.0

        # Live time stop: closes at fill + N bars. next() at bar entry_bar+N-1
        # queues the close for the open of bar entry_bar+N.
        if self.live_time_stop > 0 and (i - t.entry_bar) >= self.live_time_stop - 1:
            self.position.close()
            self._tkey = None
            return True

        long = t.is_long
        hi, lo = float(self.data.High[-1]), float(self.data.Low[-1])
        self._ext = max(self._ext, hi) if long else min(self._ext, lo)
        if self._risk <= 0:
            return False
        sign = 1.0 if long else -1.0
        mfe_r = sign * (self._ext - t.entry_price) / self._risk

        cands: list[float] = []
        if self.be_at_r > 0 and mfe_r >= self.be_at_r:
            cands.append(t.entry_price * (1 + sign * BE_BUFFER))
        if self.trail_mode != "none" and mfe_r >= self.trail_act_r:
            if self.trail_mode == "R":
                d = self.trail_dist * self._risk
            elif self.trail_mode == "atr":
                a = float(getattr(self.data, self._atr_col)[-1])
                d = self.trail_dist * a if np.isfinite(a) else None
            elif self.trail_mode == "pct":
                d = self.trail_dist * self._ext
            else:
                raise ValueError(self.trail_mode)
            if d is not None:
                cands.append(self._ext - sign * d)
        if self.line_stop:
            ln = self._line(long)
            if ln is not None:
                cands.append(ln)
        if not cands:
            return False
        new = max(cands) if long else min(cands)
        cur = t.sl
        if cur is None or (long and new > cur) or (not long and new < cur):
            t.sl = new
        return False


# --------------------------------------------------------------------------
# Leg adapters: data + class + deployed attrs + funding
# --------------------------------------------------------------------------

_CACHE: dict = {}


def _v1_setup():
    if "v1" in _CACHE:
        return _CACHE["v1"]
    import tools.multifactor_v1_live_exit_revalidation as rv
    from strategy.signals_multifactor import DayTradeMultiFactorBTC
    from tools.run_mf_deepening import PARQ, _load_slice

    df = _load_slice(PARQ["BTC"], rv.FULL_START, rv.FULL_END, attach_funding=True)
    fund = pd.read_parquet(rv.FUND_PARQ)
    if fund.index.tz is not None:
        fund.index = fund.index.tz_localize(None)

    class V1Trail(TrailMixin, DayTradeMultiFactorBTC):
        def next(self):
            if self._trail_update():
                self._entry_bar = None
                return
            super().next()

    _CACHE["v1"] = dict(df=df, fund=fund, cls=V1Trail, attrs=rv.deployed_overrides(),
                        cash=1_000_000.0, commission=5 / 1e4, margin=1 / 20, bar_h=0.25)
    return _CACHE["v1"]


def _don_setup():
    if "don" in _CACHE:
        return _CACHE["don"]
    from strategy.signals_donchian import DonchianBreakoutBTCv3
    from tools._postfrac_donchian_variants_sweep import (
        CASH,
        COMMISSION,
        MARGIN,
        _load_full_scaled_4h,
        _prep_slice,
    )
    from tools.donchian_exit_period_sweep import DEPLOYED_ATTRS

    full = _load_full_scaled_4h()
    df = _prep_slice(full, full.index[0], full.index[-1], period_entry=80, period_exit=10)
    fund = pd.read_parquet(ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet")
    if fund.index.tz is not None:
        fund.index = fund.index.tz_localize(None)

    class DonTrail(TrailMixin, DonchianBreakoutBTCv3):
        _atr_col = "ATR_1h"
        _line_long = "DonchianExitLower"
        _line_short = "DonchianExitUpper"

        def next(self):
            if self._trail_update():
                self._entry_bar = None
                return
            super().next()

    attrs = {**DEPLOYED_ATTRS, "donchian_period_exit": 10}
    _CACHE["don"] = dict(df=df, fund=fund, cls=DonTrail, attrs=attrs,
                         cash=CASH, commission=COMMISSION, margin=MARGIN, bar_h=4.0)
    return _CACHE["don"]


SOL_START = datetime(2022, 4, 1, tzinfo=UTC)
SOL_END = datetime(2026, 7, 25, tzinfo=UTC)
SOL_ATTRS = {"st_period": 14, "st_multiplier": 3.5, "st_sl_atr": 2.0, "st_tp_atr": 10.0,
             "allow_shorts": True, "st_risk_per_trade_pct": 3.5, "leverage": 3}


def _run_sol(arm_attrs: dict, commission: float, key: str) -> tuple:
    """Mirror sol_leg_blend_confirm._run, but on a fresh per-arm subclass so no
    swept attribute can leak into the next arm through the shared class."""
    import backtest as B
    from strategy.signals import StrategyParams
    from strategy.signals_supertrend import SupertrendBTC, attach_supertrend

    class SolTrail(TrailMixin, SupertrendBTC):
        _atr_col = "STAtr"
        _line_long = "STLine"
        _line_short = "STLine"

        def next(self):
            if self._trail_update():
                self._entry_bar = None
                return
            super().next()

    cls = type(f"Sol_{key}", (SolTrail,), {**SOL_ATTRS, **arm_attrs})
    name = f"trailstudy-{key}"
    B.STRATEGIES[name] = cls
    B._SUPERTREND_STRATEGIES.add(name)
    B._SUPERTREND_ATTACH_FNS[name] = (attach_supertrend, {})
    B._TF_AGNOSTIC_STRATEGIES.add(name)
    override = dataclasses.replace(StrategyParams.from_yaml(),
                                   risk_per_trade_pct=3.5, leverage=3)
    r = B.run_backtest(name, "SOL/USDT:USDT", "4h", SOL_START, SOL_END, leverage=3,
                       quiet=True, params_override=override, commission=commission,
                       return_equity=True, return_trades=True)
    return r


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _start_anchored_dd(eq: np.ndarray) -> np.ndarray:
    fwd_min = np.minimum.accumulate(eq[::-1])[::-1]
    return (fwd_min / eq - 1.0) * 100.0


def _metrics(eq: pd.Series, trades: pd.DataFrame, fcost: float, cash: float,
             bar_h: float) -> dict:
    eqv = eq.values.astype(float)
    peak = np.maximum.accumulate(eqv)
    maxdd = float(((eqv / peak) - 1).min() * 100)
    sa = _start_anchored_dd(eqv)
    ret = float(eqv[-1] / eqv[0] - 1) * 100
    net = ret - 100.0 * fcost / cash
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((1 + net / 100) ** (1 / yrs) - 1) * 100 if net > -100 else -100.0
    n = len(trades)
    if n:
        pnl = trades["PnL"].astype(float)
        gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        pf_usd = float(gp / gl) if gl > 0 else float("inf")
        # backtesting.py's own definition (per-trade ReturnPct) — the one every
        # published number in this repo quotes, so it is the parity field.
        rp = trades["ReturnPct"].astype(float)
        rl = -rp[rp < 0].sum()
        pf = float(rp[rp > 0].sum() / rl) if rl > 0 else float("inf")
        wr = float((pnl > 0).mean() * 100)
        hold_h = (trades["ExitBar"] - trades["EntryBar"]).astype(float) * bar_h
        med_hold_d = float(hold_h.median() / 24)
    else:
        pf = pf_usd = wr = med_hold_d = 0.0
    by_year = {str(y): round(float(g.iloc[-1] / g.iloc[0] - 1) * 100, 2)
               for y, g in eq.groupby(eq.index.year)}
    rets = eq.resample("1D").last().pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0
    return {
        "return_pct": round(ret, 2),
        "return_net_funding_pct": round(net, 2),
        "cagr_net_pct": round(cagr, 2),
        "max_dd_pct": round(maxdd, 2),
        "mar": round(cagr / abs(maxdd), 3) if maxdd < 0 else None,
        "sharpe_daily": round(sharpe, 3),
        "worst_start_anchored_dd_pct": round(float(sa.min()), 2),
        "pct_deploy_dates_breaching": round(float((sa <= KILL_SWITCH_DD_PCT).mean() * 100), 2),
        "trades": n,
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 3),
        "profit_factor_usd": round(pf_usd, 3),
        "median_hold_days": round(med_hold_d, 3),
        "funding_cost_pct_of_cash": round(100.0 * fcost / cash, 2),
        "by_year": by_year,
    }


def run_arm(task: tuple) -> dict:
    leg, key, arm_attrs, commission_mult = task
    t0 = time.time()
    if leg == "sol":
        # 5bps/side = sol_leg_blend_confirm.DEFAULT_COMMISSION (the published run)
        r = _run_sol(arm_attrs, 0.0005 * commission_mult, key)
        eq = r["equity_series"]
        trades = r["trades_df"]
        fcost = float(r["funding_cost_usdt"] or 0.0)
        cash = float(r["actual_cash"])
        bar_h = 4.0
        eq.index = pd.DatetimeIndex(eq.index).tz_localize(None) if eq.index.tz is not None else eq.index
    else:
        from backtest import funding_cost_for_trades
        s = _v1_setup() if leg == "v1" else _don_setup()
        bt = Backtest(s["df"], s["cls"], cash=s["cash"], commission=s["commission"] * commission_mult,
                      margin=s["margin"], trade_on_close=False, exclusive_orders=True,
                      finalize_trades=True)
        st = bt.run(**{**s["attrs"], **arm_attrs})
        trades = st._trades
        fcost, _ = funding_cost_for_trades(trades, s["df"], s["fund"])
        eq = st["_equity_curve"]["Equity"]
        cash, bar_h = s["cash"], s["bar_h"]
    m = _metrics(eq, trades, fcost, cash, bar_h)
    # Daily equity kept for the walk-forward (funding ignored there — a
    # selection device, not a headline number).
    daily = eq.resample("1D").last().ffill()
    return {"leg": leg, "key": key, "attrs": arm_attrs, "commission_mult": commission_mult,
            "metrics": m, "daily_equity": {str(k.date()): float(v) for k, v in daily.items()},
            "exit_reasons": _exit_mix(trades), "elapsed_s": round(time.time() - t0, 1)}


def _exit_mix(trades: pd.DataFrame) -> dict:
    """Share of trades held > N bars is leg-specific; keep hold quantiles instead."""
    if trades is None or not len(trades):
        return {}
    bars = (trades["ExitBar"] - trades["EntryBar"]).astype(float)
    pnl = trades["PnL"].astype(float)
    out = {"hold_bars_p50": float(bars.median()), "hold_bars_p90": float(bars.quantile(0.9)),
           "hold_bars_max": float(bars.max())}
    for n in (48,):
        long_ = bars >= n
        out[f"trades_held_ge_{n}_bars"] = int(long_.sum())
        out[f"pnl_share_of_trades_held_ge_{n}_bars_pct"] = (
            round(float(pnl[long_].sum() / pnl.sum() * 100), 1) if pnl.sum() != 0 else None)
    # top-5 trades share of total PnL — tail dependence
    out["top5_trades_pnl_share_pct"] = (
        round(float(pnl.nlargest(5).sum() / pnl.sum() * 100), 1) if pnl.sum() > 0 else None)
    return out


# --------------------------------------------------------------------------
# Grids — each leg's R and typical ATR decide the ranges (see docstring)
# --------------------------------------------------------------------------


def grids() -> dict[str, dict[str, dict]]:
    v1 = {
        "baseline": {},
        "be_0.5R": {"be_at_r": 0.5},
        "be_1.0R": {"be_at_r": 1.0},
        "be_1.5R": {"be_at_r": 1.5},
        "trail_0.5R_act1R": {"trail_mode": "R", "trail_dist": 0.5, "trail_act_r": 1.0},
        "trail_1.0R_act1R": {"trail_mode": "R", "trail_dist": 1.0, "trail_act_r": 1.0},
        "trail_0.5R_act1R_noTP": {"trail_mode": "R", "trail_dist": 0.5, "trail_act_r": 1.0, "no_tp": True},
        "trail_1.0R_act1R_noTP": {"trail_mode": "R", "trail_dist": 1.0, "trail_act_r": 1.0, "no_tp": True},
        "trail_1.0R_act1.5R_noTP": {"trail_mode": "R", "trail_dist": 1.0, "trail_act_r": 1.5, "no_tp": True},
        "trail_1.5R_act1R_noTP": {"trail_mode": "R", "trail_dist": 1.5, "trail_act_r": 1.0, "no_tp": True},
        "trail_2.0R_act1R_noTP": {"trail_mode": "R", "trail_dist": 2.0, "trail_act_r": 1.0, "no_tp": True},
        "trail_1.0R_act0_noTP": {"trail_mode": "R", "trail_dist": 1.0, "trail_act_r": 0.0, "no_tp": True},
    }
    TS = {"live_time_stop": 48}
    don = {
        "baseline_published_no_timestop": {},
        "baseline": {**TS},   # as-live
        "line_stop": {**TS, "line_stop": True},
        "be_1R": {**TS, "be_at_r": 1.0},
        "be_2R": {**TS, "be_at_r": 2.0},
    }
    for k in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        don[f"atr_{k:g}"] = {**TS, "trail_mode": "atr", "trail_dist": k}
    for k in (2.0, 3.0, 4.0):
        don[f"atr_{k:g}_act1R"] = {**TS, "trail_mode": "atr", "trail_dist": k, "trail_act_r": 1.0}
        don[f"atr_{k:g}_act2R"] = {**TS, "trail_mode": "atr", "trail_dist": k, "trail_act_r": 2.0}
    for p in (0.03, 0.05, 0.08, 0.12, 0.16):
        don[f"pct_{p*100:g}"] = {**TS, "trail_mode": "pct", "trail_dist": p}
    for k in (3.0, 4.0, 5.0, 6.0):
        don[f"atr_{k:g}_noTS"] = {"trail_mode": "atr", "trail_dist": k}
    don["line_stop_noTS"] = {"line_stop": True}
    don["pct_8_noTS"] = {"trail_mode": "pct", "trail_dist": 0.08}
    don["pct_12_noTS"] = {"trail_mode": "pct", "trail_dist": 0.12}

    sol = {
        "baseline": {},
        "line_stop": {"line_stop": True},
        "line_stop_noTP": {"line_stop": True, "no_tp": True},
        "noTP": {"no_tp": True},
        "be_1R": {"be_at_r": 1.0},
        "be_2R": {"be_at_r": 2.0},
    }
    for k in (2.0, 2.5, 3.0, 3.5, 4.5):
        sol[f"atr_{k:g}"] = {"trail_mode": "atr", "trail_dist": k}
    for k in (2.0, 3.0):
        sol[f"atr_{k:g}_act2R"] = {"trail_mode": "atr", "trail_dist": k, "trail_act_r": 2.0}
        sol[f"atr_{k:g}_act2R_noTP"] = {"trail_mode": "atr", "trail_dist": k, "trail_act_r": 2.0,
                                        "no_tp": True}
    for p in (0.05, 0.08, 0.12, 0.16, 0.20):
        sol[f"pct_{p*100:g}"] = {"trail_mode": "pct", "trail_dist": p}
    return {"v1": v1, "don": don, "sol": sol}


# Neighbour map for the plateau test: same family, adjacent value.
def neighbours(leg: str, key: str, keys: list[str]) -> list[str]:
    import re
    m = re.match(r"^(.*?)(\d+(?:\.\d+)?)(R?)(.*)$", key)
    if not m:
        return []
    pre, _num, unit, post = m.groups()
    fam = [k for k in keys if re.match(rf"^{re.escape(pre)}(\d+(?:\.\d+)?){re.escape(unit)}{re.escape(post)}$", k)]
    vals = sorted(fam, key=lambda k: float(re.match(rf"^{re.escape(pre)}(\d+(?:\.\d+)?)", k).group(1)))
    i = vals.index(key)
    return [vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)]


# --------------------------------------------------------------------------
# Walk-forward selection over the per-arm daily equity curves
# --------------------------------------------------------------------------


def _seg(daily: dict, a: str, b: str) -> pd.Series:
    s = pd.Series(daily)
    s.index = pd.to_datetime(s.index)
    return s[(s.index >= a) & (s.index < b)]


def walk_forward(arms: dict[str, dict], baseline: str, start: str, end: str,
                 train_m: int = 18, test_m: int = 6) -> dict:
    folds = []
    t = pd.Timestamp(start) + pd.DateOffset(months=train_m)
    E = pd.Timestamp(end)
    while t + pd.DateOffset(months=1) <= E:
        tr_a, te_b = t - pd.DateOffset(months=train_m), min(t + pd.DateOffset(months=test_m), E)
        best, best_score = None, -np.inf
        for k, r in arms.items():
            s = _seg(r["daily_equity"], str(tr_a.date()), str(t.date()))
            if len(s) < 30:
                continue
            ret = s.iloc[-1] / s.iloc[0] - 1
            dd = (s / s.cummax() - 1).min()
            score = ret / abs(dd) if dd < 0 else ret * 100
            if score > best_score:
                best, best_score = k, score
        def test_ret(k, a=t, b=te_b):
            s = _seg(arms[k]["daily_equity"], str(a.date()), str(b.date()))
            return float(s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 else 0.0
        folds.append({"test": f"{t.date()}..{te_b.date()}", "chosen": best,
                      "chosen_ret_pct": round(test_ret(best) * 100, 2),
                      "baseline_ret_pct": round(test_ret(baseline) * 100, 2)})
        t = t + pd.DateOffset(months=test_m)
    comp = lambda key: (np.prod([1 + f[key] / 100 for f in folds]) - 1) * 100  # noqa: E731
    return {"folds": folds, "stitched_selected_pct": round(comp("chosen_ret_pct"), 2),
            "stitched_baseline_pct": round(comp("baseline_ret_pct"), 2),
            "folds_selected_beats_baseline": sum(f["chosen_ret_pct"] > f["baseline_ret_pct"] for f in folds),
            "folds_baseline_chosen": sum(f["chosen"] == baseline for f in folds),
            "n_folds": len(folds)}


WF_SPAN = {"v1": ("2020-01-01", "2026-08-11"), "don": ("2019-11-01", "2026-08-11"),
           "sol": ("2022-04-01", "2026-07-25")}
PARITY = {
    "v1": {"key": "baseline", "return_pct": 373.2, "trades": 370, "profit_factor": 1.392},
    "don": {"key": "baseline_published_no_timestop", "return_pct": 1399.02, "trades": 219,
            "profit_factor": 1.607},
    "sol": {"key": "baseline", "return_net_funding_pct": 515.0, "trades": 119, "profit_factor": 1.63},
}


def check_parity(results: dict) -> dict:
    out = {}
    for leg, want in PARITY.items():
        got = results[leg][want["key"]]["metrics"]
        rows = {}
        for k, v in want.items():
            if k == "key":
                continue
            g = got[k]
            tol = 0.5 if "return" in k else (0 if k == "trades" else 0.01)
            rows[k] = {"want": v, "got": g, "ok": abs(g - v) <= tol}
        out[leg] = {"ok": all(r["ok"] for r in rows.values()), "fields": rows}
    return out


def evaluate(leg: str, arms: dict, stress: dict) -> dict:
    base = arms["baseline"]["metrics"]
    keys = list(arms)
    rows = {}
    for k, r in arms.items():
        m = r["metrics"]
        yrs = [y for y in m["by_year"] if y in base["by_year"]]
        yr_wins = sum(m["by_year"][y] > base["by_year"][y] for y in yrs)
        rows[k] = {
            "beats_net": m["return_net_funding_pct"] > base["return_net_funding_pct"],
            "beats_mar": (m["mar"] or -9) > (base["mar"] or -9),
            "year_wins": f"{yr_wins}/{len(yrs)}",
            "year_win_frac": yr_wins / len(yrs) if yrs else 0,
            "dd_ok": (m["worst_start_anchored_dd_pct"] >= base["worst_start_anchored_dd_pct"] - 2.0
                      and m["pct_deploy_dates_breaching"] == 0.0),
        }
    for k in rows:
        nb = neighbours(leg, k, keys)
        rows[k]["neighbours"] = nb
        rows[k]["plateau"] = bool(nb) and all(rows[n]["beats_net"] for n in nb)
        st = stress.get(k)
        rows[k]["stress_2x_beats_base"] = (
            None if st is None or "baseline" not in stress else
            st["metrics"]["return_net_funding_pct"] > stress["baseline"]["metrics"]["return_net_funding_pct"])
        rows[k]["passes_fixed_arm_gates"] = bool(
            k != "baseline" and rows[k]["beats_net"] and rows[k]["beats_mar"]
            and rows[k]["year_win_frac"] >= 0.6 and rows[k]["plateau"] and rows[k]["dd_ok"]
            and rows[k]["stress_2x_beats_base"])
    return rows


# POST-HOC probe, added AFTER the pre-registered run: the two BE arms that
# passed every gate but the plateau test only had one neighbour each on a
# coarse grid. This densifies around them to answer "plateau or spike?". It
# does not re-open the adoption rule; results go to a separate file.
PROBE = {
    "sol": {"baseline": {}, **{f"be_{r:g}R": {"be_at_r": r}
                               for r in (1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0)}},
    "v1": {"baseline": {}, **{f"be_{r:g}R": {"be_at_r": r}
                              for r in (1.0, 1.25, 1.5, 1.75)}},
}


def run_probe(workers: int) -> int:
    tasks = [(leg, k, v, cm) for leg, arms in PROBE.items() for k, v in arms.items()
             for cm in (1.0, 2.0)]
    out: dict = {}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(run_arm, tasks):
            out.setdefault(r["leg"], {}).setdefault(r["key"], {})[f"x{r['commission_mult']:g}"] = {
                "metrics": r["metrics"], "exits": r["exit_reasons"]}
    for leg, arms in out.items():
        b = arms["baseline"]["x1"]["metrics"]
        for k, v in arms.items():
            m, m2 = v["x1"]["metrics"], v["x2"]["metrics"]
            yrs = [y for y in m["by_year"] if y in b["by_year"]]
            yw = sum(m["by_year"][y] > b["by_year"][y] for y in yrs)
            print(f"PROBE {leg:3} {k:10} net={m['return_net_funding_pct']:8.2f}% "
                  f"x2cost={m2['return_net_funding_pct']:8.2f}% dd={m['max_dd_pct']:7.2f} "
                  f"MAR={m['mar']} PF={m['profit_factor']} n={m['trades']} years={yw}/{len(yrs)}",
                  file=sys.stderr)
    OUT.with_suffix(".probe.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--legs", default="v1,don,sol")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--parity-only", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return run_probe(a.workers)
    legs = a.legs.split(",")
    G = grids()
    if a.parity_only:
        G = {leg: {k: v for k, v in G[leg].items() if k.startswith("baseline")} for leg in legs}
    tasks = [(leg, k, v, 1.0) for leg in legs for k, v in G[leg].items()]
    print(f"{len(tasks)} arms on {a.workers} workers", file=sys.stderr)
    results: dict = {leg: {} for leg in legs}
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_arm, tasks):
            results[r["leg"]][r["key"]] = r
            m = r["metrics"]
            print(f"  {r['leg']:4} {r['key']:28} net={m['return_net_funding_pct']:9.2f}% "
                  f"dd={m['max_dd_pct']:7.2f} saDD={m['worst_start_anchored_dd_pct']:7.2f} "
                  f"PF={m['profit_factor']:.3f} n={m['trades']} MAR={m['mar']} ({r['elapsed_s']}s)",
                  file=sys.stderr)

    parity = check_parity({k: v for k, v in results.items()}) if set(legs) == {"v1", "don", "sol"} else {}
    for leg, p in parity.items():
        print(f"PARITY {leg}: {'OK' if p['ok'] else 'MISS'} {p['fields']}", file=sys.stderr)
    if parity and not all(p["ok"] for p in parity.values()):
        print("PARITY FAILED — not trusting any arm; stopping.", file=sys.stderr)
        OUT.with_suffix(".parity_fail.json").write_text(json.dumps(
            {"parity": parity, "metrics": {lg: {k: r["metrics"] for k, r in v.items()}
                                           for lg, v in results.items()}}, indent=1, default=str))
        return 2
    if a.parity_only:
        return 0

    # Cost stress (2x) on baseline + every arm that beats baseline net.
    stress_tasks = []
    for leg in legs:
        base = results[leg]["baseline"]["metrics"]["return_net_funding_pct"]
        for k, r in results[leg].items():
            if k == "baseline" or r["metrics"]["return_net_funding_pct"] > base:
                stress_tasks.append((leg, k, r["attrs"], 2.0))
    stress: dict = {leg: {} for leg in legs}
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_arm, stress_tasks):
            stress[r["leg"]][r["key"]] = r

    report = {"generated_at": datetime.now(UTC).isoformat(), "parity": parity, "legs": {}}
    for leg in legs:
        arms = {k: v for k, v in results[leg].items() if k != "baseline_published_no_timestop"}
        wf = walk_forward(arms, "baseline", *WF_SPAN[leg])
        ev = evaluate(leg, arms, stress[leg])
        report["legs"][leg] = {
            "arms": {k: {"attrs": r["attrs"], "metrics": r["metrics"], "exits": r["exit_reasons"],
                         "gates": ev.get(k),
                         "stress_2x": stress[leg].get(k, {}).get("metrics")}
                     for k, r in results[leg].items()},
            "walk_forward": wf,
            "recommend": [k for k, g in ev.items() if g["passes_fixed_arm_gates"]]
                         if wf["stitched_selected_pct"] > wf["stitched_baseline_pct"] else [],
        }
        print(f"WF {leg}: selected {wf['stitched_selected_pct']}% vs baseline "
              f"{wf['stitched_baseline_pct']}%  baseline chosen {wf['folds_baseline_chosen']}/{wf['n_folds']}",
              file=sys.stderr)
        print(f"RECOMMEND {leg}: {report['legs'][leg]['recommend']}", file=sys.stderr)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    # daily equity is bulky — keep it in a sidecar for plotting
    OUT.with_suffix(".equity.json").write_text(json.dumps(
        {leg: {k: r["daily_equity"] for k, r in results[leg].items()} for leg in legs}))
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
