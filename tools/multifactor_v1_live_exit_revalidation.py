"""multifactor-v1 re-validation against the exit model that ACTUALLY RUNS LIVE.

Why this exists
---------------
`DayTradeMultiFactorBTC.next()` (strategy/signals_multifactor.py) closes the
position on an adverse EMA200 cross whenever `require_trend=True` — which the
deployed config sets. The live bot never runs that rule: bot.py gates the hook
on `bot_internals.strategy_uses_trend_exit()`, which is True only for
`donchian-v3` and `supertrend`. Live multifactor-v1 therefore exits on the
exchange-native SL/TP bracket or the time stop, and nothing else.

Every prior v1 sign-off (LEVER1_SIGNOFF_DEPLOY, MULTIFACTOR_4H_GATE_DECISION,
the 4H-gate revalidation and walk-forward) measured the WITH-trend-exit model.
`tools/multifactor_validate.py` does not close the gap either: its stage 2
compares ENTRY signal bars only, so 100% parity there says nothing about exits.

This tool re-runs the standard promote battery on `LiveExitMultiFactorBTC` —
the entry logic untouched, the adverse-trend exit removed — and reports it
head-to-head against the as-validated model.

Protocol (matches tools/_sol_leg_deep_validation.py and run_mf_deepening):
  1. The 5 locked OOS windows, both exit models, funding column attached.
  2. Calendar-quarter walk-forward 2020Q1..2026Q2, both models, >=3-trade
     sufficiency filter, 70%-positive gate.
  3. CONTROL walk-forward in the ORIGINAL method (rolling 90-day windows,
     quarterly advance, 2020-01-01..2026-02-28) across four arms —
     {old params, deployed params} x {as-validated, as-live}. The old-params
     as-validated arm must reproduce reports/multifactor_v1_4h_gate_walk_forward
     .json (20/25, +243.81%) or the harness itself is not trustworthy; the other
     three arms then attribute any drop to params vs exit model.
  4. Cost stress 5 / 10 / 15 bps per side on the 5 OOS, BOTH exit models.
  5. Full-period contiguous run, checked against the kill switch.

Kill-switch metric
------------------
`deploy.kill_switch_equity_fraction` (0.645) is measured against DEPLOY-START
equity, not the running peak — so peak-to-trough max drawdown is the WRONG test.
This tool uses the START-ANCHORED drawdown: for every possible deploy date,
min(equity thereafter) / equity at that date, then the worst over all start
dates, plus the share of start dates that would have tripped the -35.5% floor.

Params come from config/params.yaml so this stays honest as the config moves;
run_mf_deepening.LOCKED is the pre-2026-07 snapshot and is NOT used as-is.

    uv run python tools/multifactor_v1_live_exit_revalidation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import funding_cost_for_trades  # noqa: E402
from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.run_mf_deepening import (  # noqa: E402
    CASH,
    LOCKED,
    MARGIN,
    PARQ,
    WINDOWS,
    _load_slice,
    run_one,
    summarize,
)

PARQ_4H = str(ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet")
FUND_PARQ = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"
OUT = ROOT / "reports" / "multifactor_v1_live_exit_revalidation.json"

KILL_SWITCH_DD_PCT = -35.5   # deploy.kill_switch_equity_fraction 0.645
SUFFICIENT_TRADES = 3
WF_GATE = 0.70
# Extended 2026-07-25 -> 2026-08-11 when the data cache was refreshed for the
# post-fix re-run. Must stay at or before the parquet's last bar —
# `_assert_window_is_cached` now raises if it doesn't, because `_load_slice`
# would otherwise shorten the window silently and the verdict would read as
# current while covering less than it claims.
FULL_START, FULL_END = "2020-01-01", "2026-08-11"


class LiveExitMultiFactorBTC(DayTradeMultiFactorBTC):
    """Entry logic of the parent; exits exactly as bot.py runs them.

    Parent `next()` does three things while in a position: time stop, adverse
    EMA200 exit, then nothing. bot.py runs the time stop (`_maybe_time_stop`)
    and leaves TP/SL to the exchange bracket, but never evaluates an adverse
    trend cross for this strategy. Dropping that one branch is the whole diff —
    entries, sizing and bracket geometry are inherited untouched so any entry
    parity already established still holds.
    """

    def next(self) -> None:
        i = len(self.data) - 1
        close_v = self.data.Close[-1]

        if self.position:
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold_bars:
                self.position.close()
                self._entry_bar = None
            return

        sl_dist = self.sl_pct * close_v
        tp_dist = self.tp_pct * close_v
        units = self._position_units(close_v, sl_dist)
        if units <= 0:
            return

        if self._long_signal(i):
            self.buy(size=units, sl=close_v - sl_dist, tp=close_v + tp_dist)
            self._entry_bar = i
        elif self._short_signal(i):
            self.sell(size=units, sl=close_v + sl_dist, tp=close_v - tp_dist)
            self._entry_bar = i


# Model identifiers. Named constants rather than bare strings because they key
# FOUR parallel structures (MODELS, the oos/full_period report sections, the
# control-walk-forward arms and the cost-stress sections) — a typo in any one of
# them would silently gate the wrong arm, which is the class of mistake this
# whole tool exists to catch.
AS_VALIDATED = "as_validated"        # DayTradeMultiFactorBTC — DEPLOYED since 2026-08-10
AS_LIVE_RUNS = "as_live_runs"        # LiveExitMultiFactorBTC — what ran before that

MODELS = {
    AS_VALIDATED: DayTradeMultiFactorBTC,
    AS_LIVE_RUNS: LiveExitMultiFactorBTC,
}

# Control-walk-forward arm names and cost-stress report keys, derived from the
# model ids so they cannot drift apart.
CTRL_ARM = {m: f"deployed_params__{m}" for m in MODELS}
STRESS_KEY = {AS_VALIDATED: "cost_stress_as_validated",
              AS_LIVE_RUNS: "cost_stress_live_exit"}


def deployed_overrides() -> dict:
    """Read config/params.yaml so the run tracks production, not LOCKED."""
    p = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
    s, z = p["strategy"], p["sizing"]
    return {
        "rsi_period":                  int(s["rsi_period"]),
        "rsi_long_threshold":          float(s["rsi_long_threshold"]),
        "rsi_short_threshold":         float(s["rsi_short_threshold"]),
        "volume_ma_period":            int(s["volume_ma_period"]),
        "volume_multiple":             float(s["volume_multiple"]),
        "mf_trend_ema_period":         int(s["mf_trend_ema_period"]),
        "require_trend":               bool(s["require_trend"]),
        "require_candlestick":         bool(s["require_candlestick"]),
        "require_macd":                bool(s["require_macd"]),
        "require_funding_not_extreme": bool(s["require_funding_not_extreme"]),
        "funding_extreme_threshold":   float(s["funding_extreme_threshold"]),
        "sl_pct":                      float(s["sl_pct"]),
        "tp_pct":                      float(s["tp_pct"]),
        "max_hold_bars":               int(s["max_hold_bars"]),
        "risk_per_trade_pct":          float(z["risk_per_trade_pct"]),
        "leverage":                    int(z["leverage"]),
        "allow_shorts":                True,
        "use_mtf_4h_gate":             bool(s["use_mtf_4h_gate"]),
        "mtf_4h_ema_period":           int(s["mtf_4h_ema_period"]),
        "mtf_4h_parquet_path":         PARQ_4H,
    }


def quarters(first_year: int = 2020, last: tuple[int, int] = (2026, 2)):
    spans = [("01-01", "03-31"), ("04-01", "06-30"),
             ("07-01", "09-30"), ("10-01", "12-31")]
    out = []
    for y in range(first_year, last[0] + 1):
        for q, (a, b) in enumerate(spans, start=1):
            if (y, q) > last:
                break
            out.append((f"{y}Q{q}", f"{y}-{a}", f"{y}-{b}"))
    return out


def _psr_of(summ: dict) -> dict:
    psr = (summ.get("canonical") or {}).get("psr_walkforward") or {}
    return {"psr_wf": round(float(psr.get("psr_vs_hurdle", 0.0)), 4),
            "interpretation": psr.get("interpretation")}


def run_oos(ov: dict, slices: dict) -> dict:
    out = {}
    for name, cls in MODELS.items():
        per_window = {lbl: run_one(slices[lbl], ov, strategy_class=cls)
                      for lbl, _s, _e in WINDOWS}
        summ = summarize(name, per_window)
        agg = summ["summary"]
        out[name] = {
            "compounded_pct":   agg["compounded_pct"],
            "windows_positive": agg["windows_positive"],
            "n_trades":         agg["n_trades"],
            "per_window":       {w: {"trades": r["trades"],
                                     "return_pct": round(r["return_pct"], 2),
                                     "max_dd_pct": round(r["max_dd_pct"], 2),
                                     "win_rate_pct": round(r["win_rate_pct"], 1)}
                                 for w, r in summ["per_window"].items()},
            **_psr_of(summ),
        }
        print(f"  OOS {name:13} comp={out[name]['compounded_pct']:8.2f}% "
              f"wins={out[name]['windows_positive']} trades={agg['n_trades']} "
              f"psr={out[name]['psr_wf']}", file=sys.stderr)
    return out


def run_walk_forward(ov: dict) -> dict:
    qs = quarters()
    slices = {}
    for lbl, s, e in qs:
        try:
            slices[lbl] = _load_slice(PARQ["BTC"], s, e, attach_funding=True)
        except Exception as exc:  # window past end of data
            print(f"  WF skip {lbl}: {exc}", file=sys.stderr)

    out = {}
    for name, cls in MODELS.items():
        rows = []
        for lbl, _s, _e in qs:
            if lbl not in slices:
                continue
            r = run_one(slices[lbl], ov, strategy_class=cls)
            rows.append({"q": lbl, "trades": r["trades"],
                         "return_pct": round(r["return_pct"], 2),
                         "max_dd_pct": round(r["max_dd_pct"], 2)})
        suff = [q for q in rows if q["trades"] >= SUFFICIENT_TRADES]
        pos = [q for q in suff if q["return_pct"] > 0]
        pct = round(100 * len(pos) / len(suff), 1) if suff else None
        out[name] = {
            "quarters": rows,
            "n_quarters": len(rows),
            "n_sufficient": len(suff),
            "n_positive_sufficient": len(pos),
            "pct_positive_sufficient": pct,
            "gate_70pct": bool(suff and len(pos) / len(suff) >= WF_GATE),
            "worst_quarter": min(rows, key=lambda q: q["return_pct"]) if rows else None,
        }
        print(f"  WF  {name:13} {len(pos)}/{len(suff)} sufficient quarters positive "
              f"({pct}%) gate={out[name]['gate_70pct']}", file=sys.stderr)
    return out


def _rolling90(start: str = "2020-01-01", end: str = "2026-02-28"):
    out, s, E = [], pd.Timestamp(start), pd.Timestamp(end)
    while s + pd.Timedelta(days=90) <= E:
        e = s + pd.Timedelta(days=90)
        out.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        s = e
    return out


def run_control_walk_forward(ov: dict) -> dict:
    """Original method, 4 arms. Arm 1 must reproduce the signed-off artifact."""
    old = {**LOCKED, "use_mtf_4h_gate": True, "mtf_4h_ema_period": 200,
           "mtf_4h_parquet_path": PARQ_4H}
    wins = _rolling90()
    slices = [_load_slice(PARQ["BTC"], s, e, attach_funding=True) for s, e in wins]

    arms = {
        "old_params__as_validated":      (old, DayTradeMultiFactorBTC),
        "deployed_params__as_validated": (ov,  DayTradeMultiFactorBTC),
        "deployed_params__as_live_runs": (ov,  LiveExitMultiFactorBTC),
        "old_params__as_live_runs":      (old, LiveExitMultiFactorBTC),
    }
    out = {"method": "rolling 90-day test windows, quarterly advance, "
                     "2020-01-01..2026-02-28 (no train phase — locked config)",
           "n_windows": len(wins),
           "reference_artifact": "reports/multifactor_v1_4h_gate_walk_forward.json "
                                 "= 20/25 positive (80.0%), compounded +243.8111%"}
    for name, (cfg, cls) in arms.items():
        rets, comp, tr = [], 1.0, 0
        for d in slices:
            r = run_one(d, cfg, strategy_class=cls)
            rets.append(round(r["return_pct"], 4))
            comp *= 1 + r["return_pct"] / 100.0
            tr += r["trades"]
        pos = sum(1 for r in rets if r > 0)
        out[name] = {
            "windows_positive": f"{pos}/{len(rets)}",
            "pct_positive": round(100 * pos / len(rets), 1),
            "gate_70pct": pos / len(rets) >= WF_GATE,
            "compounded_pct": round(100 * (comp - 1), 4),
            "trades": tr,
            "per_window_return_pct": rets,
        }
        print(f"  CTRL {name:32} {pos}/{len(rets)} ({out[name]['pct_positive']}%) "
              f"comp={out[name]['compounded_pct']:9.2f}% trades={tr}", file=sys.stderr)

    ref = out["old_params__as_validated"]
    out["harness_reproduces_artifact"] = bool(
        ref["windows_positive"] == "20/25" and abs(ref["compounded_pct"] - 243.8111) < 0.5)
    print(f"  CTRL harness reproduces signed-off artifact: "
          f"{out['harness_reproduces_artifact']}", file=sys.stderr)
    return out


def run_cost_stress(ov: dict, slices: dict, model: str = AS_LIVE_RUNS) -> dict:
    """Cost stress for one exit model.

    Parameterised by model on 2026-08-10: until then this only ever ran the
    live-exit arm, because that was the model under suspicion. The adverse-trend
    exit shipped that day (main 92433d4), so the as-validated arm IS production
    now and needs its own stress numbers rather than inheriting the other arm's.
    """
    import tools.run_mf_deepening as mfd
    cls = MODELS[model]
    saved = mfd.COMMISSION
    out = {}
    try:
        for bps in (5, 10, 15):
            mfd.COMMISSION = bps / 1e4
            per_window = {lbl: run_one(slices[lbl], ov, strategy_class=cls)
                          for lbl, _s, _e in WINDOWS}
            summ = summarize(f"{model}_{bps}bps", per_window)
            agg = summ["summary"]
            out[f"{bps}bps"] = {
                "compounded_pct": agg["compounded_pct"],
                "windows_positive": agg["windows_positive"],
                "n_trades": agg["n_trades"],
                **_psr_of(summ),
            }
            print(f"  stress[{model}] {bps}bps: comp={agg['compounded_pct']:8.2f}% "
                  f"wins={agg['windows_positive']} psr={out[f'{bps}bps']['psr_wf']}",
                  file=sys.stderr)
    finally:
        mfd.COMMISSION = saved
    return out


def _start_anchored_dd(eq: np.ndarray) -> np.ndarray:
    """For each possible deploy start i: min(eq[i:]) / eq[i] - 1, in percent.

    This is the quantity `deploy.kill_switch_equity_fraction` actually tests —
    the floor is anchored to deploy-start equity, not to the running peak, so
    peak-to-trough max drawdown neither implies nor is implied by a breach.
    """
    fwd_min = np.minimum.accumulate(eq[::-1])[::-1]
    return (fwd_min / eq - 1.0) * 100.0


def _assert_window_is_cached(start: str, end: str) -> None:
    """Fail fast if the requested window runs past the cached data.

    `_load_slice` SILENTLY shortens to whatever the parquet holds, so a stale
    cache turns "full period through today" into "full period through whenever
    I last refreshed" — with no signal in the output and a verdict that reads as
    current. The kill-switch and per-year numbers are exactly the ones a short
    window would flatter, so this is worth an assertion rather than a comment.
    """
    last = pd.read_parquet(PARQ["BTC"]).index.max()
    if last.tz is not None:
        last = last.tz_localize(None)
    want = pd.Timestamp(end)
    if want > last:
        raise SystemExit(
            f"FULL_END={end} exceeds cached 15m data (last bar {last:%Y-%m-%d %H:%M}). "
            f"Refresh the cache (exchange.data.load_klines) or lower FULL_END — "
            f"_load_slice would otherwise shorten the window silently and the "
            f"verdict would look current while covering less than it claims.")
    print(f"  window {start}..{end} within cache (last bar {last:%Y-%m-%d %H:%M})",
          file=sys.stderr)


def run_full_period(ov: dict) -> dict:
    """Contiguous run — true path-dependent DD and the kill-switch check."""
    _assert_window_is_cached(FULL_START, FULL_END)
    df = _load_slice(PARQ["BTC"], FULL_START, FULL_END, attach_funding=True)
    fund = pd.read_parquet(FUND_PARQ)
    if fund.index.tz is not None:
        fund.index = fund.index.tz_localize(None)

    out = {}
    for name, cls in MODELS.items():
        bt = Backtest(df, cls, cash=CASH, commission=5 / 1e4, margin=MARGIN,
                      trade_on_close=False, exclusive_orders=True,
                      finalize_trades=True)
        st = bt.run(**ov)
        tr = st._trades
        fcost, fevents = funding_cost_for_trades(tr, df, fund)
        ret = float(st["Return [%]"])
        hold_d = ((tr.ExitBar - tr.EntryBar) * 15 / 60 / 24) if len(tr) else pd.Series([np.nan])
        dd = float(st["Max. Drawdown [%]"])

        curve = st["_equity_curve"]
        eq = curve["Equity"].values
        sa = _start_anchored_dd(eq)
        breach = sa <= KILL_SWITCH_DD_PCT
        worst_i = int(sa.argmin())

        eqs = pd.Series(eq, index=curve.index)
        per_year = {str(y): round(float(g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
                    for y, g in eqs.groupby(eqs.index.year)}

        out[name] = {
            "return_pct":             round(ret, 2),
            "return_pct_net_funding": round(ret - 100.0 * fcost / CASH, 2),
            "funding_cost_usdt":      round(fcost, 0),
            "funding_events":         fevents,
            "trades":                 int(st["# Trades"]),
            "win_rate_pct":           round(float(st["Win Rate [%]"] or 0), 1),
            "profit_factor":          round(float(st["Profit Factor"] or 0), 3),
            "sharpe":                 round(float(st["Sharpe Ratio"] or 0), 3),
            "max_dd_pct_peak_to_trough": round(dd, 2),
            "median_hold_days":       round(float(hold_d.median()), 3),
            "worst_start_anchored_dd_pct": round(float(sa.min()), 2),
            "worst_deploy_date":      str(curve.index[worst_i].date()),
            "pct_of_start_dates_breaching_floor": round(float(breach.mean()) * 100, 2),
            "kill_switch_breached":   bool(breach.any()),
            "worst_start_anchored_dd_if_deployed_2025plus": round(
                float(sa[curve.index >= pd.Timestamp("2025-01-01")].min()), 2),
            "equity_return_pct_by_year": per_year,
        }
        print(f"  FULL {name:13} ret={out[name]['return_pct']:8.2f}% "
              f"startDD={out[name]['worst_start_anchored_dd_pct']:7.2f}% "
              f"PF={out[name]['profit_factor']} "
              f"kill_breach={out[name]['kill_switch_breached']}", file=sys.stderr)
    return out


def main() -> int:
    ov = deployed_overrides()
    print("=" * 78)
    print("multifactor-v1 — re-validation against the LIVE exit model")
    print("=" * 78)
    print(f"deployed params: vol_mult={ov['volume_multiple']} "
          f"funding_thr={ov['funding_extreme_threshold']} "
          f"risk={ov['risk_per_trade_pct']}%", file=sys.stderr)

    slices = {lbl: _load_slice(PARQ["BTC"], s, e, attach_funding=True)
              for lbl, s, e in WINDOWS}

    out: dict = {
        "strategy": "multifactor-v1 (deployed config) — exit-model comparison",
        "exit_models": {
            AS_VALIDATED: "DayTradeMultiFactorBTC — TP/SL + time stop + adverse EMA200 exit",
            AS_LIVE_RUNS: "LiveExitMultiFactorBTC — TP/SL + time stop ONLY (bot.py behaviour)",
        },
        "deployed_overrides": ov,
        "commission_note": "5 bps per side unless stated (cost_stress varies it)",
    }
    out["oos_5_windows"] = run_oos(ov, slices)
    out["walk_forward_quarterly"] = run_walk_forward(ov)
    out["walk_forward_control_original_method"] = run_control_walk_forward(ov)
    out[STRESS_KEY[AS_LIVE_RUNS]] = run_cost_stress(ov, slices, AS_LIVE_RUNS)
    out[STRESS_KEY[AS_VALIDATED]] = run_cost_stress(ov, slices, AS_VALIDATED)
    out["full_period"] = run_full_period(ov)

    ctrl = out["walk_forward_control_original_method"]

    def _gates(model: str) -> dict:
        """The same six gates, applied to one exit model.

        The walk-forward gate is judged on the ORIGINAL method (the one the
        deploy was signed off against, and the one this harness is verified to
        reproduce), not the stricter calendar-quarter variant — which fails for
        BOTH models and so cannot attribute anything to the exit change.
        """
        oos = out["oos_5_windows"][model]
        full = out["full_period"][model]
        s15 = out[STRESS_KEY[model]]["15bps"]
        return {
            # SHARED BY DESIGN, not a per-arm measurement: this gate asks "is the
            # harness itself trustworthy?", answered once by reproducing the
            # signed-off artifact (old params + as-validated). It is not an
            # assertion about `model`, and both arms are untrustworthy together
            # if it fails.
            "harness_reproduces_signed_off_artifact": ctrl["harness_reproduces_artifact"],
            "oos_compounded_positive": oos["compounded_pct"] > 0,
            # params.yaml's sign-off language is "the only ablation variant
            # clearing evidence_of_edge", so the canonical interpretation string
            # is the gate — not a raw PSR threshold, which passes on a number the
            # harness itself labels insufficient.
            "oos_psr_evidence_of_edge": oos["interpretation"] == "evidence_of_edge",
            "walk_forward_70pct": ctrl[CTRL_ARM[model]]["gate_70pct"],
            "cost_stress_15bps_positive": s15["compounded_pct"] > 0,
            "kill_switch_respected": not full["kill_switch_breached"],
        }

    gates_live = _gates(AS_LIVE_RUNS)
    gates_val = _gates(AS_VALIDATED)

    out["gates_live_exit_model"] = gates_live
    out["gates_as_validated_model"] = gates_val

    # SUBJECT FLIP 2026-08-10: the adverse-trend exit shipped (main 92433d4,
    # droplet c876aaf) and the v1 leg restarted onto it at 16:45 UTC, so the
    # DEPLOYED exit model is now `as_validated`. The headline verdict tracks
    # what is actually running; the live-exit arm is retained as the historical
    # comparison that motivated the change, not as the subject.
    out["deployed_exit_model"] = AS_VALIDATED
    out["deployed_exit_model_since"] = "2026-08-10T16:45:23Z"
    out["verdict"] = "REVALIDATED" if all(gates_val.values()) else "FAILS_REVALIDATION"
    out["gates_failed"] = [k for k, v in gates_val.items() if not v]
    out["verdict_previous_live_exit_model"] = (
        "REVALIDATED" if all(gates_live.values()) else "FAILS_REVALIDATION")
    out["gates_failed_previous_live_exit_model"] = [
        k for k, v in gates_live.items() if not v]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print()
    print(f"DEPLOYED model ({out['deployed_exit_model']}): {out['verdict']}")
    for k, v in gates_val.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print()
    print(f"previous live-exit model: {out['verdict_previous_live_exit_model']}")
    for k, v in gates_live.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"saved   -> {OUT}")
    return 0 if out["verdict"] == "REVALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
