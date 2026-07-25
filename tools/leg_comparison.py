"""
Side-by-side comparison of every candidate/deployed leg, including order cadence.

God asked for one table: the SOL candidate vs multifactor-v1 vs donchian-v3 vs
cnh-hybrid-short, with **how often each actually fires**.

Sizing caveat, stated up front because it is the easiest thing to misread:
each leg is run at ITS OWN sizing, not a common one —
  * multifactor-v1  → config/params.yaml           (risk 3.5%, lev 20)
  * donchian-v3     → config/params_donchian.yaml  (risk 2.75%, lev 20)
  * SOL supertrend  → the round-3 recommendation    (risk 4.0%, lev 3)
  * cnh-hybrid-short→ its own harness, which compounds a fixed fractional
                      net_pct per trade rather than risk-sizing off a stop.
So the `ret%`/`maxDD%` columns compare *legs as they would run*, NOT strategy
quality at equal risk. For equal-risk comparisons see tools/sol_leg_blend_confirm.py
(everything bisected to a common -30% DD). Win rate, profit factor and the
cadence columns are sizing-independent and directly comparable.

cnh-hybrid-short does not go through backtesting.py — it has its own detector +
simulator (tools/icnh_mega_sweep.py) with the Phase-1 locked params copied from
tools/cross_coin_backtest.py::_run_cnh_hybrid_per_window, so the numbers stay
comparable to the existing BTC/SOL validation.

Run: .venv/bin/python tools/leg_comparison.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import STRATEGIES, run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402
from tools.cross_coin_backtest import _load_coin_4h  # noqa: E402
from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
from tools.icnh_mega_sweep import Config, simulate_trades  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
START = datetime(2022, 4, 1, tzinfo=UTC)
END = datetime(2026, 7, 25, tzinfo=UTC)
YEARS = 4.32

# Phase-1 locked cnh-hybrid-short config (verbatim from cross_coin_backtest).
_DT_CFG = dict(
    name="hybrid_dt", pattern_type="distribution_top", direction="short",
    tf="4h", uptrend_bars=16, chop_bars=8, min_rise_pct=2.5,
    max_chop_ratio=0.55, require_chop_at_top=True,
    breakdown_mode="chop_low_or_ema24", sl_atr_mult=1.5, regime_sl_mode="off",
    tp_emas=("ema100",), entry_emas=("ema24",), dedup_bars=15,
)
_ICNH_CFG = dict(
    name="hybrid_icnh", pattern_type="inverse_cnh", direction="short",
    tf="4h", cup_len=20, handle_len=4, min_r2=0.50, min_cup_depth_atr=1.0,
    handle_max_depth_frac=0.70, peak_tolerance=6, entry_emas=("ema24",),
    sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",), dedup_bars=15,
)


def deployed_donchian_params() -> StrategyParams:
    """Load config/params_donchian.yaml *correctly*.

    `StrategyParams.from_yaml` has no donchian fields in its constructor block,
    so it silently drops `donchian_period_entry` (80 -> dataclass default 20)
    and `slope_trend_threshold_pct` (0.03 -> 0.0, i.e. regime gate OFF). That
    turns the deployed 80-bar gated breakout into a 20-bar ungated one: 454
    trades instead of 135, and max-DD -63.6% instead of -32.9%.

    The LIVE bot is unaffected — strategy/live_donchian_v3.py:119,125 reads the
    YAML dict directly via `s.get(...)`. This is a backtest-loader bug only, but
    any harness using from_yaml on a donchian config measures the wrong system.
    """
    import dataclasses

    import yaml
    path = REPO / "config" / "params_donchian.yaml"
    with open(path) as f:
        s = yaml.safe_load(f)["strategy"]
    base = StrategyParams.from_yaml(str(path))
    return dataclasses.replace(
        base,
        donchian_period_entry=int(s["donchian_period_entry"]),
        donchian_period_exit=int(s["donchian_period_exit"]),
        slope_trend_threshold_pct=float(s["slope_trend_threshold_pct"]),
        regime_ema_period=int(s["regime_ema_period"]),
        regime_slope_window=int(s["regime_slope_window"]),
    )


def _streak(flags: list[bool], val: bool) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f == val else 0
        best = max(best, cur)
    return best


def _underwater_days(eq: pd.Series) -> int:
    peak = eq.cummax()
    under = eq < peak * 0.999
    gaps, cur = [], None
    for ts, u in under.items():
        if u and cur is None:
            cur = ts
        elif not u and cur is not None:
            gaps.append((ts - cur).days)
            cur = None
    if cur is not None:
        gaps.append((under.index[-1] - cur).days)
    return max(gaps) if gaps else 0


def stats_from(label: str, entries: pd.Series, rets: np.ndarray,
               eq: pd.Series, sizing: str, direction: str) -> dict:
    """rets = per-trade net fractional return contributions (for WR/PF only)."""
    entries = pd.to_datetime(pd.Series(entries)).sort_values().reset_index(drop=True)
    gaps_d = entries.diff().dt.total_seconds().div(86400).dropna()
    wins, losses = rets[rets > 0], rets[rets < 0]
    total = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    peak = eq.cummax()
    monthly = eq.resample("ME").last().pct_change().dropna()
    return {
        "leg": label, "sizing": sizing, "direction": direction,
        "ret_pct": total,
        "cagr_pct": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / YEARS) - 1) * 100.0,
        "max_dd_pct": ((eq / peak - 1.0) * 100.0).min(),
        "win_rate_pct": 100.0 * (rets > 0).mean() if len(rets) else float("nan"),
        "profit_factor": (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
        "trades": len(rets),
        "trades_per_yr": len(rets) / YEARS,
        "median_days_between": float(gaps_d.median()) if len(gaps_d) else float("nan"),
        "mean_days_between": float(gaps_d.mean()) if len(gaps_d) else float("nan"),
        "max_days_between": float(gaps_d.max()) if len(gaps_d) else float("nan"),
        "max_lose_streak": _streak(list(rets > 0), False),
        "days_underwater": _underwater_days(eq),
        "pos_months_pct": 100.0 * (monthly > 0).mean() if len(monthly) else float("nan"),
    }


def bt_leg(label: str, key: str, symbol: str, tf: str, sizing: str,
           direction: str, params_path: str | None = None,
           attrs: dict | None = None, risk: float | None = None,
           leverage: int | None = None,
           params: StrategyParams | None = None) -> dict:
    cls = STRATEGIES[key]
    for k, v in (attrs or {}).items():
        setattr(cls, k, v)
    if risk is not None:
        for ra in ("st_risk_per_trade_pct", "rider_risk_per_trade_pct",
                   "sol_risk_per_trade_pct"):
            if hasattr(cls, ra):
                setattr(cls, ra, risk)
    if params is None and params_path:
        params = StrategyParams.from_yaml(params_path)
    if risk is not None:
        import dataclasses
        base = params or StrategyParams.from_yaml()
        params = dataclasses.replace(base, risk_per_trade_pct=risk,
                                     leverage=leverage or base.leverage)
    lev = leverage if leverage is not None else (params.leverage if params else None)
    r = run_backtest(key, symbol, tf, START, END, leverage=lev, quiet=True,
                     params_override=params, return_trades=True, return_equity=True)
    t = r["trades_df"].sort_values("EntryTime")
    eq = r["equity_series"].astype(float)
    return stats_from(label, t["EntryTime"], t["ReturnPct"].to_numpy(), eq,
                      sizing, direction)


def cnh_leg(label: str, coin: str) -> dict:
    df = _load_coin_4h(coin)
    # _load_coin_4h keeps the parquet's tz-aware index; everything downstream
    # here (and the seed timestamp below) is tz-naive, so normalise once.
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    sub = df.loc[str(START.date()):str(END.date())]
    hits = find_hybrid_patterns(sub, Config(**_DT_CFG), Config(**_ICNH_CFG))
    dt_idx = [h for h, s in hits if s == "DT"]
    ic_idx = [h for h, s in hits if s == "ICNH"]
    trades = (simulate_trades(sub, dt_idx, Config(**_DT_CFG), label)
              + simulate_trades(sub, ic_idx, Config(**_ICNH_CFG), label))
    trades.sort(key=lambda t: t["entry_ts"])
    nets = np.array([t["net_pct"] for t in trades])
    ent = pd.to_datetime([t["entry_ts"] for t in trades])
    ext = pd.to_datetime([t["exit_ts"] for t in trades])
    # Equity curve stamped at EXIT time (that is when P&L is realised), so the
    # drawdown reflects realised equity. Intra-trade excursions are invisible in
    # this harness, so max_dd here is a floor, not the true peak-to-trough.
    eq = pd.Series(np.cumprod(1.0 + nets), index=ext).sort_index()
    eq = pd.concat([pd.Series([1.0], index=[pd.Timestamp(START).tz_localize(None)]), eq])
    return stats_from(label, ent, nets, eq,
                      "fixed-fraction per trade (own harness)", "short only")


def main() -> int:
    rows = []
    print("Running legs...", flush=True)

    rows.append(bt_leg("multifactor-v1 (BTC 15m)", "multifactor-v1",
                       "BTC/USDT:USDT", "15m",
                       "params.yaml risk 3.5% lev 20", "long+short"))
    print("  v1 done", flush=True)

    rows.append(bt_leg("donchian-v3 (BTC 4h)", "donchian-v3",
                       "BTC/USDT:USDT", "4h",
                       "params_donchian.yaml risk 2.75% lev 20", "long+short",
                       params=deployed_donchian_params()))
    print("  donchian done", flush=True)

    # Kept visible: what the naive from_yaml load measures instead. Same YAML,
    # same risk — the only difference is the two silently-dropped keys.
    rows.append(bt_leg("  └ donchian via from_yaml (WRONG)", "donchian-v3",
                       "BTC/USDT:USDT", "4h",
                       "from_yaml drops entry=80 + gate=0.03", "long+short",
                       params_path=str(REPO / "config" / "params_donchian.yaml")))

    rows.append(bt_leg("SOL supertrend (4h) *candidate*", "supertrend",
                       "SOL/USDT:USDT", "4h", "risk 4.0% lev 3 (round-3 rec)",
                       "long+short",
                       attrs={"st_period": 14, "st_multiplier": 3.5,
                              "st_sl_atr": 2.0, "st_tp_atr": 10.0,
                              "allow_shorts": True},
                       risk=4.0, leverage=3))
    print("  SOL supertrend done", flush=True)

    for coin in ("BTC", "SOL"):
        try:
            rows.append(cnh_leg(f"cnh-hybrid-short ({coin} 4h)", coin))
            print(f"  cnh {coin} done", flush=True)
        except Exception as exc:
            print(f"  [warn] cnh {coin}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print()
    print("=" * 132)
    print(f"LEG COMPARISON — {START.date()} → {END.date()} ({YEARS} yr), "
          "each leg at its OWN sizing (see header note)")
    print("=" * 132)
    hdr = (f"{'leg':<34}{'ret%':>9}{'CAGR%':>7}{'maxDD%':>8}{'WR%':>6}{'PF':>6}"
           f"{'n':>5}{'n/yr':>6}{'medGap':>7}{'maxGap':>7}{'loseStk':>8}"
           f"{'daysUW':>7}{'posMo%':>7}")
    print(hdr)
    print("-" * 132)
    for r in rows:
        print(f"{r['leg']:<34}{r['ret_pct']:>9.1f}{r['cagr_pct']:>7.1f}"
              f"{r['max_dd_pct']:>8.1f}{r['win_rate_pct']:>6.1f}"
              f"{r['profit_factor']:>6.2f}{r['trades']:>5}{r['trades_per_yr']:>6.1f}"
              f"{r['median_days_between']:>7.1f}{r['max_days_between']:>7.0f}"
              f"{r['max_lose_streak']:>8}{r['days_underwater']:>7}"
              f"{r['pos_months_pct']:>7.1f}")
    print("-" * 132)
    print("medGap/maxGap = days between consecutive ENTRIES (order cadence). "
          "loseStk = worst run of consecutive losers.")
    print("daysUW = longest stretch below the previous equity high.")

    print()
    print("Sizing / direction per leg:")
    for r in rows:
        print(f"  {r['leg']:<34} {r['direction']:<12} {r['sizing']}")

    path = REPO / "reports" / "leg_comparison.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
