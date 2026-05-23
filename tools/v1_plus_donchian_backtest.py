"""Backtest multifactor-v1 + donchian-v3 + their 50/50 combined portfolio
on the same 5 OOS windows that locked v1 for production.

Windows match `reports/path2_oos_results.json`:
  2022H1, 2023H1, 2024H1, 2024H2, 2025H1

Donchian-v3 uses the best_combo from `reports/oos_donchian-v3_*.json`
(commit bf4a9dc): 40-bar entry channel, 10-bar exit, ATR-SL 1.5×,
slope-gate OFF, time-stop 48 bars on 4h timeframe.

Output: reports/v1_donchian_combined_<UTC>.json + console table.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import run_backtest  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402

WINDOWS: list[tuple[str, str, str]] = [
    ("2022H1", "2022-01-01", "2022-06-30"),
    ("2023H1", "2023-01-01", "2023-06-30"),
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-05-31"),
]

SYMBOL = "BTC/USDT:USDT"


def donchian_params(combo: str) -> StrategyParams:
    """Two historical best_combos from reports/oos_donchian-v3_*.json:

    - "agg" (40/10/gate-off): tuned on 2022-06..2024-12 IS → 2025H1 OOS (-0.5%).
       Looser entry channel, no regime gate. Trades more.
    - "cons" (80/20/gate-on): tuned on 2020-04..2021-12 IS → 2022H1 OOS (+23.3%).
       Tighter channel, slope gate at 3% requires confirmed trend. Trades less.

    Running BOTH gives an honest bracket — each is the period-appropriate
    walk-forward choice for one window but pure forward-test on the others.
    """
    p = StrategyParams.from_yaml()
    if combo == "agg":
        return dataclasses.replace(
            p,
            donchian_period_entry=40, donchian_period_exit=10,
            atr_sl_multiple=1.5, atr_tp_multiple=1.5, atr_trail_multiple=0.0,
            leverage=20, regime_ema_period=120, regime_slope_window=30,
            slope_trend_threshold_pct=0.0, time_stop_bars=48,
            volume_multiple=1.0,
        )
    if combo == "cons":
        return dataclasses.replace(
            p,
            donchian_period_entry=80, donchian_period_exit=20,
            atr_sl_multiple=1.5, atr_tp_multiple=1.5, atr_trail_multiple=0.0,
            leverage=20, regime_ema_period=120, regime_slope_window=30,
            slope_trend_threshold_pct=0.03, time_stop_bars=48,
            volume_multiple=1.0,
        )
    raise ValueError(f"unknown combo: {combo}")


def to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def annualised_sharpe(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(365))


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100.0)


def run_one(window_label: str, start: str, end: str) -> dict:
    print(f"\n=== Window {window_label} ({start} → {end}) ===")
    s = to_dt(start)
    e = to_dt(end)

    # multifactor-v1 (15m + 1h, default params from config/params.yaml)
    print("  multifactor-v1 (15m) ...")
    v1 = run_backtest(
        strategy_name="multifactor-v1",
        symbol=SYMBOL,
        timeframe="15m",
        start=s,
        end=e,
        quiet=True,
        return_equity=True,
    )
    print(
        f"    trades={v1['trades']} ret={v1.get('after_funding_pct', v1['backtest_return_pct']):+.2f}%"
        f" sharpe={v1.get('sharpe', float('nan')):+.2f} dd={v1.get('max_drawdown_pct', float('nan')):.2f}%"
    )

    out = {
        "window": window_label,
        "start": start,
        "end": end,
        "v1": {
            "trades": v1["trades"],
            "ret_pct": v1.get("after_funding_pct", v1["backtest_return_pct"]),
            "sharpe": v1.get("sharpe"),
            "max_dd_pct": v1.get("max_drawdown_pct"),
            "win_rate_pct": v1.get("win_rate_pct"),
        },
    }
    v1_norm = (v1["equity_series"] / float(v1["equity_series"].iloc[0])).resample("1D").last().ffill()

    for combo in ("agg", "cons"):
        print(f"  donchian-v3 (4h, {combo}) ...")
        d3 = run_backtest(
            strategy_name="donchian-v3",
            symbol=SYMBOL,
            timeframe="4h",
            start=s,
            end=e,
            params_override=donchian_params(combo),
            quiet=True,
            return_equity=True,
        )
        print(
            f"    trades={d3['trades']} ret={d3.get('after_funding_pct', d3['backtest_return_pct']):+.2f}%"
            f" sharpe={d3.get('sharpe', float('nan')):+.2f} dd={d3.get('max_drawdown_pct', float('nan')):.2f}%"
        )

        d3_norm = (d3["equity_series"] / float(d3["equity_series"].iloc[0])).resample("1D").last().ffill()
        common = v1_norm.index.intersection(d3_norm.index)
        v1n = v1_norm.loc[common]
        d3n = d3_norm.loc[common]
        combined = 0.5 * v1n + 0.5 * d3n
        v1_dr = v1n.pct_change().dropna()
        d3_dr = d3n.pct_change().dropna()
        combined_dr = combined.pct_change().dropna()

        corr = float(v1_dr.corr(d3_dr)) if len(v1_dr) > 5 else float("nan")
        combined_ret_pct = (float(combined.iloc[-1]) - 1.0) * 100.0
        combined_dd_pct = max_drawdown_pct(combined)
        combined_sharpe = annualised_sharpe(combined_dr)
        print(
            f"    combined 50/50 ret={combined_ret_pct:+.2f}%"
            f" sharpe={combined_sharpe:+.2f} dd={combined_dd_pct:.2f}%"
            f" corr(v1,d3)={corr:+.2f}"
        )

        out[f"donchian_v3_{combo}"] = {
            "trades": d3["trades"],
            "ret_pct": d3.get("after_funding_pct", d3["backtest_return_pct"]),
            "sharpe": d3.get("sharpe"),
            "max_dd_pct": d3.get("max_drawdown_pct"),
            "win_rate_pct": d3.get("win_rate_pct"),
        }
        out[f"combined_50_50_{combo}"] = {
            "ret_pct": combined_ret_pct,
            "sharpe": combined_sharpe,
            "max_dd_pct": combined_dd_pct,
            "corr_v1_d3_daily": corr,
        }
    return out


def main() -> int:
    results = []
    for label, start, end in WINDOWS:
        try:
            results.append(run_one(label, start, end))
        except Exception as exc:
            print(f"  FAILED {label}: {exc}")
            results.append({"window": label, "error": repr(exc)})

    # Cumulative across windows (compounded)
    def cumret(side: str) -> float:
        eq = 1.0
        for r in results:
            if "error" in r or side not in r:
                continue
            pct = r[side]["ret_pct"]
            if pct is None or not math.isfinite(pct):
                continue
            eq *= 1.0 + pct / 100.0
        return (eq - 1.0) * 100.0

    print("\n=== Summary across 5 windows ===")
    print(
        f"{'window':>8} {'v1 %':>8} "
        f"{'d3-agg %':>10} {'agg combo':>10} {'agg ρ':>7} "
        f"{'d3-con %':>10} {'con combo':>10} {'con ρ':>7}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['window']:>8}  ERROR")
            continue
        print(
            f"{r['window']:>8} "
            f"{r['v1']['ret_pct']:>+8.2f} "
            f"{r['donchian_v3_agg']['ret_pct']:>+10.2f} "
            f"{r['combined_50_50_agg']['ret_pct']:>+10.2f} "
            f"{r['combined_50_50_agg']['corr_v1_d3_daily']:>+7.2f} "
            f"{r['donchian_v3_cons']['ret_pct']:>+10.2f} "
            f"{r['combined_50_50_cons']['ret_pct']:>+10.2f} "
            f"{r['combined_50_50_cons']['corr_v1_d3_daily']:>+7.2f}"
        )

    print(
        f"\nCompounded across 5 windows:"
        f"\n  v1                       {cumret('v1'):+8.2f}%"
        f"\n  donchian-v3 (agg 40/10)  {cumret('donchian_v3_agg'):+8.2f}%"
        f"\n  combined 50/50  (agg)    {cumret('combined_50_50_agg'):+8.2f}%"
        f"\n  donchian-v3 (cons 80/20) {cumret('donchian_v3_cons'):+8.2f}%"
        f"\n  combined 50/50  (cons)   {cumret('combined_50_50_cons'):+8.2f}%"
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / "reports" / f"v1_donchian_combined_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
