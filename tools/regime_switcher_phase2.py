"""Phase 2 + 3 of the vol-regime switcher: realistic threshold-based
switcher with train/test split.

Design (based on Phase 1 findings):
  - Daily decision: compute ATR(14) percentile rank over rolling 90d, shifted 1d.
  - If pctile < THRESHOLD → use v1 (mean-reversion in calm markets).
  - Else → use Donchian-v3 cons (breakout in volatile markets).
  - Hysteresis: switch only when N consecutive days agree (avoid whipsaws).

Validation:
  - Train: 2019-10..2022-12 (sweep threshold + hysteresis, pick by Sharpe).
  - Test:  2023-01..2026-05 (apply trained params, report OOS).
  - Also: 5 OOS windows (2022H1..2025H1) for apples-to-apples vs prior reports.

Output:
  reports/regime_switcher_<UTC>.json     — per-config stats + trained params
  reports/regime_switcher_<UTC>_eq.csv   — switcher daily equity curve
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr  # noqa: E402

DATA = ROOT / "data" / "historical"
REPORTS = ROOT / "reports"

LOOKBACK = 90
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"


def latest_ts() -> str:
    cands = sorted(REPORTS.glob("full_history_*_v1_equity.csv"))
    if not cands:
        raise RuntimeError("no equity CSVs found")
    return cands[-1].stem.replace("full_history_", "").replace("_v1_equity", "")


def daily_atr_pctile() -> pd.Series:
    k1h = pd.read_parquet(DATA / "BTC_USDT_USDT_1h.parquet")
    ts_col = next((c for c in k1h.columns if c.lower() in ("timestamp", "ts", "time", "datetime")), None)
    if ts_col:
        k1h["_ts"] = pd.to_datetime(k1h[ts_col], utc=True)
        k1h = k1h.set_index("_ts")
    k1h = k1h.sort_index()
    if k1h.index.tz is not None:
        k1h.index = k1h.index.tz_convert("UTC").tz_localize(None)
    daily = k1h[["high", "low", "close"]].resample("1D").agg(
        {"high": "max", "low": "min", "close": "last"}
    )
    a = atr(daily["high"], daily["low"], daily["close"], 14)
    pct = a.rolling(LOOKBACK, min_periods=30).apply(
        lambda s: s.rank(pct=True).iloc[-1] if len(s) else np.nan, raw=False
    )
    return pct.shift(1).dropna()


def daily_returns(name: str, ts: str) -> pd.Series:
    df = pd.read_csv(REPORTS / f"full_history_{ts}_{name}_equity.csv", index_col=0, parse_dates=True)
    eq = df["equity_norm"]
    if eq.index.tz is not None:
        eq.index = eq.index.tz_convert("UTC").tz_localize(None)
    return eq.resample("1D").last().ffill().pct_change().dropna()


def apply_switcher(pct: pd.Series, v1: pd.Series, d3: pd.Series,
                   threshold: float, hyst_days: int) -> tuple[pd.Series, pd.Series, int]:
    """Return (daily_returns_of_switcher, choice_per_day, num_switches).

    choice_per_day: 1 for v1, 0 for d3.
    Hysteresis: must agree for `hyst_days` consecutive days before switching.
    """
    df = pd.concat({"pct": pct, "v1": v1, "d3": d3}, axis=1).dropna()
    # 1 = v1 favored (low vol), 0 = d3 favored
    raw_choice = (df["pct"] < threshold).astype(int)
    if hyst_days <= 1:
        choice = raw_choice
    else:
        # Use Donchian (0) unless v1 has been favored for hyst_days consecutive days,
        # and stick until d3 has been favored for hyst_days consecutive days.
        choice = pd.Series(0, index=df.index, dtype=int)
        run = 0
        current = 0   # start in d3
        for i, raw in enumerate(raw_choice.values):
            if raw == current:
                run = 0
            else:
                run += 1
                if run >= hyst_days:
                    current = raw
                    run = 0
            choice.iloc[i] = current
    rets = np.where(choice == 1, df["v1"], df["d3"])
    switcher_ret = pd.Series(rets, index=df.index)
    num_switches = int((choice.diff().abs() > 0).sum())
    return switcher_ret, choice, num_switches


def stats(rets: pd.Series, label: str = "") -> dict:
    if rets.empty:
        return {"label": label, "n": 0}
    eq = (1 + rets).cumprod()
    peak = eq.cummax()
    dd = float((eq / peak - 1).min() * 100)
    sharpe = float(rets.mean() / rets.std() * math.sqrt(365)) if rets.std() > 0 else 0.0
    ret_pct = float((eq.iloc[-1] - 1) * 100)
    return {
        "label": label,
        "n": int(len(rets)),
        "ret_pct": ret_pct,
        "sharpe": sharpe,
        "max_dd_pct": dd,
    }


def main() -> int:
    ts = latest_ts()
    print(f"Using backtest equity from run {ts}")
    pct = daily_atr_pctile()
    v1 = daily_returns("v1", ts)
    d3 = daily_returns("d3cons", ts)

    df = pd.concat({"pct": pct, "v1": v1, "d3": d3}, axis=1).dropna()
    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:]
    print(f"train: {len(train):,} days ({train.index.min().date()} → {train.index.max().date()})")
    print(f"test:  {len(test):,} days ({test.index.min().date()} → {test.index.max().date()})")

    # === Sweep threshold × hysteresis on train ===
    print("\n=== Threshold sweep (train, picking by Sharpe) ===")
    print(f"{'threshold':>10} {'hyst':>5} {'switches':>9} {'ret %':>10} {'sharpe':>8} {'maxDD %':>9}")
    best = None
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
        for hyst in (1, 2, 3, 5):
            sw_ret, choice, switches = apply_switcher(
                train["pct"], train["v1"], train["d3"], threshold, hyst
            )
            st = stats(sw_ret)
            print(f"{threshold:>10.2f} {hyst:>5d} {switches:>9d} "
                  f"{st['ret_pct']:>+10.2f} {st['sharpe']:>+8.3f} {st['max_dd_pct']:>+9.2f}")
            if best is None or st["sharpe"] > best["sharpe"]:
                best = {"threshold": threshold, "hyst": hyst, **st, "switches": switches}

    print(f"\nBest train config: threshold={best['threshold']:.2f}, hyst={best['hyst']}, "
          f"Sharpe={best['sharpe']:+.3f}, ret={best['ret_pct']:+.2f}%, switches={best['switches']}")

    # === Apply best to TEST (true OOS) ===
    test_sw_ret, test_choice, test_switches = apply_switcher(
        test["pct"], test["v1"], test["d3"],
        best["threshold"], best["hyst"]
    )

    # Baselines on test
    test_v1 = stats(test["v1"], "v1 alone (test)")
    test_d3 = stats(test["d3"], "Donchian-cons alone (test)")
    test_50_50 = stats(0.5 * test["v1"] + 0.5 * test["d3"], "50/50 (test)")
    test_switcher = stats(test_sw_ret, "Switcher (test, OOS)")
    test_switcher["switches"] = test_switches
    test_switcher["pct_days_v1"] = float((test_choice == 1).mean() * 100)

    print("\n=== TEST (true OOS, 2023-01..present) ===")
    for s in (test_v1, test_d3, test_50_50, test_switcher):
        sw = f"  switches={s.get('switches', '-')} {('pct_v1=' + str(round(s.get('pct_days_v1', 0), 1)) + '%') if 'pct_days_v1' in s else ''}"
        print(f"  {s['label']:<32} ret={s['ret_pct']:+8.2f}% sharpe={s['sharpe']:+.2f} dd={s['max_dd_pct']:+6.2f}%{sw}")

    # === Full continuous (in-sample + OOS) for HTML rendering ===
    full_sw_ret, full_choice, full_switches = apply_switcher(
        df["pct"], df["v1"], df["d3"], best["threshold"], best["hyst"]
    )
    full_50_50 = 0.5 * df["v1"] + 0.5 * df["d3"]
    full_v1 = stats(df["v1"], "v1")
    full_d3 = stats(df["d3"], "Donchian-cons")
    full_combo = stats(full_50_50, "50/50 combined")
    full_sw = stats(full_sw_ret, "Switcher")
    full_sw["switches"] = full_switches
    full_sw["pct_days_v1"] = float((full_choice == 1).mean() * 100)

    print("\n=== Full 6.7-year continuous (train+test combined) ===")
    for s in (full_v1, full_d3, full_combo, full_sw):
        sw = f"  switches={s.get('switches', '-')} {('pct_v1=' + str(round(s.get('pct_days_v1', 0), 1)) + '%') if 'pct_days_v1' in s else ''}"
        print(f"  {s['label']:<32} ret={s['ret_pct']:+8.2f}% sharpe={s['sharpe']:+.2f} dd={s['max_dd_pct']:+6.2f}%{sw}")

    # Save full switcher equity curve + JSON
    sw_eq = (1 + full_sw_ret).cumprod()
    sw_eq.index.name = "ts"
    pd.DataFrame({"equity_norm": sw_eq, "choice_v1": full_choice}).to_csv(
        REPORTS / f"regime_switcher_{ts}_eq.csv"
    )
    out = {
        "ts": ts,
        "lookback_days": LOOKBACK,
        "train_window": [str(train.index.min().date()), str(train.index.max().date())],
        "test_window":  [str(test.index.min().date()),  str(test.index.max().date())],
        "trained_params": {"threshold": best["threshold"], "hyst": best["hyst"]},
        "test_results": {
            "v1": test_v1, "donchian_cons": test_d3,
            "combo_50_50": test_50_50, "switcher": test_switcher,
        },
        "full_results": {
            "v1": full_v1, "donchian_cons": full_d3,
            "combo_50_50": full_combo, "switcher": full_sw,
        },
    }
    out_path = REPORTS / f"regime_switcher_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
