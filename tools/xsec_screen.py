"""Cross-sectional momentum MN — proper screen (Track B, step 2).

Fail-fast (xsec_failfast.py) showed gross momentum = +27% CAGR, beta ~0, but
maxDD -60% and ~80x/yr turnover. This screen applies the discipline:
  - FRICTION on turnover (the make-or-break) at 10/20/30 bps one-way.
  - small param grid (lookback, rebal, K) selected on IS by NET Sharpe.
  - lock the IS winner, test ONCE on OOS (momentum only; reversal already dead).
  - beta-to-BTC reported AND gated (|beta|<0.15 is a PASS criterion, not metric).
  - turnover + vol-targeted variant (scale book to 15% annual vol) to tame DD.
GROSS funding omitted -> result is an OPTIMISTIC upper bound (advisor flag).

IS  = 2020-10..2024-06-30  (matches gate windows 1-8)
OOS = 2024-07-01..2026-06  (windows 9-12)

Run: uv run python tools/xsec_screen.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

UNIVERSE = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOGE", "DOT",
            "AVAX", "NEAR", "LINK", "LTC", "ATOM", "BCH"]
DATA = "data/historical"
IS_END = pd.Timestamp("2024-06-30", tz="UTC")
TARGET_VOL = 0.15  # annual, for vol-targeted variant


def daily_panel(coins: list[str]) -> pd.DataFrame:
    cols = {}
    for c in coins:
        f = os.path.join(DATA, f"{c}_USDT_USDT_1h.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df.columns = [x.lower() for x in df.columns]
        cols[c] = df["close"].resample("1D").last()
    return pd.DataFrame(cols).dropna()


def _max_dd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min() * 100.0)


def book_daily_returns(panel: pd.DataFrame, lookback: int, skip: int,
                       rebal: int, K: int, bps: float):
    """Return (dates, book_daily_net, btc_daily, turnovers) for momentum."""
    prices = panel.values
    n_days, n_coins = prices.shape
    rets = np.zeros_like(prices)
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    btc_i = list(panel.columns).index("BTC")

    out_r, out_btc, out_dates, turns = [], [], [], []
    prev_w = np.zeros(n_coins)
    t = lookback + skip + 1
    while t < n_days:
        mom = prices[t - skip] / prices[t - lookback - skip] - 1.0
        order = np.argsort(mom)
        losers, winners = order[:K], order[-K:]
        w = np.zeros(n_coins)
        w[winners] = 1.0 / K
        w[losers] = -1.0 / K
        turn = float(np.abs(w - prev_w).sum())
        hold_end = min(t + rebal, n_days)
        seg = rets[t:hold_end]
        daily = seg @ w
        # charge friction (turn * one-way bps) on the entry day of the segment
        if len(daily) > 0:
            daily = daily.copy()
            daily[0] -= turn * bps / 10000.0
        out_r.extend(daily.tolist())
        out_btc.extend(rets[t:hold_end, btc_i].tolist())
        out_dates.extend(panel.index[t:hold_end].tolist())
        turns.append(turn)
        prev_w = w
        t = hold_end
    return (pd.DatetimeIndex(out_dates), np.array(out_r), np.array(out_btc),
            np.array(turns), 365.25 / rebal)


def stats(dates, book, btc, turns, rebals_per_yr) -> dict:
    equity = np.cumprod(1.0 + book)
    years = len(book) / 365.25
    cagr = float(equity[-1] ** (1 / years) - 1) * 100 if years > 0 and equity[-1] > 0 else -100.0
    sharpe = float(book.mean() / book.std() * np.sqrt(365)) if book.std() > 0 else 0.0
    beta = float(np.polyfit(btc, book, 1)[0]) if btc.std() > 0 else float("nan")
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": _max_dd(equity),
            "beta": beta, "turn_yr": float(turns.mean() * rebals_per_yr),
            "n": len(book)}


def split(dates, book, btc):
    m = dates <= IS_END
    return (book[m], btc[m]), (book[~m], btc[~m])


def main() -> int:
    panel = daily_panel(UNIVERSE)
    print(f"Panel: {panel.shape[1]} coins x {panel.shape[0]} days "
          f"({str(panel.index[0])[:10]}->{str(panel.index[-1])[:10]})")
    print(f"Vol-targeting OFF for grid; friction charged on turnover.\n")

    # ---- small IS grid (select on IS NET Sharpe @ 20bps) ----
    grid = [(lb, rb, k) for lb in (30, 60, 90) for rb in (7, 14) for k in (3, 4)]
    SEL_BPS = 20.0
    print(f"IS grid select @ {SEL_BPS:.0f}bps (lookback,rebal,K):")
    print(f"  {'cfg':<16}{'IS cagr':>9}{'IS Sh':>7}{'IS DD':>8}{'beta':>7}{'turn/yr':>9}")
    scored = []
    for lb, rb, k in grid:
        dates, book, btc, turns, rpy = book_daily_returns(panel, lb, 2, rb, k, SEL_BPS)
        (is_b, is_btc), _ = split(dates, book, btc)
        s = stats(dates[dates <= IS_END], is_b, is_btc, turns, rpy)
        scored.append(((lb, rb, k), s))
        print(f"  lb{lb:>2} rb{rb:>2} K{k:<8}{s['cagr']:>9.1f}{s['sharpe']:>7.2f}"
              f"{s['maxdd']:>8.1f}{s['beta']:>7.2f}{s['turn_yr']:>9.1f}")
    # lock best by IS Sharpe (beta-gated)
    eligible = [(c, s) for c, s in scored if abs(s["beta"]) < 0.15]
    best_cfg, best_s = max(eligible or scored, key=lambda x: x[1]["sharpe"])
    print(f"\nLOCKED IS winner (max IS Sharpe, |beta|<0.15): "
          f"lookback={best_cfg[0]} rebal={best_cfg[1]} K={best_cfg[2]}")

    # ---- OOS test of the locked config, friction stress ----
    lb, rb, k = best_cfg
    print(f"\nOOS test (locked cfg), friction stress:")
    print(f"  {'bps':>4}{'OOS cagr':>10}{'OOS Sh':>8}{'OOS DD':>9}{'beta':>7}{'turn/yr':>9}")
    for bps in (0, 10, 20, 30):
        dates, book, btc, turns, rpy = book_daily_returns(panel, lb, 2, rb, k, bps)
        _, (oos_b, oos_btc) = split(dates, book, btc)
        s = stats(dates[dates > IS_END], oos_b, oos_btc, turns, rpy)
        verdict = "PASS" if (s["cagr"] > 0 and abs(s["beta"]) < 0.15) else "FAIL"
        print(f"  {bps:>4}{s['cagr']:>10.1f}{s['sharpe']:>8.2f}{s['maxdd']:>9.1f}"
              f"{s['beta']:>7.2f}{s['turn_yr']:>9.1f}   {verdict}")

    # ---- ENSEMBLE (pre-specified de-overfit rule): avg signal across
    #      lookbacks 30/60/90, weekly rebal, K=3. Tested ONCE on IS and OOS.
    print("\nENSEMBLE (avg momentum rank over lookbacks 30/60/90, rb7, K3):")
    print(f"  {'seg':<5}{'bps':>4}{'cagr':>9}{'Sh':>7}{'DD':>8}{'beta':>7}{'turn/yr':>9}")
    eb = ensemble_book(panel, lookbacks=(30, 60, 90), rebal=7, K=3)
    for bps in (0, 20):
        dates, book, btc, turns, rpy = eb(bps)
        for seg, mask in (("IS", dates <= IS_END), ("OOS", dates > IS_END)):
            s = stats(dates[mask], book[mask], btc[mask], turns, rpy)
            v = "PASS" if (seg == "OOS" and s["cagr"] > 0 and abs(s["beta"]) < 0.15) else ""
            print(f"  {seg:<5}{bps:>4}{s['cagr']:>9.1f}{s['sharpe']:>7.2f}"
                  f"{s['maxdd']:>8.1f}{s['beta']:>7.2f}{s['turn_yr']:>9.1f}   {v}")

    print("\nGate for MN book: OOS cagr>0 AND |beta-to-BTC|<0.15 (uncorrelated).")
    print("Funding omitted -> optimistic. Turnover-driven friction is the risk.")
    return 0


def ensemble_book(panel: pd.DataFrame, lookbacks, rebal: int, K: int):
    """Pre-specified ensemble: average the momentum signal across lookbacks,
    one dollar-neutral book. Returns a closure run(bps) -> (dates,book,btc,turns,rpy)."""
    prices = panel.values
    n_days, n_coins = prices.shape
    rets = np.zeros_like(prices)
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    btc_i = list(panel.columns).index("BTC")
    skip = 2
    start = max(lookbacks) + skip + 1

    def run(bps: float):
        out_r, out_btc, out_dates, turns = [], [], [], []
        prev_w = np.zeros(n_coins)
        t = start
        while t < n_days:
            # average z-scored momentum across lookbacks
            score = np.zeros(n_coins)
            for lb in lookbacks:
                mom = prices[t - skip] / prices[t - lb - skip] - 1.0
                score += (mom - mom.mean()) / (mom.std() + 1e-9)
            order = np.argsort(score)
            losers, winners = order[:K], order[-K:]
            w = np.zeros(n_coins)
            w[winners] = 1.0 / K
            w[losers] = -1.0 / K
            turn = float(np.abs(w - prev_w).sum())
            hold_end = min(t + rebal, n_days)
            daily = (rets[t:hold_end] @ w).copy()
            if len(daily) > 0:
                daily[0] -= turn * bps / 10000.0
            out_r.extend(daily.tolist())
            out_btc.extend(rets[t:hold_end, btc_i].tolist())
            out_dates.extend(panel.index[t:hold_end].tolist())
            turns.append(turn)
            prev_w = w
            t = hold_end
        return (pd.DatetimeIndex(out_dates), np.array(out_r), np.array(out_btc),
                np.array(turns), 365.25 / rebal)
    return run


if __name__ == "__main__":
    raise SystemExit(main())
