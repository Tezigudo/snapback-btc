"""Cross-sectional market-neutral — FAIL-FAST gross check (Track B, step 1).

Before building the full walk-forward+funding harness, answer one question:
does a reasonable cross-sectional long-short show ANY gross edge, and is it
actually beta-neutral? If flat/negative gross, stop. (advisor point 4)

Config (single reasonable point, NOT a grid):
  universe   = 14 full-history coins
  signal     = trailing LOOKBACK-day return, skip most recent SKIP days
  book       = long top-K, short bottom-K, equal-weight, dollar-neutral
  rebalance  = every REBAL days, hold REBAL days
  costs      = GROSS (no friction, no funding) — this is the optimistic check

Reports: gross total/annualized, Sharpe, maxDD, beta-to-BTC, turnover.
Also runs the REVERSAL direction for contrast (advisor point 1: survivorship
biases reversal UP and momentum DOWN — distrust reversal if it wins).

Run: uv run python tools/xsec_failfast.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

UNIVERSE = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOGE", "DOT",
            "AVAX", "NEAR", "LINK", "LTC", "ATOM", "BCH"]
LOOKBACK = 30   # days
SKIP = 2        # skip most-recent days (avoid 1-2d microstructure reversal)
REBAL = 7       # rebalance weekly
K = 3           # long top-3, short bottom-3
DATA = "data/historical"


def daily_panel(coins: list[str]) -> pd.DataFrame:
    cols = {}
    for c in coins:
        f = os.path.join(DATA, f"{c}_USDT_USDT_1h.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df.columns = [x.lower() for x in df.columns]
        daily = df["close"].resample("1D").last()
        cols[c] = daily
    panel = pd.DataFrame(cols).dropna()
    return panel


def _max_dd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min() * 100.0)


def run(panel: pd.DataFrame, direction: str) -> dict:
    """direction='mom' longs winners; 'rev' longs losers."""
    prices = panel.values
    dates = panel.index
    n_days, n_coins = prices.shape
    rets = np.zeros_like(prices)
    rets[1:] = prices[1:] / prices[:-1] - 1.0

    book_rets, btc_rets, turnovers = [], [], []
    btc_i = list(panel.columns).index("BTC")
    prev_w = np.zeros(n_coins)

    t = LOOKBACK + SKIP + 1
    while t < n_days:
        # signal: trailing return from t-LOOKBACK-SKIP to t-SKIP
        p_old = prices[t - LOOKBACK - SKIP]
        p_new = prices[t - SKIP]
        mom = p_new / p_old - 1.0
        order = np.argsort(mom)              # ascending: losers first
        losers, winners = order[:K], order[-K:]
        w = np.zeros(n_coins)
        if direction == "mom":
            w[winners] = 1.0 / K
            w[losers] = -1.0 / K
        else:  # reversal
            w[losers] = 1.0 / K
            w[winners] = -1.0 / K

        # hold REBAL days, accumulate book return (rebalanced at entry)
        hold_end = min(t + REBAL, n_days)
        seg = rets[t:hold_end]               # (h, n_coins) daily returns
        # dollar-neutral book daily return = sum(w * coin_daily_ret)
        daily_book = seg @ w
        book_rets.extend(daily_book.tolist())
        btc_rets.extend(rets[t:hold_end, btc_i].tolist())
        turnovers.append(float(np.abs(w - prev_w).sum()))
        prev_w = w
        t = hold_end

    book = np.array(book_rets)
    btc = np.array(btc_rets)
    equity = np.cumprod(1.0 + book)
    total = float(equity[-1] - 1.0) * 100.0
    years = len(book) / 365.25
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    sharpe = float(book.mean() / book.std() * np.sqrt(365)) if book.std() > 0 else 0.0
    beta = float(np.polyfit(btc, book, 1)[0]) if btc.std() > 0 else float("nan")
    corr = float(np.corrcoef(btc, book)[0, 1]) if btc.std() > 0 else float("nan")
    avg_turn = float(np.mean(turnovers))
    rebals_per_yr = 365.25 / REBAL
    return {
        "direction": direction, "total_pct": total, "cagr_pct": cagr,
        "sharpe": sharpe, "maxdd_pct": _max_dd(equity), "beta_btc": beta,
        "corr_btc": corr, "avg_turnover": avg_turn,
        "annual_turnover": avg_turn * rebals_per_yr,
        "n_days": len(book), "first": str(dates[0])[:10], "last": str(dates[-1])[:10],
    }


def main() -> int:
    panel = daily_panel(UNIVERSE)
    print(f"Panel: {panel.shape[1]} coins, {panel.shape[0]} days "
          f"({str(panel.index[0])[:10]} -> {str(panel.index[-1])[:10]})")
    print(f"Config: lookback={LOOKBACK}d skip={SKIP}d rebal={REBAL}d "
          f"K={K}/{K} dollar-neutral, GROSS (no friction/funding)\n")
    print(f"{'dir':<5}{'total%':>9}{'cagr%':>8}{'sharpe':>8}{'maxDD%':>9}"
          f"{'beta':>7}{'corr':>7}{'turn/yr':>9}")
    for d in ("mom", "rev"):
        r = run(panel, d)
        print(f"{r['direction']:<5}{r['total_pct']:>9.1f}{r['cagr_pct']:>8.1f}"
              f"{r['sharpe']:>8.2f}{r['maxdd_pct']:>9.1f}{r['beta_btc']:>7.2f}"
              f"{r['corr_btc']:>7.2f}{r['annual_turnover']:>9.1f}")
    print("\nNote (advisor): survivorship biases REVERSAL up / MOMENTUM down")
    print("(dead losers like LUNA absent). If reversal wins, suspect artifact.")
    print("Beta/corr to BTC must be ~0 for this to be the variance-reducer it's for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
