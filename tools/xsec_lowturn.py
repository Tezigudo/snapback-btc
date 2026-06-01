"""Cross-sectional momentum MN — LOW-TURNOVER refinement (Track B, step 3).

The base book failed because ~58x/yr turnover ate the edge at 20bps. This tests
whether reducing turnover rescues it, with HONEST discipline:
  - turnover levers: longer rebalance + a HYSTERESIS no-trade band (a held
    position stays until it falls out of the top/bottom-(K+BAND), instead of
    churning every rebalance).
  - pre-specified small family; select best on IS by NET Sharpe @ 20bps; LOCK;
    test ONCE on OOS at 0/20/30 bps. No re-picking after seeing OOS.
  - PASS = OOS net cagr>0 @ 20bps AND |beta-to-BTC|<0.15.
Funding still omitted -> optimistic.

Run: uv run python tools/xsec_lowturn.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

UNIVERSE = ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOGE", "DOT",
            "AVAX", "NEAR", "LINK", "LTC", "ATOM", "BCH"]
DATA = "data/historical"
IS_END = pd.Timestamp("2024-06-30", tz="UTC")
SKIP = 2


def daily_panel(coins):
    cols = {}
    for c in coins:
        f = os.path.join(DATA, f"{c}_USDT_USDT_1h.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df.columns = [x.lower() for x in df.columns]
        cols[c] = df["close"].resample("1D").last()
    return pd.DataFrame(cols).dropna()


def _max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min() * 100.0)


def select_with_hysteresis(score, held_long, held_short, K, band):
    """Return (new_long_set, new_short_set) keeping exactly K each, with a
    hysteresis band: a held long survives if still ranked in the top-(K+band)."""
    order = np.argsort(score)           # ascending
    n = len(score)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)          # 0 = lowest score (worst), n-1 = best
    top = set(int(i) for i in order[-K:])
    bot = set(int(i) for i in order[:K])
    top_band = set(int(i) for i in order[-(K + band):])
    bot_band = set(int(i) for i in order[:K + band])

    # longs: keep held that survive the band, then fill from fresh top-K
    longs = [i for i in held_long if i in top_band]
    for i in sorted(top, key=lambda x: -rank[x]):
        if len(longs) >= K:
            break
        if i not in longs:
            longs.append(i)
    longs = longs[:K]
    # shorts: keep held that survive, fill from fresh bottom-K
    shorts = [i for i in held_short if i in bot_band]
    for i in sorted(bot, key=lambda x: rank[x]):
        if len(shorts) >= K:
            break
        if i not in shorts:
            shorts.append(i)
    shorts = shorts[:K]
    # avoid overlap (a coin can't be both); shorts lose ties
    shorts = [i for i in shorts if i not in longs][:K]
    return set(longs), set(shorts)


def run(panel, lookback, rebal, K, band, bps):
    prices = panel.values
    n_days, n_coins = prices.shape
    rets = np.zeros_like(prices)
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    btc_i = list(panel.columns).index("BTC")

    out_r, out_btc, out_dates, turns = [], [], [], []
    prev_w = np.zeros(n_coins)
    held_long, held_short = set(), set()
    t = lookback + SKIP + 1
    while t < n_days:
        score = prices[t - SKIP] / prices[t - lookback - SKIP] - 1.0
        held_long, held_short = select_with_hysteresis(
            score, held_long, held_short, K, band)
        w = np.zeros(n_coins)
        for i in held_long:
            w[i] = 1.0 / K
        for i in held_short:
            w[i] = -1.0 / K
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


def stats(book, btc, turns, rpy):
    equity = np.cumprod(1.0 + book)
    years = len(book) / 365.25
    cagr = float(equity[-1] ** (1 / years) - 1) * 100 if years > 0 and equity[-1] > 0 else -100.0
    sharpe = float(book.mean() / book.std() * np.sqrt(365)) if book.std() > 0 else 0.0
    beta = float(np.polyfit(btc, book, 1)[0]) if btc.std() > 0 else float("nan")
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": _max_dd(equity),
            "beta": beta, "turn_yr": float(turns.mean() * rpy)}


def main() -> int:
    panel = daily_panel(UNIVERSE)
    print(f"Panel {panel.shape[1]}x{panel.shape[0]} "
          f"({str(panel.index[0])[:10]}->{str(panel.index[-1])[:10]})\n")

    family = [(lb, rb, K, band)
              for lb in (60, 90, 120)
              for rb in (14, 21, 30)
              for K in (3, 4)
              for band in (2, 3)]
    SEL = 20.0
    print(f"IS selection @ {SEL:.0f}bps (lookback,rebal,K,band):")
    print(f"  {'cfg':<22}{'IS cagr':>9}{'IS Sh':>7}{'IS DD':>8}{'beta':>7}{'turn/yr':>9}")
    scored = []
    for lb, rb, K, band in family:
        d, book, btc, turns, rpy = run(panel, lb, rb, K, band, SEL)
        m = d <= IS_END
        s = stats(book[m], btc[m], turns, rpy)
        scored.append(((lb, rb, K, band), s))
    # show top 8 by IS Sharpe
    for cfg, s in sorted(scored, key=lambda x: -x[1]["sharpe"])[:8]:
        lb, rb, K, band = cfg
        print(f"  lb{lb} rb{rb} K{K} b{band:<8}{s['cagr']:>9.1f}{s['sharpe']:>7.2f}"
              f"{s['maxdd']:>8.1f}{s['beta']:>7.2f}{s['turn_yr']:>9.1f}")
    eligible = [(c, s) for c, s in scored if abs(s["beta"]) < 0.15]
    best_cfg, _ = max(eligible or scored, key=lambda x: x[1]["sharpe"])
    print(f"\nLOCKED IS winner (max IS Sharpe @20bps, |beta|<0.15): "
          f"lookback={best_cfg[0]} rebal={best_cfg[1]} K={best_cfg[2]} band={best_cfg[3]}")

    lb, rb, K, band = best_cfg
    print(f"\nOOS test (locked cfg), friction stress:")
    print(f"  {'bps':>4}{'OOS cagr':>10}{'OOS Sh':>8}{'OOS DD':>9}{'beta':>7}{'turn/yr':>9}")
    for bps in (0, 10, 20, 30):
        d, book, btc, turns, rpy = run(panel, lb, rb, K, band, bps)
        m = d > IS_END
        s = stats(book[m], btc[m], turns, rpy)
        v = "PASS" if (s["cagr"] > 0 and abs(s["beta"]) < 0.15) else "FAIL"
        print(f"  {bps:>4}{s['cagr']:>10.1f}{s['sharpe']:>8.2f}{s['maxdd']:>9.1f}"
              f"{s['beta']:>7.2f}{s['turn_yr']:>9.1f}   {v}")
    print("\nPASS = OOS net cagr>0 @20bps AND |beta|<0.15. Funding omitted (optimistic).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
