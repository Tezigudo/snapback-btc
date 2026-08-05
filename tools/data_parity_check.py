"""Is the chart God reads the same series the bot trades?

Answers one question with evidence: does the data SOURCE change anything? The
bot fires orders on Binance, so Binance is ground truth; this checks whether
the TradingView chart agrees, both bar-by-bar and in backtested P&L.

Verdict as of 2026-08-05 (2.3 years, 5000 4h bars): they are the SAME DATA.
TradingView does not originate crypto prices — `BINANCE:BTCUSDT.P` redistributes
Binance's own feed. BTC 4h Close matched 5000/5000, and because every gate in
all three strategies reads CLOSE, entries came out bit-identical on all three.
The two things that DO bite are not the vendor:

  1. SPOT vs PERP. `BINANCE:BTCUSDT` (spot) and `BINANCE:BTCUSDT.P` (perp) share
     0 of 500 identical closes, mean $30.40 apart. The bot trades the PERP.
  2. TradingView serves bars on a UTC+7 clock here, so the 4h bar labelled 15:00
     on the chart is Binance's 08:00 UTC bar. Shift by 7h before comparing
     anything to bot logs, telemetry, or bars_since_flip.

METHOD — why this is attributable to data and not to strategy plumbing:
signals come from the LIVE evaluators the deployed bot calls (strategy/live_*.py)
with the deployed configs, and ONE simulator runs both sides over an identical
window. Deliberately does NOT route through backtest.py / StrategyParams.from_yaml,
which silently drops donchian keys (entry 80->20, slope gate off) and would
measure a system that isn't deployed.

The simulator is a comparison instrument, NOT a re-validation harness. Absolute
numbers will not match the signed-off backtests (different window, simplified
friction: fees only, no funding). Only the TV-vs-Binance DELTA is meaningful.

USAGE:
  uv run python -m tools.data_parity_check                  # all three
  uv run python -m tools.data_parity_check --strategy donchian-v3
  uv run python -m tools.data_parity_check --bars 2000 --json out.json

Needs the unofficial tvdatafeed, which is deliberately NOT a project dependency:
  pip install --no-deps "git+https://github.com/rongardF/tvdatafeed.git" websocket-client
`--no-deps` matters — a plain install drags its own pandas/numpy that then shadow
the venv's and silently change indicator math.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
import urllib.request

import pandas as pd
import yaml

# Repo root FIRST — the venv's editable-install .pth points at a DIFFERENT
# checkout (snapback-droplet-wt), so without this we would evaluate that
# checkout's strategy code instead of this working tree's. Same guard every
# other tool in tools/ uses.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from strategy.live_donchian_v3 import channel_exit_signal, evaluate_signal_donchian_v3
from strategy.live_multifactor_v1 import evaluate_signal as evaluate_signal_v1
from strategy.live_supertrend import evaluate_signal_supertrend

REPO = pathlib.Path(__file__).resolve().parent.parent
FAPI = "https://fapi.binance.com/fapi/v1/klines"
FEE_BPS = 5.0
TV_UTC_OFFSET_H = 7          # TradingView serves UTC+7 for this account
INTERVAL_MS = {"4h": 4 * 3600_000, "15m": 15 * 60_000}

# strategy -> (label, tv symbol, binance symbol, interval, config, warmup bars)
PLANS = {
    "donchian-v3": ("donchian-v3 / BTC 4h", "BTCUSDT.P", "BTCUSDT", "4h",
                    "params_donchian.yaml", 160),
    "supertrend": ("supertrend / SOL 4h", "SOLUSDT.P", "SOLUSDT", "4h",
                   "params_sol_supertrend.yaml", 160),
    "multifactor-v1": ("multifactor-v1 / BTC 15m", "BTCUSDT.P", "BTCUSDT", "15m",
                       "params.yaml", 300),
}

log = logging.getLogger("snapback.data_parity")


# ─────────────────────────────── data ───────────────────────────────
def fetch_binance(symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    """Paginated klines so Binance can cover TradingView's full range."""
    rows: list = []
    cur = start_ms
    while True:
        url = f"{FAPI}?symbol={symbol}&interval={interval}&startTime={cur}&limit=1500"
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.load(r)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1500:
            break
        cur = batch[-1][0] + INTERVAL_MS[interval]
        time.sleep(0.25)
    df = pd.DataFrame(rows, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df.index.name = "ts"
    return df[["Open", "High", "Low", "Close", "Volume"]].drop_duplicates()


def fetch_tradingview(symbol: str, interval: str, n_bars: int) -> pd.DataFrame:
    try:
        from tvDatafeed import Interval, TvDatafeed
    except ImportError:
        raise SystemExit(
            "tvdatafeed is not installed (deliberately not a project dep).\n"
            '  pip install --no-deps "git+https://github.com/rongardF/tvdatafeed.git" '
            "websocket-client"
        ) from None
    iv = {"4h": Interval.in_4_hour, "15m": Interval.in_15_minute}[interval]
    d = TvDatafeed().get_hist(symbol=symbol, exchange="BINANCE",
                              interval=iv, n_bars=n_bars)
    if d is None or d.empty:
        raise SystemExit(f"TradingView returned no data for BINANCE:{symbol} {interval}")
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                          "close": "Close", "volume": "Volume"})
    d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
    d.index = pd.to_datetime(d.index) - pd.Timedelta(hours=TV_UTC_OFFSET_H)
    d.index.name = "ts"
    return d[~d.index.duplicated(keep="last")]


def diff_bars(tv: pd.DataFrame, bn: pd.DataFrame) -> dict:
    """Bar-by-bar OHLCV comparison over the overlapping index."""
    common = tv.index.intersection(bn.index)
    a, b = tv.loc[common], bn.loc[common]
    out: dict = {"overlap_bars": len(common),
                 "from": str(common[0]), "to": str(common[-1]), "fields": {}}
    for col in ("Open", "High", "Low", "Close", "Volume"):
        d = (a[col] - b[col]).abs()
        bad = d[d > 0]
        out["fields"][col] = {
            "mismatched_bars": len(bad),
            "max_abs_diff": round(float(d.max()), 6) if len(d) else 0.0,
            "worst_bars": [
                {"ts": str(ts), "tv": float(a.loc[ts, col]), "bn": float(b.loc[ts, col])}
                for ts in bad.abs().nlargest(3).index
            ],
        }
    return out


# ───────────────────────────── simulator ─────────────────────────────
def simulate(bars: pd.DataFrame, strategy: str, params: dict, warmup: int,
             bars_4h: pd.DataFrame | None = None) -> dict:
    """Evaluate on the last CLOSED bar, fill at the NEXT bar's open.

    SL/TP are checked intrabar against High/Low; a bar spanning both is scored
    as the STOP, which is the live worst case. Exits mirror each live leg:
    donchian = SL + 10-bar channel cross + 48-bar time stop and NO TP leg;
    supertrend = SL/TP brackets + opposite-flip exit; multifactor-v1 = SL/TP
    only, since the live bot never trend-exits.
    """
    s = params["strategy"]
    risk_pct = float(params["sizing"]["risk_per_trade_pct"]) / 100.0
    fee = FEE_BPS / 1e4

    equity = peak = 1.0
    max_dd = 0.0
    trades: list[dict] = []
    pos: dict | None = None

    for i in range(warmup, len(bars)):
        window = bars.iloc[:i]           # last row is the most recent CLOSED bar
        bar = bars.iloc[i]
        ts = bars.index[i]

        if pos is not None:
            exit_px = exit_reason = None
            hit_sl = (bar["Low"] <= pos["sl"] if pos["side"] == "long"
                      else bar["High"] >= pos["sl"])
            hit_tp = pos["tp"] is not None and (
                bar["High"] >= pos["tp"] if pos["side"] == "long"
                else bar["Low"] <= pos["tp"])
            if hit_sl:
                exit_px, exit_reason = pos["sl"], "sl"
            elif hit_tp:
                exit_px, exit_reason = pos["tp"], "tp"
            elif strategy == "donchian-v3":
                # returns (bool, debug) — a bare tuple is always truthy, unpack it
                crossed, _ = channel_exit_signal(window, pos["side"], params)
                if crossed:
                    exit_px, exit_reason = float(bar["Open"]), "channel"
                elif i - pos["i"] >= int(s.get("time_stop_bars", 48)):
                    exit_px, exit_reason = float(bar["Open"]), "time"
            elif strategy == "supertrend":
                flip, _, _, _ = evaluate_signal_supertrend(window, 0.0, params)
                if flip and flip != pos["side"]:
                    exit_px, exit_reason = float(bar["Open"]), "flip"

            if exit_px is not None:
                move = ((exit_px - pos["entry"]) if pos["side"] == "long"
                        else (pos["entry"] - exit_px))
                net = (move / pos["entry"]) * pos["lev"] - 2 * fee * pos["lev"]
                equity *= 1 + net
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1)
                trades.append({"entry_ts": str(pos["ts"]), "exit_ts": str(ts),
                               "side": pos["side"], "reason": exit_reason,
                               "ret": net})
                pos = None

        if pos is None:
            side, sl_d, tp_d = _entry_signal(strategy, window, params, s, bars_4h)
            if side and sl_d and sl_d == sl_d and sl_d > 0:
                entry = float(bar["Open"])
                sl = entry - sl_d if side == "long" else entry + sl_d
                tp = None
                if tp_d and tp_d == tp_d and tp_d > 0:
                    tp = entry + tp_d if side == "long" else entry - tp_d
                pos = {"side": side, "entry": entry, "sl": sl, "tp": tp,
                       "i": i, "ts": ts, "lev": risk_pct / (sl_d / entry)}

    wins = [t for t in trades if t["ret"] > 0]
    gross_win = sum(t["ret"] for t in wins)
    gross_loss = abs(sum(t["ret"] for t in trades if t["ret"] <= 0))
    return {
        "trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2) if trades else 0.0,
        "total_return_pct": round((equity - 1) * 100, 2),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else None,
        "max_dd_pct": round(max_dd * 100, 2),
        "entries": [(t["entry_ts"], t["side"]) for t in trades],
    }


def _entry_signal(strategy, window, params, s, bars_4h):
    if strategy == "donchian-v3":
        side, sl_d, _, _ = evaluate_signal_donchian_v3(window, 0.0, params)
        return side, sl_d, None
    if strategy == "supertrend":
        side, sl_d, tp_d, _ = evaluate_signal_supertrend(window, 0.0, params)
        return side, sl_d, tp_d
    # multifactor-v1. bars_4h MUST be passed: when it is None the evaluator
    # fetches 4h bars from Binance itself, which would contaminate the
    # TradingView side of this very comparison.
    sub = bars_4h[bars_4h.index <= window.index[-1]] if bars_4h is not None else None
    side, _ = evaluate_signal_v1(window, 0.0, params, bars_4h=sub)
    px = float(window["Close"].iloc[-1])
    return side, float(s["sl_pct"]) * px, float(s["tp_pct"]) * px


# ─────────────────────────────── driver ───────────────────────────────
def run_one(strategy: str, n_bars: int) -> dict:
    label, tv_sym, bn_sym, interval, cfg_name, warmup = PLANS[strategy]
    with open(REPO / "config" / cfg_name) as f:
        params = yaml.safe_load(f)

    tv = fetch_tradingview(tv_sym, interval, n_bars)
    bn = fetch_binance(bn_sym, interval, int(tv.index[0].timestamp() * 1000))

    tv_4h = bn_4h = None
    if strategy == "multifactor-v1":            # 4h trend gate, same source each side
        tv_4h = fetch_tradingview(tv_sym, "4h", n_bars)
        bn_4h = fetch_binance(bn_sym, "4h", int(tv_4h.index[0].timestamp() * 1000))

    lo, hi = max(tv.index[0], bn.index[0]), min(tv.index[-1], bn.index[-1])
    tvw, bnw = tv.loc[lo:hi], bn.loc[lo:hi]

    print(f"\n===== {label} =====")
    print(f"  window {lo} .. {hi}   TV {len(tvw)} bars / Binance {len(bnw)} bars")

    bars_diff = diff_bars(tvw, bnw)
    print(f"  bar diff over {bars_diff['overlap_bars']} bars:")
    for col, f in bars_diff["fields"].items():
        flag = "" if f["mismatched_bars"] == 0 else "  <-- differs"
        print(f"    {col:<7} mismatched: {f['mismatched_bars']:>4}"
              f"   max abs diff: {f['max_abs_diff']}{flag}")

    tv_r = simulate(tvw, strategy, params, warmup, tv_4h)
    bn_r = simulate(bnw, strategy, params, warmup, bn_4h)

    print(f"  {'metric':<18}{'TradingView':>14}{'Binance':>14}{'delta':>12}")
    for k in ("trades", "win_rate_pct", "total_return_pct", "profit_factor",
              "max_dd_pct"):
        a, b = tv_r[k], bn_r[k]
        delta = round(a - b, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
        print(f"  {k:<18}{a!s:>14}{b!s:>14}{delta!s:>12}")

    same = tv_r["entries"] == bn_r["entries"]
    only_tv = set(tv_r["entries"]) - set(bn_r["entries"])
    only_bn = set(bn_r["entries"]) - set(tv_r["entries"])
    print(f"  entries identical: {same}"
          f"   only-TV: {len(only_tv)}   only-Binance: {len(only_bn)}")
    for x in sorted(only_tv)[:5]:
        print(f"    only TradingView: {x}")
    for x in sorted(only_bn)[:5]:
        print(f"    only Binance:     {x}")

    return {"label": label, "strategy": strategy,
            "window": [str(lo), str(hi)], "bar_diff": bars_diff,
            "tv": {k: v for k, v in tv_r.items() if k != "entries"},
            "binance": {k: v for k, v in bn_r.items() if k != "entries"},
            "entries_identical": same,
            "only_tv": len(only_tv), "only_binance": len(only_bn)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", choices=[*PLANS, "all"], default="all")
    ap.add_argument("--bars", type=int, default=5000,
                    help="TradingView bars to request (free tier caps near 5000)")
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)   # tvdatafeed is chatty on import + connect

    names = list(PLANS) if args.strategy == "all" else [args.strategy]
    results = [run_one(n, args.bars) for n in names]

    ok = all(r["entries_identical"] for r in results)
    print("\n" + ("PARITY: every strategy produced identical entries on both sources."
                  if ok else
                  "PARITY BROKEN: at least one strategy's entries differ — see above."))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
