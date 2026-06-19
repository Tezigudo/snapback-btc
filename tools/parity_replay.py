"""parity_replay.py — historical PARITY-REPLAY validation for the forming-bar fix.

Goal
----
Prove on REAL historical Binance data that:
  (a) the FIXED live path (bot._maybe_enter after `df = df.iloc[:-1]`) produces
      the SAME entry signals as the backtest, and
  (b) the OLD forming-bar behavior (no slice — evaluator sees the partial
      forming bar as the last row) produces ~zero entry signals.

This harness ADDS a validation layer only. It does NOT touch the fix commits,
strategy code, or risk code. It places NO orders, does NOT deploy, and reads
only PUBLIC Binance klines/funding via exchange.data (the SAME loader the
backtest uses).

Three arms, all driven over the same window of CLOSED 15m bars:

  FIXED-LIVE  — for each closed bar t, build the frame ENDING at t (what the
                fixed bot sees after iloc[:-1]) and call the REAL live evaluator
                bot_internals.evaluate_for_strategy -> strategy.live_multifactor_v1
                .evaluate_signal. SL/TP from SignalDecision (sl_pct/tp_pct × close).

  OLD-BUG     — for each bar t, build the frame whose LAST row is the FORMING
                bar of t+1 (a partial next bar carrying a fraction of its
                eventual volume, as ccxt returns mid-candle). Call the SAME
                evaluator WITHOUT the slice. The volume gate (cur_vol >
                2×SMA20) can essentially never pass on a fractional-volume
                forming bar -> expected ~zero signals.

  BACKTEST    — the REAL backtest predicate. We build the indicator arrays
                exactly as strategy.signals_multifactor.DayTradeMultiFactorBTC
                .init() does (same rsi/sma/ema/macd/_build_4h_ema_aligned calls)
                and call the class's OWN _long_signal(i)/_short_signal(i)
                methods via a thin shim. No gate logic is reimplemented here —
                the real predicate methods are bound to the shim and invoked.

Then we compare FIXED-LIVE vs BACKTEST per (bar timestamp, side) and report a
parity table + the OLD-BUG signal count.

Run:
    .venv/bin/python -m tools.parity_replay --months 4
    .venv/bin/python -m tools.parity_replay --start 2026-02-01 --end 2026-06-01

Data is cached under data/historical/ by exchange.data (two-sided cache), so
re-runs are offline/reproducible after the first fetch.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot_internals import evaluate_for_strategy  # noqa: E402  (prod wrapper; see note)
from exchange.data import load_funding, load_klines  # noqa: E402
from strategy.live_multifactor_v1 import evaluate_signal as live_evaluate_signal  # noqa: E402
from strategy.indicators import (  # noqa: E402
    bearish_engulfing,
    bullish_engulfing,
    ema,
    hammer,
    macd,
    rsi,
    sma,
)
from strategy.signals import StrategyParams  # noqa: E402
from strategy.signals_multifactor import (  # noqa: E402
    DayTradeMultiFactorBTC,
    _build_4h_ema_aligned,
)

SYMBOL = "BTC/USDT:USDT"
STRATEGY_NAME = "multifactor-v1"
_4H_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"

# Live evaluator self-fetches 4H if not injected. We inject a cached slice so
# the run is deterministic and offline-reproducible. The live module loads
# _LIVE_4H_DAYS_BACK=180d of 4H; we mirror that.
_LIVE_4H_DAYS_BACK = 180


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _cache_path(label: str) -> Path:
    safe = SYMBOL.replace("/", "_").replace(":", "_")
    return ROOT / "data" / "historical" / f"{safe}_{label}.parquet"


def fetch_window(
    start: datetime, end: datetime, offline: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (k15_capitalised, k4h_lowercase, k4h_full_180d, funding_per_15m_bar).

    offline=True (default): read the parquet cache DIRECTLY (no network). This
    is the reproducible path once the cache is populated, and avoids the
    forward-top-up call load_klines makes when end=now (which triggers a
    Binance request -> 418 ban risk under per-bar load).

    offline=False: use exchange.data.load_klines/load_funding (the SAME loader
    the backtest uses) to fetch+cache the window first.

    k15 columns are Capitalised to match what evaluate_signal expects
    (Close/Volume). The per-15m-bar funding series mirrors prepare_strategy_data:
    most-recent known funding event, forward-filled onto the 15m index (the SAME
    value the backtest's self.data.Funding[-1] sees, and what the live bot's
    scalar fetch_funding_rate approximates).
    """
    warm_days = (end - start).days + 60  # extra history for EMA200 warmup
    if offline:
        k15 = pd.read_parquet(_cache_path("15m"))
        k4h_full = pd.read_parquet(_cache_path("4h"))
        fund = pd.read_parquet(_cache_path("funding"))
    else:
        k15 = load_klines(SYMBOL, "15m", days_back=warm_days, end=end)
        # 4H for the live-arm injection window (180d back, like the bot).
        k4h_full = load_klines(SYMBOL, "4h", days_back=max(warm_days, _LIVE_4H_DAYS_BACK), end=end)
        fund = load_funding(SYMBOL, days_back=warm_days, end=end)

    def _naive(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    k15 = _naive(k15)
    k4h_full = _naive(k4h_full)
    fund = _naive(fund)

    # Capitalised copy for the live evaluator + backtest predicate.
    k15_cap = k15.copy()
    k15_cap.columns = [c.capitalize() for c in k15_cap.columns]

    # Per-15m-bar funding (ffill last known event), matching prepare_strategy_data.
    if fund is None or fund.empty:
        funding_15m = pd.Series(np.nan, index=k15_cap.index)
    else:
        funding_15m = fund["funding_rate"].reindex(k15_cap.index, method="ffill")

    return k15_cap, k4h_full, k4h_full, funding_15m


# --------------------------------------------------------------------------- #
# BACKTEST reference — drive the REAL predicate methods, no gate reimplementation
# --------------------------------------------------------------------------- #
class _BarShim:
    """Minimal stand-in for backtesting.py's `self.data` accessor.

    The real DayTradeMultiFactorBTC._long_signal/_short_signal read only
    self.data.Close[-1], self.data.Volume[-1], self.data.Funding[-1]. We point
    [-1] at bar i. The predicate methods are bound to an object carrying the
    precomputed indicator arrays + class threshold attributes, so the gate logic
    executed is the real backtest code verbatim.
    """

    def __init__(self, close, volume, funding, i):
        self._close = close
        self._volume = volume
        self._funding = funding
        self._i = i

    class _Col:
        def __init__(self, arr, i):
            self._arr = arr
            self._i = i

        def __getitem__(self, k):
            if k == -1:
                return self._arr[self._i]
            raise IndexError("shim only serves [-1]")

    @property
    def Close(self):
        return self._Col(self._close, self._i)

    @property
    def Volume(self):
        return self._Col(self._volume, self._i)

    @property
    def Funding(self):
        return self._Col(self._funding, self._i)


class _BacktestPredicate:
    """Carries indicator arrays + class attrs, exposes the REAL bound predicate
    methods. Mirrors DayTradeMultiFactorBTC.init() array construction exactly."""

    # Bind the unbound predicate methods from the real strategy class.
    _long_signal = DayTradeMultiFactorBTC._long_signal
    _short_signal = DayTradeMultiFactorBTC._short_signal

    def __init__(self, k15: pd.DataFrame, funding_15m: pd.Series, params: StrategyParams):
        close = k15["Close"]
        open_ = k15["Open"]
        high = k15["High"]
        low = k15["Low"]
        volume = k15["Volume"]

        # SAME calls as DayTradeMultiFactorBTC.init():
        self._rsi = rsi(close, params.rsi_period).values
        self._vol_sma = sma(volume, params.volume_ma_period).values
        self._trend_ema = ema(close, params.mf_trend_ema_period).values
        _, _, self._macd_hist = (s.values for s in macd(close, 12, 26, 9))
        self._bull_engulf = bullish_engulfing(open_, high, low, close).values
        self._bear_engulf = bearish_engulfing(open_, high, low, close).values
        self._hammer = hammer(open_, high, low, close).values

        # 4H EMA gate aligned the SAME way the backtest does it.
        self.use_mtf_4h_gate = True
        self.mtf_4h_ema_period = int(getattr(params, "mtf_4h_ema_period", 200))
        self._ema_4h_200 = _build_4h_ema_aligned(
            pd.DatetimeIndex(k15.index), _4H_PARQUET, self.mtf_4h_ema_period
        )

        # Threshold/flag class attrs the predicate reads.
        self.rsi_long_threshold = float(params.rsi_long_threshold)
        self.rsi_short_threshold = float(params.rsi_short_threshold)
        self.volume_multiple = float(params.volume_multiple)
        self.require_trend = bool(params.require_trend)
        self.require_candlestick = bool(params.require_candlestick)
        self.require_macd = bool(params.require_macd)
        self.require_funding_not_extreme = bool(params.require_funding_not_extreme)
        self.funding_extreme_threshold = float(params.funding_extreme_threshold)
        self.allow_shorts = True

        self._close_arr = close.values
        self._vol_arr = volume.values
        self._fund_arr = funding_15m.reindex(k15.index).values

    def signal_at(self, i: int) -> str | None:
        # Point the shim's data accessor at bar i, then call the REAL predicate.
        self.data = _BarShim(self._close_arr, self._vol_arr, self._fund_arr, i)
        if self._long_signal(i):
            return "long"
        if self._short_signal(i):
            return "short"
        return None


# --------------------------------------------------------------------------- #
# Forming-bar synthesis (OLD-BUG arm)
# --------------------------------------------------------------------------- #
def _forming_bar_row(next_full: pd.Series, prev_close: float, frac: float) -> pd.Series:
    """Synthesize a partial FORMING bar as ccxt would return it mid-candle.

    A bar ~early in its life has: open = its real open, high/low/close drifting
    near open, and volume = a small fraction of the eventual bar volume. We use
    `frac` of the eventual volume (default tiny) and set close=open (price hasn't
    moved much yet). This is the worst case the live volume gate faced before the
    fix: cur_vol is a fraction of a normal bar, so cur_vol > 2×SMA20 is unreachable.
    """
    o = float(next_full["Open"])
    return pd.Series(
        {
            "Open": o,
            "High": max(o, float(next_full["High"])) if frac >= 1.0 else o,
            "Low": min(o, float(next_full["Low"])) if frac >= 1.0 else o,
            "Close": o,
            "Volume": float(next_full["Volume"]) * frac,
        }
    )


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def run_replay(
    start: datetime, end: datetime, forming_frac: float, warmup_bars: int, offline: bool
) -> dict:
    params = StrategyParams.from_yaml()
    p_dict = _params_dict_from_yaml()

    # Guard: assert the function we call directly (live_evaluate_signal) is the
    # exact one bot_internals.evaluate_for_strategy dispatches to for
    # multifactor-v1. If a refactor ever re-routes the bot's entry point, this
    # harness must follow it — fail loud rather than validate a stale path.
    import bot_internals as _bi
    assert _bi.evaluate_signal is live_evaluate_signal, (
        "live evaluator drift: bot_internals.evaluate_signal != "
        "strategy.live_multifactor_v1.evaluate_signal"
    )
    _ = evaluate_for_strategy  # documents the production wrapper (see _live_decision)

    k15, k4h_full, _, funding_15m = fetch_window(start, end, offline)

    naive_start = start.replace(tzinfo=None)
    naive_end = end.replace(tzinfo=None)

    # Bars we will EVALUATE (inside the visible window, after global warmup).
    full_index = k15.index
    eval_mask = (full_index >= naive_start) & (full_index <= naive_end)
    eval_positions = np.flatnonzero(np.asarray(eval_mask))
    # Drop the very last bar — OLD-BUG needs a t+1 forming bar to exist.
    eval_positions = eval_positions[eval_positions < len(full_index) - 1]
    # Respect warmup: need enough preceding bars for EMA200 etc.
    eval_positions = eval_positions[eval_positions >= warmup_bars]

    backtest_pred = _BacktestPredicate(k15, funding_15m, params)

    fixed_live: dict[pd.Timestamp, dict] = {}
    old_bug: dict[pd.Timestamp, dict] = {}
    backtest: dict[pd.Timestamp, dict] = {}

    # 4H slice injected into the live evaluator (cached, deterministic).
    k4h_for_live = k4h_full

    for pos in eval_positions:
        ts = full_index[pos]

        # ---- BACKTEST reference (real predicate) ----
        bt_side = backtest_pred.signal_at(pos)
        if bt_side is not None:
            backtest[ts] = {"side": bt_side, "close": float(k15["Close"].iloc[pos])}

        # Funding scalar the live evaluator receives for this bar = the ffilled
        # last-known event at ts (same value the backtest's Funding[-1] sees).
        fr = funding_15m.iloc[pos]
        fr = float(fr) if np.isfinite(fr) else 0.0

        # ---- FIXED-LIVE: frame ends at CLOSED bar t (== iloc[:-1] result) ----
        frame_fixed = k15.iloc[: pos + 1]
        side, price, sl, tp = _live_decision(frame_fixed, fr, p_dict, k4h_for_live)
        if side is not None:
            fixed_live[ts] = {"side": side, "price": price, "sl": sl, "tp": tp}

        # ---- OLD-BUG: append a FORMING t+1 bar, no slice ----
        next_full = k15.iloc[pos + 1]
        forming = _forming_bar_row(next_full, float(k15["Close"].iloc[pos]), forming_frac)
        frame_old = pd.concat(
            [k15.iloc[: pos + 1], pd.DataFrame([forming], index=[full_index[pos + 1]])]
        )
        side_old, _, _, _ = _live_decision(frame_old, fr, p_dict, k4h_for_live)
        if side_old is not None:
            old_bug[full_index[pos + 1]] = {"side": side_old}

    return {
        "params": params,
        "window": (full_index[eval_positions[0]], full_index[eval_positions[-1]]),
        "n_eval": len(eval_positions),
        "fixed_live": fixed_live,
        "old_bug": old_bug,
        "backtest": backtest,
        "k15": k15,
    }


def _params_dict_from_yaml() -> dict:
    import yaml

    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


def _live_decision(
    frame: pd.DataFrame, funding_rate: float, p_dict: dict, k4h: pd.DataFrame
) -> tuple[str | None, float, float, float]:
    """Call the REAL live evaluator and reproduce bot_internals.evaluate_for_strategy's
    multifactor-v1 SL/TP math (read verbatim from bot_internals.py:130-140).

    We call strategy.live_multifactor_v1.evaluate_signal DIRECTLY with an
    injected bars_4h slice. This is the exact function bot_internals dispatches
    to for multifactor-v1; the only reason we bypass evaluate_for_strategy is
    that the wrapper does NOT forward a bars_4h kwarg, so it would force the
    evaluator to self-fetch 180d of 4H from Binance on EVERY bar (2879 network
    calls -> HTTP 418 IP ban). Injecting the cached 4H frame is causally
    identical to the production self-fetch (same load_klines source, same EMA
    function, same most-recent-closed-4h-bar rule) but offline + deterministic.

    Returns (side, price, sl_price, tp_price). price == cur_close == frame
    Close[-1] (the closed bar). SL/TP from sl_pct/tp_pct as in bot_internals.
    """
    side, dbg = live_evaluate_signal(frame, funding_rate, p_dict, bars_4h=k4h)
    fallback_price = float(frame["Close"].iloc[-1])
    price = dbg.get("cur_close", fallback_price) if isinstance(dbg, dict) else fallback_price
    s = p_dict["strategy"]
    sl_pct = float(s["sl_pct"])
    tp_pct = float(s["tp_pct"])
    if side == "long":
        sl_price = price - sl_pct * price
        tp_price = price + tp_pct * price
    elif side == "short":
        sl_price = price + sl_pct * price
        tp_price = price - tp_pct * price
    else:
        sl_price = tp_price = float("nan")
    return side, float(price), float(sl_price), float(tp_price)


# --------------------------------------------------------------------------- #
# Compare + report
# --------------------------------------------------------------------------- #
def compare(res: dict) -> None:
    fixed = res["fixed_live"]
    bt = res["backtest"]

    fixed_keys = {(ts, v["side"]) for ts, v in fixed.items()}
    bt_keys = {(ts, v["side"]) for ts, v in bt.items()}

    matched = fixed_keys & bt_keys
    only_fixed = fixed_keys - bt_keys     # extra (live fired, backtest didn't)
    only_bt = bt_keys - fixed_keys        # missing (backtest fired, live didn't)

    # Same-bar opposite-side mismatches.
    fixed_ts = {ts: v["side"] for ts, v in fixed.items()}
    bt_ts = {ts: v["side"] for ts, v in bt.items()}
    mismatched_side = [
        ts for ts in (set(fixed_ts) & set(bt_ts)) if fixed_ts[ts] != bt_ts[ts]
    ]

    union = len(fixed_keys | bt_keys)
    parity_pct = (len(matched) / union * 100.0) if union else 100.0

    # Entry-price deltas on matched signals (live price vs backtest close).
    price_deltas = []
    for ts, side in matched:
        live_p = fixed[ts]["price"]
        bt_p = bt[ts]["close"]
        price_deltas.append(abs(live_p - bt_p))
    max_dp = max(price_deltas) if price_deltas else 0.0
    mean_dp = (sum(price_deltas) / len(price_deltas)) if price_deltas else 0.0

    w0, w1 = res["window"]
    print("=" * 72)
    print("PARITY-REPLAY  —  forming-bar fix vs backtest on REAL Binance data")
    print("=" * 72)
    print(f"symbol         : {SYMBOL}   strategy: {STRATEGY_NAME}")
    print(f"eval window    : {w0}  ->  {w1}")
    print(f"closed bars ev : {res['n_eval']}")
    print()
    print("--- FIXED-LIVE  vs  BACKTEST (signal set, per bar+side) ---")
    print(f"  backtest signals : {len(bt_keys)}")
    print(f"  fixed-live sigs  : {len(fixed_keys)}")
    print(f"  matched          : {len(matched)}")
    print(f"  missing (bt only): {len(only_bt)}")
    print(f"  extra   (lv only): {len(only_fixed)}")
    print(f"  side-mismatch    : {len(mismatched_side)}")
    print(f"  PARITY           : {parity_pct:.2f}%   (matched / union)")
    print(f"  entry-price dlt  : max ${max_dp:.4f}  mean ${mean_dp:.4f}  "
          f"(expect ~0: both anchor to closed-bar close)")
    print()
    print("--- OLD-BUG arm (forming-bar evaluated, no slice) ---")
    print(f"  old-bug signals  : {len(res['old_bug'])}   (expect ~0)")
    print()

    if only_bt:
        print("  MISSING detail (backtest fired, fixed-live did not):")
        for ts, side in sorted(only_bt)[:20]:
            print(f"    {ts}  {side}")
    if only_fixed:
        print("  EXTRA detail (fixed-live fired, backtest did not):")
        for ts, side in sorted(only_fixed)[:20]:
            print(f"    {ts}  {side}")

    # Verdict
    print()
    if union == 0:
        verdict = "INCONCLUSIVE (zero signals in window — widen window)"
    elif parity_pct >= 99.0 and len(res["old_bug"]) == 0:
        verdict = "YES — fix makes live match backtest; old bug produced 0 signals"
    elif parity_pct >= 90.0:
        verdict = "PARTIAL — high parity but residual mismatches (see detail)"
    else:
        verdict = "NO — material divergence; investigate"
    print(f"VERDICT: {verdict}")
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--months", type=int, default=4, help="lookback window in months")
    p.add_argument("--start", help="YYYY-MM-DD (UTC), overrides --months")
    p.add_argument("--end", help="YYYY-MM-DD (UTC), default now")
    p.add_argument("--forming-frac", type=float, default=0.02,
                   help="fraction of eventual volume the synthetic forming bar carries")
    p.add_argument("--warmup-bars", type=int, default=300,
                   help="min preceding 15m bars before a bar is evaluated")
    p.add_argument("--fetch", action="store_true",
                   help="fetch+cache from Binance first (default: offline, read parquet cache)")
    args = p.parse_args()

    end = (datetime.fromisoformat(args.end).replace(tzinfo=UTC)
           if args.end else datetime.now(UTC))
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    else:
        start = end - timedelta(days=args.months * 30)

    t0 = datetime.now(UTC)
    res = run_replay(start, end, args.forming_frac, args.warmup_bars, offline=not args.fetch)
    compare(res)
    print(f"runtime: {(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
