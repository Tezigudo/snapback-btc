"""multifactor-v1 deepening: per-factor ablation + MTF + multi-coin sweep.

Runs the locked-config baseline plus 7 variants across 5 OOS windows on BTC,
then runs baseline on ETH + SOL.  Aggregates trades, computes PSR, and writes
a single JSON to reports/multifactor_v1_deepening.json.

NOT a refactor: signals_multifactor.py is untouched. Variants use the existing
config-json mechanism plus a small in-process subclass for the MTF 4H gate.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import ema  # noqa: E402
from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
    equity_impact_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20

# OOS windows from PATH2_RESULTS.html (5 windows, 2.5 years)
WINDOWS = [
    ("2022H1", "2022-01-01", "2022-06-30"),
    ("2023H1", "2023-01-01", "2023-06-30"),
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
]

# Locked params.yaml values
LOCKED = {
    "rsi_period":                  14,
    "rsi_long_threshold":          35.0,
    "rsi_short_threshold":         70.0,
    "volume_ma_period":            20,
    "volume_multiple":             2.0,
    "mf_trend_ema_period":         200,
    "require_trend":               True,
    "require_candlestick":         False,
    "require_macd":                False,
    "require_funding_not_extreme": True,
    "funding_extreme_threshold":   0.0005,
    "sl_pct":                      0.015,
    "tp_pct":                      0.030,
    "max_hold_bars":               1344,
    "risk_per_trade_pct":          2.75,
    "leverage":                    20,
    "allow_shorts":                True,
}

PARQ = {
    "BTC": ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet",
    "ETH": ROOT / "data" / "historical" / "ETH_USDT_USDT_15m.parquet",
    "SOL": ROOT / "data" / "historical" / "SOL_USDT_USDT_15m.parquet",
}
FUND_PARQ = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_slice(parquet: Path, start: str, end: str,
                attach_funding: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    if attach_funding and FUND_PARQ.exists():
        fund = pd.read_parquet(FUND_PARQ)
        if fund.index.tz is not None:
            fund.index = fund.index.tz_localize(None)
        # 15m bars (left) get the most recent funding rate (right) via backward merge.
        # Lookahead-safe: funding rates are observed at their timestamp.
        left = pd.DataFrame(index=df.index)
        right = pd.DataFrame({"Funding": fund["funding_rate"].values}, index=fund.index)
        merged = pd.merge_asof(left, right,
                               left_index=True, right_index=True,
                               direction="backward")
        df["Funding"] = merged["Funding"].values
        # Fill warm-up NaNs with 0.0 (no funding info → don't block).
        df["Funding"] = df["Funding"].fillna(0.0)

    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sliced) == 0:
        raise ValueError(f"Empty slice {start}..{end} from {parquet.name}")
    return sliced


# -----------------------------------------------------------------------------
# MTF 4H gate subclass
# -----------------------------------------------------------------------------

def _load_4h_ema200_aligned(dates_15m: pd.DatetimeIndex, parquet_4h: Path,
                             period: int = 200) -> np.ndarray:
    """Lookahead-safe 4H EMA200 aligned to 15m bars (copy of divergence_v2 pattern).

    Bar T 4H open closes at T+4h. We only know the 4H EMA at T at 15m timestamps
    >= T+4h. Backward merge_asof on close timestamps achieves this.
    """
    df4 = pd.read_parquet(parquet_4h)
    if df4.index.tz is not None:
        df4.index = df4.index.tz_localize(None)
    e = ema(df4["close"], period).values
    close_times = df4.index + pd.Timedelta(hours=4)
    # Force matching datetime64 precision (pandas refuses cross-precision merges).
    close_times_us = pd.DatetimeIndex(close_times.astype("datetime64[us]"))
    left_idx = pd.DatetimeIndex(dates_15m.astype("datetime64[us]"))
    right = pd.DataFrame({"ema4h": e}, index=close_times_us).sort_index()
    left = pd.DataFrame(index=left_idx)
    merged = pd.merge_asof(left, right, left_index=True, right_index=True,
                            direction="backward")
    return merged["ema4h"].values


class MultiFactorMTF4H(DayTradeMultiFactorBTC):
    """Subclass adding a 4H EMA200 regime gate.

    Longs only when 15m close > 4H_EMA200, shorts only when 15m close < 4H_EMA200.
    The 4H EMA series is pre-attached as `_ema4h` on `data.df` by the runner.
    """

    def init(self) -> None:
        super().init()
        # Pulled from a column the runner attaches before Backtest construction.
        # backtesting.py exposes extra columns via self.data._df or self.data.<Col>.
        # We added it as a capitalised column 'Ema4h' so backtesting.py exposes
        # self.data.Ema4h (camelcase indexing).
        self._ema4h = np.asarray(self.data.Ema4h)

    def _long_signal(self, i: int) -> bool:
        v = self._ema4h[i]
        if not (np.isfinite(v) and self.data.Close[-1] > v):
            return False
        return super()._long_signal(i)

    def _short_signal(self, i: int) -> bool:
        v = self._ema4h[i]
        if not (np.isfinite(v) and self.data.Close[-1] < v):
            return False
        return super()._short_signal(i)


# -----------------------------------------------------------------------------
# Variant definitions (overrides applied on top of LOCKED)
# -----------------------------------------------------------------------------

VARIANTS = {
    "baseline":         {},
    "no_volume":        {"volume_multiple": 0.0},     # any vol > 0 passes
    "no_trend":         {"require_trend": False},
    "no_funding":       {"require_funding_not_extreme": False},
    "rsi_30_70":        {"rsi_long_threshold": 30.0, "rsi_short_threshold": 70.0},
    "add_candlestick":  {"require_candlestick": True},  # inverse: what if we re-enable?
    "add_macd":         {"require_macd": True},         # inverse
    "mtf_4h_gate":      {},  # uses MultiFactorMTF4H subclass
}


# -----------------------------------------------------------------------------
# Backtest driver
# -----------------------------------------------------------------------------

def run_one(df: pd.DataFrame, overrides: dict, strategy_class=DayTradeMultiFactorBTC):
    config = {**LOCKED, **overrides}
    bt = Backtest(df, strategy_class, cash=CASH, commission=COMMISSION,
                   margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                   finalize_trades=True)
    stats = bt.run(**config)
    trades_df = getattr(stats, "_trades", None)
    pnl_pct = []
    eq_impact_pnl_pct = []
    if trades_df is not None and len(trades_df):
        if "ReturnPct" in trades_df.columns:
            pnl_pct = (trades_df["ReturnPct"].values * 100.0).tolist()
        # CANONICAL (v2): equity-impact returns (PnL / equity-at-entry) for this
        # single contiguous window — sizing-aware PSR input.
        eq_impact_pnl_pct = equity_impact_returns(stats, cash=CASH).tolist()
    return {
        "trades": int(stats.get("# Trades", 0)),
        "return_pct": float(stats.get("Return [%]", 0.0) or 0.0),
        "max_dd_pct": float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
        "win_rate_pct": float(stats.get("Win Rate [%]") or 0.0),
        "equity_final": float(stats.get("Equity Final [$]", CASH) or CASH),
        "pnl_pct": pnl_pct,
        "eq_impact_pnl_pct": eq_impact_pnl_pct,
    }


def aggregate(per_window: dict) -> dict:
    """Compute compounded return + flatten trades."""
    all_pnl = []
    n_trades = 0
    n_pos = 0
    n_total_windows = 0
    compounded = 1.0
    rets_per_window = []
    for w, r in per_window.items():
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        rets_per_window.append(r["return_pct"])
        n_total_windows += 1
        if r["return_pct"] > 0:
            n_pos += 1
    return {
        "n_trades":         n_trades,
        "compounded_pct":   round((compounded - 1.0) * 100.0, 4),
        "windows_positive": f"{n_pos}/{n_total_windows}",
        "per_window_return_pct": [round(x, 4) for x in rets_per_window],
        "_all_pnl_pct": all_pnl,
    }


def summarize(label: str, per_window: dict) -> dict:
    agg = aggregate(per_window)
    pnl = np.asarray(agg.pop("_all_pnl_pct"))
    # LEGACY (v1) stitched-per-trade PSR — N-inflated, sizing-blind. Kept for
    # observability/diff only; NEVER the verdict input.
    legacy_psr_stitched = (
        compute_psr(pnl, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(pnl) >= 2
        else {"n_trades": int(len(pnl)), "psr_vs_hurdle": 0.0,
              "interpretation": "insufficient_evidence"}
    )

    # CANONICAL (v2) dual-emit block — single source of truth (methodology #1).
    # per_window is a dict keyed by window label; build the list the canonical
    # core expects, carrying return_pct / pnl_pct / eq_impact_pnl_pct / trades.
    pw_list = [
        {
            "label":             w,
            "return_pct":        r["return_pct"],
            "trades":            r["trades"],
            "pnl_pct":           r.get("pnl_pct", []),
            "eq_impact_pnl_pct": r.get("eq_impact_pnl_pct", []),
        }
        for w, r in per_window.items()
    ]
    canon = build_canonical_block(pw_list, aggregation_method=AGGREGATION_VERSION)
    return {
        "label": label,
        "summary": agg,
        "per_window": {w: {k: v for k, v in r.items()
                           if k not in ("pnl_pct", "eq_impact_pnl_pct")}
                        for w, r in per_window.items()},
        # Canonical headline PSR = canonical["psr_walkforward"]. The legacy
        # stitched PSR stays under `psr` for backcompat + as `legacy_psr_stitched`.
        "psr": legacy_psr_stitched,
        "legacy_psr_stitched": legacy_psr_stitched,
        "canonical": canon,
        "aggregation_method": canon["aggregation_method"],
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    out: dict = {
        "strategy": "multifactor-v1",
        "cash": CASH,
        "commission": COMMISSION,
        "margin": MARGIN,
        "windows": [w[0] for w in WINDOWS],
        "locked_config": LOCKED,
    }

    # Preload BTC sliced frames (with funding) per window once.
    btc_slices: dict[str, pd.DataFrame] = {}
    print("[mf_deep] loading BTC slices ...", file=sys.stderr)
    for label, start, end in WINDOWS:
        btc_slices[label] = _load_slice(PARQ["BTC"], start, end, attach_funding=True)
        print(f"  {label} bars={len(btc_slices[label])}", file=sys.stderr)

    # ---- A. Per-variant on BTC ----
    out["variants"] = {}
    for vname, ov in VARIANTS.items():
        print(f"[mf_deep] variant={vname} ov={ov}", file=sys.stderr)
        per_window = {}
        if vname == "mtf_4h_gate":
            # Attach 4H EMA200 column to each slice on the fly.
            cls = MultiFactorMTF4H
            for label, _s, _e in WINDOWS:
                df = btc_slices[label].copy()
                df["Ema4h"] = _load_4h_ema200_aligned(
                    df.index, ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
                )
                per_window[label] = run_one(df, {}, strategy_class=cls)
        else:
            cls = DayTradeMultiFactorBTC
            for label, _s, _e in WINDOWS:
                per_window[label] = run_one(btc_slices[label], ov, strategy_class=cls)
        out["variants"][vname] = summarize(vname, per_window)
        s = out["variants"][vname]["summary"]
        print(f"  -> trades={s['n_trades']} compounded={s['compounded_pct']}% "
              f"wins={s['windows_positive']}", file=sys.stderr)

    # Deltas relative to baseline.
    base_comp = out["variants"]["baseline"]["summary"]["compounded_pct"]
    for vname, v in out["variants"].items():
        v["delta_compounded_pp"] = round(
            v["summary"]["compounded_pct"] - base_comp, 4
        )

    # ---- C. Multi-coin (baseline only) ----
    out["multi_coin"] = {}
    for coin in ("ETH", "SOL"):
        print(f"[mf_deep] multi-coin={coin}", file=sys.stderr)
        per_window = {}
        for label, start, end in WINDOWS:
            try:
                df = _load_slice(PARQ[coin], start, end, attach_funding=False)
            except ValueError as exc:
                print(f"  {label}: SKIP ({exc})", file=sys.stderr)
                # Empty stub so downstream aggregator can cope.
                per_window[label] = {"trades": 0, "return_pct": 0.0,
                                       "max_dd_pct": 0.0, "win_rate_pct": 0.0,
                                       "equity_final": CASH, "pnl_pct": []}
                continue
            per_window[label] = run_one(df, {})
        out["multi_coin"][coin] = summarize(coin, per_window)
        s = out["multi_coin"][coin]["summary"]
        print(f"  -> trades={s['n_trades']} compounded={s['compounded_pct']}% "
              f"wins={s['windows_positive']}", file=sys.stderr)

    # ---- Write output ----
    out_path = ROOT / "reports" / "multifactor_v1_deepening.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[mf_deep] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
