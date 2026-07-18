"""multifactor-v1 + 4H EMA200 gate: ETH and SOL transferability test.

Runs the locked config + use_mtf_4h_gate=True on ETH and SOL across 5 OOS
windows.  Each coin uses its own 4H parquet (injected via mtf_4h_parquet_path
config key, which backtesting.py forwards to the class attribute — no code
changes to signals_multifactor.py needed).

Funding is NOT attached for alts (mirrors pre-gate deepening run, which used
attach_funding=False for ETH/SOL).  BTC funding was attached via a separate
funding parquet that only covers BTC.

Outputs: reports/multifactor_v1_4h_multi_coin.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.signals_multifactor import DayTradeMultiFactorBTC  # noqa: E402
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    build_canonical_block,
)
from tools.psr_eval import compute_psr  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (mirrors run_mf_deepening.py exactly)
# ---------------------------------------------------------------------------

CASH = 1_000_000.0
COMMISSION = 0.0005
MARGIN = 1.0 / 20

WINDOWS = [
    ("2022H1", "2022-01-01", "2022-06-30"),
    ("2023H1", "2023-01-01", "2023-06-30"),
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
]

# Locked params (identical to run_mf_deepening.py LOCKED dict)
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

PARQ_15M = {
    "ETH": ROOT / "data" / "historical" / "ETH_USDT_USDT_15m.parquet",
    "SOL": ROOT / "data" / "historical" / "SOL_USDT_USDT_15m.parquet",
}
PARQ_4H = {
    "ETH": ROOT / "data" / "historical" / "ETH_USDT_USDT_4h.parquet",
    "SOL": ROOT / "data" / "historical" / "SOL_USDT_USDT_4h.parquet",
}

# Pre-gate numbers from multifactor_v1_deepening.json (multi_coin section)
PRE_GATE = {
    "ETH": -31.5631,
    "SOL": 13.9102,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_slice(parquet: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(sliced) == 0:
        raise ValueError(f"Empty slice {start}..{end} from {parquet.name}")
    return sliced


# ---------------------------------------------------------------------------
# Backtest driver
# ---------------------------------------------------------------------------

def run_one(df: pd.DataFrame, config: dict) -> dict:
    bt = Backtest(df, DayTradeMultiFactorBTC, cash=CASH, commission=COMMISSION,
                  margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run(**config)
    trades_df = getattr(stats, "_trades", None)
    pnl_pct = []
    if trades_df is not None and len(trades_df):
        if "ReturnPct" in trades_df.columns:
            pnl_pct = (trades_df["ReturnPct"].values * 100.0).tolist()
    return {
        "trades":        int(stats.get("# Trades", 0)),
        "return_pct":    float(stats.get("Return [%]", 0.0) or 0.0),
        "max_dd_pct":    float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
        "win_rate_pct":  float(stats.get("Win Rate [%]") or 0.0),
        "equity_final":  float(stats.get("Equity Final [$]", CASH) or CASH),
        "pnl_pct":       pnl_pct,
    }


def aggregate_and_psr(per_window: dict) -> dict:
    all_pnl = []
    n_trades = 0
    n_pos = 0
    compounded = 1.0
    rets_per_window = []
    for w, r in per_window.items():
        n_trades += r["trades"]
        all_pnl.extend(r["pnl_pct"])
        rp = r["return_pct"] / 100.0
        compounded *= (1.0 + rp)
        rets_per_window.append(r["return_pct"])
        if r["return_pct"] > 0:
            n_pos += 1
    n_total = len(per_window)

    # LEGACY (kept for observability/diff only): stitched per-trade ReturnPct
    # PSR across DISJOINT OOS windows. N-inflated + sizing-blind — never the
    # headline. contiguous=False so Lo serial-corr correction is a no-op on the
    # spurious cross-window autocorrelation.
    pnl_arr = np.asarray(all_pnl)
    legacy_psr_stitched = (
        compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95, contiguous=False)
        if len(pnl_arr) >= 2 else {
            "n_trades": int(len(pnl_arr)), "psr_vs_hurdle": 0.0,
            "interpretation": "insufficient_evidence",
        }
    )
    legacy_psr_stitched["deprecation"] = (
        "stitched_per_trade_pl_pct_psr_is_N_inflated"
    )

    # CANONICAL (methodology debt #1): equity-curve aggregation. Per-window
    # return_pct is stats["Return [%]"] (sizing-aware engine headline), so this
    # is the 5-OOS equity-curve family. Headline PSR = psr_walkforward, computed
    # on the n-window return series (n == n_windows) — defeats N-inflation.
    canon_windows = [
        {
            "label":            w,
            "return_pct":       r["return_pct"],
            "trades":           r["trades"],
            "pnl_pct":          r.get("pnl_pct", []),
            "eq_impact_pnl_pct": [],
        }
        for w, r in per_window.items()
    ]
    canon = build_canonical_block(
        canon_windows, aggregation_method=AGGREGATION_VERSION
    )
    # Headline canonical PSR (window-level equity-curve series).
    psr = dict(canon["psr_walkforward"])

    summary = {
        "n_trades":              n_trades,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{n_total}",
        "per_window_return_pct": [round(x, 4) for x in rets_per_window],
    }
    per_window_clean = {
        w: {k: v for k, v in r.items() if k != "pnl_pct"}
        for w, r in per_window.items()
    }
    return {
        "summary":              summary,
        "per_window":           per_window_clean,
        "psr":                  psr,                    # canonical headline
        "legacy_psr_stitched":  legacy_psr_stitched,    # observability only
        "canonical":            canon,                  # full dual-emit block
        "aggregation_method":   canon["aggregation_method"],
    }


def verdict(coin: str, result: dict) -> str:
    s = result["summary"]
    p = result["psr"]["psr_vs_hurdle"]
    comp = s["compounded_pct"]
    wins = int(s["windows_positive"].split("/")[0])
    if comp > 0 and wins >= 3 and p > 0.5:
        return "transfers_with_gate"
    elif comp > 0:
        return "partial_with_gate"
    else:
        return "still_coin_specific"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    out: dict = {
        "strategy": "multifactor-v1-4h-gate",
        "cash": CASH,
        "commission": COMMISSION,
        "margin": MARGIN,
        "windows": [w[0] for w in WINDOWS],
        "locked_config_plus_gate": {**LOCKED, "use_mtf_4h_gate": True},
        "coins": {},
        "delta_vs_pre_gate": {},
        "verdicts": {},
    }

    for coin in ("ETH", "SOL"):
        parquet_15m = PARQ_15M[coin]
        parquet_4h = PARQ_4H[coin]
        print(f"\n[mf_4h_multi] === {coin} ===", file=sys.stderr)
        print(f"  15m parquet : {parquet_15m.name}", file=sys.stderr)
        print(f"  4H  parquet : {parquet_4h.name}", file=sys.stderr)

        # Config = LOCKED + gate ON + per-coin 4H parquet path
        config = {
            **LOCKED,
            "use_mtf_4h_gate":      True,
            "mtf_4h_parquet_path":  str(parquet_4h),
        }

        per_window = {}
        for label, start, end in WINDOWS:
            print(f"  {label} ...", file=sys.stderr)
            try:
                df = _load_slice(parquet_15m, start, end)
            except ValueError as exc:
                print(f"  {label}: SKIP ({exc})", file=sys.stderr)
                per_window[label] = {
                    "trades": 0, "return_pct": 0.0, "max_dd_pct": 0.0,
                    "win_rate_pct": 0.0, "equity_final": CASH, "pnl_pct": [],
                }
                continue
            result = run_one(df, config)
            per_window[label] = result
            print(
                f"  {label}: trades={result['trades']} return={result['return_pct']:.4f}%",
                file=sys.stderr,
            )

        coin_result = aggregate_and_psr(per_window)
        out["coins"][coin] = coin_result

        comp = coin_result["summary"]["compounded_pct"]
        pre = PRE_GATE[coin]
        out["delta_vs_pre_gate"][coin] = {
            "pre_gate_compounded":  pre,
            "post_gate_compounded": comp,
            "delta_pp":             round(comp - pre, 4),
        }

        v = verdict(coin, coin_result)
        out["verdicts"][coin] = v
        s = coin_result["summary"]
        psr_val = coin_result["psr"]["psr_vs_hurdle"]
        print(
            f"  -> compounded={s['compounded_pct']}% wins={s['windows_positive']} "
            f"PSR={psr_val:.4f} delta={comp - pre:+.2f}pp  VERDICT: {v}",
            file=sys.stderr,
        )

    # Write output
    out_path = ROOT / "reports" / "multifactor_v1_4h_multi_coin.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[mf_4h_multi] wrote {out_path}", file=sys.stderr)

    # Summary to stdout
    print("\n=== SUMMARY ===")
    for coin in ("ETH", "SOL"):
        s = out["coins"][coin]["summary"]
        p = out["coins"][coin]["psr"]["psr_vs_hurdle"]
        d = out["delta_vs_pre_gate"][coin]
        v = out["verdicts"][coin]
        print(
            f"{coin}: compounded={s['compounded_pct']:+.2f}%  wins={s['windows_positive']}  "
            f"PSR={p:.4f}  delta={d['delta_pp']:+.2f}pp  -> {v}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
