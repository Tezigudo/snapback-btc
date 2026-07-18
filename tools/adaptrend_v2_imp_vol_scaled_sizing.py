"""
AdaptiveTrendV2 vol-scaled-sizing improvement evaluation.

Runs BOTH:
  - AdaptiveTrendV2 (base, 11-window OOS reference)
  - AdaptiveTrendV2_vol_scaled_sizing (one-feature variant)
through the IDENTICAL prefix-buffered harness, on the 11 OOS windows from
ADAPTIVE_TREND_EXTENDED_VERDICT, $1M cash, real funding.

For each class:
  - Aggregates per-trade pnl_pct across windows -> compute_psr for PSR+MinTRL.
  - Tracks per-window net%, trades, wins/11, compounded.

Writes reports/adaptrend_v2_imp_vol_scaled_sizing.json with both result blocks
and the delta vs base.

Run from repo root:
    .venv/bin/python tools/adaptrend_v2_imp_vol_scaled_sizing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from backtesting import Backtest  # noqa: E402

from backtest import funding_cost_for_trades  # noqa: E402
from strategy.signals_adaptive_trend_v2 import AdaptiveTrendV2  # noqa: E402
from strategy.signals_adaptive_trend_v2_vol_scaled_sizing import (  # noqa: E402
    AdaptiveTrendV2_vol_scaled_sizing,
)
from tools._adaptrend_v2_run import (  # noqa: E402
    _load_15m_with_prefix,
    _load_funding_slice,
)
from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,  # noqa: F401  (referenced via build_canonical_block tag)
    build_canonical_block,
    equity_impact_returns,
)
from tools.psr_eval import compute_psr  # noqa: E402

# 11 OOS windows — matches ADAPTIVE_TREND_EXTENDED_VERDICT.md
OOS_WINDOWS = [
    ("2020-07-01", "2020-12-31", "2020_H2"),
    ("2021-01-01", "2021-06-30", "2021_H1"),
    ("2021-07-01", "2021-12-31", "2021_H2"),
    ("2022-01-01", "2022-06-30", "2022_H1"),
    ("2022-07-01", "2022-12-31", "2022_H2"),
    ("2023-01-01", "2023-06-30", "2023_H1"),
    ("2023-07-01", "2023-12-31", "2023_H2"),
    ("2024-01-01", "2024-06-30", "2024_H1"),
    ("2024-07-01", "2024-12-31", "2024_H2"),
    ("2025-01-01", "2025-06-30", "2025_H1"),
    ("2025-07-01", "2025-12-31", "2025_H2"),
]

CASH = 1_000_000
COMMISSION = 0.0005
MARGIN = 1.0 / 20

_DEFAULT_PARQUET = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
_DEFAULT_FUNDING = ROOT / "data" / "historical" / "BTC_USDT_USDT_funding.parquet"


def _run_one_window(
    strategy_cls,
    start: str,
    end: str,
    cash: float,
    config: dict | None = None,
) -> tuple[dict, pd.DataFrame | None, object, pd.Timestamp]:
    """Run one OOS window through the prefix-buffered harness.

    Returns (summary_dict, trades_oos_df_or_None, stats, oos_start_ts).
    `stats` is the raw backtesting.py result; the canonical equity-curve
    aggregation needs it (via equity_impact_returns) for the per-window PSR.
    `oos_start_ts` is passed as window_start so equity-impact returns are
    attributed to OOS trades only (prefix trades are excluded after equity
    has compounded through them).
    """
    config = config or {}
    prefix_months = int(config.get("fit_window_months", 6))

    data_15m_full, oos_start_ts = _load_15m_with_prefix(
        _DEFAULT_PARQUET, start, end, prefix_months
    )
    funding = _load_funding_slice(_DEFAULT_FUNDING, start, end)

    bt = Backtest(
        data_15m_full,
        strategy_cls,
        cash=cash,
        commission=COMMISSION,
        margin=MARGIN,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    run_kwargs = dict(config)
    run_kwargs.setdefault("trade_start_ns", oos_start_ts.value)
    stats = bt.run(**run_kwargs)

    trades_df = getattr(stats, "_trades", None)
    if trades_df is not None and len(trades_df) > 0 and "EntryTime" in trades_df.columns:
        trades_oos = trades_df[trades_df["EntryTime"] >= oos_start_ts].copy()
    else:
        trades_oos = trades_df

    n_trades = int(len(trades_oos)) if trades_oos is not None else 0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        gross_pnl_usdt = float(trades_oos["PnL"].sum())
    else:
        gross_pnl_usdt = 0.0
    gross_return_pct = gross_pnl_usdt / cash * 100.0

    win_rate = 0.0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        win_rate = float((trades_oos["PnL"] > 0).mean() * 100.0)

    funding_cost_usdt = 0.0
    funding_events = 0
    if trades_oos is not None and len(trades_oos) > 0 and not funding.empty:
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_oos, data_15m_full, funding
        )

    gross_final_equity = cash * (1.0 + gross_return_pct / 100.0)
    net_final_equity = gross_final_equity - funding_cost_usdt
    net_return_pct = (net_final_equity / cash - 1.0) * 100.0

    max_dd_pct = 0.0
    if trades_oos is not None and len(trades_oos) > 0 and "PnL" in trades_oos.columns:
        ordered = (
            trades_oos.sort_values("ExitTime")
            if "ExitTime" in trades_oos.columns
            else trades_oos
        )
        equity = cash + ordered["PnL"].cumsum()
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max * 100.0
        max_dd_pct = float(dd.min()) if len(dd) > 0 else 0.0

    summary = {
        "start": start,
        "end": end,
        "trades": n_trades,
        "gross_return_pct": round(gross_return_pct, 4),
        "net_return_pct": round(net_return_pct, 4),
        "funding_cost_usdt": round(funding_cost_usdt, 2),
        "funding_events": funding_events,
        "win_rate_pct": round(win_rate, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "final_equity_net": round(net_final_equity, 2),
    }
    return summary, trades_oos, stats, oos_start_ts


def _run_class_over_windows(
    strategy_cls, label: str, config: dict | None = None
) -> dict:
    """Run a strategy class over all 11 OOS windows.

    Dual-emit:
      - legacy_psr_stitched: compute_psr on the stitched per-trade ReturnPct
        union (N-inflated, sizing-blind — kept for observability/diff only).
      - canonical: build_canonical_block on per-window dicts. The headline
        canonical PSR is canonical["psr_walkforward"] (compute_psr on the
        n-window net-return series); per-window PSR uses equity-impact returns.
    """
    rows = []
    all_pnl_pct: list[float] = []
    per_window_canon: list[dict] = []
    for start, end, win_label in OOS_WINDOWS:
        print(f"[{label}] {win_label} {start}..{end}", flush=True)
        summary, trades_oos, stats, oos_start_ts = _run_one_window(
            strategy_cls, start, end, CASH, config=config
        )
        summary["label"] = win_label
        rows.append(summary)
        print(
            f"  -> trades={summary['trades']:3d} net={summary['net_return_pct']:+6.2f}% "
            f"dd={summary['max_dd_pct']:+6.2f}%",
            flush=True,
        )
        win_pnl_pct: list[float] = []
        if trades_oos is not None and len(trades_oos) > 0 and "ReturnPct" in trades_oos.columns:
            # ReturnPct from backtesting.py is fractional; convert to percent to match pnl_pct convention.
            win_pnl_pct = (trades_oos["ReturnPct"].astype(float) * 100.0).tolist()
            all_pnl_pct.extend(win_pnl_pct)
        # Equity-impact series (PnL/equity-at-entry %) for this contiguous window.
        # window_start=oos_start_ts attributes returns to OOS trades only (the
        # prefix is empty by design, but be strict for consistency).
        eq_impact = equity_impact_returns(
            stats, cash=CASH, window_start=oos_start_ts
        ).tolist()
        per_window_canon.append(
            {
                "label": win_label,
                # canonical v2 headline = funding-adjusted net (sizing-aware).
                "return_pct": summary["net_return_pct"],
                "trades": summary["trades"],
                # gross per-trade ReturnPct% -> synthetic stitched legacy ref.
                "pnl_pct": win_pnl_pct,
                # sizing-aware equity-impact -> feeds psr_per_window.
                "eq_impact_pnl_pct": eq_impact,
            }
        )

    # Aggregates
    nets = [r["net_return_pct"] for r in rows]
    wins = sum(1 for n in nets if n > 0)
    total_trades = sum(r["trades"] for r in rows)
    comp = 1.0
    for n in nets:
        comp *= 1.0 + n / 100.0
    compounded_pct = (comp - 1.0) * 100.0

    pnl_arr = np.array(all_pnl_pct, dtype=float)
    psr = compute_psr(pnl_arr, sr_hurdle=0.0, confidence=0.95)

    # Canonical equity-curve dual-emit (methodology debt #1). Funding-net
    # headline -> tag v2_equity_curve_funding_adjusted.
    canon = build_canonical_block(
        per_window_canon,
        aggregation_method="v2_equity_curve_funding_adjusted",
    )

    return {
        "label": label,
        "rows": rows,
        "compounded_pct": round(compounded_pct, 4),
        "wins": wins,
        "windows": len(rows),
        "total_trades": total_trades,
        # legacy stitched PSR — observability only, NOT the canonical verdict.
        "psr_block": psr,
        "legacy_psr_stitched": psr,
        "n_pnl_samples": int(len(pnl_arr)),
        # canonical v2 equity-curve aggregation (primary metric).
        "canonical": canon,
        "aggregation_method": canon["aggregation_method"],
        "_per_window_return_pct": canon["per_window_return_pct"],
    }


def main() -> int:
    print("=== AdaptiveTrendV2 vol_scaled_sizing improvement eval ===", flush=True)
    print(f"  windows={len(OOS_WINDOWS)} cash={CASH}", flush=True)

    print("\n--- Base: AdaptiveTrendV2 ---", flush=True)
    base = _run_class_over_windows(AdaptiveTrendV2, label="v2_base", config={})

    print("\n--- Variant: AdaptiveTrendV2_vol_scaled_sizing ---", flush=True)
    variant = _run_class_over_windows(
        AdaptiveTrendV2_vol_scaled_sizing,
        label="v2_vol_scaled_sizing",
        config={"use_vol_scaled_sizing": True, "target_vol_annualised": 0.15},
    )

    # Delta
    delta = {
        "compounded_pp": round(variant["compounded_pct"] - base["compounded_pct"], 4),
        "wins_delta": variant["wins"] - base["wins"],
        "trades_delta": variant["total_trades"] - base["total_trades"],
        "point_sharpe_delta": round(
            variant["psr_block"]["point_sharpe"] - base["psr_block"]["point_sharpe"], 6
        ),
        "psr_delta": round(
            variant["psr_block"]["psr_vs_hurdle"] - base["psr_block"]["psr_vs_hurdle"], 6
        ),
        "min_trl_delta": variant["psr_block"]["min_trl"] - base["psr_block"]["min_trl"],
    }

    # Verdict
    dpp = delta["compounded_pp"]
    psr_improved = variant["psr_block"]["psr_vs_hurdle"] > base["psr_block"]["psr_vs_hurdle"]
    if dpp > 5.0 and psr_improved:
        verdict = "ACCRETIVE"
    elif -5.0 <= dpp <= 5.0:
        verdict = "NEUTRAL"
    elif dpp < -5.0:
        verdict = "HURTING"
    else:
        # dpp > 5 but PSR not improved
        verdict = "NEUTRAL_compounded_up_psr_flat"

    # --- Canonical self-check (bit-for-bit): re-derive the headline PSR from
    # the PERSISTED canonical per-window return series and assert it matches
    # the psr_walkforward the canonical block reports. ---
    canonical_selfcheck = {}
    for cls_label, blk in (("base", base), ("variant", variant)):
        canon = blk["canonical"]
        persisted_returns = np.asarray(blk["_per_window_return_pct"], dtype=float)
        recomputed = compute_psr(persisted_returns, contiguous=False)
        headline = canon["psr_walkforward"]
        match = (
            recomputed["psr_vs_hurdle"] == headline["psr_vs_hurdle"]
            and recomputed["n_trades"] == headline["n_trades"]
        )
        canonical_selfcheck[cls_label] = {
            "canonical_psr": headline["psr_vs_hurdle"],
            "recomputed_psr": recomputed["psr_vs_hurdle"],
            "matches_headline": bool(match),
        }
        assert match, (
            f"CANONICAL SELF-CHECK FAILED [{cls_label}]: "
            f"recomputed={recomputed['psr_vs_hurdle']} (n={recomputed['n_trades']}) "
            f"!= headline={headline['psr_vs_hurdle']} (n={headline['n_trades']})"
        )

    # Strip the internal helper key before persisting.
    for blk in (base, variant):
        blk.pop("_per_window_return_pct", None)

    out = {
        "experiment": "adaptrend_v2_imp_vol_scaled_sizing",
        "id": "vol_scaled_sizing",
        "base": base,
        "variant": variant,
        "delta": delta,
        "verdict": verdict,
        "canonical_psr_selfcheck": canonical_selfcheck,
    }

    out_path = ROOT / "reports" / "adaptrend_v2_imp_vol_scaled_sizing.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"  base    : comp={base['compounded_pct']:+.2f}% "
        f"wins={base['wins']}/{base['windows']} trades={base['total_trades']} "
        f"SR={base['psr_block']['point_sharpe']:.4f} "
        f"PSR={base['psr_block']['psr_vs_hurdle']:.3f} "
        f"MinTRL={base['psr_block']['min_trl']}",
        flush=True,
    )
    print(
        f"  variant : comp={variant['compounded_pct']:+.2f}% "
        f"wins={variant['wins']}/{variant['windows']} trades={variant['total_trades']} "
        f"SR={variant['psr_block']['point_sharpe']:.4f} "
        f"PSR={variant['psr_block']['psr_vs_hurdle']:.3f} "
        f"MinTRL={variant['psr_block']['min_trl']}",
        flush=True,
    )
    print(f"  delta   : Δcomp={delta['compounded_pp']:+.2f}pp verdict={verdict}", flush=True)
    print(
        f"  canon   : base psr_wf={base['canonical']['psr_walkforward']['psr_vs_hurdle']:.4f} "
        f"variant psr_wf={variant['canonical']['psr_walkforward']['psr_vs_hurdle']:.4f} "
        f"(legacy stitched base={base['legacy_psr_stitched']['psr_vs_hurdle']:.3f} "
        f"variant={variant['legacy_psr_stitched']['psr_vs_hurdle']:.3f})",
        flush=True,
    )
    print(f"  saved   : {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
