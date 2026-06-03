"""Deterministic synthetic tests for tools/aggregate.py — methodology debt #1.

Runs in <5s, no parquet/exchange reads.  See tools/aggregate.py docstring
for the canonical rule.  Reference: backcompat_baselines.json after Phase 2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.aggregate import (  # noqa: E402
    AGGREGATION_VERSION,
    aggregate_windows,
    build_canonical_block,
    equity_impact_returns,
    legacy_stitched_psr,
    phase2_gate,
    window_return_pct,
)
from tools.psr_eval import compute_psr  # noqa: E402


# ----- Stub helpers ---------------------------------------------------------


class _StubStats(dict):
    """A dict that also exposes `_trades` like backtesting.py stats."""

    def __init__(self, *args, _trades: pd.DataFrame | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._trades = _trades


# ----- Test 1: window_return_pct prefers stats['Return [%]'] -----------------


def test_window_return_pct_matches_stats():
    stats = _StubStats({"Return [%]": 12.34})
    assert window_return_pct(stats) == pytest.approx(12.34, abs=1e-9)


def test_window_return_pct_zero_when_missing():
    stats = _StubStats({})
    assert window_return_pct(stats) == 0.0
    stats2 = _StubStats({"Return [%]": None})
    assert window_return_pct(stats2) == 0.0


# ----- Test 2: equity_impact_returns under fractional sizing -----------------


def test_equity_impact_returns_under_fractional_sizing():
    # Known PnL=[+275, -137.5, +550], cash=10000
    # equity_at_entry = [10000, 10275, 10137.5]
    # eq_imp_pct      = [2.75, -1.3382...,  5.4253...]
    trades = pd.DataFrame(
        {
            "EntryTime": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "ExitTime":  pd.date_range("2024-01-01 00:30", periods=3, freq="1h"),
            "PnL":       [275.0, -137.5, 550.0],
            "ReturnPct": [0.0275, -0.01375, 0.055],  # irrelevant here
        }
    )
    stats = _StubStats({"Return [%]": 6.875}, _trades=trades)
    eq_imp = equity_impact_returns(stats, cash=10000.0)
    expected = np.array([2.75, -1.3382, 5.4253])
    assert eq_imp == pytest.approx(expected, abs=1e-3)


# ----- Test 3: equity_impact diverges from ReturnPct under fractional --------


def test_equity_impact_diverges_from_returnpct_under_fractional():
    # Two stub runs with IDENTICAL ReturnPct but DIFFERENT PnL (different sizing).
    # prod(1+ReturnPct) is the same; equity-impact is different.
    rp = np.array([0.02, -0.01, 0.03])
    same_ret_pct = np.prod(1 + rp) - 1

    # Run A: small sizing
    trades_A = pd.DataFrame(
        {
            "EntryTime": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "ExitTime":  pd.date_range("2024-01-01 00:30", periods=3, freq="1h"),
            "PnL":       [200.0, -100.0, 300.0],
            "ReturnPct": rp,
        }
    )
    stats_A = _StubStats({"Return [%]": same_ret_pct * 100}, _trades=trades_A)

    # Run B: 5x larger PnL, same ReturnPct -> same prod, different eq-impact
    trades_B = pd.DataFrame(
        {
            "EntryTime": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "ExitTime":  pd.date_range("2024-01-01 00:30", periods=3, freq="1h"),
            "PnL":       [1000.0, -500.0, 1500.0],
            "ReturnPct": rp,
        }
    )
    stats_B = _StubStats({"Return [%]": same_ret_pct * 100}, _trades=trades_B)

    eqA = equity_impact_returns(stats_A, cash=10000.0)
    eqB = equity_impact_returns(stats_B, cash=10000.0)

    # prod(1+ReturnPct) identical in both
    prod_A = np.prod(1 + rp)
    prod_B = np.prod(1 + rp)
    assert prod_A == pytest.approx(prod_B, abs=1e-12)

    # equity-impact arrays MUST differ
    assert not np.allclose(eqA, eqB)


# ----- Test 4: aggregate_windows compounded uses equity-curve returns --------


def test_aggregate_windows_compounded_uses_equity_curve():
    per_window = [
        {"label": "w1", "return_pct": 10.0, "trades": 5, "pnl_pct": [], "eq_impact_pnl_pct": []},
        {"label": "w2", "return_pct": -5.0, "trades": 4, "pnl_pct": [], "eq_impact_pnl_pct": []},
        {"label": "w3", "return_pct": 20.0, "trades": 6, "pnl_pct": [], "eq_impact_pnl_pct": []},
    ]
    agg = aggregate_windows(per_window)
    expected = (1.10 * 0.95 * 1.20 - 1.0) * 100.0
    assert agg["compounded_pct"] == pytest.approx(expected, abs=1e-6)
    assert agg["windows_positive"] == "2/3"
    assert agg["aggregation_method"] == "v2_equity_curve"
    assert agg["aggregation_version"] == AGGREGATION_VERSION


# ----- Test 5: psr_walkforward n == n_windows (defeats N-inflation) ----------


def test_psr_walkforward_n_equals_n_windows():
    # 5 windows with synthetic returns; psr_walkforward n_trades MUST be 5,
    # not 50 (would be the stitched count).
    per_window = [
        {"label": f"w{i}", "return_pct": float(r), "trades": 10,
         "pnl_pct": [0.5] * 10, "eq_impact_pnl_pct": [0.5] * 10}
        for i, r in enumerate([5.0, -3.0, 8.0, 1.0, 4.0])
    ]
    agg = aggregate_windows(per_window)
    assert agg["psr_walkforward"]["n_trades"] == 5
    assert agg["n_trades_sum"] == 50  # stitched count preserved as a separate field


# ----- Test 6: psr_per_window uses equity-impact NOT ReturnPct ---------------


def test_psr_per_window_uses_equity_impact_not_returnpct():
    # Build a window where ReturnPct stream and equity-impact stream
    # produce different PSRs.  Per-window PSR must equal the eq-impact one.
    rp = np.array([2.0, -1.0, 2.0, -1.0, 3.0])              # ReturnPct %
    eq = np.array([0.5, -0.3, 0.4, -0.2, 0.6])              # eq-impact %, smaller magnitude
    per_window = [{
        "label": "single",
        "return_pct": float(np.sum(eq)),
        "trades": len(eq),
        "pnl_pct": rp.tolist(),
        "eq_impact_pnl_pct": eq.tolist(),
    }]
    agg = aggregate_windows(per_window)
    eq_psr = compute_psr(eq, sr_hurdle=0.0, confidence=0.95)
    rp_psr = compute_psr(rp, sr_hurdle=0.0, confidence=0.95)
    # Must use eq-impact, NOT ReturnPct
    assert agg["psr_per_window"][0]["point_sharpe"] == pytest.approx(
        eq_psr["point_sharpe"], abs=1e-6
    )
    # And must DIFFER from the ReturnPct PSR (load-bearing — confirms
    # the canonical fix actually changed something).
    assert agg["psr_per_window"][0]["point_sharpe"] != pytest.approx(
        rp_psr["point_sharpe"], abs=1e-3
    )


# ----- Test 7: legacy_stitched_psr is deprecation-flagged --------------------


def test_legacy_stitched_psr_is_deprecation_flagged():
    per_window = [
        {"label": "w1", "pnl_pct": [1.0, -0.5, 2.0], "trades": 3, "return_pct": 2.5,
         "eq_impact_pnl_pct": []},
        {"label": "w2", "pnl_pct": [0.7, -1.0],      "trades": 2, "return_pct": -0.3,
         "eq_impact_pnl_pct": []},
    ]
    out = legacy_stitched_psr(per_window)
    assert out["deprecation"] == "stitched_per_trade_pl_pct_psr_is_N_inflated"
    # n_trades reflects the stitched count (N-inflation, preserved for diff)
    assert out["n_trades"] == 5


# ----- Test 8: dual-emit present in canonical block --------------------------


def test_dual_emit_present_in_canonical_block():
    per_window = [
        {"label": "w1", "return_pct": 4.0, "trades": 3,
         "pnl_pct": [1.0, 1.5, 1.5], "eq_impact_pnl_pct": [1.0, 1.5, 1.5]},
        {"label": "w2", "return_pct": -1.0, "trades": 2,
         "pnl_pct": [-0.5, -0.5], "eq_impact_pnl_pct": [-0.5, -0.5]},
    ]
    block = build_canonical_block(per_window)
    # canonical
    assert "compounded_pct" in block
    assert "psr_walkforward" in block
    # legacy
    assert "legacy_compounded_pct" in block
    assert "legacy_psr_stitched" in block
    assert block["legacy_psr_stitched"]["deprecation"] == \
        "stitched_per_trade_pl_pct_psr_is_N_inflated"
    assert block["aggregation_method"] == "v2_equity_curve"


# ----- Test 9: walkforward aggregation_method tagged distinctly --------------


def test_walkforward_aggregation_method_tagged_distinctly():
    per_window = [
        {"label": "q1", "return_pct": 2.0, "trades": 8,
         "pnl_pct": [0.25] * 8, "eq_impact_pnl_pct": []},
        {"label": "q2", "return_pct": -1.0, "trades": 6,
         "pnl_pct": [-0.16] * 6, "eq_impact_pnl_pct": []},
    ]
    agg = aggregate_windows(per_window, aggregation_method="v2_walkforward")
    assert agg["aggregation_method"] == "v2_walkforward"
    # Firewall — bogus tag rejected
    with pytest.raises(ValueError):
        aggregate_windows(per_window, aggregation_method="bogus")


# ----- Test 10: funding-skip dual-emit semantics ----------------------------


def test_funding_skip_funding_adjusted_method_allowed():
    per_window = [
        {"label": "w1", "return_pct": 3.0, "trades": 4,
         "pnl_pct": [1.0, 1.0, 0.5, 0.5], "eq_impact_pnl_pct": [1.0, 1.0, 0.5, 0.5]},
    ]
    agg = aggregate_windows(per_window, aggregation_method="v2_equity_curve_funding_adjusted")
    assert agg["aggregation_method"] == "v2_equity_curve_funding_adjusted"


# ----- Test 11: phase2 baseline gate logic ----------------------------------


def _wf_series(n_pos: int, n_neg: int) -> list[float]:
    """Build a WF quarterly return series with `n_pos` wins / `n_neg` losses.

    Wins are large enough that the WF-PSR comfortably clears the 0.90 floor,
    so each case below isolates exactly the dimension it intends to test
    (positive-rate vs PSR vs n-guard) rather than confounding the two.
    """
    return [3.0] * n_pos + [-1.0] * n_neg


def test_phase2_gate_true_wf_series_pass_and_breach():
    # ----- REDESIGNED PATH: gate consumes the true WF quarterly series -----

    # Case (a) — deployed pass: 20/25 positive (80%) >> 70% floor; PSR >> 0.90.
    s_pass = _wf_series(20, 5)
    assert phase2_gate({}, {}, deployed=True, wf_quarterly_returns=s_pass) == "PROCEED"

    # Case (b) — WF positive-rate breach on a DEPLOYED strategy -> HALT.
    # 16/25 = 64% < 70% floor. (PSR of this series still clears 0.90, so the
    # positive-rate is the sole thing failing — exactly the proxy-masked case.)
    s_low_rate = _wf_series(16, 9)
    assert phase2_gate({}, {}, deployed=True, wf_quarterly_returns=s_low_rate) == "HALT_AND_SURFACE"

    # Case (c) — SCOPE DECISION (B): the SAME breach on a NON-deployed
    # candidate is now graded (no more deployed=False short-circuit) and
    # returns the SHELF severity tag rather than a free PROCEED.
    assert phase2_gate({}, {}, deployed=False, wf_quarterly_returns=s_low_rate) == "SHELF"


def test_phase2_gate_psr_floor_breach():
    # ISOLATE the PSR floor: 8 small wins + 2 big losses. Positive-rate is
    # 80% (>= 0.70, so wf_pos_floor is NOT the cause) but the two large
    # losses drag the Sharpe negative -> PSR well below 0.90. This is the
    # ONLY test where the PSR term is the sole reason for the breach, so it
    # genuinely guards `psr < psr_floor` (drop that term and this fails).
    series = [0.5] * 8 + [-3.0] * 2
    arr = np.asarray(series, dtype=float)
    assert (arr > 0).mean() >= 0.70  # positive-rate passes -> NOT the cause
    psr = compute_psr(arr, sr_hurdle=0.0, confidence=0.95)["psr_vs_hurdle"]
    assert psr < 0.90  # the PSR floor is the sole failing dimension
    assert phase2_gate({}, {}, deployed=True, wf_quarterly_returns=series) == "HALT_AND_SURFACE"
    assert phase2_gate({}, {}, deployed=False, wf_quarterly_returns=series) == "SHELF"


def test_phase2_gate_insufficient_wf_evidence():
    # Guard (D): fewer than WF_MIN_QUARTERS quarters -> refuse to render a
    # pass/fail verdict rather than wave a thin series through.
    thin = _wf_series(5, 0)  # n=5 < 8
    assert phase2_gate({}, {}, deployed=True, wf_quarterly_returns=thin) == "insufficient_wf_evidence"
    assert phase2_gate({}, {}, deployed=False, wf_quarterly_returns=thin) == "insufficient_wf_evidence"


def test_phase2_gate_wf_result_mapping_carrier():
    # The WF series may also arrive as a Mapping with a `per_window` list
    # (the WF JSON shape) via the `wf_result` parameter.
    wf_json = {"per_window": [{"return_pct": r} for r in _wf_series(20, 5)]}
    assert phase2_gate({}, {}, deployed=True, wf_result=wf_json) == "PROCEED"
    wf_json_bad = {"per_window": [{"return_pct": r} for r in _wf_series(16, 9)]}
    assert phase2_gate({}, {}, deployed=False, wf_result=wf_json_bad) == "SHELF"


def test_phase2_gate_backcompat_proxy_fallback_keeps_deployed_proceed():
    # BACKCOMPAT PATH: when NO WF series is supplied (the deployed runner's
    # existing call), the gate falls back to the legacy 5-OOS proxy fields so
    # the DEPLOYED decision stays byte-identical to the pre-redesign gate.
    canon_proxy = {
        "compounded_pct": 77.952,
        "psr_walkforward": {"psr_vs_hurdle": 0.996555},  # n=5 proxy
        "windows_positive_pct": 100.0,                   # 5/5 proxy
    }
    # This is exactly what tools/_postfrac_mf_4h_btc_run.py passes.
    assert phase2_gate({}, canon_proxy, deployed=True) == "PROCEED"

    # Backcompat proxy ALSO still grades non-deployed candidates now (no
    # short-circuit). A proxy breach on a non-deployed strategy -> SHELF.
    canon_proxy_low = {
        "psr_walkforward": {"psr_vs_hurdle": 0.85},
        "windows_positive_pct": 100.0,
    }
    assert phase2_gate({}, canon_proxy_low, deployed=False) == "SHELF"


# ----- Test 12: PRICE_SCALE invariance --------------------------------------


def test_price_scale_invariance():
    # Two "runs" with identical trades — only stitched/eq-impact pct are
    # invariant to OHLC scaling under the harness contract.
    per_window = [
        {"label": "w1", "return_pct": 7.5, "trades": 5,
         "pnl_pct": [1.5, 2.0, -0.5, 1.0, 3.5],
         "eq_impact_pnl_pct": [1.5, 2.0, -0.5, 1.0, 3.5]},
        {"label": "w2", "return_pct": -2.0, "trades": 4,
         "pnl_pct": [-0.5, -1.0, -0.5, 0.0],
         "eq_impact_pnl_pct": [-0.5, -1.0, -0.5, 0.0]},
    ]
    # Aggregating twice (same input, simulating two PRICE_SCALE runs that
    # emit identical post-harness per-window dicts) MUST yield identical
    # compounded + psr.
    agg1 = aggregate_windows(per_window)
    agg2 = aggregate_windows(per_window)
    assert agg1["compounded_pct"] == agg2["compounded_pct"]
    assert agg1["psr_walkforward"]["psr_vs_hurdle"] == \
        agg2["psr_walkforward"]["psr_vs_hurdle"]


# ----- Smoke test 13: equity_impact respects window_start filter ------------


def test_equity_impact_respects_window_start_filter():
    trades = pd.DataFrame(
        {
            "EntryTime": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "ExitTime":  pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-02"]),
            "PnL":       [100.0, 200.0, 300.0],
            "ReturnPct": [0.01, 0.02, 0.03],
        }
    )
    stats = _StubStats({"Return [%]": 6.0}, _trades=trades)
    eq = equity_impact_returns(stats, cash=10000.0,
                                window_start=pd.Timestamp("2024-02-01"))
    # First trade filtered out of the OOS attribution -> 2 entries
    assert len(eq) == 2
    # First kept trade: PnL=200, equity_at_entry=10100 (cash + warm-prefix
    # PnL of +100 from the filtered-out first trade). The warm prefix
    # compounds equity BEFORE the window starts; it is no longer discarded.
    assert eq[0] == pytest.approx(1.9802, abs=1e-4)   # 200 / 10100 * 100
    # Second kept trade: PnL=300, equity_at_entry=10000+100+200=10300
    assert eq[1] == pytest.approx(2.9126, abs=1e-4)   # 300 / 10300 * 100


# ----- Smoke test 14: warm prefix compounds equity before the OOS window -----


def test_equity_impact_window_returns_match_full_series_positionally():
    """The OOS window slice must equal the full-series returns at those same
    trades — proving the warm prefix is compounded into equity_at_entry and
    NOT reset to `cash` at the first kept trade.

    Distinct from test 13: uses a TWO-trade warm prefix and asserts the
    windowed result is positionally identical to the un-windowed full series
    (not just a hand-computed value), which is the load-bearing invariant.
    """
    trades = pd.DataFrame(
        {
            "EntryTime": pd.to_datetime(
                ["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"]
            ),
            "ExitTime": pd.to_datetime(
                ["2024-01-02", "2024-01-16", "2024-02-02", "2024-03-02"]
            ),
            "PnL": [100.0, 50.0, 200.0, 300.0],
            "ReturnPct": [0.01, 0.005, 0.02, 0.03],
        }
    )
    stats = _StubStats({"Return [%]": 6.5}, _trades=trades)

    full = equity_impact_returns(stats, cash=10000.0)            # all 4 trades
    windowed = equity_impact_returns(
        stats, cash=10000.0, window_start=pd.Timestamp("2024-02-01")
    )

    # Two trades fall inside the OOS window (2024-02-01 and 2024-03-01).
    assert len(windowed) == 2
    # The windowed slice equals the LAST TWO entries of the full series — i.e.
    # the warm prefix (+100, +50) is compounded into equity before the window.
    assert windowed == pytest.approx(full[2:], abs=1e-9)
    # Concretely: first OOS trade sees equity_at_entry = 10000+100+50 = 10150,
    # NOT cash (10000). 200 / 10150 * 100 = 1.97044...
    assert windowed[0] == pytest.approx(200.0 / 10150.0 * 100.0, abs=1e-9)
    # And strictly less than the buggy cash-reset value (200/10000*100 = 2.0).
    assert windowed[0] < 2.0
