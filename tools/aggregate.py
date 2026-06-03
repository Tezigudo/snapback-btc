"""Canonical return-aggregation module — methodology debt #1 fix.

Single source of truth for how the 5-OOS and walk-forward family of runners
turn raw `backtesting.py` stats into headline `compounded_pct` and PSR.

Background
----------
Three runners (rv_band, kc_squeeze, turn_of_candle_15m) historically computed
their headline multi-window return as

    np.prod(1.0 + ReturnPct[OOS-filtered]) - 1.0

where `ReturnPct` is `backtesting.py`'s price-move pct (sign*(exit/entry-1))
— NOT the equity-impact return.  Under fractional risk-based sizing
(risk_per_trade_pct=2.75 etc.) notional != equity, so that aggregation is
sizing-blind and drifts vs the canonical `stats["Return [%]"]` used by 14
sibling runners (and by the locked +45.52% / +55.73% / +77.93% references).

Walk-forward runners legitimately use `prod(1+ReturnPct)` per quarter
because the underlying engine's headline spans train+OOS, so they cannot
read `stats["Return [%]"]` for OOS-only attribution.  Their semantics are
clearly different — tag them with `aggregation_method="v2_walkforward"`
so the two families NEVER get cross-compared.

This module exports the deterministic helpers used by every runner:

    AGGREGATION_VERSION         — string constant
    window_return_pct(stats)    — canonical per-window headline
    equity_impact_returns(stats, cash)   — per-trade PnL/equity-at-entry
    aggregate_windows(per_window, aggregation_method=...)
                                — canonical aggregate (compounded, PSR)
    legacy_stitched_psr(per_window)
                                — deprecation-flagged stitched-PSR for
                                  backcompat diff only.  N-inflated; NEVER
                                  use as primary verdict input.
    phase2_gate(v1_locked, v2_result, deployed)
                                — rollout safety helper

Rules of use
------------
- v1 fields (`legacy_*`) are dual-emitted forever for audit; never drop.
- `aggregation_method` MUST be present in every runner output.
- 5-OOS family: per-window `return_pct` MUST come from `window_return_pct(stats)`.
- WF family: per-window `return_pct` MAY come from `prod(1+ReturnPct)`, tag
  `aggregation_method="v2_walkforward"`.
- Primary multi-window PSR = `aggregate_windows(...)["psr_walkforward"]` —
  compute_psr on the WINDOW-level return series (n == n_windows).  This
  defeats the N-inflation seen with stitched per-trade ReturnPct.
- Per-window PSR (where available) MUST be computed on equity-impact
  returns within that single contiguous window (PnL/equity-at-entry),
  NOT on `ReturnPct`.
- PRICE_SCALE is orthogonal to this fix: it only fixes integer-unit
  truncation, it does NOT change either aggregation method.

References
----------
- BTC_SOL_PORTFOLIO_VERDICT.md (debt #2 — stitched per-trade PSR is
  N-inflated; canonical fix is psr_walkforward on quarter-series).
- CLAUDE.md "BEFORE DEPLOY/PUSH" — if Phase 2 baseline gate trips on a
  deployed strategy, HALT and surface to user.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from tools.psr_eval import compute_psr

# Version constant — stamp every emit with this for audit.
AGGREGATION_VERSION = "v2_equity_curve"

# Allowed aggregation_method tags (firewall — anything else is a bug).
ALLOWED_METHODS = (
    "v2_equity_curve",            # canonical 5-OOS family
    "v2_walkforward",             # WF family (per-quarter prod(1+ReturnPct))
    "v2_equity_curve_funding_adjusted",  # soft-drift funding-net runner
)


# ---------------------------------------------------------------------------
# Canonical per-window read-outs
# ---------------------------------------------------------------------------

def window_return_pct(stats: Mapping[str, Any]) -> float:
    """Return the canonical headline per-window return percent.

    By rule, this is `stats["Return [%]"]` — the engine's equity-curve
    Return%, sizing-aware (notional via fractional risk + leverage all
    reduce to a single % change of starting cash).  Drifters used
    `prod(1+ReturnPct)` instead, which is sizing-blind.
    """
    val = stats.get("Return [%]", 0.0) if hasattr(stats, "get") else 0.0
    if val is None:
        return 0.0
    return float(val)


def equity_impact_returns(
    stats: Mapping[str, Any],
    cash: float,
    *,
    window_start: pd.Timestamp | None = None,
) -> np.ndarray:
    """Per-trade equity-impact return series for a single contiguous window.

    Builds `PnL_i / equity_at_entry_i * 100` in trade-execution order
    (sorted by ExitTime).  This is the sizing-aware PSR input.

    Parameters
    ----------
    stats:
        A backtesting.py stats object (or a dict containing `_trades`).
    cash:
        Starting equity for the window.
    window_start:
        If supplied, only trades with `EntryTime >= window_start` are
        included (OOS attribution for warm-prefix harnesses).

    Returns
    -------
    np.ndarray of equity-impact returns IN PERCENT (e.g. 2.75 means +2.75%).
    """
    trades_df = getattr(stats, "_trades", None)
    if trades_df is None:
        if isinstance(stats, Mapping) and "_trades" in stats:
            trades_df = stats["_trades"]
    if trades_df is None or len(trades_df) == 0:
        return np.array([], dtype=float)

    df = trades_df.copy()
    if "PnL" not in df.columns or len(df) == 0:
        return np.array([], dtype=float)

    if "ExitTime" in df.columns:
        df = df.sort_values("ExitTime")          # sort FULL df before cumsum

    pnls = df["PnL"].astype(float).values
    # Equity at each trade's entry: starting cash compounded by ALL previous
    # PnL (including warm-prefix trades before window_start).
    cum_pnl_prev = np.concatenate(([0.0], np.cumsum(pnls)[:-1]))
    equity_at_entry = float(cash) + cum_pnl_prev
    # Guard div-by-zero (degenerate path) — replace with cash to keep finite
    equity_at_entry = np.where(equity_at_entry == 0.0, float(cash), equity_at_entry)
    returns = (pnls / equity_at_entry) * 100.0

    # OOS attribution: keep only window trades, but AFTER equity has been
    # compounded through the warm prefix. Mask is derived from the SORTED df
    # so it stays positionally aligned with `returns`.
    if window_start is not None and "EntryTime" in df.columns:
        mask = (df["EntryTime"] >= window_start).values
        returns = returns[mask]
    return returns


# ---------------------------------------------------------------------------
# Canonical multi-window aggregation
# ---------------------------------------------------------------------------

def _safe_compute_psr(returns: np.ndarray) -> dict:
    if len(returns) < 2:
        return {
            "n_trades": int(len(returns)),
            "psr_vs_hurdle": 0.0,
            "interpretation": "insufficient_evidence",
        }
    return compute_psr(returns, sr_hurdle=0.0, confidence=0.95)


def aggregate_windows(
    per_window: list[Mapping[str, Any]],
    *,
    aggregation_method: str = AGGREGATION_VERSION,
) -> dict:
    """Canonical multi-window aggregate block (dual-emit body).

    Each `per_window` entry SHOULD carry:
      - return_pct        — the canonical per-window headline (already
                            computed by `window_return_pct(stats)` for
                            5-OOS, or by prod(1+ReturnPct) for WF).
      - eq_impact_pnl_pct — equity-impact return series for the window
                            (used for per-window PSR).
      - pnl_pct           — legacy per-trade ReturnPct % series (stitched
                            for backcompat PSR diffing).
      - trades            — int

    Returns a dict with `aggregation_method`, `compounded_pct`,
    `windows_positive`, `per_window_return_pct`, `psr_per_window` and
    `psr_walkforward`.

    Note: `psr_walkforward` is `compute_psr` on the n-window
    return-series.  `n_trades` in that block == number of windows.
    That is intentional — it defeats the N-inflation seen when PSR is
    computed on the stitched per-trade ReturnPct union.
    """
    if aggregation_method not in ALLOWED_METHODS:
        raise ValueError(
            f"aggregation_method={aggregation_method!r} not in {ALLOWED_METHODS}"
        )

    per_window_return_pct: list[float] = []
    n_pos = 0
    compounded = 1.0
    n_trades_sum = 0
    psr_per_window: list[dict] = []

    for r in per_window:
        ret_pct = float(r.get("return_pct", 0.0) or 0.0)
        per_window_return_pct.append(round(ret_pct, 4))
        compounded *= (1.0 + ret_pct / 100.0)
        if ret_pct > 0.0:
            n_pos += 1
        n_trades_sum += int(r.get("trades", 0) or 0)

        eq_imp = r.get("eq_impact_pnl_pct")
        if eq_imp is None:
            eq_imp = []
        eq_arr = np.asarray(list(eq_imp), dtype=float)
        psr_per_window.append(
            {
                "label": r.get("label"),
                **_safe_compute_psr(eq_arr),
            }
        )

    psr_walkforward = _safe_compute_psr(
        np.asarray(per_window_return_pct, dtype=float)
    )

    return {
        "aggregation_method":    aggregation_method,
        "aggregation_version":   AGGREGATION_VERSION,
        "n_windows":             len(per_window),
        "n_trades_sum":          n_trades_sum,
        "compounded_pct":        round((compounded - 1.0) * 100.0, 4),
        "windows_positive":      f"{n_pos}/{len(per_window)}",
        "windows_positive_pct":  round(100.0 * n_pos / max(len(per_window), 1), 2),
        "per_window_return_pct": per_window_return_pct,
        "psr_per_window":        psr_per_window,
        "psr_walkforward":       psr_walkforward,
    }


def legacy_stitched_psr(per_window: list[Mapping[str, Any]]) -> dict:
    """Stitched-per-trade-ReturnPct PSR (DEPRECATED — keep for diff only).

    Concatenates every window's `pnl_pct` (per-trade ReturnPct %) into a
    single series and runs `compute_psr` on it.  This is what every
    runner used to emit pre-fix.  It is N-inflated and sizing-blind:
    treats trades across disjoint windows as if drawn from one
    contiguous return-generating process.

    Returned dict carries a `deprecation` string so downstream code
    can detect that this is BACKCOMPAT not canonical.
    """
    stitched: list[float] = []
    for r in per_window:
        pnl = r.get("pnl_pct")
        if pnl:
            stitched.extend([float(x) for x in pnl])
    arr = np.asarray(stitched, dtype=float)
    psr = _safe_compute_psr(arr)
    psr["deprecation"] = "stitched_per_trade_pl_pct_psr_is_N_inflated"
    psr["n_trades"] = int(len(arr))  # restore stitched count after _safe_compute_psr round
    return psr


# ---------------------------------------------------------------------------
# Phase 2 rollout safety gate
# ---------------------------------------------------------------------------

def phase2_gate(
    v1_locked: Mapping[str, Any],
    v2_result: Mapping[str, Any],
    *,
    deployed: bool = False,
    psr_floor: float = 0.90,
    wf_pos_floor: float = 0.70,
) -> str:
    """Decide whether the v1->v2 re-baseline of a strategy may proceed.

    Returns one of:
      - "PROCEED" — v2 metrics within tolerance, no live-bot risk.
      - "HALT_AND_SURFACE" — at least one safety threshold breached;
        rollout must pause and the user must be informed per CLAUDE.md
        "BEFORE DEPLOY/PUSH".

    Parameters
    ----------
    v1_locked, v2_result:
        Dicts each containing `compounded_pct`, `psr_walkforward`
        (dict from compute_psr) and `windows_positive_pct` (0..100).
    deployed:
        If True, applies the strict gate (this strategy is live on
        mainnet — any breach triggers HALT).
    psr_floor:
        Minimum acceptable v2 PSR (default 0.90 from plan).
    wf_pos_floor:
        Minimum acceptable v2 WF positive-quarter rate as a fraction
        (default 0.70 from plan).
    """
    psr_block = v2_result.get("psr_walkforward", {})
    psr = float(psr_block.get("psr_vs_hurdle", 0.0) or 0.0)
    wf_pos_pct = float(v2_result.get("windows_positive_pct", 0.0) or 0.0) / 100.0

    if deployed:
        if psr < psr_floor:
            return "HALT_AND_SURFACE"
        if wf_pos_pct < wf_pos_floor:
            return "HALT_AND_SURFACE"
    return "PROCEED"


# ---------------------------------------------------------------------------
# Result-assembly helper (cuts boilerplate in runners)
# ---------------------------------------------------------------------------

def build_canonical_block(
    per_window: list[Mapping[str, Any]],
    *,
    aggregation_method: str = AGGREGATION_VERSION,
) -> dict:
    """Build the canonical+legacy dual-emit block in one call.

    Returns a dict with the canonical (v2) view plus a SYNTHETIC
    stitched-per-trade reference (audit-only, NOT a real historical method):
      - compounded_pct, psr_walkforward, psr_per_window (canonical)
      - legacy_compounded_pct, legacy_psr_stitched (synthetic stitched ref;
        kept for the audit contract — see inline note. The locked historical
        numbers were per-window compounding, == compounded_pct, NOT this.)
      - aggregation_method, aggregation_version

    Each runner can do:
        canon = build_canonical_block(per_window)
        result = {"strategy_id": ..., **canon, ...}
    """
    canon = aggregate_windows(per_window, aggregation_method=aggregation_method)

    # SYNTHETIC stitched-per-trade reference — NOT a historical method.
    # This is prod(1+pnl_pct/100) over the per-trade ReturnPct union across
    # all windows (sizing-blind, N-inflated).  IMPORTANT: no historical
    # report ever used stitched-prod — the project's locked references
    # (+45.52% AdaptiveTrend, +55.73%/+77.93% multifactor) were computed by
    # PER-WINDOW compounding, which is what `canon["compounded_pct"]` does.
    # So `legacy_compounded_pct` is a straw-man baseline kept only because
    # the module contract (docstring line 41) says legacy_* fields are
    # dual-emitted forever for audit; `legacy_delta_pp` is therefore a diff
    # against a method nobody used, NOT an apples-to-apples v1 delta.  Do not
    # interpret it as such.  (A "rebase on per-window compounding" would just
    # equal canon["compounded_pct"], making the delta a tautological ~0.)
    stitched: list[float] = []
    for r in per_window:
        pnl = r.get("pnl_pct")
        if pnl:
            stitched.extend([float(x) for x in pnl])
    if stitched:
        c = 1.0
        for p in stitched:
            c *= 1.0 + p / 100.0
        legacy_compounded_pct = round((c - 1.0) * 100.0, 4)
    else:
        legacy_compounded_pct = 0.0

    legacy_psr = legacy_stitched_psr(per_window)

    return {
        **canon,
        "legacy_compounded_pct": legacy_compounded_pct,
        "legacy_psr_stitched":   legacy_psr,
        "legacy_delta_pp":       round(canon["compounded_pct"] - legacy_compounded_pct, 4),
    }
