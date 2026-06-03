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

def _safe_compute_psr(returns: np.ndarray, *, contiguous: bool = True) -> dict:
    if len(returns) < 2:
        return {
            "n_trades": int(len(returns)),
            "psr_vs_hurdle": 0.0,
            "psr_lo_adjusted": 0.0,
            "interpretation": "insufficient_evidence",
        }
    return compute_psr(
        returns, sr_hurdle=0.0, confidence=0.95, contiguous=contiguous
    )


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

    # contiguous=False: the walk-forward series is n DISJOINT OOS windows, so
    # serial correlation is spurious -> Lo correction is a no-op here (Trap 2).
    psr_walkforward = _safe_compute_psr(
        np.asarray(per_window_return_pct, dtype=float),
        contiguous=False,
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
    # contiguous=False: stitched across disjoint windows -> Lo no-op (Trap 2).
    psr = _safe_compute_psr(arr, contiguous=False)
    psr["deprecation"] = "stitched_per_trade_pl_pct_psr_is_N_inflated"
    psr["n_trades"] = int(len(arr))  # restore stitched count after _safe_compute_psr round
    return psr


# ---------------------------------------------------------------------------
# Phase 2 rollout safety gate
# ---------------------------------------------------------------------------

#: Minimum number of walk-forward quarters required for the gate to
#: render a real verdict. Below this the WF positive-rate / WF-PSR are
#: too noisy to gate on, so we refuse to pass silently. n=8 is two
#: years of quarterly windows — the smallest series where a 70% bar
#: (>= 6/8) is meaningfully distinguishable from a coin flip.
WF_MIN_QUARTERS: int = 8


def phase2_gate(
    v1_locked: Mapping[str, Any],
    v2_result: Mapping[str, Any],
    *,
    deployed: bool = False,
    psr_floor: float = 0.90,
    wf_pos_floor: float = 0.70,
    wf_quarterly_returns: Iterable[float] | None = None,
    wf_result: Mapping[str, Any] | None = None,
) -> str:
    """Decide whether the v1->v2 re-baseline of a strategy may proceed.

    Returns one of:
      - "PROCEED" — metrics within tolerance, no rollout risk.
      - "HALT_AND_SURFACE" — a DEPLOYED strategy breached a floor; the
        rollout must pause and the user must be informed per CLAUDE.md
        "BEFORE DEPLOY/PUSH".
      - "SHELF" — a NON-deployed candidate breached a floor; the v1->v2
        re-baseline is rejected but there is no live-bot risk.
      - "insufficient_wf_evidence" — fewer than ``WF_MIN_QUARTERS``
        walk-forward quarters were supplied; the gate refuses to render
        a pass/fail rather than wave a thin series through.

    REDESIGN (Phase-2 gate, judgment call #1)
    -----------------------------------------
    The old gate read its WF positive-rate from
    ``v2_result["windows_positive_pct"]`` — but for the 5-OOS runner
    family that field is the *5-OOS* n_pos/n_windows (e.g. 5/5 = 100%),
    used as a PROXY for the true walk-forward quarterly positive-rate.
    Those are different distributions: the 5-OOS family hand-picks five
    disjoint stress windows, whereas the walk-forward series rolls a
    90-day test window forward quarterly across the whole history. The
    5-OOS proxy is systematically optimistic (a 5/5 run reads as 100%
    while its true WF rate can be 55-80%), which MASKED genuine WF
    failures (e.g. adaptrend_v1: 5-OOS proxy 100% vs true WF 55%).

    This redesign consumes the ACTUAL walk-forward quarterly series when
    it is supplied (``wf_quarterly_returns`` or ``wf_result``), and
    recomputes BOTH gate inputs from it:
      * wf_pos_pct = mean(series > 0)            (the real positive-rate)
      * wf_psr     = compute_psr(series)["psr_vs_hurdle"]   (n ~= 25)

    Why the PSR fix alone changes nothing, and the rate fix is the teeth
    ----------------------------------------------------------------------
    Every recorded WF-PSR already clears any sane floor (deployed mf+4H =
    0.9987 at n=25; adaptrend_v1 = 0.9932 at n=20). The old gate fed PSR
    from the 5-OOS window-series (n=5), where the standard error is so
    wide (sr_se ~ 0.31) that 0.90 was trivially cleared by any 5/5 run —
    "N-deflation". Moving PSR onto the n~=25 quarterly series tightens
    the standard error enough that 0.90 becomes a *real* bar, but no
    recorded strategy is anywhere near it, so the PSR change flips NO
    decision. ALL decision movement comes from swapping the optimistic
    5-OOS positive-rate proxy (100%) for the true WF positive-rate.

    SCOPE (the load-bearing choice)
    -------------------------------
    The two floor checks now run for ANY strategy being graded, not only
    the live one. The old gate short-circuited every ``deployed=False``
    call to "PROCEED" — which is exactly what hid the masked failures.
    ``deployed`` is retained only as a SEVERITY TAG on a breach:
    deployed -> "HALT_AND_SURFACE", non-deployed -> "SHELF".

    Backward compatibility (deployed bot must stay byte-unchanged)
    -------------------------------------------------------------
    The sole production call site
    (tools/_postfrac_mf_4h_btc_run.py) passes the 5-OOS ``canon`` block
    with NO walk-forward series. To keep the DEPLOYED strategy's gate
    decision byte-identical, when no WF series is supplied this falls
    back to the legacy proxy fields already inside ``v2_result``
    (``psr_walkforward`` + ``windows_positive_pct``) exactly as before.
    For the deployed mf+4H both paths agree on PROCEED (proxy: 100% &
    PSR 0.9966; true WF: 80% & PSR 0.9987), so the live decision is
    genuinely unchanged regardless of which input is used. NEW callers
    that pass the true WF series get the redesigned, tougher evaluation.

    Parameters
    ----------
    v1_locked:
        Locked-reference block (accepted for the audit contract; not
        read by the gate body).
    v2_result:
        Canonical v2 block. Carries the legacy proxy fields
        (``psr_walkforward`` dict from compute_psr, ``windows_positive_pct``
        in 0..100) used ONLY when no WF series is supplied.
    deployed:
        Severity tag. A floor breach on a deployed strategy returns
        "HALT_AND_SURFACE"; on a non-deployed candidate it returns
        "SHELF". (No longer gates WHETHER the floors are evaluated.)
    psr_floor:
        Minimum acceptable WF-PSR. Kept at the literal 0.90 (matches the
        plan + backcompat_baselines.json phase2_psr_floor). NOTE: 0.90 is
        meaningful ONLY because n changed. On the n=5 window-series the
        PSR sr_se is ~0.31, so 0.90 was cleared by any 5/5 run (pure
        N-deflation); on the n~=25 quarterly series the standard error
        tightens enough that 0.90 is a genuine bar. We do NOT tighten to
        0.95 — the redesign already adds teeth via the corrected WF
        positive-rate, and the deployed v1 clears 0.95 anyway (0.9987).
    wf_pos_floor:
        Minimum acceptable WF positive-quarter rate as a fraction
        (default 0.70 from plan).
    wf_quarterly_returns:
        TRUE walk-forward per-quarter return series (e.g. the
        per_window[].return_pct array from the WF JSON). When supplied,
        this is the PRIMARY input and overrides the legacy proxy.
    wf_result:
        Alternative carrier for the WF series — a Mapping with a
        ``per_window`` list of ``{"return_pct": ...}`` (the WF JSON
        shape). Used only if ``wf_quarterly_returns`` is None.
    """
    # --- Resolve the true walk-forward quarterly series, if supplied. ----
    series: list[float] | None = None
    if wf_quarterly_returns is not None:
        series = [float(x) for x in wf_quarterly_returns]
    elif wf_result is not None:
        per_window = wf_result.get("per_window") or wf_result.get("windows") or []
        series = [
            float(e["return_pct"])
            for e in per_window
            if e.get("return_pct") is not None
        ] or None

    if series is not None:
        # REDESIGNED PATH — recompute both gate inputs from the true WF
        # quarterly series (n ~= 25), not the optimistic 5-OOS proxy.
        n_q = len(series)
        if n_q < WF_MIN_QUARTERS:
            # Refuse to pass a thin series silently (guard (D)).
            return "insufficient_wf_evidence"
        arr = np.asarray(series, dtype=float)
        wf_pos_pct = float((arr > 0).mean())
        psr = float(
            compute_psr(arr, sr_hurdle=0.0, confidence=0.95)["psr_vs_hurdle"]
        )
    else:
        # BACKCOMPAT PATH — no WF series supplied (the deployed runner's
        # existing call). Read the legacy proxy fields exactly as the old
        # gate did, so the DEPLOYED decision stays byte-identical.
        psr_block = v2_result.get("psr_walkforward", {})
        psr = float(psr_block.get("psr_vs_hurdle", 0.0) or 0.0)
        wf_pos_pct = float(v2_result.get("windows_positive_pct", 0.0) or 0.0) / 100.0

    # --- Floors now evaluated for ANY strategy (scope decision B). -------
    breach = (psr < psr_floor) or (wf_pos_pct < wf_pos_floor)
    if breach:
        return "HALT_AND_SURFACE" if deployed else "SHELF"
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


# ---------------------------------------------------------------------------
# True weighted-equity-curve portfolio PSR (methodology debt #2)
# ---------------------------------------------------------------------------
# MOVED here verbatim from tools/portfolio_psr.py as part of the canonical-PSR
# dedup. tools/portfolio_psr.py is now a thin re-export shim pointing here.
# Bodies are character-for-character identical to the pre-merge originals so
# the dedup is provably behavior-preserving (see
# tests/test_unified_psr_equivalence.py).
#
# Sum-then-diff pipeline:
#     1. Superimpose normalized leg-equity curves on a union DatetimeIndex
#        (ffill gaps; default 1.0 before the first tick on a leg = capital
#        sits in cash before the first fill).
#     2. Weight-sum into one portfolio equity series per window.
#     3. Resample to '1D' (mitigates intraday autocorrelation -- documented
#        caveat, not eliminated), pct_change, drop leading NaN.
#     4. Concatenate the per-window return arrays AFTER differencing
#        (NEVER diff across a window boundary -- that would inject a
#        spurious return at the gap).
#     5. Feed the concatenated array to ``compute_psr``.
#
# Why this absorbs correlation: Var(w_b*r_b + w_s*r_s) carries
# 2*w_b*w_s*Cov(r_b, r_s); the stitched per-trade union destroys that cross
# term and inflates N by ~sqrt(2) via sqrt(n-1) in psr_eval.compute_psr.
#
# Sharpe units: per-period (default daily), NOT per-trade. Reviewers must NOT
# compare the new ``point_sharpe_period`` to historical trade-level Sharpes
# from other strategies in this repo.


def _normalized_equity(eq: "pd.Series | None") -> "pd.Series | None":
    """Normalize a leg equity series to start=1.0; return None on bad input."""
    if eq is None or len(eq) == 0:
        return None
    e0 = float(eq.iloc[0])
    if e0 <= 0 or not np.isfinite(e0):
        return None
    return eq / e0


def build_portfolio_equity_curve(
    btc_eq: "pd.Series | None",
    sol_eq: "pd.Series | None",
    w_btc: float,
    w_sol: float,
) -> "pd.Series | None":
    """Time-aligned weighted-sum portfolio equity for ONE window.

    Union DatetimeIndex, ffill gaps, default 1.0 before the first tick
    on a leg (capital sits in cash before the first fill). The resulting
    series carries the BTC<->SOL co-movement structure into its own vol.
    """
    btc_n = _normalized_equity(btc_eq)
    sol_n = _normalized_equity(sol_eq)
    if btc_n is None and sol_n is None:
        return None
    if btc_n is not None and sol_n is not None:
        idx = btc_n.index.union(sol_n.index).sort_values()
        b = btc_n.reindex(idx).ffill().fillna(1.0)
        s = sol_n.reindex(idx).ffill().fillna(1.0)
    elif btc_n is not None:
        b = btc_n
        s = pd.Series(1.0, index=btc_n.index)
    else:
        s = sol_n  # type: ignore[assignment]
        b = pd.Series(1.0, index=sol_n.index)  # type: ignore[union-attr]
    return w_btc * b + w_sol * s


def equity_to_period_returns(
    port_eq: "pd.Series | None",
    resample_period: "str | None" = "1D",
) -> np.ndarray:
    """Per-window equity -> period returns (default daily).

    NEVER diff across a window boundary. Caller concatenates the arrays.
    Pass ``None`` for ``resample_period`` to skip resampling (returns at
    native frequency; will inflate N and bias PSR upward under any residual
    autocorrelation).
    """
    if port_eq is None or len(port_eq) < 2:
        return np.asarray([], dtype=float)
    if resample_period:
        eq = port_eq.resample(resample_period).last().dropna()
    else:
        eq = port_eq.dropna()
    if len(eq) < 2:
        return np.asarray([], dtype=float)
    return eq.pct_change().dropna().values.astype(float)


def aggregate_portfolio_psr(
    per_window_eq: "dict[str, pd.Series | None]",
    resample_period: str = "1D",
    sr_hurdle: float = 0.0,
    confidence: float = 0.95,
) -> dict:
    """Headline portfolio PSR across non-contiguous windows.

    N becomes the count of PERIOD RETURNS (e.g. ~1080 daily obs across 6
    H1 windows), NOT the trade count. Per-window: equity -> resample ->
    pct_change. Cross-window: concatenate post-differencing -- the gap
    between windows never produces a return observation.
    """
    pieces: list[np.ndarray] = []
    per_window_n: dict[str, int] = {}
    for label, eq in per_window_eq.items():
        r = equity_to_period_returns(eq, resample_period=resample_period)
        per_window_n[label] = int(len(r))
        if len(r) > 0:
            pieces.append(r)
    if not pieces:
        return {
            "psr_equity_curve":     None,
            "psr_interpretation":   "insufficient_evidence",
            "point_sharpe_period":  None,
            "n_periods_total":      0,
            "per_window_n_periods": per_window_n,
            "resample_period":      resample_period,
            "sharpe_units":         "per_period (NOT per-trade)",
        }
    combined = np.concatenate(pieces)
    psr = compute_psr(combined, sr_hurdle=sr_hurdle, confidence=confidence)
    return {
        "psr_equity_curve":     psr.get("psr_vs_hurdle"),
        "psr_interpretation":   psr.get("interpretation"),
        "point_sharpe_period":  psr.get("point_sharpe"),
        "n_periods_total":      int(len(combined)),
        "per_window_n_periods": per_window_n,
        "resample_period":      resample_period,
        "sharpe_units":         (
            "per_period (NOT per-trade -- do not compare to historical "
            "trade-level Sharpe)"
        ),
    }


# ---------------------------------------------------------------------------
# Unified PSR dispatcher (thin sugar — additive; runner migration deferred)
# ---------------------------------------------------------------------------

def aggregate_psr(
    per_window: list[Mapping[str, Any]] | None = None,
    *,
    portfolio_weights: Mapping[str, float] | None = None,
    per_window_eq: "Mapping[str, pd.Series | None] | None" = None,
    aggregation_method: str = AGGREGATION_VERSION,
    resample_period: str = "1D",
    sr_hurdle: float = 0.0,
    confidence: float = 0.95,
) -> dict:
    """Single PSR entrypoint. portfolio_weights is None -> 5-OOS / WF
    path (delegates to aggregate_windows on per_window return-dicts).
    portfolio_weights given -> portfolio path (delegates to
    aggregate_portfolio_psr on per_window_eq {label: equity Series}).

    Delegation is LITERAL (no fusion of the two input shapes) so behavior is
    byte-identical to calling the two underlying public functions directly:

      - portfolio_weights is None     -> aggregate_windows(per_window, ...)
      - portfolio_weights is not None -> aggregate_portfolio_psr(per_window_eq, ...)

    The weights themselves are applied UPSTREAM in
    ``build_portfolio_equity_curve`` when the caller builds ``per_window_eq``;
    here ``portfolio_weights`` is the explicit branch selector and is
    validated to be 2 weights summing to ~1.0.
    """
    if portfolio_weights is None:
        if per_window is None:
            raise ValueError(
                "aggregate_psr: per_window is required when portfolio_weights "
                "is None (5-OOS / walk-forward path)."
            )
        return aggregate_windows(per_window, aggregation_method=aggregation_method)

    # Portfolio branch.
    if per_window_eq is None:
        raise ValueError(
            "aggregate_psr: per_window_eq is required when portfolio_weights "
            "is given (portfolio path)."
        )
    if len(portfolio_weights) != 2:
        raise ValueError(
            f"aggregate_psr: portfolio_weights must have exactly 2 weights, "
            f"got {len(portfolio_weights)}."
        )
    wsum = float(sum(portfolio_weights.values()))
    if abs(wsum - 1.0) > 1e-6:
        raise ValueError(
            f"aggregate_psr: portfolio_weights must sum to ~1.0, got {wsum}."
        )
    return aggregate_portfolio_psr(
        dict(per_window_eq),
        resample_period=resample_period,
        sr_hurdle=sr_hurdle,
        confidence=confidence,
    )
