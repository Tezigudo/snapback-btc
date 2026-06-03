"""Runner-level smoke test for the canonical-PSR migration corpus.

This is the test that residual methodology-debt #4 (per
``METHODOLOGY_JUDGMENT_CALLS_VERDICT.md``) asked for: a single pytest that
walks **every discovered runner report JSON** under ``reports/`` and asserts the
canonical-PSR invariants hold bit-for-bit. The per-runner round-trip asserts
embedded in the runners only fire when a runner is manually re-run; this test
fires under ``pytest`` over the persisted corpus.

Run from repo root::

    .venv/bin/python -m pytest tools/tests/test_runner_smoke.py -q

------------------------------------------------------------------------------
Two-source discovery (the load-bearing design choice)
------------------------------------------------------------------------------
Discovery is deliberately TWO-SOURCE so invariant (a) cannot be
green-by-construction:

  HAS-BLOCK SET (data-driven) — every ``reports/*.json`` that is dict-typed and
      already carries a ``canonical`` block (direct or nested per-arm). Drives
      (b)-headline, (b)-secondary, (c)-Lo, and the decision-provenance footgun
      scan. This is the "what is migrated on disk" view.

  REGISTRY EXPECTED-SET (contract-driven) — a hand-maintained REGISTRY of every
      ``build_canonical_block`` SINGLE-report importer in ``tools/*.py`` and the
      ``reports/<name>.json`` it CONTRACTUALLY must emit, cross-checked against a
      live importer grep (the META-ASSERT, so the registry cannot silently rot).
      This is the ONLY set that can fire on a MISSING / partial block. If (a)
      discovered off "reports that already have a block", a build_canonical_block
      importer whose JSON predates the migration (``canonical: null``) would be
      invisible -> a false green. So (a) is driven off the contract, not the data.

------------------------------------------------------------------------------
The HARD invariants (these define ``smoke_green``)
------------------------------------------------------------------------------
  (a) DUAL-EMIT COMPLETENESS — for every runner in the REGISTRY expected-set,
      its report JSON exists AND carries the FULL canonical block
      (aggregation_method, aggregation_version, n_windows, n_trades_sum,
      compounded_pct, windows_positive, windows_positive_pct,
      per_window_return_pct, psr_per_window, psr_walkforward,
      legacy_compounded_pct, legacy_psr_stitched, legacy_delta_pp), with
      aggregation_method in ALLOWED_METHODS and the stitched sibling
      self-labelled with the ``deprecation`` flag. A registry runner whose JSON
      is missing/partial (``canonical: null`` from a pre-migration write) is an
      (a) FAILURE. Known-stale ones (runner migrated in code, JSON never re-run)
      are quarantined via ``xfail(strict=False)`` with a loud reason + enumerated
      in the straggler inventory; a stale report OUTSIDE the allowlist hard-REDs.

  (b) HEADLINE == CANONICAL RECOMPUTE  *** the load-bearing footgun check ***
      The report's ADVERTISED headline PSR (top-level ``report['psr']
      ['psr_vs_hurdle']`` — the value the runner also prints to stdout) MUST
      equal an INDEPENDENT ``compute_psr(canonical['per_window_return_pct'],
      sr_hurdle=0, confidence=0.95, contiguous=False)['psr_vs_hurdle']`` within
      1e-6. This catches the runner that COMPUTES a canonical (N-deflated,
      window-series) PSR but lets a STITCHED (N-inflated, per-trade-union) PSR
      leak into the headline. Realized on exactly two reports today
      (``postfrac_mf_4h_btc`` deployed/byte-frozen, ``postfrac_adx_dr`` shelved):
      both xfail with a tracked reason. A THIRD divergence is a real finding and
      hard-REDs.

  (b-secondary) INTEGRITY — ``psr_walkforward['psr_vs_hurdle']`` ==
      ``compute_psr(per_window_return_pct, contiguous=False)``. This is
      GREEN-BY-CONSTRUCTION (aggregate_windows builds psr_walkforward from that
      same array) and is NOT (b); kept only as a tamper-detector on the
      persisted block. Labelled clearly so it is never mistaken for the headline
      check.

  (c) LO ADDITIVE FIELDS — the additive Lo (2002) keys (``psr_lo_adjusted``,
      ``sr_lo_adjusted``, ``lo_eta``) present on the PERSISTED psr_walkforward.
      Asserted on the persisted block (the strong guard: catches a runner
      emitting a non-Lo-aware PSR), NOT on a fresh recompute (recompute is always
      green because current compute_psr emits them on every path). All current
      migrated reports carry them; a report that previously did and later drops
      them is an additive-contract regression.

------------------------------------------------------------------------------
The FOOTGUN (flagged, mostly not hard-failed — known/stale, not regressions)
------------------------------------------------------------------------------
  The footgun = a runner that lets a STITCHED PSR leak into the advertised
  headline (invariant b) or into the promote/clear DECISION. The decision-leak
  scan (test_footgun_decision_provenance) looks for any PSR numeric inside a
  decision/verdict/gates field that equals ``legacy_psr_stitched`` and differs
  from ``psr_walkforward``. The kc_squeeze verdict leak the spec named was FIXED
  by the f995dfe migration (``gates.psr_basis == "canonical_psr_walkforward"``,
  ``gates.psr_actual == psr_walkforward``, stitched parked in
  ``gates.psr_stitched_legacy``); the scan confirms no decision currently reads a
  stitched-only PSR. A new decision-leak hard-REDs unless allowlisted.

  NAMED LIMITATION: this test catches the STITCHED-vs-canonical PSR axis only.
  It CANNOT catch a 5-OOS report whose windows_positive_pct=100% is used as an
  optimistic proxy for a true walk-forward positive-rate (~55%) — a 5-OOS JSON
  has no quarterly WF series to recompute against. That proxy-optimism is the
  phase2_gate() redesign's job for NEW callers; this smoke test is the
  structural guardrail for the stitched axis.

Stragglers and footguns are ENUMERATED (test_stragglers_enumerated), not asserted
away. ``smoke_green`` is true iff the hard invariants pass on the clean migrated
corpus (xfails do not break green). An anti-vacuous guard asserts the corpus is
non-empty so a glob/path bug cannot make green trivially true.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aggregate import ALLOWED_METHODS  # noqa: E402
from tools.psr_eval import compute_psr  # noqa: E402

REPORTS_DIR = ROOT / "reports"
TOOLS_DIR = ROOT / "tools"

# Bit-for-bit tolerance: persisted PSR is rounded to 6dp by compute_psr and the
# per_window_return_pct series is rounded to 4dp before PSR, so an independent
# recompute agrees to < 1e-6 (verified across the full migrated corpus).
PSR_TOL = 1e-6

# Walk-forward / stitched PSR is computed contiguous=False (disjoint OOS
# windows -> Lo no-op). The independent recompute MUST match that flag.
WF_CONTIGUOUS = False

# Lo (2002) additive sibling keys expected on a current-code PSR emit.
LO_KEYS = ("psr_lo_adjusted", "sr_lo_adjusted", "lo_eta")

# The full canonical (dual-emit) block contract — every field aggregate_windows
# + build_canonical_block stamps. Used by invariant (a).
FULL_BLOCK_KEYS = (
    "aggregation_method",
    "aggregation_version",
    "n_windows",
    "n_trades_sum",
    "compounded_pct",
    "windows_positive",
    "windows_positive_pct",
    "per_window_return_pct",
    "psr_per_window",
    "psr_walkforward",
    "legacy_compounded_pct",
    "legacy_psr_stitched",
    "legacy_delta_pp",
)

# The self-labelling flag the stitched sibling must carry.
STITCHED_DEPRECATION_FLAG = "stitched_per_trade_pl_pct_psr_is_N_inflated"

# Fields that mark a report as recording a promote/clear DECISION.
DECISION_KEYS = ("verdict", "gates", "disposition", "promote", "decision",
                 "phase2_gate_decision", "psr_cleared")


# ===========================================================================
# REGISTRY — the (a) expected-set contract.
# ===========================================================================
# Every ``build_canonical_block`` SINGLE-report importer in tools/*.py and the
# report it CONTRACTUALLY must emit, with whether the persisted block currently
# `conforms` (full block on disk). The grep meta-assert below cross-checks that
# this registry lists every live single-report importer, so it cannot silently
# rot. `conforms=False` entries are KNOWN-stale: runner migrated in code, JSON
# predates the migration (``canonical: null``). They are quarantined (xfail) and
# enumerated as stragglers; they do NOT regenerate this run (scope: enumerate,
# do not fix). A stale report OUTSIDE this registry hard-REDs invariant (a).
#
# Shape tag: "direct" = report['canonical'] is one full block;
#            "nested" = report['canonical'] = {arm: full_block, ...}.
#
# Legitimate EXCLUSIONS (build_canonical_block importers NOT in the expected-set,
# each with a stated reason — they emit a nested/array/multi shape, not one
# contractual single canonical block):
#   - _postfrac_donchian_variants_sweep.py  (variant SWEEP, array shape)
#   - run_mf_4h_multi_coin.py                (multi-COIN scan, per-coin blocks)
#   - run_mf_deepening.py                    (multi-arm deepening sweep)
#   - run_mf_stack2_comparison.py            (stack comparison, multi-block)
#   - _divergence_final_verdict.py           (no single reports/<name>.json out_path)
#   - adaptrend_extended_sweep.py            (SWEEP; writes adaptive_trend_extended_psr.json
#                                             with a partial direct block + report-top
#                                             legacy sibling — handled in the has-block set)
REGISTRY: dict[str, dict[str, Any]] = {
    # ---- direct full blocks (migrated, on disk) ----------------------------
    "kc_squeeze_5oos.json":
        {"runner": "_postfrac_kc_squeeze.py", "shape": "direct", "conforms": True},
    "postfrac_adaptrendV1_imp_funding_skip.json":
        {"runner": "_postfrac_adaptrend_v1_funding_skip.py", "shape": "direct", "conforms": True},
    "postfrac_adx_dr.json":
        {"runner": "_postfrac_adx_dr.py", "shape": "direct", "conforms": True},
    "postfrac_funding_extreme.json":
        {"runner": "_postfrac_funding_extreme.py", "shape": "direct", "conforms": True},
    "postfrac_mf_baseline.json":
        {"runner": "_postfrac_mf_baseline.py", "shape": "direct", "conforms": True},
    "postfrac_taker_flow.json":
        {"runner": "_postfrac_taker_flow.py", "shape": "direct", "conforms": True},
    "postfrac_tsmom_intraday.json":
        {"runner": "_postfrac_tsmom_intraday.py", "shape": "direct", "conforms": True},
    "turn_of_candle_15m_oos.json":
        {"runner": "_postfrac_turn_of_candle_15m.py", "shape": "direct", "conforms": True},
    "postfrac_mf_4h_btc.json":
        {"runner": "_postfrac_mf_4h_btc_run.py", "shape": "direct", "conforms": True},
    "postfrac_walkforward_adaptrend_v1_volsize.json":
        {"runner": "_postfrac_walkforward_adaptrend_v1_volsize.py", "shape": "direct", "conforms": True},
    "postfrac_walkforward_adaptrend_v1_rv_band.json":
        {"runner": "_postfrac_wf_adaptrend_v1_rv_band.py", "shape": "direct", "conforms": True},
    # ---- nested per-arm full blocks (migrated, on disk) --------------------
    "adaptrend_v2_imp_funding_skip.json":
        {"runner": "adaptrend_v2_imp_funding_skip.py", "shape": "nested", "conforms": True},
    "adaptrend_v2_imp_half_out_at_1R.json":
        {"runner": "_adaptrend_v2_imp_half_out_run.py", "shape": "nested", "conforms": True},
    "adaptrend_v2_imp_mtf_h1_confirmation.json":
        {"runner": "_adaptrend_v2_imp_mtf_h1_confirmation_run.py", "shape": "nested", "conforms": True},
    "adaptrend_v2_imp_regime_gate_adx.json":
        {"runner": "_adaptrend_v2_imp_regime_gate_adx_run.py", "shape": "nested", "conforms": True},
    "adaptrend_v2_imp_session_volume_filter.json":
        {"runner": "_adaptrend_v2_imp_session_volume_filter_run.py", "shape": "nested", "conforms": True},
    # ---- arm-nested per-arm full blocks (migrated; canonical lives in each arm
    #      sub-dict report[arm]['canonical'], NOT at report top. Arm keys vary:
    #      base/test, base/adx, base/vol, set_5_OOS/set_11_OOS_extended,
    #      base/variant, v2_base/v2_with_improvement, base/treatment. These were
    #      previously MISLABELLED conforms=False ("stale") because discovery only
    #      checked report['canonical']; _canonical_units now finds the arm-nested
    #      layout and test_a validates each arm's block (keys + recompute). -----
    "postfrac_adaptrend_v1.json":
        {"runner": "_postfrac_adaptrend_v1.py", "shape": "nested", "conforms": True},
    "postfrac_adaptrend_v1_adx_gate.json":
        {"runner": "_postfrac_adaptrend_v1_adx.py", "shape": "nested", "conforms": True},
    "postfrac_adaptrend_v1_h1_conf.json":
        {"runner": "_postfrac_adaptrend_v1_h1_conf.py", "shape": "nested", "conforms": True},
    "postfrac_adaptrend_v1_half_out_1r.json":
        {"runner": "_postfrac_adaptrend_v1_half_out.py", "shape": "nested", "conforms": True},
    "postfrac_adaptrend_v1_session_volume.json":
        {"runner": "_postfrac_adaptrend_v1_session_volume.py", "shape": "nested", "conforms": True},
    "postfrac_adaptrend_v1_time_stop.json":
        {"runner": "_postfrac_adaptrend_v1_time_stop.py", "shape": "nested", "conforms": True},
    "postfrac_adaptrend_v1_vol_gate.json":
        {"runner": "_postfrac_adaptrend_v1_vol_gate.py", "shape": "nested", "conforms": True},
    "postfrac_adaptrend_v1_volsize.json":
        {"runner": "_postfrac_adaptrend_v1_volsize.py", "shape": "nested", "conforms": True},
    "adaptrend_v2_imp_vol_scaled_sizing.json":
        {"runner": "adaptrend_v2_imp_vol_scaled_sizing.py", "shape": "nested", "conforms": True},
    "adaptrend_v2_imp_regime_gate_vol.json":
        {"runner": "adaptrend_v2_imp_regime_gate_vol.py", "shape": "nested", "conforms": True},
    "adaptrend_v2_imp_time_stop_24h.json":
        {"runner": "_adaptrend_v2_imp_time_stop_24h.py", "shape": "nested", "conforms": True},
    # ---- GENUINE straggler: runner emits no per-arm canonical in its on-disk
    #      report (sanity script, not a verdict-bearing OOS report). Honest xfail.
    "adaptrend_v2_sanity.json":
        {"runner": "adaptrend_v2_sanity.py", "shape": "direct", "conforms": False},
}

# Allowlist of registry reports whose (a) failure is KNOWN-stale (runner migrated
# in code, JSON never re-run). An (a) failure on any report NOT in this set is a
# genuine regression and hard-REDs. Derived from REGISTRY conforms=False so the
# two can never drift.
KNOWN_STALE_A: frozenset[str] = frozenset(
    name for name, meta in REGISTRY.items() if not meta["conforms"]
)

# Reports whose top-level headline PSR is a known stitched-leak (invariant b),
# quarantined with a tracked reason. A (b) failure outside this set hard-REDs.
#   - postfrac_mf_4h_btc: DEPLOYED, byte-frozen, deferred sign-off (HARD RULE) —
#     headline psr=0.97755 is the stitched value; phase2_gate reads the canonical
#     psr_walkforward (so it PASSES decision-provenance). NOT regenerated.
#   - postfrac_adx_dr: SHELVED; stale stitched headline (psr=2e-06 == stitched vs
#     wf=0.102377). No decision field reads the headline -> cosmetic only.
KNOWN_HEADLINE_LEAK: dict[str, str] = {
    "postfrac_mf_4h_btc.json":
        "DEPLOYED+byte-frozen (deferred sign-off, HARD RULE): top-level psr is "
        "the stitched value; phase2_gate() reads canon['psr_walkforward'] so the "
        "live decision is unaffected. Not regenerated this run.",
    "postfrac_adx_dr.json":
        "SHELVED strategy: stale stitched headline (psr==legacy_psr_stitched, "
        "!=psr_walkforward). No decision field reads the headline -> cosmetic.",
}


# ===========================================================================
# Loaders & shape helpers
# ===========================================================================

def _load(fp: str) -> Any:
    with open(fp, "r") as fh:
        return json.load(fh)


def _all_report_paths() -> list[str]:
    return sorted(glob.glob(str(REPORTS_DIR / "*.json")))


def _is_canonical_block(obj: Any) -> bool:
    """A DIRECT build_canonical_block body has psr_walkforward at top level."""
    return isinstance(obj, dict) and "psr_walkforward" in obj


def _canonical_units(report: dict) -> list[tuple[str, dict]]:
    """Yield (arm_label, canonical_block) for a report.

    DIRECT     : canonical = {..., psr_walkforward, legacy_psr_stitched, ...}
                 -> [("<top>", canonical)]
    NESTED     : canonical = {"base": {...}, "improvement"/"fs"/"treat": {...}}
                 -> one entry per arm that is itself a canonical block.
    ARM-NESTED : NO top-level canonical; each arm sub-dict carries its OWN block
                 report["base"]["canonical"], report["test"]["canonical"], ...
                 (A/B ablation + set_5_OOS/set_11_OOS_extended layouts).
                 Arm key varies (base/test, base/adx, set_5_OOS/..., v2_base/...).
                 -> one entry per arm sub-dict whose ["canonical"] is a block.
    """
    canon = report.get("canonical")
    if isinstance(canon, dict):
        if _is_canonical_block(canon):
            return [("<top>", canon)]
        units = [(arm, blk) for arm, blk in canon.items() if _is_canonical_block(blk)]
        if units:
            return units
    # ARM-NESTED fallback: canonical lives inside each arm sub-dict, not at the
    # report top. Generic over arm-key names (guarded by _is_canonical_block).
    arm_units: list[tuple[str, dict]] = []
    for arm, sub in report.items():
        if isinstance(sub, dict) and _is_canonical_block(sub.get("canonical")):
            arm_units.append((arm, sub["canonical"]))
    return arm_units


def _legacy_sibling(report: dict, arm: str, blk: dict) -> Any:
    """Resolve the deprecation-flagged stitched sibling for a unit.

    Dual-emit has two valid on-disk layouts:
      - in-block  : build_canonical_block folds legacy_psr_stitched INTO the
                    block (the standard migrated shape; nested per-arm).
      - report-top: a sweep runner emits the canonical block without an in-block
                    legacy and writes legacy_psr_stitched at report top-level
                    (adaptive_trend_extended_psr). Only valid for the DIRECT
                    "<top>" arm — a nested arm cannot borrow a single report-level
                    legacy.
    """
    if isinstance(blk.get("legacy_psr_stitched"), dict):
        return blk["legacy_psr_stitched"]
    if arm == "<top>" and isinstance(report.get("legacy_psr_stitched"), dict):
        return report["legacy_psr_stitched"]
    return None


# ===========================================================================
# HAS-BLOCK discovery (data-driven) — drives (b), (b-secondary), (c), footgun.
# ===========================================================================

def _discover_has_block() -> tuple[list[str], list[str], list[str]]:
    """Return (has_block_paths, decision_paths, skipped_list_typed)."""
    has_block: list[str] = []
    decision: list[str] = []
    skipped: list[str] = []
    for fp in _all_report_paths():
        try:
            d = _load(fp)
        except (json.JSONDecodeError, OSError):
            skipped.append(f"{os.path.basename(fp)} (unreadable)")
            continue
        if not isinstance(d, dict):
            # List-shaped sweeps/scans are not single-canonical runner reports.
            skipped.append(f"{os.path.basename(fp)} (top-level {type(d).__name__})")
            continue
        if _canonical_units(d):
            has_block.append(fp)
        if any(k in d for k in DECISION_KEYS):
            decision.append(fp)
    return has_block, decision, skipped


HAS_BLOCK_REPORTS, DECISION_REPORTS, SKIPPED_LIST_TYPED = _discover_has_block()

# Anti-vacuous floor: discovery now finds 33 has-block reports (11 direct +
# 5 top-nested + 11 arm-nested A/B + adaptive_trend sweep/WF reports). Floor 30
# leaves margin while still catching a broken glob/path (vacuous green).
MIN_HAS_BLOCK_REPORTS = 30


# ===========================================================================
# REGISTRY grep meta-assert (drift-guard) — keeps REGISTRY honest.
# ===========================================================================

def _imports_build_canonical_block(src: str) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "aggregate" in node.module:
            if any(a.name == "build_canonical_block" for a in node.names):
                return True
    return False


def _live_bcb_importers() -> set[str]:
    """Basenames of tools/*.py that import build_canonical_block from tools.aggregate."""
    out: set[str] = set()
    for fp in sorted(glob.glob(str(TOOLS_DIR / "*.py"))):
        if os.path.basename(fp) == "aggregate.py":
            continue  # the module itself defines it, not an importer
        try:
            src = open(fp).read()
        except OSError:
            continue
        if _imports_build_canonical_block(src):
            out.add(os.path.basename(fp))
    return out


# Runners that import build_canonical_block but are LEGITIMATELY excluded from the
# expected-set (sweeps / multi-result / no single report out_path), each reason
# stated in the REGISTRY docstring above. The meta-assert tolerates exactly these.
REGISTRY_EXCLUDED_RUNNERS: frozenset[str] = frozenset({
    "_postfrac_donchian_variants_sweep.py",
    "run_mf_4h_multi_coin.py",
    "run_mf_deepening.py",
    "run_mf_stack2_comparison.py",
    "_divergence_final_verdict.py",
    "adaptrend_extended_sweep.py",
    # rv_band OOS family: ONE runner that emits a per-RV_TAG report family
    # (reports/postfrac_adaptrend_v1_rv_band_<RV_TAG>.json), not one contractual
    # single canonical report. The variants are the known-deferred Mode-2
    # footgun (verdict reads a PSR with no canonical block on disk) — enumerated
    # in the straggler inventory + allowlisted in KNOWN_DECISION_LEAK, not
    # asserted as a single (a) registry entry.
    "_postfrac_adaptrend_v1_rv_band.py",
})


def test_registry_meta_assert_no_drift():
    """Every live build_canonical_block importer is either in the REGISTRY
    (expected-set) or in the stated excluded set. Guards against a NEW runner
    being added without a corresponding (a) registry entry (which would let its
    missing/partial report slip through invariant (a) unchecked).
    """
    live = _live_bcb_importers()
    registry_runners = {meta["runner"] for meta in REGISTRY.values()}
    accounted = registry_runners | REGISTRY_EXCLUDED_RUNNERS
    unaccounted = sorted(live - accounted)
    assert not unaccounted, (
        "build_canonical_block importer(s) neither in the (a) REGISTRY nor in "
        "the stated excluded set — the registry has drifted. Add each to "
        "REGISTRY (with its reports/<name>.json + conforms flag) or to "
        f"REGISTRY_EXCLUDED_RUNNERS with a reason: {unaccounted}"
    )
    # And the reverse: every registry runner must actually still import bcb.
    stale_registry = sorted(registry_runners - live)
    assert not stale_registry, (
        "REGISTRY lists runner(s) that no longer import build_canonical_block "
        f"(dead registry entries): {stale_registry}"
    )


# ===========================================================================
# Anti-vacuous guard — a glob/path bug must NOT yield a trivially green run.
# ===========================================================================

def test_corpus_non_empty_anti_vacuous():
    assert REPORTS_DIR.is_dir(), f"reports dir missing: {REPORTS_DIR}"
    assert len(HAS_BLOCK_REPORTS) >= MIN_HAS_BLOCK_REPORTS, (
        f"Discovered only {len(HAS_BLOCK_REPORTS)} has-block reports; expected "
        f">= {MIN_HAS_BLOCK_REPORTS}. Either the glob/path is broken (vacuous "
        f"green) or the migrated corpus shrank. Found: "
        f"{[os.path.basename(p) for p in HAS_BLOCK_REPORTS]}"
    )
    assert REGISTRY, "REGISTRY is empty — (a) would assert nothing."


# ===========================================================================
# (a) DUAL-EMIT COMPLETENESS — driven off the REGISTRY contract.
# ===========================================================================

def _full_block_missing_keys(blk: dict) -> list[str]:
    return [k for k in FULL_BLOCK_KEYS if k not in blk]


_A_PARAMS = sorted(REGISTRY.keys())


@pytest.mark.parametrize("report_name", _A_PARAMS, ids=_A_PARAMS)
def test_a_dual_emit_completeness(report_name, request):
    meta = REGISTRY[report_name]
    # Known-stale (runner migrated, JSON not re-run) -> loud xfail, not silent.
    if report_name in KNOWN_STALE_A:
        request.applymarker(pytest.mark.xfail(
            reason=(
                f"(a) STRAGGLER: {report_name} — runner {meta['runner']} is "
                f"migrated in code (writes report['canonical']=build_canonical_"
                f"block(...)), but the on-disk JSON predates the f995dfe "
                f"migration (canonical: null). Fix = re-run the runner. Scope "
                f"this run: ENUMERATE, do not regenerate."
            ),
            strict=False,
        ))
    fp = REPORTS_DIR / report_name
    assert fp.exists(), (
        f"(a) {report_name}: registry runner {meta['runner']} contractually "
        f"must emit this report but it is MISSING from reports/."
    )
    report = _load(str(fp))
    units = _canonical_units(report)
    assert units, (
        f"(a) {report_name}: report['canonical'] is absent/null/not-a-block "
        f"(a build_canonical_block importer must emit the full block)."
    )
    expected_shape = meta["shape"]
    if expected_shape == "direct":
        assert len(units) == 1 and units[0][0] == "<top>", (
            f"(a) {report_name}: registry expects a DIRECT block, got "
            f"{[u[0] for u in units]}"
        )
    else:  # nested
        assert all(arm != "<top>" for arm, _ in units) and len(units) >= 2, (
            f"(a) {report_name}: registry expects a NESTED per-arm block, got "
            f"{[u[0] for u in units]}"
        )
    for arm, blk in units:
        where = f"{report_name}[{arm}]"
        missing = _full_block_missing_keys(blk)
        assert not missing, (
            f"(a) {where}: canonical block missing required key(s) {missing} "
            f"(dual-emit completeness)."
        )
        assert blk["aggregation_method"] in ALLOWED_METHODS, (
            f"(a) {where}: aggregation_method={blk['aggregation_method']!r} not "
            f"in ALLOWED_METHODS {ALLOWED_METHODS}."
        )
        legacy = blk.get("legacy_psr_stitched")
        assert isinstance(legacy, dict), (
            f"(a) {where}: legacy_psr_stitched missing/not-a-dict."
        )
        assert legacy.get("deprecation") == STITCHED_DEPRECATION_FLAG, (
            f"(a) {where}: legacy_psr_stitched not self-labelled with the "
            f"deprecation flag {STITCHED_DEPRECATION_FLAG!r} (got "
            f"{legacy.get('deprecation')!r})."
        )
        # Cheap integrity extras (non-load-bearing): compounded_pct reproduces
        # from per_window_return_pct, and windows_positive string == n_pos/n.
        pw = np.asarray(blk["per_window_return_pct"], dtype=float)
        recomp_comp = (np.prod(1.0 + pw / 100.0) - 1.0) * 100.0
        assert abs(recomp_comp - float(blk["compounded_pct"])) < 1e-2, (
            f"(a) {where}: compounded_pct={blk['compounded_pct']} != recompute "
            f"from per_window_return_pct ({recomp_comp:.4f})."
        )
        n_pos = int((pw > 0).sum())
        assert blk["windows_positive"] == f"{n_pos}/{len(pw)}", (
            f"(a) {where}: windows_positive={blk['windows_positive']} != "
            f"{n_pos}/{len(pw)} from per_window_return_pct."
        )


# ===========================================================================
# (b) HEADLINE == CANONICAL RECOMPUTE  *** load-bearing footgun check ***
# ===========================================================================

def _has_numeric_headline(report: dict) -> bool:
    psr = report.get("psr")
    if isinstance(psr, dict):
        return isinstance(psr.get("psr_vs_hurdle"), (int, float))
    return isinstance(psr, (int, float))


def _headline_psr(report: dict) -> float:
    psr = report.get("psr")
    if isinstance(psr, dict):
        return float(psr["psr_vs_hurdle"])
    return float(psr)


# Headline check applies to DIRECT-block reports carrying a numeric top-level
# `psr`. Nested per-arm reports have no single headline; None-headline reports
# (adaptrendV1_imp_funding_skip, adaptive_trend_extended_psr) cannot misadvertise
# an absent value -> both are N/A (skipped, not failed).
def _b_headline_params() -> list[str]:
    out: list[str] = []
    for fp in HAS_BLOCK_REPORTS:
        report = _load(fp)
        units = _canonical_units(report)
        if len(units) == 1 and units[0][0] == "<top>" and _has_numeric_headline(report):
            out.append(fp)
    return out


_B_HEADLINE_PARAMS = _b_headline_params()


@pytest.mark.parametrize(
    "fp", _B_HEADLINE_PARAMS,
    ids=[os.path.basename(p) for p in _B_HEADLINE_PARAMS],
)
def test_b_headline_equals_canonical_recompute(fp, request):
    name = os.path.basename(fp)
    if name in KNOWN_HEADLINE_LEAK:
        request.applymarker(pytest.mark.xfail(
            reason=f"(b) HEADLINE LEAK [{name}]: {KNOWN_HEADLINE_LEAK[name]}",
            strict=False,
        ))
    report = _load(fp)
    (_, blk), = _canonical_units(report)
    pw = blk.get("per_window_return_pct")
    assert pw, f"(b) {name}: no per_window_return_pct to recompute from."
    headline = _headline_psr(report)
    recomputed = compute_psr(
        np.asarray(pw, dtype=float),
        sr_hurdle=0.0,
        confidence=0.95,
        contiguous=WF_CONTIGUOUS,
    )["psr_vs_hurdle"]
    assert abs(headline - float(recomputed)) < PSR_TOL, (
        f"(b) {name}: ADVERTISED headline psr.psr_vs_hurdle={headline} does NOT "
        f"match independent compute_psr(canonical.per_window_return_pct, "
        f"contiguous={WF_CONTIGUOUS})={recomputed} within {PSR_TOL}. The headline "
        f"is a STITCHED (N-inflated) PSR leaking past the canonical window-series "
        f"PSR. If this is a NEW report, it is a genuine footgun — do NOT xfail; "
        f"fix the runner to advertise canon['psr_walkforward']."
    )


# ===========================================================================
# (b-secondary) INTEGRITY — psr_walkforward == recompute. GREEN-BY-CONSTRUCTION.
#   NOT the headline check; kept only as a tamper-detector on the persisted block.
# ===========================================================================

@pytest.mark.parametrize(
    "fp", HAS_BLOCK_REPORTS,
    ids=[os.path.basename(p) for p in HAS_BLOCK_REPORTS],
)
def test_b_secondary_integrity_psr_walkforward(fp):
    report = _load(fp)
    for arm, blk in _canonical_units(report):
        where = f"{os.path.basename(fp)}[{arm}]"
        pw = blk.get("per_window_return_pct")
        assert pw, f"(b-sec) {where}: no per_window_return_pct."
        persisted = blk["psr_walkforward"].get("psr_vs_hurdle")
        assert persisted is not None, f"(b-sec) {where}: persisted psr is None."
        recomputed = compute_psr(
            np.asarray(pw, dtype=float),
            sr_hurdle=0.0,
            confidence=0.95,
            contiguous=WF_CONTIGUOUS,
        )["psr_vs_hurdle"]
        assert abs(float(persisted) - float(recomputed)) < PSR_TOL, (
            f"(b-sec) {where}: persisted psr_walkforward={persisted} != recompute "
            f"{recomputed} (tamper/staleness on the persisted block)."
        )


# ===========================================================================
# (c) LO ADDITIVE FIELDS present on the PERSISTED psr_walkforward.
# ===========================================================================

def _lo_fields_present(blk: dict) -> bool:
    wf = blk.get("psr_walkforward", {})
    return all(k in wf for k in LO_KEYS)


_LO_BEARING_UNITS = [
    (fp, arm)
    for fp in HAS_BLOCK_REPORTS
    for arm, blk in _canonical_units(_load(fp))
    if _lo_fields_present(blk)
]


@pytest.mark.parametrize(
    "fp,arm", _LO_BEARING_UNITS,
    ids=[f"{os.path.basename(fp)}[{arm}]" for fp, arm in _LO_BEARING_UNITS],
)
def test_c_lo_fields_present(fp, arm):
    report = _load(fp)
    blk = dict(_canonical_units(report))[arm]
    wf = blk["psr_walkforward"]
    for k in LO_KEYS:
        assert k in wf, (
            f"(c) {os.path.basename(fp)}[{arm}]: Lo field {k!r} dropped from a "
            f"report that previously carried it — additive-contract regression."
        )
    # Under contiguous=False (disjoint windows) the Lo correction is a documented
    # no-op -> lo_eta must be exactly 1.0 (psr_eval Trap 2).
    assert wf["lo_eta"] == 1.0, (
        f"(c) {os.path.basename(fp)}[{arm}]: lo_eta={wf['lo_eta']} but the "
        f"walk-forward series is contiguous=False (disjoint windows) where Lo "
        f"must be a no-op (lo_eta == 1.0). See psr_eval Trap 2."
    )


def test_c_lo_corpus_non_empty():
    # If this collapses to 0, (c) would silently assert nothing.
    assert len(_LO_BEARING_UNITS) >= 16, (
        f"Only {len(_LO_BEARING_UNITS)} Lo-bearing canonical units found; "
        f"expected >= 16 (the post-Lo-patch corpus). (c) would be vacuous."
    )


# ===========================================================================
# FOOTGUN — DECISION PROVENANCE. The promote/clear decision must NOT read a
# stitched-only PSR. Hard-RED on a new leak; xfail known-deferred.
# ===========================================================================

# Known-deferred decision-leak reports. The kc_squeeze verdict leak the spec
# named was FIXED by f995dfe (gates.psr_basis == "canonical_psr_walkforward"), so
# kc_squeeze is NOT here. What remains are Mode-2 leaks: stale-on-disk reports
# (canonical: null) whose runner is migrated but JSON never re-run — the verdict
# reads a PSR with no canonical block to compare against. These are the SAME
# stragglers (a) flags; they are enumerated, not hard-failed. Resolution =
# regenerate (scope: out of this run). A leak NOT covered here hard-REDs:
#   - the 6 stale ablations (== KNOWN_STALE_A ablation members), and
#   - the rv_band OOS variant family (one deferred runner, many RV_TAG reports).
# Matched by prefix so every RV_TAG variant is covered without listing each.
KNOWN_DECISION_LEAK_PREFIXES: tuple[str, ...] = (
    "postfrac_adaptrend_v1_rv_band_",   # rv_band OOS variant family (1 runner)
)


def _is_known_deferred_decision_leak(report_name: str) -> bool:
    """A decision leak is known-deferred iff its report is a KNOWN-stale (a)
    registry straggler (canonical: null, runner migrated) or an rv_band variant.
    Both are pre-regeneration artifacts, not live wrong-PSR-gated verdicts.
    """
    if report_name in KNOWN_STALE_A:
        return True
    return any(report_name.startswith(p) for p in KNOWN_DECISION_LEAK_PREFIXES)


def _decision_psr_value(report: dict) -> tuple[str, float] | None:
    """Return (field_path, value) of the PSR a decision READS, if any.

    Recognizes the on-disk decision shapes:
      - gates.psr_actual                 (5-OOS gate runner: kc_squeeze)
      - verdict.<cand>_psr*              (A/B ablation runners) — gated on a
                                         `psr_not_worse` boolean; return the
                                         CANDIDATE PSR (non-base_* numeric).
    A value parked in a clearly-labelled legacy/diff field (gates.psr_stitched_
    legacy, *_legacy, *_diff) is NOT the read value and is ignored. Returns None
    if the decision is not PSR-driven.
    """
    gates = report.get("gates")
    if isinstance(gates, dict) and isinstance(gates.get("psr_actual"), (int, float)):
        return ("gates.psr_actual", float(gates["psr_actual"]))
    verdict = report.get("verdict")
    if isinstance(verdict, dict) and "psr_not_worse" in verdict:
        def _is_candidate_psr(k: str) -> bool:
            kl = k.lower()
            if "psr" not in kl:
                return False
            if kl.startswith("base_"):       # comparison anchor, not the read
                return False
            if kl in ("psr_not_worse",) or kl.startswith("delta_"):
                return False
            if "legacy" in kl or "stitched" in kl or kl.endswith("_diff"):
                return False
            return isinstance(verdict[k], (int, float))
        cands = [k for k in verdict if _is_candidate_psr(k)]
        if cands:
            return (f"verdict.{cands[0]}", float(verdict[cands[0]]))
    return None


def _classify_decision_leak(fp: str, report: dict) -> dict | None:
    """A decision-leak iff the decision PSR equals legacy_psr_stitched and
    differs from psr_walkforward (Mode 1), OR there is no canonical block to
    compare against yet the decision reads a PSR (Mode 2 — stitched by
    construction). Returns the leak record, or None if clean.
    """
    dec = _decision_psr_value(report)
    if dec is None:
        return None
    field_path, dec_psr = dec
    units = _canonical_units(report)
    name = os.path.basename(fp)
    if units:
        _, blk = units[0]
        wf = blk.get("psr_walkforward", {}).get("psr_vs_hurdle")
        legacy = blk.get("legacy_psr_stitched", {}).get("psr_vs_hurdle")
        reads_stitched = legacy is not None and abs(dec_psr - float(legacy)) < PSR_TOL
        reads_canonical = wf is not None and abs(dec_psr - float(wf)) < PSR_TOL
        if reads_stitched and not reads_canonical:
            return {
                "report": name, "mode": "mode1_decision_reads_stitched",
                "decision_field": field_path, "decision_psr": dec_psr,
                "stitched_psr": float(legacy),
                "canonical_psr": (None if wf is None else float(wf)),
            }
        return None
    return {
        "report": name, "mode": "mode2_no_canonical_decision_reads_psr",
        "decision_field": field_path, "decision_psr": dec_psr,
        "stitched_psr": dec_psr, "canonical_psr": None,
    }


def _collect_decision_leaks() -> list[dict]:
    out: list[dict] = []
    for fp in DECISION_REPORTS:
        leak = _classify_decision_leak(fp, _load(fp))
        if leak is not None:
            out.append(leak)
    return out


DECISION_LEAKS = _collect_decision_leaks()


def test_footgun_decision_provenance():
    """Hard-RED on any decision-leak not in KNOWN_DECISION_LEAK.

    This is the mechanized manual-verify-panel catch: a promote/clear verdict
    gated on the N-inflated stitched PSR. The kc_squeeze leak the spec named is
    FIXED (gates.psr_basis == 'canonical_psr_walkforward'); this guards against a
    NEW one re-appearing.
    """
    unexpected = [
        lk for lk in DECISION_LEAKS
        if not _is_known_deferred_decision_leak(lk["report"])
    ]
    assert not unexpected, (
        "Decision-provenance footgun: promote/clear verdict reads a STITCHED-only "
        "PSR (== legacy_psr_stitched, != psr_walkforward) on a report that is "
        "NEITHER a known-stale (a) straggler NOR an rv_band variant. A "
        "wrong-PSR-gated verdict must never pass clean. If genuinely deferred, "
        "add a tracked reason; otherwise fix the runner:\n"
        f"{json.dumps(unexpected, indent=2, default=str)}"
    )


# ===========================================================================
# STRADDLE early-warning (flag-not-fail) — fires even when wiring is currently
# correct: WARN when feeding the gate the stitched PSR WOULD flip its outcome.
# ===========================================================================

class StitchedPsrFootgun(UserWarning):
    """A report where swapping canonical->stitched PSR would change the gate.

    Flagged via warnings.warn so it surfaces in the pytest summary without
    failing. CI can escalate with `-W error::...StitchedPsrFootgun`.
    """


def test_straddle_early_warning(recwarn):
    """For every gated report, recompute the PSR-floor outcome BOTH ways
    (canonical psr_walkforward vs legacy_psr_stitched) and WARN on every report
    where the two would render different psr_cleared verdicts — the latent
    footgun site, before a future edit trips it.
    """
    import warnings
    straddles: list[dict] = []
    for fp in DECISION_REPORTS:
        report = _load(fp)
        gates = report.get("gates")
        if not isinstance(gates, dict):
            continue
        floor = gates.get("psr_min")
        if not isinstance(floor, (int, float)):
            continue
        units = _canonical_units(report)
        if not units:
            continue
        _, blk = units[0]
        wf = blk.get("psr_walkforward", {}).get("psr_vs_hurdle")
        legacy = blk.get("legacy_psr_stitched", {}).get("psr_vs_hurdle")
        if wf is None or legacy is None:
            continue
        cleared_canonical = float(wf) >= float(floor)
        cleared_stitched = float(legacy) >= float(floor)
        if cleared_canonical != cleared_stitched:
            rec = {
                "report": os.path.basename(fp), "psr_min": float(floor),
                "psr_walkforward": float(wf), "legacy_psr_stitched": float(legacy),
                "cleared_canonical": cleared_canonical,
                "cleared_stitched": cleared_stitched,
            }
            straddles.append(rec)
            warnings.warn(
                f"STRADDLE [{rec['report']}]: feeding the gate the stitched PSR "
                f"would FLIP psr_cleared (canonical={cleared_canonical} vs "
                f"stitched={cleared_stitched}; floor={floor}, "
                f"wf={wf}, stitched={legacy}). Latent footgun site.",
                category=StitchedPsrFootgun,
            )
    # Always passes — this is the flag-not-fail early-warning channel.
    assert isinstance(straddles, list)


# ===========================================================================
# STRAGGLER ENUMERATION — visible, non-failing inventory.
# ===========================================================================

def _stale_a_reports() -> list[str]:
    """Registry reports whose on-disk block is missing/partial (the (a) stragglers)."""
    out: list[str] = []
    for name, meta in sorted(REGISTRY.items()):
        fp = REPORTS_DIR / name
        if not fp.exists():
            out.append(f"{name} (MISSING)")
            continue
        units = _canonical_units(_load(str(fp)))
        if not units or any(_full_block_missing_keys(blk) for _, blk in units):
            out.append(name)
    return out


def _stale_lo_reports() -> list[str]:
    out: list[str] = []
    for fp in HAS_BLOCK_REPORTS:
        units = _canonical_units(_load(fp))
        if units and not all(_lo_fields_present(blk) for _, blk in units):
            out.append(os.path.basename(fp))
    return out


def _unflagged_legacy_reports() -> list[str]:
    """Canonical reports whose stitched sibling lacks the N-inflation flag."""
    out: list[str] = []
    for fp in HAS_BLOCK_REPORTS:
        report = _load(fp)
        for arm, blk in _canonical_units(report):
            legacy = _legacy_sibling(report, arm, blk) or {}
            if legacy.get("deprecation") != STITCHED_DEPRECATION_FLAG:
                out.append(os.path.basename(fp))
                break
    return out


def test_stragglers_enumerated(capsys):
    """Always-pass: prints the straggler/footgun inventory under -s.

    Never asserts the lists are empty (they legitimately aren't, pre-regeneration
    of the stale set); its job is to make the residual surface VISIBLE, not gate.
    """
    stale_a = _stale_a_reports()
    stale_lo = _stale_lo_reports()
    unflagged = _unflagged_legacy_reports()
    headline_leaks = sorted(
        os.path.basename(p) for p in _B_HEADLINE_PARAMS
        if os.path.basename(p) in KNOWN_HEADLINE_LEAK
    )
    with capsys.disabled():
        print("\n=== runner smoke: straggler inventory ===")
        print(f"registry runners (a)        : {len(REGISTRY)}")
        print(f"has-block reports checked   : {len(HAS_BLOCK_REPORTS)}")
        print(f"decision reports scanned    : {len(DECISION_REPORTS)}")
        print(f"Lo-bearing units (hard c)   : {len(_LO_BEARING_UNITS)}")
        print(f"(b) headline params         : {len(_B_HEADLINE_PARAMS)}")
        print(f"skipped (list-typed/unread) : {SKIPPED_LIST_TYPED}")
        print(f"(a) STALE registry stragglers: {stale_a}")
        print(f"(b) known headline leaks (xfail): {headline_leaks}")
        print(f"(c) stale pre-Lo writes     : {stale_lo}")
        print(f"unflagged stitched sibling  : {unflagged}")
        dl_deferred = sorted(
            lk["report"] for lk in DECISION_LEAKS
            if _is_known_deferred_decision_leak(lk["report"])
        )
        dl_unexpected = sorted(
            lk["report"] for lk in DECISION_LEAKS
            if not _is_known_deferred_decision_leak(lk["report"])
        )
        print(f"decision leaks/known-deferred: {dl_deferred}")
        print(f"decision leaks/UNEXPECTED(RED): {dl_unexpected}")
    assert isinstance(stale_a, list)
