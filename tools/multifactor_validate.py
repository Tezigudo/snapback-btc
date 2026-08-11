"""multifactor-v1 + 4H EMA200 gate — live/backtest parity validation.

Pattern adapted from tools/hybrid_phase3_validate.py.

Three-stage validation:

(stage 0) PARAMS PROVENANCE — cheap, and the reason the other stages mean anything.
    Build the validated params from `config/params.yaml` — the file the live bot
    actually reads — and DIFF them against `tools/run_mf_deepening.LOCKED`, the
    research-era dict this tool used to validate with. Until 2026-08-10 the tool
    imported LOCKED directly, so it happily certified vol 2.0 / funding 0.0005 /
    risk 2.75 while production ran 1.5 / 0.0015 / 3.5. A green run said nothing
    about the deployed config. LOCKED is now imported for COMPARISON ONLY.

(stage 1) IMPL CROSS-CHECK — fast.
    Compare the new in-class gate (`DayTradeMultiFactorBTC.use_mtf_4h_gate=True`)
    against the existing `MultiFactorMTF4H` subclass in tools/run_mf_deepening.py
    on a single OOS window. They should produce IDENTICAL trade lists. If they
    don't, the in-class implementation is wrong — investigate before going on.

(stage 2) ENTRY PARITY.
    For each 15m bar in a 90-day window, evaluate BOTH paths and compare:
      (a) Backtest: re-run DayTradeMultiFactorBTC over the window, record the
          set of bar timestamps where _long_signal/_short_signal would return
          True at that bar.
      (b) Live: invoke evaluate_signal(bars_15m=window_up_to_bar_i,
                                       funding_rate=bar_i_funding,
                                       params=params,
                                       bars_4h=<injected slice>) bar by bar.
    Compute per-bar agreement % and dump mismatches.

(stage 3) EXIT PARITY — added 2026-08-10, and the whole reason this file changed.
    Stage 2 compares ENTRY SIGNAL BARS ONLY. That is why this tool reported
    "100% parity across 25,702 bars" for months while the live bot was missing an
    exit rule entirely: `DayTradeMultiFactorBTC.next()` closes on an adverse
    EMA(200) cross, and `strategy_uses_trend_exit()` did not list multifactor-v1,
    so live v1 ran on SL/TP/time-stop alone. Cost, per the 2026-08-01
    re-validation: walk-forward 64% vs the 70% gate, OOS 3/5, and a kill-floor
    breach on 0.41% of deploy dates. A passing parity check that cannot see
    exits is worse than no check, because it is read as coverage.

    So this stage compares, bar by bar over the same window, next()'s in-position
    branch against `trend_exit_signal_multifactor_v1` — for a hypothetical long
    AND a hypothetical short at every bar, in two shapes:
      (a) full-prefix  — pure algebraic parity, must be EXACT.
      (b) rolling 1500 — the shape the bot actually calls, whose EMA is seeded at
          the window start rather than at genesis. This is the one that can drift.

Gate:
    Stage 0: no drift between params.yaml and the params under test.
    Stage 1: 100% trade-list equality.
    Stage 2: ≥99.5% signal-bar parity (stricter than 99% used elsewhere since
             this is a LOCKED LIVE strategy).
    Stage 3: 100% full-prefix exit parity, ≥99.5% as-live rolling-window parity.

Output: reports/multifactor_4h_parity_validation.json

Run:
    uv run python tools/multifactor_validate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from backtesting import Backtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.live_multifactor_v1 import (  # noqa: E402
    evaluate_signal,
    trend_exit_signal_multifactor_v1,
)
from strategy.signals_multifactor import (  # noqa: E402
    DayTradeMultiFactorBTC,
    _build_4h_ema_aligned,
)

# PSR-MIGRATION NOTE (methodology debt #1): this validator computes NO PSR.
# It is a trade-list-equality + per-bar signal-parity check whose verdict is an
# agreement-%, not a Sharpe/PSR. It imports only data/config/strategy symbols
# (CASH, COMMISSION, LOCKED, MARGIN, MultiFactorMTF4H, _load_slice) from
# run_mf_deepening — it NEVER calls that module's run_one/aggregate/summarize
# path, so it never reaches the (previously stitched, now already-canonical)
# PSR aggregation there. There is therefore no PSR call site to migrate here;
# the inherited base (run_mf_deepening.summarize) was already migrated to the
# canonical dual-emit. No change to the parity logic and no touch to the live
# path (strategy.live_multifactor_v1 stays a read-only import).
from tools.run_mf_deepening import (  # noqa: E402
    CASH,
    COMMISSION,
    LOCKED,
    MARGIN,
    MultiFactorMTF4H,
    _load_slice,
)

PARITY_GATE = 99.5
EXIT_PARITY_GATE = 100.0        # full-prefix exit parity is algebraic — must be exact
LIVE_FETCH_BARS = 1500          # bot.py: fetch_ohlcv(..., limit=1500)

CONFIG_PATH = ROOT / "config" / "params.yaml"


def _deployed_strategy_params() -> tuple[dict, str | None]:
    """The params the LIVE BOT reads, flattened the way the backtest wants them.

    Returns (params, load_error). `bot.py` loads config/params.yaml wholesale;
    the backtest class takes sizing knobs (`risk_per_trade_pct`, `leverage`) as
    class attributes, but the YAML keeps them under `sizing:`. Flatten
    strategy+sizing into one dict so the validated config is literally the
    deployed one.

    Read as YAML rather than imported from a research module ON PURPOSE — see
    the stage-0 note in the module docstring. Importing a frozen dict is how this
    tool ended up certifying a config that had not been in production since July.

    NEVER RAISES. This runs at module import (DEPLOYED is a module constant), so
    an unreadable or malformed params.yaml would otherwise take down every
    importer — including the test suite — with a bare traceback, before `main()`
    could turn it into the structured FAIL that stage 0 promises. A load failure
    is returned as an error string and becomes that FAIL instead.
    """
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
    except (OSError, yaml.YAMLError) as e:
        return {}, f"{type(e).__name__}: {e}"
    if not isinstance(cfg, dict):
        return {}, f"params.yaml did not parse to a mapping (got {type(cfg).__name__})"
    merged = dict(cfg.get("strategy") or {})
    for k in ("risk_per_trade_pct", "leverage"):
        if k in (cfg.get("sizing") or {}):
            merged[k] = cfg["sizing"][k]
    return merged, None


DEPLOYED, DEPLOYED_LOAD_ERROR = _deployed_strategy_params()

# Locked LIVE params, read from config/params.yaml, PLUS the 4H gate toggle.
LIVE_PARAMS: dict = {
    "symbol": "BTC/USDT:USDT",
    "strategy": {
        **DEPLOYED,
        "use_mtf_4h_gate": True,
        "mtf_4h_ema_period": 200,
    },
}


def _class_kwargs(cls, params: dict) -> tuple[dict, list[str]]:
    """Split `params` into (settable on cls, dropped).

    backtesting.py's `bt.run(**kw)` raises on an attribute the Strategy class
    doesn't define, so params destined for a class must be filtered. Filtering
    SILENTLY is its own bug class — dropping a key that matters makes the run
    quietly test defaults instead (cf. the `from_yaml`-drops-donchian-keys
    gotcha). Hence the dropped list is returned and reported, never swallowed.
    """
    keep, dropped = {}, []
    for k, v in params.items():
        if hasattr(cls, k):
            keep[k] = v
        else:
            dropped.append(k)
    return keep, sorted(dropped)

# 90-day stage-2 window. Use the most-recent slice that's fully cached.
STAGE2_START = "2025-04-01"
STAGE2_END = "2025-06-30"

# Stage-1 cross-check window (shortest OOS slice with reasonable trade count).
STAGE1_START = "2024-01-01"
STAGE1_END = "2024-06-30"

PARQ_15M = ROOT / "data" / "historical" / "BTC_USDT_USDT_15m.parquet"
PARQ_4H = ROOT / "data" / "historical" / "BTC_USDT_USDT_4h.parquet"
OUT = ROOT / "reports" / "multifactor_4h_parity_validation.json"


# ---------------------------------------------------------------------------
# Stage 0 — params provenance
# ---------------------------------------------------------------------------

def stage0_params_provenance() -> dict:
    """Diff the params under test against the research-era LOCKED dict.

    This stage exists because its absence made every other stage meaningless
    for months: the tool imported LOCKED and validated vol 2.0 / funding 0.0005
    / risk 2.75 while the droplet ran 1.5 / 0.0015 / 3.5. Drift here is reported
    as INFO, not FAIL — LOCKED is a frozen research artifact and is SUPPOSED to
    fall behind production; what must never happen again is validating it by
    accident. The verdict fails only if config/params.yaml can't be read or is
    missing keys the strategy needs — and that failure is STRUCTURED (a report
    plus a non-zero exit), never a bare traceback.
    """
    print("[validate stage 0] params provenance", file=sys.stderr)

    if DEPLOYED_LOAD_ERROR is not None:
        print(f"  FAIL: could not load {CONFIG_PATH}: {DEPLOYED_LOAD_ERROR}",
              file=sys.stderr)
        return {
            "stage": 0,
            "source_of_truth": str(CONFIG_PATH.relative_to(ROOT)),
            "load_error": DEPLOYED_LOAD_ERROR,
            "params_under_test": {},
            "drift_vs_research_locked": {},
            "drift_key_count": 0,
            "missing_required_keys": [],
            "verdict": "FAIL",
        }

    drift = {}
    for k in sorted(set(LOCKED) | set(DEPLOYED)):
        a, b = LOCKED.get(k, "<absent>"), DEPLOYED.get(k, "<absent>")
        if a != b:
            drift[k] = {"research_locked": a, "deployed_params_yaml": b}

    required = ("rsi_period", "rsi_long_threshold", "rsi_short_threshold",
                "volume_ma_period", "volume_multiple", "mf_trend_ema_period",
                "require_trend", "funding_extreme_threshold", "sl_pct", "tp_pct")
    missing = [k for k in required if k not in DEPLOYED]

    result = {
        "stage": 0,
        "source_of_truth": str(CONFIG_PATH.relative_to(ROOT)),
        "params_under_test": DEPLOYED,
        "drift_vs_research_locked": drift,
        "drift_key_count": len(drift),
        "missing_required_keys": missing,
        "verdict": "PASS" if not missing else "FAIL",
    }
    print(f"  params from {result['source_of_truth']}; "
          f"{len(drift)} key(s) differ from research LOCKED; "
          f"verdict={result['verdict']}", file=sys.stderr)
    for k, v in drift.items():
        print(f"    drift: {k}: LOCKED={v['research_locked']} "
              f"DEPLOYED={v['deployed_params_yaml']}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Stage 1 — impl cross-check (new in-class gate vs old subclass)
# ---------------------------------------------------------------------------

def _trade_signatures(stats) -> list[tuple]:
    """Build a canonical, hashable trade signature for diffing.

    Uses (EntryTime, ExitTime, Size sign, EntryPrice rounded) — robust across
    columns that vary by backtesting.py version. Two strategies are
    behaviourally identical iff trade signatures match.
    """
    df = getattr(stats, "_trades", None)
    if df is None or len(df) == 0:
        return []
    out = []
    for _, r in df.iterrows():
        out.append((
            str(r["EntryTime"]),
            str(r["ExitTime"]),
            int(np.sign(float(r["Size"]))),
            round(float(r["EntryPrice"]), 2),
        ))
    return out


def stage1_impl_cross_check() -> dict:
    print(f"[validate stage 1] window={STAGE1_START}..{STAGE1_END}", file=sys.stderr)

    df = _load_slice(PARQ_15M, STAGE1_START, STAGE1_END, attach_funding=True)

    # Path A: new in-class gate via DayTradeMultiFactorBTC with use_mtf_4h_gate=True.
    kw_a, dropped_a = _class_kwargs(DayTradeMultiFactorBTC, DEPLOYED)
    config_a = {**kw_a, "use_mtf_4h_gate": True, "mtf_4h_ema_period": 200,
                "mtf_4h_parquet_path": str(PARQ_4H)}
    bt_a = Backtest(df, DayTradeMultiFactorBTC, cash=CASH, commission=COMMISSION,
                     margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                     finalize_trades=True)
    stats_a = bt_a.run(**config_a)
    sigs_a = _trade_signatures(stats_a)

    # Path B: existing MultiFactorMTF4H subclass from the deepening tool. It
    # reads `self.data.Ema4h` (attached by the runner), so we replicate that.
    kw_b, dropped_b = _class_kwargs(MultiFactorMTF4H, DEPLOYED)
    df_b = df.copy()
    df_b["Ema4h"] = _build_4h_ema_aligned(
        df_b.index, PARQ_4H, ema_period=200,
    )
    bt_b = Backtest(df_b, MultiFactorMTF4H, cash=CASH, commission=COMMISSION,
                     margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                     finalize_trades=True)
    stats_b = bt_b.run(**kw_b)
    sigs_b = _trade_signatures(stats_b)

    only_a = sorted(set(sigs_a) - set(sigs_b))
    only_b = sorted(set(sigs_b) - set(sigs_a))
    common = sorted(set(sigs_a) & set(sigs_b))

    result = {
        "stage": 1,
        "window": [STAGE1_START, STAGE1_END],
        "trades_path_a_in_class": len(sigs_a),
        "trades_path_b_subclass": len(sigs_b),
        "trades_common": len(common),
        "trades_only_in_path_a": only_a[:50],   # cap diff dump
        "trades_only_in_path_b": only_b[:50],
        "stats_a_return_pct": float(stats_a.get("Return [%]", 0.0) or 0.0),
        "stats_b_return_pct": float(stats_b.get("Return [%]", 0.0) or 0.0),
        # Surfaced, not swallowed: a deployed key the class can't accept means
        # this stage silently tested a default instead of production.
        "deployed_keys_dropped_path_a": dropped_a,
        "deployed_keys_dropped_path_b": dropped_b,
        "verdict": "PASS" if (not only_a and not only_b) else "FAIL",
    }
    print(
        f"  trades A={len(sigs_a)} B={len(sigs_b)} common={len(common)}  "
        f"verdict={result['verdict']}",
        file=sys.stderr,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 2 — live evaluator vs backtest per-bar parity
# ---------------------------------------------------------------------------

class _BarRecorder(DayTradeMultiFactorBTC):
    """Subclass that records, for EVERY bar in the window, whether
    _long_signal / _short_signal would return True if asked.

    We don't piggyback on the parent's next() because the parent only calls
    _long_signal/_short_signal when there's no active position — so trades
    that are open during a bar would suppress the recording, biasing the
    parity-check sample. We probe both signal functions unconditionally
    here at every bar.
    """

    def init(self) -> None:
        super().init()
        self._fired_long: list[int] = []
        self._fired_short: list[int] = []

    def next(self) -> None:
        i = len(self.data) - 1
        # Probe both directions unconditionally. The parent's signal methods
        # have no side effects beyond returning a bool.
        try:
            if super()._long_signal(i):
                self._fired_long.append(i)
        except Exception:  # noqa: BLE001
            pass
        try:
            if super()._short_signal(i):
                self._fired_short.append(i)
        except Exception:  # noqa: BLE001
            pass


def stage2_live_vs_backtest() -> dict:
    print(f"[validate stage 2] window={STAGE2_START}..{STAGE2_END}", file=sys.stderr)
    df = _load_slice(PARQ_15M, STAGE2_START, STAGE2_END, attach_funding=True)
    print(f"  bars={len(df)}", file=sys.stderr)

    # --- Backtest path: collect signal bars ---
    kw_bt, _dropped = _class_kwargs(DayTradeMultiFactorBTC, DEPLOYED)
    config_bt = {**kw_bt, "use_mtf_4h_gate": True, "mtf_4h_ema_period": 200,
                  "mtf_4h_parquet_path": str(PARQ_4H)}
    bt = Backtest(df, _BarRecorder, cash=CASH, commission=COMMISSION,
                   margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                   finalize_trades=True)
    stats = bt.run(**config_bt)
    strat = stats._strategy  # backtesting.py exposes the strategy instance
    bt_long = set(strat._fired_long)
    bt_short = set(strat._fired_short)
    print(f"  backtest: long_fires={len(bt_long)} short_fires={len(bt_short)}",
          file=sys.stderr)

    # --- Live path: per-bar evaluate_signal with injected bars_4h ---
    # Pre-load 4H bars covering the window plus 180-day warmup.
    bars_4h_full = pd.read_parquet(PARQ_4H)
    if bars_4h_full.index.tz is not None:
        bars_4h_full.index = bars_4h_full.index.tz_localize(None)
    cutoff_lo = pd.Timestamp(STAGE2_START) - pd.Timedelta(days=180)
    bars_4h_window = bars_4h_full.loc[cutoff_lo:].copy()

    warm = (max(DEPLOYED["mf_trend_ema_period"], DEPLOYED["volume_ma_period"],
                DEPLOYED["rsi_period"]) + 5)
    live_long: set[int] = set()
    live_short: set[int] = set()

    t0 = time.time()
    for i in range(warm, len(df)):
        bars_slice = df.iloc[: i + 1]
        funding_v = float(bars_slice["Funding"].iloc[-1]) if "Funding" in bars_slice.columns else 0.0
        # Only pass 4H bars whose CLOSE time is <= current 15m bar's timestamp.
        # The live evaluator's _compute_4h_ema_at_15m_close enforces this too,
        # but slicing here keeps the live path's EMA from accidentally peeking
        # at a bar that closed after the 15m bar's wall clock.
        cur_ts = bars_slice.index[-1]
        cur_4h = bars_4h_window.loc[: cur_ts]
        side, _dbg = evaluate_signal(bars_slice, funding_v, LIVE_PARAMS,
                                      bars_4h=cur_4h)
        if side == "long":
            live_long.add(i)
        elif side == "short":
            live_short.add(i)
        if i % 500 == 0:
            print(f"    bar {i}/{len(df)} t={time.time()-t0:.1f}s", file=sys.stderr)

    print(f"  live:     long_fires={len(live_long)} short_fires={len(live_short)} "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)

    # Per-bar comparison: each bar i is "agree" iff bt-decision matches live-decision.
    # 4-state per bar: (bt_long, bt_short, live_long, live_short)
    n_total = len(df) - warm
    n_agree = 0
    mismatches: list[dict] = []
    for i in range(warm, len(df)):
        bt_dec = "long" if i in bt_long else ("short" if i in bt_short else None)
        lv_dec = "long" if i in live_long else ("short" if i in live_short else None)
        if bt_dec == lv_dec:
            n_agree += 1
        else:
            if len(mismatches) < 50:  # cap dump
                mismatches.append({
                    "bar_idx": i,
                    "ts": str(df.index[i]),
                    "backtest": bt_dec,
                    "live": lv_dec,
                })

    agreement_pct = 100.0 * n_agree / max(n_total, 1)
    verdict = "PASS" if agreement_pct >= PARITY_GATE else "FAIL"

    return {
        "stage": 2,
        "window": [STAGE2_START, STAGE2_END],
        "warmup_bars": warm,
        "evaluable_bars": n_total,
        "bars_in_agreement": n_agree,
        "agreement_pct": round(agreement_pct, 4),
        "parity_gate_pct": PARITY_GATE,
        "backtest_long_fires": len(bt_long),
        "backtest_short_fires": len(bt_short),
        "live_long_fires": len(live_long),
        "live_short_fires": len(live_short),
        "mismatches_sample": mismatches,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Stage 3 — exit parity (the gap that let the live exit go missing)
# ---------------------------------------------------------------------------

class _ExitRecorder(DayTradeMultiFactorBTC):
    """Records, at EVERY bar, what next()'s adverse-trend branch WOULD decide
    for a hypothetical long and a hypothetical short.

    Mirrors signals_multifactor.py's in-position branch exactly:

        t = self._trend_ema[i]
        if np.isfinite(t):
            long  closes when close_v < t
            short closes when close_v > t

    Probing both sides unconditionally (rather than reading real positions) is
    the same trick `_BarRecorder` uses for entries: whether a position happens
    to be open must not bias the sample. Reads the class's OWN `_trend_ema`, so
    if the backtest changes its indicator this stage moves with it.
    """

    def init(self) -> None:
        super().init()
        self._exit_long: list[int] = []
        self._exit_short: list[int] = []

    def next(self) -> None:
        if not self.require_trend:
            return
        i = len(self.data) - 1
        t = self._trend_ema[i]
        if not np.isfinite(t):
            return
        close_v = self.data.Close[-1]
        if close_v < t:
            self._exit_long.append(i)
        if close_v > t:
            self._exit_short.append(i)


def stage3_exit_parity() -> dict:
    """Bar-for-bar EXIT parity: backtest branch vs the live exit function.

    Two live shapes, because they answer different questions:
      full_prefix  — every bar from window start. Pure algebraic parity; ewm is
                     a causal recursion seeded at bar 0 in BOTH paths, so this
                     must be EXACT. A miss here is a logic bug.
      rolling_1500 — the last 1500 bars only, which is what bot.py fetches. Its
                     EMA is seeded at the window start, not at genesis, so the
                     initial condition decays as (1-2/201)^k instead of being
                     shared. This is the shape that can drift, and because the
                     exit is a THRESHOLD CROSSING a tiny drift can flip a bar.
    """
    print(f"[validate stage 3] window={STAGE2_START}..{STAGE2_END}", file=sys.stderr)
    df = _load_slice(PARQ_15M, STAGE2_START, STAGE2_END, attach_funding=True)

    kw, dropped = _class_kwargs(DayTradeMultiFactorBTC, DEPLOYED)
    config_bt = {**kw, "use_mtf_4h_gate": True, "mtf_4h_ema_period": 200,
                 "mtf_4h_parquet_path": str(PARQ_4H)}
    bt = Backtest(df, _ExitRecorder, cash=CASH, commission=COMMISSION,
                  margin=MARGIN, trade_on_close=False, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run(**config_bt)
    strat = stats._strategy
    bt_exit_long = set(strat._exit_long)
    bt_exit_short = set(strat._exit_short)
    print(f"  backtest: exit_long_bars={len(bt_exit_long)} "
          f"exit_short_bars={len(bt_exit_short)}", file=sys.stderr)

    period = int(DEPLOYED["mf_trend_ema_period"])
    shapes = {"full_prefix": None, "rolling_1500": LIVE_FETCH_BARS}
    out_shapes: dict = {}

    for shape_name, window in shapes.items():
        t0 = time.time()
        agree = 0
        n_total = 0
        mismatches: list[dict] = []
        fired_long = fired_short = 0

        # Start at period-1, not period: the live guard is `len(bars) < period`,
        # so at i = period-1 the prefix is exactly `period` bars and the function
        # DOES evaluate. Starting a bar later would silently exclude the first
        # comparable bar — the backtest records an exit there — and this stage
        # exists precisely because a check that quietly skips things gets read
        # as coverage.
        for i in range(period - 1, len(df)):
            lo = 0 if window is None else max(0, i + 1 - window)
            bars = df.iloc[lo: i + 1]
            lv_long, _ = trend_exit_signal_multifactor_v1(bars, "long", LIVE_PARAMS)
            lv_short, _ = trend_exit_signal_multifactor_v1(bars, "short", LIVE_PARAMS)
            fired_long += bool(lv_long)
            fired_short += bool(lv_short)

            n_total += 1
            ok = (lv_long == (i in bt_exit_long)) and (lv_short == (i in bt_exit_short))
            if ok:
                agree += 1
            elif len(mismatches) < 50:
                mismatches.append({
                    "bar_idx": i,
                    "ts": str(df.index[i]),
                    "backtest_long_exit": i in bt_exit_long,
                    "live_long_exit": bool(lv_long),
                    "backtest_short_exit": i in bt_exit_short,
                    "live_short_exit": bool(lv_short),
                })

        pct = 100.0 * agree / max(n_total, 1)
        gate = EXIT_PARITY_GATE if window is None else PARITY_GATE
        out_shapes[shape_name] = {
            "live_window_bars": window or "full prefix",
            "evaluable_bars": n_total,
            "bars_in_agreement": agree,
            "agreement_pct": round(pct, 4),
            "gate_pct": gate,
            "live_long_exit_bars": fired_long,
            "live_short_exit_bars": fired_short,
            "mismatches_sample": mismatches,
            "verdict": "PASS" if pct >= gate else "FAIL",
        }
        print(f"  {shape_name}: {pct:.4f}% (gate {gate}%) "
              f"{out_shapes[shape_name]['verdict']} ({time.time()-t0:.1f}s)",
              file=sys.stderr)

    # A stage that never sees an exit fire would "pass" vacuously — exactly the
    # failure mode of the old entry-only check. Require real signal on both sides.
    exercised = len(bt_exit_long) >= 10 and len(bt_exit_short) >= 10
    verdict = ("PASS" if all(s["verdict"] == "PASS" for s in out_shapes.values())
               and exercised else "FAIL")

    return {
        "stage": 3,
        "what": "adverse-EMA200 exit: next() in-position branch vs "
                "trend_exit_signal_multifactor_v1",
        "window": [STAGE2_START, STAGE2_END],
        "trend_ema_period": period,
        "backtest_exit_long_bars": len(bt_exit_long),
        "backtest_exit_short_bars": len(bt_exit_short),
        "non_vacuous": exercised,
        "deployed_keys_dropped": dropped,
        "shapes": out_shapes,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("multifactor-v1 + 4H EMA200 gate — parity validation")
    print("=" * 78)

    out: dict = {"strategy": "multifactor-v1+4h-gate"}

    out["stage0_params_provenance"] = stage0_params_provenance()
    if out["stage0_params_provenance"]["verdict"] != "PASS":
        # Stages 1-3 would run on empty params and quietly measure the backtest
        # class DEFAULTS — a green parity number for a config nobody deployed,
        # which is the precise failure stage 0 was added to prevent. Stop here
        # and still emit a report.
        out["overall_verdict"] = "FAIL"
        out["aborted_after_stage"] = 0
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print()
        print("stage 0: FAIL — "
              f"{out['stage0_params_provenance'].get('load_error') or 'missing required keys'}")
        print("stages 1-3 SKIPPED (would validate class defaults, not production)")
        print(f"saved   → {OUT}")
        return 1

    out["stage1_impl_cross_check"] = stage1_impl_cross_check()
    out["stage2_entry_parity"] = stage2_live_vs_backtest()
    out["stage3_exit_parity"] = stage3_exit_parity()

    # Back-compat: the old key name is what existing reports and any consumer
    # of reports/multifactor_4h_parity_validation.json look for.
    out["stage2_live_vs_backtest"] = out["stage2_entry_parity"]

    overall = all(out[k]["verdict"] == "PASS" for k in (
        "stage0_params_provenance", "stage1_impl_cross_check",
        "stage2_entry_parity", "stage3_exit_parity"))
    out["overall_verdict"] = "PASS" if overall else "FAIL"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    s3 = out["stage3_exit_parity"]
    print()
    print(f"stage 0: {out['stage0_params_provenance']['verdict']}  "
          f"({out['stage0_params_provenance']['drift_key_count']} key(s) differ "
          f"from research LOCKED)")
    print(f"stage 1: {out['stage1_impl_cross_check']['verdict']}")
    print(f"stage 2 (entry): {out['stage2_entry_parity']['verdict']}  "
          f"({out['stage2_entry_parity']['agreement_pct']}%)")
    print(f"stage 3 (exit):  {s3['verdict']}  "
          f"full-prefix {s3['shapes']['full_prefix']['agreement_pct']}% / "
          f"as-live {s3['shapes']['rolling_1500']['agreement_pct']}%")
    print(f"overall: {out['overall_verdict']}")
    print(f"saved   → {OUT}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
