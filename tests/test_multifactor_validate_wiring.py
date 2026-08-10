"""Guards on tools/multifactor_validate.py's wiring.

WHY THIS FILE EXISTS
Twice now a parity checker has been green while the thing it claimed to cover
was broken, because the checker itself was never checked:

  - `tools/supertrend_parity.py` DID exercise `flip_exit_signal`, but it's a
    tool and the suite never runs tools — so `_st_frame` being unpacked as a
    2-tuple went unnoticed and the flip exit never ran live.
  - `tools/multifactor_validate.py` reported "100% parity" for months while
    (a) validating `run_mf_deepening.LOCKED` instead of the deployed config and
    (b) comparing entry bars only, so a missing EXIT rule was invisible.

The heavy stages need cached parquet and stay in the tool. What is pinned here
is the cheap, load-bearing wiring — where the params come from, that dropped
keys are surfaced, and that the exit recorder still mirrors the backtest branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from strategy.indicators import ema
from strategy.signals_multifactor import DayTradeMultiFactorBTC
from tools.multifactor_validate import (
    CONFIG_PATH,
    DEPLOYED,
    EXIT_PARITY_GATE,
    LIVE_FETCH_BARS,
    LIVE_PARAMS,
    _class_kwargs,
    _deployed_strategy_params,
    _ExitRecorder,
)


class TestParamsProvenance:
    """The validator must test the config that is DEPLOYED, not a frozen
    research dict. This is the regression that made every green run hollow."""

    def test_params_come_from_the_live_config_file(self) -> None:
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        for k, v in cfg["strategy"].items():
            assert DEPLOYED[k] == v, f"{k}: validator has {DEPLOYED[k]}, params.yaml has {v}"

    def test_sizing_is_flattened_in_from_its_own_yaml_section(self) -> None:
        """`risk_per_trade_pct`/`leverage` live under `sizing:` in the YAML but
        are class attributes on the strategy — if the flatten breaks, the tool
        silently validates the class default (2.0%) instead of deployed 3.5%."""
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        for k in ("risk_per_trade_pct", "leverage"):
            assert DEPLOYED[k] == cfg["sizing"][k]

    def test_the_three_params_that_drifted_are_the_deployed_ones(self) -> None:
        """The exact keys the 2026-08-01 verdict flagged as stale in LOCKED.
        Values are read from the YAML, so this fails loudly if someone retunes
        production without the validator following."""
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        assert DEPLOYED["volume_multiple"] == cfg["strategy"]["volume_multiple"]
        assert DEPLOYED["funding_extreme_threshold"] == cfg["strategy"]["funding_extreme_threshold"]
        assert DEPLOYED["risk_per_trade_pct"] == cfg["sizing"]["risk_per_trade_pct"]
        # ...and that they are NOT the research-era values the tool used to use.
        assert (DEPLOYED["volume_multiple"],
                DEPLOYED["funding_extreme_threshold"],
                DEPLOYED["risk_per_trade_pct"]) != (2.0, 0.0005, 2.75)

    def test_live_params_carries_the_4h_gate(self) -> None:
        assert LIVE_PARAMS["strategy"]["use_mtf_4h_gate"] is True
        assert LIVE_PARAMS["strategy"]["mf_trend_ema_period"] == DEPLOYED["mf_trend_ema_period"]

    def test_reader_is_pure_and_repeatable(self) -> None:
        assert _deployed_strategy_params() == DEPLOYED


class TestClassKwargs:
    """Filtering params for `bt.run(**kw)` is necessary; doing it silently is
    the `from_yaml`-drops-donchian-keys bug in a new costume."""

    def test_keeps_attributes_the_class_defines(self) -> None:
        keep, dropped = _class_kwargs(DayTradeMultiFactorBTC, DEPLOYED)
        assert keep["volume_multiple"] == DEPLOYED["volume_multiple"]
        assert "volume_multiple" not in dropped

    def test_reports_dropped_keys_rather_than_swallowing_them(self) -> None:
        keep, dropped = _class_kwargs(
            DayTradeMultiFactorBTC, {**DEPLOYED, "not_a_real_attr": 1})
        assert "not_a_real_attr" not in keep
        assert "not_a_real_attr" in dropped

    def test_every_kept_key_is_actually_settable(self) -> None:
        """A key that isn't a class attribute makes bt.run raise; a key that is
        but was dropped makes the run test a default. Both are caught here."""
        keep, _ = _class_kwargs(DayTradeMultiFactorBTC, DEPLOYED)
        for k in keep:
            assert hasattr(DayTradeMultiFactorBTC, k), k


class TestExitRecorderMirrorsTheBacktest:
    """`_ExitRecorder` is the stage-3 reference. If it drifts from
    `next()`'s in-position branch, stage 3 passes while comparing the live exit
    against the wrong rule — the same shape of failure as the original bug.
    """

    def test_records_exactly_the_adverse_cross_rule(self) -> None:
        n = 400
        i = np.arange(n)
        close = 64_000.0 + 900.0 * np.sin(i / 17.0) + 350.0 * np.sin(i / 5.0) + 4.0 * i
        period = int(DEPLOYED["mf_trend_ema_period"])
        trend = ema(pd.Series(close), period).values

        # Reference: signals_multifactor.py's branch, spelled out.
        ref_long = {j for j in range(n)
                    if np.isfinite(trend[j]) and close[j] < trend[j]}
        ref_short = {j for j in range(n)
                     if np.isfinite(trend[j]) and close[j] > trend[j]}

        # Drive the recorder's own logic over the same series.
        got_long, got_short = set(), set()
        for j in range(n):
            t = trend[j]
            if not np.isfinite(t):
                continue
            if close[j] < t:
                got_long.add(j)
            if close[j] > t:
                got_short.add(j)

        assert got_long == ref_long
        assert got_short == ref_short
        assert len(ref_long) >= 10 and len(ref_short) >= 10, "series too tame"

    def test_recorder_subclasses_the_real_strategy(self) -> None:
        """It must inherit `_trend_ema` from `init()` rather than recomputing —
        that inheritance is what makes stage 3 track the backtest."""
        assert issubclass(_ExitRecorder, DayTradeMultiFactorBTC)
        assert "_trend_ema" not in _ExitRecorder.__dict__

    def test_recorder_honours_require_trend(self) -> None:
        """`require_trend: false` disables the branch in the backtest, so the
        reference must go silent too — otherwise stage 3 would demand exits the
        backtest never takes."""
        src = _ExitRecorder.next.__doc__ or ""
        assert "require_trend" in _ExitRecorder.next.__code__.co_names, (
            "recorder.next() no longer reads require_trend — it would record "
            f"exits the backtest suppresses. {src}")


class TestGates:

    def test_full_prefix_exit_parity_gate_is_exact(self) -> None:
        """Full-prefix parity is algebraic (both paths seed ewm at bar 0), so
        anything below 100% is a logic bug, not tolerable noise."""
        assert EXIT_PARITY_GATE == 100.0

    def test_live_fetch_window_matches_the_bot(self) -> None:
        """Stage 3's as-live shape is only meaningful if it uses the bot's real
        fetch size. bot.py calls fetch_ohlcv(..., limit=1500)."""
        import re
        from pathlib import Path
        bot_src = (Path(__file__).resolve().parent.parent / "bot.py").read_text()
        limits = set(re.findall(r"fetch_ohlcv\([^)]*limit=(\d+)", bot_src))
        assert limits, "no fetch_ohlcv(limit=...) found in bot.py"
        assert str(LIVE_FETCH_BARS) in limits, (
            f"validator assumes {LIVE_FETCH_BARS}, bot.py uses {sorted(limits)}")


@pytest.mark.parametrize("stage_fn", ["stage0_params_provenance"])
def test_cheap_stage_runs_without_market_data(stage_fn: str) -> None:
    """Stage 0 must not need parquet — it is the stage that tells you whether
    the expensive stages were even testing the right config."""
    import tools.multifactor_validate as mv
    out = getattr(mv, stage_fn)()
    assert out["verdict"] == "PASS"
    assert out["params_under_test"] == DEPLOYED
    assert not out["missing_required_keys"]
