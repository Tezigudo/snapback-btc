"""The leg dashboard must not read a deposit as profit, or invent a kill line.

Regression cover for 2026-09-09. `tools/build_dashboard.py` runs from cron every
60 s and measured P&L as `cur_equity / deploy_start_equity`, a RAW snapshot
frozen at first boot. Every later deposit moved the numerator and left the basis
behind, so the page contradicted itself:

    v1 dashboard card:   +10.46%  /  +$14.96      <- deploy_start_equity 142.93
    v1 bot log, SAME page: -14.25%                <- principal 183.93

donchian was worse -- start 50.50 against principal 172.53 renders +253.96% for
a leg that is up 3.60%.

Separately the kill card computed `deploy_start_equity * 0.82`, which was the
deploy-era rule. The live switch is `principal * kill_switch_equity_fraction`
(0.645 on every leg), so the page named a floor the bot does not enforce. That
constant is now read from the leg's config, never hardcoded.

Fourth reader of the same pair, after PR #25 (monitor.py) and PR #30 (bot.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import build_dashboard as bd  # noqa: E402


def _db(*, deploy_start: str, principal: str | None, equity_after: float) -> dict:
    """A minimal read_db() payload. meta values are TEXT, as sqlite returns."""
    meta = {"deploy_start_equity": deploy_start,
            "deploy_start_ts": "2026-07-25T05:45:27+00:00"}
    if principal is not None:
        meta["principal_anchor"] = principal
    return {
        "meta": meta,
        # (ts, side, qty, price, pnl_usd, reason, equity_after)
        "fills": [("2026-09-09T03:00:00+00:00", "long", 0.01, 100.0,
                   None, "entry", equity_after)],
        "events": [],
    }


def _render(db: dict, instance: str = "v1") -> str:
    return bd.render_html(db, [], instance)


class TestPnlBasis:

    def test_v1_deposit_is_not_reported_as_profit(self, caplog):
        """The live numbers. $142.93 at deploy, $183.93 actually funded."""
        html = _render(_db(deploy_start="142.93", principal="183.93",
                           equity_after=157.89))
        assert "-14.1" in html          # truth
        assert "10.4" not in html       # the figure the bug printed
        assert "vs principal" in html

    def test_donchian_stale_start_does_not_render_a_fake_250_percent(self):
        """start 50.50 vs principal 172.53 -- the widest gap in the book."""
        html = _render(_db(deploy_start="50.50", principal="172.53",
                           equity_after=178.75), instance="donchian")
        assert "+3.6" in html
        assert "253" not in html

    def test_basis_is_principal_when_present(self):
        """Clean round numbers so the arithmetic is unambiguous:
        110/100 = +10.00%, while the old basis of 25 would give +340.00%."""
        html = _render(_db(deploy_start="25.00", principal="100.00",
                           equity_after=110.00))
        assert "+10.00%" in html
        assert "340" not in html

    def test_missing_principal_falls_back_but_says_so(self):
        """A pre-Part-C leg has no ledger. Falling back is correct; printing the
        unadjusted number WITHOUT the caveat is what kept this invisible."""
        html = _render(_db(deploy_start="100.00", principal=None,
                           equity_after=110.00))
        assert "+10.00%" in html
        assert "NOT transfer-adjusted" in html


class TestKillLine:

    def test_kill_line_uses_the_configured_fraction_not_0_82(self):
        """0.645 of principal 100.00 = 64.50. The old rule would have printed
        0.82 of deploy-start 25.00 = 20.50."""
        html = _render(_db(deploy_start="25.00", principal="100.00",
                           equity_after=110.00))
        assert "$64.50" in html
        assert "$20.50" not in html
        assert "-35.5% from principal" in html

    def test_stale_18_percent_label_is_gone(self):
        html = _render(_db(deploy_start="100.00", principal="100.00",
                           equity_after=100.00))
        assert "-18% from start" not in html

    def test_unreadable_config_renders_a_dash_not_a_number(self):
        """An invented kill level is worse than none -- you would plan around
        it. read_kill_fraction returns None on any failure."""
        with patch.object(bd, "read_kill_fraction", return_value=None):
            html = _render(_db(deploy_start="100.00", principal="100.00",
                               equity_after=110.00))
        assert "config unreadable" in html

    def test_read_kill_fraction_reads_every_configured_leg(self):
        """Reads the real YAML on disk, so a config rename breaks the test
        rather than silently returning None and dashing out the kill card."""
        for inst in bd.INSTANCE_FILES:
            assert bd.read_kill_fraction(inst) == 0.645, inst

    def test_read_kill_fraction_rejects_a_nonsense_value(self):
        """Out-of-range means the config is wrong, not that we should multiply
        equity by it."""
        with patch.object(bd.yaml, "safe_load",
                          return_value={"deploy": {"kill_switch_equity_fraction": 42}}):
            assert bd.read_kill_fraction("v1") is None
