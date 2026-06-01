"""Cross-coin Phase 1 walk-forward backtest.

For each (coin × strategy) pair, run the BTC-tuned strategy on the coin's
full 4h / 15m history, split into IS (windows 1..8) and OOS (windows 9..12),
and capture per-window + aggregate metrics. The point is candidate screening:
identify which pairs are worth a deep Phase 2-5 validation, not to ship them.

Strategies:
  - multifactor-v1 (15m): uses RSI/EMA/funding confluence
  - donchian-v3 cons (4h): 80/20 channel, slope-gate on, 1.5x ATR SL/TP
  - cnh-hybrid-short-v1 (4h): DT + ICnH detectors with dedup=15

Coins:
  - BTC (baseline, should reproduce existing numbers)
  - ETH, SOL, ADA, WLD

Output:
  - reports/cross_coin_results_<UTC>.json — raw structured data
  - reports/cross_coin_summary_<UTC>.html — master ranking + hyperlinks
  - reports/cross_coin_{coin}_{strategy}_<UTC>.html — per-pair detail page

WLD has only ~2.5yr history; its IS set will be smaller. Results flagged in
the report.

Run:
    uv run python tools/cross_coin_backtest.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import run_backtest  # noqa: E402
from exchange.data import load_klines  # noqa: E402
from strategy.signals import StrategyParams  # noqa: E402
from strategy.indicators import atr, ema  # noqa: E402
from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
from tools.icnh_mega_sweep import Config, simulate_trades, WINDOWS  # noqa: E402

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COINS = ["BTC", "ETH", "SOL", "ADA", "WLD", "PAXG"]

# IS/OOS split mirrors hybrid_walkforward.py: 8 IS + 4 OOS.
IS_LABELS = {w[0] for w in WINDOWS[:8]}
OOS_LABELS = {w[0] for w in WINDOWS[8:]}

ATR_LEN = 14


# ===== Helpers =====


def _symbol(coin: str) -> str:
    return f"{coin}/USDT:USDT"


def _load_coin_4h(coin: str) -> pd.DataFrame:
    """Load 4h OHLCV + attach EMA/ATR indicators for the cnh-hybrid detector.

    Mirrors `icnh_mega_sweep.load_tf` but parametrised on coin.
    """
    df = load_klines(symbol=_symbol(coin), timeframe="4h", days_back=2200).copy()
    if df.empty:
        return df
    # Lowercase to match the detectors' expected column names.
    df.columns = [c.lower() for c in df.columns]
    df["ema7"] = ema(df["close"], 7)
    df["ema24"] = ema(df["close"], 24)
    df["ema50"] = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 200)
    df["atr14"] = atr(df["high"], df["low"], df["close"], ATR_LEN)
    return df


def _annualised_sharpe(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(365))


def _max_dd_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min() * 100.0)


def _donchian_cons_params() -> StrategyParams:
    p = StrategyParams.from_yaml()
    return dataclasses.replace(
        p,
        donchian_period_entry=80, donchian_period_exit=20,
        atr_sl_multiple=1.5, atr_tp_multiple=1.5, atr_trail_multiple=0.0,
        leverage=20, regime_ema_period=120, regime_slope_window=30,
        slope_trend_threshold_pct=0.03, time_stop_bars=48,
        volume_multiple=1.0,
    )


def _rider_v1_params() -> StrategyParams:
    """Params for rider-v1 walk-forward.

    All rider-specific config (risk%, sl_atr, tp_atr, channel N, ATR period,
    EMA period, time-stop) lives as class attrs on DonchianRiderV1.
    We only need to pass leverage=3 via StrategyParams so run_backtest picks
    it up via eff_leverage = leverage or params.leverage.
    """
    p = StrategyParams.from_yaml()
    return dataclasses.replace(p, leverage=3)


def _to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# ===== Per-strategy runners =====


def _run_backtest_per_window(
    coin: str, strategy: str, timeframe: str,
    params_override: StrategyParams | None = None,
) -> list[dict]:
    """Run a v1 or donchian backtest per 12-window slice.

    Returns a list of per-window dicts. A window with insufficient data
    (Binance lists a coin after window start) is recorded with trades=0.
    """
    sym = _symbol(coin)
    out: list[dict] = []
    for label, start, end in WINDOWS:
        s_dt = _to_dt(start)
        e_dt = _to_dt(end)
        try:
            r = run_backtest(
                strategy_name=strategy, symbol=sym, timeframe=timeframe,
                start=s_dt, end=e_dt,
                params_override=params_override,
                quiet=True, return_equity=True,
            )
            eq_series = r.get("equity_series")
            daily_ret = (
                (eq_series / eq_series.iloc[0]).resample("1D").last().ffill().pct_change()
                if eq_series is not None and not eq_series.empty else pd.Series(dtype=float)
            )
            sharpe = _annualised_sharpe(daily_ret)
            out.append({
                "window": label,
                "start": start, "end": end,
                "trades": int(r.get("trades", 0)),
                "ret_pct": float(r.get("after_funding_pct", r.get("backtest_return_pct", 0.0))),
                "sharpe": float(sharpe),
                "max_dd_pct": float(r.get("max_drawdown_pct", 0.0)),
                "win_rate_pct": float(r.get("win_rate_pct") or 0.0),
            })
        except Exception as e:
            # "Missing klines" = coin wasn't listed on Binance Futures yet in
            # this window. Mark no_data=True so aggregation skips it instead
            # of treating it as a zero-PnL window (which inflates "positive
            # windows" denominators and dilutes the summary).
            msg = str(e)
            is_no_data = ("Missing" in msg and "klines" in msg)
            out.append({
                "window": label, "start": start, "end": end,
                "trades": 0, "ret_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "win_rate_pct": 0.0,
                "no_data": is_no_data,
                "error" if not is_no_data else "note":
                    "no data (coin not yet listed)" if is_no_data
                    else f"{type(e).__name__}: {e}",
            })
    return out


def _run_cnh_hybrid_per_window(coin: str) -> list[dict]:
    """Run cnh-hybrid-short-v1 with locked Phase 1 winner params per window.
    Mirrors the math in `hybrid_walkforward._run_hybrid_on_subset` so results
    are directly comparable to the existing BTC validation.
    """
    df = _load_coin_4h(coin)
    if df.empty:
        return [{"window": w[0], "start": w[1], "end": w[2],
                 "trades": 0, "win_rate": 0.0, "cum": 0.0, "sharpe": 0.0,
                 "error": "no data"} for w in WINDOWS]

    dt_cfg = Config(
        name="hybrid_dt", pattern_type="distribution_top", direction="short",
        tf="4h", uptrend_bars=16, chop_bars=8, min_rise_pct=2.5,
        max_chop_ratio=0.55, require_chop_at_top=True,
        breakdown_mode="chop_low_or_ema24",
        sl_atr_mult=1.5, regime_sl_mode="off", tp_emas=("ema100",),
        entry_emas=("ema24",), dedup_bars=15,
    )
    icnh_cfg = Config(
        name="hybrid_icnh", pattern_type="inverse_cnh", direction="short",
        tf="4h", cup_len=20, handle_len=4, min_r2=0.50,
        min_cup_depth_atr=1.0, handle_max_depth_frac=0.70, peak_tolerance=6,
        entry_emas=("ema24",), sl_atr_mult=1.5, regime_sl_mode="off",
        tp_emas=("ema100",), dedup_bars=15,
    )

    out: list[dict] = []
    for label, start, end in WINDOWS:
        sub = df.loc[start:end]
        if len(sub) < 100:
            out.append({"window": label, "start": start, "end": end,
                        "trades": 0, "win_rate": 0.0, "cum": 0.0,
                        "sharpe": 0.0, "ret_pct": 0.0,
                        "no_data": True,
                        "note": "no data (coin not yet listed)"})
            continue
        hits = find_hybrid_patterns(sub, dt_cfg, icnh_cfg)
        dt_idxs = [h for h, src in hits if src == "DT"]
        icnh_idxs = [h for h, src in hits if src == "ICNH"]
        dt_trades = simulate_trades(sub, dt_idxs, dt_cfg, label)
        icnh_trades = simulate_trades(sub, icnh_idxs, icnh_cfg, label)
        trades = dt_trades + icnh_trades
        if not trades:
            out.append({"window": label, "start": start, "end": end,
                        "trades": 0, "win_rate": 0.0, "cum": 0.0,
                        "sharpe": 0.0, "ret_pct": 0.0})
            continue
        nets = np.array([t["net_pct"] for t in trades])
        cum = float(np.prod(1.0 + nets) - 1.0)
        out.append({
            "window": label, "start": start, "end": end,
            "trades": int(len(trades)),
            "win_rate_pct": float((nets > 0).mean() * 100.0),
            "cum": cum,
            "ret_pct": cum * 100.0,
            "sharpe": float(nets.mean() / nets.std() * np.sqrt(250))
                      if nets.std() > 0 else 0.0,
        })
    return out


# ===== Aggregation =====


def _aggregate(per_window: list[dict], windows_filter: set[str]) -> dict:
    """Aggregate per-window stats across a subset (IS or OOS).

    Windows flagged `no_data=True` are excluded — they represent windows where
    the coin wasn't yet listed on Binance Futures, not zero-PnL outcomes.
    Treating them as 0% returns would dilute "positive windows" denominators
    and make short-history coins look more conservative than they are.

    Trades/year and OOS years use the actual window date spans (each H1/H2
    window is 0.5yr) so coins with partial data report a fair frequency.
    """
    in_filter = [w for w in per_window if w.get("window") in windows_filter]
    filtered = [w for w in in_filter if not w.get("no_data")]
    n_no_data = len(in_filter) - len(filtered)
    if not filtered:
        return {"n_windows": 0, "n_no_data_windows": n_no_data,
                "trades": 0, "cum_ret_pct": 0.0,
                "median_sharpe": 0.0, "worst_window_pct": 0.0,
                "positive_windows": 0, "trades_per_year": 0.0,
                "years_covered": 0.0}
    rets = [float(w.get("ret_pct", 0.0)) / 100.0 for w in filtered]
    trades = sum(int(w.get("trades", 0)) for w in filtered)
    cum = float(np.prod([1.0 + r for r in rets]) - 1.0)
    sharpes = [float(w.get("sharpe", 0.0)) for w in filtered]
    worst = min(rets) * 100.0
    positives = sum(1 for r in rets if r > 0)
    # Each H1/H2 window is ~0.5yr; total covered = 0.5 * n active windows.
    years = 0.5 * len(filtered)
    return {
        "n_windows": len(filtered),
        "n_no_data_windows": n_no_data,
        "trades": trades,
        "cum_ret_pct": cum * 100.0,
        "median_sharpe": float(np.median(sharpes)),
        "worst_window_pct": worst,
        "positive_windows": positives,
        "trades_per_year": float(trades / years) if years > 0 else 0.0,
        "years_covered": float(years),
    }


# ===== Reports =====


def _strategy_label(s: str) -> str:
    return {
        "multifactor-v1": "v1 (multifactor)",
        "donchian-v3-cons": "donchian-v3 cons",
        "cnh-hybrid-short-v1": "cnh-hybrid-short",
        "rider-v1": "rider-v1 (4h trend)",
    }.get(s, s)


def _detail_filename(coin: str, strategy: str, ts: str) -> str:
    s = strategy.replace("-", "_")
    return f"cross_coin_{coin}_{s}_{ts}.html"


def _write_detail_html(path: Path, coin: str, strategy: str,
                       per_window: list[dict], summary: dict) -> None:
    """Per-pair detail page: per-window table + IS/OOS aggregates."""
    def _row(w: dict) -> str:
        css = "oos" if w["window"] in OOS_LABELS else "is"
        if w.get("no_data"):
            css += " nodata"
            return (
                f"<tr class='{css}'>"
                f"<td>{w['window']}</td>"
                f"<td>{w.get('start','')}</td>"
                f"<td>{w.get('end','')}</td>"
                f"<td class='num'>—</td>"
                f"<td class='num'>—</td>"
                f"<td class='num'>—</td>"
                f"<td class='num'>—</td>"
                f"<td class='note'>{w.get('note','no data')}</td>"
                f"</tr>"
            )
        return (
            f"<tr class='{css}'>"
            f"<td>{w['window']}</td>"
            f"<td>{w.get('start','')}</td>"
            f"<td>{w.get('end','')}</td>"
            f"<td>{w.get('trades', 0)}</td>"
            f"<td class='num'>{w.get('ret_pct', 0.0):+.2f}%</td>"
            f"<td class='num'>{w.get('sharpe', 0.0):+.2f}</td>"
            f"<td class='num'>{w.get('win_rate_pct', 0.0):.1f}%</td>"
            f"<td class='note'>{w.get('error', w.get('note',''))}</td>"
            f"</tr>"
        )
    rows = "".join(_row(w) for w in per_window)
    is_a = summary.get("IS", {})
    oos_a = summary.get("OOS", {})

    def _agg_row(label: str, agg: dict) -> str:
        n = agg.get("n_windows", 0)
        nodata = agg.get("n_no_data_windows", 0)
        win_text = f"{n} active" + (f" (+ {nodata} no-data)" if nodata else "")
        tpy = agg.get("trades_per_year", 0.0)
        if n == 0:
            return (
                f"<tr><th>{label}</th>"
                f"<td>{win_text}</td>"
                f"<td colspan='5' class='note'>no data in any window</td></tr>"
            )
        return (
            f"<tr><th>{label}</th>"
            f"<td>{win_text}</td>"
            f"<td>{agg.get('trades',0)} trades ({tpy:.1f}/yr)</td>"
            f"<td class='num'>{agg.get('cum_ret_pct',0.0):+.2f}%</td>"
            f"<td class='num'>{agg.get('median_sharpe',0.0):+.2f}</td>"
            f"<td class='num'>{agg.get('worst_window_pct',0.0):+.2f}%</td>"
            f"<td>{agg.get('positive_windows',0)}/{n} positive</td></tr>"
        )

    html = f"""<!doctype html><html><head>
<meta charset="utf-8">
<title>{coin} × {_strategy_label(strategy)} — cross-coin Phase 1</title>
<style>
body {{ font: 14px/1.45 -apple-system, BlinkMacSystemFont, sans-serif;
       max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }}
h1 {{ margin: 0 0 6px; font-size: 22px; }}
.sub {{ color: #666; margin-bottom: 18px; }}
table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }}
th, td {{ padding: 6px 8px; border: 1px solid #ddd; text-align: left; }}
th {{ background: #f6f7f9; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.oos td {{ background: #fffbe6; }}
tr.is td {{ background: #ffffff; }}
tr.nodata td {{ background: #f4f4f5 !important; color: #aaa; }}
.note {{ color: #888; font-size: 12px; }}
.aggs th {{ background: #eaf3ff; }}
a {{ color: #2563eb; text-decoration: none; }}
</style></head><body>
<h1>{coin} × {_strategy_label(strategy)}</h1>
<div class="sub">Cross-coin Phase 1 walk-forward · 8 IS windows + 4 OOS windows ·
<a href="cross_coin_summary_{path.name.split('_')[-1].replace('.html','')}.html">← back to summary</a></div>

<h2>Aggregate metrics</h2>
<table class="aggs">
<tr><th></th><th>Windows</th><th>Trades</th><th>Cum return</th><th>Median Sharpe</th><th>Worst window</th><th>Positive</th></tr>
{_agg_row("IS (windows 1-8)", is_a)}
{_agg_row("<b>OOS (windows 9-12)</b>", oos_a)}
</table>

<h2>Per-window</h2>
<p class="sub">Yellow rows are OOS.</p>
<table>
<tr><th>Window</th><th>Start</th><th>End</th><th>Trades</th><th>Return</th>
    <th>Sharpe</th><th>Win rate</th><th>Note</th></tr>
{rows}
</table>

</body></html>"""
    path.write_text(html, encoding="utf-8")


def _write_summary_html(path: Path, results: list[dict], ts: str) -> None:
    """Ranking summary across all (coin × strategy) pairs."""
    # Rank by OOS cum return × positive windows fraction.
    def _score(r: dict) -> float:
        agg = r.get("summary", {}).get("OOS", {}) or {}
        cum = float(agg.get("cum_ret_pct", 0.0))
        n = int(agg.get("n_windows", 0)) or 1
        pos = int(agg.get("positive_windows", 0))
        return cum * (pos / n)

    ranked = sorted(results, key=_score, reverse=True)

    def _cell_num(agg: dict, key: str, suffix: str = "") -> str:
        if agg.get("n_windows", 0) == 0:
            return "<td class='num nodata'>—</td>"
        return f"<td class='num'>{agg.get(key, 0.0):+.2f}{suffix}</td>"

    rows = []
    for r in ranked:
        coin = r["coin"]
        strat = r["strategy"]
        is_a = r["summary"]["IS"]
        oos_a = r["summary"]["OOS"]
        detail = _detail_filename(coin, strat, ts)
        n_oos = int(oos_a.get("n_windows", 0))
        if n_oos == 0:
            verdict = "n/a (no OOS data)"
        elif (
            float(oos_a.get("cum_ret_pct", 0)) > 0
            and int(oos_a.get("positive_windows", 0)) >= n_oos / 2
            and float(oos_a.get("worst_window_pct", 0)) > -15.0
        ):
            verdict = "✓ promising"
        else:
            verdict = "✗ skip"
        tpy = oos_a.get("trades_per_year", 0.0)
        n_oos_str = (
            f"{oos_a.get('positive_windows', 0)}/{n_oos}"
            + (f" (+{oos_a.get('n_no_data_windows',0)} no-data)"
               if oos_a.get('n_no_data_windows', 0) else "")
        )
        oos_cum_str = (
            "—" if n_oos == 0
            else f"{oos_a.get('cum_ret_pct', 0.0):+.2f}%"
        )
        rows.append(
            f"<tr>"
            f"<td><a href='{detail}'><b>{coin}</b></a></td>"
            f"<td><a href='{detail}'>{_strategy_label(strat)}</a></td>"
            f"{_cell_num(is_a, 'cum_ret_pct', '%')}"
            f"{_cell_num(is_a, 'median_sharpe')}"
            f"<td class='num oos-cum'>{oos_cum_str}</td>"
            f"{_cell_num(oos_a, 'median_sharpe')}"
            f"{_cell_num(oos_a, 'worst_window_pct', '%')}"
            f"<td>{n_oos_str}</td>"
            f"<td class='num'>{oos_a.get('trades', 0)}</td>"
            f"<td class='num'>{tpy:.1f}</td>"
            f"<td>{verdict}</td>"
            f"</tr>"
        )

    html = f"""<!doctype html><html><head>
<meta charset="utf-8">
<title>Cross-coin Phase 1 backtest — summary</title>
<style>
body {{ font: 14px/1.45 -apple-system, BlinkMacSystemFont, sans-serif;
       max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }}
h1 {{ margin: 0 0 6px; font-size: 24px; }}
.sub {{ color: #666; margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }}
th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: left; vertical-align: top; }}
th {{ background: #f6f7f9; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.oos-cum {{ background: #fffbe6; font-weight: 600; }}
.nodata {{ color: #aaa; font-weight: normal; }}
a {{ color: #2563eb; text-decoration: none; }}
.legend {{ font-size: 13px; color: #555; margin-top: 24px; }}
.legend code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
</style></head><body>

<h1>Cross-coin Phase 1 backtest</h1>
<div class="sub">
Generated {datetime.now(UTC).isoformat(timespec='seconds')} ·
BTC-tuned params applied to each coin with no re-tuning ·
ranked by <code>OOS cum × (positive OOS windows / total)</code>
</div>

<table>
<tr>
  <th>Coin</th><th>Strategy</th>
  <th>IS cum</th><th>IS median Sharpe</th>
  <th>OOS cum</th><th>OOS median Sharpe</th>
  <th>OOS worst</th><th>OOS positive</th>
  <th>OOS trades</th><th>Trades/yr</th>
  <th>Verdict</th>
</tr>
{"".join(rows)}
</table>

<div class="legend">
<b>Verdict criteria (Phase 1 gate)</b>: OOS cum > 0  AND  ≥ half of OOS windows
positive  AND  worst OOS window > -15%. Pairs that pass are candidates for
Phase 2-5 deep validation; pairs that fail should not be deployed without
re-tuning the params for that coin.<br><br>
<b>About "—" cells and "(+ N no-data)"</b>: those windows are
periods where the coin wasn't yet listed on Binance Futures. They're
excluded from aggregates (instead of being counted as zero-PnL windows)
so short-history coins aren't artificially diluted.<br><br>
<b>About trade count</b>: low-vol coins (gold, ADA) fire fewer patterns
because the BTC-tuned <code>dedup_bars=15</code> + <code>min_rise_pct=2.5%</code> +
<code>sl_atr_mult=1.5×ATR</code> thresholds assume crypto-scale volatility. To get
more trades per year on a low-vol coin, you'd need to re-tune those knobs
per coin (Phase 2). The numbers shown here use BTC's locked params on
every coin so the cross-coin comparison is apples-to-apples.<br><br>
<b>Note on WLD</b>: only listed Jul-2023, so it has 0 IS windows on the
2020-H2..2024-H1 IS bucket. Its OOS row is the only signal that matters.<br><br>
<b>Note on PAXG (gold)</b>: only listed Mar-2025, so 8 of 12 windows are
no-data. The +87% donchian result is concentrated in gold's 2025-H2 →
2026-H1 parabolic rally; small sample + single-regime, do not extrapolate.<br><br>
<b>Note on multifactor-v1</b>: uses funding-rate data. If a coin's funding history
is sparse, v1 may run with degraded signals.
</div>

</body></html>"""
    path.write_text(html, encoding="utf-8")


# ===== Driver =====


def main() -> int:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"Cross-coin backtest start ({ts})", flush=True)

    strategies = [
        ("multifactor-v1", "15m", None),
        ("donchian-v3-cons", "4h", _donchian_cons_params()),
        ("cnh-hybrid-short-v1", None, None),  # custom runner
        ("rider-v1", "4h", _rider_v1_params()),
    ]

    results: list[dict] = []
    for coin in COINS:
        for strategy, timeframe, params in strategies:
            t0 = time.monotonic()
            print(f"\n[{coin} × {strategy}] running…", flush=True)
            try:
                if strategy == "cnh-hybrid-short-v1":
                    per_window = _run_cnh_hybrid_per_window(coin)
                else:
                    # Map the display strategy label to the backtest.py key.
                    bt_name_map = {
                        "multifactor-v1": "multifactor-v1",
                        "donchian-v3-cons": "donchian-v3",
                        "rider-v1": "rider-v1",
                    }
                    bt_name = bt_name_map.get(strategy, strategy)
                    per_window = _run_backtest_per_window(
                        coin, bt_name, timeframe, params_override=params,
                    )
                summary = {
                    "IS": _aggregate(per_window, IS_LABELS),
                    "OOS": _aggregate(per_window, OOS_LABELS),
                }
                elapsed = time.monotonic() - t0
                print(
                    f"[{coin} × {strategy}] done in {elapsed:.1f}s — "
                    f"OOS cum={summary['OOS']['cum_ret_pct']:+.2f}% "
                    f"({summary['OOS']['positive_windows']}/"
                    f"{summary['OOS']['n_windows']} positive)",
                    flush=True,
                )
                results.append({
                    "coin": coin, "strategy": strategy,
                    "per_window": per_window, "summary": summary,
                })
                # Write detail page now so a crash later still leaves something.
                _write_detail_html(
                    REPORTS_DIR / _detail_filename(coin, strategy, ts),
                    coin, strategy, per_window, summary,
                )
            except Exception as e:
                print(f"[{coin} × {strategy}] FAILED: {e}", flush=True)
                traceback.print_exc()
                results.append({
                    "coin": coin, "strategy": strategy,
                    "error": f"{type(e).__name__}: {e}",
                    "summary": {"IS": {}, "OOS": {}},
                    "per_window": [],
                })

    # Persist raw JSON
    raw_path = REPORTS_DIR / f"cross_coin_results_{ts}.json"
    raw_path.write_text(json.dumps(results, indent=2, default=str),
                        encoding="utf-8")

    summary_path = REPORTS_DIR / f"cross_coin_summary_{ts}.html"
    _write_summary_html(summary_path, results, ts)

    print(f"\n=== reports ===")
    print(f"  raw JSON : {raw_path.relative_to(ROOT)}")
    print(f"  summary  : {summary_path.relative_to(ROOT)}")
    print(f"  detail x : reports/cross_coin_<COIN>_<STRATEGY>_{ts}.html ({len(results)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
