"""AdaptiveTrend multi-asset portfolio layer (BTC+ETH+SOL) — load-bearing test.

Per ADAPTIVE_TREND_V2_VERDICT.md operator playbook #1: the paper's Sharpe lift
(arXiv 2602.11708, 1.34 -> 2.41) comes from the cross-sectional selection +
allocation layer, not the single-asset monthly re-opt. This runner tests that
architecture on the 3-asset universe we have data for:

  - per-asset Algorithm-2 monthly re-opt of (L, theta) on trailing 6mo H6
    (reuses strategy.signals_adaptive_trend_v2._simulate_h6_fit)
  - post-fit selection gates on ANNUALIZED fit Sharpe: long-eligible if
    >= GAMMA_L (1.3), short-eligible if >= GAMMA_S (1.7)  [verdict's reading
    of the paper's gamma gates; annualization = per-trade * sqrt(trades/yr)]
  - allocation: 70% budget to the long book / 30% to the short book,
    inverse-vol weights within each book (trailing 120 H6-bar return std),
    empty book leaves its budget in cash (conservative)
  - correlation overlay: total gross exposure scaled by
    CF = sqrt(N / (N + N*(N-1)*rho_bar)), rho_bar = trailing mean pairwise
    corr of H6 returns (120 bars)
  - costs: 10 bps of turnover notional per weight change + real 8h funding
  - granularity: H6 bar replay (same shortcut as the V2 fit sim, applied
    IDENTICALLY to the portfolio arm and the BTC-solo baseline arm, so the
    lift comparison is apples-to-apples; this is an architecture verdict,
    not a live-PnL estimate)

Verdict rule (from the operator playbook): if the portfolio arm does not lift
Sharpe over BTC-solo under the identical harness, the architecture does not
transfer to this universe -> SHELF Track B.

Outputs reports/adaptrend_multiasset_port.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy.indicators import atr as wilder_atr  # noqa: E402
from strategy.signals_adaptive_trend import _resample_h6  # noqa: E402
from strategy.signals_adaptive_trend_v2 import (  # noqa: E402
    _per_trade_sharpe,
    _simulate_h6_fit,
)

ASSETS = ("BTC", "ETH", "SOL")
STUDY_START = pd.Timestamp("2021-07-01")   # SOL has >=6mo fit history by then
STUDY_END = pd.Timestamp("2026-05-27")
GRID_L = (3, 4, 5, 6)
GRID_TH = (0.015, 0.02, 0.025)
ALPHA = 2.0
ATR_P = 14
FIT_MONTHS = 6
MIN_FIT_TRADES = 20
GAMMA_L, GAMMA_S = 1.3, 1.7
VOL_WIN = 120                  # H6 bars (~30 days)
COST_BPS = 10.0                # round-trip, charged on turnover
LONG_BUDGET, SHORT_BUDGET = 0.7, 0.3


def load_h6(coin: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(ROOT / "data" / "historical" / f"{coin}_USDT_USDT_15m.parquet")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    h6 = _resample_h6(df[["Open", "High", "Low", "Close"]])
    fund = pd.read_parquet(ROOT / "data" / "historical" / f"{coin}_USDT_USDT_funding.parquet")
    if fund.index.tz is not None:
        fund.index = fund.index.tz_localize(None)
    return h6, fund["funding_rate"]


def replay_h6_bar_returns(h6: pd.DataFrame, L: int, theta: float) -> pd.Series:
    """Replay the rule from flat on an H6 slice; return per-bar STRATEGY returns.

    Same entry/exit rules as _simulate_h6_fit but emitting a bar-return stream
    (needed for time-series Sharpe and for the portfolio combination) with
    intra-bar trail exits honored: when the bar's low/high pierces the trail,
    the bar's return is cut at the trail price instead of the close.
    Entries fill at the bar close (no return on the entry bar).
    """
    close = h6["Close"].to_numpy()
    high = h6["High"].to_numpy()
    low = h6["Low"].to_numpy()
    atr = wilder_atr(h6["High"], h6["Low"], h6["Close"], period=ATR_P).to_numpy()
    mom = ((h6["Close"] - h6["Close"].shift(L)) / h6["Close"].shift(L)).to_numpy()

    ret = np.zeros(len(h6))
    p, trail = 0, 0.0
    for i in range(1, len(h6)):
        m_v, a_v, c = mom[i - 1], atr[i - 1], close[i]
        if p == 0:
            if np.isfinite(m_v) and np.isfinite(a_v) and a_v > 0:
                if m_v > theta:
                    p, trail = 1, c - ALPHA * a_v
                elif m_v < -theta:
                    p, trail = -1, c + ALPHA * a_v
            continue
        if p > 0:
            # test the intra-bar pierce against the PRIOR trail (the level that
            # actually existed during the bar), THEN ratchet with this close
            if low[i] <= trail:
                ret[i] = trail / close[i - 1] - 1.0
                p = 0
            else:
                if np.isfinite(a_v) and a_v > 0:
                    trail = max(trail, c - ALPHA * a_v)
                ret[i] = c / close[i - 1] - 1.0
                if c < trail:
                    p = 0
        else:
            if high[i] >= trail:
                ret[i] = 1.0 - trail / close[i - 1]
                p = 0
            else:
                if np.isfinite(a_v) and a_v > 0:
                    trail = min(trail, c + ALPHA * a_v)
                ret[i] = 1.0 - c / close[i - 1]
                if c > trail:
                    p = 0
    return pd.Series(ret, index=h6.index)


H6_BARS_PER_YEAR = 4 * 365


def monthly_fits(h6: pd.DataFrame) -> pd.DataFrame:
    """Algorithm-2 per month: (L*, theta*), ranked by per-trade Sharpe (repo
    convention), plus the winner's TIME-SERIES annualized Sharpe on the fit
    window (paper convention — used for the gamma selection gates)."""
    months = pd.date_range(h6.index.min().normalize().replace(day=1),
                           h6.index.max(), freq="MS")
    rows = []
    for m in months:
        fit = h6.loc[(h6.index >= m - pd.DateOffset(months=FIT_MONTHS)) & (h6.index < m)]
        if len(fit) < max(GRID_L) + ATR_P + 20:
            continue
        best, best_sr, best_n = None, float("-inf"), 0
        for L in GRID_L:
            for th in GRID_TH:
                rets = _simulate_h6_fit(fit, L=L, theta=th, alpha=ALPHA, atr_period=ATR_P)
                if len(rets) < MIN_FIT_TRADES:
                    continue
                sr = _per_trade_sharpe(rets)
                if sr > best_sr:
                    best, best_sr, best_n = (L, th), sr, len(rets)
        if best is None:
            rows.append((m, np.nan, np.nan, np.nan))
            continue
        br = replay_h6_bar_returns(fit, best[0], best[1])
        ts_sharpe = (br.mean() / br.std() * np.sqrt(H6_BARS_PER_YEAR)
                     if br.std() > 0 else float("-inf"))
        rows.append((m, best[0], best[1], float(ts_sharpe)))
    out = pd.DataFrame(rows, columns=["month", "L", "theta", "fit_sharpe_ann"])
    return out.set_index("month")


def position_stream(h6: pd.DataFrame, fits: pd.DataFrame,
                    gate: bool) -> pd.DataFrame:
    """Replay entries/exits at H6 bars under monthly (L, theta).

    Returns a frame with:
      pos     — signed position held INTO each bar's return interval
      leg_ret — the leg's own bar return (intra-bar trail exits honored:
                the bar's return is cut at the trail price when pierced)

    gate=True applies the gamma eligibility gates from fit_sharpe_ann;
    gate=False trades every signal (solo baseline arms). Position carries
    across month boundaries; params/gates affect new entries only.
    """
    close = h6["Close"].to_numpy()
    high = h6["High"].to_numpy()
    low = h6["Low"].to_numpy()
    idx = h6.index
    atr = wilder_atr(h6["High"], h6["Low"], h6["Close"], period=ATR_P).to_numpy()

    # precompute MOM for each grid L once
    mom_by_L = {L: ((h6["Close"] - h6["Close"].shift(L)) / h6["Close"].shift(L)).to_numpy()
                for L in GRID_L}

    pos = np.zeros(len(h6))
    leg_ret = np.zeros(len(h6))
    p = 0
    trail = 0.0
    cur = None       # (L, theta, long_ok, short_ok)
    month_seen = None
    for i in range(1, len(h6)):
        ts = idx[i]
        mk = (ts.year, ts.month)
        if mk != month_seen:
            month_seen = mk
            m = pd.Timestamp(ts.year, ts.month, 1)
            if m in fits.index and np.isfinite(fits.loc[m, "L"]):
                f = fits.loc[m]
                long_ok = (not gate) or f["fit_sharpe_ann"] >= GAMMA_L
                short_ok = (not gate) or f["fit_sharpe_ann"] >= GAMMA_S
                cur = (int(f["L"]), float(f["theta"]), long_ok, short_ok)
            # else: keep prior params (fallback), or stay flat if none yet
        pos[i] = p                      # held into this bar
        if cur is None:
            continue
        L, th, long_ok, short_ok = cur
        m_v = mom_by_L[L][i - 1]        # just-closed bar
        a_v = atr[i - 1]
        c = close[i]

        if p == 0:
            if np.isfinite(m_v) and np.isfinite(a_v) and a_v > 0:
                if m_v > th and long_ok:
                    p, trail = 1, c - ALPHA * a_v      # enter at close
                elif m_v < -th and short_ok:
                    p, trail = -1, c + ALPHA * a_v
            continue

        # in a position through this bar: bar return with intra-bar exit.
        # Pierce is tested against the PRIOR trail (the level that existed
        # during the bar); the ratchet with this bar's close happens after.
        if p > 0:
            if low[i] <= trail:
                leg_ret[i] = trail / close[i - 1] - 1.0
                p = 0
            else:
                if np.isfinite(a_v) and a_v > 0:
                    trail = max(trail, c - ALPHA * a_v)
                leg_ret[i] = c / close[i - 1] - 1.0
                if c < trail:
                    p = 0
        else:
            if high[i] >= trail:
                leg_ret[i] = 1.0 - trail / close[i - 1]
                p = 0
            else:
                if np.isfinite(a_v) and a_v > 0:
                    trail = min(trail, c + ALPHA * a_v)
                leg_ret[i] = 1.0 - c / close[i - 1]
                if c > trail:
                    p = 0
    return pd.DataFrame({"pos": pos, "leg_ret": leg_ret}, index=idx)


def portfolio_returns(streams: dict, h6map: dict, fundmap: dict,
                      single: str | None = None) -> pd.Series:
    """Combine per-asset leg-return streams into net H6 portfolio bar returns.

    Leg returns already carry direction and intra-bar trail exits; the
    portfolio layer assigns CAPITAL WEIGHTS to active legs:
      single=<coin>: baseline arm — that coin alone at weight 1.0.
      otherwise: 70/30 long-short book budgets, inverse-vol within book,
                 CF = sqrt(N/(N+N(N-1)rho_bar)) overlay on total exposure.
    Costs: COST_BPS on signed-exposure turnover; real 8h funding on exposure.
    """
    common = None
    for c in streams:
        common = streams[c].index if common is None else common.union(streams[c].index)
    common = common[(common >= STUDY_START) & (common <= STUDY_END)]

    asset_ret = {c: h6map[c]["Close"].pct_change().reindex(common) for c in streams}
    leg_ret = {c: streams[c]["leg_ret"].reindex(common).fillna(0.0) for c in streams}
    pos = {c: streams[c]["pos"].reindex(common).fillna(0.0) for c in streams}
    vol = {c: asset_ret[c].rolling(VOL_WIN).std().shift(1) for c in streams}

    weights = pd.DataFrame(0.0, index=common, columns=list(streams))
    if single is not None:
        weights[single] = (pos[single] != 0).astype(float)
    else:
        # trailing mean pairwise correlation for CF
        retdf = pd.DataFrame(asset_ret)
        pairs = [(a, b) for i, a in enumerate(ASSETS) for b in ASSETS[i + 1:]]
        rhos = [retdf[a].rolling(VOL_WIN).corr(retdf[b]) for a, b in pairs]
        rho_bar = pd.concat(rhos, axis=1).mean(axis=1).shift(1).clip(0, 1)

        pos_l = pd.DataFrame({c: (pos[c] > 0).astype(float) for c in streams})
        pos_s = pd.DataFrame({c: (pos[c] < 0).astype(float) for c in streams})
        iv = pd.DataFrame({c: 1.0 / vol[c] for c in streams}).replace([np.inf, -np.inf], np.nan)
        for book, budget in ((pos_l, LONG_BUDGET), (pos_s, SHORT_BUDGET)):
            act = book * iv
            norm = act.div(act.sum(axis=1), axis=0).fillna(0.0)
            weights += budget * norm            # capital weight, direction lives in leg_ret
        n_active = (weights != 0).sum(axis=1)
        cf = np.sqrt(n_active / (n_active + n_active * (n_active - 1) * rho_bar))
        weights = weights.mul(cf.fillna(1.0), axis=0)

    gross = (weights * pd.DataFrame(leg_ret)).sum(axis=1)
    exposure = weights * pd.DataFrame({c: np.sign(pos[c]) for c in streams})
    turnover = exposure.diff().abs().sum(axis=1)
    cost = turnover * (COST_BPS / 1e4)
    # funding: charge signed exposure * funding_rate at each 8h funding ts
    fund_cost = pd.Series(0.0, index=common)
    for c in streams:
        f6 = fundmap[c].resample("6h", label="right", closed="right").sum()
        f = f6.reindex(common).fillna(0.0)
        fund_cost = fund_cost.add((exposure[c] * f).fillna(0.0), fill_value=0.0)
    net = gross - cost - fund_cost
    return net.dropna()


def summarize(net: pd.Series) -> dict:
    ann = np.sqrt(4 * 365)
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = (net.index.max() - net.index.min()).days / 365.25
    halves = net.groupby([net.index.year, net.index.map(lambda t: 1 if t.month <= 6 else 2)])
    half_tab = {f"{y}H{h}": round(((1 + g).prod() - 1) * 100, 2) for (y, h), g in halves}
    quarters = net.groupby(pd.PeriodIndex(net.index, freq="Q"))
    q_rets = {str(q): ((1 + g).prod() - 1) * 100 for q, g in quarters}
    q_pos = sum(1 for v in q_rets.values() if v > 0)
    return {
        "sharpe_ann": round(float(net.mean() / net.std() * ann), 3),
        "comp_pct": round(float((eq.iloc[-1] - 1) * 100), 2),
        "cagr_pct": round(float((eq.iloc[-1] ** (1 / yrs) - 1) * 100), 2),
        "max_dd_pct": round(float(dd * 100), 2),
        "quarters_positive": f"{q_pos}/{len(q_rets)} = {q_pos / len(q_rets) * 100:.1f}%",
        "half_year_returns_pct": half_tab,
    }


def main() -> int:
    h6map, fundmap, fitmap = {}, {}, {}
    for c in ASSETS:
        print(f"[port] loading {c} ...", file=sys.stderr)
        h6map[c], fundmap[c] = load_h6(c)
        print(f"[port] monthly fits {c} ...", file=sys.stderr)
        fitmap[c] = monthly_fits(h6map[c])

    streams_gated = {c: position_stream(h6map[c], fitmap[c], gate=True) for c in ASSETS}
    streams_all = {c: position_stream(h6map[c], fitmap[c], gate=False) for c in ASSETS}

    out = {"study": f"{STUDY_START.date()} .. {STUDY_END.date()}",
           "cost_bps_turnover": COST_BPS, "arms": {}}
    out["arms"]["portfolio_gated"] = summarize(
        portfolio_returns(streams_gated, h6map, fundmap))
    out["arms"]["portfolio_nogate"] = summarize(
        portfolio_returns(streams_all, h6map, fundmap))
    out["arms"]["btc_solo_baseline"] = summarize(
        portfolio_returns(streams_all, h6map, fundmap, single="BTC"))
    for c in ("ETH", "SOL"):
        out["arms"][f"{c.lower()}_solo"] = summarize(
            portfolio_returns(streams_all, h6map, fundmap, single=c))

    path = ROOT / "reports" / "adaptrend_multiasset_port.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[port] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
