"""
Backtest harness with realistic friction modeling.

Phase 1 added a buy-and-hold benchmark with fees + slippage + funding.
Phase 2 adds the real snapback-v1 strategy (RSI(2) + EMA(200) + volume +
funding confluence, multi-timeframe).

CLI:
    # benchmark
    python backtest.py --strategy buy-and-hold --tf 1h --days 30

    # strategy v1
    python backtest.py --strategy snapback-v1 --tf 15m --start 2024-01-01 --end 2025-01-01

Honest reporting: naive B&H / after fees+slip / after funding shown
separately, so friction is never hidden in a headline number.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest

from exchange.data import load_funding, load_klines
from strategy.signals import (
    FRACTIONAL_UNIT,
    SnapbackBTC,
    StrategyParams,
    prepare_strategy_data,
)

# 2026-05-17: cleaned up to only the deployable strategy + benchmark.
# Old strategies (carry, donchian, fmom, snapback-v2, multifactor v2/mtf)
# removed from this repo after PATH2_RESULTS.html locked multifactor-v1
# as the production choice. Recover from git history if needed.
from strategy.signals_multifactor import DayTradeMultiFactorBTC
from strategy.signals_multifactor_v2 import (
    DayTradeMultiFactorBTCv2Loose,
    DayTradeMultiFactorBTCv2Strict,
)
from strategy.signals_multifactor_v3 import (
    DayTradeMultiFactorBTCv3,
    V3All,
    V3AllK_2_5,
    V3AllK_3_5,
    V3AllK_5_0,
    V3AllK_6_0,
    V3AllWider2,
    V3AllWider3,
    V3AllWider4,
    V3AtrStopsOnly,
    V3DistEmaOnly,
    V3VolRegimeOnly,
)
from strategy.signals_multifactor_tuned import (
    V1Debounce2,
    V1Debounce3,
    V1Debounce4,
    V1Deluxe,
    V1Floor005,
    V1Floor010,
)
# Resurrected 2026-05-23 for v1+Donchian parallel-deploy backtest.
from strategy.signals_donchian import (
    DonchianBreakoutBTCv3, DonchianRiderV1,
    attach_donchian, attach_rider,
)
# 2026-06-14: native-4h Supertrend trend-follower for bake-off vs donchian-v3.
from strategy.signals_supertrend import SupertrendBTC, attach_supertrend
# 2026-06-14: Supertrend variants for walk-forward bake-off (experiments/walkforward.py).
from strategy.signals_supertrend_variants import (
    SupertrendADX,
    SupertrendADXDonchExit,
    SupertrendDonchExit,
    SupertrendDual,
    SupertrendEMA,
    SupertrendTrail,
    SupertrendVolAdaptive,
    attach_supertrend_adx,
    attach_supertrend_adx_donchexit,
    attach_supertrend_donchexit,
    attach_supertrend_dual,
    attach_supertrend_ema,
    attach_supertrend_trail,
    attach_supertrend_voladapt,
)

# multifactor-v1/v2/v3 use a single entry TF (15m); no second-TF prep required.
# Donchian-v3 and rider-v1 are single-TF (entry_tf == channel TF, typically 4h).
_TF_AGNOSTIC_STRATEGIES = {
    "multifactor-v1",
    "multifactor-v2-loose", "multifactor-v2-strict",
    "multifactor-v3", "v3-dist-ema-only", "v3-vol-regime-only",
    "v3-atr-stops-only", "v3-all",
    "donchian-v3",
    "rider-v1",
    "supertrend",
    "st-adx", "st-ema", "st-dual", "st-donchexit", "st-voladapt",
    "st-adx-donchexit", "st-trail",
}

# For snapback, use plain Backtest with large notional cash so 1 BTC fits as
# an integer unit. Returns are scale-invariant so headline metrics are
# unchanged vs running with $10k.
SNAPBACK_DEFAULT_CASH = 1_000_000.0

log = logging.getLogger(__name__)

# --- Friction model ----------------------------------------------------------
# Binance Futures taker fee = 0.04% per side. We bundle a 1bp slippage proxy
# into the same `commission` parameter since backtesting.py has no native
# slippage knob. Total per-side: 0.05%.
TAKER_FEE = 0.0004
SLIPPAGE_PROXY = 0.0001
COMMISSION_PER_SIDE = TAKER_FEE + SLIPPAGE_PROXY


# --- Strategies --------------------------------------------------------------
class BuyAndHold(Strategy):
    """Sanity benchmark — buy on first bar, hold forever."""

    def init(self) -> None:
        self._opened = False

    def next(self) -> None:
        if not self._opened:
            # size=1.0 fails the broker margin check after commission; 0.999
            # leaves a 0.1% cash buffer that's effectively zero capital drag.
            self.buy(size=0.999)
            self._opened = True


STRATEGIES: dict[str, type[Strategy]] = {
    "buy-and-hold": BuyAndHold,
    "snapback-v1": SnapbackBTC,
    "multifactor-v1": DayTradeMultiFactorBTC,
    "multifactor-v2-loose": DayTradeMultiFactorBTCv2Loose,
    "multifactor-v2-strict": DayTradeMultiFactorBTCv2Strict,
    "multifactor-v3": DayTradeMultiFactorBTCv3,
    "v3-dist-ema-only": V3DistEmaOnly,
    "v3-vol-regime-only": V3VolRegimeOnly,
    "v3-atr-stops-only": V3AtrStopsOnly,
    "v3-all": V3All,
    "v3-all-k-2.5": V3AllK_2_5,
    "v3-all-wider-2": V3AllWider2,
    "v3-all-k-3.5": V3AllK_3_5,
    "v3-all-wider-3": V3AllWider3,
    "v3-all-wider-4": V3AllWider4,
    "v3-all-k-5.0": V3AllK_5_0,
    "v3-all-k-6.0": V3AllK_6_0,
    # multifactor-v1 trend-exit ablation variants
    "v1-debounce-2": V1Debounce2,
    "v1-debounce-3": V1Debounce3,
    "v1-debounce-4": V1Debounce4,
    "v1-floor-0.5": V1Floor005,
    "v1-floor-1.0": V1Floor010,
    "v1-deluxe": V1Deluxe,
    # Resurrected 2026-05-23.
    "donchian-v3": DonchianBreakoutBTCv3,
    # 2026-05-30: 4h native trend-rider (long-only, fixed TP bracket).
    "rider-v1": DonchianRiderV1,
    # 2026-06-14: 4h native Supertrend trend-follower (long + short, ATR bracket).
    "supertrend": SupertrendBTC,
    # 2026-06-14: Supertrend variants for walk-forward bake-off.
    "st-adx": SupertrendADX,
    "st-ema": SupertrendEMA,
    "st-dual": SupertrendDual,
    "st-donchexit": SupertrendDonchExit,
    "st-voladapt": SupertrendVolAdaptive,
    # 2026-06-14 round 2: ADX+Donchian-exit combo + chandelier trailing stop.
    "st-adx-donchexit": SupertrendADXDonchExit,
    "st-trail": SupertrendTrail,
}

# No strategy in the current codebase needs regime columns.
_REGIME_STRATEGIES: set[str] = set()
# Donchian-v3 needs DonchianUpper/Lower/ExitUpper/ExitLower + ATR_1h columns
# attached to the entry-TF frame by attach_donchian().
_DONCHIAN_STRATEGIES: set[str] = {"donchian-v3"}
# rider-v1 needs native-4h RiderDonHi/RiderEma/RiderAtr attached by attach_rider().
_RIDER_STRATEGIES: set[str] = {"rider-v1"}
# supertrend (and variants) need native-4h STLine/STDir/STAtr attached by an
# attach_* function. _SUPERTREND_ATTACH_FNS maps each key to its attach
# function and the set of extra class-attr-derived kwargs it needs (beyond
# the shared period/multiplier/atr_period).
_SUPERTREND_STRATEGIES: set[str] = {
    "supertrend", "st-adx", "st-ema", "st-dual", "st-donchexit", "st-voladapt",
    "st-adx-donchexit", "st-trail",
}
_SUPERTREND_ATTACH_FNS: dict[str, tuple] = {
    "supertrend": (attach_supertrend, {}),
    "st-adx": (attach_supertrend_adx, {"adx_period": "st_adx_period"}),
    "st-ema": (attach_supertrend_ema, {"ema_period": "st_ema_period"}),
    "st-dual": (attach_supertrend_dual, {
        "slow_period": "st_slow_period", "slow_multiplier": "st_slow_multiplier",
    }),
    "st-donchexit": (attach_supertrend_donchexit, {"donch_period": "st_donch_period"}),
    "st-voladapt": (attach_supertrend_voladapt, {
        "multiplier_low": "st_mult_low", "multiplier_high": "st_mult_high",
        "vol_lookback": "st_vol_lookback",
    }),
    "st-adx-donchexit": (attach_supertrend_adx_donchexit, {
        "adx_period": "st_adx_period", "donch_period": "st_donch_period",
    }),
    "st-trail": (attach_supertrend_trail, {"adx_period": "st_adx_period"}),
}


# --- Funding accounting ------------------------------------------------------
def funding_cost_for_long_btc(
    data: pd.DataFrame,
    funding: pd.DataFrame,
    initial_cash: float,
    commission: float,
) -> tuple[float, int]:
    """Sum funding payments for a long buy-and-hold position over `data`'s span."""
    if funding.empty or data.empty:
        return 0.0, 0

    first_open = float(data["Open"].iloc[0])
    btc_position = initial_cash * (1.0 - commission) / first_open

    span = funding.loc[
        (funding.index >= data.index[0]) & (funding.index <= data.index[-1])
    ]
    if span.empty:
        return 0.0, 0

    prices = data["Close"].reindex(span.index, method="ffill")
    notional = btc_position * prices
    paid = (notional * span["funding_rate"]).sum()
    return float(paid), len(span)


def funding_cost_for_trades(
    trades: pd.DataFrame, data: pd.DataFrame, funding: pd.DataFrame
) -> tuple[float, int]:
    """Sum funding payments across per-trade open intervals.

    backtesting.py 0.6 Size is signed: + for long, - for short. Notional with
    sign × funding rate gives the correct sign for cost (long pays positive
    funding, short receives positive funding).
    """
    if trades is None or trades.empty or funding is None or funding.empty:
        return 0.0, 0

    total = 0.0
    events = 0
    closes = data["Close"]

    for _, t in trades.iterrows():
        entry_time = t["EntryTime"] if "EntryTime" in t else t.get("EntryTime")
        exit_time = t["ExitTime"] if "ExitTime" in t else t.get("ExitTime")
        size = float(t["Size"])  # signed
        if entry_time is None or exit_time is None or pd.isna(entry_time) or pd.isna(exit_time):
            continue
        span = funding.loc[(funding.index >= entry_time) & (funding.index <= exit_time)]
        if span.empty:
            continue
        prices = closes.reindex(span.index, method="ffill")
        notional = size * prices  # carries sign
        total += float((notional * span["funding_rate"]).sum())
        events += len(span)

    return total, events


# --- Runner ------------------------------------------------------------------
def _prepare_buy_and_hold_data(
    symbol: str, timeframe: str, start: datetime, end: datetime
) -> pd.DataFrame:
    days_back = max((end - start).days + 2, 2)
    raw = load_klines(symbol=symbol, timeframe=timeframe, days_back=days_back, end=end)
    df = raw.loc[start:end].copy()
    if df.empty:
        raise RuntimeError(
            f"No klines for {symbol} {timeframe} in {start.date()} → {end.date()}"
        )
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _prepare_snapback_data(
    symbol: str,
    start: datetime,
    end: datetime,
    params: StrategyParams,
    with_regimes: bool = False,
    with_donchian: bool = False,
    donchian_entry: int = 20,
    donchian_exit: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull 15m + 1h + funding; build indicator-augmented 15m DataFrame.

    Returns (prepared_15m_df, funding_in_span_df). Adds a warm-up buffer so
    indicators are valid by the start of the visible window.
    """
    # Warm-up needed: ema_period 1h bars to start the EMA = ema_period hours.
    # Add 50% slack.
    warmup = timedelta(hours=int(params.ema_period * 1.5))
    pull_start = start - warmup
    days_back = max((end - pull_start).days + 2, 2)

    k15 = load_klines(symbol=symbol, timeframe="15m", days_back=days_back, end=end)
    k1h = load_klines(symbol=symbol, timeframe="1h", days_back=days_back, end=end)
    fund = load_funding(symbol=symbol, days_back=days_back, end=end)

    if k15.empty or k1h.empty:
        raise RuntimeError("Missing 15m or 1h klines for the requested window.")

    prepared = prepare_strategy_data(k15, k1h, fund, params)
    # NOTE: old regime/Donchian attach steps removed with the archived strategies.
    _ = with_regimes  # parameter kept for backward-compat callers
    _ = with_donchian
    _ = donchian_entry
    _ = donchian_exit
    # prepare_strategy_data strips tz; slice bounds + funding must match.
    naive_start = start.replace(tzinfo=None) if start.tzinfo else start
    naive_end = end.replace(tzinfo=None) if end.tzinfo else end
    visible = prepared.loc[naive_start:naive_end]
    if visible.empty:
        raise RuntimeError(f"No bars after warm-up in {start} → {end}")

    if fund.index.tz is not None:
        fund = fund.copy()
        fund.index = fund.index.tz_convert("UTC").tz_localize(None)
    fund_visible = fund.loc[visible.index[0]:visible.index[-1]]

    return visible, fund_visible


def _prepare_tf_agnostic_data(
    symbol: str,
    entry_tf: str,
    start: datetime,
    end: datetime,
    with_donchian: bool = False,
    donchian_entry: int = 20,
    donchian_exit: int = 10,
    atr_period: int = 20,
    with_rider: bool = False,
    rider_donchian_n: int = 55,
    rider_atr_period: int = 14,
    rider_ema_period: int = 200,
    with_supertrend: bool = False,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
    supertrend_strategy_name: str = "supertrend",
    supertrend_extra_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull klines at entry_tf + funding; attach Donchian channels (computed
    on the SAME entry_tf — single-TF Donchian, not multi-TF) if requested.
    For rider-v1 (with_rider=True), attaches native-4h RiderDonHi/RiderEma/RiderAtr.
    For supertrend (with_supertrend=True), attaches native-4h STLine/STDir/STAtr.

    Returns (prepared_df, funding_in_span). Used for carry + donchian + rider
    backtests when entry_tf != 15m, where snapback's 15m+1h prep doesn't
    apply.
    """
    # Warm-up buffer: enough bars for the longest indicator window.
    # rider-v1 needs EMA200 (binding constraint = 200 bars * 3 slack).
    # donchian_entry bars at entry_tf is the standard constraint.
    if with_rider:
        # rider-v1: compute indicators on exactly the visible window (no pre-roll).
        # This matches the standalone validator (rider_port_validate) which slices
        # first then computes EMA/DonHi/ATR on the slice.  Pre-rolling would warm
        # EMA200 earlier than the standalone does, adding spurious early-window
        # trades and inflating trade counts / WR.  NaN bars at the head are skipped
        # by the isfinite guard in DonchianRiderV1.next().
        warmup_bars = 0
    elif with_supertrend:
        warmup_bars = max(supertrend_period, atr_period) * 3
    else:
        warmup_bars = max(donchian_entry, atr_period) * 3
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                  "1h": 60, "2h": 120, "4h": 240, "1d": 1440}.get(entry_tf, 15)
    warmup = timedelta(minutes=warmup_bars * tf_minutes)
    pull_start = start - warmup
    days_back = max((end - pull_start).days + 2, 2)

    klines = load_klines(symbol=symbol, timeframe=entry_tf, days_back=days_back, end=end)
    fund = load_funding(symbol=symbol, days_back=days_back, end=end)
    if klines.empty:
        raise RuntimeError(f"Missing {entry_tf} klines for the requested window.")

    # 2026-05-23: re-enabled for donchian-v3 resurrection.
    klines.columns = [c.capitalize() for c in klines.columns]
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    prepared = klines
    if with_donchian:
        prepared = attach_donchian(
            prepared, klines,
            period_entry=donchian_entry,
            period_exit=donchian_exit,
            atr_period=atr_period,
        )
    if with_rider:
        prepared = attach_rider(
            prepared,
            donchian_n=rider_donchian_n,
            atr_period=rider_atr_period,
            ema_period=rider_ema_period,
        )
    if with_supertrend:
        attach_fn, _ = _SUPERTREND_ATTACH_FNS[supertrend_strategy_name]
        extra = dict(supertrend_extra_kwargs or {})
        if supertrend_strategy_name == "st-voladapt":
            # attach_supertrend_voladapt has no single `multiplier` kwarg —
            # it takes multiplier_low/multiplier_high (supplied via extra).
            prepared = attach_fn(
                prepared,
                period=supertrend_period,
                atr_period=atr_period,
                **extra,
            )
        else:
            prepared = attach_fn(
                prepared,
                period=supertrend_period,
                multiplier=supertrend_multiplier,
                atr_period=atr_period,
                **extra,
            )

    naive_start = start.replace(tzinfo=None) if start.tzinfo else start
    naive_end = end.replace(tzinfo=None) if end.tzinfo else end
    visible = prepared.loc[naive_start:naive_end]
    if visible.empty:
        raise RuntimeError(f"No bars after warm-up in {start} → {end}")

    if fund.index.tz is not None:
        fund = fund.copy()
        fund.index = fund.index.tz_convert("UTC").tz_localize(None)
    fund_visible = fund.loc[visible.index[0]:visible.index[-1]]
    return visible, fund_visible


def _apply_params_to_class(cls: type[Strategy], params: StrategyParams) -> None:
    """Inject swept params as class attributes on a Strategy subclass.

    Only sets attributes that already exist on the class — so snapback params
    don't pollute Donchian, Donchian params don't pollute carry, etc.
    """
    for field in (
        "rsi_long_threshold", "rsi_short_threshold",
        "volume_multiple", "funding_long_max", "funding_short_min",
        "atr_tp_multiple", "atr_sl_multiple",
        "time_stop_bars", "risk_per_trade_pct",
        "donchian_period_entry", "donchian_period_exit", "atr_trail_multiple",
        "funding_threshold", "funding_exit_threshold", "sl_pct",
        "max_24h_change_pct",
        # carry-v3
        "atr_percentile_threshold", "dd_halt_pct",
        # carry-v4
        "trend_ema_period",
        # fmom-v1
        "momentum_lookback_bars", "momentum_threshold", "tp_pct", "require_trend_align",
        # donchian-v3 regime gate
        "regime_ema_period", "regime_slope_window", "slope_trend_threshold_pct",
        # multifactor-v1
        "mf_trend_ema_period", "macd_fast", "macd_slow", "macd_signal",
        "require_candlestick", "require_macd", "require_trend",
        "require_funding_not_extreme", "funding_extreme_threshold", "max_hold_bars",
        # multifactor-v2 trailing (legacy; v2 reborn as TA confirmation)
        "trail_activate_atr", "trail_atr_multiple",
        # multifactor-v2 TA confirmation (current).
        # `confirmations_required` is CLASS-LEVEL per variant (Loose=1, Strict=3,
        # v3=3). DO NOT include it here — _apply_params_to_class would otherwise
        # clobber each variant with the StrategyParams default (2), making all
        # variants identical. Same rationale as the v3 enable_* flags below.
        "swing_k", "swing_lookback_bars",
        "trendline_max_distance_pct", "sr_max_distance_pct",
        "sr_cluster_tolerance_pct", "fib_max_distance_pct",
        # multifactor-v3 — only THRESHOLDS go through param injection.
        # enable_dist_ema_filter / enable_atr_stops / enable_vol_regime_gate are
        # CLASS-LEVEL switches set by the variant subclass (V3DistEmaOnly etc),
        # so do NOT include them here — _apply_params_to_class would otherwise
        # reset them to StrategyParams defaults and break ablation testing.
        "max_distance_above_ema_pct", "max_distance_below_ema_pct",
        # atr_sl_k / atr_tp_k are CLASS-LEVEL per variant
        # (V3All: 1.5/3.0; V3AllWider2: 2.0/4.0; V3AllWider3: 3.0/6.0; etc).
        # Same rationale as confirmations_required and the enable_* flags:
        # if we apply StrategyParams defaults here we'd silently clobber the
        # variant's intentional choice, making ablations indistinguishable.
        "vol_regime_lookback_days", "vol_regime_max_pctile",
        # volume_ma_period already in StrategyParams default 20
    ):
        if hasattr(cls, field):
            setattr(cls, field, getattr(params, field))


def run_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    cash: float = 10_000.0,
    leverage: int | None = None,
    quiet: bool = False,
    params_override: StrategyParams | None = None,
    return_equity: bool = False,
    return_trades: bool = False,
    commission: float | None = None,
) -> dict:
    """Run a backtest.

    `params_override` lets walk-forward / sweep harnesses inject specific
    StrategyParams without touching config/params.yaml. When None, params
    come from the yaml as usual. Ignored for buy-and-hold.
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy_name}")

    # Result timeframe label (will be overwritten if strategy uses a single TF).
    _result_tf_label = timeframe

    if strategy_name == "buy-and-hold":
        data = _prepare_buy_and_hold_data(symbol, timeframe, start, end)
        funding_in_span = load_funding(
            symbol=symbol, days_back=max((end - start).days + 2, 2), end=end
        ).loc[start:end]
        if funding_in_span.index.tz is not None:
            funding_in_span = funding_in_span.copy()
            funding_in_span.index = funding_in_span.index.tz_convert("UTC").tz_localize(None)
        params = None
    else:
        params = params_override or StrategyParams.from_yaml()
        cls = STRATEGIES[strategy_name]
        # CRITICAL: read donchian periods from `params` (the source of truth for
        # this run), NOT from the class attr (which is sticky from the prior
        # backtest in the same process). The prior-bug version read class attrs
        # before _apply_params_to_class was called, so the data prep used STALE
        # channel periods while the strategy ran with NEW ones — producing
        # mismatched Donchian bands and bogus "OOS passing" results.
        donchian_entry = getattr(params, "donchian_period_entry",
                                 getattr(cls, "donchian_period_entry", 20))
        donchian_exit = getattr(params, "donchian_period_exit",
                                getattr(cls, "donchian_period_exit", 10))

        if timeframe != "15m" and strategy_name in _TF_AGNOSTIC_STRATEGIES:
            # Single-TF prep: carry / donchian / rider on arbitrary entry timeframe.
            data, funding_in_span = _prepare_tf_agnostic_data(
                symbol, timeframe, start, end,
                with_donchian=(strategy_name in _DONCHIAN_STRATEGIES),
                donchian_entry=donchian_entry,
                donchian_exit=donchian_exit,
                atr_period=params.atr_period,
                with_rider=(strategy_name in _RIDER_STRATEGIES),
                rider_donchian_n=getattr(cls, "rider_donchian_n", 55),
                rider_atr_period=getattr(cls, "rider_atr_period", 14),
                rider_ema_period=getattr(cls, "rider_ema_period", 200),
                with_supertrend=(strategy_name in _SUPERTREND_STRATEGIES),
                supertrend_period=getattr(cls, "st_period", 10),
                supertrend_multiplier=getattr(cls, "st_multiplier", 3.0),
                supertrend_strategy_name=strategy_name,
                supertrend_extra_kwargs={
                    kwarg: getattr(cls, attr)
                    for kwarg, attr in _SUPERTREND_ATTACH_FNS.get(strategy_name, (None, {}))[1].items()
                } if strategy_name in _SUPERTREND_ATTACH_FNS else None,
            )
        else:
            if timeframe != "15m":
                log.warning("strategy %s requires 15m entry; ignoring tf=%s",
                            strategy_name, timeframe)
            data, funding_in_span = _prepare_snapback_data(
                symbol, start, end, params,
                with_regimes=(strategy_name in _REGIME_STRATEGIES),
                with_donchian=(strategy_name in _DONCHIAN_STRATEGIES),
                donchian_entry=donchian_entry,
                donchian_exit=donchian_exit,
            )
        _apply_params_to_class(STRATEGIES[strategy_name], params)

    eff_leverage = leverage or (params.leverage if params else 1)
    margin = 1.0 / max(eff_leverage, 1)

    eff_commission = commission if commission is not None else COMMISSION_PER_SIDE
    if strategy_name == "buy-and-hold":
        actual_cash = cash
        bt = FractionalBacktest(
            data,
            STRATEGIES[strategy_name],
            cash=actual_cash,
            commission=eff_commission,
            margin=margin,
            trade_on_close=False,
            exclusive_orders=True,
            fractional_unit=FRACTIONAL_UNIT,
            finalize_trades=True,
        )
    else:
        # Plain Backtest with $1M cash so 1 BTC ≈ 4% of equity and integer
        # unit sizing works without FractionalBacktest's price scaling (which
        # would silently desync custom indicator columns).
        actual_cash = cash if cash != 10_000.0 else SNAPBACK_DEFAULT_CASH
        STRATEGIES[strategy_name].leverage = eff_leverage  # type: ignore[attr-defined]
        bt = Backtest(
            data,
            STRATEGIES[strategy_name],
            cash=actual_cash,
            commission=eff_commission,
            margin=margin,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True,
        )
    stats = bt.run()

    naive_return_pct = (
        float(data["Close"].iloc[-1]) / float(data["Open"].iloc[0]) - 1.0
    ) * 100.0
    bt_return_pct = float(stats["Return [%]"])

    funding_cost_usdt: float | None = None
    funding_events = 0
    after_funding_pct: float | None = None
    if strategy_name == "buy-and-hold":
        funding_cost_usdt, funding_events = funding_cost_for_long_btc(
            data, funding_in_span, initial_cash=cash, commission=COMMISSION_PER_SIDE
        )
    else:
        trades_df = getattr(stats, "_trades", None)
        funding_cost_usdt, funding_events = funding_cost_for_trades(
            trades_df, data, funding_in_span
        )

    if funding_cost_usdt is not None:
        final_equity = actual_cash * (1.0 + bt_return_pct / 100.0) - funding_cost_usdt
        after_funding_pct = (final_equity / actual_cash - 1.0) * 100.0

    result = {
        "strategy": strategy_name,
        "symbol": symbol,
        "timeframe": (
            timeframe if strategy_name == "buy-and-hold"
            else (timeframe if strategy_name in _TF_AGNOSTIC_STRATEGIES and timeframe != "15m"
                  else "15m+1h")
        ),
        "start": data.index[0],
        "end": data.index[-1],
        "bars": len(data),
        "trades": int(stats["# Trades"]),
        "naive_return_pct": naive_return_pct,
        "backtest_return_pct": bt_return_pct,
        "after_funding_pct": after_funding_pct,
        "funding_cost_usdt": funding_cost_usdt,
        "funding_events": funding_events,
        "sharpe": float(stats.get("Sharpe Ratio") or 0.0),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]") or 0.0),
        "profit_factor": _safe_pf(stats),
        "win_rate_pct": float(stats.get("Win Rate [%]") or 0.0),
        "avg_trade_pct": float(stats.get("Avg. Trade [%]") or 0.0),
        "commission_per_side": eff_commission,
        "leverage": eff_leverage,
    }

    if return_equity:
        eq = getattr(stats, "_equity_curve", None)
        if eq is not None and "Equity" in eq.columns:
            # Normalise to a returns series anchored at 1.0 so ensembling on
            # different cash bases is meaningful.
            equity = eq["Equity"].astype(float)
            result["equity_series"] = equity
            result["returns_series"] = equity / float(equity.iloc[0])
            result["actual_cash"] = actual_cash

    if return_trades:
        result["trades_df"] = getattr(stats, "_trades", None)

    if not quiet:
        _print_result(result)
    return result


def _safe_pf(stats) -> float:
    pf = stats.get("Profit Factor")
    if pf is None or pd.isna(pf):
        return 0.0
    return float(pf)


def _print_result(r: dict) -> None:
    print()
    print(f"=== {r['strategy']} | {r['symbol']} {r['timeframe']} | {r['leverage']}x ===")
    print(f"  period          : {r['start']} → {r['end']}  ({r['bars']} bars)")
    print(f"  commission/side : {r['commission_per_side']*100:.4f}%  "
          f"({TAKER_FEE*100:.4f}% fee + {SLIPPAGE_PROXY*100:.4f}% slip)")
    print(f"  trades          : {r['trades']}")
    print()
    print(f"  naive B&H       : {r['naive_return_pct']:+.2f}%   (price change, zero friction)")
    print(f"  after fees+slip : {r['backtest_return_pct']:+.2f}%   (backtesting.py)")
    if r["after_funding_pct"] is not None:
        print(f"  after funding   : {r['after_funding_pct']:+.2f}%   "
              f"(funding cost {r['funding_cost_usdt']:+.2f} USDT over {r['funding_events']} events)")
    print()
    print(f"  Sharpe          : {r['sharpe']:.2f}")
    print(f"  Max DD          : {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor   : {r['profit_factor']:.2f}")
    print(f"  Win rate        : {r['win_rate_pct']:.1f}%")
    print(f"  Avg trade       : {r['avg_trade_pct']:+.3f}%")


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Run a backtest on cached Binance Futures data.")
    p.add_argument("--strategy", default="snapback-v1", choices=list(STRATEGIES.keys()))
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--tf", default="15m", help="entry timeframe (ignored for snapback-v1 which is fixed 15m+1h)")
    p.add_argument("--start", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="YYYY-MM-DD (UTC)")
    p.add_argument("--days", type=int, help="lookback days (overrides --start)")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--leverage", type=int, help="override params.yaml leverage")
    args = p.parse_args()

    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=UTC)
        if args.end
        else datetime.now(UTC)
    )
    if args.days:
        start = end - timedelta(days=args.days)
    elif args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    else:
        start = end - timedelta(days=365)

    run_backtest(
        args.strategy, args.symbol, args.tf, start, end,
        cash=args.cash, leverage=args.leverage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
