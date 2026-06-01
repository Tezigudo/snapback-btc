"""Generate a continuous full-history SOL cnh-hybrid-short trade CSV.

Same detector/simulator as the gate (dedup=15), run over the WHOLE SOL 4h
series (not windowed), so the DCA sim has a continuous trade stream with
EntryTime/ExitTime/prices/net_pct + atr_pct at entry (for risk sizing).

Out: reports/sol_hybrid_short_trades.csv
Run: uv run python tools/gen_sol_hybrid_trades.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.icnh_mega_sweep import Config, simulate_trades  # noqa: E402
from tools.icnh_final_tune import find_hybrid_patterns  # noqa: E402
from tools.cross_coin_backtest import _load_coin_4h  # noqa: E402


def main() -> int:
    df = _load_coin_4h("SOL")
    if df.empty:
        print("no SOL data")
        return 1
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
    hits = find_hybrid_patterns(df, dt_cfg, icnh_cfg)
    dt_idxs = [h for h, src in hits if src == "DT"]
    icnh_idxs = [h for h, src in hits if src == "ICNH"]
    trades = (simulate_trades(df, dt_idxs, dt_cfg, "full")
              + simulate_trades(df, icnh_idxs, icnh_cfg, "full"))
    atr_pct = (df["atr14"] / df["close"])
    rows = []
    for t in trades:
        ets = pd.Timestamp(t["entry_ts"])
        ap = float(atr_pct.get(ets, np.nan))
        rows.append({
            "EntryTime": t["entry_ts"], "ExitTime": t["exit_ts"],
            "EntryPrice": t["entry_price"], "ExitPrice": t["exit_price"],
            "net_pct": t["net_pct"], "atr_pct": ap,
            "exit_reason": t["exit_reason"], "bars_held": t["bars_held"],
        })
    out = pd.DataFrame(rows).sort_values("EntryTime").reset_index(drop=True)
    path = ROOT / "reports" / "sol_hybrid_short_trades.csv"
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} SOL hybrid-short trades -> {path}")
    print(f"  span {out['EntryTime'].iloc[0][:10]} .. {out['EntryTime'].iloc[-1][:10]}")
    print(f"  mean net/trade = {out['net_pct'].mean()*100:+.2f}%  "
          f"WR = {(out['net_pct'] > 0).mean()*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
