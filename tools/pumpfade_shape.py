"""Deep-dive: do BLOW-OFF / EXHAUSTION shape features separate winning fades from
losing ones? (The discretionary signal my coarse mining missed.)

For each real pump-fade event, compute — POINT-IN-TIME, using only 1h bars up to
the entry decision — the features a tape reader actually uses:
  - peak_wick  : upper-wick fraction on the highest bar = seller rejection / climax
  - vol_ratio  : recent volume / peak-bar volume  (low = volume dried up = exhaustion)
  - pump_bars  : bars from day-open to the peak  (few = vertical blow-off; many = grind/trend)
  - ext_peak   : peak / EMA50(1h) - 1  (how stretched above trend at the top)
  - rolled     : 1 - entry/peak  (how far it had already rolled over at entry)
Then: winners-vs-losers medians + OOS-tested filters (with controls) + named examples.

Winner = net_ret > 0. IS entries <2025, OOS 2025+.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import pumpfade_data as pfd  # noqa: E402

CACHE = ROOT / "data" / "pumpfade"
OOS = pd.Timestamp("2025-01-01", tz="UTC")


def compute_features() -> pd.DataFrame:
    tr = pd.read_parquet(CACHE / "trades_intraday.parquet")
    tr = tr[tr.reason.isin(["TP", "STOP", "TIME", "SETTLE"]) & tr.entry_time.notna()].copy()
    tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
    tr["day_n"] = pd.to_datetime(tr["day"], utc=True).dt.normalize()
    rows = []
    for sym, grp in tr.groupby("symbol"):
        p = pfd.KLINES_DIR / f"{sym}_1h.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p)
        b = b[~b.index.duplicated()].sort_index()
        b = b[b["volume"] > 0]
        if b.empty:
            continue
        ema = b["close"].ewm(span=50, min_periods=20).mean()
        for _, e in grp.iterrows():
            et, ds = e["entry_time"], e["day_n"]
            win = b.loc[ds:et]
            if len(win) < 4:
                rows.append({**e.to_dict(), "peak_wick": np.nan})
                continue
            pk = win["high"].idxmax()
            pkbar = b.loc[pk]
            rng = max(float(pkbar["high"] - pkbar["low"]), 1e-12)
            wick = (float(pkbar["high"]) - max(float(pkbar["open"]), float(pkbar["close"]))) / rng
            pump_bars = int(len(b.loc[ds:pk]))
            vpeak = float(pkbar["volume"]) or np.nan
            vrec = float(b.loc[:et]["volume"].iloc[-3:].mean())
            vol_ratio = vrec / vpeak if vpeak and vpeak == vpeak else np.nan
            ema_at_pk = float(ema.loc[:pk].iloc[-1]) if ema.loc[:pk].notna().any() else np.nan
            ext_peak = (float(pkbar["high"]) / ema_at_pk - 1.0) if ema_at_pk and ema_at_pk == ema_at_pk else np.nan
            rows.append({
                "symbol": sym, "day": str(e["day"])[:10], "et": et, "net_ret": float(e["net_ret"]),
                "reason": e["reason"], "day_ret": float(e["day_ret"]), "entry": float(e["entry"]),
                "peak": float(e["peak"]),
                "peak_wick": round(wick, 3), "vol_ratio": round(vol_ratio, 2) if vol_ratio == vol_ratio else np.nan,
                "pump_bars": pump_bars, "ext_peak": round(ext_peak, 2) if ext_peak == ext_peak else np.nan,
                "rolled": round(1 - float(e["entry"]) / float(e["peak"]), 3),
            })
    return pd.DataFrame(rows)


def wl(df: pd.DataFrame) -> None:
    df["win"] = (df.net_ret > 0).astype(int)
    w, l = df[df.win == 1], df[df.win == 0]
    print("=== blow-off/exhaustion features: WINNERS vs LOSERS (median) ===")
    print(f"{'feature':<11}{'winners':>11}{'losers':>11}{'spread':>10}  reading")
    notes = {
        "peak_wick": "higher in winners => rejection-at-top fades better",
        "vol_ratio": "lower in winners => volume dried up (exhaustion)",
        "pump_bars": "lower in winners => sharper/vertical blow-off",
        "ext_peak": "higher in winners => more stretched snaps back",
        "rolled": "higher => entered later/deeper after rollover",
    }
    for f in ["peak_wick", "vol_ratio", "pump_bars", "ext_peak", "rolled"]:
        wv, lv = w[f].median(), l[f].median()
        print(f"{f:<11}{wv:>11.3f}{lv:>11.3f}{wv-lv:>10.3f}  {notes[f]}")


def ev(d: pd.DataFrame) -> str:
    if len(d) == 0:
        return "n=0"
    return (f"n={len(d):<4} EV {100*d.net_ret.mean():+6.2f}%  win {100*(d.net_ret>0).mean():>4.0f}%  "
            f"med {100*d.net_ret.median():+5.1f}%  worst {100*d.net_ret.min():>6.0f}%")


def test(df: pd.DataFrame, name: str, mask: pd.Series) -> None:
    s = df[mask.fillna(False)]
    isd, oos = s[s.et < OOS], s[s.et >= OOS]
    print(f"\n{name}  (keeps {len(s)}/{len(df)} = {100*len(s)/len(df):.0f}%)")
    print(f"   IS  {ev(isd)}")
    print(f"   OOS {ev(oos)}   <- must be + to be real")


def examples(df: pd.DataFrame) -> None:
    print("\n=== example REAL coins (winners vs blow-ups) with their shape ===")
    for tag, d in [("BIG WINS", df.nlargest(6, "net_ret")), ("BLOW-UPS", df.nsmallest(6, "net_ret"))]:
        print(f"-- {tag} --")
        for _, r in d.iterrows():
            print(f"  {r.symbol.replace('USDT',''):10s} {r.day}  pump+{r.day_ret*100:.0f}%  "
                  f"wick {r.peak_wick:.2f}  volR {r.vol_ratio:.2f}  bars {r.pump_bars:.0f}  "
                  f"ext {r.ext_peak:.1f}  rolled {r.rolled*100:.0f}%  -> NET {r.net_ret*100:+.0f}% ({r.reason})")


def main() -> int:
    df = compute_features()
    df = df[df.peak_wick.notna()].copy()
    print(f"events with shape features: {len(df)}\n")
    wl(df)
    print("\n=== OOS-TESTED SHAPE FILTERS (with controls) ===")
    test(df, "BLOW-OFF: wick>=0.25 (rejection at top)", df.peak_wick >= 0.25)
    test(df, "control: wick<0.05 (closed at highs = strength)", df.peak_wick < 0.05)
    test(df, "EXHAUSTION: vol_ratio<=0.4 (volume dried up)", df.vol_ratio <= 0.4)
    test(df, "control: vol_ratio>=1.0 (volume rising into entry)", df.vol_ratio >= 1.0)
    test(df, "VERTICAL: pump_bars<=6 (climactic spike)", df.pump_bars <= 6)
    test(df, "control: pump_bars>=18 (slow grind/trend)", df.pump_bars >= 18)
    test(df, "STRETCHED: ext_peak>=1.0 (>=100% above EMA50)", df.ext_peak >= 1.0)
    test(df, "COMPOSITE blow-off+exhaustion: wick>=0.2 & vol_ratio<=0.5",
         (df.peak_wick >= 0.2) & (df.vol_ratio <= 0.5))
    test(df, "COMPOSITE +vertical: wick>=0.2 & vol_ratio<=0.5 & pump_bars<=8",
         (df.peak_wick >= 0.2) & (df.vol_ratio <= 0.5) & (df.pump_bars <= 8))
    examples(df)
    df.to_parquet(CACHE / "shape_features.parquet")
    print(f"\nsaved {CACHE/'shape_features.parquet'}\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
