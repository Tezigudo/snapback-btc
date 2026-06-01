"""Reconstruct ALL real futures positions from the async-download trade CSV
(3y, 65 symbols) and report performance by side / coin-class — and isolate the
pump-fade book (SHORT alts) so we can later test the volume rule on real money.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TRADES = Path("/tmp/full_futures_trades.csv")
OUT = Path("/tmp/full_positions.json")
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "LTCUSDT", "AVAXUSDT",
          "SUIUSDT", "APTUSDT", "DOGEUSDT", "LINKUSDT", "BNBUSDT", "TIAUSDT", "ONDOUSDT"}


def f(x: str) -> float:
    try:
        return float(str(x).split()[0])
    except (ValueError, IndexError):
        return 0.0


def reconstruct(fills: list[dict]) -> list[dict]:
    fills.sort(key=lambda r: (r["t"], r["tid"]))
    out, net, cur = [], 0.0, []
    for r in fills:
        q = r["qty"] * (1 if r["side"] == "BUY" else -1)
        if abs(net) < 1e-9 and cur:
            out.append(cur); cur = []
        cur.append(r); net += q
        if abs(net) < 1e-9:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    trades = []
    for grp in out:
        if not grp:
            continue
        oside = grp[0]["side"]
        opens = [r for r in grp if r["side"] == oside]
        closes = [r for r in grp if r["side"] != oside]
        wq = lambda fs: (sum(r["px"] * r["qty"] for r in fs) / sum(r["qty"] for r in fs)) if fs else 0.0
        trades.append({
            "symbol": grp[0]["symbol"], "side": "SHORT" if oside == "SELL" else "LONG",
            "entry_px": wq(opens), "exit_px": wq(closes) if closes else None,
            "qty": sum(r["qty"] for r in opens),
            "pnl": sum(r["pnl"] for r in grp), "fee": sum(r["fee"] for r in grp),
            "open_ms": grp[0]["t"], "close_ms": grp[-1]["t"], "n": len(grp),
            "open": grp[0]["iso"], "close": grp[-1]["iso"],
        })
    return trades


def main() -> int:
    rows = list(csv.DictReader(open(TRADES, encoding="utf-8-sig")))
    bysym = defaultdict(list)
    for r in rows:
        t = int(datetime.strptime(r["Time(UTC)"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        bysym[r["Symbol"]].append({"symbol": r["Symbol"], "side": r["Side"], "px": f(r["Price"]),
                                   "qty": f(r["Quantity"]), "pnl": f(r["Realized Profit"]), "fee": f(r["Fee"]),
                                   "t": t, "iso": r["Time(UTC)"], "tid": int(r["Trade Id"])})
    trades = []
    for s, fl in bysym.items():
        trades += reconstruct(fl)
    trades.sort(key=lambda x: x["open_ms"])
    for x in trades:
        x["net"] = x["pnl"] - x["fee"]
        x["alt"] = x["symbol"] not in MAJORS
    OUT.write_text(json.dumps(trades))

    closed = [t for t in trades if t["exit_px"] is not None]
    def stat(name, ts):
        if not ts:
            print(f"  {name:<26} n=0"); return
        net = sum(t["net"] for t in ts)
        wins = sum(1 for t in ts if t["net"] > 0)
        print(f"  {name:<26} n={len(ts):<4} win {100*wins/len(ts):>4.0f}%  netPnL ${net:>+9.1f}  "
              f"avg ${net/len(ts):>+6.1f}  best ${max(t['net'] for t in ts):>+7.1f}  worst ${min(t['net'] for t in ts):>+7.1f}")

    print(f"\n=== ALL POSITIONS ({len(trades)} total, {len(closed)} closed) ===")
    stat("ALL", closed)
    print("\n-- by side --")
    stat("LONG", [t for t in closed if t["side"] == "LONG"])
    stat("SHORT", [t for t in closed if t["side"] == "SHORT"])
    print("\n-- the PUMP-FADE book (SHORT alts) vs others --")
    stat("SHORT alt (pump-fade)", [t for t in closed if t["side"] == "SHORT" and t["alt"]])
    stat("SHORT major", [t for t in closed if t["side"] == "SHORT" and not t["alt"]])
    stat("LONG alt", [t for t in closed if t["side"] == "LONG" and t["alt"]])
    stat("LONG major", [t for t in closed if t["side"] == "LONG" and not t["alt"]])
    print("\n-- top 8 winners --")
    for t in sorted(closed, key=lambda x: -x["net"])[:8]:
        print(f"   {t['symbol']:<14}{t['side']:>6} {t['open'][:10]}  ${t['net']:+8.1f}  held {(t['close_ms']-t['open_ms'])/3.6e6:.0f}h")
    print("-- top 8 losers --")
    for t in sorted(closed, key=lambda x: x["net"])[:8]:
        print(f"   {t['symbol']:<14}{t['side']:>6} {t['open'][:10]}  ${t['net']:+8.1f}  held {(t['close_ms']-t['open_ms'])/3.6e6:.0f}h")
    print(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
