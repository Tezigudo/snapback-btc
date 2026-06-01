"""READ-ONLY: pull my real Binance Futures trade history (last 365d) — income
AND the actual fills (side/price/time) — so we can classify each trade as
'pump-fade' vs 'inverted strategy' and deep-dive its chart.

Reads the bot's futures key from env (BinanceClient.from_env). Calls ONLY GET
endpoints (income, userTrades). NEVER places/cancels anything, NEVER prints the
API key. Dumps detail to /tmp and prints a per-TRADE reconstruction.

Run on the droplet:  cd /root/snapback-btc && .venv/bin/python -m tools.pull_my_futures_history
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from exchange import env as _env  # noqa: F401  (loads .env at import)
from exchange.binance_client import BinanceClient

DAYS = 2000          # ~5.5y — the full possible life of Binance USDT-M futures
WEEK_MS = 7 * 86_400_000
WIN_MS = 90 * 86_400_000   # Binance /fapi/v1/income caps the query window — must page in <=~90d chunks
OUT = Path("/tmp/my_futures_history.json")


def windowed_income(ex, start_ms: int, now_ms: int) -> list[dict]:
    """Walk income in explicit <=90-day windows (start->now). A 5y request with
    NO endTime silently returns only the recent slice — this reaches the full history."""
    rows, a = [], start_ms
    while a < now_ms:
        b = min(a + WIN_MS, now_ms)
        cursor = a
        for _ in range(60):  # page within the window
            try:
                page = ex.fapiPrivateGetIncome({"startTime": cursor, "endTime": b, "limit": 1000})
            except Exception as e:  # noqa: BLE001
                print(f"   warn income window {a}: {str(e)[:80]}"); page = []
            if not page:
                break
            rows.extend(page)
            if len(page) < 1000:
                break
            cursor = int(page[-1].get("time") or cursor) + 1
        a = b + 1
    return rows


def fetch_user_trades(ex, symbol: str, start_ms: int, now_ms: int) -> list[dict]:
    """userTrades is capped at 7-day windows — walk forward in weekly chunks."""
    out, a = [], start_ms
    while a < now_ms:
        b = min(a + WEEK_MS, now_ms)
        try:
            page = ex.fapiPrivateGetUserTrades({"symbol": symbol, "startTime": a, "endTime": b, "limit": 1000})
        except Exception as e:  # noqa: BLE001
            page = []
            print(f"   warn {symbol} window: {str(e)[:80]}")
        out.extend(page)
        a = b + 1
    return out


def _income_trade(cluster: list[dict]) -> dict:
    """Build a trade record from REALIZED_PNL income rows only (no fills/prices available)."""
    return {
        "symbol": cluster[0]["symbol"], "side": "?", "entry_px": None, "exit_px": None, "qty": None,
        "realizedPnl": round(sum(float(r["income"]) for r in cluster), 4),
        "open_ms": int(cluster[0]["time"]), "close_ms": int(cluster[-1]["time"]),
        "n_fills": len(cluster), "source": "income",
    }


def reconstruct(fills: list[dict]) -> list[dict]:
    """Group fills into positions: a run where net qty stays non-zero.
    Opening side = SELL -> SHORT, BUY -> LONG."""
    fills = sorted(fills, key=lambda f: int(f["time"]))
    trades, net, cur = [], 0.0, []
    for f in fills:
        q = float(f["qty"]) * (1 if f["side"] == "BUY" else -1)
        if abs(net) < 1e-9 and cur:  # previous trade closed
            trades.append(cur); cur = []
        cur.append(f); net += q
        if abs(net) < 1e-9:
            trades.append(cur); cur = []
    if cur:
        trades.append(cur)
    out = []
    for grp in trades:
        if not grp:
            continue
        open_side = grp[0]["side"]
        is_short = open_side == "SELL"
        opens = [f for f in grp if f["side"] == open_side]
        closes = [f for f in grp if f["side"] != open_side]

        def wavg(fs):
            qn = sum(float(f["qty"]) for f in fs)
            return sum(float(f["price"]) * float(f["qty"]) for f in fs) / qn if qn else 0.0
        out.append({
            "symbol": grp[0]["symbol"],
            "side": "SHORT" if is_short else "LONG",
            "entry_px": round(wavg(opens), 8),
            "exit_px": round(wavg(closes), 8) if closes else None,
            "qty": round(sum(float(f["qty"]) for f in opens), 6),
            "realizedPnl": round(sum(float(f.get("realizedPnl") or 0) for f in grp), 4),
            "open_ms": int(grp[0]["time"]),
            "close_ms": int(grp[-1]["time"]),
            "n_fills": len(grp),
        })
    return out


def main() -> int:
    client = BinanceClient.from_env()
    ex = client.ex
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - DAYS * 86_400_000

    income = windowed_income(ex, start_ms, now_ms)
    income_rows = [{"symbol": r.get("symbol") or "", "incomeType": r.get("incomeType") or "",
                    "income": float(r.get("income") or 0.0), "time": int(r.get("time") or 0)}
                   for r in income]
    symbols = sorted({r["symbol"] for r in income_rows if r["symbol"]})

    user_trades, all_trades = {}, []
    for s in symbols:
        st = [r["time"] for r in income_rows if r["symbol"] == s and r["time"]]
        a = (min(st) - WEEK_MS) if st else start_ms
        b = (max(st) + 2 * 86_400_000) if st else now_ms
        ut = fetch_user_trades(ex, s, a, b)
        user_trades[s] = [{"symbol": t["symbol"], "side": t["side"], "price": t["price"], "qty": t["qty"],
                           "realizedPnl": t.get("realizedPnl"), "time": int(t["time"]),
                           "positionSide": t.get("positionSide")} for t in ut]
        recon = reconstruct(user_trades[s])
        if not recon:
            # userTrades retention couldn't reach these fills — synthesize trades from
            # income REALIZED_PNL clusters (groups separated by >36h) so we still COUNT them.
            rp = sorted([r for r in income_rows if r["symbol"] == s and r["incomeType"] == "REALIZED_PNL"],
                        key=lambda r: r["time"])
            cluster, last = [], None
            for r in rp:
                if last is not None and r["time"] - last > 36 * 3_600_000 and cluster:
                    recon.append(_income_trade(cluster)); cluster = []
                cluster.append(r); last = r["time"]
            if cluster:
                recon.append(_income_trade(cluster))
        all_trades += recon

    OUT.write_text(json.dumps({"income": income_rows, "userTrades": user_trades, "trades": all_trades}))

    # per-trade reconstruction (the classification input) — no secrets
    all_trades.sort(key=lambda t: t["open_ms"])
    print(f"\n{'coin':<12}{'side':>6}{'opened':>17}{'held_h':>8}{'entry':>12}{'exit':>12}{'PnL$':>9}")
    import datetime as dt
    for t in all_trades:
        o = dt.datetime.fromtimestamp(t["open_ms"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        held = (t["close_ms"] - t["open_ms"]) / 3.6e6
        en_px = f"{t['entry_px']:.6g}" if t["entry_px"] is not None else "(income)"
        ex_px = f"{t['exit_px']:.6g}" if t["exit_px"] is not None else "(open)"
        print(f"{t['symbol']:<12}{t['side']:>6}{o:>17}{held:>8.1f}{en_px:>12}{ex_px:>12}{t['realizedPnl']:>9.2f}")
    import datetime as _dt
    if income_rows:
        lo = min(r["time"] for r in income_rows if r["time"])
        hi = max(r["time"] for r in income_rows if r["time"])
        fmt = lambda ms: _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%d")
        req = _dt.datetime.fromtimestamp(start_ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%d")
        print(f"\nrequested from {req} ({DAYS}d) -> Binance returned data {fmt(lo)} .. {fmt(hi)}")
        print("  (if 'returned from' is much later than requested, that's Binance's retention cap)")
    print(f"{len(all_trades)} trades across {len(symbols)} symbols. wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
