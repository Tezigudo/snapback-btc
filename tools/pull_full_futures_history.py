"""READ-ONLY: pull FULL futures history (~3y) via Binance's async-download API —
the API equivalent of the web "Export", which reaches far past the ~4-month
live-endpoint wall.

Flow per Binance docs:
  1. GET /fapi/v1/trade/asyn  (+ income/asyn)  -> {downloadId}   (window <= 1 year)
  2. poll GET /fapi/v1/trade/asyn/id?downloadId=..  -> {status, url} until completed
  3. download url (zip/gzip/csv), save raw CSV to /tmp

Saves /tmp/full_futures_trades.csv and /tmp/full_futures_income.csv + a manifest,
and prints headers/row-counts/date-range so we can parse locally. NEVER places
orders, NEVER prints the API key.

Run on the droplet:  cd /root/snapback-btc && .venv/bin/python -m tools.pull_full_futures_history
"""

from __future__ import annotations

import gzip
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from exchange import env as _env  # noqa: F401
from exchange.binance_client import BinanceClient

YEAR_MS = 365 * 86_400_000
LOOKBACK_YEARS = 3
POLL_TIMEOUT_S = 480
OUTDIR = Path("/tmp")


def fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def request_ids(ex, kind: str, start: int, end: int) -> list[tuple]:
    meth = getattr(ex, "fapiPrivateGetTradeAsyn" if kind == "trade" else "fapiPrivateGetIncomeAsyn")
    ids, a = [], start
    while a < end:
        b = min(a + YEAR_MS - 86_400_000, end)  # window strictly < 1 year
        try:
            r = meth({"startTime": a, "endTime": b})
            did = (r or {}).get("downloadId")
            print(f"  request {kind} {fmt(a)}..{fmt(b)} -> downloadId={did}", flush=True)
            if did:
                ids.append((kind, did, a, b))
        except Exception as e:  # noqa: BLE001
            print(f"  request {kind} {fmt(a)}..{fmt(b)} ERR: {str(e)[:140]}", flush=True)
        time.sleep(1.0)
        a = b + 1
    return ids


def poll(ex, ids: list[tuple], timeout_s: int = POLL_TIMEOUT_S) -> list[tuple]:
    ready, pending, t0 = [], list(ids), time.time()
    while pending and time.time() - t0 < timeout_s:
        still = []
        for (kind, did, a, b) in pending:
            idm = getattr(ex, "fapiPrivateGetTradeAsynId" if kind == "trade" else "fapiPrivateGetIncomeAsynId")
            try:
                s = idm({"downloadId": did}) or {}
            except Exception as e:  # noqa: BLE001
                print(f"  poll ERR {did}: {str(e)[:80]}", flush=True); still.append((kind, did, a, b)); continue
            if str(s.get("status")).lower() == "completed" and s.get("url"):
                print(f"  READY {kind} {fmt(a)}..{fmt(b)}", flush=True)
                ready.append((kind, did, a, b, s["url"]))
            elif s.get("isExpired") in (True, "true"):
                print(f"  EXPIRED {kind} {did}", flush=True)
            else:
                still.append((kind, did, a, b))
        pending = still
        if pending:
            time.sleep(8)
    for (kind, did, a, b) in pending:
        print(f"  STILL PROCESSING {kind} {did} ({fmt(a)}..{fmt(b)}) after {timeout_s}s", flush=True)
    return ready


def download_extract(url: str) -> str:
    raw = urllib.request.urlopen(url, timeout=90).read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        return "\n".join(zf.read(n).decode("utf-8", "replace") for n in zf.namelist())
    except zipfile.BadZipFile:
        pass
    try:
        return gzip.decompress(raw).decode("utf-8", "replace")
    except OSError:
        pass
    return raw.decode("utf-8", "replace")


def main() -> int:
    ex = BinanceClient.from_env().ex
    now = int(time.time() * 1000)
    start = now - LOOKBACK_YEARS * YEAR_MS
    print(f"requesting async downloads for {fmt(start)}..{fmt(now)} (trade + income)", flush=True)

    ids = request_ids(ex, "trade", start, now) + request_ids(ex, "income", start, now)
    print(f"\npolling {len(ids)} download IDs (up to {POLL_TIMEOUT_S}s)...", flush=True)
    ready = poll(ex, ids)

    manifest = []
    for kind in ("trade", "income"):
        parts = [download_extract(url) for (k, did, a, b, url) in ready if k == kind]
        if not parts:
            print(f"\n{kind}: no completed downloads", flush=True)
            continue
        # merge CSV parts: keep first header, drop repeats
        lines, header = [], None
        for txt in parts:
            for i, ln in enumerate(txt.splitlines()):
                if not ln.strip():
                    continue
                if i == 0:
                    if header is None:
                        header = ln; lines.append(ln)
                    continue
                lines.append(ln)
        out = OUTDIR / f"full_futures_{'trades' if kind == 'trade' else 'income'}.csv"
        out.write_text("\n".join(lines))
        nrows = max(len(lines) - 1, 0)
        print(f"\n{kind}: {nrows} rows -> {out}", flush=True)
        print(f"  header: {header}", flush=True)
        for ln in lines[1:4]:
            print(f"  sample: {ln}", flush=True)
        manifest.append({"kind": kind, "rows": nrows, "file": str(out), "header": header})
    (OUTDIR / "full_futures_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUTDIR/'full_futures_manifest.json'}", flush=True)
    print("If 'STILL PROCESSING' above, just re-run this in a few minutes (Binance is generating).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
