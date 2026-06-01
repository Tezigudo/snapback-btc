"""Survivorship-safe data layer for the pump-fade study.

The strategy ("short the day's top gainer after it rolls over, TP at prior
support") only means anything if we can see the coins that ACTUALLY pumped on
each historical day — including the ones that later got delisted. The live
fapi REST `/klines` endpoint returns garbage placeholder bars for delisted
symbols (flat price, zero volume, future-dated), so it is survivorship-trapped.

`data.binance.vision` (Binance's public data dump) keeps full history for
delisted symbols — daily + intraday klines AND funding — and its S3 listing
enumerates every symbol that ever traded. That is our point-in-time universe.

Two parsing gotchas this module handles:
  1. Some files have a header row ("open_time,open,..."), older ones don't.
  2. Binance switched kline open_time from MILLIseconds to MICROseconds in
     futures files around 2025 — same column, different unit. We normalise.

Cache layout (parquet, idempotent):
    data/pumpfade/universe.json                  # symbol -> first/last month
    data/pumpfade/klines/{SYMBOL}_{tf}.parquet
    data/pumpfade/funding/{SYMBOL}.parquet

No API key required. Public endpoints only.
"""

from __future__ import annotations

import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "pumpfade"
KLINES_DIR = CACHE / "klines"
FUNDING_DIR = CACHE / "funding"
for _d in (CACHE, KLINES_DIR, FUNDING_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# data.binance.vision is fronted by an HTML site; the raw S3 listing API lives
# at the s3-ap-northeast-1 host. Object downloads work off either host.
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
TIMEOUT = 40
KLINE_COLS = ["open", "high", "low", "close", "volume", "quote_volume"]


# --------------------------------------------------------------------------- #
# S3 listing helpers
# --------------------------------------------------------------------------- #
def _s3_list(prefix: str, delimiter: str = "/") -> tuple[list[str], list[str]]:
    """Return (common_prefixes, object_keys) for one S3 listing page-set.

    Paginates via Marker until IsTruncated=false. With delimiter='/', folders
    come back as CommonPrefixes and files as Contents/Key.
    """
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        params = {"prefix": prefix, "delimiter": delimiter}
        if marker:
            params["marker"] = marker
        r = requests.get(S3_LIST, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for cp in root.findall(f"{S3_NS}CommonPrefixes"):
            p = cp.findtext(f"{S3_NS}Prefix")
            if p:
                prefixes.append(p)
        for c in root.findall(f"{S3_NS}Contents"):
            k = c.findtext(f"{S3_NS}Key")
            if k:
                keys.append(k)
        truncated = (root.findtext(f"{S3_NS}IsTruncated") or "false") == "true"
        if not truncated:
            break
        # NextMarker is only present with a delimiter; else use last key/prefix.
        nm = root.findtext(f"{S3_NS}NextMarker")
        marker = nm or (prefixes + keys)[-1]
    return prefixes, keys


def enumerate_universe(quote: str = "USDT") -> list[str]:
    """All USD-M futures symbols that EVER had daily klines (survivorship-safe).

    Filters to the given quote asset. Includes delisted symbols.
    """
    prefixes, _ = _s3_list("data/futures/um/daily/klines/")
    syms = [p.rstrip("/").split("/")[-1] for p in prefixes]
    return sorted(s for s in syms if s.endswith(quote))


def list_months(symbol: str, tf: str, kind: str = "klines") -> list[str]:
    """Available monthly archive months for a symbol/timeframe, e.g. ['2022-10'].

    kind: 'klines' (needs tf) or 'fundingRate' (tf ignored).
    """
    if kind == "klines":
        prefix = f"data/futures/um/monthly/klines/{symbol}/{tf}/"
        suffix = f"{symbol}-{tf}-"
    else:
        prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
        suffix = f"{symbol}-fundingRate-"
    _, keys = _s3_list(prefix, delimiter="")
    months = []
    for k in keys:
        name = k.split("/")[-1]
        if name.endswith(".zip") and name.startswith(suffix):
            months.append(name[len(suffix) : -len(".zip")])
    return sorted(months)


# --------------------------------------------------------------------------- #
# Download + parse
# --------------------------------------------------------------------------- #
def _normalise_time(series: pd.Series) -> pd.DatetimeIndex:
    """Binance open_time is ms pre-2025, microseconds in newer futures files."""
    v = pd.to_numeric(series, errors="coerce")
    unit = "us" if v.dropna().median() > 1e14 else "ms"
    return pd.to_datetime(v, unit=unit, utc=True)


def _read_kline_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    # Header detection: first byte non-digit/non-minus => header row present.
    head = raw[:32].lstrip()
    has_header = head[:1].isalpha() if head else False
    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote", "ignore",
        ],
        usecols=range(8),
    )
    if has_header:  # names= ignored when header=0; re-apply our names
        df.columns = [
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        ]
    idx = _normalise_time(df["open_time"])
    out = df[["open", "high", "low", "close", "volume", "quote_volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    out.index = idx
    out.index.name = "open_time"
    return out[out.index.notna()].sort_index()


def _read_funding_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0])
    head = raw[:32].lstrip()
    has_header = head[:1].isalpha() if head else False
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    # Columns vary: (calc_time, [funding_interval_hours], last_funding_rate).
    df.columns = [str(c).lower() for c in df.columns] if has_header else [
        "calc_time", "funding_interval_hours", "last_funding_rate"
    ][: df.shape[1]]
    tcol = "calc_time" if "calc_time" in df.columns else df.columns[0]
    rcol = next((c for c in df.columns if "rate" in c), df.columns[-1])
    idx = _normalise_time(df[tcol])
    out = pd.DataFrame({"funding_rate": pd.to_numeric(df[rcol], errors="coerce")})
    out.index = idx
    out.index.name = "funding_time"
    return out[out.index.notna()].sort_index()


def _dl(url: str) -> bytes | None:
    r = requests.get(url, timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def _fetch_months(symbol: str, tf: str, months: list[str], kind: str = "klines") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def one(m: str) -> pd.DataFrame | None:
        if kind == "klines":
            url = f"{DL}/data/futures/um/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{m}.zip"
            parse = _read_kline_zip
        else:
            url = f"{DL}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip"
            parse = _read_funding_zip
        c = _dl(url)
        return parse(c) if c else None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(one, m) for m in months]):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
    if not frames:
        cols = KLINE_COLS if kind == "klines" else ["funding_rate"]
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def load_daily(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """All daily (1d) klines for a symbol, cached to parquet. Survivorship-safe."""
    path = KLINES_DIR / f"{symbol}_1d.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    months = list_months(symbol, "1d")
    df = _fetch_months(symbol, "1d", months) if months else pd.DataFrame(columns=KLINE_COLS)
    if not df.empty:
        df.to_parquet(path)
    return df


def load_intraday(symbol: str, tf: str, months: list[str]) -> pd.DataFrame:
    """Intraday klines for the given month list (event windows). Cached per (sym,tf)."""
    path = KLINES_DIR / f"{symbol}_{tf}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=KLINE_COLS)
    have = set(cached.index.strftime("%Y-%m")) if not cached.empty else set()
    need = [m for m in months if m not in have]
    if need:
        fresh = _fetch_months(symbol, tf, need)
        if not fresh.empty:
            cached = pd.concat([cached, fresh]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
            cached.to_parquet(path)
    return cached


def load_funding(symbol: str, months: list[str]) -> pd.DataFrame:
    path = FUNDING_DIR / f"{symbol}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["funding_rate"])
    have = set(cached.index.strftime("%Y-%m")) if not cached.empty else set()
    need = [m for m in months if m not in have]
    if need:
        fresh = _fetch_months(symbol, "", need, kind="fundingRate")
        if not fresh.empty:
            cached = pd.concat([cached, fresh]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
            cached.to_parquet(path)
    return cached


def months_spanning(start: datetime, end: datetime) -> list[str]:
    """List of 'YYYY-MM' month tags covering [start, end] inclusive."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Survivorship-safe Binance futures data dump fetcher.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("universe")
    dd = sub.add_parser("daily")
    dd.add_argument("symbol")
    pa = sub.add_parser("daily-all")
    pa.add_argument("--limit", type=int, default=0)
    pa.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    if args.cmd == "universe":
        u = enumerate_universe()
        print(f"{len(u)} USDT perps (incl. delisted). sample: {u[:10]} ... {u[-5:]}")
        (CACHE / "universe.json").write_text(__import__("json").dumps(u, indent=0))
        print(f"wrote {CACHE / 'universe.json'}")
    elif args.cmd == "daily":
        df = load_daily(args.symbol, refresh=True)
        if df.empty:
            print(f"{args.symbol}: EMPTY")
        else:
            print(f"{args.symbol}: {len(df)} days {df.index.min().date()} -> {df.index.max().date()}")
    elif args.cmd == "daily-all":
        u = enumerate_universe()
        if args.limit:
            u = u[: args.limit]
        print(f"downloading daily 1d for {len(u)} symbols (workers={args.workers})...", flush=True)
        done = {"n": 0, "empty": 0, "err": 0}

        def grab(s: str) -> str:
            try:
                df = load_daily(s)
                if df.empty:
                    done["empty"] += 1
                    return f"  {s}: EMPTY"
                return f"  {s}: {len(df)}d {df.index.min().date()}->{df.index.max().date()}"
            except Exception as e:  # noqa: BLE001
                done["err"] += 1
                return f"  {s}: ERR {type(e).__name__}: {e}"

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(grab, s) for s in u]):
                done["n"] += 1
                msg = fut.result()
                if done["n"] % 25 == 0 or "ERR" in msg:
                    print(f"[{done['n']}/{len(u)}]{msg}", flush=True)
        print(f"done. empty={done['empty']} err={done['err']} at {datetime.now(UTC):%H:%M:%S}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
