"""Tests for the pure payload mappers in tools/consolidate_futures_push.

These cover the row→ingest-shape transforms that turn ccxt responses into the
consolidate /futures/* payloads. The network + ccxt calls are NOT exercised
here (no live key); the mappers are the part that can silently corrupt the
dashboard's numbers, so they're tested directly with synthetic ccxt-shaped
dicts — the same verification strategy used on the TypeScript side.
"""

from __future__ import annotations

from tools.consolidate_futures_push import (
    build_account_payload,
    build_position_payloads,
    build_income_payloads,
)


def test_account_payload_maps_raw_fapi_fields():
    info = {
        "totalWalletBalance": "120.50",
        "totalMarginBalance": "122.10",
        "totalUnrealizedProfit": "1.60",
        "availableBalance": "100.00",
        "ignored": "x",
    }
    assert build_account_payload(info) == {
        "walletBalanceUsd": 120.50,
        "marginBalanceUsd": 122.10,
        "unrealizedPnlUsd": 1.60,
        "availableBalanceUsd": 100.00,
    }


def test_account_payload_tolerates_missing_fields():
    assert build_account_payload({}) == {
        "walletBalanceUsd": 0.0,
        "marginBalanceUsd": 0.0,
        "unrealizedPnlUsd": 0.0,
        "availableBalanceUsd": 0.0,
    }


def test_position_payloads_skip_flat_and_keep_sign():
    positions = [
        {"info": {"symbol": "BTCUSDT", "positionAmt": "0", "positionSide": "BOTH"}},  # flat → skipped
        {"info": {
            "symbol": "BTCUSDT", "positionAmt": "-0.010", "positionSide": "BOTH",
            "entryPrice": "68000", "markPrice": "67800", "unRealizedProfit": "2.0",
            "liquidationPrice": "90000", "leverage": "20",
        }},
    ]
    out = build_position_payloads(positions)
    assert len(out) == 1
    p = out[0]
    assert p["symbol"] == "BTCUSDT"
    assert p["positionAmt"] == -0.010          # sign preserved (short)
    assert p["entryPrice"] == 68000.0
    assert p["liquidationPrice"] == 90000.0
    assert p["leverage"] == 20.0


def test_position_liq_zero_becomes_null():
    positions = [{"info": {
        "symbol": "ETHUSDT", "positionAmt": "1.0", "liquidationPrice": "0",
        "entryPrice": "3000", "markPrice": "3010", "unRealizedProfit": "10", "leverage": "10",
    }}]
    out = build_position_payloads(positions)
    assert out[0]["liquidationPrice"] is None


def test_position_falls_back_to_unified_fields():
    # No raw info beyond positionAmt — unified fields fill the rest.
    positions = [{
        "info": {"positionAmt": "0.5"},
        "symbol": "SOLUSDT", "entryPrice": 150.0, "markPrice": 152.0,
        "unrealizedPnl": 1.0, "liquidationPrice": 90.0, "leverage": 5,
    }]
    out = build_position_payloads(positions)
    assert out[0]["symbol"] == "SOLUSDT"
    assert out[0]["entryPrice"] == 150.0
    assert out[0]["unrealizedPnlUsd"] == 1.0


def test_income_payloads_map_and_preserve_time():
    rows = [
        {"tranId": 9001, "symbol": "BTCUSDT", "incomeType": "REALIZED_PNL",
         "income": "5.0", "asset": "USDT", "time": 1717000000000},
        {"tranId": 9001, "symbol": "BTCUSDT", "incomeType": "COMMISSION",
         "income": "-0.20", "asset": "USDT", "time": 1717000000000},
        {"tranId": 9002, "symbol": "BTCUSDT", "incomeType": "FUNDING_FEE",
         "income": "-0.30", "asset": "USDT", "time": 1717003600000},
    ]
    out = build_income_payloads(rows)
    assert len(out) == 3
    assert out[0] == {"tranId": 9001, "symbol": "BTCUSDT", "incomeType": "REALIZED_PNL",
                      "incomeUsd": 5.0, "asset": "USDT", "ts": 1717000000000}
    # funding paid keeps its negative sign for the server to split
    assert out[2]["incomeUsd"] == -0.30


def test_income_payloads_skip_malformed():
    rows = [
        {"tranId": "not-an-int", "incomeType": "REALIZED_PNL", "income": "1", "time": "x"},
        {"tranId": 1, "incomeType": "REALIZED_PNL", "income": "2.0", "time": 1717000000000},
    ]
    out = build_income_payloads(rows)
    assert len(out) == 1
    assert out[0]["incomeUsd"] == 2.0
