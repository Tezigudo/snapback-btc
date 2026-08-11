# Re-enabling `_maybe_reprotect` with algo-aware bracket detection

**Status:** scoped, not started. **Date:** 2026-08-11.
**Blast radius:** real-money order placement on a live position. This is the exact
code that caused the 2026-07-22 `-4045` incident, so the plan below is deliberately
conservative about re-enabling it.

---

## Why bother

`_maybe_reprotect` (bot.py:695) exists to restore a reduce-only SL/TP bracket that
disappeared **without a fill** — a manual cancel on the Binance app, or a leverage
change (Binance auto-cancels *all* open orders on a leverage change).
`_detect_bracket_exit` only reacts to a bracket FILL (open→flat); it never notices a
cancel. With reprotect disabled, an externally-cancelled bracket leaves a live
position **silently unprotected until the time stop** — which, per the 2026-08-11
measurement, fires 0 times in 4.6 years. So in practice: unprotected indefinitely.

Second, smaller payoff: `_maybe_reprotect` holds the **only** line that clears
`meta.active_bracket` when flat (bot.py:720). With it disabled, that record goes
stale after every trade and has needed manual clearing twice (2026-08-05, and again
after the 08-10 SL). Cosmetic — `json.loads` is guarded — but recurring.

## What actually broke in July

`_place_brackets` creates STOP_MARKET / TAKE_PROFIT_MARKET orders, which Binance
parks on the **algo/conditional** endpoint (`/fapi/v1/openAlgoOrders`).
`fetch_open_orders()` (`/fapi/v1/openOrders`) does not return them. So
`bracket_is_intact(fetch_open_orders(...))` was `False` on every poll while a
perfectly healthy bracket rested → re-place every 60s → `-4045 Reach max stop order
limit` → alert spam (ids 89–102, 07-22 07:12–07:26). Mitigation was to comment out
the call site (main `2cbc003`).

## The part that is NOT a simple endpoint swap

Swapping `fetch_open_orders` for `fapiPrivateGetOpenAlgoOrders` is **not enough**,
because the classifier cannot read the algo shape.

`reduce_only_bracket_leg()` (bot_internals.py:522) decides SL-vs-TP from
`info.type` / `info.origType` containing `"STOP"` / `"TAKE_PROFIT"`, and reads
`reduceOnly` from `order["info"]`.

A **live** algo row captured from v1 on 2026-08-10 (both resting legs):

```json
{"algoId": "2000001347733291", "clientAlgoId": "snap-v1-1786231808674-t",
 "side": "SELL", "positionSide": "BOTH", "triggerPrice": "66858.5",
 "strategyType": null, "origQty": null, "reduceOnly": true,
 "algoStatus": "NEW", "bookTime": null}
```

Three mismatches against the classifier:

| classifier expects | algo row gives |
|---|---|
| `info.type` / `info.origType` containing STOP / TAKE_PROFIT | **no type field at all**; `strategyType` was `null` on both live rows |
| `reduceOnly` nested under `info` | `reduceOnly` top-level (and a real bool, not a string) |
| — | `clientAlgoId`, not `clientOrderId` |

So feeding algo rows to the existing classifier returns `None` for every leg, i.e.
"bracket missing" — **the same false negative as before, just from a different
cause.** Anyone who "fixes" this by only changing the fetch call reproduces the
incident.

### The discriminator that does work

The bot tags its own legs. `_place_brackets` uses `_coid(root, "s")` for the stop
and `_coid(root, "t")` for the take-profit (binance_client.py:379/389), producing
`snap-v1-{root}-s` / `-t`. That matches the live `clientAlgoId` values exactly.
COID suffix is therefore the reliable leg discriminator on the algo endpoint, and it
has the bonus of being **scoped to our own orders** — a manual order placed from the
phone will not carry the prefix and must not count toward "bracket intact".

## Work items

1. **`exchange/binance_client.py`** — extract a read-only `fetch_algo_orders(symbol)`
   from the existing `_cancel_algo_orders` (which already calls
   `fapiPrivateGetOpenAlgoOrders` and already degrades to a no-op with a warning on
   an old ccxt build). Return `[]` on any failure, never raise.
2. **`bot_internals.py`** — new `algo_bracket_leg(row, coid_prefix) -> 'sl'|'tp'|None`
   keyed on the `clientAlgoId` suffix + `reduceOnly`, and a
   `bracket_state(plain_orders, algo_rows, coid_prefix, place_tp)` that merges both
   sources. Keep `reduce_only_bracket_leg` untouched for the plain path — donchian
   and supertrend depend on it.
3. **`bot.py`** — `_maybe_reprotect` reads both sources; the post-cancel
   "did anything survive?" re-check must also read both, or the duplicate-order guard
   is as blind as the detector was.
4. **Re-place circuit breaker (NEW, non-negotiable).** The 60s throttle did not stop
   July — it just paced the spam. Add a hard per-position cap (proposal: 3 re-places
   per `signal_id`, persisted in `active_bracket`), after which it alerts once and
   stops trying. A detector bug must not be able to reach `-4045` again.
5. **Tests** — the live algo payload above as a fixture; assert a healthy algo
   bracket reads INTACT (the exact July false-negative), a genuinely missing leg reads
   missing, a foreign-prefix order does not count, and the breaker stops at N.
6. **`config/params.yaml`** — `reprotect.enabled` + `reprotect.observe_only`.

## Rollout — observe before acting

The failure mode is a detector that is wrong in the unsafe direction, and it is not
provable from unit tests alone because the truth lives on the exchange.

- **Phase 1 (observe_only):** ship with the call site re-enabled but the re-place
  branch replaced by a log line — *"would re-place: SL=<present> TP=<present>"* — on
  every poll. Ship while flat, then wait for at least one full live position.
  **Pass condition: zero "would re-place" lines across a complete trade.** That is
  the direct falsification of the July bug.
- **Phase 2:** flip `observe_only: false`. Breaker stays.

Phase 1 is free — it places no orders — and it is the only thing that actually proves
the detector against a real resting bracket.

## Deploy notes

- Touches `bot_internals.py` (HARD-FAIL in `check_deploy_drift.sh`) and `bot.py`
  (warn-only, and the droplet copy diverges by ~322 lines — cherry-pick, do not copy).
- Requires a **bot restart**, so it needs a **flat leg**. Reuse
  `v1-deferred-restart.timer`; delete `data/v1_deferred_restart.done` first or it
  exits immediately on its stamp.
- Merge to main **before** pushing to droplet — the drift script compares local
  `main`, so the other order reads a false clean.

## Explicitly NOT in scope

**The Tier-2 relay is already algo-aware** and needs no work.
`tools/consolidate_futures_push.py:356-357` reads *both* `fetch_open_orders()` and
`fapiPrivateGetOpenAlgoOrders()`, and `build_bracket_map` accepts both row shapes.
Earlier notes pairing "reprotect fix" with "unblocks Tier-2 SL/TP" are **stale** —
that half landed with the 2026-07-26 algo-aware sweep work. Verified by reading the
live cron path 2026-08-11.

## Recommendation

Worth doing, but not urgent. The bug it guards against is an *external* cancel, which
has never been observed on this account; the cost of getting it wrong is live-order
spam on a real-money leg. Do it as a deliberate piece with Phase 1 observation — not
bundled into an unrelated deploy.
