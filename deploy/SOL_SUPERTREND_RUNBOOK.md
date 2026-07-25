# sol_supertrend leg — deploy runbook

Everything in the repo is built, tested and gated. What is left is droplet-side:
credentials, funding, dry-run, enable. Run these **on the droplet**
(`/root/snapback-btc`) unless noted.

Evidence for the strategy: `SOL_LEG_VERDICT.md`. Parity: `tools/supertrend_parity.py`.

---

## Two files get confused. They are not the same thing.

| | `config/params_cnh_hybrid_short.yaml` | `.env.cnh_short` |
|---|---|---|
| Exists in the repo? | **Yes** | **No** |
| Holds | strategy params — `symbol: BTC/USDT:USDT`, `strategy_name: cnh-hybrid-short-v1`, risk 2.75%, lev 20 | the leg's Binance **API key + secret** |
| Reusable for sol_supertrend? | **No** — it selects the cnh-short strategy on BTC. The SOL leg uses `config/params_sol_supertrend.yaml`. | **Yes, if it exists on the droplet** — that is the only thing worth copying. |

So "use the cnh_short config" is right about the *credentials* and wrong about the
*strategy config*. The strategy config for this leg already exists and is
committed; only the key is missing.

---

## Step 0 — check what the droplet actually has (no secrets printed)

```bash
ls -l /root/snapback-btc/.env.*
```

Two things this answers:

1. **`.env.donchian`** — almost certainly present, since the donchian leg is live
   in its own sub-account. **Confirm it anyway.** The per-instance env guard
   (`bot._main()`) now refuses to boot any non-v1 leg without its env file, so if
   it is missing the *live donchian leg will not restart* after you pull. This is
   the one hard prerequisite before `git pull`.
2. **`.env.cnh_short`** — if present, Step 1a. If absent, Step 1b.

---

## Step 1a — `.env.cnh_short` EXISTS: derive from it

```bash
cd /root/snapback-btc
sed 's/^CONSOLIDATE_SOURCE=.*/CONSOLIDATE_SOURCE=snapback-sol-supertrend/; \
     s/^ALERT_TAG=.*/ALERT_TAG=snapback-sol-st/' \
  .env.cnh_short > .env.sol_supertrend
chmod 600 .env.sol_supertrend
grep -c BINANCE_API_KEY .env.sol_supertrend      # expect 1
grep -E '^(CONSOLIDATE_SOURCE|ALERT_TAG)=' .env.sol_supertrend
```

**Then verify the key is NOT the v1 account** (Step 2). If cnh_short was never
funded, its key may have been left pointing at the main account — copying that
would put SOL orders inside your live BTC account. Step 2 is what catches it.

## Step 1b — `.env.cnh_short` is absent: new sub-account key

Binance → sub-accounts → create/pick one for this leg → API key with **futures
trading enabled, withdrawals disabled, droplet IP allowlisted**.

```bash
cd /root/snapback-btc
cp .env.sol_supertrend.example .env.sol_supertrend
chmod 600 .env.sol_supertrend
$EDITOR .env.sol_supertrend        # paste BINANCE_API_KEY / BINANCE_API_SECRET
```

`CONSOLIDATE_SOURCE` and `ALERT_TAG` are already filled in the template.

`.env.*` is gitignored as of `d6f015c`, so this file will not be committed.

---

## Step 2 — pre-flight (read-only, places no orders)

```bash
.venv/bin/python tools/preflight_live.py --instance sol_supertrend
```

Checks env load, the SOL market spec and constraints, authenticated balance and
position reads, sizing at current price, and the risk.py ceilings — against
**this leg's own** `.env.sol_supertrend`, so the equity it prints is the
sub-account's.

**The isolation check:** compare the equity it reports with v1's.

```bash
.venv/bin/python tools/preflight_live.py --instance v1 | grep -i equity
```

If the two match, `.env.sol_supertrend` is pointing at the v1 account — **stop
and fix it** before going further.

Expect `min_qty=0.01, min_notional=$5.0, qty_step=0.01, price_step=0.01`. If it
says `min_notional=$50.0` you are running pre-`d6f015c` code.

---

## Step 3 — fund it

**$50–100 USDT.** Not the $200 quoted earlier — that was an artifact of BTC's
$50 min-notional being applied to SOL, fixed in `d6f015c`.

| Equity | signals skipped | truncated by `MAX_NOTIONAL_USD=500` |
|---|---|---|
| $25 | 0% | 0% ← hard floor |
| **$50–100** | **0%** | **0%** ← recommended |
| $500 | 0% | 9% |
| $800 | 0% | 76% ← do not exceed ~$500 |

$25 works mechanically but quantity rounds to 0.01 SOL steps, so its 0.12-SOL
median position carries ~8% sizing granularity vs ~2% at $100. Rounding always
floors, so the error is toward *less* risk.

---

## Step 4 — dry-run through at least one Supertrend flip

```bash
cd /root/snapback-btc
.venv/bin/python -m bot --instance sol_supertrend --dry-run
```

Watch for: `strategy=supertrend`, `symbol=SOL/USDT:USDT`, the per-instance env
line, and constraints matching Step 2. Then leave it running.

Flips come roughly every 12 days (median gap; 27.5 orders/yr), so a meaningful
dry-run is **days, not hours**. As of 2026-07-25 the band is DOWN since the
07-24 12:00 flip, so the next event is a flip to LONG.

Confirm in the log, on a flip: a `Signal` line with qty/sl/tp, and `notional`
between $5 and $500.

---

## Step 5 — go live

```bash
sudo cp deploy/snapback-sol-supertrend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now snapback-sol-supertrend
journalctl -u snapback-sol-supertrend -f
```

Rollback: `sudo systemctl disable --now snapback-sol-supertrend`, or
`touch data/HALT` to flatten and exit cleanly (per-leg HALT — it will not cascade
to the BTC legs, per the 2026-07-01 fix).

---

## Known characteristics — do not be surprised by these

* **Win rate 37%.** Worst backtested losing streak is 6 trades.
* **Bear-biased.** Shorts contributed +571% vs longs' +108% over a span where SOL
  fell 56%, and the leg was **-5.0% in SOL's +804% year**. Expect "makes money
  when SOL falls, roughly flat when it rips" — not CAGR 30%.
* **Quiet.** ~27 orders/yr, median 12 days apart, max gap 43 days. Silence is
  normal, not a fault.
* `MAX_CONSECUTIVE_LOSSES = 4` in risk.py is **declared but never enforced** (no
  check function, no caller). If it is ever wired up, this leg trips it — its
  natural worst streak is 6.

## Do NOT enable

`cnh_short_sol` (`config/params_cnh_hybrid_short_sol.yaml`). The SOL allowlist
entry un-blocked it, but it measured CAGR 1.3% with **1,360 days underwater**.
The allowlist is no longer what prevents it — not enabling the unit is.
