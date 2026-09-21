# Boot-resume — adopt an open position instead of flattening it

Status: **PHASE A BUILT, NOT DEPLOYED.** Written 2026-09-21, the same day reprotect Phase 2 was
armed (droplet `1577c5d`), which is what makes this safe to attempt at all.

**Implementation:** branch `feat/boot-resume-observe`, commit **`3243f97`**, branched off
`droplet` (worktree `/Users/god/Desktop/work/snapback-boot-resume-wt`). 153 insertions and **zero
deletions** in `bot.py` — the flatten block is byte-identical and the new branch sits in front of
it. Suite **392 → 415, exit 0** on both the feature branch and the droplet baseline, zero
regressions. `boot_resume.observe_only: true`, so behaviour is unchanged.

**Still open:** §4 H1 is now implemented (a `qty` field was added to `active_bracket`); §5's Phase A
gate has not been run at all — it needs the ≥5 deliberate restarts. Nothing is on the droplet.

> ## ⚠️ READ THIS BEFORE WRITING ANY CODE
>
> **All line numbers below are on the `droplet` branch, because that is what runs live.**
>
> `bot.py` differs between `main` and `droplet` by **306 insertions / 44 deletions** — every
> reference in this document sits ~250 lines lower on `main`, and some of it **does not exist on
> `main` at all** (`_daily_anchor_equity` and the whole daily-loss breaker are droplet-only; grep
> `main` and that machinery looks dead).
>
> A `bot.py` feature developed on `main` and cherry-picked is therefore a merge hazard, unlike the
> config-only Phase 2 cherry-pick. **Decide the branch strategy before implementing**, and verify
> the merged tree by running the suite on it — not by the cherry-pick succeeding. That exact trap
> has already bitten once: a cherry-picked block needed an `import json` the droplet's test file
> lacked, and only running the suite caught it.

---

## 1. The problem, stated as evidence rather than theory

`boot()` closes any open position it finds on a live restart. The path is unconditional
(`bot.py:486` onward, droplet): find `pos.side != "flat"`, call `close_position(..., close_leg="bf")`,
record a `boot_flatten` event. **There is no adopt branch.** A restart is a market close, every
time.

What it has actually cost, read from the leg state DBs:

| when | leg | what died |
|---|---|---|
| 2026-07-21 | v1 | unattended-upgrade restart *(from memory; the other three were read from the `events` tables)* |
| 2026-08-10 | sol | `boot_flatten`, event 3 |
| 2026-08-26 | donchian | `boot_flatten`, event 7 |
| 2026-09-11 | sol | `boot_flatten`, event 7, −$0.82 |

For sol this dominates the live record: **5 entries, 3 of them killed by infrastructure** (one
time-stop-zero, two boot-flattens) rather than by the market. Reading that leg's win rate as a
verdict on the strategy is reading a denominator that is mostly operations.

**What is already fixed, and must not be double-counted.** `needrestart`
(`/etc/needrestart/conf.d/50-snapback.conf`, 2026-09-16) stopped apt/unattended-upgrades from
auto-restarting the legs. That closed the *most frequent* trigger, not the mechanism. Still
uncovered: a crash, a systemd auto-restart, an OOM kill, and a **reboot** — the needrestart fix
explicitly does not cover a reboot, because `boot()` still flattens.

`tools/leg_safe_restart.py` covers the *operator* case properly: it refuses unless the leg is flat
on the exchange. **So the remaining exposure is precisely the restarts nobody chose.**

---

## 2. Why this is feasible now, and was not before

Flatten is the safe default because a freshly booted process cannot trust that a position it did
not open still has a live bracket behind it. Arming reprotect removed that objection: the bot can
now *detect* a missing bracket across both order books and *restore* it.

`_maybe_reprotect` (`bot.py:1014-1170`, droplet) already contains every primitive an adopt gate
needs:

1. reads the **exchange** position, not the DB
2. reads `meta.active_bracket`, the params stashed at entry (`bot.py:966`)
3. **guards identity** — side must match (`bot.py:1071`), entry price within 2%
4. reads **both** order books; `fetch_algo_orders` reports success separately, so an unreadable
   algo book is `UNKNOWN`, never "no bracket" (`bot.py:1086`)
5. `bracket_state(...)` merges the books and decides `intact`
6. cancel-then-confirm-then-replace, so a partial cancel cannot leave a stale leg
7. `max_replaces_per_position`, persisted **inside** `active_bracket` (`bot.py:1170`) so a restart
   cannot reset it

Adoption is not new machinery. It is **calling validation that already exists, at boot, and
branching on it.**

### Three facts that make this cheaper than expected

- **The time stop survives a restart.** `_maybe_time_stop` (`bot.py:1587`, droplet) derives the
  position's age from `SELECT ts FROM fills WHERE reason='entry' ORDER BY id DESC LIMIT 1` — the
  **fills table**, not in-memory state. An adopted position keeps its true age and a max-hold exit
  still fires on schedule. This was the largest correctness risk and it is already handled.
  *(Verified on droplet, not assumed from main.)*
- **The daily-loss breaker also survives.** `_daily_anchor_equity` (`bot.py:712`, droplet) persists
  `daily_anchor_date` / `daily_anchor_equity` in `state.meta` and re-anchors **only on UTC date
  rollover**. Confirmed live: v1's anchor was set at 00:00 UTC on 2026-09-21 and was *not* reset by
  the 09:15 restart. Adoption does not move the breaker's baseline.
- **We would only ever adopt our own positions.** The identity guard requires a matching
  `active_bracket`, which exists only for a bracket *this bot placed*. A manually opened position
  has none and falls through to the existing flatten path, unchanged.

That last point also defuses the `_open_entry_fill` concern. Its docstring (`bot.py:1560`, droplet)
warns that "a position adopted at boot has no entry row of its own" — true for a *manual* position,
but a position adopted under this plan was opened by this bot and **does** have its entry row.

---

## 3. The design

One new gate, called from the existing `pos.side != "flat"` branch in `boot()`.

```
boot():
  pos = fetch_position()
  if pos.side == "flat":   -> unchanged (orphan sweep)
  elif dry_run:            -> unchanged (leave alone)
  else:
      if _can_adopt(pos):  -> adopt: log, emit `boot_adopt`, fall through to the
                              tick loop, which owns it from there
      else:                -> FLATTEN, exactly as today
```

`_can_adopt(pos)` returns True only if **all** hold. Any failure, any exception, any unknown →
`False` → flatten. **The current behaviour must remain the default.**

1. `reprotect.enabled` is true **and** `observe_only` is false for this leg.
   *Without an armed re-placer, adopting a possibly-unprotected position is strictly worse than
   closing it.* This makes the gate fail-closed on donchian and sol by construction — neither
   config has a `reprotect:` key, and `rp.get("enabled", False)` is fail-closed.
2. `meta.active_bracket` exists, parses, and `side` matches the exchange position.
3. Entry price within 2% of `pos.entry_price` — the guard reprotect already uses.
4. **`qty` sanity** — see H1; reprotect does not check this and adoption should.
5. The algo book is **readable**. `algo_ok == False` → flatten. An unknown bracket state at boot is
   not a risk worth carrying.
6. `bracket_state(...).intact`, **or** not intact and `reprotect_count < cap`, so the tick loop's
   first `_maybe_reprotect` can restore it.

### The `active_bracket` edge cases, stated explicitly

A reader will assume adoption covers these. It does not — **all three flatten**, and the doc says
so on purpose:

- **Empty `active_bracket`, exchange NOT flat.** This is reachable: `_maybe_reprotect` clears the
  record to `""` on the first flat tick (`bot.py:1059`, droplet), and v1's meta holds exactly that
  right now. It is also what a dropped exit leaves behind — donchian's 4 Sep close wrote no fill
  and no event. Condition 2 fails ⇒ **flatten.** Correct, and deliberately not "adopt anyway".
- **Stale `active_bracket` from a previous position** whose side and entry price happen to land
  inside the 2% guard. The guard is a filter, not a proof of identity. H1's `qty` check is the
  second line of defence here, which is the main reason to add it.
- **Manually opened position.** No `active_bracket` at all ⇒ **flatten**, unchanged from today.

On adopt, emit a `boot_adopt` event mirroring `boot_flatten`'s shape, so the dashboard and any
later audit can distinguish "resumed" from "never restarted."

---

## 4. Hazards and open decisions

**H1 — qty is not guarded.** Reprotect checks side and entry price but never quantity, which is
safe *for reprotect* because it re-places against live `pos.qty`. Adoption inherits more, so a
partially-closed position could be adopted against a stale size. **Decision needed:** recommend
requiring `pos.qty` within one `qty_step` of the stashed value, flatten otherwise.

**H2 — `_reprotect_capped_alerted` resets on boot.** A position already at its re-place cap would
re-alert once after adoption. Cosmetic, but seed it from `reprotect_count` rather than `False`.

**H3 — the adopt/flatten race.** `tools/leg_safe_restart.py` documents the mirror of this and
deliberately does not close it: a position can open between check and action. Adoption is the safer
side of that race — doing nothing beats closing at market — but the gate must re-read the position
immediately before committing, as that script does.

**H4 — this widens what a bad detector can cost.** Today a false "bracket intact" costs an
unprotected position until the time stop. With adoption it also costs a *silently resumed* one. The
brakes are the same three Phase 1 validated, which is the argument for the rollout below.

**H5 — branch divergence.** See the banner. This is the highest-probability way to ship a bug here.

---

## 5. Rollout — mirror reprotect's own discipline

Reprotect earned its arming with an observe-only phase and a falsifiable gate. This should too.

- **Phase A, observe-only.** Add `boot_resume: {enabled: true, observe_only: true}`. On a restart
  with an open position, evaluate `_can_adopt`, **log the verdict and the full reason**, then
  flatten anyway. Costs nothing, changes nothing.
- **Gate to pass — needs a number, or it will stall or get waved through.** Reprotect's Phase 1
  was checkable ("zero lines across one full position"); this must be too:
  **≥5 deliberate restarts while holding a position, every verdict inspected by God, zero
  disagreements.** Unlike Phase 1 this **cannot be proven by silence** — the log line *is* the
  evidence, so a run with no restarts proves nothing. Drive it with deliberate test restarts on a
  dry-run leg; do not wait for organic ones.
- **Phase B, armed**, v1 only, exactly as reprotect was.

**Test without risking money:** dry-run legs already leave positions alone at boot, so the adopt
logic can be exercised against a dry-run leg and against synthetic `active_bracket` / order-book
fixtures in the unit suite. Both book shapes are well understood now — `algo_bracket_leg` keys on
the COID suffix.

---

## 6. What NOT to do

- **Do not copy `reprotect:` into `params_donchian.yaml` or `params_sol_supertrend.yaml`** to make
  this apply there. That arms an unvalidated detector on real money. Those legs stay fail-closed
  until they have their own Phase 1.
- **Do not run pytest on the droplet.** Parts of the suite touch real sqlite and the live state DBs
  are in `/root/snapback-btc/data/`. Verify by grep/byte comparison against a local worktree.
- **Do not gate adoption on `strategy_name`.** Same reasoning recorded inline in
  `config/params.yaml` for reprotect: a hard strategy gate silently *disables* protection on a
  legitimate switch, the failure direction that actually costs money.
- **Do not enable boot-resume on a config whose `strategy_name` implies `place_tp: false`** until
  the half-bracket path has been exercised. `config/params.yaml`'s own shipped comment names
  `donchian-v3` as the sharpest of the three dispatch paths precisely because it places **no TP
  leg**, and "nothing here has been exercised against a half-bracket." Donchian's live
  `active_bracket` read today carries exactly `"place_tp": false`. Condition 6 delegates to
  `bracket_state(..., place_tp)`, so adoption would inherit that untested shape. Unreachable today
  (donchian has no `reprotect:` key) — but `config/params.yaml` is **not pinned to one strategy**,
  which is the same trap the commit itself records.
- **Do not treat an unreadable algo book as "no bracket."** That is the July `-4045` bug, and at
  boot it would be worse.
- **Do not verify the deploy with `git merge-base --is-ancestor`.** `droplet` is a cherry-pick
  branch, so SHAs differ and that test reports present commits as ABSENT. Grep for the code.

---

## 7. Related

- `REPROTECT_ALGO_AWARE_PLAN.md` — the detector this depends on
- `config/params.yaml` — the Phase 1 evidence chain, recorded inline
- `tools/leg_safe_restart.py` — the operator-side guard, and the race discussion
- `bot.py:486-533` (droplet) — the flatten path this modifies; the branch opens at `if pos.side != "flat"` on 487
- `bot.py:1014-1170` (droplet) — `_maybe_reprotect`, the validation being reused
