# snapback-btc — Claude Code Operator Guide

This repo runs a deterministic Binance Futures BTC/USDT perpetual bot. You (Claude Code) are the **monitor cockpit**, NOT the trading runtime. The bot makes all decisions in plain Python — never call out to an LLM from the trading loop.

## Active strategy: multifactor-v1
- Locked 2026-05-17 for 90-day testnet deploy. See `DEPLOY.md` and `PATH2_RESULTS.html`.
- Config in `config/params.yaml`. Backtest: 5 OOS windows → +55.73% compounded, 4 of 5 positive.
- Worst window: 2024 H1 (-12.56%, chop). Kill switch fires at -15% equity drawdown.
- Old strategies (carry, donchian, fmom, snapback-v2, multifactor v2/mtf) **removed from repo** post-cleanup. Recover from git history if needed.

## Your role
1. Read `logs/*.jsonl` and `data/state.db`, summarize what happened.
2. Diagnose anomalies, propose tweaks to `config/params.yaml`.
3. Generate weekly Plotly HTML reports in `reports/`.
4. **NEVER edit `risk.py`** — those are hard ceilings. Propose changes to the user instead.
5. **NEVER place orders.** Read-only against state.db and the exchange.

## Hard rules
- No order placement from Claude Code, ever.
- No edits to `risk.py` (git pre-commit hook will reject anyway unless `RISK_REVIEW=1`).
- **Leverage ceiling is 20x** (raised from 3x in P3.4 per explicit user decision).
  Do NOT lower it back to 3x without the user explicitly asking. The user
  prefers 20x as the permanent default; backtests showed it doesn't change
  carry-v2 returns but the user wants the capital efficiency for live deploy.
- No changing `BINANCE_ENV` from testnet to mainnet without the full `/promote-mainnet` checklist.
- If `data/HALT` exists, do NOT remove it without explicit user ask. Bot polls every 5s and exits.
- Mainnet requires `confirm_mainnet.lock` to exist. If user asks you to create it, run `verify_identity(action_type="bot_mainnet")` first.

## Memory protocol (CogniLayer)
- BEFORE diagnostics: `memory_search("snapback ...")` for past gotchas.
- AFTER finding a root cause: `memory_write(content=..., type="error_fix", tags="snapback-btc,<topic>")`, end body with `Search: keyword1, keyword2`.
- BEFORE switching to mainnet: `verify_identity(action_type="bot_mainnet")`. If BLOCKED, STOP and read the target back to the user.

## Slash commands (`.claude/commands/`)
- `/status` — last 24h: open positions, fills, P&L, equity delta, heartbeat age
- `/diagnose` — triage: heartbeat, last error, drawdown, recent signals; proposes fixes (does not apply)
- `/backtest` — re-run backtest with current `config/params.yaml`, compare to live
- `/weekly-report` — Plotly HTML in `reports/`, plus markdown summary
- `/halt` — `touch data/HALT` (bot polls every 5s, exits clean)
- `/promote-mainnet` — guided pre-mainnet checklist, requires `verify_identity`

## Conventions
- Logs are JSONL, one event per line. Use `jq` when sampling.
- All times UTC.
- State changes always go through `data/state.db` (SQLite, WAL mode).
- Heartbeat: `data/heartbeat` mtime updated every loop tick (~5s). Stale >90s = bot down.

## Stack reference
Python 3.11+, `ccxt`, `pandas`, `pandas-ta`, `backtesting.py` (research), `freqtrade` (live later), `plotly`, `structlog`. Alerts via `alerts.py` (stdlib `smtplib`, Gmail-friendly).

See `../btc-bot-ultraplan.md` for the full strategy + phase plan.

# === COGNILAYER (auto-generated, do not delete) ===

## CogniLayer v4 Active
Persistent memory + code intelligence is ON.
ON FIRST USER MESSAGE in this session, briefly tell the user:
  'CogniLayer v4 active — persistent memory is on. Type /cognihelp for available commands.'
Say it ONCE, keep it short, then continue with their request.

## MEMORY HIERARCHY (CRITICAL — ALWAYS FOLLOW)

You have TWO memory systems. Use BOTH, but with clear priority:

### PRIMARY: CogniLayer MCP (memory_search / memory_write)
- ALWAYS use FIRST for both reading and writing
- FTS5 + vector search, heat decay, 14 fact types, code intelligence
- On-demand — loads only relevant facts (~500 tokens instead of tens of thousands)
- Store here: decisions, gotchas, patterns, error_fixes, api_contracts, procedures

### SECONDARY (FALLBACK): Auto-memory (MEMORY.md files)
- Use when CogniLayer MCP is unavailable, fails, or returns empty
- MEMORY.md is loaded into context ALWAYS at session start — keep it SHORT (max 30 lines)
- Store here only: critical user feedback, deploy workflow, 1-line pointers to CogniLayer

### RULES:
1. READING: memory_search(query) FIRST → if empty/error → read MEMORY.md files
2. WRITING: memory_write() ALWAYS → ALSO to auto-memory ONLY if critical user feedback/rule
3. NEVER duplicate content — if fact is in CogniLayer, put only a 1-line pointer in auto-memory
4. Auto-memory MEMORY.md is an INDEX, not a database — format: `- [topic] → /recall keyword`
5. If CogniLayer MCP fails → USE auto-memory as base and alert user about MCP issue

### CHECK (every ~10 prompts or before ending work):
- Did I save new findings to memory_write()? If not → save NOW
- Is session bridge current? If not → session_bridge(action="save")
- DO NOT wait for end of session — save continuously, session may crash

## Tools — HOW TO WORK

FIRST RUN ON A PROJECT:
When DNA shows "[new session]" or "[first session]":
1. Run /onboard — indexes project docs (PRD, README), builds initial memory
2. Run code_index() — builds AST index for code intelligence
Both are one-time. After that, updates are incremental.
If file_search or code_search return empty → these haven't been run yet.

UNDERSTAND FIRST (before making changes):
- memory_search(query) → what do we know? Past bugs, decisions, gotchas
- code_context(symbol) → how does the code work? Callers, callees, dependencies
- file_search(query) → search project docs (PRD, README) without reading full files
- code_search(query) → find where a function/class is defined
Use BOTH memory + code tools for complete picture. They are fast — call in parallel.

BEFORE RISKY CHANGES (mandatory):
- Renaming, deleting, or moving a function/class → code_impact(symbol) FIRST
- Changing a function's signature or return value → code_impact(symbol) FIRST
- Modifying shared utilities used across multiple files → code_impact(symbol) FIRST
- ALSO: memory_search(symbol) → check for related decisions or known gotchas
Both required. Structure tells you what breaks, memory tells you WHY it was built that way.

AFTER COMPLETING WORK:
- memory_write(content) → save important discoveries immediately
  (error_fix, gotcha, pattern, api_contract, procedure, decision)
- session_bridge(action="save", content="Progress: ...; Open: ...")
DO NOT wait for /harvest — session may crash.

## SHORT SESSIONS = BETTER PERFORMANCE
- With 200K context, session compresses sooner → faster responses
- CogniLayer bridge + memory_search replaces lost history for ~2K tokens
- After completing a coherent block of work: save bridge → suggest user starts new session
- Use /compact when session grows and work is not yet done

SUBAGENT MEMORY PROTOCOL:
When spawning Agent tool for research or exploration:
- Include in prompt: synthesize findings into consolidated memory_write(content, type, tags="subagent,<task-topic>") facts
  Assign a descriptive topic tag per subagent (e.g. tags="subagent,auth-review", tags="subagent,perf-analysis")
- Do NOT write each discovery separately — group related findings into cohesive facts
- Write to memory as the LAST step before return, not incrementally — saves turns and tokens
- Each fact must be self-contained with specific details (file paths, values, code snippets)
- When findings relate to specific files, include domain and source_file for better search and staleness detection
- End each fact with 'Search: keyword1, keyword2' — keywords INSIDE the fact survive context compaction
- Record significant negative findings too (e.g. 'no rate limiting exists in src/api/' — prevents repeat searches)
- Return: actionable summary (file paths, function names, specific values) + what was saved + keywords for memory_search
- If MCP tools unavailable or fail → include key findings directly in return text as fallback
- Launch subagents as foreground (default) for reliable MCP access — user can Ctrl+B to background later
Why: without this protocol, subagent returns dump all text into parent context (40K+ tokens).
With protocol, findings go to DB and parent gets ~500 token summary + on-demand memory_search.

BEFORE DEPLOY/PUSH:
- verify_identity(action_type="...") → mandatory safety gate
- If BLOCKED → STOP and ask the user
- If VERIFIED → READ the target server to the user and request confirmation

## VERIFY-BEFORE-ACT
When memory_search returns a fact marked ⚠ STALE:
1. Read the source file and verify the fact still holds
2. If changed → update via memory_write
3. NEVER act on STALE facts without verification

## Process Management (Windows)
- NEVER use `taskkill //F //IM node.exe` — kills ALL Node.js INCLUDING Claude Code CLI!
- Use: `npx kill-port PORT` or find PID via `netstat -ano | findstr :PORT` then `taskkill //F //PID XXXX`

## Git Rules
- Commit often, small atomic changes. Format: "[type] what and why"
- commit = Tier 1 (do it yourself). push = Tier 3 (verify_identity).

## Project DNA: snapback-btc
Stack: unknown
Style: [unknown]
Structure: .githooks, .pytest_cache, config, data, deploy, exchange, logs, reports
Deploy: [NOT SET]
Active: [new session]
Last: [first session]

## Session Continuity
State: No changes or facts in this session.

# === END COGNILAYER ===
