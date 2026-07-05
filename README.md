# Dreamland

[![CI](https://github.com/Kelsidavis/Dreamland/actions/workflows/ci.yml/badge.svg)](https://github.com/Kelsidavis/Dreamland/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple)](https://ml-explore.github.io/mlx/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/tests-1350%2B%20passing-brightgreen)]()
[![Skills](https://img.shields.io/badge/skills-100%2B%20built--in-blue)]()

**It doesn't exist.**

> **Dreamland** — the classified build site for your local AI fleet

A local AI assistant powered by MLX. Private, fast, extensible, and local-first.
Dreamland is best on Apple Silicon, and also supports Linux.

## Quick Start

```bash
pip install -e ".[all]"
dreamland setup    # browser GUI — pick backend (MLX / Ollama / llama-server / Claude) + model
dreamland chat     # start chatting
```

Skipping setup? `dreamland init` writes a starter `~/.dreamland/config.toml` you can edit by hand.
Or run `dreamland doctor` any time to verify the environment.

Launch scripts:

```bash
./launch.sh       # Linux (also works on macOS)
./launch.command  # macOS double-click friendly
```

## What Can Dreamland Do?

### Chat

```bash
dreamland chat                  # interactive chat with streaming
dreamland ask "explain monads"  # one-shot query (pipeable)
cat code.py | dreamland ask "review this"
```

### Developer CLI

```bash
dreamland review                # AI code review of git changes
dreamland review --staged       # review only staged changes
dreamland review -f security    # focus on security issues
dreamland commit                # generate commit message + commit
dreamland commit -a             # stage all and commit
dreamland watch src/*.py        # live feedback on file changes
```

### Conversation Management

```bash
dreamland history               # list conversations
dreamland history --tag work    # filter by tag
dreamland log                   # activity timeline
dreamland log --today           # today's sessions
dreamland search "auth bug"     # search across all conversations
dreamland show <id>             # view a conversation
dreamland export <id> -f html   # export (markdown/text/json/html)
dreamland import backup.json    # import conversations
dreamland gc                    # clean up old conversations
```

### In-Chat Commands

40+ slash commands for power users:

| Command | Description |
|---------|-------------|
| `/undo` | Remove last exchange |
| `/retry` | Regenerate last response |
| `/fork [title]` | Branch the conversation |
| `/diff <id>` | Compare with another conversation |
| `/compact` | Compress old messages to free context |
| `/pin` / `/pins` | Pin important messages (survive context eviction) |
| `/grep <query>` | Search within conversation |
| `/save file.py` | Extract code blocks to files |
| `/copy` / `/copy code` | Copy response to clipboard |
| `/tag` / `/tags` | Organize with tags |
| `/stats` | Session statistics + cloud cost comparison |
| `/report` | Full session summary |
| `/history` / `/resume` | Switch conversations inline |
| `/alias` / `/snippet` | Custom shortcuts and text blocks |
| `/t review @file` | Apply prompt templates |

### 60 Built-in Skills

The agent has tools for:

| Skill | Tools |
|-------|-------|
| **filesystem** | read, write, list files |
| **shell** | execute commands |
| **git** | status, diff, log, commit, branch |
| **web** | fetch URLs |
| **search** | grep-like recursive search |
| **data** | JSON/CSV parsing, math |
| **memory** | persistent cross-session memory |
| **clipboard** | read/write system clipboard |
| **system** | CPU, memory, disk, processes |
| **time** | current time, timezones, duration calc |
| **network** | DNS lookup, port check, HTTP ping, whois |
| **hash** | MD5/SHA/base64/URL encoding |
| **env** | environment variables, PATH, which |
| **regex** | test, match, replace, split |
| **convert** | length, weight, volume, temperature, speed, data, time |
| **json_tools** | diff, flatten, schema generation, validate |
| **diff** | compare files and text, similarity stats |
| **archive** | create/list/extract zip and tar archives |
| **cron** | explain, preview, and build cron expressions |
| **markdown** | tables, TOC, checklists, JSON-to-markdown |
| **http** | full HTTP requests (GET/POST/PUT/DELETE, headers, JSON) |
| **sql** | query SQLite, inspect schema, explain plans |
| **image** | dimensions, format, file size (PNG/JPEG/GIF) |
| **process** | find, inspect, tree, listening ports |
| **text** | word count, stats, transforms, frequency |
| **knowledge** | personal knowledge base with tags |
| **translate** | language detection, translation prompts |
| **security** | scan for secrets, permissions, dependency audit |
| **todo** | task management with priorities and due dates |
| **scaffold** | project boilerplate (8 templates) |
| **math** | statistics, formatting, sequences (fibonacci, primes) |
| **docker** | containers, images, logs, stats |
| **calendar** | month display, business days, countdown |
| **qr** | ASCII QR code art generation |
| **jwt** | decode and inspect JSON Web Tokens |
| **color** | hex/RGB/HSL conversion, palettes, WCAG contrast |
| **uuid** | UUIDs, passwords, cryptographic tokens |
| **yaml** | parse, validate, YAML/JSON conversion |
| **codegen** | code snippet templates (10 patterns) |


### Persistent Memory

Dreamland remembers things across sessions in a local SQLite store with three
parallel retrieval tiers fused via Reciprocal Rank Fusion:

- **BM25** (SQLite FTS5) for keyword precision
- **Vector cosine** via `sentence-transformers` (optional: `pip install "dreamland[embeddings]"`) for paraphrase recall
- **Graph co-retrieval** — pairs of memories that show up together get linked, so a hit on one pulls its neighbors

Auto-capture extracts user / preference / project / deadline facts from
every user turn via conservative regex patterns (clause-bounded negation
so "I'm not a data scientist, I'm a designer" only captures designer).
Decay + auto-forget prune stale, never-recalled fact memories;
user / preference / project entries are protected.

```bash
dreamland memory stats              # counts, recall fraction, by-source/scope, pattern health
dreamland memory inspect <key>      # entry detail + salience + related + recent recalls
dreamland memory tidy --dry-run     # see what would be pruned
dreamland memory tidy --apply       # actually prune
dreamland memory consolidate        # find + merge near-duplicates
dreamland memory export --out backup.json
dreamland memory import backup.json
dreamland memory backup             # timestamped snapshot + rotation
dreamland memory diff baseline.json # what changed since baseline
dreamland memory reembed            # backfill vectors after installing [embeddings]
dreamland memory ingest --all       # backfill captures from every saved conversation
dreamland memory extract --stdin    # LLM-based extraction for what regex missed
dreamland memory recalls --last 24  # query trail: what was asked, what came back
dreamland memory activity           # ASCII sparkline of capture rate
dreamland memory tag KEY add work   # free-form labels for grouping
dreamland memory list --scope all   # cross-project audit
dreamland memory forget --tag X     # bulk forget by tag / source / scope
dreamland memory nudge KEY          # mark useful — bumps recall_count
dreamland memory promote KEY --to global   # move between scopes
```

**Per-query introspection.** Every `to_prompt_block(query=...)` run is
logged (capped at 5000 most-recent) so `memory inspect <key>` shows
the recent queries that returned it, with rank in result. Answers
"why does the agent remember X when I asked Y?" without grepping logs.

**Auto-LLM-extract** (opt-in, `config.auto_llm_extract: true`): when
regex captures 0 on a user turn, fires a background task that runs
the local LLM extractor against the same backend. Failures are
silent; the same backend serializes the work behind the live response,
so extraction runs when the model is idle.

**Per-project scope.** Memories carry an optional `scope` string —
empty = global (visible everywhere), non-empty = restricted to
callers passing the same scope. When `dreamland chat` / `dreamland serve` /
`dreamland mcp` is launched inside a project (one with `.dreamland.md`,
`.git`, `pyproject.toml`, etc.), they auto-derive a stable scope
from the project root path. New captures land there by default;
retrieval ORs current-scope with global so universal facts still
surface. Use `--scope all` on CLI commands to audit across every
project from one terminal.

**MCP server.** Run `dreamland mcp` to expose the store over stdio to any
MCP-compatible client (Claude Code, Cursor, OpenCode, Gemini CLI):

```jsonc
// .mcp.json
{"mcpServers": {"dreamland-memory": {"command": "dreamland", "args": ["mcp"]}}}
```

Seven tools become available to the client: `memory_search`, `memory_recall`,
`memory_list`, `memory_remember`, `memory_forget`, `memory_related`,
`memory_stats`.

### Web UI

```bash
dreamland serve    # starts gateway + web UI
# Open http://127.0.0.1:18743
```

- 4 themes (Deep Space, Frost, Matrix, Solarized)
- Command palette (Ctrl+P)
- Keyboard shortcuts (Ctrl+N/K/L/E/T)
- Conversation sidebar with search
- Real-time streaming

### API

```bash
# Simple ask endpoint
curl -X POST http://127.0.0.1:18743/api/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'

# OpenAI-compatible endpoint
curl http://127.0.0.1:18743/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'
```

### Multi-worker orchestration

Give the fleet a goal; it plans, executes, checks, and repairs:

```bash
# The fleet decomposes the goal itself, runs subtasks in parallel
# across workers, reviewer-checks every result, executes generated
# Python (run_check), audits the outcome against the goal, and runs
# one repair round if the audit finds gaps.
dreamland orchestrate \
  --goal "Create stats.py with mean/median functions and a demo main block" \
  --workspace /tmp/build --verify --repair --watch
```

- **Auto-planning** — omit `--task` and a planner-role worker produces
  the task DAG (roles, dependencies, output files), sized to the
  fleet's actual concurrency.
- **Parallel by default** — dependency-aware readiness scheduling:
  each task launches the moment its dependencies finish, throttled to
  the number of connected workers.
- **Follow-through** — syntax + substance validation on extracted
  files, optional reviewer verification per task (`--verify`),
  coordinator-side execution of generated Python (`run_check`), and
  feedback-carrying retries: a rejected attempt retries with the
  specific failure appended, not a blind re-roll.
- **Goal audit + repair** (`--repair`) — a majority-vote reviewer
  panel audits the finished run against the goal; on gaps, a targeted
  repair plan (grounded in the current file contents) executes and the
  goal is re-audited once.
- **Background runs** — `--watch` streams live progress; via the API,
  `POST /api/orchestrate {"background": true}` returns an id and
  `GET/DELETE /api/orchestrate/<id>` polls or cancels.
- **Pull the artifacts** — `GET /api/orchestrate/<id>/files` lists the
  project files a run produced; `…/files/<path>` serves raw contents and
  `…/archive` the whole workspace as a zip (traversal-guarded, scoped to
  the run's recorded workspace). `dreamland pull <id> [dest]` downloads and
  unpacks a build on any machine, and the fleet panel includes a file
  explorer / code viewer with a zip download over the same endpoints.
- **Push existing code back** — `POST /api/orchestrate` accepts
  `"files": {"path": "content"}` (CLI `--file mycode.py` or the web
  panel's "+ seed files" picker), seeding the workspace so the goal
  modifies real code instead of starting from scratch: pull → edit →
  push → pull the result.
- **Project git history** — managed workspaces are git repos; seeds and
  every finished run land as commits. `GET /api/orchestrate/<id>/git/log`
  and `…/git/diff/<sha>` serve the timeline, and the web explorer's
  "history" button renders it with a colored diff viewer — a personal
  GitHub-style view of what the fleet changed, run by run. Or just
  `git clone http://coordinator:18743/git/<id>` (read-only smart HTTP),
  then iterate with `dreamland orchestrate --project <id> --goal "…"` and
  `git pull` the fleet's new commit.

Hand-authored plans still work: `dreamland orchestrate plan.json` or
repeated `--task "role:prompt@deps+tools"` specs.

### Extensible

```bash
dreamland skill-init my_tool    # generate a skill skeleton
# Edit ~/.dreamland/skills/my_tool_skill.py
# Restart dreamland — skill auto-loaded
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                  Gateway                      │
│     WebSocket + HTTP + OpenAI-compat API     │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Sessions  │  │ Routing  │  │ 60 Skills │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└──────────┬───────────────────────┬───────────┘
           │                       │
    ┌──────┴──────┐         ┌──────┴──────┐
    │   Agent     │         │  Channels   │
    │  Runtime    │         │             │
    │  (MLX)      │         │  CLI        │
    │             │         │  WebChat    │
    │  Streaming  │         │  HTTP API   │
    │  Tool loop  │         │  ...        │
    └─────────────┘         └─────────────┘
```

## Configuration

```bash
dreamland config        # show current settings
dreamland config --json # machine-readable
dreamland doctor        # diagnose your setup
dreamland bench         # benchmark model speed
```

Config lives in `~/.dreamland/config.toml`. Three built-in agent profiles: **coder**, **researcher**, **writer**.

**Tuning knobs** (all optional, defaults in parentheses):

| Field | Default | What it does |
|---|---|---|
| `auto_capture` | `true` | Regex auto-capture on every user turn |
| `auto_llm_extract` | `false` | Background LLM extraction when regex misses (one inference call per quiet turn) |
| `memory_recall_log_cap` | `5000` | Max rows in the per-query recall log (oldest pruned) |
| `dispatch_history_size` | `500` | Dispatch decision ring buffer — hours of audit at typical traffic |
| `worker_inference_timeout` | `300.0` | Seconds the coordinator waits for the next chunk from a remote worker before tearing down the WS. Bump for cold-loaded large models |
| `mdns_advertise_ip` | `""` | Override the IP advertised via mDNS. Useful on Tailscale / WireGuard / multi-homed hosts where the auto-detected IP isn't the one workers can reach |

## Contributing

```bash
pip install -e ".[dev]"      # install dev deps (pytest, ruff, etc.)
make test                    # run the full suite (~30s, 1200+ tests)
make lint                    # ruff check src/ tests/
make fmt                     # ruff format
make help                    # see all targets
```

Tests have a 60-second per-test ceiling configured in `pyproject.toml` —
runaway loops surface as `Failed: Timeout` rather than hanging the
suite.

## License

MIT
