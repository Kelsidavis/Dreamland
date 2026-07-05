# Changelog

Human-friendly summary of notable changes. Full git history is the
source of truth; this file groups commits by theme so you can tell at
a glance whether a release affects you.

## Unreleased

### Orchestrator: goal-driven multi-worker builds (2026-07-03)

The orchestrator went from "execute a hand-written task list" to a
full goal → plan → execute → check → repair pipeline, verified live
on a two-worker MLX fleet:

- **Auto-planning**: `towel orchestrate --goal "…"` with no tasks —
  a planner-role worker emits the task DAG; malformed plans retry
  with the validation error fed back; planner quirks that feedback
  can't fix (schema echo, duplicate file writers, stale task indices
  in `depends_on`) are normalized instead of rejected.
- **Follow-through**: per-task reviewer verification (`verify`),
  coordinator-side execution of generated Python (`run_check`) with
  stderr fed into the retry prompt, and rejection-carrying retries.
- **Goal audit + one adaptive repair round** (`goal_check` /
  `repair`): a majority-vote reviewer panel judges the whole outcome
  against the goal; on gaps, a repair plan grounded in current file
  contents executes and is re-audited.
- **Fleet-aware parallel scheduling (now the default)**: readiness
  scheduling launches each task when its dependencies finish,
  throttled to the connected-worker count; the planner is told the
  fleet's concurrency so it plans independent branches.
- **Background runs**: `{"background": true}` → id + live
  `GET/DELETE /api/orchestrate/<id>`; CLI `--watch` streams progress.
- **Web UI**: the fleet panel gains an Orchestrate section — enter a
  goal, toggle verify/repair, watch per-task progress live, cancel
  mid-run. Backed by the background API.
- **Resume from any machine**: opening the fleet panel re-attaches to
  a running orchestration automatically (work started on another
  machine or before a reload resumes showing live progress instead of
  a blank panel); the recent-runs picker loads any run's full status;
  `towel orchestrate --attach <id>` does the same from the CLI.
- **Chat transcripts restore too**: fixed a localStorage guard that
  skipped transcript restore on every page reload after the first
  (users saw a welcome screen over their existing conversation), and
  a fresh browser now adopts the most recent server conversation
  instead of opening a blank session — chat carries across machines.
- **Persistent history**: every run (sync and background) is recorded
  to `~/.towel/orchestrations.json` (newest 100 kept); records survive
  coordinator restarts, in-flight runs get marked `interrupted`.
  `GET /api/orchestrations` and `towel orchestrations` list recent
  runs; `GET /api/orchestrate/<id>` serves finished runs from history.
- **Local planner + auditor**: when the coordinator's own model is ≥2×
  the best connected worker (or the fleet is empty), orchestration
  PLANNING and GOAL AUDITS run on the coordinator instead of an
  under-spec worker — judgment on the big model, volume on the fleet.
  Audits get a dedicated `auditor` role (per-task verifies stay on
  workers as `reviewer`), and the audit panel shrinks to one vote when
  the big model judges. Live: a two-file build that coin-flipped under
  7B planning completed first-try in 13s with a textbook plan, and a
  goal whose audit false-negatived on every 7B attempt got its first
  correct ACHIEVED verdict, 12s, zero repairs.
  `local_planner_enabled = false` opts out.
- **Managed workspaces**: omitting `workspace_dir` now provisions
  `~/.towel/workspaces/<id>` automatically — previously extract_to
  files were silently never written, so a bare goal produced a
  "completed" run with nothing pullable.
- **Capability-matched workspace directive**: chat-fast extract_to
  tasks are told their code block is saved for them; only tool-loop
  tasks are told to call write_file. The old one-size directive primed
  no-tools workers to emit filesystem scaffolding instead of the
  requested code (observed live twice).
- **Git-backed project history**: managed workspaces are git repos —
  seed files commit as "Seed files: <goal>", each finished run commits
  as "achieved/completed/partial: <goal> [towel:<id>]", so the project
  timeline is a real git log and each run's changes are a real diff.
  `GET /api/orchestrate/<id>/git/log` and `…/git/diff/<sha>` serve
  them; the web explorer gains a "history" button with a colored diff
  viewer. Caller-supplied workspace dirs are never touched unless the
  request opts in with `"git": true`; per-invocation git identity, so
  nothing mutates user git config; history is best-effort (a missing
  git binary never fails a run).
- **Seed files in the browser**: the Orchestrate panel gains a
  "+ seed files" picker — local code files are read client-side
  (FileReader), shown as removable chips, and sent in the JSON `files`
  map; caps mirror the server (32 files / 2MB), binary files rejected.
- **Seed files (the push half)**: `POST /api/orchestrate` accepts
  `"files": {path: content}` (CLI `--file PATH` / `--file NAME=PATH`),
  written into the workspace before planning; the planner sees the
  existing project contents and plans modifications of real code.
  Verified live: pull a build, push it back with an "add a function"
  goal, pull the correctly-modified result.
- **One-command project pulls**: `GET /api/orchestrate/<id>/archive`
  serves the whole workspace as a zip (same filters as the listing,
  200MB cap); the web explorer gains a "↓ zip" download button; new
  `towel pull <id> [dest]` downloads and unpacks a build on any
  machine (zip-slip guarded). `--json` CLI outputs now print plain
  (Rich's wrapping corrupted them for scripts).
- **File explorer / artifact pulling**: `GET /api/orchestrate/<id>/files`
  lists a run's project files, `…/files/<path>` serves raw contents —
  scoped to coordinator-recorded workspaces with traversal guards, so
  external systems can pull build artifacts. The fleet panel's
  Orchestrate section gains a file browser + code viewer over the same
  endpoints, with a recent-runs picker.
- **Collaboration grounding**: dependents receive a dependency's
  current on-disk file contents (not its chat blob) plus actual
  execution output.
- **Fixed**: MLX workers crashed with `KeyError: 'prompt'` on every
  chat-fast dispatch (coordinator sends `{system, messages}`; the MLX
  runtime now renders it through the chat template).

A heavy-development day focused on fleet coordination, model awareness,
and onboarding. ~52 commits.

### Fleet coordination — major

- **Central `Dispatcher`** (`gateway/dispatcher.py`) replaces the
  scattered worker-selection logic. Seven explicit layers (pin →
  affinity → task match → role match → general role → capability
  fallback → idle preempt) each emit a structured `DispatchDecision`
  with reason code, candidates considered, and observability flags
  (`affinity_missed`, `quality_degraded`, `preempted_idle`).
- **Fixed silent `AttributeError`** in the drain/disconnect path —
  `_select_worker()` was called but never existed; sessions were
  silently orphaned.
- **Quality gating**: dispatcher filters by per-task `min_vram_mb` /
  `min_context` from `TASK_REQUIREMENTS`. Falls back to under-spec
  workers with a `quality_degraded` flag rather than refusing —
  coordinator adapts to the fleet it has.
- **CPU pressure** folded into worker scoring (small penalty so
  capability-tied workers break in favour of the calmer one).
- **`/dispatch/recent`** and **`/dispatch/explain`** endpoints + a
  Recent-decisions section in the fleet panel.
- **Fast disconnect notify**: when a worker dies mid-job, the
  coordinator wakes any blocked waiter immediately instead of
  letting them spin on the per-call timeout.
- **Periodic stale-result sweeper** for idle-task cache (with per-task
  TTLs derived from cooldowns).

### Heterogeneous fleet awareness

- Workers self-report **`available_models`** (HF cache scan for MLX,
  `/api/tags` for Ollama, `/v1/models` for llama-server, the three SDK
  aliases for Claude), **`max_param_b_est`** (largest 4-bit quant the
  box can hold), **`disk_free_gb`** + **`disk_total_gb`** (rolled into
  live_resources).
- **`worker_quality_tier`** classifier (`high` / `medium` / `low`)
  derived from VRAM + context + backend.
- **`/fleet/inventory`** aggregates every cached model across the
  fleet — "where can I find model X?" — surfaced in the fleet panel
  with a searchable list.
- **`/fleet/suggest-targets`** ranks workers by has-cached / fits /
  disk-fits / quality-tier for a given model, with a download-size
  estimate. Both the per-worker `replace` and fleet-wide `roll` flows
  in the UI call it to show "✓ cached / ↓ will download / ✗ too small"
  before the destructive action.

### Remote lifecycle management

- **`towel launcher`** daemon — HTTP server you run on each candidate
  worker host. Coordinator can POST `/launch` (auth via
  `$TOWEL_TRIGGER_TOKEN`) to spawn a fresh `towel worker` process.
- **`/fleet/spawn`**, **`/fleet/replace-worker`**, **`/fleet/upgrade`**,
  **`/fleet/rolling-replace`** — coordinator-side orchestration on top
  of the launcher: spawn, drain+respawn, run `pip install --upgrade
  towel`, walk N workers serially with a configurable delay.
- **Worker shutdown WS message** so replace flows exit cleanly instead
  of relying on launchers to kill the process.
- **`towel worker --model <name>`** flag — overrides `config.model.name`
  at startup, the primary knob the coordinator uses to distribute
  different models to different workers.

### Native tools channel (all four backends)

- **MLX, Ollama, llama-server, Claude** runtimes now route tools
  through each backend's native API (`tools=` kwarg /
  `/api/chat` tools field / OpenAI-compat tools / Anthropic
  tool-use blocks) instead of stuffing 330 tool descriptions into the
  system prompt. Slim system prompts, structured tool-call parsing
  with text-fallback for older models.

### Setup + onboarding

- **`towel setup`** — browser GUI to pick backend + model. Reads
  available backends, lists locally-cached models, writes
  `~/.towel/config.toml`.
- **First-run hint** in `towel chat` when no config exists.
- **`launch.sh` / `launch.command`** now drop the user into setup if
  config is missing.
- Consistent messaging across README quickstart, `towel init`
  next-steps, `towel doctor` suggestions.

### Chat UX

- **`/skills` slash command** parallel to the CLI + HTTP endpoint.
- **`/dispatch/explain`** preview without side effects.
- **`/memory`** endpoint with type filter, substring search, limit,
  newest-first ordering. Plus `DELETE /memory/{key}`.
- **Streaming `<tool_call>` markup hidden** from the live token
  stream (Qwen3 native format leaked raw XML before).

### Bug fixes

- **`RAGIndex._split` infinite loop** when `chunk_overlap ≥
  chunk_size` — wedged every full-suite test run for hours until
  found.
- **Tool-error regex bugs**: five `^X:\b` patterns never fired because
  `\b` after `:` requires a word character (which is always a space).
  Tool failures like `File not found: …` were silently classified as
  successes.
- **Bracket markup in CLI output** (`[busy/enabled/ready]`) was being
  parsed as Rich style — escaped to render literally.

### Dev workflow

- **Makefile** with `test`, `lint`, `fmt`, `fix`, `doctor`, `clean`.
- **pytest-timeout** with 60s per-test ceiling so future infinite
  loops surface as clean failures.
- Lint baseline cleared across `src/towel/` and `tests/`.

### Daemon logging + error visibility (later in the day)

- Every long-running Towel process — `towel serve`, `towel worker`,
  `towel setup`, `towel launcher` — now configures
  `logging.basicConfig(level=INFO, …)` so operators see timestamped
  operational events in the daemon's terminal. Worker registrations,
  dispatch decisions, idle-result sweeps, upgrade requests, and
  shutdown notifications no longer vanish into the void.
- Logging setup deduplicated into `towel.logging_setup` with an
  idempotency test.
- `_load_model_with_friendly_error` helper: every user-facing CLI
  command (chat, ask, commit, summarize, etc.) now degrades to a red
  Rich Panel pointing at `towel setup` / `towel doctor` when model
  load fails, instead of dumping a stack trace.

### Stats

- 1214 tests passing in ~28 seconds.
- Zero lint complaints (`ruff check src/towel/ tests/`).
- ~400 new tests today, mostly covering the new endpoints, dispatcher
  paths, and the load-failure / logging helpers.

— Towel: Tool Oriented Worker Execution Link. Don't Panic.
