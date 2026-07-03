"""Agent orchestrator — spawn and coordinate specialist sub-agents.

Enables multi-agent workflows where a coordinator agent delegates
subtasks to specialists (coder, researcher, reviewer, writer) and
synthesizes their results.

Usage:
    orch = Orchestrator(config, skills)
    result = await orch.run("Build a REST API for user management", [
        AgentTask(role="architect", prompt="Design the API schema"),
        AgentTask(role="coder", prompt="Implement the endpoints"),
        AgentTask(role="reviewer", prompt="Review the code for issues"),
    ])
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from towel.agent.conversation import Conversation, Role
from towel.config import TowelConfig

log = logging.getLogger("towel.agent.orchestrator")


class WorkerDispatchError(RuntimeError):
    """Raised by `RoleDispatcher.dispatch_role_task` on failure.

    Carries the worker id (if a worker was picked) so the orchestrator
    can exclude it on the next retry attempt. Without this, retries
    against a flaky-but-pickable worker bounce back to the same worker
    every time — the dispatcher's session affinity is fresh per
    subtask but the task_type routing still steers to the same
    prefer_quality/prefer_fast pick.
    """

    def __init__(self, message: str, *, worker_id: str | None = None) -> None:
        super().__init__(message)
        self.worker_id = worker_id


class TaskRejectedError(ValueError):
    """Raised when a subtask's output failed validation or review.

    Distinct from `WorkerDispatchError` (infrastructure failure — the
    worker never produced usable output) because the remedy differs:
    an infra failure retries the same prompt on a different worker,
    while a rejection retries with the rejection reason appended so
    the model can correct course instead of re-rolling blind.
    Subclasses ValueError so callers that caught the extraction
    ValueErrors keep working.
    """


class RoleDispatcher(Protocol):
    """Protocol the Orchestrator uses to dispatch a single role task.

    Implemented by the gateway server so each orchestrator subtask can land
    on the best-fit remote worker (per `_route_by_role`) instead of running
    locally on the coordinator. Defined as a Protocol so the agent package
    stays free of a hard dependency on the gateway package — that
    direction would create a circular import (gateway → orchestrator →
    gateway).
    """

    async def dispatch_role_task(
        self,
        role: str,
        role_system: str,
        prompt: str,
        *,
        session_id: str,
        max_tokens: int,
        temperature: float,
        with_tools: bool,
        task_type: str | None,
        exclude_workers: set[str] | None,
    ) -> str:
        ...


@dataclass
class AgentTask:
    """A subtask to be executed by a specialist agent."""

    role: str  # e.g., "coder", "researcher", "reviewer", "writer"
    prompt: str
    depends_on: list[int] = field(default_factory=list)  # indices of tasks this depends on
    context: str = ""  # additional context injected from parent or dependencies
    result: str = ""
    status: str = "pending"  # pending, running, completed, failed
    elapsed: float = 0.0
    # When True the subtask runs through the worker's tool loop so it
    # can call write_file/read_file/edit_file. Defaults False so simple
    # text-only roles (writer, default) stay on the faster path.
    with_tools: bool = False
    # When set, the orchestrator extracts the first fenced code block
    # from the subtask's response and writes it to this workspace-
    # relative path. Lets a chat-fast (no-tools) coder produce code
    # without needing the slow tool loop — the worker emits a fenced
    # block, the coordinator writes the file. Lives in `AgentTask`
    # rather than on the subtask prompt so callers can use the same
    # response.
    extract_to: str | None = None
    extracted_path: str | None = None
    # How many times this subtask retried before completing or failing.
    # Surfaced so operators reading the response body can see when the
    # cluster needed multiple workers to satisfy a request.
    attempts: int = 0
    # When True, a reviewer-role worker checks the completed result
    # against the task prompt before the task is marked completed. A
    # FAIL verdict counts as a failed attempt and retries with the
    # reviewer's reason appended to the prompt — this is the
    # "follow-through" loop that keeps a plausible-but-wrong response
    # from silently passing as done.
    verify: bool = False
    # True = reviewer passed the result. None = verification not
    # requested, or the reviewer was unavailable / gave no parseable
    # verdict (result accepted, unverified). False = the task
    # terminally failed review — only seen alongside status="failed".
    verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "prompt": self.prompt[:100],
            "status": self.status,
            "elapsed": f"{self.elapsed:.1f}s",
            "result_length": len(self.result),
            "attempts": self.attempts,
            "verified": self.verified,
        }


@dataclass
class OrchestratorResult:
    """Result of a multi-agent orchestration run."""

    tasks: list[AgentTask]
    synthesis: str = ""
    total_elapsed: float = 0.0

    @property
    def success(self) -> bool:
        return all(t.status == "completed" for t in self.tasks)

    def summary(self) -> str:
        lines = [f"Orchestration: {len(self.tasks)} tasks, {self.total_elapsed:.1f}s total"]
        for i, t in enumerate(self.tasks):
            icon = {
                "completed": "+",
                "failed": "!",
                "skipped": "/",
                "running": "~",
                "pending": " ",
            }.get(t.status, "?")
            lines.append(
                f"  [{icon}] {i}. {t.role}: {t.status} ({t.elapsed:.1f}s, {len(t.result)} chars)"
            )
        return "\n".join(lines)


# Role-specific system prompts
ROLE_PROMPTS: dict[str, str] = {
    "coder": (
        "You are an expert software engineer. Write clean, production-quality code. "
        "Include error handling, types, and brief comments for complex logic. "
        "Output code in fenced blocks with the language specified."
    ),
    "researcher": (
        "You are a thorough research analyst. Find relevant information, cite sources, "
        "and present findings in a structured format. Be comprehensive but concise."
    ),
    "reviewer": (
        "You are a senior code reviewer. Analyze code for bugs, security issues, "
        "performance problems, and style. Be specific — cite line numbers. "
        "Rate overall quality 1-10."
    ),
    "writer": (
        "You are a technical writer. Write clear, well-structured documentation. "
        "Use headers, bullet points, and code examples where appropriate."
    ),
    "architect": (
        "You are a software architect. Design systems with clear separation of concerns, "
        "scalability, and maintainability. Provide schemas, data flow diagrams (as ASCII), "
        "and API specifications."
    ),
    "tester": (
        "You are a QA engineer. Write comprehensive tests covering edge cases, error "
        "conditions, and typical usage. Use the appropriate testing framework."
    ),
    "debugger": (
        "You are a debugging expert. Analyze errors systematically — identify root causes, "
        "explain why the bug occurs, and provide verified fixes."
    ),
    "default": ("You are a helpful AI assistant. Be concise and accurate."),
}


# Map orchestrator roles to TaskType strings the dispatcher recognises.
# Without this, the workspace preamble the orchestrator prepends to every
# subtask prompt prevents the keyword classifier from triggering — the
# prompt no longer starts with "write …" or "plan …", so it falls all
# the way through to None and dispatches via role_match. Role_match
# happens to pick the biggest INFERENCE worker (which is fine for
# coder) but skips the dispatcher's prefer_quality preempt path that
# would, e.g., pull SparklesMint off an idle task for an architect
# request. Explicit mapping closes the gap.
ROLE_TASK_TYPES: dict[str, str] = {
    "architect": "plan",
    "coder": "generate",
    "researcher": "research",
    "reviewer": "code_review",
    "writer": "draft",
    "tester": "test_gen",
    "debugger": "analyze",
}


class Orchestrator:
    """Coordinates multiple specialist agents on a complex task.

    With `dispatcher` set, each role's subtask is dispatched to the best-fit
    remote worker via the gateway's routing pipeline — so a "coder" subtask
    can land on the bigger worker while a "writer" subtask runs in parallel
    on a smaller one. Without `dispatcher`, falls back to a local
    AgentRuntime per subtask (useful for tests and single-node setups).
    """

    def __init__(
        self,
        config: TowelConfig,
        skills: Any = None,
        memory: Any = None,
        dispatcher: RoleDispatcher | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.config = config
        self.skills = skills
        self.memory = memory
        self.dispatcher = dispatcher
        # Single retry by default. Mirrors `/api/ask`'s primary→alt
        # fallback: if a worker emits empty text or times out, a second
        # attempt typically lands on the alternate worker (since the
        # first is now busy/draining) and succeeds. Setting this to 1
        # disables retries, which is occasionally useful for explicit
        # benchmarking of a particular worker.
        self.max_attempts = max(1, int(max_attempts))

    async def _execute_with_retry(
        self,
        task: AgentTask,
        full_prompt: str,
        *,
        workspace_dir: str | None = None,
    ) -> None:
        """Run a subtask, retrying once on failure.

        Updates `task` in place — populates result/status/elapsed/attempts.
        Captures the last error message as the result on terminal failure
        so the caller and downstream synthesis still see what went wrong.

        When a `WorkerDispatchError` carries a worker_id, that worker is
        excluded from the next attempt's dispatch — so the cluster's
        prefer_quality routing doesn't bounce back to the exact worker
        that just timed out. Other RuntimeErrors (no worker available,
        etc.) don't exclude anything since there's no worker to blame.

        When the task has `extract_to` set and `workspace_dir` is
        provided, the extraction + validation runs INSIDE the retry
        loop — a written file with a SyntaxError raises to trigger
        another attempt rather than leaving broken code on disk.
        Model-quality issues are often stochastic; re-rolling the
        same prompt frequently succeeds where the first try didn't.

        Rejections (`TaskRejectedError` — syntax/substance validation or a
        reviewer FAIL when ``task.verify`` is set) carry the reason
        forward: the next attempt's prompt gets the rejection appended
        so the model can fix the specific problem instead of
        re-rolling blind. Infra failures (`WorkerDispatchError`) do
        NOT add feedback — the model never saw the prompt fail, so
        there's nothing for it to correct.
        """
        task_start = time.perf_counter()
        task.status = "running"
        last_exc: Exception | None = None
        exclude_workers: set[str] = set()
        feedback: str | None = None
        for attempt in range(self.max_attempts):
            task.attempts = attempt + 1
            attempt_prompt = full_prompt
            if feedback:
                attempt_prompt = (
                    f"{full_prompt}\n\n"
                    "[Your previous attempt was rejected]\n"
                    f"{feedback}\n"
                    "Produce a corrected response. Follow the task "
                    "instructions exactly."
                )
            try:
                task.result = await self._run_agent(
                    task.role, attempt_prompt,
                    with_tools=task.with_tools,
                    exclude_workers=exclude_workers,
                )
                # Extract-and-validate runs in-loop so a write that
                # fails syntax check counts as an attempt and triggers
                # the next retry rather than a terminal "failed" task.
                if task.extract_to and workspace_dir:
                    self._extract_and_write(task, workspace_dir)
                # Follow-through check runs last, against the fully
                # validated result — no point reviewing output that
                # already failed syntax validation.
                if task.verify:
                    await self._verify_result(task)
                task.status = "completed"
                last_exc = None
                break
            except TaskRejectedError as e:
                last_exc = e
                feedback = str(e)
                log.warning(
                    "Task (%s, attempt %d/%d) rejected: %s",
                    task.role, attempt + 1, self.max_attempts, e,
                )
                await asyncio.sleep(0)
            except WorkerDispatchError as e:
                last_exc = e
                if e.worker_id:
                    exclude_workers.add(e.worker_id)
                log.warning(
                    "Task (%s, attempt %d/%d) failed on %s: %s",
                    task.role, attempt + 1, self.max_attempts,
                    e.worker_id or "no-worker", e,
                )
                await asyncio.sleep(0)
            except Exception as e:
                last_exc = e
                log.warning(
                    "Task (%s, attempt %d/%d) failed: %s",
                    task.role, attempt + 1, self.max_attempts, e,
                )
                # Brief yield between attempts so the dispatcher can
                # release the failed worker's slot — without this, a
                # tight retry on the same event-loop tick lands back on
                # the same worker that just failed. Same idea as the
                # `await asyncio.sleep(0)` after `_preempt_idle_task`.
                await asyncio.sleep(0)
        if last_exc is not None:
            task.result = f"Error: {last_exc}"
            task.status = "failed"
        task.elapsed = time.perf_counter() - task_start

    @staticmethod
    def _extract_and_write(task: AgentTask, workspace_dir: str) -> None:
        """Pull the first fenced code block out of `task.result` and
        write it to `workspace_dir / task.extract_to`.

        Handles three common shapes the model produces:
          ```python\n...```      — language tag we strip
          ```\n...```            — no language tag
          {code with no fences}  — falls through, writes the whole
                                   stripped result if no fence found

        On a successful write `task.extracted_path` is populated with
        the absolute path so callers downstream can read it back.
        """
        import re
        from pathlib import Path
        if task.extract_to is None:
            return
        text = task.result or ""
        # Match fenced blocks; tolerate language tags and trailing
        # whitespace. DOTALL so newlines in the body are kept.
        match = re.search(
            r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```",
            text,
            re.DOTALL,
        )
        body = match.group(1) if match else text.strip()
        if not body.endswith("\n"):
            body += "\n"
        # Reject path traversal — task.extract_to should land inside
        # the workspace.
        target = (Path(workspace_dir) / task.extract_to).resolve()
        ws_root = Path(workspace_dir).resolve()
        if ws_root not in target.parents and target != ws_root:
            raise ValueError(
                f"extract_to path {task.extract_to!r} resolves outside "
                f"workspace {workspace_dir}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        task.extracted_path = str(target)
        # Syntax-validate when the file looks like Python — catches the
        # common failure mode where the model emits almost-valid code
        # with a stray bracket or import line, which Codex would catch
        # via py_compile and we'd previously discover only at run time.
        # ast.parse is in-process and ~1ms; cheap enough to always run.
        # Validation failure raises so the orchestrator retries this
        # subtask on a different worker (model-quality issue is often
        # stochastic — a re-roll succeeds where the first didn't).
        if target.suffix == ".py":
            import ast
            try:
                tree = ast.parse(body)
            except SyntaxError as exc:
                raise TaskRejectedError(
                    f"extract_to wrote {target.name} but it has a "
                    f"SyntaxError on line {exc.lineno}: {exc.msg}"
                ) from exc
            # ast.parse accepts a bare identifier ("write_file") as
            # valid Python — it's a no-op expression statement. Live
            # observation: a coder subtask returned the literal text
            # `write_file` (the tool name) and that passed parsing,
            # producing an 11-byte file. Require at least one
            # substantive top-level construct so empty-or-degenerate
            # bodies trigger a retry. `import` covers stubs that just
            # re-export; `def`/`class`/`Assign`/`AnnAssign` cover the
            # real cases.
            has_substance = any(
                isinstance(
                    node,
                    (
                        ast.FunctionDef
                        | ast.AsyncFunctionDef
                        | ast.ClassDef
                        | ast.Assign
                        | ast.AnnAssign
                        | ast.Import
                        | ast.ImportFrom
                        | ast.If
                        | ast.For
                        | ast.While
                        | ast.Try
                        | ast.With
                    ),
                )
                for node in tree.body
            )
            if not has_substance:
                raise TaskRejectedError(
                    f"extract_to wrote {target.name} but it has no "
                    "substantive code (no def/class/assignment/import) — "
                    f"got {body[:80]!r}"
                )

    async def _verify_result(self, task: AgentTask) -> None:
        """Ask a reviewer-role worker whether the result followed the
        task instructions. Raises `TaskRejectedError` (with the reviewer's
        reason) on a FAIL verdict so the retry loop re-prompts with
        that feedback.

        Best-effort by design: if the reviewer can't run (no worker
        free) or returns no parseable verdict, the result is accepted
        with ``task.verified = None`` rather than failing the task —
        a flaky reviewer must not be able to kill otherwise-good work.
        Only an explicit FAIL blocks completion.
        """
        import re
        # Cap the excerpt so a long result doesn't blow the reviewer's
        # context; the instruction-adherence signal is almost always in
        # the first few thousand chars.
        excerpt = (task.result or "")[:6000]
        prompt = (
            "You are verifying whether a completed subtask followed its "
            "instructions.\n\n"
            f"Instructions given to the worker:\n{task.prompt}\n\n"
            f"Worker's output:\n---\n{excerpt}\n---\n\n"
            "Did the output follow the instructions and accomplish the "
            "task? Minor style differences are fine; missing "
            "requirements, ignored constraints, or off-task output are "
            "not. Reply with exactly one line: 'VERDICT: PASS' or "
            "'VERDICT: FAIL — <what is wrong and what to fix>'."
        )
        try:
            text = await self._run_agent("reviewer", prompt)
        except Exception as exc:
            log.warning(
                "verify(%s): reviewer unavailable, accepting unverified: %s",
                task.role, exc,
            )
            task.verified = None
            return
        match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
        if match is None:
            log.warning(
                "verify(%s): no parseable verdict in reviewer response, "
                "accepting unverified: %r",
                task.role, text[:120],
            )
            task.verified = None
            return
        if match.group(1).upper() == "PASS":
            task.verified = True
            return
        reason = text[match.end():].strip(" -—:\n") or "no reason given"
        task.verified = False
        raise TaskRejectedError(f"reviewer rejected the output: {reason[:500]}")

    @staticmethod
    def _workspace_preamble(workspace_dir: str | None) -> str:
        """Prefix subtask prompts with a workspace-directive when set.

        Subtasks share state via files in this directory: a coder writes
        ``game.py`` there, and a downstream tester reads it back. Tool
        execution happens on the coordinator, so a single absolute path
        works for every subtask regardless of which worker runs it.
        """
        if not workspace_dir:
            return ""
        return (
            f"Shared workspace: {workspace_dir}\n"
            "Use the filesystem tools (write_file, read_file, edit_file, "
            "list_directory) against this directory so other subtasks "
            "in this orchestration can see your work. Prefer relative "
            "paths under the workspace; absolute paths outside it should "
            "be avoided unless the goal explicitly requires it.\n\n"
        )

    @staticmethod
    def _failed_deps(task: AgentTask, tasks: list[AgentTask]) -> list[int]:
        return [
            d for d in task.depends_on
            if 0 <= d < len(tasks) and tasks[d].status in ("failed", "skipped")
        ]

    @staticmethod
    def _mark_skipped(task: AgentTask, failed_deps: list[int]) -> None:
        """Short-circuit when a direct dependency didn't succeed.

        Without this, the dependent runs with the failed dep's error
        string injected as `Result from <role>` context — the worker
        either reasons against a misleading "result" or wastes time
        refusing the prompt. Marking the task `skipped` makes the
        failure cascade visible in the response and saves the worker
        turn.
        """
        task.status = "skipped"
        task.result = (
            f"Skipped: depends on task(s) {failed_deps} which did "
            "not complete successfully."
        )
        task.elapsed = 0.0
        task.attempts = 0

    @staticmethod
    def _compose_prompt(
        goal: str,
        task: AgentTask,
        tasks: list[AgentTask],
        workspace_preamble: str,
    ) -> str:
        """Build a subtask's full prompt: workspace directive, results
        from its dependencies, injected context, then goal + task."""
        dep_context = ""
        if task.depends_on:
            dep_results = []
            for dep_idx in task.depends_on:
                if dep_idx < len(tasks) and tasks[dep_idx].result:
                    dep_results.append(
                        f"[Result from {tasks[dep_idx].role} (task {dep_idx})]:\n"
                        f"{tasks[dep_idx].result}"
                    )
            if dep_results:
                dep_context = "\n\n".join(dep_results) + "\n\n"

        full_prompt = workspace_preamble
        if dep_context:
            full_prompt += f"Context from previous tasks:\n{dep_context}\n"
        if task.context:
            full_prompt += f"{task.context}\n\n"
        full_prompt += f"Goal: {goal}\n\nYour task: {task.prompt}"
        return full_prompt

    def _synthesize(self, goal: str, result: OrchestratorResult) -> None:
        if result.success and len(result.tasks) > 1:
            synthesis_parts = [f"# Results for: {goal}\n"]
            for i, t in enumerate(result.tasks):
                synthesis_parts.append(f"## {t.role.title()} (Task {i})\n{t.result}\n")
            result.synthesis = "\n".join(synthesis_parts)

    async def plan(
        self,
        goal: str,
        *,
        max_tasks: int = 8,
        verify: bool = False,
    ) -> list[AgentTask]:
        """Decompose a goal into an executable task list — no
        hand-authored plan required.

        Dispatches an architect-role subtask that returns a JSON plan,
        then validates it with the same rules the API enforces. A
        malformed plan retries with the validation error appended, so
        the planner model gets to correct its own output instead of
        the orchestration dying on the first bad comma.

        The plan guidance bakes in the fleet's known-good recipe:
        chat-fast subtasks with `extract_to` for code files (one fenced
        block, no prose) rather than the tool loop.

        Raises ValueError when no valid plan emerges after
        `self.max_attempts` tries.
        """
        roles = ", ".join(sorted(r for r in ROLE_PROMPTS if r != "default"))
        base_prompt = (
            f"Decompose the following goal into 1-{max_tasks} subtasks "
            "for specialist workers.\n\n"
            f"Goal: {goal}\n\n"
            f"Available roles: {roles}.\n\n"
            "Respond with ONLY a JSON array — no prose, no markdown "
            "outside the JSON. Each element:\n"
            "{\n"
            '  "role": "<role>",\n'
            '  "prompt": "<complete, self-contained instructions>",\n'
            '  "depends_on": [<indices of earlier tasks whose output '
            "this task needs>],\n"
            '  "extract_to": "<relative file path — ONLY for tasks '
            'that must produce a code file>"\n'
            "}\n\n"
            "Rules:\n"
            "- Each prompt must be self-contained: the worker sees only "
            "its prompt plus the outputs of its depends_on tasks — "
            "nothing else.\n"
            "- For a task that produces a code file, set extract_to and "
            "instruct the worker to answer with ONE fenced code block "
            "and no prose.\n"
            "- depends_on may only reference earlier tasks (smaller "
            "index). Omit it or use [] for independent tasks.\n"
            "- Prefer a few substantial tasks over many small ones."
        )
        feedback: str | None = None
        last_exc: Exception | None = None
        for _attempt in range(self.max_attempts):
            prompt = base_prompt
            if feedback:
                prompt = (
                    f"{base_prompt}\n\n"
                    "[Your previous plan was rejected]\n"
                    f"{feedback}\n"
                    "Return a corrected JSON array."
                )
            text = await self._run_agent("architect", prompt)
            try:
                tasks = self._parse_plan(text)
            except ValueError as exc:
                feedback = str(exc)
                last_exc = exc
                log.warning("plan(%r): invalid plan, retrying: %s", goal[:60], exc)
                continue
            if verify:
                for t in tasks:
                    t.verify = True
            log.info(
                "plan(%r): %d tasks (%s)",
                goal[:60], len(tasks), [t.role for t in tasks],
            )
            return tasks
        raise ValueError(
            f"planner produced no valid plan after {self.max_attempts} "
            f"attempts: {last_exc}"
        )

    @staticmethod
    def _parse_plan(text: str) -> list[AgentTask]:
        """Parse and validate a planner response into AgentTasks.

        Tolerates the JSON arriving inside a fenced block or surrounded
        by prose — grabs the outermost array. Raises ValueError with a
        model-actionable message on any structural problem; `plan()`
        feeds that message back for the retry.
        """
        import json
        import re

        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        candidate = match.group(1) if match else text
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end <= start:
            raise ValueError("response contains no JSON array")
        try:
            raw = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(raw, list) or not raw:
            raise ValueError("plan must be a non-empty JSON array")
        if len(raw) > 32:
            raise ValueError("plan must have 32 tasks or fewer")

        tasks: list[AgentTask] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"tasks[{i}] must be a JSON object")
            role = entry.get("role")
            if role not in ROLE_PROMPTS:
                raise ValueError(
                    f"tasks[{i}].role={role!r} is unknown; valid roles: "
                    f"{sorted(ROLE_PROMPTS)}"
                )
            prompt = entry.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"tasks[{i}].prompt must be a non-empty string")
            deps_raw = entry.get("depends_on") or []
            if not isinstance(deps_raw, list):
                raise ValueError(f"tasks[{i}].depends_on must be a list")
            deps: list[int] = []
            for d in deps_raw:
                if not isinstance(d, int) or isinstance(d, bool) or not 0 <= d < i:
                    raise ValueError(
                        f"tasks[{i}].depends_on contains {d!r} — entries "
                        f"must be integer indices of EARLIER tasks (0..{i - 1})"
                    )
                deps.append(d)
            extract_to = entry.get("extract_to")
            if extract_to is not None:
                if not isinstance(extract_to, str) or not extract_to.strip():
                    raise ValueError(
                        f"tasks[{i}].extract_to must be a non-empty string"
                    )
                if ".." in extract_to.split("/"):
                    raise ValueError(
                        f"tasks[{i}].extract_to must not contain '..'"
                    )
                extract_to = extract_to.strip()
            with_tools = entry.get("with_tools", False)
            if not isinstance(with_tools, bool):
                raise ValueError(f"tasks[{i}].with_tools must be a boolean")
            tasks.append(AgentTask(
                role=role,
                prompt=prompt.strip(),
                depends_on=deps,
                with_tools=with_tools,
                extract_to=extract_to,
            ))
        return tasks

    async def run(
        self,
        goal: str,
        tasks: list[AgentTask],
        *,
        workspace_dir: str | None = None,
    ) -> OrchestratorResult:
        """Execute a sequence of agent tasks, respecting dependencies."""
        start = time.perf_counter()
        result = OrchestratorResult(tasks=tasks)

        log.info(f"Orchestrating {len(tasks)} tasks for: {goal[:80]}")

        workspace_preamble = self._workspace_preamble(workspace_dir)

        for i, task in enumerate(tasks):
            failed_deps = self._failed_deps(task, tasks)
            if failed_deps:
                self._mark_skipped(task, failed_deps)
                log.info(
                    "Task %d (%s): skipped (failed deps %s)",
                    i, task.role, failed_deps,
                )
                continue

            full_prompt = self._compose_prompt(
                goal, task, tasks, workspace_preamble,
            )

            # Execute (extract-and-validate happens INSIDE the retry
            # loop when extract_to is set, so a syntax-error in the
            # written file triggers another attempt instead of leaving
            # broken code on disk).
            await self._execute_with_retry(
                task, full_prompt, workspace_dir=workspace_dir,
            )
            log.info(
                "Task %d (%s): %s in %.1fs (attempts=%d)",
                i, task.role, task.status, task.elapsed, task.attempts,
            )

        result.total_elapsed = time.perf_counter() - start
        self._synthesize(goal, result)
        return result

    async def run_parallel(
        self,
        goal: str,
        tasks: list[AgentTask],
        *,
        workspace_dir: str | None = None,
    ) -> OrchestratorResult:
        """Execute tasks in dependency-aware parallel waves.

        Each wave runs every still-pending task whose dependencies have
        all completed — independent tasks fan out across the fleet
        simultaneously while dependents wait for their inputs and get
        the same dependency-context injection the sequential path does.
        Previously this method ignored `depends_on` entirely, so
        `parallel=true` silently broke collaboration: dependents raced
        their dependencies and saw none of their output.

        Tasks whose dependencies failed are skipped (same cascade as
        `run`). Tasks left pending when no wave can make progress —
        only possible with a dependency cycle — are skipped too, with
        the cycle called out in the result.
        """
        start = time.perf_counter()
        result = OrchestratorResult(tasks=tasks)
        workspace_preamble = self._workspace_preamble(workspace_dir)

        log.info(
            f"Orchestrating {len(tasks)} tasks (parallel waves) for: {goal[:80]}"
        )

        async def _exec(i: int, task: AgentTask) -> None:
            full_prompt = self._compose_prompt(
                goal, task, tasks, workspace_preamble,
            )
            await self._execute_with_retry(
                task, full_prompt, workspace_dir=workspace_dir,
            )
            log.info(
                "Task %d (%s): %s in %.1fs (attempts=%d)",
                i, task.role, task.status, task.elapsed, task.attempts,
            )

        wave = 0
        while True:
            # Cascade skips first so a wave never launches a task whose
            # dependency just failed in the previous wave.
            for i, task in enumerate(tasks):
                if task.status != "pending":
                    continue
                failed_deps = self._failed_deps(task, tasks)
                if failed_deps:
                    self._mark_skipped(task, failed_deps)
                    log.info(
                        "Task %d (%s): skipped (failed deps %s)",
                        i, task.role, failed_deps,
                    )

            ready = [
                i for i, task in enumerate(tasks)
                if task.status == "pending"
                and all(
                    tasks[d].status == "completed"
                    for d in task.depends_on
                    if 0 <= d < len(tasks)
                )
            ]
            if not ready:
                break
            wave += 1
            log.info("Parallel wave %d: tasks %s", wave, ready)
            await asyncio.gather(*[_exec(i, tasks[i]) for i in ready])

        # Anything still pending here has an unresolvable dependency
        # graph (a cycle) — surface that instead of looping forever.
        for task in tasks:
            if task.status == "pending":
                task.status = "skipped"
                task.result = (
                    "Skipped: unresolvable dependencies (dependency cycle "
                    f"involving depends_on={task.depends_on})."
                )

        result.total_elapsed = time.perf_counter() - start
        self._synthesize(goal, result)
        return result

    async def _run_agent(
        self, role: str, prompt: str, *,
        with_tools: bool = False,
        exclude_workers: set[str] | None = None,
    ) -> str:
        """Run a single agent step with role-specific system prompt.

        Uses the configured remote dispatcher when present so each role's
        subtask can land on the best-fit worker; otherwise falls back to
        a local AgentRuntime. ``with_tools`` flips the dispatcher onto
        the tool-loop path so the subtask can call write_file etc.
        """
        role_system = ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])

        if self.dispatcher is not None:
            # Per-subtask session keeps role contexts isolated — a coder
            # subtask shouldn't reuse the writer's affinity-pinned worker.
            import uuid
            session_id = f"orch-{role}-{uuid.uuid4().hex[:8]}"
            return await self.dispatcher.dispatch_role_task(
                role,
                role_system,
                prompt,
                session_id=session_id,
                max_tokens=2048,
                temperature=0.4,
                with_tools=with_tools,
                task_type=ROLE_TASK_TYPES.get(role),
                exclude_workers=exclude_workers,
            )

        # Local fallback: spin up a coordinator-side AgentRuntime with
        # the role's identity. Used by tests and single-node deployments.
        import copy

        from towel.agent.runtime import AgentRuntime

        agent_config = copy.deepcopy(self.config)
        agent_config.identity = role_system

        runtime = AgentRuntime(agent_config, skills=self.skills, memory=self.memory)
        conv = Conversation(channel=f"orchestrator:{role}")
        conv.add(Role.USER, prompt)

        response = await runtime.step(conv)
        return response.content
