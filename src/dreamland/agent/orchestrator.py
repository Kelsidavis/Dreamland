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

from dreamland.agent.conversation import Conversation, Role
from dreamland.config import DreamlandConfig

log = logging.getLogger("dreamland.agent.orchestrator")


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

    ``retryable=False`` marks rejections that regenerating THIS task
    cannot fix — e.g. an import-name error rooted in a sibling task's
    file. The retry loop fails the task immediately instead of burning
    attempts; the goal-audit/repair round owns cross-file fixes.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


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
    # When True (requires extract_to on a Python file), the coordinator
    # executes the extracted file after validation; a non-zero exit or
    # timeout rejects the attempt with the error fed back into the
    # retry. Chat-fast workers cannot run code — without this, "test
    # that it works" subtasks can only hallucinate a result. Opt-in
    # because it executes model-generated code on the coordinator.
    run_check: bool = False
    # Captured stdout (trimmed) of a successful run_check. Injected
    # into dependents' context so downstream subtasks reason about the
    # program's ACTUAL output rather than the coder's claims.
    run_output: str | None = None
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
    # Goal-level audit (run_goal with goal_check=True). Per-task verify
    # checks each result against ITS OWN prompt; this checks the whole
    # outcome against the GOAL — a plan can complete every task and
    # still miss the goal. True = auditor says achieved, False = gaps
    # found (listed in goal_feedback), None = audit not requested or
    # the auditor was unavailable.
    goal_achieved: bool | None = None
    goal_feedback: str = ""
    # Number of tasks appended by the adaptive repair round (repair=True
    # and the first audit found gaps).
    repair_tasks_added: int = 0

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
    "planner": (
        "You are a planning engine that decomposes goals into machine-readable "
        "task plans. You respond with STRICT JSON only — no prose, no markdown "
        "fences, no explanations, no checklists. Your entire response must be "
        "parseable by json.loads."
    ),
    "auditor": (
        "You are a rigorous delivery auditor. You judge whether completed "
        "work achieves its stated goal, strictly from the evidence "
        "presented — execution output is ground truth, claims are not. "
        "You never speculate about content you cannot see, and you answer "
        "in exactly the verdict format requested."
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
    "planner": "plan",
    # Goal audits are judgment, like planning — the "plan" task type
    # routes them to the highest-quality worker AND makes them eligible
    # for the coordinator-local big-model path. Per-task verifies keep
    # the "reviewer" role (code_review) so the volume stays on the
    # fleet.
    "auditor": "plan",
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
        config: DreamlandConfig,
        skills: Any = None,
        memory: Any = None,
        dispatcher: RoleDispatcher | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.config = config
        self.skills = skills
        self.memory = memory
        self.dispatcher = dispatcher
        # Wall-clock ceiling for a run_check execution. 30s covers any
        # sane generated script; an infinite loop or a blocking
        # input() call gets killed and rejected instead of wedging the
        # orchestration. Instance attribute so tests can shrink it.
        self.run_check_timeout = 30.0
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
        all_tasks: list[AgentTask] | None = None,
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
                    # Execution check before the (more expensive)
                    # reviewer pass — no point reviewing code that
                    # doesn't run.
                    if task.run_check:
                        await self._run_extracted_file(
                            task, workspace_dir, all_tasks,
                        )
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
                if not e.retryable:
                    # Regenerating this task can't fix the problem
                    # (e.g. the bug lives in a sibling's file) — fail
                    # now and let the repair round handle it instead
                    # of burning the remaining attempts.
                    break
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
                if e.worker_id is None:
                    # No worker was even picked — the fleet is
                    # saturated or briefly empty (mid-reconnect). An
                    # immediate retry hits the same wall; a short real
                    # wait lets a worker finish its job or re-register.
                    await asyncio.sleep(2.0)
                else:
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
            def _substantive(node: ast.stmt) -> bool:
                if isinstance(
                    node,
                    (
                        ast.FunctionDef
                        | ast.AsyncFunctionDef
                        | ast.ClassDef
                        | ast.Assign
                        | ast.AnnAssign
                        | ast.AugAssign
                        | ast.Import
                        | ast.ImportFrom
                        | ast.If
                        | ast.For
                        | ast.While
                        | ast.Try
                        | ast.With
                        | ast.Raise
                        | ast.Assert
                    ),
                ):
                    return True
                # A bare expression statement counts only when it's a
                # call — `print("hi")` is a real script, the literal
                # text `write_file` (a stray tool name, seen live) is
                # not.
                return isinstance(node, ast.Expr) and isinstance(
                    node.value, ast.Call
                )

            has_substance = any(_substantive(node) for node in tree.body)
            if not has_substance:
                raise TaskRejectedError(
                    f"extract_to wrote {target.name} but it has no "
                    "substantive code (no def/class/assignment/import) — "
                    f"got {body[:80]!r}"
                )

    async def _exec_file_once(
        self, path: str, workspace_dir: str,
    ) -> tuple[int | None, str, str]:
        """Run one Python file to completion. Returns (returncode,
        stdout, stderr); returncode None means timeout."""
        import sys
        proc = await asyncio.create_subprocess_exec(
            sys.executable, path,
            cwd=workspace_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.run_check_timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return None, "", ""
        return (
            proc.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def _run_extracted_file(
        self,
        task: AgentTask,
        workspace_dir: str,
        all_tasks: list[AgentTask] | None = None,
    ) -> None:
        """Execute the file `_extract_and_write` just wrote and reject
        the attempt if it doesn't run cleanly.

        This is the strongest follow-through check available: chat-fast
        workers cannot execute anything, so without it a "make sure it
        runs" instruction can only ever be hallucinated. The stderr
        tail goes into the rejection so the retry prompt tells the
        model exactly what crashed.

        Runs with cwd=workspace_dir so sibling files written by earlier
        subtasks import naturally. Timeout kills the process (infinite
        loop / blocking input()) and counts as a rejection.

        Sibling-import race: under parallel scheduling an entry point
        can be written (and executed) before the library it imports
        exists — planners routinely mark such tasks independent
        despite the guidance. A ModuleNotFoundError whose module is
        another task's extract_to waits for that producer to finish
        and re-runs, instead of burning every retry regenerating a
        file that was never the problem (observed live: main.py failed
        3/3 attempts while calc.py was still generating).
        """
        import re
        path = task.extracted_path
        if not path:
            return
        name = task.extract_to or path
        rc, stdout, stderr = await self._exec_file_once(path, workspace_dir)
        if rc == 0:
            task.run_output = stdout[:2000]
            return
        if rc is None:
            raise TaskRejectedError(
                f"run_check: {name} did not finish within "
                f"{self.run_check_timeout:.0f}s — likely an infinite "
                "loop or a blocking input() call. The file must run to "
                "completion non-interactively."
            )
        missing = re.search(
            r"ModuleNotFoundError: No module named '([\w.]+)'", stderr,
        )
        if missing and all_tasks:
            mod_file = missing.group(1).split(".")[0] + ".py"
            producer = next(
                (t for t in all_tasks
                 if t is not task and t.extract_to == mod_file),
                None,
            )
            if producer is not None:
                log.info(
                    "run_check: %s imports %s produced by a sibling "
                    "task (%s) — waiting for it",
                    name, mod_file, producer.status,
                )
                # Poll the sibling's status; its extract happens inside
                # its completion, so completed ⇒ the file exists.
                for _ in range(240):
                    if producer.status not in ("pending", "running"):
                        break
                    await asyncio.sleep(0.5)
                if producer.status == "completed":
                    rc, stdout, stderr = await self._exec_file_once(
                        path, workspace_dir,
                    )
                    if rc == 0:
                        task.run_output = stdout[:2000]
                        return
        tail = stderr.strip()[-800:]
        # An import-NAME error rooted in a completed sibling's file is
        # unfixable from here: this file's import matches the goal, the
        # sibling's contents don't. Fail fast (non-retryable) so the
        # repair round fixes the right file — observed live: main.py
        # burned 3 regenerations over calc.py's missing subtract().
        name_err = re.search(
            r"ImportError: cannot import name '[\w.]+' from '([\w.]+)'",
            stderr,
        )
        if name_err and all_tasks:
            sib_file = name_err.group(1).split(".")[0] + ".py"
            producer = next(
                (t for t in all_tasks
                 if t is not task and t.extract_to == sib_file
                 and t.status == "completed"),
                None,
            )
            if producer is not None:
                raise TaskRejectedError(
                    f"run_check: {name} exited with code {rc}:\n{tail}\n"
                    f"The missing name lives in {sib_file}, produced by "
                    "another task — regenerating this file cannot fix "
                    f"it; {sib_file} must be corrected.",
                    retryable=False,
                )
        raise TaskRejectedError(
            f"run_check: {name} exited with code {rc}:\n{tail}"
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
            "not. Output that includes MORE than asked (extra "
            "functions, extra explanation) still PASSES as long as "
            "everything required is present and correct — judge for "
            "missing work, not for surplus. Reply with exactly one "
            "line: 'VERDICT: PASS' or 'VERDICT: FAIL — <what is wrong "
            "and what to fix>'."
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
    def _workspace_preamble(
        workspace_dir: str | None, task: AgentTask | None = None,
    ) -> str:
        """Per-task workspace directive.

        The directive must match what the task can actually do:

        - ``with_tools`` tasks get the filesystem-tools instruction
          (they really can call write_file etc. on the coordinator).
        - ``extract_to`` tasks (chat-fast, no tools) are told their
          code block is saved FOR them. The old one-size directive
          told these tasks to "use write_file" — which they can't —
          and primed exactly that garbage: live runs produced files
          containing os.makedirs/write_file scaffolding instead of the
          requested code.
        - tasks with neither get no workspace text at all.
        """
        if not workspace_dir or task is None:
            return ""
        if task.with_tools:
            return (
                f"Shared workspace: {workspace_dir}\n"
                "Use the filesystem tools (write_file, read_file, "
                "edit_file, list_directory) against this directory so "
                "other subtasks in this orchestration can see your "
                "work. Prefer relative paths under the workspace; "
                "absolute paths outside it should be avoided unless "
                "the goal explicitly requires it.\n\n"
            )
        if task.extract_to:
            return (
                f"Your output will be saved automatically as "
                f"{task.extract_to} in the build's shared workspace — "
                "you have no filesystem access and must not write "
                "path-handling or file-writing code. Respond with the "
                "file's contents in ONE fenced code block.\n\n"
            )
        return ""

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
        workspace_dir: str | None,
    ) -> str:
        """Build a subtask's full prompt: workspace directive (shaped
        by the task's capabilities), results from its dependencies,
        injected context, then goal + task."""
        workspace_preamble = Orchestrator._workspace_preamble(
            workspace_dir, task,
        )
        dep_context = ""
        if task.depends_on:
            dep_results = []
            for dep_idx in task.depends_on:
                if not (0 <= dep_idx < len(tasks)):
                    continue
                dep = tasks[dep_idx]
                file_body: str | None = None
                if dep.extracted_path:
                    # Disk truth beats the chat blob: the dependency's
                    # canonical output is the file it wrote, not its
                    # full response (prose + possibly stale code —
                    # observed live: a dependent reproduced an earlier
                    # draft from the result text instead of the
                    # validated file). Read fresh each compose so a
                    # repair-rewritten file propagates.
                    from pathlib import Path
                    p = Path(dep.extracted_path)
                    if p.is_file():
                        file_body = p.read_text(
                            encoding="utf-8", errors="replace",
                        )[:4000]
                if file_body is not None:
                    dep_results.append(
                        f"[File {dep.extract_to} produced by "
                        f"{dep.role} (task {dep_idx}) — current "
                        f"contents]:\n```\n{file_body}\n```"
                    )
                elif dep.result:
                    dep_results.append(
                        f"[Result from {dep.role} (task {dep_idx})]:\n"
                        f"{dep.result}"
                    )
                # Ground-truth beats claims: when the dependency's
                # file was actually executed (run_check), give the
                # dependent the real program output too.
                if dep.run_output is not None:
                    dep_results.append(
                        f"[Actual execution output of "
                        f"{dep.extract_to} (task {dep_idx})]:\n"
                        f"{dep.run_output}"
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
        workspace_dir: str | None = None,
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
        # "planner" is the internal role plan() itself dispatches as —
        # its strict-JSON system prompt would make any subtask assigned
        # to it emit JSON instead of real output (live observation: a
        # planner-role subtask produced a write_file JSON blob where a
        # Python file was expected). "default" adds nothing over the
        # specific roles. Keep both out of the menu.
        # Seeded / pre-existing project files ground the plan: a goal
        # like "add logging to app.py" must plan MODIFICATIONS of what
        # exists, not invent files from scratch.
        files_block = self._workspace_files_block(workspace_dir)
        base_prompt = (
            f"Decompose the following goal into 1-{max_tasks} subtasks "
            "for specialist workers.\n\n"
            f"Goal: {goal}\n\n"
            + (
                "The project already contains these files. A task that "
                "modifies one must set extract_to to that EXACT path "
                "and produce the complete updated file.\n"
                f"{files_block}\n\n"
                if files_block else ""
            )
            + f"{self._plan_schema_guide()}"
        )
        return await self._plan_loop(base_prompt, label=goal[:60], verify=verify)

    @staticmethod
    def _workspace_files_block(
        workspace_dir: str | None,
        *,
        limit_files: int = 8,
        limit_chars: int = 2000,
    ) -> str:
        """Trimmed contents of a workspace's existing files, for
        grounding the initial plan when the caller seeded the workspace
        (or re-runs against a previous build)."""
        if not workspace_dir:
            return ""
        from pathlib import Path
        root = Path(workspace_dir)
        if not root.is_dir():
            return ""
        blocks: list[str] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(
                part.startswith(".") or part == "__pycache__"
                for part in rel.parts
            ):
                continue
            if len(blocks) >= limit_files:
                blocks.append("… (more files not shown)")
                break
            body = p.read_text(encoding="utf-8", errors="replace")[:limit_chars]
            blocks.append(f"--- {rel} ---\n{body}")
        return "\n".join(blocks)

    def _plan_schema_guide(self) -> str:
        """The JSON schema + rules block shared by every planning
        prompt (initial decomposition and repair planning)."""
        # "planner" is the internal role plan() itself dispatches as —
        # its strict-JSON system prompt would make any subtask assigned
        # to it emit JSON instead of real output (live observation: a
        # planner-role subtask produced a write_file JSON blob where a
        # Python file was expected). "default" adds nothing over the
        # specific roles. Keep both out of the menu.
        roles = ", ".join(
            sorted(
                r for r in ROLE_PROMPTS
                if r not in ("default", "planner", "auditor")
            )
        )
        return (
            f"Available roles: {roles}.\n\n"
            "Respond with ONLY a JSON array — no prose, no markdown "
            "outside the JSON. Each element:\n"
            "{\n"
            '  "role": "<role>",\n'
            '  "prompt": "<complete, self-contained instructions>",\n'
            '  "depends_on": [<indices of earlier tasks whose output '
            "this task needs>],\n"
            '  "extract_to": "<relative file path — ONLY for tasks '
            'that must produce a code file>",\n'
            '  "run_check": true  // ONLY with extract_to on a Python '
            "file that should run to completion — the coordinator "
            "executes it and feeds errors back\n"
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
            "- Do NOT include a meta task like 'plan the work' or "
            "'decide the structure' — this plan IS the planning. Every "
            "task must produce concrete output.\n"
            "- Workers cannot execute code. Do NOT add a 'run it' or "
            "'test that it works' task — set run_check on the task "
            "that produces the file instead.\n"
            "- run_check executes the file in the shared workspace, so "
            "a file that IMPORTS another generated file must "
            "depends_on the task producing that file (or the import "
            "fails). Put run_check on the entry-point file; library "
            "modules with no __main__ effect don't need it.\n"
            "- Each file is produced by exactly ONE task. Downstream "
            "tasks receive its contents via depends_on — they must "
            "not set extract_to to the same file.\n"
            "- When the goal names specific files, use EXACTLY those "
            "file names in extract_to.\n"
            "- Only set extract_to on a task whose entire output is "
            "that one file's contents.\n"
            "- Prefer a few substantial tasks over many small ones."
            + self._parallelism_hint()
        )

    def _parallelism_hint(self) -> str:
        """Extra planning rule when the fleet can actually run tasks
        concurrently: chains waste workers. Skipped on a single-worker
        fleet where it would just add prompt tokens."""
        slots = self._concurrency_slots()
        if slots < 2:
            return ""
        return (
            f"\n- The fleet runs up to {slots} tasks CONCURRENTLY. "
            "Tasks without depends_on start immediately in parallel. "
            "Only add depends_on when a task genuinely needs another "
            "task's output — do not chain independent work (e.g. two "
            "unrelated files can be written simultaneously by two "
            "tasks with no depends_on)."
        )

    async def _plan_loop(
        self,
        base_prompt: str,
        *,
        label: str,
        verify: bool,
    ) -> list[AgentTask]:
        """Dispatch a planning prompt and parse the result, retrying
        with the validation error fed back on a malformed plan."""
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
            # Dispatch as "planner", not "architect" — the architect
            # role prompt asks for schemas and ASCII diagrams, which
            # directly fights the JSON-only requirement here. Live
            # observation: a 7B worker under the architect identity
            # returned markdown checklists for every planning attempt.
            try:
                text = await self._run_agent("planner", prompt)
            except WorkerDispatchError as exc:
                # Infra failure (worker died, empty text): burn the
                # attempt but don't add feedback — the model never saw
                # this prompt fail.
                last_exc = exc
                log.warning("plan(%r): dispatch failed, retrying: %s", label, exc)
                continue
            try:
                tasks = self._parse_plan(text)
            except ValueError as exc:
                feedback = str(exc)
                last_exc = exc
                log.warning(
                    "plan(%r): invalid plan, retrying: %s — raw response: %r",
                    label, exc, text[:400],
                )
                continue
            if verify:
                for t in tasks:
                    t.verify = True
            log.info(
                "plan(%r): %d tasks (%s)",
                label, len(tasks), [t.role for t in tasks],
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
            if role not in ROLE_PROMPTS or role in ("planner", "auditor"):
                valid = sorted(
                    r for r in ROLE_PROMPTS if r not in ("planner", "auditor")
                )
                raise ValueError(
                    f"tasks[{i}].role={role!r} is not usable in a plan; "
                    f"valid roles: {valid}"
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
                    # Drop, don't reject. Repair planners in particular
                    # reference the PREVIOUS run's task indices (live:
                    # depends_on=[4] in a 1-task repair plan, repeated
                    # through every feedback retry — the reference is
                    # semantically meaningful to the model, so feedback
                    # can't train it away). Prompts are required to be
                    # self-contained, so a dropped dep costs context
                    # enrichment, not correctness.
                    log.info(
                        "plan: dropping invalid depends_on %r on tasks[%d] "
                        "— must reference an earlier task in THIS plan",
                        d, i,
                    )
                    continue
                deps.append(d)
            extract_to = entry.get("extract_to")
            if extract_to is not None:
                if not isinstance(extract_to, str):
                    raise ValueError(
                        f"tasks[{i}].extract_to must be a string path"
                    )
                # Planners routinely echo the schema with "" or null
                # for tasks that produce no file — treat both as
                # absent rather than failing the whole plan (live
                # observation: three consecutive plans rejected over
                # an empty extract_to on a research task).
                extract_to = extract_to.strip() or None
                if extract_to and ".." in extract_to.split("/"):
                    raise ValueError(
                        f"tasks[{i}].extract_to must not contain '..'"
                    )
            with_tools = entry.get("with_tools", False)
            if not isinstance(with_tools, bool):
                raise ValueError(f"tasks[{i}].with_tools must be a boolean")
            run_check = entry.get("run_check", False)
            if not isinstance(run_check, bool):
                raise ValueError(f"tasks[{i}].run_check must be a boolean")
            # run_check without a file is meaningless — planners echo
            # the schema flag onto no-file tasks despite the guidance
            # (live observation, 3/3 attempts), so drop it instead of
            # failing the plan. The API keeps its strict 400 for
            # explicit callers; leniency is planner-only.
            if run_check and not extract_to:
                log.info(
                    "plan: dropping run_check on tasks[%d] (%s) — no "
                    "extract_to",
                    i, role,
                )
                run_check = False
            tasks.append(AgentTask(
                role=role,
                prompt=prompt.strip(),
                depends_on=deps,
                with_tools=with_tools,
                extract_to=extract_to,
                run_check=run_check,
            ))
        # One writer per file. Live observation: a 7B planner gave five
        # of seven tasks extract_to=prime.py, each silently clobbering
        # the previous version — the shipped file was whichever task
        # happened to run last. Demote later duplicates to no-file
        # tasks (they still read the file via depends_on) instead of
        # rejecting: planners repeat this despite the guidance, and
        # rejection was observed to burn every retry. Hand-authored
        # plans via the API may still overwrite deliberately; this
        # normalization is planner-only.
        seen_targets: dict[str, int] = {}
        for i, t in enumerate(tasks):
            if not t.extract_to:
                continue
            first = seen_targets.get(t.extract_to)
            if first is not None:
                log.info(
                    "plan: tasks[%d] (%s) duplicates extract_to=%r from "
                    "tasks[%d] — demoting to a no-file task",
                    i, t.role, t.extract_to, first,
                )
                t.extract_to = None
                t.run_check = False
                continue
            seen_targets[t.extract_to] = i
        # A plan that writes Python but never executes any of it gives
        # the goal audit zero ground truth — the auditor then judges
        # from truncated text excerpts and hallucinates gaps (observed
        # live: 'missing parenthesis' on files that ran fine). Ensure
        # at least one execution point: the LAST .py-producing task is
        # the entry point by dependency order; executing a library
        # module instead is harmless (no __main__ effect, exit 0).
        py_tasks = [t for t in tasks if (t.extract_to or "").endswith(".py")]
        if py_tasks and not any(t.run_check for t in py_tasks):
            py_tasks[-1].run_check = True
            log.info(
                "plan: no run_check on any .py task — enabling it on "
                "the last file task (%s)", py_tasks[-1].extract_to,
            )
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
                goal, task, tasks, workspace_dir,
            )

            # Execute (extract-and-validate happens INSIDE the retry
            # loop when extract_to is set, so a syntax-error in the
            # written file triggers another attempt instead of leaving
            # broken code on disk).
            await self._execute_with_retry(
                task, full_prompt, workspace_dir=workspace_dir,
                all_tasks=tasks,
            )
            log.info(
                "Task %d (%s): %s in %.1fs (attempts=%d)",
                i, task.role, task.status, task.elapsed, task.attempts,
            )

        result.total_elapsed = time.perf_counter() - start
        self._synthesize(goal, result)
        return result

    def _concurrency_slots(self) -> int:
        """How many subtasks to dispatch concurrently.

        Sized from the dispatcher's live worker count when it exposes
        one (the gateway does) — dispatching more concurrent subtasks
        than the fleet has workers converts the excess into
        no-worker-available retry failures instead of throughput.
        Falls back to a modest constant for dispatchers without the
        accessor (tests, custom integrations).
        """
        counter = getattr(self.dispatcher, "available_worker_count", None)
        if callable(counter):
            try:
                return max(1, int(counter()))
            except Exception:
                return 4
        return 4

    async def run_parallel(
        self,
        goal: str,
        tasks: list[AgentTask],
        *,
        workspace_dir: str | None = None,
    ) -> OrchestratorResult:
        """Execute tasks with dependency-aware readiness scheduling.

        Every task launches the moment its dependencies complete — no
        wave barrier, so a slow branch never blocks an unrelated ready
        task (previously tasks were gathered in waves and the whole
        wave waited on its slowest member). Dependents get the same
        dependency-context injection the sequential path does.

        Concurrency is throttled to the fleet's worker count (see
        `_concurrency_slots`): ready tasks beyond that queue on the
        semaphore instead of dispatching into a saturated fleet and
        burning their retries on no-worker-available errors.

        Tasks whose dependencies failed are skipped (same cascade as
        `run`). Tasks left pending when nothing is running and nothing
        can launch — only possible with a dependency cycle — are
        skipped with the cycle called out in the result.
        """
        start = time.perf_counter()
        result = OrchestratorResult(tasks=tasks)

        slots = self._concurrency_slots()
        log.info(
            "Orchestrating %d tasks (parallel, %d slots) for: %s",
            len(tasks), slots, goal[:80],
        )
        sem = asyncio.Semaphore(slots)

        async def _exec(i: int, task: AgentTask) -> None:
            async with sem:
                # Compose inside the slot: dependencies are complete at
                # launch time, so their results are final here.
                full_prompt = self._compose_prompt(
                    goal, task, tasks, workspace_dir,
                )
                await self._execute_with_retry(
                    task, full_prompt, workspace_dir=workspace_dir,
                    all_tasks=tasks,
                )
            log.info(
                "Task %d (%s): %s in %.1fs (attempts=%d)",
                i, task.role, task.status, task.elapsed, task.attempts,
            )

        launched: set[int] = set()
        running: set[asyncio.Task[None]] = set()
        while True:
            # Cascade skips first so we never launch a task whose
            # dependency just failed.
            for i, task in enumerate(tasks):
                if task.status != "pending" or i in launched:
                    continue
                failed_deps = self._failed_deps(task, tasks)
                if failed_deps:
                    self._mark_skipped(task, failed_deps)
                    log.info(
                        "Task %d (%s): skipped (failed deps %s)",
                        i, task.role, failed_deps,
                    )
                    continue
                if all(
                    tasks[d].status == "completed"
                    for d in task.depends_on
                    if 0 <= d < len(tasks)
                ):
                    launched.add(i)
                    running.add(asyncio.create_task(_exec(i, task)))

            if not running:
                break
            done, running = await asyncio.wait(
                running, return_when=asyncio.FIRST_COMPLETED,
            )
            for finished in done:
                # Task failures are captured in AgentTask.status; an
                # exception here is an orchestrator bug — propagate.
                finished.result()

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

    async def _refresh_run_outputs(
        self,
        result: OrchestratorResult,
        workspace_dir: str | None,
    ) -> None:
        """Re-execute every run_check file against the FINAL workspace
        so the goal audit judges current evidence.

        A task's run_output is captured when its file is written — but
        a later task or repair round can rewrite a sibling the file
        imports, silently invalidating that snapshot. A failed re-run
        overwrites run_output with the error, which is exactly the
        evidence the audit needs to demand a repair. Best-effort:
        execution problems are recorded, never raised.
        """
        if not workspace_dir:
            return
        for task in result.tasks:
            if not (task.run_check and task.status == "completed"
                    and task.extracted_path):
                continue
            import sys
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, task.extracted_path,
                    cwd=workspace_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.run_check_timeout,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                task.run_output = (
                    f"[re-run of {task.extract_to} timed out after "
                    f"{self.run_check_timeout:.0f}s]"
                )
                continue
            except OSError as exc:
                task.run_output = f"[re-run of {task.extract_to} failed: {exc}]"
                continue
            if proc.returncode != 0:
                tail = stderr.decode("utf-8", errors="replace").strip()[-500:]
                task.run_output = (
                    f"[re-run of {task.extract_to} exited "
                    f"{proc.returncode}]\n{tail}"
                )
            else:
                task.run_output = stdout.decode(
                    "utf-8", errors="replace",
                )[:2000]

    @staticmethod
    def _orchestration_summary(
        result: OrchestratorResult, workspace_dir: str | None,
    ) -> str:
        """Compact ground-truth digest of an orchestration for the
        goal auditor and the repair planner: per-task status, artifact
        paths, ACTUAL execution output where available, and result
        excerpts."""
        lines: list[str] = []
        for i, t in enumerate(result.tasks):
            line = f"- task {i} ({t.role}): {t.status}"
            if t.extract_to:
                line += f", wrote {t.extract_to}"
            lines.append(line)
            if t.run_output is not None:
                lines.append(
                    f"  actual execution output: {t.run_output[:400]!r}"
                )
            if t.result:
                lines.append(f"  result excerpt: {t.result[:300]}")
        if workspace_dir:
            from pathlib import Path
            ws = Path(workspace_dir)
            if ws.is_dir():
                files = sorted(
                    str(p.relative_to(ws))
                    for p in ws.rglob("*") if p.is_file()
                )[:40]
                if files:
                    lines.append(f"Workspace files: {', '.join(files)}")
        return "\n".join(lines)

    async def check_goal(
        self,
        goal: str,
        result: OrchestratorResult,
        workspace_dir: str | None,
    ) -> tuple[bool | None, str]:
        """Audit the whole orchestration outcome against the goal.

        Per-task ``verify`` asks "did this subtask follow ITS prompt?";
        this asks "taken together, did the run achieve THE GOAL?" — a
        plan can complete every task and still miss the goal (wrong
        file name, a requirement no subtask covered, an artifact whose
        actual output contradicts the spec).

        When the fleet has capacity, a PANEL of up to three auditors
        votes concurrently and the majority verdict wins. A single
        auditor's verdict was observed (three separate live runs) to
        false-negative on outcomes whose execution output was plainly
        correct; independent votes make one hallucinated gap an
        outvoted minority instead of the final word. A tie counts as
        INCOMPLETE — the follow-through bias: repair is one bounded
        round and gets re-audited, while a wrongly-passed goal is
        silent.

        Returns ``(True, "")`` on an ACHIEVED majority, ``(False,
        gaps)`` on INCOMPLETE, and ``(None, note)`` when no auditor
        produced a parseable verdict — same fail-open stance as
        `_verify_result`: a flaky panel must not sink a finished
        orchestration.
        """
        summary = self._orchestration_summary(result, workspace_dir)
        prompt = (
            "You are auditing whether a multi-agent orchestration "
            "achieved its goal.\n\n"
            f"Goal: {goal}\n\n"
            f"What actually happened:\n{summary}\n\n"
            "Judge ONLY against the stated goal, using this rule of "
            "evidence: 'actual execution output' lines are ground "
            "truth from really running the files. If the execution "
            "output demonstrates the behavior the goal requires, the "
            "verdict is ACHIEVED — do not speculate about code you "
            "cannot see. Result excerpts are truncated; a function "
            "not visible in an excerpt is NOT a gap. Only report a "
            "gap you can point to in the goal that no evidence "
            "satisfies. Reply with exactly one line: 'VERDICT: "
            "ACHIEVED' or 'VERDICT: INCOMPLETE — <each concrete gap "
            "and which file/task it concerns>'."
        )
        # Panel sizing: one strong judge beats three weak votes. When
        # the dispatcher reports judgment calls land on the
        # coordinator's big model (see the local-planner path), a
        # single vote suffices — and avoids serializing three large
        # generations on the local runtime. Otherwise vote with up to
        # three fleet workers.
        local_judge = getattr(self.dispatcher, "prefers_local_judgment", None)
        if callable(local_judge) and local_judge():
            panel = 1
        else:
            panel = min(3, self._concurrency_slots())
        votes = await asyncio.gather(
            *[self._audit_once(prompt) for _ in range(panel)],
        )
        valid = [(v, gaps) for v, gaps in votes if v is not None]
        if not valid:
            # All auditors unavailable/unparseable; surface one note.
            return None, votes[0][1]
        achieved = sum(1 for v, _ in valid if v)
        incomplete_gaps = [gaps for v, gaps in valid if not v]
        log.info(
            "check_goal: panel=%d valid=%d achieved=%d incomplete=%d",
            panel, len(valid), achieved, len(incomplete_gaps),
        )
        if achieved > len(incomplete_gaps):
            return True, ""
        # Merge distinct gap lists so the repair planner sees every
        # complaint the majority raised.
        merged = "\n".join(dict.fromkeys(incomplete_gaps))
        return False, merged[:1500]

    async def _audit_once(self, prompt: str) -> tuple[bool | None, str]:
        """One auditor's vote: (True, ''), (False, gaps), or
        (None, note) for unavailable/unparseable."""
        import re
        try:
            text = await self._run_agent("auditor", prompt)
        except Exception as exc:
            log.warning("check_goal: auditor unavailable: %s", exc)
            return None, f"goal audit unavailable: {exc}"
        match = re.search(r"VERDICT:\s*(ACHIEVED|INCOMPLETE)", text, re.IGNORECASE)
        if match is None:
            log.warning("check_goal: no parseable verdict: %r", text[:120])
            return None, "goal audit returned no parseable verdict"
        if match.group(1).upper() == "ACHIEVED":
            return True, ""
        gaps = text[match.end():].strip(" -—:\n") or "no gaps listed"
        return False, gaps[:1500]

    @staticmethod
    def _current_file_contents(
        result: OrchestratorResult,
        workspace_dir: str | None,
        *,
        limit_files: int = 4,
        limit_chars: int = 2500,
    ) -> str:
        """Trimmed current contents of the files this orchestration
        extracted — the repair planner and repair workers must ground
        their fixes in what is ACTUALLY on disk, not in stale result
        excerpts."""
        if not workspace_dir:
            return ""
        from pathlib import Path
        blocks: list[str] = []
        seen: set[str] = set()
        for t in result.tasks:
            if not t.extract_to or t.extract_to in seen:
                continue
            seen.add(t.extract_to)
            p = Path(workspace_dir) / t.extract_to
            if not p.is_file():
                continue
            body = p.read_text(encoding="utf-8", errors="replace")[:limit_chars]
            blocks.append(
                f"Current contents of {t.extract_to}:\n```\n{body}\n```"
            )
            if len(blocks) >= limit_files:
                break
        return "\n\n".join(blocks)

    async def plan_repair(
        self,
        goal: str,
        gaps: str,
        result: OrchestratorResult,
        *,
        workspace_dir: str | None = None,
        max_tasks: int = 4,
        verify: bool = False,
    ) -> list[AgentTask]:
        """Generate a short, targeted plan that fixes the audited gaps.

        The repair planner sees the goal, the audit's gap list, the
        ground-truth digest (statuses, artifacts, real execution
        output), and the CURRENT contents of extracted files — so it
        patches what is actually broken instead of re-planning the
        whole goal from scratch. Each repair task that rewrites a file
        also gets that file's current contents injected as context,
        because repair prompts can't rely on depends_on: the previous
        run's tasks aren't in this plan's index space.

        Repair tasks MAY rewrite existing files: the one-writer-per-
        file rule holds within a single plan, not across rounds.
        """
        summary = self._orchestration_summary(result, workspace_dir)
        files_block = self._current_file_contents(result, workspace_dir)
        base_prompt = (
            "A multi-agent orchestration ran but did NOT fully achieve "
            "its goal. Produce a SHORT repair plan of 1-"
            f"{max_tasks} subtasks that fixes ONLY the gaps below — do "
            "not redo work that already succeeded.\n\n"
            f"Goal: {goal}\n\n"
            f"Gaps found by the auditor:\n{gaps}\n\n"
            f"What already happened:\n{summary}\n\n"
            + (f"{files_block}\n\n" if files_block else "")
            + "A repair task that rewrites an existing file must produce "
            "the COMPLETE corrected file, not a diff. Repair tasks run "
            "fresh — they cannot depend_on tasks from the previous "
            "run, so each prompt must carry everything the worker "
            "needs.\n\n"
            f"{self._plan_schema_guide()}"
        )
        tasks = await self._plan_loop(
            base_prompt, label=f"repair:{goal[:48]}", verify=verify,
        )
        # Ground each file-rewriting repair task in the file's current
        # contents — its prompt was written by a planner that saw them,
        # but the worker executing it won't have, and depends_on can't
        # bridge runs.
        if workspace_dir:
            from pathlib import Path
            for t in tasks:
                if not t.extract_to:
                    continue
                p = Path(workspace_dir) / t.extract_to
                if p.is_file():
                    body = p.read_text(
                        encoding="utf-8", errors="replace",
                    )[:4000]
                    t.context = (
                        f"Current contents of {t.extract_to} (rewrite "
                        f"this file completely):\n```\n{body}\n```\n"
                        + t.context
                    )
        return tasks

    async def run_goal(
        self,
        goal: str,
        tasks: list[AgentTask],
        *,
        workspace_dir: str | None = None,
        parallel: bool = False,
        goal_check: bool = False,
        repair: bool = False,
        verify: bool = False,
    ) -> OrchestratorResult:
        """Execute tasks, optionally audit the outcome against the
        goal, and optionally run one adaptive repair round.

        The full follow-through pipeline:

        1. run / run_parallel (per-task retries, validation, verify,
           run_check all apply as configured on the tasks)
        2. goal_check: a reviewer-role audit of the WHOLE outcome
        3. repair (requires goal_check): if the audit found gaps, ask
           the planner for a targeted repair plan, execute it in the
           same workspace, and re-audit once

        One repair round only — a goal the fleet can't reach in two
        audited passes needs a human (or a better goal), not an
        unbounded loop burning workers.
        """
        runner = self.run_parallel if parallel else self.run
        result = await runner(goal, tasks, workspace_dir=workspace_dir)
        if not goal_check:
            return result

        start_check = time.perf_counter()
        await self._refresh_run_outputs(result, workspace_dir)
        achieved, feedback = await self.check_goal(goal, result, workspace_dir)
        result.goal_achieved = achieved
        result.goal_feedback = feedback

        if repair and achieved is False:
            log.info("run_goal: audit found gaps, planning repair: %s",
                     feedback[:200])
            try:
                repair_tasks = await self.plan_repair(
                    goal, feedback, result,
                    workspace_dir=workspace_dir, verify=verify,
                )
            except ValueError as exc:
                result.goal_feedback += f"\n(repair planning failed: {exc})"
                result.total_elapsed += time.perf_counter() - start_check
                return result
            await runner(goal, repair_tasks, workspace_dir=workspace_dir)
            result.tasks.extend(repair_tasks)
            result.repair_tasks_added = len(repair_tasks)
            await self._refresh_run_outputs(result, workspace_dir)
            achieved, feedback = await self.check_goal(
                goal, result, workspace_dir,
            )
            result.goal_achieved = achieved
            result.goal_feedback = feedback
            # Re-synthesize over the full task list (initial + repair).
            result.synthesis = ""
            self._synthesize(goal, result)

        result.total_elapsed += time.perf_counter() - start_check
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

        from dreamland.agent.runtime import AgentRuntime

        agent_config = copy.deepcopy(self.config)
        agent_config.identity = role_system

        runtime = AgentRuntime(agent_config, skills=self.skills, memory=self.memory)
        conv = Conversation(channel=f"orchestrator:{role}")
        conv.add(Role.USER, prompt)

        response = await runtime.step(conv)
        return response.content
