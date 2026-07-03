"""Tests for the multi-agent orchestrator."""

import asyncio

from towel.agent.orchestrator import (
    ROLE_PROMPTS,
    AgentTask,
    Orchestrator,
    OrchestratorResult,
)


class TestAgentTask:
    def test_defaults(self):
        t = AgentTask(role="coder", prompt="write hello world")
        assert t.status == "pending"
        assert t.result == ""
        assert t.depends_on == []

    def test_to_dict(self):
        t = AgentTask(
            role="reviewer",
            prompt="check code",
            status="completed",
            elapsed=2.5,
            result="looks good",
        )
        d = t.to_dict()
        assert d["role"] == "reviewer"
        assert d["status"] == "completed"
        assert d["result_length"] == 10


class TestOrchestratorResult:
    def test_success_all_completed(self):
        tasks = [
            AgentTask(role="coder", prompt="x", status="completed", result="code"),
            AgentTask(role="reviewer", prompt="y", status="completed", result="ok"),
        ]
        r = OrchestratorResult(tasks=tasks)
        assert r.success

    def test_failure_if_any_failed(self):
        tasks = [
            AgentTask(role="coder", prompt="x", status="completed", result="code"),
            AgentTask(role="reviewer", prompt="y", status="failed", result="error"),
        ]
        r = OrchestratorResult(tasks=tasks)
        assert not r.success

    def test_summary(self):
        tasks = [
            AgentTask(role="coder", prompt="write", status="completed", elapsed=1.2, result="done"),
        ]
        r = OrchestratorResult(tasks=tasks, total_elapsed=1.5)
        s = r.summary()
        assert "1 tasks" in s
        assert "coder" in s
        assert "completed" in s


class TestRolePrompts:
    def test_all_roles_have_prompts(self):
        expected = {
            "coder",
            "researcher",
            "reviewer",
            "writer",
            "architect",
            "tester",
            "debugger",
            "default",
        }
        assert expected.issubset(set(ROLE_PROMPTS.keys()))

    def test_prompts_are_nonempty(self):
        for role, prompt in ROLE_PROMPTS.items():
            assert len(prompt) > 20, f"Prompt for {role} is too short"


class TestOrchestrator:
    def test_instantiation(self):
        from towel.config import TowelConfig

        config = TowelConfig()
        orch = Orchestrator(config)
        assert orch is not None

    def test_task_dependency_tracking(self):
        tasks = [
            AgentTask(role="architect", prompt="design"),
            AgentTask(role="coder", prompt="implement", depends_on=[0]),
            AgentTask(role="reviewer", prompt="review", depends_on=[1]),
        ]
        assert tasks[1].depends_on == [0]
        assert tasks[2].depends_on == [1]


class _RecordingDispatcher:
    """In-process RoleDispatcher used to verify the orchestrator's
    role-routing contract without touching real workers."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

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
        self.calls.append({
            "role": role,
            "role_system": role_system,
            "prompt": prompt,
            "session_id": session_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "with_tools": with_tools,
            "task_type": task_type,
            "exclude_workers": set(exclude_workers) if exclude_workers else set(),
        })
        return f"[{role} result for: {prompt[:40]}]"


class TestOrchestratorWithDispatcher:
    """End-to-end: confirm that `dispatcher`, when set, replaces the
    local AgentRuntime path. These tests pin the contract the gateway
    must satisfy (see RoleDispatcher Protocol)."""

    def test_dispatcher_invoked_per_task(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="architect", prompt="design API"),
            AgentTask(role="coder", prompt="write impl"),
        ]
        result = asyncio.run(orch.run("Build a thing", tasks))
        assert result.success
        assert len(dispatcher.calls) == 2
        assert dispatcher.calls[0]["role"] == "architect"
        assert dispatcher.calls[1]["role"] == "coder"
        # Each subtask must get its own session id so role affinities
        # don't bleed between them.
        assert dispatcher.calls[0]["session_id"] != dispatcher.calls[1]["session_id"]
        # The dispatcher receives the role's system identity verbatim
        # — the gateway needs this to set identity_override.
        assert "software engineer" in dispatcher.calls[1]["role_system"].lower() \
            or "code" in dispatcher.calls[1]["role_system"].lower()

    def test_dispatcher_receives_dependency_context(self):
        """When a task depends on a prior task, its prompt must
        include the prior task's result — that's how piecemeal
        coordination actually shares state across workers."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="architect", prompt="design"),
            AgentTask(role="coder", prompt="implement", depends_on=[0]),
        ]
        asyncio.run(orch.run("Build a thing", tasks))
        # The second call's prompt must mention the first task's result
        # (the recording dispatcher echoes the role+prompt as its result).
        second_prompt = dispatcher.calls[1]["prompt"]
        assert "Result from architect" in second_prompt
        # The architect role's result starts with "[architect result for:"
        # — its exact suffix depends on prompt truncation, so just check
        # the role tag round-tripped into the next subtask's prompt.
        assert "[architect result for:" in second_prompt

    def test_dispatcher_run_parallel_independent_sessions(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="file a"),
            AgentTask(role="coder", prompt="file b"),
            AgentTask(role="coder", prompt="file c"),
        ]
        asyncio.run(orch.run_parallel("Build three files", tasks))
        assert all(t.status == "completed" for t in tasks)
        # Three distinct session_ids — parallel subtasks must not
        # share a session or the dispatcher's session-pinning code
        # serializes them onto one worker.
        sids = {c["session_id"] for c in dispatcher.calls}
        assert len(sids) == 3

    def test_role_to_task_type_mapping_flows_to_dispatcher(self):
        """Without this, the workspace preamble the orchestrator
        prepends to subtask prompts prevented the keyword classifier
        from triggering (prompt no longer starts with 'write …') —
        coder/architect/tester subtasks fell through to role_match
        and skipped the dispatcher's prefer_quality preempt path.
        Explicit role→task_type mapping closes the gap."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="architect", prompt="plan it"),
            AgentTask(role="coder", prompt="write it"),
            AgentTask(role="tester", prompt="test it"),
            AgentTask(role="reviewer", prompt="review it"),
            AgentTask(role="writer", prompt="document it"),
            AgentTask(role="researcher", prompt="research it"),
            AgentTask(role="debugger", prompt="debug it"),
            AgentTask(role="default", prompt="something else"),
        ]
        asyncio.run(orch.run("g", tasks))
        types_by_role = {c["role"]: c["task_type"] for c in dispatcher.calls}
        assert types_by_role["architect"] == "plan"
        assert types_by_role["coder"] == "generate"
        assert types_by_role["tester"] == "test_gen"
        assert types_by_role["reviewer"] == "code_review"
        assert types_by_role["writer"] == "draft"
        assert types_by_role["researcher"] == "research"
        assert types_by_role["debugger"] == "analyze"
        # `default` has no mapping — falls through to classifier.
        assert types_by_role["default"] is None

    def test_with_tools_flows_through_to_dispatcher(self):
        """A subtask declared with_tools=True must hand that down to
        the dispatcher — without this, "coder" subtasks can never call
        write_file etc., which makes piecemeal artifact building
        impossible regardless of how good the planning is."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="writer", prompt="explain x"),               # no tools
            AgentTask(role="coder", prompt="write x.py", with_tools=True),
        ]
        asyncio.run(orch.run("g", tasks))
        assert dispatcher.calls[0]["with_tools"] is False
        assert dispatcher.calls[1]["with_tools"] is True

    def test_dispatcher_error_propagates_as_failed_task(self):
        from towel.config import TowelConfig

        class _BrokenDispatcher:
            async def dispatch_role_task(self, *args, **kwargs) -> str:  # noqa: ARG002
                raise RuntimeError("worker timed out")

        # max_attempts=1 disables the default retry so the test is
        # checking the failure-propagation path, not the retry path.
        orch = Orchestrator(
            TowelConfig(), dispatcher=_BrokenDispatcher(), max_attempts=1,
        )
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run("g", tasks))
        assert not result.success
        assert tasks[0].status == "failed"
        assert "worker timed out" in tasks[0].result
        assert tasks[0].attempts == 1

    def test_retry_recovers_when_second_attempt_succeeds(self):
        """When a subtask fails once then succeeds, the orchestrator
        marks the task completed and records the attempt count.
        This is the codex-style "primary worker emitted empty text →
        alt worker answered" pattern."""
        from towel.agent.orchestrator import WorkerDispatchError
        from towel.config import TowelConfig

        attempts = {"count": 0}
        seen_excludes: list[set[str]] = []

        class _FlakyDispatcher:
            async def dispatch_role_task(self, *args, **kwargs) -> str:  # noqa: ARG002
                attempts["count"] += 1
                seen_excludes.append(
                    set(kwargs.get("exclude_workers") or ())
                )
                if attempts["count"] == 1:
                    raise WorkerDispatchError(
                        "primary returned empty", worker_id="primary",
                    )
                return "real answer on retry"

        orch = Orchestrator(TowelConfig(), dispatcher=_FlakyDispatcher())
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert tasks[0].status == "completed"
        assert tasks[0].result == "real answer on retry"
        assert tasks[0].attempts == 2
        # First attempt had no exclude_workers; second attempt
        # excludes the worker that just failed.
        assert seen_excludes[0] == set()
        assert seen_excludes[1] == {"primary"}

    def test_retry_gives_up_after_max_attempts(self):
        from towel.config import TowelConfig

        attempt_log: list[int] = []

        class _AlwaysFails:
            async def dispatch_role_task(self, *args, **kwargs) -> str:  # noqa: ARG002
                attempt_log.append(1)
                raise RuntimeError(f"fail #{len(attempt_log)}")

        orch = Orchestrator(
            TowelConfig(), dispatcher=_AlwaysFails(), max_attempts=3,
        )
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run("g", tasks))
        assert not result.success
        # Tried exactly max_attempts times.
        assert len(attempt_log) == 3
        assert tasks[0].attempts == 3
        # Final error message reflects the last failure.
        assert "fail #3" in tasks[0].result

    def test_workspace_dir_injected_into_prompts(self):
        """When workspace_dir is set, every subtask must see a
        workspace directive — that's how a `coder` subtask knows where
        to write so the next subtask can read from the same place."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="write game.py"),
            AgentTask(role="tester", prompt="read game.py and add tests"),
        ]
        asyncio.run(orch.run("g", tasks, workspace_dir="/tmp/orch-test"))
        for call in dispatcher.calls:
            assert "/tmp/orch-test" in call["prompt"]
            assert "Shared workspace" in call["prompt"]

    def test_workspace_dir_absent_no_preamble(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="x")]
        asyncio.run(orch.run("g", tasks))
        assert "Shared workspace" not in dispatcher.calls[0]["prompt"]

    def test_workspace_dir_parallel_too(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="file a"),
            AgentTask(role="coder", prompt="file b"),
        ]
        asyncio.run(
            orch.run_parallel("g", tasks, workspace_dir="/tmp/orch-par"),
        )
        for call in dispatcher.calls:
            assert "/tmp/orch-par" in call["prompt"]

    def test_failed_dep_skips_dependents(self):
        """When a subtask fails after all retries, dependent subtasks
        should be `skipped` rather than run with the dep's error
        string injected as context — that wastes worker time and
        produces nonsensical output."""
        from towel.config import TowelConfig

        class _FailFirst:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools, task_type,
                exclude_workers,
            ):
                self.calls.append(role)
                if role == "architect":
                    raise RuntimeError("architect timed out")
                return f"{role}-result"

        dispatcher = _FailFirst()
        orch = Orchestrator(
            TowelConfig(), dispatcher=dispatcher, max_attempts=1,
        )
        tasks = [
            AgentTask(role="architect", prompt="plan"),
            AgentTask(role="coder", prompt="impl", depends_on=[0]),
            AgentTask(role="reviewer", prompt="review", depends_on=[1]),
        ]
        result = asyncio.run(orch.run("g", tasks))
        # Architect ran and failed.
        assert tasks[0].status == "failed"
        # Coder and reviewer were skipped — never dispatched.
        assert tasks[1].status == "skipped"
        assert tasks[2].status == "skipped"
        assert "depends on task(s) [0]" in tasks[1].result
        assert "depends on task(s) [1]" in tasks[2].result
        assert dispatcher.calls == ["architect"]  # no waste on dependents
        assert not result.success

    def test_skipped_task_no_synthesis(self):
        """When any task is skipped, the run is not 'success' and the
        markdown synthesis block stays empty — operators reading the
        response don't get a misleadingly-complete summary."""
        from towel.config import TowelConfig

        class _FailDep:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools, task_type,
                exclude_workers,
            ):
                if role == "architect":
                    raise RuntimeError("nope")
                return f"{role}-result"

        orch = Orchestrator(
            TowelConfig(), dispatcher=_FailDep(), max_attempts=1,
        )
        tasks = [
            AgentTask(role="architect", prompt="plan"),
            AgentTask(role="coder", prompt="impl", depends_on=[0]),
        ]
        result = asyncio.run(orch.run("g", tasks))
        assert not result.success
        assert result.synthesis == ""

    def test_extract_to_writes_fenced_code_block(self, tmp_path):
        """Lets a no-tools chat-fast coder produce code without going
        through the slow tool loop. Models often wrap code in ```python
        fences; the orchestrator extracts the first block and writes
        it to the workspace path the caller specified."""
        from towel.config import TowelConfig

        class _Echo:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                return (
                    "Here is the function:\n\n```python\n"
                    "def hello():\n    return 'hi'\n```\n\nDone."
                )

        orch = Orchestrator(TowelConfig(), dispatcher=_Echo())
        tasks = [
            AgentTask(role="coder", prompt="write hello", extract_to="hello.py"),
        ]
        ws = str(tmp_path / "ws")
        result = asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        assert result.success
        # Extracted body landed on disk.
        target = tmp_path / "ws" / "hello.py"
        assert target.exists()
        body = target.read_text(encoding="utf-8")
        assert "def hello()" in body
        assert "return 'hi'" in body
        # Python fence stripped — no triple-backticks in body.
        assert "```" not in body
        assert tasks[0].extracted_path == str(target.resolve())

    def test_extract_to_no_fence_writes_whole_response(self, tmp_path):
        """When the model doesn't use fences, write the whole stripped
        body anyway — a code-shaped response without backticks is
        still useful."""
        from towel.config import TowelConfig

        class _Plain:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                return "def f(): return 1\n"

        orch = Orchestrator(TowelConfig(), dispatcher=_Plain())
        tasks = [
            AgentTask(role="coder", prompt="x", extract_to="f.py"),
        ]
        ws = str(tmp_path / "ws")
        result = asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        assert result.success
        assert (tmp_path / "ws" / "f.py").read_text(encoding="utf-8") == "def f(): return 1\n"

    def test_extract_to_retries_on_python_syntax_error(self, tmp_path):
        """When extract_to writes a .py file with a SyntaxError, the
        orchestrator should treat that as a failed attempt and retry —
        not leave broken code on disk and call the subtask completed.
        Model-quality issues are stochastic; a re-roll often succeeds."""
        from towel.config import TowelConfig

        attempts = {"n": 0}

        class _Flaky:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                attempts["n"] += 1
                # First attempt: syntactically invalid Python.
                # Second attempt: clean fenced block.
                if attempts["n"] == 1:
                    return "```python\ndef broken(\n    return 'oops'\n```"
                return "```python\ndef ok():\n    return 'fine'\n```"

        orch = Orchestrator(TowelConfig(), dispatcher=_Flaky(), max_attempts=3)
        tasks = [AgentTask(role="coder", prompt="x", extract_to="hello.py")]
        ws = str(tmp_path / "ws")
        result = asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        # Second attempt's valid code is what landed on disk.
        body = (tmp_path / "ws" / "hello.py").read_text(encoding="utf-8")
        assert "def ok():" in body
        assert "def broken" not in body
        assert tasks[0].attempts == 2
        assert tasks[0].status == "completed"
        assert result.success

    def test_extract_to_syntax_failure_marks_task_failed_after_attempts(self, tmp_path):
        """When every attempt produces invalid code, the task ends
        `failed` with the SyntaxError surfaced as the result. The
        partial file from the last attempt remains on disk (operator
        can inspect it) but the orchestrator reports failure."""
        from towel.config import TowelConfig

        class _AlwaysBroken:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                return "```python\ndef broken(\n    return 'oops'\n```"

        orch = Orchestrator(
            TowelConfig(), dispatcher=_AlwaysBroken(), max_attempts=2,
        )
        tasks = [AgentTask(role="coder", prompt="x", extract_to="bad.py")]
        ws = str(tmp_path / "ws")
        result = asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        assert not result.success
        assert tasks[0].status == "failed"
        assert "SyntaxError" in tasks[0].result
        assert tasks[0].attempts == 2

    def test_extract_to_rejects_bare_identifier_no_substance(self, tmp_path):
        """ast.parse accepts a single identifier as valid Python — but
        a file containing just `write_file` is not real code. Live
        observation: a coder subtask returned the literal tool-name
        text, parsed cleanly, wrote 11 bytes to disk. The substance
        check catches that."""
        from towel.config import TowelConfig

        attempts = {"n": 0}

        class _Flaky:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    return "write_file"  # bare identifier
                return "```python\ndef ok():\n    return 'fine'\n```"

        orch = Orchestrator(TowelConfig(), dispatcher=_Flaky(), max_attempts=3)
        tasks = [AgentTask(role="coder", prompt="x", extract_to="m.py")]
        ws = str(tmp_path / "ws")
        result = asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        assert result.success
        assert tasks[0].attempts == 2
        body = (tmp_path / "ws" / "m.py").read_text(encoding="utf-8")
        assert "def ok" in body
        assert "write_file" not in body

    def test_extract_to_rejects_path_traversal(self, tmp_path):
        """A model-suggested `extract_to` shouldn't be able to write
        outside the workspace. Path resolution + ancestor check."""
        from towel.config import TowelConfig

        class _Echo:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                return "```python\nx = 1\n```"

        orch = Orchestrator(TowelConfig(), dispatcher=_Echo())
        tasks = [
            AgentTask(role="coder", prompt="x", extract_to="../escape.py"),
        ]
        ws = str(tmp_path / "ws")
        (tmp_path / "ws").mkdir()
        asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        # Task marked failed because the extract path escaped.
        assert tasks[0].status == "failed"
        assert "escape" not in (tmp_path / "escape.py").exists().__str__() or \
            not (tmp_path / "escape.py").exists()
        # And the original error surfaces in the task result.
        assert "outside workspace" in tasks[0].result

    def test_retry_max_attempts_floor_is_one(self):
        """max_attempts<=0 should clamp to 1 — orchestrator must always
        try at least once per subtask, never zero-attempts."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=0)
        tasks = [AgentTask(role="coder", prompt="x")]
        asyncio.run(orch.run("g", tasks))
        assert len(dispatcher.calls) == 1
        assert tasks[0].attempts == 1


class TestRetryFeedback:
    """A rejected attempt must retry WITH the rejection reason in the
    prompt — re-rolling blind wastes the retry on the same mistake."""

    def test_syntax_rejection_feeds_back_into_retry_prompt(self, tmp_path):
        from towel.config import TowelConfig

        class _Flaky:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return "```python\ndef broken(\n```"
                return "```python\ndef ok():\n    return 1\n```"

        dispatcher = _Flaky()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = [AgentTask(role="coder", prompt="write f.py", extract_to="f.py")]
        ws = tmp_path / "ws"
        ws.mkdir()
        result = asyncio.run(orch.run("g", tasks, workspace_dir=str(ws)))
        assert result.success
        # The second prompt must carry the rejection and the reason.
        assert "rejected" in dispatcher.prompts[1]
        assert "SyntaxError" in dispatcher.prompts[1]
        # The first prompt must NOT (no feedback yet).
        assert "rejected" not in dispatcher.prompts[0]

    def test_infra_failure_does_not_add_feedback(self):
        """WorkerDispatchError retries the same prompt verbatim — the
        model never saw the failure, so there is nothing to correct."""
        from towel.agent.orchestrator import WorkerDispatchError
        from towel.config import TowelConfig

        class _FlakyInfra:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    raise WorkerDispatchError("timeout", worker_id="w1")
                return "fine"

        dispatcher = _FlakyInfra()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = [AgentTask(role="writer", prompt="write docs")]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert dispatcher.prompts[0] == dispatcher.prompts[1]


class TestVerify:
    """`verify=True` routes completed results through a reviewer-role
    check; FAIL retries with the reviewer's feedback, PASS marks the
    task verified."""

    def test_fail_then_pass_retries_with_feedback(self):
        from towel.config import TowelConfig

        class _Reviewer:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.reviews = 0

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.calls.append({"role": role, "prompt": prompt})
                if role == "reviewer":
                    self.reviews += 1
                    if self.reviews == 1:
                        return "VERDICT: FAIL — missing the goodbye() function"
                    return "VERDICT: PASS"
                return "def hello(): ..."

        dispatcher = _Reviewer()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = [AgentTask(role="coder", prompt="write hello and goodbye", verify=True)]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert tasks[0].verified is True
        assert tasks[0].attempts == 2
        # Retry prompt carries the reviewer's reason.
        coder_prompts = [c["prompt"] for c in dispatcher.calls if c["role"] == "coder"]
        assert len(coder_prompts) == 2
        assert "goodbye" in coder_prompts[1]
        assert "rejected" in coder_prompts[1]

    def test_terminal_fail_marks_task_failed_and_unverified(self):
        from towel.config import TowelConfig

        class _AlwaysFail:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    return "VERDICT: FAIL — wrong output"
                return "something off-task"

        orch = Orchestrator(TowelConfig(), dispatcher=_AlwaysFail(), max_attempts=2)
        tasks = [AgentTask(role="coder", prompt="do x", verify=True)]
        result = asyncio.run(orch.run("g", tasks))
        assert not result.success
        assert tasks[0].status == "failed"
        assert tasks[0].verified is False
        assert "reviewer rejected" in tasks[0].result

    def test_reviewer_unavailable_accepts_unverified(self):
        """A flaky reviewer must not kill otherwise-good work."""
        from towel.agent.orchestrator import WorkerDispatchError
        from towel.config import TowelConfig

        class _NoReviewer:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    raise WorkerDispatchError("no worker available")
                return "result"

        orch = Orchestrator(TowelConfig(), dispatcher=_NoReviewer())
        tasks = [AgentTask(role="coder", prompt="do x", verify=True)]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert tasks[0].status == "completed"
        assert tasks[0].verified is None

    def test_unparseable_verdict_accepts_unverified(self):
        from towel.config import TowelConfig

        class _Rambler:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    return "Well, it looks mostly fine to me I suppose."
                return "result"

        orch = Orchestrator(TowelConfig(), dispatcher=_Rambler())
        tasks = [AgentTask(role="coder", prompt="do x", verify=True)]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert tasks[0].verified is None


class TestParallelWaves:
    """run_parallel must respect depends_on: dependents wait for their
    dependencies and receive their output as context, same as the
    sequential path."""

    def test_dependent_sees_dependency_results(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="architect", prompt="design part A"),
            AgentTask(role="architect", prompt="design part B"),
            AgentTask(role="coder", prompt="implement both", depends_on=[0, 1]),
        ]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert result.success
        coder_call = next(c for c in dispatcher.calls if c["role"] == "coder")
        assert "design part A" in coder_call["prompt"]
        assert "design part B" in coder_call["prompt"]
        assert "Context from previous tasks" in coder_call["prompt"]

    def test_independent_tasks_share_a_wave(self):
        """Both roots must be in flight simultaneously — the wave
        gathers them together rather than serializing."""
        from towel.config import TowelConfig

        class _Barrier:
            def __init__(self) -> None:
                self.in_flight = 0
                self.max_in_flight = 0

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                await asyncio.sleep(0.01)
                self.in_flight -= 1
                return "ok"

        dispatcher = _Barrier()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="a"),
            AgentTask(role="coder", prompt="b"),
            AgentTask(role="reviewer", prompt="c", depends_on=[0, 1]),
        ]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert result.success
        assert dispatcher.max_in_flight == 2

    def test_failed_dep_skips_dependent_in_parallel(self):
        from towel.agent.orchestrator import WorkerDispatchError
        from towel.config import TowelConfig

        class _FailFirst:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if "part A" in prompt:
                    raise WorkerDispatchError("boom")
                return "ok"

        orch = Orchestrator(TowelConfig(), dispatcher=_FailFirst(), max_attempts=1)
        tasks = [
            AgentTask(role="coder", prompt="part A"),
            AgentTask(role="coder", prompt="part B"),
            AgentTask(role="reviewer", prompt="review", depends_on=[0]),
        ]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert not result.success
        assert tasks[0].status == "failed"
        assert tasks[1].status == "completed"
        assert tasks[2].status == "skipped"
        assert "did not complete" in tasks[2].result

    def test_dependency_cycle_terminates_as_skipped(self):
        """A cycle must terminate with the tasks skipped — not hang."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="a", depends_on=[1]),
            AgentTask(role="coder", prompt="b", depends_on=[0]),
        ]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert not result.success
        assert all(t.status == "skipped" for t in tasks)
        assert "cycle" in tasks[0].result

    def test_parallel_synthesis_on_success(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="a"),
            AgentTask(role="writer", prompt="b"),
        ]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert result.success
        assert "# Results for: g" in result.synthesis


class TestPlan:
    """`plan()` decomposes a goal into validated AgentTasks via an
    architect-role dispatch, retrying with feedback on a bad plan."""

    def test_plan_parses_json_array(self):
        from towel.config import TowelConfig

        class _Planner:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                return (
                    '[{"role": "coder", "prompt": "write calc.py",'
                    ' "extract_to": "calc.py"},'
                    ' {"role": "tester", "prompt": "test it",'
                    ' "depends_on": [0]}]'
                )

        orch = Orchestrator(TowelConfig(), dispatcher=_Planner())
        tasks = asyncio.run(orch.plan("build a calculator"))
        assert [t.role for t in tasks] == ["coder", "tester"]
        assert tasks[0].extract_to == "calc.py"
        assert tasks[1].depends_on == [0]

    def test_plan_tolerates_fenced_json_with_prose(self):
        from towel.config import TowelConfig

        class _Chatty:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                return (
                    "Here is the plan:\n```json\n"
                    '[{"role": "writer", "prompt": "write the docs"}]\n'
                    "```\nGood luck!"
                )

        orch = Orchestrator(TowelConfig(), dispatcher=_Chatty())
        tasks = asyncio.run(orch.plan("document the project"))
        assert len(tasks) == 1
        assert tasks[0].role == "writer"

    def test_plan_retries_with_feedback_on_invalid_plan(self):
        from towel.config import TowelConfig

        class _BadThenGood:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return '[{"role": "codr", "prompt": "typo role"}]'
                return '[{"role": "coder", "prompt": "fixed"}]'

        dispatcher = _BadThenGood()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = asyncio.run(orch.plan("g"))
        assert len(tasks) == 1
        assert tasks[0].role == "coder"
        # Second prompt carried the validation error.
        assert "codr" in dispatcher.prompts[1]
        assert "rejected" in dispatcher.prompts[1]

    def test_plan_gives_up_after_max_attempts(self):
        import pytest

        from towel.config import TowelConfig

        class _AlwaysBad:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                return "no json here"

        orch = Orchestrator(TowelConfig(), dispatcher=_AlwaysBad(), max_attempts=2)
        with pytest.raises(ValueError, match="no valid plan"):
            asyncio.run(orch.plan("g"))

    def test_plan_drops_invalid_dependencies(self):
        """Out-of-range / forward depends_on entries are dropped, not
        rejected — repair planners reference the previous run's task
        indices and repeat them through every feedback retry (live),
        so rejection deadlocks planning. Prompts are self-contained by
        rule, so a dropped dep costs context, not correctness."""
        from towel.config import TowelConfig

        class _Forward:
            def __init__(self) -> None:
                self.count = 0

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.count += 1
                return (
                    '[{"role": "coder", "prompt": "a", "depends_on": [4]},'
                    ' {"role": "coder", "prompt": "b", "depends_on": [0]}]'
                )

        dispatcher = _Forward()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = asyncio.run(orch.plan("g"))
        assert dispatcher.count == 1
        assert tasks[0].depends_on == []
        # Valid in-plan deps survive.
        assert tasks[1].depends_on == [0]

    def test_plan_verify_flag_applies_to_all_tasks(self):
        from towel.config import TowelConfig

        class _Planner:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                return (
                    '[{"role": "coder", "prompt": "a"},'
                    ' {"role": "writer", "prompt": "b"}]'
                )

        orch = Orchestrator(TowelConfig(), dispatcher=_Planner())
        tasks = asyncio.run(orch.plan("g", verify=True))
        assert all(t.verify for t in tasks)


class TestRunCheck:
    """`run_check=True` executes the extracted file coordinator-side;
    a crash or timeout rejects the attempt with the error fed back —
    real execution instead of a hallucinated 'I ran it and it works'."""

    def test_crash_feeds_stderr_back_and_retry_recovers(self, tmp_path):
        from towel.config import TowelConfig

        class _BadThenGood:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return "```python\nraise RuntimeError('kaboom')\n```"
                return "```python\nprint('recovered')\n```"

        dispatcher = _BadThenGood()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = [AgentTask(
            role="coder", prompt="write app.py",
            extract_to="app.py", run_check=True,
        )]
        ws = tmp_path / "ws"
        ws.mkdir()
        result = asyncio.run(orch.run("g", tasks, workspace_dir=str(ws)))
        assert result.success
        assert tasks[0].attempts == 2
        # The retry prompt carries the crash.
        assert "kaboom" in dispatcher.prompts[1]
        assert "run_check" in dispatcher.prompts[1]
        # Successful run captured real stdout.
        assert tasks[0].run_output == "recovered\n"

    def test_timeout_rejects(self, tmp_path):
        from towel.config import TowelConfig

        class _Sleeper:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                return "```python\nimport time\ntime.sleep(30)\n```"

        orch = Orchestrator(TowelConfig(), dispatcher=_Sleeper(), max_attempts=1)
        orch.run_check_timeout = 0.5
        tasks = [AgentTask(
            role="coder", prompt="x", extract_to="slow.py", run_check=True,
        )]
        ws = tmp_path / "ws"
        ws.mkdir()
        result = asyncio.run(orch.run("g", tasks, workspace_dir=str(ws)))
        assert not result.success
        assert tasks[0].status == "failed"
        assert "did not finish" in tasks[0].result

    def test_run_output_injected_into_dependent_context(self, tmp_path):
        from towel.config import TowelConfig

        class _Coder:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                if role == "coder":
                    return "```python\nprint(6 * 7)\n```"
                return "confirmed"

        dispatcher = _Coder()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="write answer.py",
                      extract_to="answer.py", run_check=True),
            AgentTask(role="reviewer", prompt="confirm the output",
                      depends_on=[0]),
        ]
        ws = tmp_path / "ws"
        ws.mkdir()
        result = asyncio.run(orch.run("g", tasks, workspace_dir=str(ws)))
        assert result.success
        reviewer_prompt = dispatcher.prompts[-1]
        assert "Actual execution output" in reviewer_prompt
        assert "42" in reviewer_prompt

    def test_plan_accepts_run_check(self):
        from towel.config import TowelConfig

        class _Planner:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                return (
                    '[{"role": "coder", "prompt": "write it",'
                    ' "extract_to": "x.py", "run_check": true}]'
                )

        orch = Orchestrator(TowelConfig(), dispatcher=_Planner())
        tasks = asyncio.run(orch.plan("g"))
        assert tasks[0].run_check is True

    def test_plan_drops_run_check_without_extract_to(self):
        """Planners echo run_check onto no-file tasks despite the
        guidance; that normalizes to False rather than failing the
        plan (rejection was observed to burn all attempts live)."""
        from towel.config import TowelConfig

        class _Echoey:
            def __init__(self) -> None:
                self.count = 0

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.count += 1
                return '[{"role": "coder", "prompt": "x", "run_check": true}]'

        dispatcher = _Echoey()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = asyncio.run(orch.plan("g"))
        assert len(tasks) == 1
        assert tasks[0].run_check is False
        assert dispatcher.count == 1

    def test_plan_demotes_duplicate_extract_targets(self):
        """Planners give several tasks the same extract_to despite the
        guidance; later duplicates lose extract_to/run_check instead of
        failing the plan (rejection burned every retry live) — the
        first writer wins, dependents read via depends_on."""
        from towel.config import TowelConfig

        class _Dup:
            def __init__(self) -> None:
                self.count = 0

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.count += 1
                return (
                    '[{"role": "coder", "prompt": "a", "extract_to": "x.py",'
                    ' "run_check": true},'
                    ' {"role": "reviewer", "prompt": "b",'
                    ' "extract_to": "x.py", "run_check": true,'
                    ' "depends_on": [0]}]'
                )

        dispatcher = _Dup()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=2)
        tasks = asyncio.run(orch.plan("g"))
        assert dispatcher.count == 1
        assert len(tasks) == 2
        assert tasks[0].extract_to == "x.py"
        assert tasks[0].run_check is True
        assert tasks[1].extract_to is None
        assert tasks[1].run_check is False

    def test_plan_tolerates_empty_extract_to(self):
        """Planners echo the schema with "" for no-file tasks — that
        must normalize to None, not fail the plan."""
        from towel.config import TowelConfig

        class _Planner:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                return (
                    '[{"role": "researcher", "prompt": "look", "extract_to": ""},'
                    ' {"role": "coder", "prompt": "code", "extract_to": "x.py"}]'
                )

        orch = Orchestrator(TowelConfig(), dispatcher=_Planner())
        tasks = asyncio.run(orch.plan("g"))
        assert tasks[0].extract_to is None
        assert tasks[1].extract_to == "x.py"


class TestGoalCheckAndRepair:
    """run_goal audits the WHOLE outcome against the goal and, with
    repair=True, runs one adaptive repair round on audit gaps —
    per-task verify can pass every subtask while the goal is missed."""

    def test_goal_check_achieved(self):
        from towel.config import TowelConfig

        class _Auditor:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.calls.append({"role": role, "prompt": prompt})
                if role == "reviewer":
                    return "VERDICT: ACHIEVED"
                return "done"

        dispatcher = _Auditor()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="do x")]
        result = asyncio.run(orch.run_goal(
            "the goal", tasks, goal_check=True,
        ))
        assert result.goal_achieved is True
        assert result.goal_feedback == ""
        assert result.repair_tasks_added == 0
        audit = dispatcher.calls[-1]
        assert audit["role"] == "reviewer"
        assert "the goal" in audit["prompt"]
        # The audit prompt carries the ground-truth digest.
        assert "task 0 (coder)" in audit["prompt"]

    def test_goal_check_off_by_default(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run_goal("g", tasks))
        assert result.goal_achieved is None
        # No reviewer dispatch happened.
        assert all(c["role"] == "coder" for c in dispatcher.calls)

    def test_repair_round_fixes_gaps(self, tmp_path):
        """INCOMPLETE audit → planner produces a repair task → it runs
        → second audit passes."""
        from towel.config import TowelConfig

        class _World:
            def __init__(self) -> None:
                self.audits = 0
                self.roles: list[str] = []
                self.repair_prompt: str | None = None

            def available_worker_count(self) -> int:
                # Single worker → single-auditor panel, so the audit
                # sequencing below stays deterministic.
                return 1

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.roles.append(role)
                if role == "reviewer":
                    self.audits += 1
                    if self.audits == 1:
                        return "VERDICT: INCOMPLETE — missing goodbye() in app.py"
                    return "VERDICT: ACHIEVED"
                if role == "planner":
                    self.repair_prompt = prompt
                    return (
                        '[{"role": "coder", "prompt": "add goodbye() to '
                        'app.py, produce the complete file",'
                        ' "extract_to": "app.py"}]'
                    )
                return "```python\ndef goodbye():\n    return 'bye'\n```"

        dispatcher = _World()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="write app.py")]
        ws = tmp_path / "ws"
        ws.mkdir()
        result = asyncio.run(orch.run_goal(
            "app.py with hello() and goodbye()", tasks,
            workspace_dir=str(ws), goal_check=True, repair=True,
        ))
        assert result.goal_achieved is True
        assert result.repair_tasks_added == 1
        assert len(result.tasks) == 2
        assert result.tasks[1].extract_to == "app.py"
        assert result.tasks[1].status == "completed"
        # Repair planner saw the audit gaps and the ground truth.
        assert "goodbye" in dispatcher.repair_prompt
        assert "task 0 (coder)" in dispatcher.repair_prompt
        # coder, audit, planner, repair-coder, audit
        assert dispatcher.audits == 2

    def test_repair_not_run_when_achieved(self):
        from towel.config import TowelConfig

        class _HappyAuditor:
            def __init__(self) -> None:
                self.planner_called = False

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "planner":
                    self.planner_called = True
                if role == "reviewer":
                    return "VERDICT: ACHIEVED"
                return "done"

        dispatcher = _HappyAuditor()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run_goal(
            "g", tasks, goal_check=True, repair=True,
        ))
        assert result.goal_achieved is True
        assert dispatcher.planner_called is False

    def test_auditor_unavailable_fails_open(self):
        from towel.agent.orchestrator import WorkerDispatchError
        from towel.config import TowelConfig

        class _NoAuditor:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    raise WorkerDispatchError("no worker")
                return "done"

        orch = Orchestrator(TowelConfig(), dispatcher=_NoAuditor())
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run_goal(
            "g", tasks, goal_check=True, repair=True,
        ))
        # Unknown verdict: no repair attempted, tasks stand.
        assert result.goal_achieved is None
        assert result.repair_tasks_added == 0
        assert result.success

    def test_repair_planning_failure_keeps_first_audit(self):
        from towel.config import TowelConfig

        class _BrokenPlanner:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    return "VERDICT: INCOMPLETE — gap X"
                if role == "planner":
                    return "utter nonsense"
                return "done"

        orch = Orchestrator(TowelConfig(), dispatcher=_BrokenPlanner(), max_attempts=1)
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run_goal(
            "g", tasks, goal_check=True, repair=True,
        ))
        assert result.goal_achieved is False
        assert "gap X" in result.goal_feedback
        assert "repair planning failed" in result.goal_feedback
        assert result.repair_tasks_added == 0

    def test_repair_tasks_grounded_in_current_file_contents(self, tmp_path):
        """A repair task rewriting a file must receive that file's
        CURRENT contents — depends_on can't bridge runs."""
        from towel.config import TowelConfig

        class _World:
            def __init__(self) -> None:
                self.audits = 0
                self.coder_prompts: list[str] = []

            def available_worker_count(self) -> int:
                return 1

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    self.audits += 1
                    if self.audits == 1:
                        return "VERDICT: INCOMPLETE — app.py lacks goodbye()"
                    return "VERDICT: ACHIEVED"
                if role == "planner":
                    # The planner prompt must include the current file.
                    assert "MARKER_ORIGINAL_HELLO" in prompt
                    return (
                        '[{"role": "coder", "prompt": "rewrite app.py '
                        'with hello() and goodbye()",'
                        ' "extract_to": "app.py"}]'
                    )
                self.coder_prompts.append(prompt)
                return (
                    "```python\ndef hello():\n    return 'MARKER_ORIGINAL_HELLO'\n"
                    "def goodbye():\n    return 'bye'\n```"
                )

        dispatcher = _World()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="write app.py",
                           extract_to="app.py")]
        ws = tmp_path / "ws"
        ws.mkdir()
        result = asyncio.run(orch.run_goal(
            "app.py with hello() and goodbye()", tasks,
            workspace_dir=str(ws), goal_check=True, repair=True,
        ))
        assert result.goal_achieved is True
        assert result.repair_tasks_added == 1
        # The repair coder saw the file it was rewriting.
        assert "MARKER_ORIGINAL_HELLO" in dispatcher.coder_prompts[-1]
        assert "Current contents of app.py" in dispatcher.coder_prompts[-1]


class TestReadinessScheduling:
    """run_parallel launches a task the moment its deps complete —
    no wave barrier — and throttles concurrency to fleet capacity."""

    def test_dependent_starts_before_slow_sibling_finishes(self):
        """A(slow) and B(fast) are independent; C depends on B. Under
        wave scheduling C waited for A; readiness scheduling starts C
        as soon as B completes, while A is still running."""
        import time as _time

        from towel.config import TowelConfig

        class _Timeline:
            def __init__(self) -> None:
                self.events: list[tuple[str, float]] = []
                self.a_done_at: float | None = None

            def available_worker_count(self) -> int:
                return 3

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if "slow task A" in prompt:
                    await asyncio.sleep(0.2)
                    self.a_done_at = _time.monotonic()
                    return "A done"
                if "fast task B" in prompt:
                    await asyncio.sleep(0.01)
                    return "B done"
                # C — record when it started relative to A finishing.
                self.events.append(("C-started", _time.monotonic()))
                return "C done"

        dispatcher = _Timeline()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="slow task A"),
            AgentTask(role="coder", prompt="fast task B"),
            AgentTask(role="reviewer", prompt="review B output", depends_on=[1]),
        ]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert result.success
        c_started = dispatcher.events[0][1]
        assert dispatcher.a_done_at is not None
        # C must have STARTED before A finished — pipelined, not
        # wave-barriered.
        assert c_started < dispatcher.a_done_at

    def test_concurrency_throttled_to_fleet_size(self):
        from towel.config import TowelConfig

        class _OneWorker:
            def __init__(self) -> None:
                self.in_flight = 0
                self.max_in_flight = 0

            def available_worker_count(self) -> int:
                return 1

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                await asyncio.sleep(0.01)
                self.in_flight -= 1
                return "ok"

        dispatcher = _OneWorker()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt=f"t{i}") for i in range(4)]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert result.success
        # Never more in flight than the fleet has workers.
        assert dispatcher.max_in_flight == 1

    def test_fleet_size_saturated_not_exceeded(self):
        from towel.config import TowelConfig

        class _TwoWorkers:
            def __init__(self) -> None:
                self.in_flight = 0
                self.max_in_flight = 0

            def available_worker_count(self) -> int:
                return 2

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                await asyncio.sleep(0.02)
                self.in_flight -= 1
                return "ok"

        dispatcher = _TwoWorkers()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt=f"t{i}") for i in range(5)]
        result = asyncio.run(orch.run_parallel("g", tasks))
        assert result.success
        # Both workers used, never oversubscribed.
        assert dispatcher.max_in_flight == 2

    def test_planner_hint_mentions_concurrency_on_multi_worker_fleet(self):
        from towel.config import TowelConfig

        class _BigFleet:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def available_worker_count(self) -> int:
                return 3

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                return '[{"role": "coder", "prompt": "x"}]'

        dispatcher = _BigFleet()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        asyncio.run(orch.plan("g"))
        assert "3 tasks CONCURRENTLY" in dispatcher.prompts[0]

    def test_planner_hint_absent_on_single_worker(self):
        from towel.config import TowelConfig

        class _Solo:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def available_worker_count(self) -> int:
                return 1

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                return '[{"role": "coder", "prompt": "x"}]'

        dispatcher = _Solo()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        asyncio.run(orch.plan("g"))
        assert "CONCURRENTLY" not in dispatcher.prompts[0]


class TestDiskTruthDepContext:
    def test_dependent_sees_current_file_not_chat_blob(self, tmp_path):
        """When a dependency wrote a file, the dependent's context
        carries the file's CURRENT on-disk contents, not the raw chat
        response (prose + possibly stale drafts)."""
        from towel.config import TowelConfig

        class _Coder:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                self.prompts.append(prompt)
                if role == "coder":
                    return (
                        "Here is an early draft: DRAFT_MARKER\n"
                        "```python\ndef final():\n    return 'FINAL_MARKER'\n```"
                    )
                return "reviewed"

        dispatcher = _Coder()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="write mod.py", extract_to="mod.py"),
            AgentTask(role="reviewer", prompt="review mod.py", depends_on=[0]),
        ]
        ws = tmp_path / "ws"
        ws.mkdir()
        result = asyncio.run(orch.run("g", tasks, workspace_dir=str(ws)))
        assert result.success
        reviewer_prompt = dispatcher.prompts[-1]
        # The extracted file body is present…
        assert "FINAL_MARKER" in reviewer_prompt
        assert "current" in reviewer_prompt
        # …and the chat prose around it is NOT.
        assert "DRAFT_MARKER" not in reviewer_prompt

    def test_dep_without_file_still_passes_result(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="researcher", prompt="find facts"),
            AgentTask(role="writer", prompt="write it up", depends_on=[0]),
        ]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert "Result from researcher" in dispatcher.calls[1]["prompt"]


class TestAuditPanel:
    """On a multi-worker fleet the goal audit is a majority-vote panel
    — one hallucinated gap becomes an outvoted minority instead of the
    final word (single-auditor false-negatives observed live 3x)."""

    def test_majority_outvotes_false_negative(self):
        from towel.config import TowelConfig

        class _Panel:
            def __init__(self) -> None:
                self.reviews = 0

            def available_worker_count(self) -> int:
                return 3

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    self.reviews += 1
                    if self.reviews == 2:
                        return "VERDICT: INCOMPLETE — hallucinated gap"
                    return "VERDICT: ACHIEVED"
                return "done"

        dispatcher = _Panel()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run_goal("g", tasks, goal_check=True))
        assert dispatcher.reviews == 3
        # 2 ACHIEVED vs 1 INCOMPLETE → achieved.
        assert result.goal_achieved is True

    def test_tie_counts_as_incomplete_with_merged_gaps(self):
        from towel.agent.orchestrator import WorkerDispatchError
        from towel.config import TowelConfig

        class _Split:
            def __init__(self) -> None:
                self.reviews = 0

            def available_worker_count(self) -> int:
                return 3

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    self.reviews += 1
                    if self.reviews == 1:
                        return "VERDICT: ACHIEVED"
                    if self.reviews == 2:
                        return "VERDICT: INCOMPLETE — missing avg key"
                    # Third auditor unavailable → 1-1 among valid votes.
                    raise WorkerDispatchError("no worker")
                return "done"

        dispatcher = _Split()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run_goal("g", tasks, goal_check=True))
        # Tie → follow-through bias: INCOMPLETE, gaps preserved.
        assert result.goal_achieved is False
        assert "missing avg key" in result.goal_feedback

    def test_single_worker_fleet_audits_once(self):
        from towel.config import TowelConfig

        class _Solo:
            def __init__(self) -> None:
                self.reviews = 0

            def available_worker_count(self) -> int:
                return 1

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *, session_id, max_tokens,
                temperature, with_tools, task_type, exclude_workers,
            ) -> str:
                if role == "reviewer":
                    self.reviews += 1
                    return "VERDICT: ACHIEVED"
                return "done"

        dispatcher = _Solo()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run_goal("g", tasks, goal_check=True))
        assert dispatcher.reviews == 1
        assert result.goal_achieved is True
