"""Dreamland configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel, Field


def _resolve_home() -> Path:
    """Resolve the data/config directory, honoring the towel->dreamland
    rename without disrupting existing installs:

    1. $DREAMLAND_HOME wins; $TOWEL_HOME still honored (legacy).
    2. ~/.dreamland when it exists (or when nothing legacy does).
    3. ~/.towel when it exists and ~/.dreamland doesn't — existing
       installs keep their config, conversations, memory, workspaces,
       and orchestration history with zero migration.
    """
    env = os.environ.get("DREAMLAND_HOME") or os.environ.get("TOWEL_HOME")
    if env:
        return Path(env)
    new = Path.home() / ".dreamland"
    legacy = Path.home() / ".towel"
    # Decide by config presence, not bare directory existence — stray
    # side-effect files (a test run, a lock file) must not be able to
    # flip the home away from the directory holding the user's actual
    # config and data.
    if (new / "config.toml").exists():
        return new
    if (legacy / "config.toml").exists():
        return legacy
    if new.exists():
        return new
    if legacy.exists():
        return legacy
    return new


DREAMLAND_HOME = _resolve_home()
# Legacy alias — user plugins/skills may import this name.
TOWEL_HOME = DREAMLAND_HOME


class ModelConfig(BaseModel):
    """MLX model configuration."""

    name: str = "Eldadalbajob/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-3Bit"
    max_tokens: int = 4096
    context_window: int = 262144
    auto_context: bool = True
    min_context_window: int = 32768
    temperature: float = 0.7
    top_p: float = 0.95
    turboquant: bool = True
    turboquant_bits: int = 3
    turboquant_qjl_ratio: float = 0.5


class AgentProfile(BaseModel):
    """A named agent profile — model + identity + behavior."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    identity: str = "You are Dreamland, a helpful local AI assistant. It doesn't exist."
    skills: list[str] = Field(default_factory=list)  # empty = all skills
    description: str = ""

    def effective_identity(self, base_identity: str) -> str:
        """Return this profile's identity, falling back to base."""
        if self.identity and self.identity != AgentProfile.model_fields["identity"].default:
            return self.identity
        return base_identity


class GatewayConfig(BaseModel):
    """Gateway server configuration."""

    host: str = "127.0.0.1"
    port: int = 18742  # 42 * 446 + 10, because 42
    ws_path: str = "/ws"


class ChannelDefaults(BaseModel):
    """Default channel routing configuration."""

    enabled: list[str] = Field(default_factory=lambda: ["cli", "webchat"])


class SecurityConfig(BaseModel):
    """Tool-gating policy — protects against a rogue/misaligned model.

    Dreamland hands the model a broad tool surface (shell, HTTP, secrets,
    persistence, SSH). This policy is the enforcement point that decides
    whether a tool may run at all. See ``dreamland.policy``.

    * ``audit``  — log every tool call but allow all (full visibility,
      no enforcement). Safe default for trusted/aligned models.
    * ``enforce`` — refuse tools whose risk tier is in ``blocked_risks``
      unless named in ``allow_tools``. Recommended when running
      abliterated or otherwise untrusted models.

    Environment variables (``DREAMLAND_TOOL_POLICY`` etc.) override these
    when set, so an operator can lock a deployment down without editing
    config. Empty/unset env falls back to what's saved here.
    """

    tool_policy: str = "audit"  # "audit" | "enforce"
    blocked_risks: list[str] = Field(
        default_factory=lambda: ["exec", "exfil", "secret", "persist", "lateral"]
    )
    allow_tools: list[str] = Field(default_factory=list)
    deny_tools: list[str] = Field(default_factory=list)


# Built-in agent profiles
DEFAULT_AGENTS: dict[str, dict[str, Any]] = {
    "coder": {
        "model": {"name": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", "context_window": 32768},
        "identity": (
            "You are Dreamland (coder mode), an expert software engineer. "
            "Write clean, efficient code. Explain your reasoning. "
            "Use tools to read files and run commands. It doesn't exist."
        ),
        "description": "Code generation and software engineering",
    },
    "researcher": {
        "model": {"name": "mlx-community/Llama-3.3-70B-Instruct-4bit", "context_window": 16384},
        "identity": (
            "You are Dreamland (researcher mode), a thorough research assistant. "
            "Analyze information carefully, cite sources, and present balanced views. "
            "Use tools to fetch web content and read files. It doesn't exist."
        ),
        "description": "Research, analysis, and information synthesis",
    },
    "writer": {
        "model": {"name": "mlx-community/Llama-3.3-70B-Instruct-4bit", "temperature": 0.9},
        "identity": (
            "You are Dreamland (writer mode), a creative writing assistant. "
            "Help with drafting, editing, and refining text. "
            "Adapt your tone to match the user's needs. It doesn't exist."
        ),
        "description": "Creative and technical writing",
    },
}


class DreamlandConfig(BaseModel):
    """Root configuration for Dreamland."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    channels: ChannelDefaults = Field(default_factory=ChannelDefaults)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    skills_dirs: list[str] = Field(default_factory=lambda: ["~/.dreamland/skills", "./skills"])
    identity: str = "You are Dreamland, a helpful local AI assistant. It doesn't exist."
    agents: dict[str, AgentProfile] = Field(default_factory=dict)
    default_agent: str = ""

    # Backend preference — written by the setup GUI / persistent across runs.
    # Empty means "let the CLI auto-detect (or fall back to MLX)".
    backend: str = ""
    # Backend-specific connection settings; the corresponding CLI flag still
    # overrides each one at command time.
    ollama_url: str = ""
    llama_url: str = ""
    claude_model: str = ""

    # When False, the llama backend only CONNECTS to an existing llama-server
    # and never spawns one itself. Set False when a managed service (e.g. the
    # dreamland-llama systemd unit) owns the server — this prevents Dreamland from
    # falling back to a stale ~/.local/bin/llama-server build that can't load
    # the configured model (e.g. an old build that crashes on Qwen3.6).
    llama_auto_start: bool = True

    # Heuristic auto-capture: extract user/preference/project facts from
    # every user turn and write them to memory. Conservative patterns
    # (role, employer, project, deadline, preference, explicit remember)
    # so false positives stay low. Set to False to disable entirely.
    auto_capture: bool = True

    # Orchestration planning on the coordinator's own model when it is
    # at least twice the size of the best connected worker (or the
    # fleet is empty). The plan is the highest-leverage single call in
    # an orchestration and runs once, so the latency is bounded. Set
    # False to always dispatch planning to workers.
    local_planner_enabled: bool = True

    # When regex auto-capture produces zero captures on a user turn,
    # fire a background LLM call to extract memories the patterns
    # missed (multi-sentence context, paraphrase, indirect mention).
    # Off by default because it costs one extra inference call per
    # quiet turn. Failures are silent — the user's response is never
    # blocked, but a slow backend still serializes the work.
    auto_llm_extract: bool = False

    # Cap on the per-query recall_log table. 5000 rows is ~750KB and
    # well under any actual storage concern; long-running daemons
    # auditing several months of activity may want to bump it.
    memory_recall_log_cap: int = 5000

    # Background idle tasks (lint, type-check, email triage, …) that the
    # coordinator dispatches to otherwise-idle workers. Disabled by default:
    # the common deployment is a single GPU where llama-server is --parallel 1,
    # so an in-flight idle generation monopolises the GPU and a real chat turn
    # can't start until it yields — which reads to the user as "stopped
    # responding", and preemption can't free a mid-flight generation. Opt in
    # (set True) on a multi-GPU fleet where spare workers can stay productive
    # without stealing the GPU an interactive request needs.
    idle_tasks_enabled: bool = False

    # In-memory dispatch decision history (ring buffer). 50 was the
    # original default which only covers minutes on a busy
    # coordinator. 500 trades ~50KB for several hours of audit at
    # typical traffic — still negligible. Bump higher for long-
    # running daemons where post-hoc debugging matters.
    dispatch_history_size: int = 500

    # Max seconds the coordinator waits for any single chunk from a
    # remote worker during inference. The previous hard-coded 120s
    # was too tight for cold-loaded large models — a 30B llama-server
    # on a quiet GPU routinely takes 90-180s to produce its first
    # token, and the timeout fires before the model is ready. 300s
    # is conservative; bump higher for very large models or slow
    # hosts. Failures here log a TimeoutError and tear down the WS
    # connection, so a too-low value also costs the operator a
    # reconnect cycle.
    worker_inference_timeout: float = 300.0

    # Chat-fast timeout — how long `_quick_remote_infer` waits for the
    # worker's non-streaming response on the /api/ask chat path and
    # /v1/chat/completions non-streaming. Chat is meant to feel
    # interactive, so this is shorter than `worker_inference_timeout`.
    # On worker timeout we surface a structured error to the caller
    # rather than holding the HTTP connection open. Bump higher on
    # fleets where small models cold-start slowly.
    chat_fast_timeout: float = 60.0

    # Override the IP address advertised via mDNS. Empty means
    # auto-detect via the "connect to 8.8.8.8" trick, which on
    # machines with Tailscale / WireGuard / multiple interfaces
    # often picks the wrong one — workers connect to the VPN IP
    # rather than the LAN IP. Setting this to the wired LAN address
    # makes discovery deterministic. Leave empty in single-interface
    # setups.
    mdns_advertise_ip: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> DreamlandConfig:
        """Load config from TOML file, falling back to defaults.

        On corruption (bad TOML, OS read error, model validation
        failure), rename the bad file aside to a sibling
        ``.corrupted-<ts>`` and fall back to defaults. Same pattern
        the JSON-backed stores got (5512834, 98d1c68, 8a86987,
        c62847f). Without this guard, a corrupted config.toml made
        the coordinator crash on startup — the recovery flow was
        "delete the file by hand" which also destroyed any
        operator-set non-default values.
        """
        config_path = path or DREAMLAND_HOME / "config.toml"
        if not config_path.exists():
            return cls()
        try:
            data: dict[str, Any] = toml.load(config_path)
            return cls.model_validate(data)
        except Exception as exc:
            from datetime import UTC, datetime
            backup = config_path.with_name(
                f"{config_path.name}.corrupted-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
            )
            try:
                config_path.replace(backup)
                import logging
                logging.getLogger("dreamland.config").warning(
                    "Failed to load config: %s. Backed up the bad file "
                    "to %s and falling back to defaults.",
                    exc, backup,
                )
            except OSError:
                pass
            return cls()

    def save(self, path: Path | None = None) -> None:
        """Save config to TOML file.

        Atomic write: dumps to a sibling .tmp then renames. Without
        this, a kill / disk-full mid-write leaves a half-written
        config.toml that load() backs up and replaces with defaults
        — silently undoing every operator-tuned setting (model name,
        memory paths, gateway host/port, …).
        """
        config_path = path or DREAMLAND_HOME / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = config_path.with_name(config_path.name + ".tmp")
        tmp.write_text(toml.dumps(self.model_dump()))
        tmp.replace(config_path)

    def list_agents(self) -> dict[str, AgentProfile]:
        """List all available agents (user-defined + built-in)."""
        result: dict[str, AgentProfile] = {}
        # Built-in agents first
        for name, data in DEFAULT_AGENTS.items():
            result[name] = AgentProfile.model_validate(data)
        # User agents from agents.toml
        agents_file = DREAMLAND_HOME / "agents.toml"
        if agents_file.exists():
            try:
                import toml as _toml

                for name, data in _toml.load(agents_file).items():
                    result[name] = AgentProfile.model_validate(data)
            except Exception:
                pass
        # Config-defined override everything
        result.update(self.agents)
        return result

    def get_agent(self, name: str) -> AgentProfile | None:
        """Get an agent profile by name (config, agents.toml, or built-in)."""
        all_agents = self.list_agents()
        return all_agents.get(name)

    def resolve_agent(self, agent_name: str | None = None) -> tuple[ModelConfig, str]:
        """Resolve an agent name to (model_config, identity).

        Falls back to: explicit agent -> default_agent config -> base config.
        """
        name = agent_name or self.default_agent
        if name:
            profile = self.get_agent(name)
            if profile:
                return profile.model, profile.effective_identity(self.identity)
        return self.model, self.identity
