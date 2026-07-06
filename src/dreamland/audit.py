"""Durable tool-call audit log.

Every tool the agent invokes is appended here as one JSON line, so a
rogue-model episode can be reconstructed after the fact — what ran, with
what arguments, and whether it succeeded, was blocked, or errored.
Conversation transcripts capture the chat; this captures the *actions*.

Design:

* **Append-only JSONL** at ``~/.dreamland/audit/tool-calls.jsonl`` (override
  with ``DREAMLAND_AUDIT_PATH``). One self-contained object per line so the
  log survives partial writes and is trivial to grep/tail.
* **Never breaks execution.** Logging is best-effort; any failure here is
  swallowed so the audit layer can't take down a tool call.
* **Secret-aware.** Argument values whose key looks like a credential are
  redacted, and every value is length-capped so the log can't be used to
  smuggle or bloat data.
* **Risk-tagged.** Known dangerous tools (shell, exfil, persistence,
  secret-read) are flagged so the log can be filtered/alerted on without
  re-deriving the threat model.
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("dreamland.audit")

# Tools that matter most in a rogue-model post-mortem. The tag is purely
# advisory metadata in the log line — it does not gate anything (that's
# the gating layer's job, step 4). Grouped by the capability they grant.
_RISK_TAGS: dict[str, str] = {
    # arbitrary code / host control
    "run_command": "exec",
    "run_shell": "exec",
    # outbound / exfil
    "webhook_post": "exfil",
    "slack_notify": "exfil",
    "http_request": "exfil",
    "http_get": "exfil",
    "http_post": "exfil",
    "send_email": "exfil",
    "gmail_send": "exfil",
    "dns_lookup": "exfil",
    # secret read
    "get_env": "secret",
    "list_env": "secret",
    "read_dotenv": "secret",
    "keychain_get": "secret",
    # persistence
    "cron_add": "persist",
    "crontab_set": "persist",
    "systemd_enable": "persist",
    # lateral movement
    "ssh_run": "lateral",
    "ssh_exec": "lateral",
    # memory / self-influence
    "remember": "memory",
    "forget": "memory",
}

# Argument keys whose values are redacted regardless of content.
_SECRET_KEY_HINT = ("password", "passwd", "secret", "token", "api_key",
                    "apikey", "credential", "auth", "private_key")

_MAX_VALUE_LEN = 512
_MAX_RESULT_LEN = 512


# Active session for the current async task. The skill registry's audit
# call and the filesystem write-hook read this when no explicit session
# is threaded through — set by the gateway around a chat turn so
# tool activity can be attributed to the conversation that caused it.
_active_session: ContextVar[str | None] = ContextVar(
    "dreamland_active_session", default=None,
)


def set_active_session(session: str | None):
    """Bind the active session for this task; returns the reset token."""
    return _active_session.set(session)


def reset_active_session(token) -> None:
    _active_session.reset(token)


def get_active_session() -> str | None:
    return _active_session.get()


def risk_tag(tool_name: str) -> str:
    """Return the risk category for a tool, or 'low' if untagged."""
    return _RISK_TAGS.get(tool_name, "low")


def _audit_path() -> Path:
    override = os.environ.get("DREAMLAND_AUDIT_PATH")
    if override:
        return Path(override)
    # Follow the resolved home (which honors the legacy ~/.towel
    # fallback) instead of hardcoding ~/.dreamland — a hardcoded path
    # silently split the audit log from the rest of the data dir.
    from dreamland.config import DREAMLAND_HOME
    return DREAMLAND_HOME / "audit" / "tool-calls.jsonl"


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact secret-looking values and length-cap everything else."""
    out: dict[str, Any] = {}
    for k, v in (arguments or {}).items():
        kl = str(k).lower()
        if any(h in kl for h in _SECRET_KEY_HINT):
            out[k] = "<redacted>"
            continue
        s = v if isinstance(v, int | float | bool) or v is None else str(v)
        if isinstance(s, str) and len(s) > _MAX_VALUE_LEN:
            s = s[:_MAX_VALUE_LEN] + f"...<+{len(s) - _MAX_VALUE_LEN} chars>"
        out[k] = s
    return out


def audit_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    status: str,
    result: Any = None,
    error: str | None = None,
    duration_ms: float | None = None,
    session: str | None = None,
) -> None:
    """Append one audit record. Best-effort; never raises.

    ``status`` is one of: ``ok`` | ``error`` | ``blocked``.
    """
    try:
        result_preview = None
        if result is not None:
            result_preview = str(result)
            if len(result_preview) > _MAX_RESULT_LEN:
                result_preview = (
                    result_preview[:_MAX_RESULT_LEN]
                    + f"...<+{len(result_preview) - _MAX_RESULT_LEN} chars>"
                )
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "tool": tool_name,
            "risk": risk_tag(tool_name),
            "status": status,
            "session": session or _active_session.get(),
            "args": _redact(arguments),
            "result": result_preview,
            "error": error,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        }
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # never let auditing break a tool call
        log.debug("audit write failed: %s", exc)
