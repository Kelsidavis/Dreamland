"""Persistence for chat-generated projects.

When the agent writes files during a normal chat (via the filesystem
tools), that work is a *project* just as much as an orchestration run
is — but it lived nowhere the Projects panel could see it. This store
records, per chat session, the directory tree the session produced
files in, so those projects consolidate alongside orchestration runs.

A "project" here is identified by its session and rooted at the common
parent directory of every file that session wrote — the session's
working directory in practice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dreamland.config import DREAMLAND_HOME

DEFAULT_CHAT_PROJECTS_PATH = DREAMLAND_HOME / "chat_projects.json"

# Keep the newest N sessions; records are tiny (root + counters), but an
# unbounded map still grows forever on a long-lived coordinator.
MAX_RECORDS = 100


class ChatProjectStore:
    """JSON-backed map of session id -> chat-project record."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CHAT_PROJECTS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        """Load records, backing up a corrupt file rather than letting
        the next save() overwrite (and destroy) it — same pattern as
        the other stores."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._back_up_corrupt(exc)
            return {}
        if not isinstance(data, dict):
            self._back_up_corrupt(
                ValueError(f"top-level shape is {type(data).__name__}")
            )
            return {}
        return {
            k: v for k, v in data.items()
            if isinstance(k, str) and isinstance(v, dict)
        }

    def save(self, records: dict[str, dict[str, Any]]) -> None:
        """Persist, keeping only the newest MAX_RECORDS by last_ts."""
        if len(records) > MAX_RECORDS:
            newest = sorted(
                records.items(),
                key=lambda kv: kv[1].get("last_ts", ""),
                reverse=True,
            )[:MAX_RECORDS]
            records = dict(newest)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(records, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    @staticmethod
    def upsert(
        records: dict[str, dict[str, Any]],
        session: str,
        file_path: str,
        ts: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Record one file write against a session, generalizing the
        project root to the common parent of everything it has written.

        Returns the updated record. Root generalization is monotonic:
        each new write can only widen the root, never narrow it, so the
        root converges on the session's working directory.
        """
        parent = str(Path(file_path).expanduser().resolve().parent)
        rec = records.get(session)
        if rec is None:
            rec = {
                "root": parent,
                "count": 1,
                "first_ts": ts,
                "last_ts": ts,
                "title": title,
            }
        else:
            existing = rec.get("root") or parent
            try:
                rec["root"] = os.path.commonpath([existing, parent])
            except ValueError:
                # Different drives (Windows) — keep the existing root.
                rec["root"] = existing
            rec["count"] = int(rec.get("count", 0)) + 1
            rec["last_ts"] = ts
            if title and not rec.get("title"):
                rec["title"] = title
        records[session] = rec
        return rec

    def _back_up_corrupt(self, exc: Exception) -> None:
        import logging
        import time
        backup = self.path.with_suffix(f".corrupted-{int(time.time())}")
        try:
            self.path.rename(backup)
        except OSError:
            return
        logging.getLogger("dreamland.persistence").warning(
            "chat_projects store corrupt (%s); moved to %s", exc, backup,
        )
