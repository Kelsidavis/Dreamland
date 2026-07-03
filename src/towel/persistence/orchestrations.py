"""Persistence for orchestration run history.

Background orchestrations previously lived only in gateway memory — a
coordinator restart erased every record, and finished runs vanished
with no way to ask "what did the fleet build yesterday and did it pass
its goal audit?". This store keeps the terminal snapshot of each run
(the same dict the status endpoint serves) in a single JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from towel.config import TOWEL_HOME

DEFAULT_ORCHESTRATIONS_PATH = TOWEL_HOME / "orchestrations.json"

# Most-recent cap. Records embed per-task results, so an unbounded file
# grows by tens of KB per run; 100 runs of history is plenty for the
# fleet panel and CLI listing while keeping the file readable.
MAX_RECORDS = 100


class OrchestrationStore:
    """JSON-backed store of finished orchestration records, keyed by id."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_ORCHESTRATIONS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        """Load persisted records.

        On corruption, rename the bad file to a sibling
        ``.corrupted-<ts>`` before returning ``{}`` — same pattern as
        WorkerStateStore/session_pins: without the rename, the next
        save() overwrites the corrupt file with in-memory state and
        silently destroys whatever history it held.
        """
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._back_up_corrupt(exc)
            return {}
        if not isinstance(data, dict):
            self._back_up_corrupt(
                ValueError(
                    f"top-level shape is {type(data).__name__}, expected dict"
                ),
            )
            return {}
        return {
            k: v for k, v in data.items()
            if isinstance(k, str) and isinstance(v, dict)
        }

    def save(self, records: dict[str, dict[str, Any]]) -> None:
        """Persist records, keeping only the newest MAX_RECORDS.

        Recency comes from each record's ``created_at`` (ISO-8601,
        lexicographically sortable); records missing it sort oldest.
        """
        if len(records) > MAX_RECORDS:
            newest = sorted(
                records.items(),
                key=lambda kv: kv[1].get("created_at", ""),
                reverse=True,
            )[:MAX_RECORDS]
            records = dict(newest)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(records, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def _back_up_corrupt(self, exc: Exception) -> None:
        import time
        backup = self.path.with_suffix(f".corrupted-{int(time.time())}")
        try:
            self.path.rename(backup)
        except OSError:
            return
        import logging
        logging.getLogger("towel.persistence").warning(
            "orchestrations store corrupt (%s); moved to %s", exc, backup,
        )
