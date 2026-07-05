"""Tests for orchestration history persistence."""

import json

from dreamland.persistence.orchestrations import MAX_RECORDS, OrchestrationStore


class TestOrchestrationStore:
    def test_roundtrip(self, tmp_path):
        store = OrchestrationStore(path=tmp_path / "orch.json")
        records = {
            "abc": {"goal": "g", "state": "completed", "created_at": "2026-07-03T00:00:00+00:00"},
        }
        store.save(records)
        assert store.load() == records

    def test_missing_file_loads_empty(self, tmp_path):
        store = OrchestrationStore(path=tmp_path / "orch.json")
        assert store.load() == {}

    def test_corrupt_file_backed_up(self, tmp_path):
        path = tmp_path / "orch.json"
        path.write_text("{not json")
        store = OrchestrationStore(path=path)
        assert store.load() == {}
        assert not path.exists()
        assert list(tmp_path.glob("orch.corrupted-*"))

    def test_cap_keeps_newest(self, tmp_path):
        store = OrchestrationStore(path=tmp_path / "orch.json")
        big = {
            f"id{i}": {"goal": "g", "state": "completed",
                       "created_at": f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00"}
            for i in range(MAX_RECORDS + 20)
        }
        evicted = store.save(big)
        loaded = store.load()
        assert len(loaded) == MAX_RECORDS
        # The dropped 20 are the oldest (smallest created_at) and are
        # returned to the caller for resource cleanup…
        assert f"id{MAX_RECORDS + 19}" in loaded
        assert "id0" not in loaded
        assert len(evicted) == 20
        assert "id0" in evicted
        # …and removed from the caller's dict so memory matches disk.
        assert "id0" not in big
        assert len(big) == MAX_RECORDS

    def test_save_under_cap_evicts_nothing(self, tmp_path):
        store = OrchestrationStore(path=tmp_path / "orch.json")
        records = {"a": {"goal": "g", "state": "completed",
                         "created_at": "2026-07-03T00:00:00+00:00"}}
        assert store.save(records) == {}

    def test_non_dict_top_level_backed_up(self, tmp_path):
        path = tmp_path / "orch.json"
        path.write_text(json.dumps(["not", "a", "dict"]))
        store = OrchestrationStore(path=path)
        assert store.load() == {}
        assert list(tmp_path.glob("orch.corrupted-*"))
