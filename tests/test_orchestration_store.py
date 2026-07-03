"""Tests for orchestration history persistence."""

import json

from towel.persistence.orchestrations import MAX_RECORDS, OrchestrationStore


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
        store.save(big)
        loaded = store.load()
        assert len(loaded) == MAX_RECORDS
        # The dropped 20 are the oldest (smallest created_at).
        assert f"id{MAX_RECORDS + 19}" in loaded
        assert "id0" not in loaded

    def test_non_dict_top_level_backed_up(self, tmp_path):
        path = tmp_path / "orch.json"
        path.write_text(json.dumps(["not", "a", "dict"]))
        store = OrchestrationStore(path=path)
        assert store.load() == {}
        assert list(tmp_path.glob("orch.corrupted-*"))
