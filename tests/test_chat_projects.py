"""Tests for chat-project consolidation."""

from dreamland.persistence.chat_projects import MAX_RECORDS, ChatProjectStore


class TestChatProjectStore:
    def test_roundtrip(self, tmp_path):
        store = ChatProjectStore(path=tmp_path / "cp.json")
        recs = {"webchat-a": {"root": "/x/y", "count": 2,
                              "first_ts": "2026-07-05T00:00:00+00:00",
                              "last_ts": "2026-07-05T00:01:00+00:00",
                              "title": "t"}}
        store.save(recs)
        assert store.load() == recs

    def test_missing_file_empty(self, tmp_path):
        assert ChatProjectStore(path=tmp_path / "cp.json").load() == {}

    def test_corrupt_backed_up(self, tmp_path):
        p = tmp_path / "cp.json"
        p.write_text("{bad")
        store = ChatProjectStore(path=p)
        assert store.load() == {}
        assert not p.exists()
        assert list(tmp_path.glob("cp.corrupted-*"))

    def test_upsert_new_session(self, tmp_path):
        recs: dict = {}
        rec = ChatProjectStore.upsert(
            recs, "webchat-a", str(tmp_path / "proj" / "a.py"),
            "2026-07-05T00:00:00+00:00", title="My Chat",
        )
        assert rec["root"] == str(tmp_path / "proj")
        assert rec["count"] == 1
        assert rec["title"] == "My Chat"
        assert recs["webchat-a"] is rec

    def test_upsert_generalizes_root(self, tmp_path):
        recs: dict = {}
        ChatProjectStore.upsert(recs, "s", str(tmp_path / "proj" / "sub1" / "a.py"),
                                "2026-07-05T00:00:00+00:00")
        ChatProjectStore.upsert(recs, "s", str(tmp_path / "proj" / "sub2" / "b.py"),
                                "2026-07-05T00:01:00+00:00")
        # Root widens to the common parent; count accumulates.
        assert recs["s"]["root"] == str(tmp_path / "proj")
        assert recs["s"]["count"] == 2
        assert recs["s"]["last_ts"] == "2026-07-05T00:01:00+00:00"

    def test_cap_keeps_newest(self, tmp_path):
        store = ChatProjectStore(path=tmp_path / "cp.json")
        recs = {
            f"s{i}": {"root": f"/r{i}", "count": 1,
                      "first_ts": "x", "last_ts": f"2026-01-{i % 28 + 1:02d}"}
            for i in range(MAX_RECORDS + 10)
        }
        store.save(recs)
        assert len(store.load()) == MAX_RECORDS


class TestWriteHookIntegration:
    """The filesystem write actually fires the observer with the
    contextvar-bound session."""

    def test_write_hook_fires_with_active_session(self, tmp_path):
        import asyncio

        from dreamland import audit
        from dreamland.skills.builtin import filesystem as fs

        captured = []
        fs.set_write_observer(lambda sess, path: captured.append((sess, path)))
        try:
            skill = fs.FileSystemSkill()
            target = tmp_path / "proj" / "hello.py"
            token = audit.set_active_session("webchat-xyz")
            try:
                asyncio.run(skill.execute("write_file", {
                    "path": str(target), "content": "print('hi')\n",
                }))
            finally:
                audit.reset_active_session(token)
        finally:
            fs.set_write_observer(None)
        assert captured
        sess, path = captured[-1]
        assert sess == "webchat-xyz"
        assert path.endswith("hello.py")

    def test_audit_uses_active_session_fallback(self, tmp_path, monkeypatch):
        from dreamland import audit

        monkeypatch.setenv("DREAMLAND_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
        token = audit.set_active_session("webchat-fallback")
        try:
            audit.audit_tool_call("write_file", {"path": "/x"}, status="ok")
        finally:
            audit.reset_active_session(token)
        import json
        line = (tmp_path / "audit.jsonl").read_text().strip()
        assert json.loads(line)["session"] == "webchat-fallback"
