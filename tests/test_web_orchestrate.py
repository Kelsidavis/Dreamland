"""Grep-tests pinning the fleet panel's Orchestrate section — the web
surface for the background orchestration API. Mirrors the convention in
test_cancel.py: assert the load-bearing ids/functions exist in
index.html so a refactor can't silently drop the feature."""

from pathlib import Path

HTML = (
    Path(__file__).parent.parent / "src" / "towel" / "web" / "index.html"
).read_text()


class TestOrchestratePanel:
    def test_section_elements_exist(self):
        for element_id in (
            "orch-goal",
            "orch-workspace",
            "orch-verify",
            "orch-repair",
            "orch-start-btn",
            "orch-cancel-btn",
            "orch-status",
            "orch-tasks",
        ):
            assert element_id in HTML, f"missing #{element_id}"

    def test_uses_background_api(self):
        assert "background: true" in HTML
        assert "/api/orchestrate/" in HTML

    def test_buttons_wired(self):
        assert "orchStart" in HTML
        assert "orchCancel" in HTML
        assert "orchPoll" in HTML

    def test_output_escaped(self):
        # Task results/run_output are model-generated text rendered
        # into innerHTML — they must pass through the escaper.
        assert "orchEsc" in HTML


class TestFileExplorer:
    def test_explorer_elements_exist(self):
        for element_id in (
            "orch-history-select",
            "orch-files-btn",
            "orch-files-list",
            "orch-file-view",
            "orch-file-head",
        ):
            assert element_id in HTML, f"missing #{element_id}"

    def test_uses_files_api(self):
        assert "/files" in HTML
        assert "orchFilesLoad" in HTML
        assert "orchFileView" in HTML

    def test_file_paths_escaped_and_encoded(self):
        # Paths come from disk and land in innerHTML + URLs.
        assert "encodeURIComponent" in HTML
