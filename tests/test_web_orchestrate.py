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


class TestResumeAcrossMachines:
    """Opening the fleet panel must re-attach to server-side work —
    an orchestration started on another machine (or before a reload)
    resumes showing live progress instead of a blank panel."""

    def test_attach_functions_exist(self):
        assert "orchAttach" in HTML
        assert "orchAutoAttach" in HTML

    def test_panel_open_auto_attaches(self):
        assert "orchHistoryRefresh().then(orchAutoAttach)" in HTML

    def test_running_run_preferred(self):
        assert "r.state === 'running'" in HTML

    def test_history_select_loads_run(self):
        # Selecting a past run attaches to it (status + tasks), not
        # just the file browser.
        assert "orchStop(); orchAttach(id);" in HTML


class TestChatSessionResume:
    """Chat transcripts must restore on page load and carry across
    machines: the server owns conversations, localStorage only recalls
    which one THIS browser was in."""

    def test_resume_session_exists(self):
        assert "resumeSession" in HTML
        assert "transcriptRestored" in HTML

    def test_fresh_browser_adopts_latest_server_conversation(self):
        assert "allConversations[0].id" in HTML

    def test_no_persistent_loaded_guard(self):
        # The old localStorage guard skipped transcript restore on
        # every reload after the first — users saw a welcome screen
        # over their existing conversation.
        assert "towel-session-loaded" not in HTML

    def test_reconnect_does_not_reclobber_transcript(self):
        assert "Reconnects skip this" in HTML


class TestArchiveDownload:
    def test_download_button_exists(self):
        assert "orch-download-btn" in HTML

    def test_uses_archive_endpoint(self):
        assert "/archive" in HTML


class TestSeedFileWidget:
    def test_widget_elements_exist(self):
        for element_id in (
            "orch-addfiles-btn",
            "orch-file-input",
            "orch-seed-list",
        ):
            assert element_id in HTML, f"missing #{element_id}"

    def test_reads_files_client_side(self):
        assert "orchAddSeedFiles" in HTML
        assert "f.text()" in HTML

    def test_caps_mirror_server(self):
        assert "32 files" in HTML or "32) {" in HTML
        assert "2 * 1024 * 1024" in HTML

    def test_binary_rejected(self):
        assert "\\u0000" in HTML

    def test_seeds_sent_in_body(self):
        assert "body.files = orchSeeds" in HTML

    def test_no_raw_nul_bytes(self):
        from pathlib import Path
        raw = (
            Path(__file__).parent.parent / "src" / "towel" / "web" / "index.html"
        ).read_bytes()
        assert b"\x00" not in raw


class TestGitHistoryUI:
    def test_history_elements_exist(self):
        assert "orch-git-btn" in HTML
        assert "orchGitLog" in HTML
        assert "orchGitDiff" in HTML

    def test_uses_git_endpoints(self):
        assert "/git/log" in HTML
        assert "/git/diff/" in HTML

    def test_diff_lines_escaped(self):
        # Diff content is model-generated code rendered via innerHTML —
        # every line must pass through the escaper.
        assert "const esc = orchEsc(line);" in HTML
