"""Grep-tests pinning the fleet panel's Orchestrate section — the web
surface for the background orchestration API. Mirrors the convention in
test_cancel.py: assert the load-bearing ids/functions exist in
index.html so a refactor can't silently drop the feature."""

from pathlib import Path

HTML = (
    Path(__file__).parent.parent / "src" / "dreamland" / "web" / "index.html"
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
            "orch-runs-list",
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
        assert "orchRunsRefresh().then(orchAutoAttach)" in HTML

    def test_running_run_preferred(self):
        assert "r.state === 'running'" in HTML

    def test_run_card_click_attaches(self):
        # Clicking a run card attaches to it (status + tasks + files).
        assert "orchSelectRun" in HTML
        assert "orchAttach(id)" in HTML


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
        assert "dreamland-session-loaded" not in HTML

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
            Path(__file__).parent.parent / "src" / "dreamland" / "web" / "index.html"
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


class TestCloneCommand:
    def test_history_shows_clone_command(self):
        assert "git clone " in HTML
        assert "clone_path" in HTML


class TestProjectContinueUI:
    def test_checkbox_exists(self):
        assert "orch-project-continue" in HTML

    def test_sends_project_when_checked(self):
        assert "body.project = orchSelectedId" in HTML


class TestDreamlandTheme:
    def test_dreamland_is_default_theme(self):
        assert 'data-theme="dreamland"' in HTML
        assert "'dreamland', 'deep-space', 'frost', 'matrix', 'solarized'" in HTML

    def test_theme_palette_defined(self):
        assert '[data-theme="dreamland"]' in HTML
        assert "--hazard" in HTML

    def test_welcome_hero_elements(self):
        for marker in ("RESTRICTED AREA", "saucer", "stars", "radar-ping"):
            assert marker in HTML, marker

    def test_favicon_present(self):
        assert 'rel="icon"' in HTML


class TestProjectsPanel:
    """The Projects panel is the first-class surface for goal-driven
    builds — composer + runs list on the left, live run detail with
    files/history on the right."""

    def test_panel_elements_exist(self):
        for element_id in (
            "projects-overlay",
            "projects-panel",
            "tb-projects",
            "projects-close-btn",
            "projects-refresh-btn",
            "orch-runs-list",
        ):
            assert element_id in HTML, f"missing #{element_id}"

    def test_orchestrate_left_fleet_panel(self):
        # The fleet panel no longer hosts the orchestrate section.
        fleet = HTML.split('id="fleet-panel"')[1].split('id="projects-overlay"')[0]
        assert "orch-goal" not in fleet
        assert "Orchestrate</h3>" not in fleet

    def test_shortcut_and_palette(self):
        assert "Ctrl+Shift+P" in HTML
        assert "openProjectsPanel()" in HTML


class TestChatPolish:
    """Chat-surface QoL: copy affordances, scroll pill, responsive
    layout with a slide-in sidebar for narrow screens."""

    def test_code_copy_button(self):
        assert "code-copy" in HTML
        assert "Copy code" in HTML

    def test_message_copy_button(self):
        assert "msg-copy" in HTML

    def test_scroll_pill(self):
        assert "scroll-bottom-btn" in HTML

    def test_responsive_breakpoint(self):
        assert "@media (max-width: 900px)" in HTML
        assert "tb-menu" in HTML
        assert "#sidebar.open" in HTML

    def test_clipboard_used(self):
        assert "navigator.clipboard.writeText" in HTML


class TestToolChips:
    """Saved transcripts render tool-call markup as collapsible chips —
    the noise folds away, the payload stays inspectable."""

    def test_chip_pipeline_exists(self):
        for marker in ("extractToolCalls", "toolChipHtml",
                       "renderTranscriptMarkdown", "tool-chip"):
            assert marker in HTML, marker

    def test_history_renderer_uses_chips(self):
        assert "renderTranscriptMarkdown(text)" in HTML

    def test_chip_content_escaped(self):
        # Tool payloads are model output rendered via innerHTML.
        assert "orchEsc(inner)" in HTML


class TestBuildNotifications:
    def test_notification_on_finish(self):
        assert "orchNotifyDone" in HTML
        assert "Notification.permission" in HTML
        assert "document.hidden" in HTML


class TestChatProjectsUI:
    def test_run_card_handles_chat_source(self):
        assert "r.source === 'chat'" in HTML
        assert "💬 chat" in HTML

    def test_detail_opens_files_for_chat(self):
        assert "data.source === 'chat' || data.state === 'files'" in HTML


class TestContinueButton:
    def test_continue_button_present(self):
        assert 'id="orch-continue-btn"' in HTML

    def test_continue_handler_posts_to_endpoint(self):
        assert "function orchContinue" in HTML
        assert "/continue" in HTML

    def test_continue_shown_only_when_goal_incomplete(self):
        # Button is gated on a finished run with an incomplete goal audit.
        assert "data.goal_achieved === false" in HTML
        assert "orchContinue" in HTML  # wired to a click listener


class TestCaffeinate:
    def test_checkbox_present(self):
        assert 'id="orch-caffeinate"' in HTML
        assert "caffeinate" in HTML

    def test_auto_continue_logic_present(self):
        assert "function orchMaybeCaffeinate" in HTML
        # Fires another repair round automatically on an incomplete run.
        assert "orchContinue()" in HTML

    def test_bounded_by_round_cap(self):
        assert "ORCH_CAFFEINE_MAX_ROUNDS" in HTML

    def test_stops_on_no_progress(self):
        # Same audit gaps two rounds running → stop, don't spin the fleet.
        assert "orchCaffeineLastFeedback" in HTML
        assert "no progress" in HTML

    def test_reset_on_new_run(self):
        assert "orchResetCaffeine" in HTML


class TestProjectsPanelUX:
    """The projects area groups individual projects to click between and
    gives richer at-a-glance feedback (state pill, progress, verdict)."""

    def test_projects_grouped_builds_and_chat(self):
        assert "class=\"proj-group\"" in HTML
        assert "builds " in HTML
        assert "chat projects" in HTML

    def test_project_card_helper(self):
        assert "function orchRunCard" in HTML

    def test_progress_bar_and_state_dot(self):
        assert "rc-bar" in HTML          # per-card progress bar
        assert "rc-dot" in HTML          # status dot
        assert "rc-pulse" in HTML        # live pulse animation

    def test_relative_time_helper(self):
        assert "function orchAgo" in HTML
        assert "ago" in HTML

    def test_goal_verdict_chip(self):
        assert "function orchGoalChip" in HTML
        assert "✓ goal" in HTML and "✗ goal" in HTML

    def test_detail_status_header_is_rich(self):
        # State pill + progress bar in the detail header, not a plain line.
        assert "od-pill" in HTML
        assert "od-bar" in HTML

    def test_loading_and_empty_feedback(self):
        assert "loading projects…" in HTML
        assert "No projects yet" in HTML
        assert "loading project…" in HTML  # per-project select feedback

    def test_sort_control_present(self):
        assert 'id="orch-sort"' in HTML
        assert 'value="newest"' in HTML
        assert 'value="running"' in HTML

    def test_sort_logic_present(self):
        assert "function orchSortRuns" in HTML
        assert "orchSortMode" in HTML
