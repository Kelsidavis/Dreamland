"""Tests for dreamland init and the starter config."""

import pytest
import toml

from dreamland.cli.main import STARTER_CONFIG, cli
from dreamland.config import DreamlandConfig


class TestStarterConfig:
    def test_valid_toml(self):
        data = toml.loads(STARTER_CONFIG)
        assert "model" in data
        assert "gateway" in data

    def test_loads_as_dreamland_config(self):
        data = toml.loads(STARTER_CONFIG)
        config = DreamlandConfig.model_validate(data)
        assert config.model.name == "mlx-community/Llama-3.3-70B-Instruct-4bit"
        assert config.model.context_window == 8192
        assert config.model.max_tokens == 4096
        assert config.gateway.port == 18742
        assert "It doesn't exist" in config.identity

    def test_has_comments(self):
        assert "# " in STARTER_CONFIG
        assert "Quick start" in STARTER_CONFIG
        assert "dreamland chat" in STARTER_CONFIG
        assert "dreamland doctor" in STARTER_CONFIG

    def test_documents_agent_profiles(self):
        assert "agents" in STARTER_CONFIG.lower()
        assert "coder" in STARTER_CONFIG
        assert "default_agent" in STARTER_CONFIG

    def test_documents_skills_dirs(self):
        assert "skills_dirs" in STARTER_CONFIG
        assert "~/.dreamland/skills" in STARTER_CONFIG

    def test_context_window_gt_max_tokens(self):
        data = toml.loads(STARTER_CONFIG)
        assert data["model"]["context_window"] > data["model"]["max_tokens"]


class TestInitCommand:
    def test_creates_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dreamland.cli.main.DREAMLAND_HOME", tmp_path)
        monkeypatch.setattr("dreamland.config.DREAMLAND_HOME", tmp_path)

        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        config_path = tmp_path / "config.toml"
        assert config_path.exists()

        # Verify it's valid
        data = toml.loads(config_path.read_text())
        config = DreamlandConfig.model_validate(data)
        assert config.gateway.port == 18742

    def test_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dreamland.cli.main.DREAMLAND_HOME", tmp_path)
        monkeypatch.setattr("dreamland.config.DREAMLAND_HOME", tmp_path)

        from click.testing import CliRunner

        runner = CliRunner()
        runner.invoke(cli, ["init"])

        assert (tmp_path / "skills").is_dir()
        assert (tmp_path / "memory").is_dir()
        assert (tmp_path / "conversations").is_dir()

    def test_does_not_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dreamland.cli.main.DREAMLAND_HOME", tmp_path)
        config_path = tmp_path / "config.toml"
        config_path.write_text("existing = true")

        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["init"])

        assert "already exists" in result.output
        assert config_path.read_text() == "existing = true"

    def test_output_shows_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dreamland.cli.main.DREAMLAND_HOME", tmp_path)
        monkeypatch.setattr("dreamland.config.DREAMLAND_HOME", tmp_path)

        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["init"])

        assert "Config" in result.output
        assert "Skills" in result.output
        assert "Memory" in result.output
        assert "dreamland doctor" in result.output


class TestCLICommands:
    """Verify all CLI commands are registered and show help."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "bench",
            "commit",
            "config",
            "dashboard",
            "doc",
            "doctor",
            "explain",
            "fix",
            "gc",
            "history",
            "log",
            "marketplace",
            "plugins",
            "refactor",
            "review",
            "search",
            "show",
            "skills",
            "status",
            "summarize",
            "templates",
            "test-gen",
            "version",
            "watch",
            "webhook",
        ],
    )
    def test_command_help(self, cmd):
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0
        assert cmd in result.output.lower() or "usage" in result.output.lower()
