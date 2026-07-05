"""Tests for Telegram channel."""


class TestTelegramChannel:
    def test_instantiation(self):
        from dreamland.channels.telegram import TelegramChannel

        ch = TelegramChannel(token="fake:token")
        assert ch.name == "telegram"
        assert ch.token == "fake:token"

    def test_cli_registered(self):
        from dreamland.cli.main import cli

        assert "telegram" in [c.name for c in cli.commands.values()]

    def test_help(self):
        from click.testing import CliRunner

        from dreamland.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["telegram", "--help"])
        assert result.exit_code == 0
        assert "telegram" in result.output.lower()
        assert "token" in result.output.lower()
