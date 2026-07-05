"""Tests for deployment generator."""

from dreamland.cli.deploy import generate_deploy


class TestDeploy:
    def test_docker(self, tmp_path):
        files = generate_deploy("docker", tmp_path)
        assert len(files) == 3
        names = {f.name for f in files}
        assert "Dockerfile" in names
        assert "docker-compose.yml" in names
        assert ".env.example" in names
        assert "dreamland" in (tmp_path / "Dockerfile").read_text()

    def test_systemd(self, tmp_path):
        files = generate_deploy("systemd", tmp_path, user="kelsi")
        assert any("dreamland.service" in str(f) for f in files)
        content = (tmp_path / "dreamland.service").read_text()
        assert "kelsi" in content

    def test_heroku(self, tmp_path):
        files = generate_deploy("heroku", tmp_path)
        assert any("Procfile" in str(f) for f in files)
        assert "dreamland serve" in (tmp_path / "Procfile").read_text()

    def test_fly(self, tmp_path):
        files = generate_deploy("fly", tmp_path)
        assert any("fly.toml" in str(f) for f in files)

    def test_all(self, tmp_path):
        files = generate_deploy("all", tmp_path)
        names = {f.name for f in files}
        assert "Dockerfile" in names
        assert "dreamland.service" in names
        assert "Procfile" in names
        assert "fly.toml" in names

    def test_cli_registered(self):
        from dreamland.cli.main import cli

        assert "deploy" in [c.name for c in cli.commands.values()]
