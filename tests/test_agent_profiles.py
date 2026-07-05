"""Tests for agent profiles and model switching."""

from dreamland.config import DEFAULT_AGENTS, AgentProfile, DreamlandConfig, ModelConfig


class TestAgentProfile:
    def test_default_agents_exist(self):
        assert "coder" in DEFAULT_AGENTS
        assert "researcher" in DEFAULT_AGENTS
        assert "writer" in DEFAULT_AGENTS

    def test_get_builtin_agent(self):
        config = DreamlandConfig()
        profile = config.get_agent("coder")
        assert profile is not None
        assert "coder" in profile.model.name.lower() or profile.identity != config.identity
        assert profile.description

    def test_get_nonexistent_agent(self):
        config = DreamlandConfig()
        assert config.get_agent("nonexistent") is None

    def test_user_defined_agent(self):
        config = DreamlandConfig()
        config.agents["custom"] = AgentProfile(
            model=ModelConfig(name="mlx-community/tiny-model"),
            identity="You are a custom agent.",
            description="My custom agent",
        )
        profile = config.get_agent("custom")
        assert profile is not None
        assert profile.model.name == "mlx-community/tiny-model"
        assert "custom" in profile.identity

    def test_user_agent_overrides_builtin(self):
        config = DreamlandConfig()
        config.agents["coder"] = AgentProfile(
            identity="My custom coder",
            description="Overridden coder",
        )
        profile = config.get_agent("coder")
        assert profile is not None
        assert "My custom coder" in profile.identity

    def test_list_agents(self):
        config = DreamlandConfig()
        agents = config.list_agents()
        assert "coder" in agents
        assert "researcher" in agents
        assert "writer" in agents

    def test_list_includes_user_agents(self):
        config = DreamlandConfig()
        config.agents["mybot"] = AgentProfile(description="test")
        agents = config.list_agents()
        assert "mybot" in agents


class TestResolveAgent:
    def test_resolve_none_returns_base(self):
        config = DreamlandConfig()
        model, identity = config.resolve_agent(None)
        assert model == config.model
        assert identity == config.identity

    def test_resolve_named_agent(self):
        config = DreamlandConfig()
        model, identity = config.resolve_agent("coder")
        # Coder should have a different model or identity
        assert "coder" in identity.lower() or model.name != config.model.name

    def test_resolve_default_agent(self):
        config = DreamlandConfig()
        config.default_agent = "researcher"
        model, identity = config.resolve_agent(None)
        assert "research" in identity.lower()

    def test_explicit_overrides_default(self):
        config = DreamlandConfig()
        config.default_agent = "researcher"
        model, identity = config.resolve_agent("coder")
        assert "coder" in identity.lower()

    def test_resolve_unknown_falls_back(self):
        config = DreamlandConfig()
        model, identity = config.resolve_agent("doesnt_exist")
        assert model == config.model
        assert identity == config.identity
