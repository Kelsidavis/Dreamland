"""Shared pytest fixtures and safety guards for the Dreamland test suite."""

from __future__ import annotations

# ── Data-home isolation ──────────────────────────────────────────────
# MUST run before importing anything from ``dreamland``: config resolves
# ``DREAMLAND_HOME`` once at import time, and every ``DEFAULT_*_PATH``
# (conversations, orchestrations, session pins, worker state, chat
# projects, audit log) is frozen from it right then. A fixture that
# monkeypatches the env var later is therefore too late — the defaults
# are already bound to the developer's real ~/.dreamland / ~/.towel.
#
# Without this, any store a test constructs *without* an explicit tmp
# path reads and writes real user data. That is not hypothetical: the
# chat-project write-observer once generalized a project root across
# runs straight into the live chat_projects.json. Point the home at a
# throwaway dir so a stray default-path store can never touch real data.
# ``setdefault`` respects an explicitly-exported home (deliberate runs).
import atexit
import os
import shutil
import tempfile

_isolated_home = False
if not os.environ.get("DREAMLAND_HOME") and not os.environ.get("TOWEL_HOME"):
    _test_home = tempfile.mkdtemp(prefix="dreamland-test-home-")
    os.environ["DREAMLAND_HOME"] = _test_home
    atexit.register(shutil.rmtree, _test_home, ignore_errors=True)
    _isolated_home = True

import pytest  # noqa: E402  — must follow the data-home isolation above

from dreamland.agent.runtime import AgentRuntime  # noqa: E402

# When we set the throwaway home, fail loudly at collection if it didn't
# actually take — e.g. a future import pulled ``dreamland.config`` in
# before this file ran, freezing the defaults onto real user data.
# Checked once at import, not per test (test_config deliberately reloads
# config with other homes). Skipped when the home was exported
# deliberately — that run owns whatever it points at.
if _isolated_home:
    import dreamland.config as _cfg

    assert str(_cfg.DREAMLAND_HOME) == os.environ["DREAMLAND_HOME"], (
        f"test data-home isolation failed: DREAMLAND_HOME resolved to "
        f"{_cfg.DREAMLAND_HOME}, not the throwaway {os.environ['DREAMLAND_HOME']}. "
        f"dreamland.config was imported before conftest set the env var."
    )


@pytest.fixture(autouse=True)
def _block_real_model_download(monkeypatch):
    """Never let a test trigger a real ``mlx_lm`` model download.

    The default configured model is an 80B HF repo. A test that accidentally
    reaches ``AgentRuntime._load_model_sync`` (e.g. a gateway endpoint test
    that falls back to the local MLX agent because no worker is connected)
    would call ``mlx_lm.load`` -> ``huggingface_hub.snapshot_download`` and
    hang on the network until the per-test timeout fires.

    On a dev box without ``mlx`` installed this already fails fast (ImportError)
    which is why the suite is green there; on CI runners that *do* have ``mlx``
    it would otherwise wedge. Make the load fail fast and deterministically so
    both environments behave identically. Tests that genuinely need generation
    stub ``agent.generate`` / ``agent.step`` / ``agent.stream`` directly and so
    never reach this method; tests that specifically exercise loading can
    re-patch it after this fixture runs.
    """

    def _blocked(self: AgentRuntime):  # pragma: no cover - guard, not behavior
        raise RuntimeError(
            "Real model loading is disabled under pytest (would download the "
            "configured model). Stub agent.generate/step/stream instead."
        )

    monkeypatch.setattr(AgentRuntime, "_load_model_sync", _blocked, raising=False)
