"""Tests for `app.config.Settings` — specifically `anthropic_api_key`.

Every Anthropic call site used to read `os.environ.get("ANTHROPIC_API_KEY")`
directly, bypassing `Settings` (and therefore `.env`) entirely. These tests
guard the fix: the key must load from `.env` like every other setting, with
normal precedence — an explicit environment variable still wins.

The global `_reset_settings_cache` fixture (conftest.py) disables `.env`
loading for the rest of the suite, so tests that simulate "no key configured"
via `monkeypatch.delenv` are not at the mercy of this repo's real `.env`. The
tests below re-enable it against a throwaway file of their own, so they never
touch that real file or its real key.
"""

from __future__ import annotations

import asyncio
import inspect

from app.config import Settings, get_settings


def test_anthropic_api_key_defaults_to_empty_string(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert get_settings().anthropic_api_key == ""


def test_anthropic_api_key_is_read_from_a_dot_env_file(tmp_path, monkeypatch):
    """A real key in .env must be picked up — the bug this fix addresses."""
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", str(env_file))

    assert Settings().anthropic_api_key == "sk-ant-from-dotenv"


def test_explicit_environment_variable_wins_over_dot_env_file(tmp_path, monkeypatch):
    """Normal precedence: an explicit env var still beats .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
    monkeypatch.setitem(Settings.model_config, "env_file", str(env_file))

    assert Settings().anthropic_api_key == "sk-ant-from-environment"


def test_no_call_site_reads_os_environ_for_the_key_directly():
    """Static regression guard for the reported bug.

    Six call sites used to read `os.environ.get("ANTHROPIC_API_KEY")`
    directly, bypassing `Settings` (and therefore `.env`) entirely. Assert the
    source of each no longer does, rather than only checking behaviour that a
    reintroduced `os.environ` read could still accidentally satisfy.
    """
    import app.orchestrator.aggregator as aggregator
    import app.orchestrator.router_llm as router_llm
    import app.services.chat as chat
    import app.services.rca as rca

    for module in (aggregator, router_llm, chat, rca):
        source = inspect.getsource(module)
        assert 'os.environ.get("ANTHROPIC_API_KEY")' not in source, module.__name__
        assert "get_settings().anthropic_api_key" in source, module.__name__


def test_anthropic_router_call_uses_the_key_from_dot_env_not_os_environ(
    tmp_path, monkeypatch
):
    """End-to-end proof for one representative call site.

    A key present only via `.env` (never in `os.environ`) must still reach
    the Anthropic client — the exact case that broke before this fix, since
    `os.environ.get(...)` would have returned ``None`` here.
    """
    import anthropic

    import app.orchestrator.router_llm as router_llm

    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", str(env_file))

    seen_keys = []

    class _FakeTextBlock:
        type = "text"
        text = "ROUTED"

    class _FakeResponse:
        content = [_FakeTextBlock()]

    class _FakeMessages:
        def create(self, **_kwargs):
            return _FakeResponse()

    class _FakeAnthropic:
        def __init__(self, api_key):
            seen_keys.append(api_key)
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    result = asyncio.run(router_llm.anthropic_router_call("route this"))

    assert seen_keys == ["sk-ant-from-dotenv"]
    assert result == "ROUTED"
