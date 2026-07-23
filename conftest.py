"""Pytest bootstrap: put the backend root on sys.path so ``app`` is importable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch):
    """Every test sees a freshly-built ``Settings``, built from ``os.environ`` alone.

    Two things ``get_settings`` normally does are wrong for a test run:

    * It is ``@lru_cache``d for the whole process, so once anything has
      called it, later ``monkeypatch.setenv``/``delenv`` calls (e.g. on
      ``ANTHROPIC_API_KEY``) would silently do nothing — the cached instance
      keeps answering with whatever the environment held the first time.
    * It reads this repo's real ``.env``, which carries a real
      ``ANTHROPIC_API_KEY``. Without disabling that here, every test that
      does ``monkeypatch.delenv("ANTHROPIC_API_KEY")`` to simulate "no key
      configured" would silently get the real key back from ``.env`` instead
      (dotenv is a source independent of ``os.environ``) — the test's own
      environment would no longer be authoritative, and "no LLM configured"
      tests would start making real API calls.
    """
    from app.config import Settings, get_settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
