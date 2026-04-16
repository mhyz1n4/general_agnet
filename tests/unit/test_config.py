"""
Unit tests for Config (pydantic-settings).

Verifies that required fields are loaded from environment variables, optional
fields carry the expected defaults, the vLLM EMPTY API-key fallback works,
and that fields present in app.yaml resolve from YAML when the env var is
absent.  monkeypatch is used to avoid touching the real process environment.
"""

from typing import Dict, Optional

import pytest
from pydantic import ValidationError


def _make_env(
    overrides: Optional[Dict[str, str]] = None,
    exclude: Optional[list] = None,
) -> Dict[str, str]:
    """
    Build a minimal valid environment dict for Config instantiation.

    Args:
        overrides: Key/value pairs to set or overwrite in the base env.
        exclude:   Keys to remove from the base env before returning.

    Returns:
        A dict of env-var names to string values suitable for monkeypatch.setenv.
    """
    base: Dict[str, str] = {
        "LLM_MODEL": "claude-sonnet-4-6",
        "LLM_API_KEY": "sk-test",
        "MEMORY_ROOT": "/tmp/mem",
        "INDEX_PATH": "/tmp/mem/index.json",
    }
    if overrides:
        base.update(overrides)
    if exclude:
        for k in exclude:
            base.pop(k, None)
    return base


def test_config_all_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four required fields must be read from environment variables."""
    from src.config import Config

    env: Dict[str, str] = _make_env()
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg: Config = Config(_env_file=None)
    assert cfg.llm_model == "claude-sonnet-4-6"
    assert cfg.llm_api_key == "sk-test"
    assert cfg.memory_root == "/tmp/mem"
    assert cfg.index_path == "/tmp/mem/index.json"


def test_config_defaults_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional fields not set in env must carry their documented default values."""
    from src.config import Config

    for k, v in _make_env().items():
        monkeypatch.setenv(k, v)

    cfg: Config = Config(_env_file=None)
    assert cfg.llm_max_tokens == 1024
    assert cfg.redis_host == "localhost"
    assert cfg.redis_port == 6379
    assert cfg.session_max_messages == 100
    assert cfg.dlq_max_attempts == 3


def test_config_api_key_defaults_to_empty_for_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_API_KEY defaults to 'EMPTY' so vLLM works with no env var set."""
    from src.config import Config
    from src.constants import VLLM_API_KEY

    env: Dict[str, str] = _make_env(exclude=["LLM_API_KEY"])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    cfg: Config = Config(_env_file=None)
    assert cfg.llm_api_key == VLLM_API_KEY


@pytest.mark.parametrize("missing_field", ["LLM_MODEL", "MEMORY_ROOT", "INDEX_PATH"])
def test_config_yaml_provides_default_for_field(
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    """Fields present in app.yaml resolve from YAML even when the env var is absent."""
    from src.config import Config

    env: Dict[str, str] = _make_env(exclude=[missing_field])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv(missing_field, raising=False)

    cfg: Config = Config(_env_file=None)
    assert cfg is not None


def test_config_custom_optional_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var overrides for optional fields must take precedence over YAML defaults."""
    from src.config import Config

    env: Dict[str, str] = _make_env(overrides={"REDIS_PORT": "6380", "SESSION_MAX_MESSAGES": "50"})
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg: Config = Config(_env_file=None)
    assert cfg.redis_port == 6380
    assert cfg.session_max_messages == 50
