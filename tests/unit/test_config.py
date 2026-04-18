"""
Unit tests for Config (pydantic-settings).

V1.1: Redis, DLQ, and index_path fields have been removed.
Verifies that required fields are loaded from environment variables,
optional fields carry expected defaults, and YAML resolution works.
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
    }
    if overrides:
        base.update(overrides)
    if exclude:
        for k in exclude:
            base.pop(k, None)
    return base


def test_config_all_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """All required fields must be read from environment variables."""
    from src.config import Config

    env: Dict[str, str] = _make_env()
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg: Config = Config(_env_file=None)
    assert cfg.llm_model == "claude-sonnet-4-6"
    assert cfg.llm_api_key == "sk-test"
    assert cfg.memory_root == "/tmp/mem"


def test_config_defaults_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional fields not set in env must carry their documented default values."""
    from src.config import Config

    for k, v in _make_env().items():
        monkeypatch.setenv(k, v)

    cfg: Config = Config(_env_file=None)
    assert cfg.llm_max_tokens == 1024
    assert cfg.session_inactivity_timeout_seconds == 300
    assert cfg.tool_timeout_seconds == 10
    assert cfg.max_context_chars == 8000
    assert cfg.max_tool_calls == 10


def test_config_no_redis_or_dlq_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """V1.1 Config must not have Redis, DLQ, or index_path fields."""
    from src.config import Config

    for k, v in _make_env().items():
        monkeypatch.setenv(k, v)

    cfg: Config = Config(_env_file=None)
    assert not hasattr(cfg, "redis_host")
    assert not hasattr(cfg, "redis_port")
    assert not hasattr(cfg, "dlq_path")
    assert not hasattr(cfg, "index_path")
    assert not hasattr(cfg, "session_max_messages")


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


@pytest.mark.parametrize("missing_field", ["LLM_MODEL", "MEMORY_ROOT"])
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

    env: Dict[str, str] = _make_env(overrides={"MAX_TOOL_CALLS": "20"})
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg: Config = Config(_env_file=None)
    assert cfg.max_tool_calls == 20
