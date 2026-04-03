"""Unit tests for Config (pydantic-settings)."""

import pytest
from pydantic import ValidationError


def _make_env(overrides=None, exclude=None):
    base = {
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


def test_config_all_required_fields(monkeypatch):
    from src.config import Config

    env = _make_env()
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg = Config(_env_file=None)
    assert cfg.llm_model == "claude-sonnet-4-6"
    assert cfg.llm_api_key == "sk-test"
    assert cfg.memory_root == "/tmp/mem"
    assert cfg.index_path == "/tmp/mem/index.json"


def test_config_defaults_applied(monkeypatch):
    from src.config import Config

    for k, v in _make_env().items():
        monkeypatch.setenv(k, v)

    cfg = Config(_env_file=None)
    assert cfg.llm_max_tokens == 1024
    assert cfg.redis_host == "localhost"
    assert cfg.redis_port == 6379
    assert cfg.session_max_messages == 100
    assert cfg.dlq_max_attempts == 3


def test_config_missing_api_key_raises(monkeypatch):
    """LLM_API_KEY has no YAML default — must always come from env / .env."""
    from src.config import Config

    env = _make_env(exclude=["LLM_API_KEY"])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Config(_env_file=None)


@pytest.mark.parametrize("missing_field", ["LLM_MODEL", "MEMORY_ROOT", "INDEX_PATH"])
def test_config_yaml_provides_default_for_field(monkeypatch, missing_field):
    """Fields present in app.yaml resolve from YAML even when env var is absent."""
    from src.config import Config

    env = _make_env(exclude=[missing_field])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv(missing_field, raising=False)

    # Should NOT raise — YAML provides the default
    cfg = Config(_env_file=None)
    assert cfg is not None


def test_config_custom_optional_values(monkeypatch):
    from src.config import Config

    env = _make_env(overrides={"REDIS_PORT": "6380", "SESSION_MAX_MESSAGES": "50"})
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg = Config(_env_file=None)
    assert cfg.redis_port == 6380
    assert cfg.session_max_messages == 50
