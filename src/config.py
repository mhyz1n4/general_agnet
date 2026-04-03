"""
Application configuration.

Priority (highest → lowest):
  1. Explicit init values
  2. Environment variables / .env file
  3. config/v1/app.yaml  (default values, version-controlled)

Secrets (e.g. llm_api_key) must be supplied via environment variable or .env.
They are intentionally absent from the YAML file.

The YAML is structured hierarchically for readability.  The YamlSettingsSource
flattens it into the same flat namespace used by the pydantic model fields:

    llm.model        → llm_model
    redis.host       → redis_host
    session.tool_timeout_seconds → session_tool_timeout_seconds  (*)

(*) Collision guard: if a flattened key doesn't match any model field it is
    silently ignored, so adding new YAML sections never breaks existing code.
"""

import os
from typing import Any, Dict, Tuple, Type

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Resolve config directory relative to this file so the app works regardless
# of the working directory.
_CONFIG_V1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "v1")
_APP_YAML = os.path.join(_CONFIG_V1, "app.yaml")


class YamlSettingsSource(PydanticBaseSettingsSource):
    """
    Custom pydantic-settings source that loads defaults from a YAML file.

    The YAML uses flat keys that match Config field names exactly (see
    config/v1/app.yaml). The top-level ``version`` key is stripped.
    Unknown keys in the YAML are silently ignored, so adding new sections
    to the file never breaks running code.
    """

    def __init__(self, settings_cls: Type[BaseSettings], yaml_path: str) -> None:
        super().__init__(settings_cls)
        self._data: Dict[str, Any] = self._load(yaml_path)

    def _load(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        raw.pop("version", None)
        # Keep only scalar values; nested dicts are not used in Config v1.
        return {k: v for k, v in raw.items() if not isinstance(v, dict)}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> Dict[str, Any]:
        # Only emit keys that are present and non-None so that pydantic
        # model defaults are not accidentally overwritten with None.
        return {k: v for k, v in self._data.items() if v is not None}


class Config(BaseSettings):
    """
    Validated application configuration.

    Sources (highest → lowest priority):
      init args > env vars > .env file > config/v1/app.yaml > field defaults
    """

    # LLM — api_key has no YAML default (secret)
    llm_model: str
    llm_api_key: str
    llm_max_tokens: int = 1024

    # Memory (file system)
    memory_root: str
    index_path: str

    # Redis (short-term session memory)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 3600
    session_max_messages: int = 100

    # Session behaviour
    session_inactivity_timeout_seconds: int = 300
    tool_timeout_seconds: int = 10
    max_context_chars: int = 8000

    # Logging (mirrored from logging.yaml for components that need them)
    log_path: str = "./logs/app.jsonl"
    log_level: str = "INFO"

    # Dead-letter queue
    dlq_path: str = "./memory/dlq.jsonl"
    dlq_max_attempts: int = 3
    dlq_retry_interval_seconds: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls, _APP_YAML),
        )
