"""
Application configuration.

Priority (highest -> lowest):
  1. Explicit init values
  2. Environment variables / .env file
  3. config/v1/app.yaml  (default values, version-controlled)

Secrets (e.g. llm_api_key) must be supplied via environment variable or .env.
They are intentionally absent from the YAML file.

The YAML uses flat keys that match Config field names exactly (see
config/v1/app.yaml). Unknown keys in the YAML are silently ignored, so
adding new sections to the file never breaks running code.
"""

import os
from typing import Dict, Tuple, Type, Union

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from src.constants import (
    DEFAULT_COMPACT_BATCH_SIZE,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONVERSATION_MESSAGES,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_STRANDS_LOG_LEVEL,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_SESSION_INACTIVITY_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL_ID,
)

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
        self._data: Dict[str, Union[str, int, float, bool]] = self._load(yaml_path)

    def _load(self, path: str) -> Dict[str, Union[str, int, float, bool]]:
        """
        Parse the YAML file at *path* and return only scalar values.

        Nested dicts are discarded; the top-level ``version`` key is stripped.
        Returns an empty dict if the file is absent.

        Args:
            path: Filesystem path to the YAML config file.

        Returns:
            Flat dict mapping field names to scalar config values.
        """
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, object] = yaml.safe_load(f) or {}
        raw.pop("version", None)
        # Keep only scalar values; nested dicts are not used in Config v1.
        return {
            k: v  # type: ignore[misc]
            for k, v in raw.items()
            if isinstance(v, (str, int, float, bool))
        }

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Union[str, int, float, bool, None], str, bool]:
        """
        Return the YAML-sourced value for a pydantic settings field.

        Required by ``PydanticBaseSettingsSource``.  The third element of
        the tuple (``bool``) indicates whether the value comes from a
        sequence; always ``False`` here since YAML values are scalars.

        Args:
            field:      Pydantic ``FieldInfo`` for the model field.
            field_name: The field's attribute name on the ``Config`` model.

        Returns:
            ``(value, field_name, False)`` where *value* is the YAML scalar
            or ``None`` if the field is not present in the YAML.
        """
        return self._data.get(field_name), field_name, False

    def __call__(self) -> Dict[str, Union[str, int, float, bool]]:
        """
        Return the full settings dict used by pydantic-settings.

        Only non-None values are emitted so that pydantic field defaults
        are not accidentally shadowed by a ``None`` from an absent YAML key.

        Returns:
            Dict of field-name -> scalar value for all present YAML keys.
        """
        # Only emit keys that are present and non-None so that pydantic
        # model defaults are not accidentally overwritten with None.
        return {k: v for k, v in self._data.items() if v is not None}


class Config(BaseSettings):
    """
    Validated application configuration.

    Sources (highest -> lowest priority):
      init args > env vars > .env file > config/v1/app.yaml > field defaults
    """

    # LLM
    llm_model: str = VLLM_MODEL_ID
    llm_api_key: str = VLLM_API_KEY
    llm_api_endpoint: str = VLLM_BASE_URL
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS

    # Memory (ReMeLight working directory)
    memory_root: str

    # Session behaviour
    session_inactivity_timeout_seconds: int = DEFAULT_SESSION_INACTIVITY_TIMEOUT
    tool_timeout_seconds: int = DEFAULT_TOOL_TIMEOUT_SECONDS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS

    # Conversation compaction
    max_conversation_messages: int = DEFAULT_MAX_CONVERSATION_MESSAGES
    compact_batch_size: int = DEFAULT_COMPACT_BATCH_SIZE

    # Logging — base directory; session subfolder is computed at runtime
    log_dir: str = DEFAULT_LOG_DIR
    log_level: str = DEFAULT_LOG_LEVEL
    strands_log_level: str = DEFAULT_STRANDS_LOG_LEVEL

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(  # type: ignore[override]
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Define the priority order of configuration sources.

        Returns sources highest-priority first:
          1. ``init_settings``   — values passed directly to ``Config()``.
          2. ``env_settings``    — environment variables.
          3. ``dotenv_settings`` — ``.env`` file.
          4. ``YamlSettingsSource`` — ``config/v1/app.yaml`` (lowest priority).

        The standard ``file_secret_settings`` source is intentionally
        excluded; secrets must come from env vars or ``.env``.

        Args:
            settings_cls:        The ``Config`` class itself.
            init_settings:       Init-argument source provided by pydantic.
            env_settings:        Environment-variable source.
            dotenv_settings:     ``.env`` file source.
            file_secret_settings: File-based secrets source (not used here).

        Returns:
            Tuple of sources in descending priority order.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls, _APP_YAML),
        )
