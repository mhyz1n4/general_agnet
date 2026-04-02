"""
Application configuration loaded from environment variables / .env file.

All components receive Config as a constructor argument — no component reads
.env directly. This keeps components testable and decoupled from the environment.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # LLM
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

    # Logging
    log_path: str = "./logs/app.jsonl"
    log_level: str = "INFO"

    # Dead-letter queue
    dlq_path: str = "./memory/dlq.jsonl"
    dlq_max_attempts: int = 3
    dlq_retry_interval_seconds: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
