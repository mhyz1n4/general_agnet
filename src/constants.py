"""
Global application constants.

All magic numbers and fixed string literals used across the codebase are
declared here as named constants.  Import from this module instead of
duplicating literals inline.

Groups:
  - Memory types
  - Hashing / ID generation
  - Retrieval
  - Query intent values
  - Orchestration / session
  - Redis defaults
  - Logging defaults
  - LLM / model defaults
  - Session behaviour defaults
  - Dead-letter queue defaults
  - File paths
  - Filesystem / serialisation
  - Context formatting
"""

from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Memory types
# ---------------------------------------------------------------------------

MEMORY_TYPE_EPISODIC: str = "episodic"
MEMORY_TYPE_SEMANTIC: str = "semantic"
MEMORY_TYPE_PROCEDURAL: str = "procedural"

VALID_MEMORY_TYPES: FrozenSet[str] = frozenset(
    {MEMORY_TYPE_EPISODIC, MEMORY_TYPE_SEMANTIC, MEMORY_TYPE_PROCEDURAL}
)

# Maps memory type → storage subdirectory name.
MEMORY_TYPES_DIR_MAP: Dict[str, str] = {
    MEMORY_TYPE_EPISODIC: "conversations",
    MEMORY_TYPE_SEMANTIC: "knowledge",
    MEMORY_TYPE_PROCEDURAL: "procedures",
}

DEFAULT_MEMORY_TOPIC: str = "general"
MEMORY_DATE_FOLDER_FORMAT: str = "%Y-%m-%d"
UNKNOWN_SESSION_ID: str = "unknown_session"

# ---------------------------------------------------------------------------
# Hashing / ID generation
# ---------------------------------------------------------------------------

CONTENT_HASH_LENGTH: int = 16   # SHA-256 hex prefix length used for dedup
MEMORY_ID_HEX_LENGTH: int = 8   # UUID hex prefix for generated memory IDs
SESSION_ID_HEX_LENGTH: int = 12  # UUID hex prefix for session IDs

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_RETRIEVAL_LIMIT: int = 3        # context blocks returned per query turn
DEFAULT_RETRIEVER_SEARCH_LIMIT: int = 5  # max results from KeywordRetriever.search()
QUERY_LOG_PREVIEW_LENGTH: int = 80      # characters shown in log previews of queries
MIN_KEYWORD_LENGTH: int = 3             # _extract_keywords drops words ≤ this length

# ---------------------------------------------------------------------------
# Query intent values  (mirrors Intent enum string values for decoupled use)
# ---------------------------------------------------------------------------

INTENT_RECALL_HISTORY: str = "recall_history"
INTENT_CURRENT_SESSION: str = "current_session"
INTENT_GENERAL_TASK: str = "general_task"

# ---------------------------------------------------------------------------
# Orchestration / session
# ---------------------------------------------------------------------------

EXIT_COMMANDS: FrozenSet[str] = frozenset({"exit", "quit", "bye", "/exit", "/quit"})
CHARS_PER_TOKEN: int = 4                # rough approximation for Claude/GPT-class models
CONTEXT_BUDGET_ALERT_THRESHOLD: float = 0.9
CONSECUTIVE_FAILURES_BEFORE_GRACEFUL: int = 2

# ---------------------------------------------------------------------------
# Redis defaults
# ---------------------------------------------------------------------------

REDIS_DEFAULT_PREFIX: str = "mem:"
REDIS_DEFAULT_HOST: str = "localhost"
REDIS_DEFAULT_PORT: int = 6379
REDIS_DEFAULT_TTL: int = 3600           # seconds — 1 hour

# ---------------------------------------------------------------------------
# Logging defaults
# ---------------------------------------------------------------------------

DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_STRANDS_LOG_LEVEL: str = "WARNING"   # set to DEBUG to trace Strands internals
DEFAULT_LOG_DIR: str = "./logs"
LOG_DIR_MAX_BYTES: int = 500 * 1024 * 1024  # 500 MB
DEFAULT_TRACE_ID: str = "unset"

# Session log subfolder format: logs/session_{session_id}_{YYYY-MM-DD}/
SESSION_LOG_SUBDIR_PREFIX: str = "session_"
SESSION_LOG_DATE_FORMAT: str = "%Y-%m-%d"

# Per-component log filenames written inside the session subfolder.
# Keys are Python logger namespace prefixes; the root logger catches everything
# else and writes to DEFAULT_APP_LOG_FILENAME.
DEFAULT_APP_LOG_FILENAME: str = "app.jsonl"
DEFAULT_METRICS_FILENAME: str = "metrics.jsonl"
LOG_COMPONENT_FILES: Dict[str, str] = {
    "src.memory": "memory.jsonl",
    "src.agents": "agent.jsonl",
    "src.tools": "agent.jsonl",
}

# ---------------------------------------------------------------------------
# LLM / model defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_MAX_TOKENS: int = 1024
DEFAULT_CLAUDE_MODEL: str = "claude-3-5-sonnet-20240620"
DEFAULT_OPENAI_MODEL: str = "gpt-4o"
DEFAULT_AGENT_MODEL: str = "claude-sonnet-4-6"  # Strands agent model
SUMMARY_MAX_TOKENS: int = 256   # token budget for the post-session summary call
LLM_PING_MAX_TOKENS: int = 1    # minimal completion used to validate the API key

# ---------------------------------------------------------------------------
# Session behaviour defaults
# ---------------------------------------------------------------------------

DEFAULT_SESSION_MAX_MESSAGES: int = 100
DEFAULT_SESSION_INACTIVITY_TIMEOUT: int = 300   # seconds
DEFAULT_TOOL_TIMEOUT_SECONDS: int = 10
DEFAULT_MAX_CONTEXT_CHARS: int = 8000
DEFAULT_MAX_TOOL_CALLS: int = 10   # max tool-call retries per agent invocation

# ---------------------------------------------------------------------------
# Dead-letter queue defaults
# ---------------------------------------------------------------------------

DEFAULT_DLQ_PATH: str = "./memory/dlq.jsonl"
DEFAULT_DLQ_MAX_ATTEMPTS: int = 3
DEFAULT_DLQ_RETRY_INTERVAL: int = 60   # seconds

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DEFAULT_METRICS_PATH: str = "./logs/metrics.jsonl"  # flat fallback; runtime uses session subfolder

# ---------------------------------------------------------------------------
# Filesystem / serialisation
# ---------------------------------------------------------------------------

WRITE_TEST_FILENAME: str = ".write_test"
WRITE_TEST_CONTENT: str = "ok"
INDEX_LOCK_SUFFIX: str = ".lock"
INDEX_TMP_SUFFIX: str = ".tmp"
JSON_INDENT: int = 2

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

CONTEXT_BLOCK_DELIMITER: str = "--- Memory ---"
MEMORY_TIMESTAMP_DISPLAY_FORMAT: str = "%Y-%m-%d %H:%M UTC"  # embedded in stored content for LLM readability

# ---------------------------------------------------------------------------
# Local vLLM server (OpenAI-compatible)
# ---------------------------------------------------------------------------

VLLM_BASE_URL: str = "http://localhost:8000/v1"
VLLM_MODEL_ID: str = "/model"
# vLLM accepts any non-empty string as the API key when auth is disabled.
VLLM_API_KEY: str = "EMPTY"
