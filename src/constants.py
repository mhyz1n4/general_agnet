"""
Global application constants.

All magic numbers and fixed string literals used across the codebase are
declared here as named constants.  Import from this module instead of
duplicating literals inline.

Groups:
  - Memory types
  - ID generation
  - Retrieval
  - Orchestration / session
  - Logging defaults
  - LLM / model defaults
  - Session behaviour defaults
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

DEFAULT_MEMORY_TOPIC: str = "general"
MEMORY_DATE_FOLDER_FORMAT: str = "%Y-%m-%d"

# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

MEMORY_ID_HEX_LENGTH: int = 8   # UUID hex prefix for generated memory IDs
SESSION_ID_HEX_LENGTH: int = 12  # UUID hex prefix for session IDs

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_RETRIEVAL_LIMIT: int = 3        # context blocks returned per query turn
QUERY_LOG_PREVIEW_LENGTH: int = 80      # characters shown in log previews of queries

# ---------------------------------------------------------------------------
# Orchestration / session
# ---------------------------------------------------------------------------

EXIT_COMMANDS: FrozenSet[str] = frozenset({"exit", "quit", "bye", "/exit", "/quit"})
CHARS_PER_TOKEN: int = 4                # rough approximation for Claude/GPT-class models
CONTEXT_BUDGET_ALERT_THRESHOLD: float = 0.9
CONSECUTIVE_FAILURES_BEFORE_GRACEFUL: int = 2

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
LLM_PING_MAX_TOKENS: int = 1    # minimal completion used to validate the API key

# ---------------------------------------------------------------------------
# Session behaviour defaults
# ---------------------------------------------------------------------------

DEFAULT_SESSION_INACTIVITY_TIMEOUT: int = 300   # seconds
# Wall-clock cap on a single agent.__call__: covers the full LLM ↔ tool loop
# for one user turn, not a single tool call.  Per-tool timeouts live inside
# the tool modules themselves (e.g. _HTTP_TIMEOUT_SECONDS in web_search.py).
DEFAULT_AGENT_TURN_TIMEOUT_SECONDS: int = 60
DEFAULT_MAX_CONTEXT_CHARS: int = 8000
DEFAULT_MAX_TOOL_CALLS: int = 10   # max tool-call retries per agent invocation
DEFAULT_MAX_CONVERSATION_MESSAGES: int = 40   # triggers compaction (matches Strands default)
DEFAULT_COMPACT_BATCH_SIZE: int = 20          # messages per compaction batch
# Cadence at which the orchestrator appends a metrics snapshot to
# ``metrics.jsonl`` between session-end flushes.
METRICS_FLUSH_INTERVAL_TURNS: int = 5

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DEFAULT_METRICS_PATH: str = "./logs/metrics.jsonl"  # flat fallback; runtime uses session subfolder

# ---------------------------------------------------------------------------
# Filesystem / serialisation
# ---------------------------------------------------------------------------

WRITE_TEST_FILENAME: str = ".write_test"
WRITE_TEST_CONTENT: str = "ok"

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

CONTEXT_BLOCK_DELIMITER: str = "--- Memory ---"
MEMORY_TIMESTAMP_DISPLAY_FORMAT: str = "%Y-%m-%d %H:%M UTC"  # embedded in stored content for LLM readability

# Truncation cap for compacted episodic write-back: keep entries small enough
# to fit many in a single retrieval window without dominating context.
TURN_SUMMARY_MAX_CHARS: int = 280

# ---------------------------------------------------------------------------
# Raw-turn archive
# Full-fidelity user/assistant transcript appended once per turn.  Lives next
# to ``memory_root`` so it ships with backups but is excluded from search.
# ---------------------------------------------------------------------------

ARCHIVE_DIR_NAME: str = "archive"
ARCHIVE_SESSION_FILENAME_FORMAT: str = "session_{session_id}.jsonl"

# ---------------------------------------------------------------------------
# Local vLLM server (OpenAI-compatible)
# ---------------------------------------------------------------------------

VLLM_BASE_URL: str = "http://localhost:8000/v1"
VLLM_MODEL_ID: str = "/model"
# vLLM accepts any non-empty string as the API key when auth is disabled.
VLLM_API_KEY: str = "EMPTY"

# ---------------------------------------------------------------------------
# External tools (V1.1 M2)
# ---------------------------------------------------------------------------

DEFAULT_TAVILY_ENDPOINT: str = "https://api.tavily.com/search"
DEFAULT_TAVILY_MAX_RESULTS: int = 5

# ---------------------------------------------------------------------------
# Per-tool budget defaults
# Per-turn caps protecting against runaway loops.  The global
# DEFAULT_MAX_TOOL_CALLS still bounds the total call count per turn.
# ---------------------------------------------------------------------------

DEFAULT_TOOL_BUDGET_WEB_SEARCH: int = 10
DEFAULT_TOOL_BUDGET_RUN_PYTHON: int = 2
DEFAULT_TOOL_BUDGET_DELEGATE_TO_RESEARCH: int = 1

# ---------------------------------------------------------------------------
# Research sub-agent defaults
# ---------------------------------------------------------------------------

# Per-delegation cap on tool calls inside the research sub-agent.  Keeps a
# stuck sub-agent from consuming the parent's per-turn budget.
DEFAULT_SUB_AGENT_MAX_TOOL_CALLS: int = 3

# Wall-clock timeout (seconds) applied to each ``delegate_to_research``
# invocation.  Enforced by the module-level executor in
# ``src/agents/research.py``.
DEFAULT_SUB_AGENT_TIMEOUT_SECONDS: int = 30
