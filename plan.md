# Plan: V1 Implementation — Memory File System, Orchestrator, Strands Tools

## Context
Existing: LLM clients, dual-backend memory (FileStorage/JSONIndexer/KeywordRetriever + RedisStorage),
query routing (RegexClassifier, TemporalExtractor), MemoryManager, and tests.

This plan builds the remaining v1 components:
- **Memory file system**: typed `.md` directory structure + content-hash deduplication + DLQ for failed saves
- **Strands Agent tools**: `memorize` tool via `@tool` + closure; all hooks/tools have structured logging
- **Full orchestrator**: config, logging, hooks, prompt templates, chat loop, entry point
- **Integration tests**: real Redis + real filesystem; LLM interactions mocked

---

## What Already Exists (do not rebuild)
| Component | File | Status |
|---|---|---|
| LLM clients | `src/client/` | Complete |
| Storage (JSON flat) | `src/memory/file_system/storage.py` | Kept for existing tests |
| Indexer | `src/memory/file_system/indexer.py` | Reused unchanged |
| Retriever | `src/memory/file_system/retriever.py` | Reused unchanged |
| Redis storage | `src/memory/redis/storage.py` | Complete |
| Memory manager | `src/memory/manager.py` | Extend only |
| Query routing | `src/query/` | Complete |
| Unit tests | `tests/test_memory_system.py`, `tests/test_redis_storage.py` | Must stay green |

---

## Strands Dynamic System Prompt Decision

Strands `Agent.system_prompt` is fixed at init time.
**Solution**: inject memory context into the **user message** each turn (not the system prompt).

- `system_prompt.j2` → rendered **once at startup** (static: persona + guidelines + examples)
- `context_prefix.j2` → rendered **per turn** with `{{ memory_context }}`; prepended to user input
- Per-turn call: `agent(f"{rendered_prefix}\n\n{user_input}")`

---

## New Files — Dependency-Ordered

### Phase 1 — Foundation (no internal deps, write in parallel)
| File | Purpose |
|---|---|
| `requirements.txt` (update) | Add `pydantic-settings>=2.0.0`, `jinja2>=3.1.0`, `strands-agents[anthropic]>=0.1.0`, `rich>=13.0.0` |
| `.env.example` | Document all env vars |
| `src/config.py` | `Config(BaseSettings)` — single `.env` source, validated at startup |
| `src/logging_config.py` | JSON structured logging + `contextvars.ContextVar` trace_id |
| `src/hooks/__init__.py` | Empty package |
| `src/hooks/base.py` | `HookResult` dataclass + `BaseHook(ABC)` |
| `src/tools/__init__.py` | Empty package |
| `src/prompts/__init__.py` | Empty package |

### Phase 2 — Memory File System (depends on Phase 1)
| File | Purpose |
|---|---|
| `src/memory/file_system/typed_storage.py` | `TypedMarkdownStorage(BaseStorage)` — `.md` files in typed dirs, manifest for O(1) lookup |
| `src/memory/dlq.py` | `DeadLetterQueue` — persistent JSONL file, retry logic for failed saves |
| `src/memory/manager.py` (extend) | Add `get_context_with_keys()` + content-hash dedup + DLQ integration |

### Phase 3 — Prompts & Tools (depends on Phase 1 + 2)
| File | Purpose |
|---|---|
| `src/prompts/loader.py` | `render_prompt(name, **ctx) -> str` via `jinja2.FileSystemLoader` |
| `src/prompts/system_prompt.j2` | Static system prompt (persona + guidelines + examples) |
| `src/prompts/context_prefix.j2` | Per-turn memory context prefix template |
| `src/tools/memorize.py` | `create_memorize_tool(memory_manager)` via Strands `@tool` + closure + logging |

### Phase 4 — Hooks (depends on Phase 1–3)
| File | Purpose |
|---|---|
| `src/hooks/pre_session.py` | FS check (abort) → Redis ping (degrade) → LLM ping (abort) + logging |
| `src/hooks/post_session.py` | Daemon thread: flush Redis→FS, LLM summary, metrics.jsonl + logging |
| `src/hooks/pre_mem_fetch.py` | Query normalisation + logging |
| `src/hooks/post_mem_fetch.py` | Truncate context, save retrieval_event + logging |

### Phase 5 — Orchestrator + Entry Point (depends on all above)
| File | Purpose |
|---|---|
| `src/orchestrator.py` | `SessionMetrics` + `Orchestrator` + `run()` loop |
| `main.py` | Wire components, PreSessionHook, degrade handling, start loop |

### Phase 6 — Tests + Infra
| File | Purpose |
|---|---|
| `tests/conftest.py` | Session-scoped fixtures: start/stop Redis, create/clean temp dirs |
| `tests/integration/test_memory_integration.py` | Real Redis + real FS; LLM mocked |
| `tests/integration/test_hooks_integration.py` | Hooks against real Redis + FS; LLM mocked |
| `tests/integration/test_orchestrator_integration.py` | Full turn loop; LLM mocked |
| `tests/unit/test_config.py` | Config validation |
| `tests/unit/test_typed_storage.py` | TypedMarkdownStorage unit tests |
| `tests/unit/test_manager_extensions.py` | `get_context_with_keys` + dedup + DLQ |
| `tests/unit/test_tools.py` | Memorize tool with mock memory_manager |
| `tests/unit/test_hooks.py` | Hooks with all deps mocked |
| `tests/unit/test_dlq.py` | DLQ enqueue/retry/persistence |
| `tests/golden/` | 20–30 JSON golden examples |
| `Makefile` (update) | Add `test-unit`, `test-integration`, `test-infra-start`, `test-infra-stop` |

---

## Key Implementation Details

### `src/config.py`
```python
class Config(BaseSettings):
    llm_model: str
    llm_api_key: str
    llm_max_tokens: int = 1024

    memory_root: str
    index_path: str

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 3600
    session_max_messages: int = 100

    session_inactivity_timeout_seconds: int = 300
    tool_timeout_seconds: int = 10
    max_context_chars: int = 8000

    log_path: str = "./logs/app.jsonl"
    log_level: str = "INFO"

    # DLQ
    dlq_path: str = "./memory/dlq.jsonl"
    dlq_max_attempts: int = 3
    dlq_retry_interval_seconds: int = 60

    model_config = SettingsConfigDict(env_file=".env")
```

---

### `src/logging_config.py`
- Custom `logging.Formatter` emitting one JSON line per record:
  `{timestamp (ISO8601), level, trace_id, component, event, data}`
- `trace_id` from `contextvars.ContextVar[str]` (default `"unset"`)
- Expose: `set_trace_id(value)`, `get_logger(component: str) -> logging.Logger`
- File handler → `config.log_path` (creates `logs/` if missing)
- `StreamHandler(stderr)` at `WARNING` for operator alerts

**Every hook and tool must call `get_logger(__name__)` at module level and log:**
- Entry to `run()` / `execute()` at `DEBUG`
- Completion with outcome at `DEBUG`
- Errors at `ERROR` with full `data` dict including the exception message

---

### `src/memory/file_system/typed_storage.py` — `TypedMarkdownStorage`

**Directory structure**:
```
{memory_root}/
  conversations/YYYY-MM/{key}.md      # episodic
  knowledge/{topic_or_general}/{key}.md    # semantic
  procedures/{topic_or_general}/{key}.md   # procedural
  manifest.json                       # key → relative path
```

**`.md` file format**:
```markdown
# {key}

{content}
```

```python
class TypedMarkdownStorage(BaseStorage):
    MEMORY_TYPES = {
        "episodic": "conversations",
        "semantic": "knowledge",
        "procedural": "procedures",
    }

    def __init__(self, memory_root: str):
        self.memory_root = memory_root
        self._lock = threading.Lock()
        os.makedirs(memory_root, exist_ok=True)

    def _resolve_path(self, key: str, data: Dict) -> str:
        """Derive relative path from data["metadata"]["type"/"topic"/"timestamp"]."""
        # episodic  → conversations/YYYY-MM/{key}.md  (from timestamp or today)
        # semantic  → knowledge/{topic or "general"}/{key}.md
        # procedural→ procedures/{topic or "general"}/{key}.md

    def _load_manifest(self) -> Dict[str, str]: ...    # reads manifest.json or {}
    def _save_manifest(self, manifest: Dict) -> None:  # atomic write via temp + os.replace

    def save(self, key: str, data: Any) -> None:
        """Write .md file; update manifest atomically; create subdirs as needed."""

    def load(self, key: str) -> Optional[Any]:
        """Lookup manifest → read .md → return {"id": key, "content": text, "metadata": {}}."""
        # Returns None if key not in manifest (no filesystem scan)
```

---

### `src/memory/dlq.py` — `DeadLetterQueue`

**Purpose**: when `MemoryManager.save_message()` fails, the message is persisted to a
JSONL file and retried on a background timer. After `dlq_max_attempts` failures the
entry is marked `dead` and left for manual inspection.

```python
@dataclass
class DLQEntry:
    message_id: str
    content: str
    metadata: Dict[str, Any]
    error: str
    attempts: int = 0
    last_attempt_ts: str = ""
    dead: bool = False

class DeadLetterQueue:
    def __init__(self, dlq_path: str, max_attempts: int = 3): ...

    def enqueue(self, message_id: str, content: str, metadata: Dict, error: str) -> None:
        """Append DLQEntry to dlq.jsonl atomically. Log at ERROR with alert=True."""

    def retry_all(self, memory_manager: "MemoryManager") -> int:
        """
        Read all non-dead entries. For each:
          - increment attempts
          - call memory_manager.save_message()
          - on success: remove from DLQ
          - on failure: update attempts; if >= max_attempts: mark dead
        Returns number of successfully retried entries.
        """

    def start_background_retry(
        self, memory_manager: "MemoryManager", interval_seconds: int
    ) -> threading.Timer:
        """Start a recurring threading.Timer that calls retry_all(). Returns the timer."""

    def _load_entries(self) -> List[DLQEntry]: ...
    def _save_entries(self, entries: List[DLQEntry]) -> None: ...  # atomic write
```

**Integration in `MemoryManager.save_message()`**:
```python
try:
    self.storage.save(message_id, data)
    self.indexer.add(message_id, content, metadata)
    if self.session_storage:
        self.session_storage.save(message_id, data)
except Exception as e:
    logger.error(f"save_message failed for {message_id}: {e}")
    if self.dlq:
        self.dlq.enqueue(message_id, content, metadata, str(e))
    raise   # still raise so callers know it failed
```

`MemoryManager.__init__` gains optional `dlq: Optional[DeadLetterQueue] = None`.

---

### `src/memory/manager.py` — Extensions

**`get_context_with_keys()`** (existing `get_context()` delegates to it):
```python
def get_context_with_keys(self, query: str, limit: int = 3) -> Tuple[str, List[str]]:
    # Same routing logic as current get_context()
    # Collect result.key from each SearchResult
    # Return (formatted_string, [key1, key2, ...])

def get_context(self, query: str, limit: int = 3) -> str:
    context, _ = self.get_context_with_keys(query, limit)
    return context   # backward-compatible, no interface break
```

**Content-hash deduplication** in `save_message()`:
```python
content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
index = self.indexer._load_index()
for existing_key, entry in index.items():
    if entry.get("metadata", {}).get("content_hash") == content_hash:
        logger.debug(f"Dedup: skipping {message_id}, hash {content_hash} matches {existing_key}")
        return
metadata = {**(metadata or {}), "content_hash": content_hash}
```

---

### `src/tools/memorize.py` — Logging
```python
logger = get_logger(__name__)

def create_memorize_tool(memory_manager: MemoryManager):
    @tool
    def memorize(content: str, type: str, topic: str = "") -> str:
        """..."""
        logger.debug("memorize called", extra={"data": {"type": type, "topic": topic}})
        if not content.strip():
            logger.warning("memorize: empty content rejected")
            return "Error: content cannot be empty."
        if type not in VALID_TYPES:
            logger.warning(f"memorize: invalid type '{type}'")
            return f"Error: type must be one of {sorted(VALID_TYPES)}."
        message_id = f"{type}_{uuid4().hex[:8]}"
        metadata = {"type": type, "topic": topic.strip() or None,
                    "timestamp": datetime.utcnow().isoformat()}
        try:
            memory_manager.save_message(message_id, content.strip(), metadata)
            logger.debug(f"memorize: saved {message_id}")
            return f"Saved to memory with key: {message_id}"
        except Exception as e:
            logger.error(f"memorize: save failed: {e}")
            return f"Error saving to memory: {str(e)}"
    return memorize
```

**All four hooks follow the same pattern**: `logger = get_logger(__name__)` at module level,
`logger.debug("hook_name: starting", ...)` at entry, error/outcome logged before return.

---

### Hooks — Logging Pattern (apply to all four)
```python
logger = get_logger(__name__)

class PreMemFetchHook(BaseHook):
    def run(self, query: str) -> HookResult:
        logger.debug("pre_mem_fetch: normalising query", extra={"data": {"query": query[:100]}})
        # ... normalisation ...
        logger.debug("pre_mem_fetch: done", extra={"data": {"result": cleaned[:100]}})
        return HookResult(success=True, message=cleaned)
```

---

### Integration Tests — Real Redis + Real FS

**`tests/conftest.py`** — session-scoped fixtures:
```python
import subprocess, time, pytest, redis, shutil, tempfile, os

REDIS_TEST_PORT = 6380   # separate port from any dev Redis

@pytest.fixture(scope="session")
def redis_server():
    """Start a local redis-server on port 6380; yield; shut down."""
    proc = subprocess.Popen(
        ["redis-server", "--port", str(REDIS_TEST_PORT), "--loglevel", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for Redis to be ready
    client = redis.Redis(port=REDIS_TEST_PORT)
    for _ in range(20):
        try:
            client.ping(); break
        except redis.ConnectionError:
            time.sleep(0.1)
    yield proc
    proc.terminate(); proc.wait()

@pytest.fixture(scope="session")
def test_memory_root(tmp_path_factory):
    """Session-scoped temp dir for memory storage."""
    root = tmp_path_factory.mktemp("memory")
    yield str(root)
    shutil.rmtree(str(root), ignore_errors=True)

@pytest.fixture
def clean_redis(redis_server):
    """Per-test Redis flush using test port."""
    client = redis.Redis(port=REDIS_TEST_PORT)
    client.flushall()
    yield client
    client.flushall()
```

**LLM client mocking** in integration tests:
```python
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.completion.return_value = MagicMock(content=[MagicMock(text="Test response")])
    return mock
```

**`Makefile` additions**:
```makefile
test-unit:
    . .venv/bin/activate && PYTHONPATH=. pytest tests/unit/ -v

test-integration:
    . .venv/bin/activate && PYTHONPATH=. pytest tests/integration/ -v -m integration

test-infra-start:
    redis-server --port 6380 --daemonize yes --logfile /tmp/redis-test.log
    @echo "Test Redis started on port 6380"

test-infra-stop:
    redis-cli -p 6380 shutdown nosave || true
    @echo "Test Redis stopped"
    rm -rf test_memory_data/
    @echo "Test filesystem cleaned"

test-all: test-unit test-integration
```

---

## Test Cases

### `tests/unit/test_config.py`
- Each required field missing individually → `ValidationError`
- All required fields + all defaults applied correctly
- Values via env vars (no `.env`) → passes
- DLQ config fields have correct defaults

### `tests/unit/test_typed_storage.py`
- Episodic → `conversations/YYYY-MM/{key}.md` exists, key in manifest
- Semantic + topic → `knowledge/{topic}/{key}.md`
- Semantic without topic → `knowledge/general/{key}.md`
- Procedural → `procedures/{topic_or_general}/{key}.md`
- `load` existing → `{"id": key, "content": ..., "metadata": {}}`
- `load` missing key → `None` (no scan)
- Delete `.md` file manually → `load` returns `None` gracefully
- Two concurrent saves → no manifest corruption (threading test)
- Same key saved twice → file overwritten, one manifest entry
- Content with markdown characters preserved on round-trip

### `tests/unit/test_dlq.py`
- `enqueue` → entry in dlq.jsonl, `attempts=0`, `dead=False`
- `retry_all` success → entry removed from file, count=1
- `retry_all` failure → `attempts` incremented, entry still in file
- `retry_all` failure × max_attempts → `dead=True`, not retried again
- `retry_all` mixed (some pass, some fail) → correct entries removed
- DLQ survives process restart (file persisted to disk)
- Concurrent enqueue + retry → no file corruption (lock test)

### `tests/unit/test_manager_extensions.py`
- `get_context_with_keys` returns `(str, list)` with correct keys
- `get_context` returns `str` — backward compat unchanged
- Dedup: same content saved twice → second call returns early, index unchanged
- Dedup: different content, same key → both saved (hash on content not key)
- `save_message` fails → DLQ `enqueue` called, exception re-raised
- `save_message` fails, no DLQ configured → exception re-raised, no crash

### `tests/unit/test_tools.py` (mock `MemoryManager`)
- Valid episodic → returns key string, `save_message` called once
- Valid semantic + topic → topic in metadata
- Empty content → error string, `save_message` NOT called
- Whitespace-only content → error string
- Invalid type → error string with valid options
- `type="Episodic"` (wrong case) → error (case-sensitive)
- `save_message` raises `OSError` → returns error string (no raise from tool)
- Unicode content (emoji, CJK) → passed through, `save_message` called
- Empty `topic` → stored as `None` in metadata
- Very long content (10k chars) → no truncation, full content to `save_message`
- Logger called at entry and on success/failure

### `tests/unit/test_hooks.py` (all deps mocked)

**`PreMemFetchHook`**
- `""` → `""`; `"!@#$%"` → `""`
- `"what's up"` → `"what's up"` (apostrophe kept)
- `"Hello World"` → `"hello world"`; `"hello  world"` → `"hello world"`
- `"  hello "` → `"hello"` (stripped)
- 5000-char query → processed without error
- Logger called at entry and exit

**`PostMemFetchHook`**
- Context under limit → unchanged; at limit → unchanged
- Over limit (3 blocks) → top blocks kept; single oversized block → returned as-is
- `""` → `""`, no crash; `retrieved_keys=[]` → retrieval_event with empty list
- Logger called on truncation

**`PreSessionHook`**
- All healthy → `HookResult(success=True, degraded=False)`; FS missing → creates it
- FS not writable → `success=False`; Redis `ConnectionError` → `degraded=True`
- LLM `AuthenticationError` → `success=False`; LLM `RateLimitError` → `degraded=True`
- Redis fail + LLM rate-limit (both non-fatal) → `degraded=True`, message mentions both
- Logger called at each step, errors logged at ERROR

**`PostSessionHook`**
- 0 Redis keys → no flush, metrics written; 1 key fails load → skipped, rest flushed
- LLM summary raises → logged, metrics still written
- Metrics dir missing → created automatically
- 0 turns → no crash; 200 turns → last 20 passed to LLM
- Logger called for each flush key and on summary

### `tests/integration/test_memory_integration.py` (real Redis port 6380, real FS)
- Save 3 messages, search retrieves correct ranking
- Redis session mirroring: save → verify key in Redis with TTL
- Session eviction: fill to `session_max_messages + 1` → oldest evicted to FS
- TypedMarkdownStorage: save episodic → `.md` file in `conversations/YYYY-MM/`
- Deduplication: save same content twice → file count unchanged
- DLQ retry: make FS read-only → save fails → DLQ entry created → restore FS → retry succeeds

### `tests/integration/test_hooks_integration.py` (real Redis + FS; LLM mocked)
- `PreSessionHook`: all healthy, real Redis, real FS → passes
- `PreSessionHook`: Redis on wrong port → `degraded=True`
- `PostSessionHook`: save 5 messages to Redis → run hook → all 5 in FS
- `PostSessionHook`: LLM mocked → summary saved as `.md` in `conversations/`

### `tests/integration/test_orchestrator_integration.py` (real Redis + FS; LLM + Strands mocked)
- Full turn: input → memory fetch → mock agent response → turn saved to FS
- Exit command → close session → hook runs → metrics.jsonl written
- Memory round-trip: save a message, run a turn querying it → context appears in agent call

---

## `.env.example`
```bash
# Required
LLM_MODEL=claude-sonnet-4-6
LLM_API_KEY=sk-ant-...
MEMORY_ROOT=./memory
INDEX_PATH=./memory/index.json

# Optional (defaults shown)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=3600
SESSION_MAX_MESSAGES=100
SESSION_INACTIVITY_TIMEOUT_SECONDS=300
TOOL_TIMEOUT_SECONDS=10
MAX_CONTEXT_CHARS=8000
LOG_PATH=./logs/app.jsonl
LOG_LEVEL=INFO
LLM_MAX_TOKENS=1024
DLQ_PATH=./memory/dlq.jsonl
DLQ_MAX_ATTEMPTS=3
DLQ_RETRY_INTERVAL_SECONDS=60
```

---

## First Implementation Step
Copy this plan to the repo as `plan.md` (per CLAUDE.md planning rules):
```bash
cp /home/mhyz1n4/.claude/plans/mutable-greeting-melody.md plan.md
git add plan.md && git commit -m "docs: add v1 implementation plan"
```

---

## Verification
1. `make setup` → install all deps
2. `make test-unit` → all unit tests green (no Redis/FS infra needed)
3. `make test-infra-start` → Redis running on port 6380
4. `make test-integration` → integration tests green (real Redis + FS, LLM mocked)
5. `make test-infra-stop` → Redis stopped, test dirs cleaned
6. `cp .env.example .env` → fill `LLM_API_KEY`
7. `python main.py` → terminal starts
8. Type `"remember that I prefer Python"` → memorize tool logs appear, `.md` file in `memory/knowledge/`
9. Type `"what language do I prefer?"` → memory context injected in agent call
10. Type `"exit"` → `logs/metrics.jsonl` written, DLQ background timer stopped
11. `cat logs/app.jsonl | python -m json.tool | head -30` → JSON structured logs with trace_id
