# Summary
This is a general personal assistant agent system.


# Main Components

## V1

### 1. Configuration Management  *(missing from original — added)*
- a. All runtime config loaded from a single `.env` file and validated at startup via `pydantic-settings`. Fail fast if required values are absent.
- b. Config categories: LLM (model name, API key, max tokens per call), Redis (host, port, TTL, max session messages), file system (memory root path), session (inactivity timeout seconds), logging (level, output path).
- c. Config is injected into components as constructor arguments — no component reads `.env` directly. This keeps components testable and decoupled from the environment.

---

### 2. LLM Client  *(missing from original — added)*
- a. Abstract `LLMClient` base class with `completion()` and `function_call()` methods. Concrete implementations: `ClaudeClient`, `OpenAIClient`. See `src/client/`.
- b. Singleton per subclass — one client instance per provider per process lifetime.
- c. All LLM calls go through the client. No component calls the provider SDK directly. This isolates API changes and makes latency instrumentation a single location.
- d. `function_call()` accepts a typed tool schema list and returns a structured tool-call result. The orchestrator parses this; the client does not interpret it.

---

### 3. Memory
- a. **Long-term memory**: filesystem storage. Each memory item is a `.md` file. A companion JSON index (`index.json`) maps keys to keywords + metadata for fast search without loading file content.
- b. **Short-term memory**: Redis session store. Stores full message objects with TTL. Max session window = `SESSION_MAX_MESSAGES` (config). When the window is full, the oldest message is evicted and written to long-term memory before deletion.
- c. **File system structure** (3-level max, v2-compatible):
  ```
  memory/
  ├── conversations/          # episodic — what happened
  │   └── YYYY-MM/
  │       └── {session_id}.md
  ├── knowledge/              # semantic — facts, preferences, notes
  │   └── {topic}/
  │       └── {id}.md
  └── procedures/             # procedural — how-tos, workflows
      └── {topic}/
          └── {id}.md
  ```
  Top level encodes **memory type** so v2 vector indexing can embed each type separately and apply type-aware retrieval strategies.
- d. **Memory typing**: every saved item carries a `type` field (`episodic | semantic | procedural`). The query layer uses this to scope searches — a "how do I…" query only searches `procedures/`, not `conversations/`.
- e. **Deduplication**: before writing a new item, compute a content hash. If an identical hash exists in the index, skip the write and update the timestamp on the existing entry instead.
- f. **No memory decay in v1**. Long-term storage grows monotonically. V2 will add a compaction pass that summarises old episodic entries and drops superseded semantic entries.

---

### 4. Query
- a. Intent classification via `RegexClassifier` (`src/query/classifier.py`). Three intents: `recall_history`, `current_session`, `general_task`.
- b. Temporal extraction via `TemporalExtractor` (`src/query/preprocessor.py`) for `recall_history` queries. Extracts date range; v1 logs it for debug; v2 passes it as a retriever filter.
- c. Retrieval: `KeywordRetriever` searches the JSON index for keyword intersection, loads matching files, scores by match count. Returns up to `QUERY_RESULT_LIMIT` (config) `SearchResult` objects.
- d. **Fallback**: if keyword search returns zero results, retry with individual query tokens (split compound query). If still zero, return empty context — do not fabricate.
- e. **Session eviction policy**: Redis session window is capped at `SESSION_MAX_MESSAGES`. On overflow: evict the oldest message → write it to `conversations/YYYY-MM/{session_id}.md` in the filesystem backend → index it → delete from Redis. This runs synchronously before the new message is written.
- f. **Query preprocessing** (also enforced by pre-mem-fetch hook): strip non-alphanumeric characters except spaces and apostrophes, lowercase, collapse whitespace.

---

### 5. Orchestrator
- a. Chat loop: `user input → preprocess query → classify intent → fetch memory context → build prompt → LLM call → parse response → execute tool if needed → return result to user → save turn to session memory`.
- b. **Context window budget**: before every LLM call, count tokens in: system prompt + memory context + conversation history + pending tool results. If total exceeds `LLM_MAX_CONTEXT_TOKENS * 0.9` (10% safety margin), truncate conversation history oldest-first until it fits. Memory context and system prompt are never truncated.
- c. **Error recovery**: tool call failures return a structured error to the LLM with an instruction to either retry once or inform the user. After two consecutive tool failures on the same turn, return a graceful error message to the user and log the incident.
- d. **Timeout**: all tool calls are wrapped with a `TOOL_TIMEOUT_SECONDS` (config) deadline. On timeout, treat as a tool failure (see above).
- e. **Session lifecycle**: session closes on (1) explicit user command, (2) inactivity exceeding `SESSION_INACTIVITY_TIMEOUT_SECONDS` (config), or (3) unrecoverable system error. In all cases the post-session hook runs before exit.
- f. **No multi-step tool chaining in v1**. Each turn allows at most one tool call. Multi-hop reasoning is a v2 feature (requires graph-based orchestration).

---

### 6. Sub-agents
- a. **Memorize sub-agent**: triggered when the user explicitly says "remember this" or similar. Steps: classify what to remember (type: episodic/semantic/procedural) → determine file path and key → write to filesystem via `MemoryManager.save_message()` → verify the file exists on disk → return success/failure to orchestrator. This is the only write path for explicit user-directed memory saves; the post-session hook handles automatic session saves.
- b. ~~Memory retrieval sub-agent~~ — **removed**. Retrieval is handled entirely by the query component + `MemoryManager.get_context()`. A separate sub-agent was redundant.
- c. ~~Memorize sub-agent distinct from memorize tool~~ — **merged**. The memorize tool (item 7a) is the atomic write operation. The memorize sub-agent above is the multi-step verification wrapper around it.
- d. **Sub-agent communication protocol**: sub-agents are Python functions called synchronously by the orchestrator. They return a typed `SubAgentResult(success: bool, output: str, error: Optional[str])`. The orchestrator checks `success` and handles `False` via the error recovery policy (item 5c).
- e. Notion note-taking sub-agent: **deferred to v2**. External dependency adds network failure modes inappropriate for a v1 POC.

---

### 7. Hooks
- a. **Pre-session hook**: verify Redis connectivity, verify filesystem memory root exists and is writable, verify LLM API key is valid (one lightweight ping call). **Failure policy**: Redis failure → log warning, continue with filesystem-only mode (no short-term memory). Filesystem failure → abort with clear error. API key failure → abort.
- b. **Post-session hook**: runs asynchronously in a background thread after the session closes. Steps: (1) flush remaining Redis session messages to long-term filesystem memory, (2) call LLM to produce a one-paragraph session summary and save it to `conversations/YYYY-MM/{session_id}.md`, (3) write session metrics (token count, turn count, memory hits/misses, duration) to the metrics log. If this hook fails, log the error — it must not surface to the user.
- c. **Pre-mem-fetch hook**: clean and normalise user query (see query item 4f). Runs synchronously; fast.
- d. **Post-mem-fetch hook**: if retrieved context exceeds `MAX_CONTEXT_CHARS` (config), truncate to the top-N results by relevance score. Save the query + retrieved key list to Redis session memory as a `retrieval_event` entry for session-scoped recall.

---

### 8. Tools
- a. **Memorize tool**: atomic write of a single item to filesystem memory via `MemoryManager`. Input schema: `{content: str, type: "episodic|semantic|procedural", topic: Optional[str]}`. Returns `{success: bool, key: str, path: str}`. Called by the memorize sub-agent; also callable directly by the LLM via function calling.
- b. Notion note-taking tool: **deferred to v2**.
- c. **Tool contract**: all tools return a consistent `{success: bool, result: Any, error: Optional[str]}` envelope. The orchestrator checks `success` before injecting `result` into the next LLM prompt. On `success=False` the orchestrator follows the error recovery policy (item 5c).

---

### 9. Prompts
- a. **Chat orchestrator system prompt**: defines the assistant persona, available tools (injected from schema at runtime), memory context format, and response style. Includes 2–3 few-shot examples of correct tool-call decisions to anchor routing accuracy.
- b. **Tool prompts**: each tool has a `description` string (used by the LLM for routing) and a JSON schema (used for argument generation). Both live in the tool's definition file alongside the implementation.
- c. **Sub-agent prompts**: each sub-agent has a focused single-task system prompt. No cross-agent prompt reuse in v1 — prompts are tightly scoped.
- d. **Prompt format**: Python string templates (`string.Template`) with `$variable` substitution. Stored as `.txt` files in `src/prompts/`. Versioned with the codebase. No Jinja2 dependency in v1.
- e. **Prompt testing**: each prompt has a corresponding golden input/output pair in the evaluation dataset (item 10b). A prompt change that breaks a golden pair is a test failure.

---

### 10. Evaluation
- a. Unit and integration tests for all components (`pytest`). Memory system, query routing, tool execution, hook behaviour.
- b. **Golden dataset**: 20–30 hand-crafted `{user_input, expected_intent, expected_memory_keys_retrieved, expected_tool_called}` examples. Enough to catch regressions without being a full eval harness. Stored in `tests/golden/`.
- c. **Latency benchmark**: measure P50/P95 for (1) memory retrieval, (2) full orchestrator turn (excluding LLM latency). Target: memory retrieval < 100ms, full turn overhead < 200ms.
- d. **Accuracy metric for v1**: % of golden examples where the correct tool was called AND the correct memory keys were retrieved. Binary per example — no partial credit.

---

### 11. Metrics and Logging
- a. **Structured JSON logs**: every log line is a JSON object with fields: `timestamp`, `level`, `trace_id`, `component`, `event`, `data`. Written to `logs/app.jsonl`. No plain text logs.
- b. **Trace ID**: a UUID generated at the start of each user turn. Passed through memory fetch, tool calls, and LLM calls so a single query can be reconstructed across all log lines.
- c. **Metrics** (appended to `logs/metrics.jsonl` per session): LLM token count (prompt + completion), LLM latency (ms), memory hit count, memory miss count, tool calls made, tool failures, session duration, short-term → long-term eviction count, index size (bytes), storage size (bytes).
- d. **Critical alerts** (logged at `ERROR` level with `alert: true` field, printed to stderr for operator visibility): memory root not writable, Redis unavailable at session start, LLM API returning errors for >2 consecutive calls, session token count exceeds 90% of context limit.
- e. **Log rotation**: `logs/` directory capped at 500MB total. When exceeded, oldest log files are deleted. Implemented in the pre-session hook health check.


---

## V2

### 1. Memory
- a. Expand to vector DB (e.g. ChromaDB, pgvector) to support semantic query understanding. Vector DB acts as a second index layer alongside the existing JSON keyword index — hybrid retrieval in v2 query.
- b. Enhance file system with backup/snapshot support for durability. Consider periodic tar+gz snapshots to a configurable backup path.
- c. Add mid-term memory tier: `user_preferences/` and `project_context/` directories sit between session memory and general long-term memory. More precise context fetch for recurring tasks.
- d. Memory compaction: summarise episodic entries older than N days into a single summary file. Drop individual entries after compaction.

### 2. Query
- a. Hybrid retrieval: keyword BM25 + vector embedding similarity. Combine scores with a weighted sum. Configurable weight ratio.
- b. LLM-directed retrieval as an option: pass a `search_memory` tool to the LLM and let it decide what to search for, instead of the regex classifier.

### 3. Tools
- a. MCP (Model Context Protocol) for external service integrations: Gmail, Calendar, Notion.
- b. Skills for streamlined multi-step workflows that would otherwise consume excessive context.
