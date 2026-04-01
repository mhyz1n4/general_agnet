# AI Changes Log

---

## 2026-03-30 — v1 Design Hole Fixes (Fixes 1–5, 7)

### Fix 1 — Redis files moved to `memory/redis/` subfolder

`redis_storage.py` and `redis_storage_design.md` were loose in `src/memory/` while
file-system components already lived in `src/memory/file_system/`. They are now
reorganised into a dedicated subfolder:

```
src/memory/redis/
├── __init__.py       # exports RedisStorage
├── storage.py        # formerly redis_storage.py (import path updated)
└── design.md         # formerly redis_storage_design.md
```

`src/memory/__init__.py` re-exports `RedisStorage` under a try/except so the package
works even when the `redis` pip package is not installed.

---

### Fix 2 — Retrieval pipeline wired into `MemoryManager.get_context()`

`MemoryManager` previously called `self.retriever.search()` directly, bypassing the
classifier and temporal extractor entirely. `get_context()` now:

1. Calls `self.classifier.classify(query)` when a classifier is injected.
2. Short-circuits with `""` for `current_session` intent (v1 limitation — Redis has no
   indexer/retriever; placeholder for v2 Redis session scan).
3. Calls `self.temporal_extractor.extract(query)` for `recall_history` intent and logs
   the detected date range (v2 will pass this range to the retriever as a filter).
4. Falls through to keyword search for `recall_history` and `general_task`.

Both `classifier` and `temporal_extractor` are optional constructor args — existing
callers that pass neither are unaffected.

---

### Fix 3 — Dual-backend support added to `MemoryManager`

`MemoryManager.__init__` now accepts an optional `session_storage: BaseStorage`
parameter (intended for `RedisStorage`). When provided:

- `save_message()` mirrors every message to `session_storage` after writing to the
  long-term backend. Redis TTL handles automatic expiry with no extra logic needed.
- `get_context()` is already aware of session intent (Fix 2) and will route to the
  session backend in v2 once a Redis retriever exists.

No breaking change — `session_storage` defaults to `None`.

---

### Fix 4 — `src/retrieval/` renamed to `src/query/`

The `src/retrieval/` folder contained high-level *query routing* logic (intent
classification, temporal extraction). Having two things named "retrieval" —
`src/retrieval/` and `src/memory/file_system/retriever.py` — created a naming
collision. The folder is now `src/query/` to clearly distinguish query preprocessing
from storage-level retrieval.

```
src/query/
├── __init__.py        # exports RegexClassifier, Intent, TemporalExtractor
├── classifier.py      # formerly src/retrieval/classifier.py
└── preprocessor.py    # formerly src/retrieval/preprocessor.py
```

---

### Fix 5 — LLMClient singleton already correct (no change needed)

On inspection, `LLMClient.base.py` already uses a class-keyed dict
(`_instances: Dict[Type["LLMClient"], "LLMClient"] = {}`) with `cls` as the lookup
key. Each subclass gets its own singleton slot — `OpenAIClient` and `ClaudeClient` do
not collide. The earlier concern was based on a misread; the implementation is correct.

---

### Fix 7 — Redis unit tests added

`tests/test_redis_storage.py` adds 9 mock-based tests covering:

- `save()`: default TTL, explicit TTL, zero-TTL (no expiry), custom prefix
- `load()`: successful deserialisation, missing key returns `None`, corrupt JSON
  returns `None`
- `delete()`: returns `True` when key exists, `False` when missing

Tests use `unittest.mock.patch` on `redis.Redis` — no live Redis instance required.

---

### Files changed

| Action   | Path |
|----------|------|
| Created  | `src/memory/redis/__init__.py` |
| Created  | `src/memory/redis/storage.py` |
| Moved    | `src/memory/redis/design.md` (was `redis_storage_design.md`) |
| Deleted  | `src/memory/redis_storage.py` |
| Deleted  | `src/memory/redis_storage_design.md` |
| Modified | `src/memory/__init__.py` |
| Modified | `src/memory/manager.py` |
| Created  | `src/query/__init__.py` |
| Created  | `src/query/classifier.py` |
| Created  | `src/query/preprocessor.py` |
| Deleted  | `src/retrieval/classifier.py` |
| Deleted  | `src/retrieval/preprocessor.py` |
| Created  | `tests/test_redis_storage.py` |
| Created  | `AI_changes.md` |
