"""
ReMeLight-backed implementation of ``MemoryProvider``.

ReMeLight is an async framework: ``memory_search``, ``check_context``, and
``compact_memory`` are coroutines backed by a persistent service context
(SQLite file store, file watcher, thread pool). The orchestrator consumes
``MemoryProvider`` synchronously, so this adapter owns a background asyncio
loop in a daemon thread and runs ReMeLight coroutines on it via
``run_coroutine_threadsafe``.

V1.1 configuration (FTS-only, stubbed LLM):
    - SQLite file store with FTS5; no vectors, no embeddings API call.
    - ``stub`` LLM backend — Compactor/Summarizer return empty strings instead
      of crashing on startup when no real LLM is wired in.

Save path writes a YAML-frontmattered ``.md`` file under ``memory/``; the
ReMeLight file watcher picks it up (sub-second latency in practice) and
indexes it. Session turns are appended to per-day JSONL files under
``dialog/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agentscope.message import Msg

from reme.reme_light import ReMeLight

from src.constants import (
    DEFAULT_MEMORY_TOPIC,
    MEMORY_DATE_FOLDER_FORMAT,
    MEMORY_TYPE_SEMANTIC,
    VALID_MEMORY_TYPES,
)
from src.memory.provider import (
    MemoryItem,
    MemoryType,
    Message,
    ReasoningContext,
    SearchFilters,
    Summary,
)

from .stub_llm import STUB_LLM_BACKEND_NAME, register_stub_llm


logger = logging.getLogger(__name__)

_FRONTMATTER_DELIM = "---"


class ReMeLightProvider:
    """``MemoryProvider`` backed by a ReMeLight application instance."""

    def __init__(
        self,
        memory_root: str,
        session_id: str,
        as_llm_config: Optional[dict[str, Any]] = None,
        file_store_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            memory_root: Working directory for ReMeLight (memory/, dialog/,
                tool_result/ created underneath).
            session_id: Active session identifier used to tag dialog turns.
            as_llm_config: Optional override for the LLM backend config.
                Defaults to the registered ``stub`` backend.
            file_store_config: Optional override for file store config.
                Defaults to SQLite with FTS-only.
        """
        register_stub_llm()

        self._memory_root = Path(memory_root).absolute()
        self._session_id = session_id
        self._memory_write_lock = threading.Lock()
        self._dialog_lock = threading.Lock()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="remelight-loop",
            daemon=True,
        )
        self._loop_thread.start()

        self._reme = ReMeLight(
            working_dir=str(self._memory_root),
            default_as_llm_config=as_llm_config
            or {"backend": STUB_LLM_BACKEND_NAME, "model_name": "stub"},
            default_file_store_config=file_store_config
            or {
                "backend": "sqlite",
                "vector_enabled": False,
                "fts_enabled": True,
            },
            enable_load_env=False,
        )
        self._run_sync(self._reme.start())

    # ------------------------------------------------------------------ #
    # Protocol methods
    # ------------------------------------------------------------------ #

    def search(self, query: str, filters: SearchFilters) -> list[MemoryItem]:
        """
        Query ReMeLight's FTS index and return matching memory items.

        Over-fetches ``filters.limit * 3`` raw results from the backend then
        applies ``types`` / ``topics`` post-filters in Python, truncating to
        ``filters.limit``. Returns ``[]`` for a blank query or when the
        backend payload is not valid JSON.

        Args:
            query:   Search text; blank strings short-circuit to ``[]``.
            filters: Structured filter set applied after the backend query.

        Returns:
            Matching ``MemoryItem`` list with ``relevance_score`` populated
            from the backend score. May be shorter than ``filters.limit``
            when filters are selective relative to the over-fetch multiplier.
        """
        if not query.strip():
            return []
        response = self._run_sync(
            self._reme.memory_search(
                query=query,
                max_results=filters.limit * 3,
                min_score=0.001,
            )
        )
        raw_text = response.content[0]["text"] if response.content else "[]"
        try:
            raw_results = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("memory_search returned non-JSON payload: %s", raw_text[:200])
            return []

        items: list[MemoryItem] = []
        for raw in raw_results:
            snippet = raw.get("snippet", "")
            item_type, topic, body = _parse_frontmatter(snippet)
            if filters.types is not None and item_type not in filters.types:
                continue
            if filters.topics is not None and topic not in filters.topics:
                continue
            items.append(
                MemoryItem(
                    key=raw.get("path", ""),
                    content=body,
                    type=item_type,
                    topic=topic,
                    timestamp=None,
                    relevance_score=raw.get("score"),
                )
            )
            if len(items) >= filters.limit:
                break
        return items

    def save(
        self,
        content: str,
        type: MemoryType,
        topic: Optional[str] = None,
    ) -> MemoryItem:
        """
        Append a YAML-frontmattered memory entry to today's ``.md`` file.

        Writes under ``memory/{YYYY-MM-DD}.md`` inside ``memory_root``. The
        ReMeLight file watcher picks up the change asynchronously (sub-second
        in practice) and indexes it. The write is serialised by
        ``_memory_write_lock`` so concurrent saves from this process do not
        interleave.

        Args:
            content: Memory body text.
            type:    Memory type literal.
            topic:   Optional topic; falls back to ``DEFAULT_MEMORY_TOPIC``.

        Returns:
            ``MemoryItem`` whose ``key`` is the absolute file path and
            ``timestamp`` is the UTC ISO-8601 write time.
        """
        topic = topic or DEFAULT_MEMORY_TOPIC
        now = datetime.now(timezone.utc)
        date_str = now.strftime(MEMORY_DATE_FOLDER_FORMAT)
        key = uuid.uuid4().hex[:12]

        memory_dir = self._memory_root / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        path = memory_dir / f"{date_str}.md"

        entry = _format_entry(content=content, type=type, topic=topic, key=key)
        with self._memory_write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(entry)

        return MemoryItem(
            key=str(path),
            content=content,
            type=type,
            topic=topic,
            timestamp=now.isoformat(),
        )

    def compact(self, messages: list[Message]) -> Summary:
        """
        Delegate message summarisation to ReMeLight's compactor.

        With the V1.1 stub LLM wired in, the returned ``text`` is empty —
        only ``source_count`` is meaningful. Once a real LLM is registered
        this method returns the actual summary.

        Args:
            messages: Messages to compact; empty input short-circuits to an
                      empty ``Summary``.

        Returns:
            ``Summary`` whose ``source_count`` is ``len(messages)`` and whose
            ``text`` is the compactor output (empty string under the stub LLM).
        """
        if not messages:
            return Summary(text="", source_count=0)
        msgs = [_to_agentscope_msg(m) for m in messages]
        result = self._run_sync(self._reme.compact_memory(messages=msgs, language=""))
        text = result if isinstance(result, str) else ""
        return Summary(text=text, source_count=len(messages))

    def check_context(
        self,
        messages: list[Message],
        budget_tokens: int,
    ) -> list[Message]:
        """
        Trim ``messages`` to fit within ``budget_tokens`` via ReMeLight.

        Converts messages to AgentScope ``Msg`` objects, calls
        ``ContextChecker`` with the budget as both ``memory_compact_threshold``
        and a reserved quarter, then maps the kept AgentScope messages back to
        the original dicts by matching content text.

        Args:
            messages:      Full conversation history (oldest-first).
            budget_tokens: Token budget for the returned slice.

        Returns:
            The subset of ``messages`` whose content was retained by the
            checker. Order follows the original ``messages`` list.
        """
        if not messages:
            return []
        msgs = [_to_agentscope_msg(m) for m in messages]
        _, messages_to_keep, _ = self._run_sync(
            self._reme.check_context(
                messages=msgs,
                memory_compact_threshold=max(budget_tokens, 1),
                memory_compact_reserve=max(budget_tokens // 4, 1),
            )
        )
        kept_texts = {m.get_text_content() for m in messages_to_keep}
        return [m for m in messages if m.get("content", "") in kept_texts]

    def save_session_turn(self, message: Message) -> None:
        """
        Append a dialog turn to ``dialog/{YYYY-MM-DD}.jsonl``.

        Missing ``timestamp`` is filled with the current UTC ISO-8601 time;
        missing or empty ``session_id`` falls back to the provider's
        configured ``session_id``. Writes are serialised by
        ``_dialog_lock`` so concurrent saves from this process do not
        interleave within the file.

        Args:
            message: Message dict following the ``Message`` TypedDict schema.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime(MEMORY_DATE_FOLDER_FORMAT)
        dialog_dir = self._memory_root / "dialog"
        dialog_dir.mkdir(parents=True, exist_ok=True)
        path = dialog_dir / f"{date_str}.jsonl"

        record = {
            "role": message.get("role", ""),
            "content": message.get("content", ""),
            "timestamp": message.get("timestamp") or now.isoformat(),
            "session_id": message.get("session_id") or self._session_id,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._dialog_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def get_session_history(self, session_id: str) -> list[Message]:
        """
        Scan all daily JSONL files and return turns for ``session_id``.

        Malformed lines are logged and skipped. Files are iterated in
        lexicographic (date) order so the returned history is roughly
        chronological.

        Args:
            session_id: Session identifier to filter on.

        Returns:
            List of persisted message records. Empty when the dialog directory
            does not exist or no record matches.
        """
        dialog_dir = self._memory_root / "dialog"
        if not dialog_dir.exists():
            return []

        history: list[Message] = []
        for jsonl_path in sorted(dialog_dir.glob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed dialog line in %s", jsonl_path)
                        continue
                    if record.get("session_id") == session_id:
                        history.append(record)  # type: ignore[arg-type]
        return history

    def pre_reasoning_hook(self, context: ReasoningContext) -> ReasoningContext:
        """
        Trim ``context.messages`` via ``check_context`` before the LLM call.

        Returns ``context`` unchanged when it has no messages or a non-positive
        budget. Otherwise emits a fresh ``ReasoningContext`` with the trimmed
        messages and the original ``memory_items`` / ``budget_tokens``.

        Args:
            context: Input reasoning context.

        Returns:
            Possibly a new ``ReasoningContext`` with trimmed messages.
        """
        if not context.messages or context.budget_tokens <= 0:
            return context
        kept = self.check_context(context.messages, context.budget_tokens)
        return ReasoningContext(
            messages=kept,
            memory_items=context.memory_items,
            budget_tokens=context.budget_tokens,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Shut down the background loop and ReMeLight instance."""
        try:
            self._run_sync(self._reme.close())
        except Exception:  # noqa: BLE001
            logger.exception("ReMeLight close raised")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2)
            if self._loop_thread.is_alive():
                logger.warning("remelight-loop thread did not stop within 2s")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """Background-thread target: bind ``self._loop`` and run it forever."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_sync(self, coro: Any) -> Any:
        """
        Run ``coro`` on the provider's background event loop and block for
        the result.

        Args:
            coro: Awaitable to schedule on the background loop.

        Returns:
            The coroutine's return value.

        Raises:
            Exception: Re-raises whatever the coroutine raised.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _format_entry(content: str, type: str, topic: str, key: str) -> str:
    """Produce a frontmattered memory block ready to append to a dated .md file."""
    return (
        f"\n{_FRONTMATTER_DELIM}\n"
        f"type: {type}\n"
        f"topic: {topic}\n"
        f"key: {key}\n"
        f"{_FRONTMATTER_DELIM}\n"
        f"{content.rstrip()}\n"
    )


def _parse_frontmatter(snippet: str) -> tuple[MemoryType, Optional[str], str]:
    """
    Extract (type, topic, body) from a snippet returned by memory_search.

    Snippets with no frontmatter default to type=semantic, topic=None, body=full snippet.
    """
    lines = snippet.split("\n")
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return (MEMORY_TYPE_SEMANTIC, None, snippet)  # type: ignore[return-value]

    meta: dict[str, str] = {}
    body_start = len(lines)
    for idx in range(1, len(lines)):
        stripped = lines[idx].strip()
        if stripped == _FRONTMATTER_DELIM:
            body_start = idx + 1
            break
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            meta[k.strip()] = v.strip()

    raw_type = meta.get("type", MEMORY_TYPE_SEMANTIC)
    item_type: MemoryType = (
        raw_type if raw_type in VALID_MEMORY_TYPES else MEMORY_TYPE_SEMANTIC
    )  # type: ignore[assignment]
    topic = meta.get("topic") or None
    body = "\n".join(lines[body_start:]).strip()
    return (item_type, topic, body)


def _to_agentscope_msg(message: Message) -> Msg:
    """
    Convert a ``Message`` TypedDict into an AgentScope ``Msg``.

    Roles outside ``{"user", "assistant", "system"}`` are collapsed to
    ``"user"`` so the downstream ReMeLight API accepts the payload.

    Args:
        message: Source message dict.

    Returns:
        AgentScope ``Msg`` with matching role, content, and timestamp.
    """
    role = message.get("role", "user")
    if role not in ("user", "assistant", "system"):
        role = "user"
    return Msg(
        name=role,
        role=role,  # type: ignore[arg-type]
        content=message.get("content", ""),
        timestamp=message.get("timestamp"),
    )
