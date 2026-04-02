"""
File system indexing implementation for the memory system.

This module provides the JSONIndexer class, which maintains a central JSON file
for indexing content with metadata and keywords.

Concurrency model
-----------------
Two locks are layered to cover every access scenario:

1. ``threading.Lock`` (self._thread_lock) — prevents races between threads in
   the *same* process.  ``fcntl.flock`` / file-based locks are per-process on
   Linux; they do not block two threads in the same process from entering the
   critical section simultaneously, so a thread lock is still needed.

2. ``filelock.FileLock`` (self._file_lock) — holds an OS-level advisory lock on
   ``{index_path}.lock`` for the entire read-modify-write cycle.  This prevents
   data loss when multiple processes (e.g. two terminal sessions) write the
   index concurrently.  The final write is still an atomic ``os.replace`` as an
   additional safeguard.
"""

import os
import json
import logging
import threading
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from filelock import FileLock

from ..base import BaseIndexer

logger = logging.getLogger(__name__)


class JSONIndexer(BaseIndexer):
    """
    File system based indexer using a central JSON file.

    Thread-safe and cross-process safe: every read-modify-write cycle holds
    both a threading.Lock (intra-process) and a FileLock (inter-process).
    """

    def __init__(self, index_path: str) -> None:
        """
        Initialize the JSONIndexer with a target index file path.

        Args:
            index_path: Absolute or relative path to the JSON index file.
        """
        self.index_path: str = index_path
        self._thread_lock: threading.Lock = threading.Lock()
        self._file_lock: FileLock = FileLock(f"{index_path}.lock")
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        if not os.path.exists(self.index_path):
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    # ------------------------------------------------------------------
    # Low-level helpers  (must be called while both locks are held)
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, Any]:
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error(f"Error loading index at {self.index_path}, returning empty.")
            return {}

    def _save_index(self, index_data: Dict[str, Any]) -> None:
        tmp_path: str = f"{self.index_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.index_path)
        except Exception as exc:
            logger.error(f"Error saving index to {self.index_path}: {exc}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _locked(self):
        """Context manager: acquire thread lock then file lock."""
        class _Ctx:
            def __init__(self_, outer):
                self_._outer = outer

            def __enter__(self_):
                self_._outer._thread_lock.acquire()
                self_._outer._file_lock.acquire()
                return self_

            def __exit__(self_, *_):
                self_._outer._file_lock.release()
                self_._outer._thread_lock.release()

        return _Ctx(self)

    # ------------------------------------------------------------------
    # BaseIndexer interface
    # ------------------------------------------------------------------

    def add(self, key: str, content: str, metadata: Dict[str, Any]) -> None:
        """Index new content by extracting keywords and updating the index file."""
        with self._locked():
            index = self._load_index()
            index[key] = {
                "metadata": metadata,
                "keywords": self._extract_keywords(content),
            }
            self._save_index(index)

    def update(self, key: str, content: str, metadata: Dict[str, Any]) -> None:
        """Update an existing index entry (alias for add)."""
        self.add(key, content, metadata)

    def find_by_content_hash(self, content_hash: str) -> Optional[str]:
        """
        Return the key of any entry whose metadata.content_hash matches, or None.

        This is the sole approved way for MemoryManager to perform deduplication
        — it does not expose the internal index structure.
        """
        with self._locked():
            index = self._load_index()
        for key, entry in index.items():
            if entry.get("metadata", {}).get("content_hash") == content_hash:
                return key
        return None

    def touch(self, key: str) -> None:
        """
        Refresh the stored timestamp on an existing entry.

        Called when MemoryManager detects duplicate content — keeps the entry's
        last-seen time current without re-writing the content or keywords.
        Silently no-ops if the key does not exist.
        """
        with self._locked():
            index = self._load_index()
            if key not in index:
                return
            index[key].setdefault("metadata", {})["timestamp"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._save_index(index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract unique words longer than 3 characters from content."""
        words: List[str] = re.findall(r"\w+", content.lower())
        return list(set(w for w in words if len(w) > 3))
