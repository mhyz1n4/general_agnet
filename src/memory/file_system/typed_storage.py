"""
TypedMarkdownStorage — file-system storage that organises .md files
into a typed directory structure with a manifest for O(1) key lookup.

Directory layout:
    {memory_root}/
      conversations/{session_id}/{YYYY-MM-DD}/{key}.md   # episodic
      knowledge/{topic_or_general}/{key}.md               # semantic
      procedures/{topic_or_general}/{key}.md              # procedural
      manifest.json                                       # key → relative path
"""

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..base import BaseStorage
from ..types import MemoryMetadata, StorageRecord
from src.constants import (
    DEFAULT_MEMORY_TOPIC,
    JSON_INDENT,
    MEMORY_DATE_FOLDER_FORMAT,
    MEMORY_TYPE_EPISODIC,
    MEMORY_TYPES_DIR_MAP,
    UNKNOWN_SESSION_ID,
)


class TypedMarkdownStorage(BaseStorage):
    """
    BaseStorage implementation using typed .md files and a manifest index.

    Directory layout::

        {memory_root}/
          conversations/{session_id}/{YYYY-MM-DD}/{key}.md   # episodic
          knowledge/{topic_or_general}/{key}.md               # semantic
          procedures/{topic_or_general}/{key}.md              # procedural
          manifest.json                                       # key → relative path

    **Metadata is NOT stored in .md files.**  The ``.md`` file contains only
    the plain text content.  Metadata (type, topic, timestamp, content_hash,
    session_id, etc.) is managed exclusively by ``JSONIndexer`` in
    ``index.json``.  As a result, ``load()`` always returns ``metadata={}``
    by design — callers that need metadata must query the indexer directly.
    This is an intentional separation of concerns: storage handles raw content,
    the indexer handles structured metadata and search.
    """

    MEMORY_TYPES: Dict[str, str] = MEMORY_TYPES_DIR_MAP

    def __init__(self, memory_root: str) -> None:
        self.memory_root = memory_root
        self._lock = threading.Lock()
        os.makedirs(memory_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manifest_path(self) -> str:
        """Return the absolute path to the manifest.json file."""
        return os.path.join(self.memory_root, "manifest.json")

    def _load_manifest(self) -> Dict[str, str]:
        """
        Load the manifest from disk.

        Returns:
            Dict mapping logical key → relative path within ``memory_root``.
            Returns an empty dict if the file does not exist or is corrupt.
        """
        path = self._manifest_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self, manifest: Dict[str, str]) -> None:
        """
        Atomically write the manifest to disk via temp-file + ``os.replace``.

        Args:
            manifest: Complete key → relative-path mapping to persist.

        Raises:
            OSError: If the temp file cannot be written or renamed.
        """
        path = self._manifest_path()
        dir_ = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=JSON_INDENT)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _resolve_path(self, key: str, data: StorageRecord) -> str:
        """
        Derive the relative storage path from record metadata.

        Routing rules:
          - ``episodic``   → ``conversations/{session_id}/{YYYY-MM-DD}/{key}.md``
          - ``semantic``   → ``knowledge/{topic|general}/{key}.md``
          - ``procedural`` → ``procedures/{topic|general}/{key}.md``

        For episodic entries, ``session_id`` is taken from
        ``metadata["session_id"]``, falling back to ``"unknown_session"``
        when absent.  ``YYYY-MM-DD`` is derived from ``metadata["timestamp"]``,
        falling back to the current UTC date.

        Args:
            key:  Logical storage key used as the filename stem.
            data: The ``StorageRecord`` whose ``metadata`` drives routing.

        Returns:
            Relative path string (from ``memory_root``) including filename.
        """
        metadata = data.get("metadata", {})
        mem_type = metadata.get("type", MEMORY_TYPE_EPISODIC)
        base_dir = self.MEMORY_TYPES.get(mem_type, MEMORY_TYPES_DIR_MAP[MEMORY_TYPE_EPISODIC])

        if mem_type == MEMORY_TYPE_EPISODIC:
            session_id = (metadata.get("session_id") or "").strip() or UNKNOWN_SESSION_ID
            ts = metadata.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                dt = datetime.now(timezone.utc)
            date_dir = dt.strftime(MEMORY_DATE_FOLDER_FORMAT)
            return os.path.join(base_dir, session_id, date_dir, f"{key}.md")
        else:
            topic = (metadata.get("topic") or "").strip() or DEFAULT_MEMORY_TOPIC
            # Sanitise topic for filesystem use
            safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)
            return os.path.join(base_dir, safe_topic, f"{key}.md")

    # ------------------------------------------------------------------
    # BaseStorage interface
    # ------------------------------------------------------------------

    def save(self, key: str, data: StorageRecord) -> None:
        """
        Persist a record as a ``.md`` file and update the manifest atomically.

        The file is written via temp-file + ``os.replace`` to prevent partial
        writes.  The manifest is updated under ``self._lock`` immediately after.

        Args:
            key:  Unique identifier; used as the markdown filename stem.
            data: ``StorageRecord`` whose ``content`` is written as the file body.

        Raises:
            OSError: If the file or manifest cannot be written.
        """

        with self._lock:
            rel_path = self._resolve_path(key, data)
            abs_path = os.path.join(self.memory_root, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            content = data.get("content", "")
            md_body = f"# {key}\n\n{content}\n"

            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(abs_path), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(md_body)
                os.replace(tmp, abs_path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

            manifest = self._load_manifest()
            manifest[key] = rel_path
            self._save_manifest(manifest)

    def load(self, key: str) -> Optional[StorageRecord]:
        """
        Load a record by looking up its path in the manifest and reading the file.

        Args:
            key: Unique identifier to look up.

        Returns:
            A ``StorageRecord`` with ``id``, ``content``, and an empty
            ``metadata`` dict, or ``None`` if the key is not in the manifest
            or the file has been deleted.
        """
        with self._lock:
            manifest = self._load_manifest()
            rel_path = manifest.get(key)

        if rel_path is None:
            return None

        abs_path = os.path.join(self.memory_root, rel_path)
        if not os.path.exists(abs_path):
            return None

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return None

        # Strip the `# {key}\n\n` header
        lines = raw.split("\n", 2)
        content = lines[2] if len(lines) > 2 else raw

        # metadata is intentionally empty — see class docstring.
        # The canonical source of metadata is JSONIndexer's index.json.
        return {"id": key, "content": content.rstrip("\n"), "metadata": {}}

    def delete(self, key: str) -> bool:
        """
        Delete the ``.md`` file for *key* and remove it from the manifest.

        Args:
            key: Unique identifier of the entry to delete.

        Returns:
            ``True`` if the key existed and was deleted; ``False`` if it was
            not found in the manifest.
        """
        with self._lock:
            manifest = self._load_manifest()
            rel_path = manifest.pop(key, None)
            if rel_path is None:
                return False

            abs_path = os.path.join(self.memory_root, rel_path)
            try:
                os.unlink(abs_path)
            except OSError:
                pass
            self._save_manifest(manifest)
        return True

    def list_keys(self) -> List[str]:
        """
        Return all logical keys currently tracked in the manifest.

        Returns:
            A list of key strings.  Order is not guaranteed.
        """
        with self._lock:
            return list(self._load_manifest().keys())

    def exists(self, key: str) -> bool:
        """
        Return ``True`` if *key* is present in the manifest.

        Args:
            key: Unique identifier to check.

        Returns:
            ``True`` if the key exists in the manifest, ``False`` otherwise.
            Does not verify that the backing ``.md`` file exists on disk.
        """
        with self._lock:
            return key in self._load_manifest()
