"""
TypedMarkdownStorage — file-system storage that organises .md files
into a typed directory structure with a manifest for O(1) key lookup.

Directory layout:
    {memory_root}/
      conversations/YYYY-MM/{key}.md      # episodic
      knowledge/{topic_or_general}/{key}.md    # semantic
      procedures/{topic_or_general}/{key}.md   # procedural
      manifest.json                        # key → relative path
"""

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..base import BaseStorage


class TypedMarkdownStorage(BaseStorage):
    """BaseStorage implementation using typed .md files and a manifest index."""

    MEMORY_TYPES: Dict[str, str] = {
        "episodic": "conversations",
        "semantic": "knowledge",
        "procedural": "procedures",
    }

    def __init__(self, memory_root: str) -> None:
        self.memory_root = memory_root
        self._lock = threading.Lock()
        os.makedirs(memory_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manifest_path(self) -> str:
        return os.path.join(self.memory_root, "manifest.json")

    def _load_manifest(self) -> Dict[str, str]:
        path = self._manifest_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self, manifest: Dict[str, str]) -> None:
        path = self._manifest_path()
        dir_ = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _resolve_path(self, key: str, data: Dict[str, Any]) -> str:
        """Derive relative path from data['metadata'] type/topic/timestamp."""
        metadata = data.get("metadata", {})
        mem_type = metadata.get("type", "episodic")
        base_dir = self.MEMORY_TYPES.get(mem_type, "conversations")

        if mem_type == "episodic":
            ts = metadata.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                dt = datetime.now(timezone.utc)
            month_dir = dt.strftime("%Y-%m")
            return os.path.join(base_dir, month_dir, f"{key}.md")
        else:
            topic = (metadata.get("topic") or "").strip() or "general"
            # Sanitise topic for filesystem use
            safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)
            return os.path.join(base_dir, safe_topic, f"{key}.md")

    # ------------------------------------------------------------------
    # BaseStorage interface
    # ------------------------------------------------------------------

    def save(self, key: str, data: Any) -> None:
        """Write .md file; update manifest atomically; create subdirs as needed."""
        if not isinstance(data, dict):
            data = {"id": key, "content": str(data), "metadata": {}}

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

    def load(self, key: str) -> Optional[Any]:
        """Lookup manifest → read .md → return structured dict or None."""
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

        return {"id": key, "content": content.rstrip("\n"), "metadata": {}}

    def delete(self, key: str) -> bool:
        """Delete .md file and remove from manifest. Returns True if existed."""
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

    def exists(self, key: str) -> bool:
        manifest = self._load_manifest()
        return key in manifest
