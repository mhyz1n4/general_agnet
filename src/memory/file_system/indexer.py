"""
File system indexing implementation for the memory system.

This module provides the JSONIndexer class, which maintains a central JSON file
for indexing content with metadata and keywords.
"""

import os
import json
import logging
import threading
import re
from typing import Dict, Any, List
from ..base import BaseIndexer

logger = logging.getLogger(__name__)


class JSONIndexer(BaseIndexer):
    """
    File system based indexer using a central JSON file.

    This class manages an index that maps keys to metadata and keywords extracted
    from the content, enabling efficient keyword-based retrieval.
    Thread-safe updates are ensured through a mutex lock.
    """

    def __init__(self, index_path: str):
        """
        Initialize the JSONIndexer with a target index file path.

        Args:
            index_path: Absolute or relative path to the JSON index file.
        """
        self.index_path: str = index_path
        self.lock: threading.Lock = threading.Lock()
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """
        Check if the index file exists, creating it with an empty structure if not.

        Returns:
            None
        """
        if not os.path.exists(self.index_path):
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_index(self) -> Dict[str, Any]:
        """
        Load the index data from the JSON file.

        Returns:
            A dictionary representing the index structure. Returns an empty dict
            if the file is missing or corrupted.
        """
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error(f"Error loading index at {self.index_path}, returning empty.")
            return {}

    def _save_index(self, index_data: Dict[str, Any]) -> None:
        """
        Save index data to the JSON file atomically using a temporary file.

        Args:
            index_data: The dictionary structure to be saved.

        Returns:
            None

        Raises:
            Exception: If file writing or atomic replacement fails.
        """
        # Temporary file for atomic-like write
        tmp_path: str = f"{self.index_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.index_path)
        except Exception as e:
            logger.error(f"Error saving index to {self.index_path}: {str(e)}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def add(self, key: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Index new content by extracting keywords and updating the index file.

        Args:
            key: Unique identifier for the content.
            content: Text content from which keywords will be extracted.
            metadata: Additional structured data associated with the key.

        Returns:
            None
        """
        with self.lock:
            index: Dict[str, Any] = self._load_index()
            index[key] = {
                "metadata": metadata,
                "keywords": self._extract_keywords(content)
            }
            self._save_index(index)

    def update(self, key: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Update an existing index entry. (Currently a wrapper for add).

        Args:
            key: Unique identifier for the content.
            content: The updated text content.
            metadata: The updated metadata.

        Returns:
            None
        """
        self.add(key, content, metadata)

    def _extract_keywords(self, content: str) -> List[str]:
        """
        Simple keyword extraction from text.

        Cleans the text, tokenizes it, and returns unique words with length > 3 characters.

        Args:
            content: The text string to process.

        Returns:
            A list of extracted keywords.
        """
        words: List[str] = re.findall(r"\w+", content.lower())
        return list(set(w for w in words if len(w) > 3))
