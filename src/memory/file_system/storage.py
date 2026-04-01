"""
File system storage implementation for the memory system.

This module provides the FileStorage class, which persists and retrieves data
from the local file system using JSON format.
"""

import os
import json
import logging
from typing import Any, Optional
from ..base import BaseStorage

logger = logging.getLogger(__name__)


class FileStorage(BaseStorage):
    """
    File system based storage implementation.

    This class manages the persistence of data onto the local file system.
    Each entry is stored as an individual JSON file within a specified base directory.
    """

    def __init__(self, base_path: str):
        """
        Initialize the FileStorage with a base directory.

        Args:
            base_path: Root directory where all files will be stored.
        """
        self.base_path: str = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_path(self, key: str) -> str:
        """
        Convert a unique key into a sanitized file path.

        Args:
            key: Unique identifier for the data.

        Returns:
            Absolute file path within the base storage directory.
        """
        # Sanitize key to avoid path traversal
        safe_key: str = "".join([c for c in key if c.isalnum() or c in ("-", "_")]).rstrip()
        return os.path.join(self.base_path, f"{safe_key}.json")

    def save(self, key: str, data: Any) -> None:
        """
        Persist data as a JSON file.

        Args:
            key: Unique identifier for the data.
            data: The object to be serialized and saved.

        Returns:
            None

        Raises:
            Exception: If file writing or directory creation fails.
        """
        path: str = self._get_path(key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Successfully saved data to {path}")
        except Exception as e:
            logger.error(f"Error saving data to {path}: {str(e)}")
            raise

    def load(self, key: str) -> Optional[Any]:
        """
        Load data from its corresponding JSON file.

        Args:
            key: Unique identifier for the data.

        Returns:
            The deserialized data if found, otherwise None.

        Raises:
            Exception: If the file exists but cannot be read or parsed.
        """
        path: str = self._get_path(key)
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}")
            raise
