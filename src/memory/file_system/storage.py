"""
File system storage implementation for the memory system.

This module provides the FileStorage class, which persists and retrieves data
from the local file system using JSON format.
"""

import os
import json
import logging
from typing import List, Optional
from ..base import BaseStorage
from ..types import StorageRecord

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

    def save(self, key: str, data: StorageRecord) -> None:
        """
        Persist a StorageRecord as a JSON file.

        Args:
            key: Unique identifier for the data.
            data: The StorageRecord to serialise and save.

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

    def load(self, key: str) -> Optional[StorageRecord]:
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

    def list_keys(self) -> List[str]:
        """
        Return all stored keys by scanning .json files in the base directory.

        Returns:
            A list of keys derived from file names (without the .json extension).
        """
        try:
            return [
                f[:-5]  # strip ".json"
                for f in os.listdir(self.base_path)
                if f.endswith(".json") and os.path.isfile(os.path.join(self.base_path, f))
            ]
        except OSError:
            return []

    def delete(self, key: str) -> bool:
        """
        Delete the JSON file for the given key.

        Args:
            key: Unique identifier for the data.

        Returns:
            True if the file was deleted, False if it did not exist.
        """
        path = self._get_path(key)
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.error(f"Error deleting key {key}: {str(exc)}")
            return False
