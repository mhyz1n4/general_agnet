import json
import logging
import redis
from typing import Any, Optional
from ..base import BaseStorage

logger = logging.getLogger(__name__)


class RedisStorage(BaseStorage):
    """
    Redis implementation of BaseStorage for Short-Term Memory.
    Uses Redis Strings with a configurable prefix to store JSON-encoded data.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "mem:",
        default_ttl: Optional[int] = 3600  # Default 1 hour TTL for short-term memory
    ):
        """
        Initialize the Redis storage client.

        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
            password: Redis server password.
            prefix: Prefix to use for all keys stored via this instance.
            default_ttl: Default Time-To-Live in seconds for new keys.
        """
        self.prefix = prefix
        self.default_ttl = default_ttl
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True  # Ensure we get strings back from Redis
            )
            # Test connectivity
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def _get_full_key(self, key: str) -> str:
        """Helper to append prefix to a key."""
        return f"{self.prefix}{key}"

    def save(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Save data to Redis as a JSON string.

        Args:
            key: Unique identifier for the data.
            data: The content to be stored (must be JSON serializable).
            ttl: Optional override for the default TTL. Pass 0 to store without expiry.
        """
        full_key = self._get_full_key(key)
        try:
            json_data = json.dumps(data)
            expiration = ttl if ttl is not None else self.default_ttl

            if expiration:
                self.client.set(full_key, json_data, ex=expiration)
            else:
                self.client.set(full_key, json_data)

            logger.debug(f"Saved key '{full_key}' to Redis (TTL: {expiration}s)")
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data for key {key}: {str(e)}")
            raise
        except redis.RedisError as e:
            logger.error(f"Error saving to Redis: {str(e)}")
            raise

    def load(self, key: str) -> Optional[Any]:
        """
        Load data from Redis and deserialize from JSON.

        Args:
            key: Unique identifier for the data.

        Returns:
            The deserialized data if found, otherwise None.
        """
        full_key = self._get_full_key(key)
        try:
            data = self.client.get(full_key)
            if data is None:
                return None
            return json.loads(data)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to deserialize data from key {full_key}: {str(e)}")
            return None
        except redis.RedisError as e:
            logger.error(f"Error loading from Redis: {str(e)}")
            return None

    def delete(self, key: str) -> bool:
        """
        Explicitly delete a key from Redis.

        Args:
            key: The unique identifier.

        Returns:
            True if the key was deleted, False if it did not exist.
        """
        full_key = self._get_full_key(key)
        try:
            return bool(self.client.delete(full_key))
        except redis.RedisError as e:
            logger.error(f"Error deleting from Redis: {str(e)}")
            return False
