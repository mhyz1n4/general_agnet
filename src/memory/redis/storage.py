import json
import logging
import redis
from typing import Optional
from ..base import BaseStorage
from ..types import StorageRecord
from src.constants import REDIS_DEFAULT_HOST, REDIS_DEFAULT_PORT, REDIS_DEFAULT_PREFIX, REDIS_DEFAULT_TTL

logger = logging.getLogger(__name__)


class RedisStorage(BaseStorage):
    """
    Redis implementation of BaseStorage for Short-Term Memory.
    Uses Redis Strings with a configurable prefix to store JSON-encoded data.
    """

    def __init__(
        self,
        host: str = REDIS_DEFAULT_HOST,
        port: int = REDIS_DEFAULT_PORT,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = REDIS_DEFAULT_PREFIX,
        default_ttl: Optional[int] = REDIS_DEFAULT_TTL,
        client: Optional[redis.Redis] = None,
    ):
        """
        Initialize the Redis storage client.

        Args:
            host: Redis server host. Ignored when *client* is provided.
            port: Redis server port. Ignored when *client* is provided.
            db: Redis database number. Ignored when *client* is provided.
            password: Redis server password. Ignored when *client* is provided.
            prefix: Prefix to use for all keys stored via this instance.
            default_ttl: Default Time-To-Live in seconds for new keys.
            client: Optional pre-built ``redis.Redis`` instance. When supplied,
                connection parameters above are ignored and no ``ping`` is
                issued — the caller is responsible for verifying connectivity.
        """
        self.prefix = prefix
        self.default_ttl = default_ttl
        if client is not None:
            self.client = client
        else:
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
        """
        Prepend the instance prefix to *key* to produce the full Redis key.

        Args:
            key: Logical storage key (without prefix).

        Returns:
            The full Redis key string used for all Redis operations.
        """
        return f"{self.prefix}{key}"

    def save(self, key: str, data: StorageRecord, ttl: Optional[int] = None) -> None:
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

    def load(self, key: str) -> Optional[StorageRecord]:
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

    def list_keys(self) -> list[str]:
        """
        Return all logical keys stored under this instance's prefix.

        Scans Redis for all keys matching ``{prefix}*`` and strips the
        prefix before returning.  Returns an empty list on any Redis error.

        Returns:
            A list of logical key strings (prefix removed).
        """
        try:
            full_keys: list[str] = list(self.client.keys(f"{self.prefix}*"))
            return [k[len(self.prefix):] for k in full_keys]
        except Exception as exc:
            logger.error(f"Error listing Redis keys: {str(exc)}")
            return []

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
