"""
Unit tests for RedisStorage using a mocked Redis client.

These tests validate the BaseStorage interface contract and RedisStorage-specific
behaviour (TTL handling, key prefixing, graceful error recovery) without requiring
a live Redis instance.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from src.memory.redis.storage import RedisStorage


@pytest.fixture
def redis_storage():
    """RedisStorage instance backed by a MagicMock Redis client."""
    with patch("src.memory.redis.storage.redis.Redis") as mock_cls:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_cls.return_value = mock_client
        storage = RedisStorage(host="localhost", port=6379)
        yield storage, mock_client


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

def test_save_uses_default_ttl(redis_storage):
    storage, client = redis_storage
    storage.save("msg1", {"content": "hello"})
    client.set.assert_called_once_with(
        "mem:msg1",
        json.dumps({"content": "hello"}),
        ex=3600,
    )


def test_save_with_explicit_ttl(redis_storage):
    storage, client = redis_storage
    storage.save("msg1", {"content": "hello"}, ttl=60)
    client.set.assert_called_once_with(
        "mem:msg1",
        json.dumps({"content": "hello"}),
        ex=60,
    )


def test_save_with_zero_ttl_stores_without_expiry(redis_storage):
    """ttl=0 should store permanently (no ex= argument)."""
    storage, client = redis_storage
    storage.save("msg1", {"content": "hello"}, ttl=0)
    client.set.assert_called_once_with(
        "mem:msg1",
        json.dumps({"content": "hello"}),
    )


def test_save_applies_custom_prefix(redis_storage):
    storage, client = redis_storage
    storage.prefix = "session:"
    storage.save("msg2", "data")
    client.set.assert_called_once_with(
        "session:msg2",
        json.dumps("data"),
        ex=3600,
    )


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def test_load_returns_deserialized_data(redis_storage):
    storage, client = redis_storage
    client.get.return_value = json.dumps({"content": "hello"})
    result = storage.load("msg1")
    assert result == {"content": "hello"}
    client.get.assert_called_once_with("mem:msg1")


def test_load_returns_none_for_missing_key(redis_storage):
    storage, client = redis_storage
    client.get.return_value = None
    assert storage.load("missing") is None


def test_load_returns_none_on_corrupt_json(redis_storage):
    storage, client = redis_storage
    client.get.return_value = "{ invalid json }"
    assert storage.load("bad_key") is None


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

def test_delete_returns_true_when_key_exists(redis_storage):
    storage, client = redis_storage
    client.delete.return_value = 1
    assert storage.delete("msg1") is True
    client.delete.assert_called_once_with("mem:msg1")


def test_delete_returns_false_when_key_missing(redis_storage):
    storage, client = redis_storage
    client.delete.return_value = 0
    assert storage.delete("missing") is False
