"""
Top-level pytest configuration and shared fixtures.

Session-scoped fixtures:
  - redis_server: starts redis-server on port 6380, tears down after session.
  - test_memory_root: session-scoped temp dir for filesystem memory.

Per-test fixture:
  - clean_redis: flushes all keys in the test Redis before and after each test.
"""

import shutil
import subprocess
import time

import pytest

REDIS_TEST_PORT = 6380


@pytest.fixture(scope="session")
def redis_server():
    """Start a local redis-server on port 6380 for the test session."""
    proc = subprocess.Popen(
        [
            "redis-server",
            "--port",
            str(REDIS_TEST_PORT),
            "--loglevel",
            "warning",
            "--save",
            "",  # disable RDB snapshots
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for Redis to be ready
    try:
        import redis as _redis

        client = _redis.Redis(port=REDIS_TEST_PORT)
        for _ in range(30):
            try:
                client.ping()
                break
            except _redis.ConnectionError:
                time.sleep(0.1)
        else:
            proc.terminate()
            proc.wait()
            pytest.skip("Could not connect to test Redis on port 6380")
    except ImportError:
        proc.terminate()
        proc.wait()
        pytest.skip("redis package not installed")

    yield proc

    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def test_memory_root(tmp_path_factory):
    """Session-scoped temp dir for memory storage."""
    root = tmp_path_factory.mktemp("memory")
    yield str(root)
    shutil.rmtree(str(root), ignore_errors=True)


@pytest.fixture
def clean_redis(redis_server):
    """Flush all test Redis keys before and after each test."""
    import redis

    client = redis.Redis(port=REDIS_TEST_PORT, decode_responses=True)
    client.flushall()
    yield client
    client.flushall()
