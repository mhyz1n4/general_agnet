"""Marker for integration tests."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as integration tests (require Redis + FS)")
