"""
Unit tests for init_session_logging.

Each test runs in an isolated tmp directory and resets the global
``_configured`` flag so tests don't bleed into each other.
Verifies file creation, idempotency, logger routing (memory/agent vs root),
propagation suppression, and JSON record format.
"""

import json
import logging
import os
from pathlib import Path

import pytest

import src.logging_config as logging_config_module
from src.logging_config import init_session_logging, get_logger
from src.constants import DEFAULT_APP_LOG_FILENAME, LOG_COMPONENT_FILES


@pytest.fixture(autouse=True)
def reset_logging(tmp_path: Path) -> None:
    """
    Reset logging state before every test.

    Clears the _configured guard so init_session_logging runs fresh each time,
    and removes all file handlers added by previous tests to avoid file-lock
    interference.
    """
    logging_config_module._configured = False

    yield

    loggers_to_clean = [logging.getLogger()] + [
        logging.getLogger(ns) for ns in LOG_COMPONENT_FILES
    ]
    for log in loggers_to_clean:
        for h in list(log.handlers):
            h.close()
            log.removeHandler(h)
        log.propagate = True

    logging_config_module._configured = False


class TestInitSessionLogging:
    """init_session_logging creates structured log files and routes by component namespace."""

    def test_creates_session_dir(self, tmp_path: Path) -> None:
        """The session directory must be created if it does not already exist."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir)
        assert os.path.isdir(session_dir)

    def test_app_log_file_created(self, tmp_path: Path) -> None:
        """The main application log file must exist after init_session_logging."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir)
        assert os.path.exists(os.path.join(session_dir, DEFAULT_APP_LOG_FILENAME))

    def test_memory_log_file_created(self, tmp_path: Path) -> None:
        """The memory-specific JSONL log file must be created on init."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir)
        assert os.path.exists(os.path.join(session_dir, "memory.jsonl"))

    def test_agent_log_file_created(self, tmp_path: Path) -> None:
        """The agent-specific JSONL log file must be created on init."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir)
        assert os.path.exists(os.path.join(session_dir, "agent.jsonl"))

    def test_idempotent_second_call_is_noop(self, tmp_path: Path) -> None:
        """A second call to init_session_logging must not add duplicate handlers."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir)
        root_handler_count: int = len(logging.getLogger().handlers)
        init_session_logging(str(tmp_path / "other_dir"))
        assert len(logging.getLogger().handlers) == root_handler_count

    def test_root_logger_writes_to_app_log(self, tmp_path: Path) -> None:
        """Events logged via an unconfigured namespace must appear in the app log."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir, log_level="DEBUG")
        logger = get_logger("some.other.component")
        logger.info("root test event")
        app_log: str = os.path.join(session_dir, DEFAULT_APP_LOG_FILENAME)
        content: str = open(app_log).read()
        assert "root test event" in content

    def test_memory_logger_writes_to_memory_log(self, tmp_path: Path) -> None:
        """Events from the src.memory namespace must appear in memory.jsonl."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir, log_level="DEBUG")
        logger = get_logger("src.memory.manager")
        logger.info("memory event")
        memory_log: str = os.path.join(session_dir, "memory.jsonl")
        content: str = open(memory_log).read()
        assert "memory event" in content

    def test_memory_logger_does_not_write_to_app_log(self, tmp_path: Path) -> None:
        """Memory-namespace events must NOT propagate to the root app log (propagation=False)."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir, log_level="DEBUG")
        logger = get_logger("src.memory.manager")
        logger.info("memory only event")
        app_log: str = os.path.join(session_dir, DEFAULT_APP_LOG_FILENAME)
        content: str = open(app_log).read()
        assert "memory only event" not in content

    def test_agent_logger_writes_to_agent_log(self, tmp_path: Path) -> None:
        """Events from the src.agents namespace must appear in agent.jsonl."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir, log_level="DEBUG")
        logger = get_logger("src.agents.memorize")
        logger.info("agent event")
        agent_log: str = os.path.join(session_dir, "agent.jsonl")
        content: str = open(agent_log).read()
        assert "agent event" in content

    def test_tools_logger_writes_to_agent_log(self, tmp_path: Path) -> None:
        """Events from the src.tools namespace must also be routed to agent.jsonl."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir, log_level="DEBUG")
        logger = get_logger("src.tools.memorize")
        logger.info("tool event")
        agent_log: str = os.path.join(session_dir, "agent.jsonl")
        content: str = open(agent_log).read()
        assert "tool event" in content

    def test_agent_logger_does_not_write_to_app_log(self, tmp_path: Path) -> None:
        """Agent-namespace events must NOT propagate to the root app log (propagation=False)."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir, log_level="DEBUG")
        logger = get_logger("src.agents.memorize")
        logger.info("agent only event")
        app_log: str = os.path.join(session_dir, DEFAULT_APP_LOG_FILENAME)
        content: str = open(app_log).read()
        assert "agent only event" not in content

    def test_log_records_are_valid_json(self, tmp_path: Path) -> None:
        """Every line in the JSONL log must be valid JSON with required fields."""
        session_dir: str = str(tmp_path / "session_abc_2026-04-05")
        init_session_logging(session_dir, log_level="DEBUG")
        get_logger("src.memory.storage").info("json check", extra={"data": {"k": "v"}})
        line: str = open(os.path.join(session_dir, "memory.jsonl")).readline()
        obj: dict = json.loads(line)
        assert obj["event"] == "json check"
        assert obj["data"] == {"k": "v"}
        assert "timestamp" in obj
        assert "level" in obj
        assert "component" in obj
