"""
Pre-session hook — validates system prerequisites before starting a chat session.

V1.2 checks (in order):
  1. Memory root is writable and ReMeLight directory structure exists (abort on
     failure).  Creates ``memory/``, ``MEMORY.md``, and ``memory.md`` if missing.
  2. LLM API key is valid via a lightweight ping (abort on AuthenticationError,
     degrade on RateLimitError).
  3. Log rotation (delete oldest log files if logs/ exceeds 500MB).
"""

import os
import shutil
from typing import Optional

from src.client.base import LLMClient
from src.constants import (
    DEFAULT_AGENT_MODEL,
    LLM_PING_MAX_TOKENS,
    LOG_DIR_MAX_BYTES,
    SESSION_LOG_SUBDIR_PREFIX,
    WRITE_TEST_CONTENT,
    WRITE_TEST_FILENAME,
)
from src.logging_config import get_logger
from .base import BaseHook, HookResult

logger = get_logger(__name__)


class PreSessionHook(BaseHook):
    """Validate system health before a chat session starts."""

    def __init__(
        self,
        memory_root: str,
        llm_client: Optional[LLMClient] = None,
        llm_model: str = DEFAULT_AGENT_MODEL,
        log_dir: str = "./logs",
    ) -> None:
        """
        Initialise the pre-session hook.

        Args:
            memory_root: Path to the memory root directory.
            llm_client:  Optional LLM client for API key validation.
            llm_model:   Model identifier for the LLM ping call.
            log_dir:     Path to the log directory for rotation.
        """
        self.memory_root = memory_root
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.log_dir = log_dir

    def run(self, **kwargs: object) -> HookResult:
        """Execute all pre-session health checks."""
        logger.debug("pre_session: starting health checks")
        issues: list[str] = []
        degraded = False

        # 1. Filesystem check
        fs_result = self._check_filesystem()
        if not fs_result:
            return HookResult(
                success=False,
                message="Memory root is not writable. Cannot start session.",
            )

        # 2. LLM check
        llm_result = self._check_llm()
        if llm_result == "auth_error":
            return HookResult(
                success=False,
                message="LLM API key is invalid. Cannot start session.",
            )
        if llm_result == "rate_limit":
            degraded = True
            issues.append("LLM rate-limited at startup — may affect first response")

        # 3. Log rotation
        self._rotate_logs()

        msg = "Session ready"
        if issues:
            msg += " (degraded: " + "; ".join(issues) + ")"

        logger.debug(
            "pre_session: done",
            extra={"data": {"degraded": degraded, "issues": issues}},
        )
        return HookResult(success=True, message=msg, degraded=degraded)

    def _check_filesystem(self) -> bool:
        """
        Verify that the memory root is writable and ReMeLight directories exist.

        Creates the memory root, the ``memory/`` subdirectory (where ``.md``
        entries are stored and watched), and the ``MEMORY.md`` / ``memory.md``
        index files if they are missing.

        Returns:
            ``True`` if the directory is writable; ``False`` otherwise.
        """
        try:
            os.makedirs(self.memory_root, exist_ok=True)
            test_path = os.path.join(self.memory_root, WRITE_TEST_FILENAME)
            with open(test_path, "w") as f:
                f.write(WRITE_TEST_CONTENT)
            os.unlink(test_path)

            # ReMeLight file watcher expects these paths to exist on startup.
            memory_subdir = os.path.join(self.memory_root, "memory")
            os.makedirs(memory_subdir, exist_ok=True)

            for index_file in ("MEMORY.md", "memory.md"):
                path = os.path.join(self.memory_root, index_file)
                if not os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as f:
                        pass  # empty file — placeholder for file watcher
                    logger.debug(
                        "pre_session: created missing index file",
                        extra={"data": {"path": path}},
                    )

            logger.debug("pre_session: filesystem check passed")
            return True
        except OSError as exc:
            logger.error(
                "pre_session: filesystem not writable",
                extra={"data": {"path": self.memory_root, "error": str(exc), "alert": True}},
            )
            return False

    def _check_llm(self) -> str:
        """
        Perform a lightweight LLM API ping to validate the API key.

        Returns:
            ``'ok'``         — API key is valid (or no client configured).
            ``'auth_error'`` — Authentication failed; session cannot start.
            ``'rate_limit'`` — Rate-limited at startup; session can continue
                               in degraded mode.
        """
        if self.llm_client is None:
            return "ok"
        try:
            self.llm_client.completion(
                messages=[{"role": "user", "content": "ping"}],
                model=self.llm_model,
                max_tokens=LLM_PING_MAX_TOKENS,
            )
            logger.debug("pre_session: LLM check passed")
            return "ok"
        except Exception as exc:
            exc_name = type(exc).__name__
            if "Authentication" in exc_name or "auth" in exc_name.lower():
                logger.error(
                    "pre_session: LLM authentication failed",
                    extra={"data": {"error": str(exc), "alert": True}},
                )
                return "auth_error"
            if "RateLimit" in exc_name or "rate" in exc_name.lower():
                logger.warning(
                    "pre_session: LLM rate-limited",
                    extra={"data": {"error": str(exc)}},
                )
                return "rate_limit"
            logger.warning(
                "pre_session: LLM ping failed (non-fatal)",
                extra={"data": {"error": str(exc)}},
            )
            return "ok"

    def _rotate_logs(self) -> None:
        """
        Evict old log entries when the log directory exceeds the size cap.

        Entries are sorted by modification time (oldest first) and removed
        until the total size is at or below ``LOG_DIR_MAX_BYTES``.
        """
        if not os.path.isdir(self.log_dir):
            return
        try:
            entries: list[tuple[float, str, int, bool]] = []
            total = 0

            for name in os.listdir(self.log_dir):
                path = os.path.join(self.log_dir, name)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    entries.append((os.path.getmtime(path), path, size, False))
                    total += size
                elif os.path.isdir(path) and name.startswith(SESSION_LOG_SUBDIR_PREFIX):
                    size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fnames in os.walk(path)
                        for f in fnames
                    )
                    entries.append((os.path.getmtime(path), path, size, True))
                    total += size

            if total <= LOG_DIR_MAX_BYTES:
                return

            entries.sort()
            for _, path, size, is_dir in entries:
                if total <= LOG_DIR_MAX_BYTES:
                    break
                if is_dir:
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
                total -= size
                logger.info(
                    "pre_session: rotated old logs",
                    extra={"data": {"path": path, "is_dir": is_dir}},
                )
        except Exception as exc:
            logger.warning(
                "pre_session: log rotation failed",
                extra={"data": {"error": str(exc)}},
            )
