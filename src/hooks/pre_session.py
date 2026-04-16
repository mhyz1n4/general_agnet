"""
Pre-session hook — validates system prerequisites before starting a chat session.

Checks (in order):
  1. Memory root is writable (abort on failure).
  2. Redis connectivity (degrade on failure).
  3. LLM API key is valid via a lightweight ping (abort on AuthenticationError,
     degrade on RateLimitError).
  4. Log rotation (delete oldest log files if logs/ exceeds 500MB).

Returns a HookResult with degraded=True if the session can continue in a
reduced-capability mode (no Redis), or success=False if it must abort.
"""

import logging
import os
import shutil
from typing import Optional

import redis as redis_lib

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
        redis_client: Optional[redis_lib.Redis] = None,
        llm_client: Optional[LLMClient] = None,
        llm_model: str = DEFAULT_AGENT_MODEL,
        log_dir: str = "./logs",
    ) -> None:
        self.memory_root = memory_root
        self.redis_client = redis_client
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.log_dir = log_dir

    def run(self, **kwargs: object) -> HookResult:
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

        # 2. Redis check
        redis_ok = self._check_redis()
        if not redis_ok:
            degraded = True
            issues.append("Redis unavailable — session memory disabled")

        # 3. LLM check
        llm_result = self._check_llm()
        if llm_result == "auth_error":
            return HookResult(
                success=False,
                message="LLM API key is invalid. Cannot start session.",
            )
        if llm_result == "rate_limit":
            degraded = True
            issues.append("LLM rate-limited at startup — may affect first response")

        # 4. Log rotation
        self._rotate_logs()

        msg = "Session ready"
        if issues:
            msg += " (degraded: " + "; ".join(issues) + ")"

        logger.debug(
            "pre_session: done",
            extra={"data": {"degraded": degraded, "issues": issues}},
        )
        return HookResult(success=True, message=msg, degraded=degraded)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_filesystem(self) -> bool:
        """
        Verify that the memory root directory is writable.

        Creates the directory if it does not exist, then performs a
        write-and-delete test to confirm the filesystem is not read-only.

        Returns:
            ``True`` if the directory is writable; ``False`` otherwise.
        """
        try:
            os.makedirs(self.memory_root, exist_ok=True)
            test_path = os.path.join(self.memory_root, WRITE_TEST_FILENAME)
            with open(test_path, "w") as f:
                f.write(WRITE_TEST_CONTENT)
            os.unlink(test_path)
            logger.debug("pre_session: filesystem check passed")
            return True
        except OSError as exc:
            logger.error(
                "pre_session: filesystem not writable",
                extra={"data": {"path": self.memory_root, "error": str(exc), "alert": True}},
            )
            return False

    def _check_redis(self) -> bool:
        """
        Ping Redis to verify connectivity.

        If no ``redis_client`` is configured, Redis is treated as absent
        rather than failed — the method returns ``True`` immediately.

        Returns:
            ``True`` if Redis is reachable or not configured; ``False`` on
            any connection or command error.
        """
        if self.redis_client is None:
            return True  # Redis not configured — not a failure
        try:
            self.redis_client.ping()
            logger.debug("pre_session: Redis check passed")
            return True
        except Exception as exc:
            logger.warning(
                "pre_session: Redis unavailable",
                extra={"data": {"error": str(exc), "alert": True}},
            )
            return False

    def _check_llm(self) -> str:
        """
        Perform a lightweight LLM API ping to validate the API key.

        If no ``llm_client`` is configured the check is skipped and ``'ok'``
        is returned.  Exception classification is based on the exception
        class name so it works across different provider SDKs.

        Returns:
            ``'ok'``         — API key is valid (or no client configured).
            ``'auth_error'`` — Authentication failed; session cannot start.
            ``'rate_limit'`` — Rate-limited at startup; session can continue
                               in degraded mode.
        """
        if self.llm_client is None:
            return "ok"
        try:
            # Lightweight ping — one token completion
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

        Scans ``self.log_dir`` for:
          - Session subdirectories (``session_*``) — deleted as a whole unit
            to preserve per-session log integrity.
          - Legacy root-level files — deleted individually (backward compat).

        Entries are sorted by modification time (oldest first) and removed
        until the total size is at or below ``LOG_DIR_MAX_BYTES``.  Errors
        are logged at WARNING and do not abort the session.
        """
        if not os.path.isdir(self.log_dir):
            return
        try:
            entries: list[tuple[float, str, int, bool]] = []  # (mtime, path, size, is_dir)
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

            entries.sort()  # oldest first
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
