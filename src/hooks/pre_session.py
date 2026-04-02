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
from typing import Any, Optional

from src.logging_config import get_logger
from .base import BaseHook, HookResult

logger = get_logger(__name__)

_LOG_DIR_MAX_BYTES = 500 * 1024 * 1024  # 500 MB


class PreSessionHook(BaseHook):
    """Validate system health before a chat session starts."""

    def __init__(
        self,
        memory_root: str,
        redis_client: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        log_dir: str = "./logs",
    ) -> None:
        self.memory_root = memory_root
        self.redis_client = redis_client
        self.llm_client = llm_client
        self.log_dir = log_dir

    def run(self, **kwargs: Any) -> HookResult:
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
        try:
            os.makedirs(self.memory_root, exist_ok=True)
            test_path = os.path.join(self.memory_root, ".write_test")
            with open(test_path, "w") as f:
                f.write("ok")
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
        """Returns 'ok', 'auth_error', or 'rate_limit'."""
        if self.llm_client is None:
            return "ok"
        try:
            # Lightweight ping — one token completion
            self.llm_client.completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
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
        """Delete oldest log files if the log directory exceeds 500 MB."""
        if not os.path.isdir(self.log_dir):
            return
        try:
            files = [
                (os.path.getmtime(os.path.join(self.log_dir, f)), os.path.join(self.log_dir, f))
                for f in os.listdir(self.log_dir)
                if os.path.isfile(os.path.join(self.log_dir, f))
            ]
            total = sum(os.path.getsize(p) for _, p in files)
            if total <= _LOG_DIR_MAX_BYTES:
                return

            files.sort()  # oldest first
            for _, path in files:
                if total <= _LOG_DIR_MAX_BYTES:
                    break
                size = os.path.getsize(path)
                os.unlink(path)
                total -= size
                logger.info("pre_session: deleted old log file", extra={"data": {"path": path}})
        except Exception as exc:
            logger.warning(
                "pre_session: log rotation failed",
                extra={"data": {"error": str(exc)}},
            )
