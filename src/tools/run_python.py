"""
Python code execution tool (V1.1 M6) — subprocess + rlimit, feature-flagged OFF.

Security model
--------------
This tool runs arbitrary Python in a child interpreter.  It is **not** a
security sandbox: rlimits prevent accidental resource exhaustion (fork bombs,
memory blowups, unbounded file writes) but a determined attacker with control
over the ``code`` parameter can still exfiltrate data via the network, read
files the parent process can read, or otherwise abuse the container.

The feature flag (``Config.enable_code_exec``) defaults to False.  The
confirmation hook (``@requires_confirmation``) adds a second gate.  Anyone
who flips the flag on in a deployment accepts arbitrary-code-execution risk.
Docker/E2B/Firecracker-based sandboxes are documented in ``design_v1_1.md``
§7 as V1.2/V2 upgrades.
"""

from __future__ import annotations

import resource
import subprocess
import sys
from typing import Optional

from strands import tool

from src.constants import CHARS_PER_TOKEN
from src.logging_config import get_logger
from src.tools.confirm import requires_confirmation
from src.tools.envelope import ToolResult, err, ok

logger = get_logger(__name__)

# Per-call subprocess limits.  Treated like rlimit defaults — fixed
# implementation details, not operator knobs.  If a deployment ever needs to
# tune these, promote to Config; until then, hard-coded keeps the surface area
# small and the values close to the code that enforces them.
_DEFAULT_TIMEOUT_SECONDS = 10
_TIMEOUT_CAP_SECONDS = 30        # never let the model override past this
_MEMORY_BYTES = 256 * 1024 * 1024  # 256 MiB address space
_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MiB per-file write cap
_MAX_OUTPUT_BYTES = 64 * 1024     # truncate stdout/stderr above this
_CONFIRM_DESCRIPTION = "execute Python code on the host"


def _make_preexec(memory_bytes: int, cpu_seconds: int, file_size_bytes: int):
    """
    Build a ``preexec_fn`` that applies rlimits in the child before exec.

    Values match the plan's defaults: 256MB address space, 15s CPU, 1MB
    per-file size, 32 open files.
    """

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))

    return _apply


def create_run_python_tool(enable: bool):
    """
    Build the ``run_python`` tool, or return ``None`` when the feature is off.

    Returning ``None`` when the flag is off lets ``main.py`` skip registering
    the tool entirely — the agent never sees it, so the LLM cannot try to
    call it and trip the confirmation hook on every turn.
    """
    if not enable:
        logger.info("run_python: disabled by config; skipping registration")
        return None

    @tool
    @requires_confirmation(_CONFIRM_DESCRIPTION)
    def run_python(code: str, timeout_s: Optional[int] = None) -> ToolResult:
        """
        Run a snippet of Python code in a subprocess with strict resource limits.

        Do NOT use this tool for tasks the agent can reason about directly.
        It is intended for one-off computations, data munging, or small
        scripts the user explicitly asks to execute.

        Args:
            code:      The Python source to execute. Must be non-empty.
            timeout_s: Optional per-call wall-clock timeout (seconds).

        Returns:
            ``ToolResult`` envelope. ``data`` carries ``{stdout, stderr,
            returncode, truncated}`` on success or failure; ``ok`` reflects
            the child's exit status (``returncode == 0``).
        """
        if not code.strip():
            return err("code cannot be empty")

        timeout = min(int(timeout_s or _DEFAULT_TIMEOUT_SECONDS), _TIMEOUT_CAP_SECONDS)
        preexec = _make_preexec(
            memory_bytes=_MEMORY_BYTES,
            # CPU rlimit slightly higher than wall-clock so wall-timeout
            # surfaces first with a clean ``TimeoutExpired`` error.
            cpu_seconds=timeout + 5,
            file_size_bytes=_FILE_SIZE_BYTES,
        )

        logger.info(
            "run_python: executing",
            extra={"data": {"code_len": len(code), "timeout_s": timeout}},
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "run_python: timeout",
                extra={"data": {"timeout_s": timeout}},
            )
            return err(f"timed out after {timeout}s", timeout_s=timeout)
        except Exception as exc:
            logger.error(
                "run_python: subprocess failed",
                extra={"data": {"error": str(exc)}},
            )
            return err(f"subprocess error: {exc}")

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        truncated = False
        if len(stdout) > _MAX_OUTPUT_BYTES:
            stdout = stdout[:_MAX_OUTPUT_BYTES]
            truncated = True
        if len(stderr) > _MAX_OUTPUT_BYTES:
            stderr = stderr[:_MAX_OUTPUT_BYTES]
            truncated = True

        data = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": completed.returncode,
            "truncated": truncated,
        }
        meta = {
            "timeout_s": timeout,
            "output_tokens_est": (len(stdout) + len(stderr)) // CHARS_PER_TOKEN,
        }
        logger.info(
            "run_python: completed",
            extra={"data": {
                "returncode": completed.returncode,
                "stdout_bytes": len(stdout),
                "stderr_bytes": len(stderr),
                "truncated": truncated,
                "output_tokens_est": meta["output_tokens_est"],
                "timeout_s": timeout,
            }},
        )
        if completed.returncode == 0:
            return ok(data, **meta)
        return err(f"process exited {completed.returncode}", data=data, **meta)

    return run_python


def create_run_python_tool_from_config(config):
    """Factory convenience: build run_python from a Config instance."""
    return create_run_python_tool(
        enable=getattr(config, "enable_code_exec", False),
    )
