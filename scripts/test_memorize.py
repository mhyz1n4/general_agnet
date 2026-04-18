"""
Standalone smoke-test for the memorize tool call flow.

Simulates a single orchestrator turn with user input:
    "memorize this: I ate an orange today"

The Qwen3 model served by vLLM emits tool calls as <tool_call> XML inside
the content field.  vLLM's qwen3_xml parser does not convert these to the
OpenAI tool_calls format reliably (streaming: name=null; non-streaming:
tool_calls=[]).  This script bypasses the broken conversion, parses the
<tool_call> block directly, and executes the tool — proving the end-to-end
logic is correct and pinning the integration failure to the vLLM layer.

Run from the repo root:
    . .venv/bin/activate && PYTHONPATH=. python scripts/test_memorize.py
"""

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from urllib import request as urllib_request

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Bootstrap (mirrors main.py)
# ---------------------------------------------------------------------------
from src.config import Config
from src.logging_config import get_logger, init_session_logging
from src.constants import (
    SESSION_ID_HEX_LENGTH,
    SESSION_LOG_DATE_FORMAT,
    SESSION_LOG_SUBDIR_PREFIX,
)

config = Config()
session_id = "smoketest_" + uuid.uuid4().hex[:SESSION_ID_HEX_LENGTH]
session_date = datetime.now(timezone.utc).strftime(SESSION_LOG_DATE_FORMAT)
session_log_dir = os.path.join(
    config.log_dir,
    f"{SESSION_LOG_SUBDIR_PREFIX}{session_id}_{session_date}",
)
init_session_logging(
    session_log_dir,
    log_level="DEBUG",
    strands_log_level="WARNING",
)
logger = get_logger(__name__)

print(f"\n{'='*60}")
print(f"  Memorize smoke test  |  session: {session_id}")
print(f"  Endpoint : {config.llm_api_endpoint}")
print(f"  Model    : {config.llm_model}")
print(f"  Log dir  : {session_log_dir}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Memory provider (V1.1 — StubMemoryProvider for smoke test)
# ---------------------------------------------------------------------------
from src.memory.stub_provider import StubMemoryProvider
from src.memory.provider import SearchFilters
from src.tools.memorize import create_memorize_tool
from src.prompts.loader import render_prompt

provider = StubMemoryProvider()
memorize_fn = create_memorize_tool(provider, session_id=session_id)

# ---------------------------------------------------------------------------
# Tool registry — maps tool name -> callable (no Strands dispatcher needed)
# ---------------------------------------------------------------------------
TOOLS = {
    "memorize": memorize_fn,
}

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "memorize",
            "description": (
                "Save information to long-term memory. "
                "Call only when the user explicitly asks to remember, save, or note something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to remember."},
                    "type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural"],
                        "description": "episodic=events, semantic=facts/preferences, procedural=how-tos.",
                    },
                    "topic": {"type": "string", "description": "Optional category label."},
                },
                "required": ["content", "type"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_THINKING_RE = re.compile(
    r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE
)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def _strip_thinking(text: str) -> str:
    """Remove CoT thinking blocks from text."""
    return _THINKING_RE.sub("", text).strip()


def _extract_tool_calls(content: str) -> list[dict]:
    """Parse all <tool_call>JSON</tool_call> blocks from content."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(content):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError as e:
            logger.warning("test: failed to parse tool_call JSON", extra={"data": {"error": str(e), "raw": m.group(1)[:200]}})
    return calls


def _call_api(messages: list[dict]) -> dict:
    """Single non-streaming call to vLLM; returns the choice dict."""
    payload = json.dumps({
        "model": config.llm_model,
        "messages": messages,
        "tools": TOOL_DEFS,
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": config.llm_max_tokens,
    }).encode()
    req = urllib_request.Request(
        f"{config.llm_api_endpoint}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {config.llm_api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib_request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]


# ---------------------------------------------------------------------------
# Agentic loop — handles <tool_call> in content, up to max_turns
# ---------------------------------------------------------------------------
USER_INPUT = "memorize this: I ate an orange today"
system_prompt = render_prompt("system_prompt")

messages = [
    {"role": "system",  "content": system_prompt},
    {"role": "user",    "content": USER_INPUT},
]

print(f"[User]\n{USER_INPUT}\n")
print(f"[Sending to LLM...]\n{'-'*40}")

MAX_TURNS = 5
tool_calls_executed: list[dict] = []

for turn in range(MAX_TURNS):
    choice = _call_api(messages)
    msg    = choice["message"]
    content = msg.get("content") or ""
    finish  = choice.get("finish_reason", "")

    logger.debug(
        "test: LLM response",
        extra={"data": {"turn": turn, "finish_reason": finish, "content_len": len(content)}},
    )

    # --- Check if vLLM properly populated tool_calls (future: if parser is fixed) ---
    api_tool_calls = msg.get("tool_calls") or []

    # --- Parse <tool_call> XML from content (current workaround) ---
    xml_tool_calls = _extract_tool_calls(content)

    all_calls = []
    if api_tool_calls:
        print(f"  [turn {turn}] vLLM tool_calls API populated — {len(api_tool_calls)} call(s)")
        for tc in api_tool_calls:
            fn = tc.get("function", {})
            all_calls.append({
                "name": fn.get("name"),
                "arguments": json.loads(fn.get("arguments", "{}")),
                "source": "api",
            })
    elif xml_tool_calls:
        print(f"  [turn {turn}] <tool_call> XML parsed from content — {len(xml_tool_calls)} call(s)")
        for tc in xml_tool_calls:
            all_calls.append({
                "name": tc.get("name"),
                "arguments": tc.get("arguments", {}),
                "source": "xml",
            })

    if all_calls:
        messages.append({"role": "assistant", "content": content})
        for tc in all_calls:
            name = tc["name"]
            args = tc["arguments"]
            print(f"  → Executing tool: {name}({args})")
            if name in TOOLS:
                result = TOOLS[name](**args)
                print(f"  ← Result: {result}")
                tool_calls_executed.append({"tool": name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result,
                })
            else:
                print(f"  [WARN] Unknown tool: {name}")
        continue

    # No tool calls — model is done
    visible = _strip_thinking(content)
    print(f"\n[Assistant response]\n{'-'*40}")
    print(f"(raw {len(content)} chars, finish={finish})")
    print(visible if visible else "(no visible content after stripping thinking)")
    break

else:
    print(f"\n[WARN] Reached max turns ({MAX_TURNS}) without a final response")

# ---------------------------------------------------------------------------
# Verify memory was written
# ---------------------------------------------------------------------------
print(f"\n[Memory verification]\n{'-'*40}")
if tool_calls_executed:
    print(f"  Tool calls executed: {len(tool_calls_executed)}")
    for tc in tool_calls_executed:
        print(f"    {tc['tool']}({tc['args']}) → {tc['result']}")

results = provider.search("orange", SearchFilters(limit=5))
if results:
    print(f"\n  PASS — {len(results)} entry(ies) found in memory")
    for r in results:
        print(f"    [{r.type}] {r.content[:200]}")
else:
    print("\n  FAIL — no entries found in memory matching 'orange'.")
    print(f"  (All items: {list(provider._items.values())})")

print()
