"""
V1.1 M0 eval runner.

Parametrises a single pytest function over every fixture loaded from
``seed.yaml``.  For each fixture the runner:

  1. Builds a fresh orchestrator with the fixture's ``seed_memory``.
  2. Runs every message in ``turns`` sequentially against the live LLM.
  3. Asserts every ``expected_substrings`` entry appears (case-insensitive)
     in the final assistant response.
  4. Asserts the aggregate wall-clock latency stays under ``max_latency_ms``.
  5. Asserts the orchestrator's recorded tool-call count stays under
     ``max_tool_calls``.

Fail-on-regression is the contract: a missing substring, blown latency
budget, or excess tool use causes the fixture to fail.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest


def _fixture_ids(fixtures: List[Dict[str, Any]]) -> List[str]:
    """Return a stable parametrize id per fixture, falling back to index."""
    return [f.get("name", f"fixture_{i}") for i, f in enumerate(fixtures)]


from tests.eval import load_fixtures

_FIXTURES = load_fixtures()


@pytest.mark.eval
@pytest.mark.parametrize("fixture", _FIXTURES, ids=_fixture_ids(_FIXTURES))
def test_eval_fixture(fixture: Dict[str, Any], eval_orchestrator) -> None:
    """Run a single golden fixture end-to-end against the live LLM."""
    turns: List[str] = fixture["turns"]
    expected_substrings: List[str] = fixture["expected_substrings"]
    max_latency_ms = fixture.get("max_latency_ms")
    max_tool_calls = fixture.get("max_tool_calls")

    orch = eval_orchestrator(seed_memory=fixture.get("seed_memory"))

    t0 = time.perf_counter()
    final_response: str = ""
    for user_input in turns:
        final_response = orch._process_turn(user_input)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    lowered = final_response.lower()
    missing = [s for s in expected_substrings if s.lower() not in lowered]
    assert not missing, (
        f"Missing expected substrings {missing} in response: {final_response!r}"
    )

    if max_latency_ms is not None:
        assert elapsed_ms <= max_latency_ms, (
            f"Latency {elapsed_ms:.0f}ms exceeded budget {max_latency_ms}ms "
            f"for fixture {fixture.get('name')!r}"
        )

    if max_tool_calls is not None:
        calls_made = orch.metrics.tool_calls_made
        assert calls_made <= max_tool_calls, (
            f"Tool calls {calls_made} exceeded budget {max_tool_calls} "
            f"for fixture {fixture.get('name')!r}"
        )
