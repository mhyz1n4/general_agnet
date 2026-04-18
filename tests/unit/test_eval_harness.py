"""
Unit tests for the V1.1 M0 eval harness wiring.

These validate that the fixture loader returns well-formed records and that
every fixture carries the fields the runner asserts on.  They do NOT spin up
an LLM — that is the job of ``tests/eval/test_eval_runner.py``.
"""

from __future__ import annotations

import pytest

from tests.eval import load_fixtures

_REQUIRED_KEYS = ("name", "turns", "expected_substrings")


@pytest.fixture(scope="module")
def fixtures():
    """Load the seed fixtures once per module."""
    return load_fixtures()


def test_seed_fixtures_present(fixtures) -> None:
    """The seed corpus must carry the 5 fixtures the plan requires."""
    assert len(fixtures) >= 5, f"expected at least 5 seed fixtures, got {len(fixtures)}"


def test_every_fixture_has_required_fields(fixtures) -> None:
    """Every fixture must carry name, turns, and expected_substrings."""
    for fx in fixtures:
        for key in _REQUIRED_KEYS:
            assert key in fx, f"fixture {fx.get('name')!r} missing field {key!r}"
        assert isinstance(fx["turns"], list) and fx["turns"], (
            f"fixture {fx['name']!r} has empty or non-list turns"
        )
        assert isinstance(fx["expected_substrings"], list) and fx["expected_substrings"], (
            f"fixture {fx['name']!r} has empty or non-list expected_substrings"
        )


def test_fixture_names_unique(fixtures) -> None:
    """Fixture names are used as pytest ids — they must be unique."""
    names = [fx["name"] for fx in fixtures]
    assert len(names) == len(set(names)), f"duplicate fixture names: {names}"


def test_seed_memory_records_are_well_formed(fixtures) -> None:
    """Every seed_memory entry must be ``{content, type, topic?}``."""
    for fx in fixtures:
        for rec in fx.get("seed_memory") or []:
            assert "content" in rec, f"{fx['name']}: seed_memory record missing content"
            assert rec.get("type") in {"episodic", "semantic", "procedural"}, (
                f"{fx['name']}: invalid seed_memory type {rec.get('type')!r}"
            )
