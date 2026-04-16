"""
Unit tests for RegexClassifier.

Verifies that every documented pattern routes to the correct Intent, that
word-boundary matching is respected (e.g. "pasta" should not trigger the
RECALL_HISTORY "past" pattern), and that the default fallback is GENERAL_TASK.
"""

import pytest

from src.query.classifier import Intent, RegexClassifier


@pytest.fixture
def clf() -> RegexClassifier:
    """Return a fresh RegexClassifier with its default built-in rules."""
    return RegexClassifier()


# ---------------------------------------------------------------------------
# RECALL_HISTORY patterns
# ---------------------------------------------------------------------------


class TestRecallHistoryPatterns:
    """Each keyword in the RECALL_HISTORY rule set should trigger the intent."""

    def test_recall_keyword(self, clf: RegexClassifier) -> None:
        """'recall' should route to RECALL_HISTORY."""
        assert clf.classify("please recall that meeting") == Intent.RECALL_HISTORY

    def test_remember_keyword(self, clf: RegexClassifier) -> None:
        """'remember' should route to RECALL_HISTORY."""
        assert clf.classify("do you remember what I said?") == Intent.RECALL_HISTORY

    def test_search_keyword(self, clf: RegexClassifier) -> None:
        """'search' should route to RECALL_HISTORY."""
        assert clf.classify("search my notes") == Intent.RECALL_HISTORY

    def test_last_time_keyword(self, clf: RegexClassifier) -> None:
        """'last time' should route to RECALL_HISTORY."""
        assert clf.classify("what did we discuss last time?") == Intent.RECALL_HISTORY

    def test_previously_keyword(self, clf: RegexClassifier) -> None:
        """'previously' should route to RECALL_HISTORY."""
        assert clf.classify("as previously discussed") == Intent.RECALL_HISTORY

    def test_history_keyword(self, clf: RegexClassifier) -> None:
        """'history' should route to RECALL_HISTORY."""
        assert clf.classify("show me my history") == Intent.RECALL_HISTORY

    def test_past_keyword(self, clf: RegexClassifier) -> None:
        """'past' should route to RECALL_HISTORY."""
        assert clf.classify("in the past we talked about Python") == Intent.RECALL_HISTORY


# ---------------------------------------------------------------------------
# CURRENT_SESSION patterns
# ---------------------------------------------------------------------------


class TestCurrentSessionPatterns:
    """Each keyword in the CURRENT_SESSION rule set should trigger the intent."""

    def test_in_this_session(self, clf: RegexClassifier) -> None:
        """'in this session' should route to CURRENT_SESSION."""
        assert clf.classify("what did I say in this session?") == Intent.CURRENT_SESSION

    def test_just_now(self, clf: RegexClassifier) -> None:
        """'just now' should route to CURRENT_SESSION."""
        assert clf.classify("I just now mentioned that") == Intent.CURRENT_SESSION

    def test_recently(self, clf: RegexClassifier) -> None:
        """'recently' should route to CURRENT_SESSION."""
        assert clf.classify("I recently asked about this") == Intent.CURRENT_SESSION

    def test_what_did_i_just_say(self, clf: RegexClassifier) -> None:
        """'what did i just say' should route to CURRENT_SESSION."""
        assert clf.classify("what did i just say?") == Intent.CURRENT_SESSION


# ---------------------------------------------------------------------------
# GENERAL_TASK fallback
# ---------------------------------------------------------------------------


class TestGeneralTaskFallback:
    """Queries with no matching pattern should fall back to GENERAL_TASK."""

    def test_unknown_query(self, clf: RegexClassifier) -> None:
        """A plain factual question with no recall keywords routes to GENERAL_TASK."""
        assert clf.classify("what is the capital of France?") == Intent.GENERAL_TASK

    def test_empty_string(self, clf: RegexClassifier) -> None:
        """An empty query should return GENERAL_TASK without raising."""
        assert clf.classify("") == Intent.GENERAL_TASK

    def test_greeting(self, clf: RegexClassifier) -> None:
        """A simple greeting has no recall keywords and routes to GENERAL_TASK."""
        assert clf.classify("hello, how are you?") == Intent.GENERAL_TASK


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    """Patterns should match regardless of the input casing."""

    def test_uppercase_recall(self, clf: RegexClassifier) -> None:
        """'RECALL' in uppercase should still trigger RECALL_HISTORY."""
        assert clf.classify("RECALL my preferences") == Intent.RECALL_HISTORY

    def test_mixed_case_remember(self, clf: RegexClassifier) -> None:
        """'Remember' in title case should still trigger RECALL_HISTORY."""
        assert clf.classify("Remember what I told you?") == Intent.RECALL_HISTORY


# ---------------------------------------------------------------------------
# Word-boundary enforcement
# ---------------------------------------------------------------------------


class TestWordBoundary:
    """Patterns use \\b so partial word matches must NOT trigger an intent."""

    def test_pasta_does_not_trigger_past(self, clf: RegexClassifier) -> None:
        """'pasta' contains 'past' but \\bpast\\b should not match it."""
        assert clf.classify("I love eating pasta") == Intent.GENERAL_TASK

    def test_researching_does_not_trigger_search(self, clf: RegexClassifier) -> None:
        """'researching' contains 'search' but \\bsearch\\b should not match it."""
        assert clf.classify("I am researching this topic") == Intent.GENERAL_TASK
