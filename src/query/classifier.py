"""
Query intent classification for the memory retrieval system.

``RegexClassifier`` maps user queries to ``Intent`` values using word-boundary
regex patterns.  Intent drives routing in ``MemoryManager``:

  ``RECALL_HISTORY``  → full long-term retriever search
  ``CURRENT_SESSION`` → session-only search (stub in v1, returns empty)
  ``GENERAL_TASK``    → full long-term retriever search (same path as
                        RECALL_HISTORY in v1; kept separate for future routing)

Pattern notes:
  - All patterns use ``\\b`` word boundaries to prevent substring false-positives
    (e.g. ``\\bpast\\b`` must not match "pasta").
  - Matching is case-insensitive via ``.lower()`` before ``re.search``.
  - Iteration order of ``self.rules`` determines priority; the first matching
    intent wins.
"""

import re
from enum import Enum
from typing import List, Dict

from src.constants import INTENT_CURRENT_SESSION, INTENT_GENERAL_TASK, INTENT_RECALL_HISTORY


class Intent(Enum):
    """
    Enumeration of query intent categories recognised by ``RegexClassifier``.

    Attributes:
        RECALL_HISTORY:  User is asking about something from the past
                         (e.g. "what did I say about…", "last week").
        CURRENT_SESSION: User is asking about the active session
                         (e.g. "what did I just say", "earlier today").
        GENERAL_TASK:    All other queries — factual questions, creative
                         tasks, or instructions that don't involve recall.
    """

    RECALL_HISTORY = INTENT_RECALL_HISTORY
    CURRENT_SESSION = INTENT_CURRENT_SESSION
    GENERAL_TASK = INTENT_GENERAL_TASK


class RegexClassifier:
    """
    Classifies user queries into intents using predefined regex patterns.
    """

    def __init__(self) -> None:
        """
        Initialise the classifier with built-in regex rules.

        Each ``Intent`` is mapped to a list of regex patterns.  Patterns are
        tested with ``re.search`` (case-insensitive via ``.lower()``).
        ``GENERAL_TASK`` is the implicit default and has no patterns — it is
        returned when no other intent matches.
        """
        self.rules: Dict[Intent, List[str]] = {
            Intent.RECALL_HISTORY: [
                r"\brecall\b",
                r"\bremember\b",
                r"\bsearch\b",
                r"\blast time\b",
                r"\bpreviously\b",
                r"\bhistory\b",
                r"\bpast\b",
            ],
            Intent.CURRENT_SESSION: [
                r"\bin this session\b",
                r"\bjust now\b",
                r"\brecently\b",
                r"\bwhat did i just say\b",
            ]
        }

    def classify(self, query: str) -> Intent:
        """
        Classifies the query based on the first matching rule.
        Defaults to GENERAL_TASK if no matches are found.
        """
        query_lower = query.lower()
        for intent, patterns in self.rules.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        return Intent.GENERAL_TASK
