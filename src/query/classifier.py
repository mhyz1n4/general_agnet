import re
from enum import Enum
from typing import List, Dict


class Intent(Enum):
    RECALL_HISTORY = "recall_history"
    CURRENT_SESSION = "current_session"
    GENERAL_TASK = "general_task"


class RegexClassifier:
    """
    Classifies user queries into intents using predefined regex patterns.
    """

    def __init__(self):
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
