import re
from datetime import datetime, timedelta
from typing import Optional, Tuple


class TemporalExtractor:
    """
    Extracts temporal markers from user queries and converts them
    to relative date ranges or specific timestamps.
    """

    def __init__(self) -> None:
        """
        Initialise the extractor with built-in temporal patterns.

        Patterns are matched with ``re.search`` on a lowercased query.
        Each pattern maps to a handler method that returns a
        ``(start_datetime, end_datetime)`` tuple.
        """
        # Basic patterns for v1
        self.patterns = {
            r"\btoday\b": self._get_today,
            r"\byesterday\b": self._get_yesterday,
            r"\blast session\b": self._get_last_session,
            r"\b(\d+)\s+days?\s+ago\b": self._get_days_ago,
        }

    def extract(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Extract a date range from a query containing a temporal marker.

        Supported markers: ``today``, ``yesterday``, ``last session``,
        and ``N days ago`` (where N is a positive integer).

        Args:
            query: Raw user query string.

        Returns:
            ``(start_datetime, end_datetime)`` if a marker is found,
            or ``None`` if no temporal pattern matches.
        """
        query_lower = query.lower()
        for pattern, handler in self.patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                if "days ago" in pattern:
                    return handler(int(match.group(1)))
                return handler()
        return None

    def _get_today(self) -> Tuple[datetime, datetime]:
        """Return ``(midnight_today, now)`` for the current local day."""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now

    def _get_yesterday(self) -> Tuple[datetime, datetime]:
        """Return ``(midnight_yesterday, 23:59:59.999999_yesterday)``."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    def _get_last_session(self) -> Tuple[datetime, datetime]:
        """
        Return a date range approximating the previous session.

        v1 simplification: returns ``(now - 24 hours, now)``.
        v2 will use stored session metadata for precise boundaries.
        """
        # In v1, treat "last session" as the last 24 hours.
        # v2: check session metadata for precise session boundaries.
        now = datetime.now()
        start = now - timedelta(hours=24)
        return start, now

    def _get_days_ago(self, days: int) -> Tuple[datetime, datetime]:
        """
        Return ``(midnight, 23:59:59.999999)`` for the day *N* days ago.

        Args:
            days: Number of days to look back (must be a positive integer).

        Returns:
            ``(start_of_day, end_of_day)`` for the target date.
        """
        now = datetime.now()
        target_day = now - timedelta(days=days)
        start = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = target_day.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
