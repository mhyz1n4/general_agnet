"""End-to-end eval harness (V1.1 M0).

Golden-fixture tests that exercise the live orchestrator against a real LLM
endpoint. Skipped automatically when the endpoint is unreachable so the rest
of the suite can run unimpeded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

_FIXTURES_PATH = Path(__file__).parent / "fixtures" / "seed.yaml"


def load_fixtures() -> List[Dict[str, Any]]:
    """Parse ``seed.yaml`` and return the list of fixture dicts."""
    with open(_FIXTURES_PATH) as f:
        doc = yaml.safe_load(f)
    return list(doc.get("fixtures", []))
