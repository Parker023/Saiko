import json
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def remotive_payload() -> dict:
    """A real Remotive response, captured once and committed.

    Session scope: read the file once for the whole run, not per test.
    """
    return json.loads((FIXTURES / "remotive.json").read_text())


@pytest.fixture(scope="session")
def arbeitnow_payload() -> dict:
    return json.loads((FIXTURES / "arbeitnow.json").read_text())
