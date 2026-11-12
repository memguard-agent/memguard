"""Pin machine-local memwatch configuration so tests never depend on this box's files."""
import pytest

from memguard import memwatch

#: Stand-ins for a machine's exemption file (~/.config/memguard/exempt).
TEST_EXEMPT_PATTERNS = ("protected_service", "redis-server", "/Applications/Database.app")


@pytest.fixture(autouse=True)
def _pin_memwatch_config(monkeypatch):
    monkeypatch.setattr(memwatch, "EXEMPT_PATTERNS", TEST_EXEMPT_PATTERNS)
    monkeypatch.setattr(memwatch, "REPO_PATH_MARKERS", ("/Users/x/code/",))
    monkeypatch.setattr(memwatch, "DISK_WARN_GB", 0.0)   # never this Mac's disk
    monkeypatch.setattr(memwatch, "SWAP_ROOM_GB", 0.0)   # ...nor its swap room
