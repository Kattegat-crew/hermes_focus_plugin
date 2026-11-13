import sys
from pathlib import Path

import pytest

# Ensure root package and neural_context are importable
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


@pytest.fixture(autouse=True)
def isolate_hermes_test_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default Hermes config, storage, and receipt paths inside each test's temp home."""
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for name in ("HERMES_MEMORY_DB", "FOCUS_MEMORY_DB", "HERMES_MEMORY_DIR", "HERMES_MEMORY_PATH"):
        monkeypatch.delenv(name, raising=False)
