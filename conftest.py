"""Put the repo root on sys.path so tests can `from backend import ...`.

There is no installed package (`pyproject.toml` sets `package = false`), so
without this pytest imports the test module with only `tests/` on the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
