from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_SRC = Path(__file__).resolve().parents[1] / "src"

for path in (str(PROJECT_ROOT), str(METADATA_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
