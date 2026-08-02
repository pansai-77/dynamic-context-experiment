"""Add project root and metadata_experiment/src to sys.path for scripts."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METADATA_SRC = EXPERIMENT_ROOT / "src"

for path in (str(PROJECT_ROOT), str(METADATA_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
