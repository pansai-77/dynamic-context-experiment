"""Add project root to sys.path when running scripts directly."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root = str(PROJECT_ROOT)
if root not in sys.path:
    sys.path.insert(0, root)
