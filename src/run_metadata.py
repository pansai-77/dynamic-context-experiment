from __future__ import annotations
from dataclasses import asdict
from datetime import UTC, datetime
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import Settings

PACKAGE_VERSIONS = (
    "mlx",
    "mlx-lm",
    "transformers",
    "sentence-transformers",
    "qdrant-client",
    "pandas",
    "PyMuPDF",
)

def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"

def _serialize_settings(settings: Settings) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in asdict(settings).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized

def build_run_metadata(settings: Settings, methods: list[str]) -> dict[str, Any]:
    return {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "methods": methods,
        "settings": _serialize_settings(settings),
        "dependencies": {
            package: _package_version(package) for package in PACKAGE_VERSIONS
        },
    }

def export_run_metadata(settings: Settings, methods: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_run_metadata(settings, methods)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
