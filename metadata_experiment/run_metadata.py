from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import MetadataSettings


RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")

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


def _serialize_settings(settings: MetadataSettings) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in asdict(settings).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def create_run_directory(results_dir: Path, run_at: datetime | None = None) -> Path:
    timestamp = (run_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def is_run_directory(path: Path) -> bool:
    return path.is_dir() and RUN_DIR_PATTERN.fullmatch(path.name) is not None


def find_latest_run_directory(results_dir: Path) -> Path:
    run_dirs = sorted(
        (path for path in results_dir.iterdir() if is_run_directory(path)),
        key=lambda path: path.name,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No timestamped run directories found under {results_dir}. "
            "Run metadata_experiment/scripts/run_experiment.py first."
        )
    return run_dirs[0]


def resolve_run_directory(run_dir: Path | None, results_dir: Path) -> Path:
    if run_dir is None:
        return find_latest_run_directory(results_dir)
    if not run_dir.is_absolute():
        run_dir = results_dir / run_dir
    if not is_run_directory(run_dir):
        raise ValueError(f"Not a timestamped run directory: {run_dir}")
    return run_dir


def build_run_metadata(
    settings: MetadataSettings,
    methods: list[str],
    run_directory: Path | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "run_directory": str(run_directory) if run_directory else None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "methods": methods,
        "settings": _serialize_settings(settings),
        "dependencies": {
            package: _package_version(package) for package in PACKAGE_VERSIONS
        },
    }
    if notes:
        metadata["notes"] = notes
    return metadata


def export_run_metadata(
    settings: MetadataSettings,
    methods: list[str],
    output_path: Path,
    run_directory: Path | None = None,
    notes: list[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_run_metadata(settings, methods, run_directory, notes)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
