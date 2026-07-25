from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from src.config import Settings
from src.run_metadata import (
    build_run_metadata,
    create_run_directory,
    export_run_metadata,
    find_latest_run_directory,
)

def test_build_run_metadata_includes_settings_and_dependencies(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "20260726_093512"
    run_dir.mkdir()
    metadata = build_run_metadata(settings, ["Baseline (Top-8)"], run_dir)
    assert metadata["methods"] == ["Baseline (Top-8)"]
    assert metadata["settings"]["temperature"] == settings.temperature
    assert metadata["run_directory"] == str(run_dir)
    assert "mlx-lm" in metadata["dependencies"]
    assert "run_timestamp_utc" in metadata

def test_export_run_metadata_writes_json(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "20260726_093512"
    run_dir.mkdir()
    output_path = run_dir / "run_config.json"
    export_run_metadata(settings, ["No RAG"], output_path, run_directory=run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["methods"] == ["No RAG"]
    assert output_path.exists()

def test_create_run_directory_uses_timestamp_format(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, datetime(2026, 7, 26, 9, 35, 12))
    assert run_dir.name == "20260726_093512"
    assert run_dir.exists()

def test_find_latest_run_directory(tmp_path: Path) -> None:
    (tmp_path / "20260726_093512").mkdir()
    (tmp_path / "20260726_120000").mkdir()
    assert find_latest_run_directory(tmp_path).name == "20260726_120000"
