from __future__ import annotations
import json
from pathlib import Path
from src.config import Settings
from src.run_metadata import build_run_metadata, export_run_metadata

def test_build_run_metadata_includes_settings_and_dependencies(tmp_path: Path) -> None:
    settings = Settings()
    metadata = build_run_metadata(settings, ["Baseline (Top-8)"])
    assert metadata["methods"] == ["Baseline (Top-8)"]
    assert metadata["settings"]["temperature"] == settings.temperature
    assert "mlx-lm" in metadata["dependencies"]
    assert "run_timestamp_utc" in metadata

def test_export_run_metadata_writes_json(tmp_path: Path) -> None:
    settings = Settings()
    output_path = tmp_path / "run_config.json"
    export_run_metadata(settings, ["No RAG"], output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["methods"] == ["No RAG"]
    assert output_path.exists()
