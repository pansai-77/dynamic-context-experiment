from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from metadata_experiment.config import MetadataSettings
from metadata_experiment.run_metadata import (
    build_run_metadata,
    create_run_directory,
    export_run_metadata,
)


def test_build_run_metadata_includes_settings_and_notes(tmp_path: Path) -> None:
    settings = MetadataSettings(
        root_dir=tmp_path,
        experiment_dir=tmp_path / "metadata_experiment",
        results_dir=tmp_path / "results",
    )
    run_dir = tmp_path / "results" / "20260804_093512"
    run_dir.mkdir(parents=True)
    metadata = build_run_metadata(
        settings,
        ["Query-Aware Top-4", "Query-Aware + Metadata Top-4"],
        run_dir,
        notes=["Retrieval Time includes two embeddings for method B."],
    )
    assert metadata["methods"] == [
        "Query-Aware Top-4",
        "Query-Aware + Metadata Top-4",
    ]
    assert metadata["settings"]["random_seed"] == settings.random_seed
    assert metadata["run_directory"] == str(run_dir)
    assert "qdrant-client" in metadata["dependencies"]
    assert metadata["notes"] == ["Retrieval Time includes two embeddings for method B."]


def test_export_run_metadata_writes_json(tmp_path: Path) -> None:
    settings = MetadataSettings(
        root_dir=tmp_path,
        experiment_dir=tmp_path / "metadata_experiment",
        results_dir=tmp_path / "results",
    )
    run_dir = tmp_path / "results" / "20260804_093512"
    run_dir.mkdir(parents=True)
    output_path = run_dir / "run_config.json"
    export_run_metadata(settings, ["Query-Aware Top-4"], output_path, run_directory=run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["methods"] == ["Query-Aware Top-4"]
    assert output_path.exists()


def test_create_run_directory_uses_timestamp_format(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_dir = create_run_directory(results_dir, datetime(2026, 8, 4, 9, 35, 12))
    assert run_dir.name == "20260804_093512"
    assert run_dir.exists()
