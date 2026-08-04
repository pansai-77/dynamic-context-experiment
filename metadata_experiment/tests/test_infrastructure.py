from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from metadata_experiment.classification import ClassificationCache, compute_cache_version
from metadata_experiment.classification_prompts import CLASSIFICATION_PROMPT_VERSION
from metadata_experiment.config import MetadataSettings
from metadata_experiment.index_metadata import (
    MetadataIndexManifest,
    expected_manifest,
    manifest_path,
    read_index_manifest,
    verify_index_metadata,
    write_index_manifest,
)
from metadata_experiment.topics import TOPIC_TAXONOMY_VERSION, topic_names
from metadata_experiment.run_metadata import (
    build_run_metadata,
    create_run_directory,
    export_run_metadata,
)
from src.models import Chunk


def _sample_manifest(**overrides) -> MetadataIndexManifest:
    payload = {
        "collection": "huozhe_meta",
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "chunk_strategy": "continuous_sentence_aware",
        "chunk_target_size": 600,
        "chunk_max_size": 800,
        "chunk_min_size": 100,
        "chunk_overlap": 100,
        "source_files": ["活着.pdf"],
        "chunk_ids": ["c0001", "c0002"],
        "topics": topic_names(),
        "topic_taxonomy_version": TOPIC_TAXONOMY_VERSION,
        "classification_prompt_version": CLASSIFICATION_PROMPT_VERSION,
    }
    payload.update(overrides)
    return MetadataIndexManifest(**payload)


def test_manifest_roundtrip(tmp_path: Path):
    manifest = _sample_manifest()
    path = tmp_path / "metadata_index_manifest.json"
    write_index_manifest(path, manifest)
    loaded = read_index_manifest(path)
    assert loaded == manifest


def test_expected_manifest_uses_sorted_chunk_ids(tmp_path: Path):
    cfg = MetadataSettings(
        root_dir=tmp_path,
        experiment_dir=tmp_path / "metadata_experiment",
        book_file=tmp_path / "data/book/活着.pdf",
        qdrant_path=tmp_path / "qdrant_storage_metadata",
    )
    chunks = [
        Chunk("c0002", "text", "活着.pdf", 1, 1, 1, 2),
        Chunk("c0001", "text", "活着.pdf", 1, 1, 1, 1),
    ]
    manifest = expected_manifest(cfg, chunks, ["家庭生活"])
    assert manifest.chunk_ids == ["c0001", "c0002"]
    assert manifest.source_files == ["活着.pdf"]


def test_cache_rejects_stale_topic_vocabulary(tmp_path: Path):
    cache_version = compute_cache_version(
        llm_model="mlx-community/Qwen2.5-3B-Instruct-4bit",
        temperature=0.0,
        max_new_tokens=128,
    )
    cache_dir = tmp_path / "cache"
    cache = ClassificationCache(cache_dir, cache_version)
    cache._entries["c0001"] = {
        "topics": ["老牛晚年陪伴"],
        "raw_response": '{"topics": ["老牛晚年陪伴"]}',
        "prompt_version": CLASSIFICATION_PROMPT_VERSION,
        "cache_version": cache_version,
    }
    assert cache.get("c0001") is None


def test_verify_index_metadata_detects_stale_taxonomy(tmp_path: Path, monkeypatch):
    cfg = MetadataSettings(
        root_dir=tmp_path,
        experiment_dir=tmp_path / "metadata_experiment",
        book_file=tmp_path / "data/book/活着.pdf",
        qdrant_path=tmp_path / "qdrant_storage_metadata",
    )
    write_index_manifest(
        manifest_path(cfg.qdrant_path),
        _sample_manifest(topic_taxonomy_version="2026-08-04-v5"),
    )
    monkeypatch.setattr(
        "metadata_experiment.index_metadata.list_collection_chunk_ids",
        lambda *_args, **_kwargs: ["c0001", "c0002"],
    )
    with pytest.raises(ValueError, match="topic_taxonomy_version"):
        verify_index_metadata(cfg)


def test_verify_index_metadata_detects_setting_mismatch(tmp_path: Path, monkeypatch):
    cfg = MetadataSettings(
        root_dir=tmp_path,
        experiment_dir=tmp_path / "metadata_experiment",
        book_file=tmp_path / "data/book/活着.pdf",
        qdrant_path=tmp_path / "qdrant_storage_metadata",
        chunk_target_size=600,
    )
    write_index_manifest(
        manifest_path(cfg.qdrant_path),
        _sample_manifest(chunk_target_size=500),
    )
    monkeypatch.setattr(
        "metadata_experiment.index_metadata.list_collection_chunk_ids",
        lambda *_args, **_kwargs: ["c0001", "c0002"],
    )

    with pytest.raises(ValueError, match="chunk_target_size"):
        verify_index_metadata(cfg)


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
