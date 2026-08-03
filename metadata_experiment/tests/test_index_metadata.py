from __future__ import annotations

import csv
from pathlib import Path

import pytest

from metadata_experiment.config import MetadataSettings
from metadata_experiment.index_metadata import (
    MetadataIndexManifest,
    expected_manifest,
    manifest_path,
    read_index_manifest,
    verify_index_metadata,
    write_index_manifest,
)
from metadata_experiment.logic import load_gold
from metadata_experiment.metrics import ranking_metrics
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
        "topics": ["家庭生活"],
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


def test_gold_loader_keeps_blank_chunk_ids_empty(tmp_path: Path):
    gold_file = tmp_path / "gold_annotations.csv"
    with gold_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Question ID", "Gold Topics", "Gold Chunk IDs"],
        )
        writer.writeheader()
        writer.writerow({
            "Question ID": "Q01",
            "Gold Topics": "老牛陪伴",
            "Gold Chunk IDs": "",
        })

    gold = load_gold(gold_file)
    assert gold["Q01"]["topics"] == ["老牛陪伴"]
    assert gold["Q01"]["chunks"] == []


def test_ranking_metrics_stays_blank_without_gold_chunks():
    assert ranking_metrics(["c0001", "c0002"], []) == (None, None)


def test_gold_topics_do_not_imply_hit_at_4():
    hit, mrr = ranking_metrics(["c9999"], ["c0001"])
    assert hit == 0.0
    assert mrr == 0.0
