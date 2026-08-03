from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.index_metadata import (
    IndexMetadata,
    expected_metadata,
    read_index_metadata,
    verify_index_metadata,
    write_index_metadata,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        root_dir=tmp_path,
        book_dir=tmp_path / "data" / "book",
        book_file=tmp_path / "data" / "book" / "活着.pdf",
        questions_file=tmp_path / "data" / "questions" / "questions.csv",
        qdrant_path=tmp_path / "qdrant_storage",
        results_dir=tmp_path / "results",
        collection_name="huozhe",
        embedding_model="BAAI/bge-small-zh-v1.5",
        chunk_strategy="continuous_sentence_aware",
        chunk_target_size=600,
        chunk_max_size=800,
        chunk_min_size=100,
        chunk_overlap=100,
    )


def test_write_and_read_index_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    metadata = expected_metadata(settings, ["活着.pdf"])
    write_index_metadata(settings.qdrant_path, metadata)
    loaded = read_index_metadata(settings.qdrant_path)
    assert loaded == metadata


def test_verify_index_metadata_raises_when_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="Index metadata not found"):
        verify_index_metadata(settings)


def test_verify_index_metadata_raises_on_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    write_index_metadata(
        settings.qdrant_path,
        IndexMetadata(
            embedding_model="old-model",
            chunk_strategy="continuous_sentence_aware",
            target_size=600,
            max_size=800,
            min_size=100,
            overlap=100,
            source_files=["活着.pdf"],
        ),
    )
    with pytest.raises(ValueError, match="does not match current settings"):
        verify_index_metadata(settings)


def test_verify_index_metadata_raises_on_source_file_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    write_index_metadata(
        settings.qdrant_path,
        IndexMetadata(
            embedding_model=settings.embedding_model,
            chunk_strategy=settings.chunk_strategy,
            target_size=settings.chunk_target_size,
            max_size=settings.chunk_max_size,
            min_size=settings.chunk_min_size,
            overlap=settings.chunk_overlap,
            source_files=["old-book.pdf"],
        ),
    )
    with pytest.raises(ValueError, match="source_files"):
        verify_index_metadata(settings)


def test_verify_index_metadata_accepts_matching_config(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    write_index_metadata(settings.qdrant_path, expected_metadata(settings, ["活着.pdf"]))
    loaded = verify_index_metadata(settings)
    assert loaded.source_files == ["活着.pdf"]
