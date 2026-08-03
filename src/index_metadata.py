from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings

METADATA_FILENAME = "index_metadata.json"


@dataclass(frozen=True)
class IndexMetadata:
    embedding_model: str
    chunk_strategy: str
    target_size: int
    max_size: int
    min_size: int
    overlap: int
    source_files: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def metadata_path(qdrant_path: Path) -> Path:
    return qdrant_path / METADATA_FILENAME


def write_index_metadata(qdrant_path: Path, metadata: IndexMetadata) -> None:
    qdrant_path.mkdir(parents=True, exist_ok=True)
    metadata_path(qdrant_path).write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_index_metadata(qdrant_path: Path) -> IndexMetadata | None:
    path = metadata_path(qdrant_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "chunk_strategy" in payload:
        return IndexMetadata(
            embedding_model=str(payload["embedding_model"]),
            chunk_strategy=str(payload["chunk_strategy"]),
            target_size=int(payload["target_size"]),
            max_size=int(payload["max_size"]),
            min_size=int(payload["min_size"]),
            overlap=int(payload["overlap"]),
            source_files=[str(name) for name in payload["source_files"]],
        )
    # Legacy page-local index metadata (pre-continuous chunking).
    return IndexMetadata(
        embedding_model=str(payload["embedding_model"]),
        chunk_strategy="page_local_fixed",
        target_size=int(payload["chunk_size"]),
        max_size=int(payload["chunk_size"]),
        min_size=0,
        overlap=int(payload["chunk_overlap"]),
        source_files=[str(name) for name in payload["source_files"]],
    )


def expected_metadata(settings: Settings, source_files: list[str]) -> IndexMetadata:
    return IndexMetadata(
        embedding_model=settings.embedding_model,
        chunk_strategy=settings.chunk_strategy,
        target_size=settings.chunk_target_size,
        max_size=settings.chunk_max_size,
        min_size=settings.chunk_min_size,
        overlap=settings.chunk_overlap,
        source_files=sorted(source_files),
    )


def verify_index_metadata(settings: Settings) -> IndexMetadata:
    stored = read_index_metadata(settings.qdrant_path)
    if stored is None:
        raise FileNotFoundError(
            f"Index metadata not found at {metadata_path(settings.qdrant_path)}. "
            "Run scripts/build_index.py after changing the PDF, embedding model, or chunk settings."
        )

    mismatches: list[str] = []
    if stored.embedding_model != settings.embedding_model:
        mismatches.append(
            f"embedding_model: index={stored.embedding_model!r}, config={settings.embedding_model!r}"
        )
    if stored.chunk_strategy != settings.chunk_strategy:
        mismatches.append(
            f"chunk_strategy: index={stored.chunk_strategy!r}, config={settings.chunk_strategy!r}"
        )
    if stored.target_size != settings.chunk_target_size:
        mismatches.append(
            f"target_size: index={stored.target_size}, config={settings.chunk_target_size}"
        )
    if stored.max_size != settings.chunk_max_size:
        mismatches.append(
            f"max_size: index={stored.max_size}, config={settings.chunk_max_size}"
        )
    if stored.min_size != settings.chunk_min_size:
        mismatches.append(
            f"min_size: index={stored.min_size}, config={settings.chunk_min_size}"
        )
    if stored.overlap != settings.chunk_overlap:
        mismatches.append(
            f"overlap: index={stored.overlap}, config={settings.chunk_overlap}"
        )
    if settings.book_file is not None:
        expected_source_files = sorted([settings.book_file.name])
        if sorted(stored.source_files) != expected_source_files:
            mismatches.append(
                f"source_files: index={stored.source_files!r}, config={expected_source_files!r}"
            )

    if mismatches:
        raise ValueError(
            "Qdrant index does not match current settings:\n- "
            + "\n- ".join(mismatches)
            + "\nRun scripts/build_index.py to rebuild the index."
        )

    return stored
