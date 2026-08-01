from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings

METADATA_FILENAME = "index_metadata.json"

@dataclass(frozen=True)
class IndexMetadata:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
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
    return IndexMetadata(
        embedding_model=str(payload["embedding_model"]),
        chunk_size=int(payload["chunk_size"]),
        chunk_overlap=int(payload["chunk_overlap"]),
        source_files=[str(name) for name in payload["source_files"]],
    )

def expected_metadata(settings: Settings, source_files: list[str]) -> IndexMetadata:
    return IndexMetadata(
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
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
    if stored.chunk_size != settings.chunk_size:
        mismatches.append(
            f"chunk_size: index={stored.chunk_size}, config={settings.chunk_size}"
        )
    if stored.chunk_overlap != settings.chunk_overlap:
        mismatches.append(
            f"chunk_overlap: index={stored.chunk_overlap}, config={settings.chunk_overlap}"
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
