from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from config import MetadataSettings, settings


@dataclass(frozen=True)
class IndexMetadata:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    source_files: list[str]
    collection_name: str
    chunk_embedding_policy: str = "text_only"

    def to_dict(self) -> dict:
        return asdict(self)


def write_index_metadata(path: Path, metadata: IndexMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_index_metadata(path: Path) -> IndexMetadata | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return IndexMetadata(
        embedding_model=str(payload["embedding_model"]),
        chunk_size=int(payload["chunk_size"]),
        chunk_overlap=int(payload["chunk_overlap"]),
        source_files=[str(name) for name in payload["source_files"]],
        collection_name=str(payload.get("collection_name", "huozhe_meta")),
        chunk_embedding_policy=str(payload.get("chunk_embedding_policy", "text_only")),
    )


def expected_metadata(cfg: MetadataSettings, source_files: list[str]) -> IndexMetadata:
    return IndexMetadata(
        embedding_model=cfg.embedding_model,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        source_files=sorted(source_files),
        collection_name=cfg.collection_name,
        chunk_embedding_policy="text_only",
    )


def verify_index_metadata(cfg: MetadataSettings = settings) -> IndexMetadata:
    stored = read_index_metadata(cfg.index_metadata_path)
    if stored is None:
        raise FileNotFoundError(
            f"Index metadata not found at {cfg.index_metadata_path}. "
            "Run metadata_experiment/scripts/build_metadata_index.py first."
        )

    mismatches: list[str] = []
    if stored.embedding_model != cfg.embedding_model:
        mismatches.append(
            f"embedding_model: index={stored.embedding_model!r}, config={cfg.embedding_model!r}"
        )
    if stored.chunk_size != cfg.chunk_size:
        mismatches.append(f"chunk_size: index={stored.chunk_size}, config={cfg.chunk_size}")
    if stored.chunk_overlap != cfg.chunk_overlap:
        mismatches.append(
            f"chunk_overlap: index={stored.chunk_overlap}, config={cfg.chunk_overlap}"
        )
    if stored.collection_name != cfg.collection_name:
        mismatches.append(
            f"collection_name: index={stored.collection_name!r}, config={cfg.collection_name!r}"
        )
    if cfg.book_file is not None:
        expected_source_files = sorted([cfg.book_file.name])
        if sorted(stored.source_files) != expected_source_files:
            mismatches.append(
                f"source_files: index={stored.source_files!r}, config={expected_source_files!r}"
            )

    if mismatches:
        raise ValueError(
            "Metadata experiment index does not match current settings:\n- "
            + "\n- ".join(mismatches)
            + "\nRebuild with metadata_experiment/scripts/build_metadata_index.py."
        )
    return stored
