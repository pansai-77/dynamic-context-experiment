from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from config import MetadataSettings, settings


@dataclass(frozen=True)
class IndexMetadata:
    embedding_model: str
    chunk_strategy: str
    target_size: int
    max_size: int
    min_size: int
    overlap: int
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
    if "chunk_strategy" in payload:
        return IndexMetadata(
            embedding_model=str(payload["embedding_model"]),
            chunk_strategy=str(payload["chunk_strategy"]),
            target_size=int(payload["target_size"]),
            max_size=int(payload["max_size"]),
            min_size=int(payload["min_size"]),
            overlap=int(payload["overlap"]),
            source_files=[str(name) for name in payload["source_files"]],
            collection_name=str(payload.get("collection_name", "huozhe_meta")),
            chunk_embedding_policy=str(payload.get("chunk_embedding_policy", "text_only")),
        )
    return IndexMetadata(
        embedding_model=str(payload["embedding_model"]),
        chunk_strategy="page_local_fixed",
        target_size=int(payload["chunk_size"]),
        max_size=int(payload["chunk_size"]),
        min_size=0,
        overlap=int(payload["chunk_overlap"]),
        source_files=[str(name) for name in payload["source_files"]],
        collection_name=str(payload.get("collection_name", "huozhe_meta")),
        chunk_embedding_policy=str(payload.get("chunk_embedding_policy", "text_only")),
    )


def expected_metadata(cfg: MetadataSettings, source_files: list[str]) -> IndexMetadata:
    return IndexMetadata(
        embedding_model=cfg.embedding_model,
        chunk_strategy=cfg.chunk_strategy,
        target_size=cfg.chunk_target_size,
        max_size=cfg.chunk_max_size,
        min_size=cfg.chunk_min_size,
        overlap=cfg.chunk_overlap,
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
    if stored.chunk_strategy != cfg.chunk_strategy:
        mismatches.append(
            f"chunk_strategy: index={stored.chunk_strategy!r}, config={cfg.chunk_strategy!r}"
        )
    if stored.target_size != cfg.chunk_target_size:
        mismatches.append(f"target_size: index={stored.target_size}, config={cfg.chunk_target_size}")
    if stored.max_size != cfg.chunk_max_size:
        mismatches.append(f"max_size: index={stored.max_size}, config={cfg.chunk_max_size}")
    if stored.min_size != cfg.chunk_min_size:
        mismatches.append(f"min_size: index={stored.min_size}, config={cfg.chunk_min_size}")
    if stored.overlap != cfg.chunk_overlap:
        mismatches.append(f"overlap: index={stored.overlap}, config={cfg.chunk_overlap}")
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
