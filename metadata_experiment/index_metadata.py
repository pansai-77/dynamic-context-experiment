from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import MetadataSettings, settings
from .classification_prompts import CLASSIFICATION_PROMPT_VERSION
from .topics import TOPIC_TAXONOMY_VERSION, topic_names


MANIFEST_FILENAME = "metadata_index_manifest.json"
EXP1_COLLECTION = "huozhe"


@dataclass(frozen=True)
class MetadataIndexManifest:
    collection: str
    embedding_model: str
    chunk_strategy: str
    chunk_target_size: int
    chunk_max_size: int
    chunk_min_size: int
    chunk_overlap: int
    source_files: list[str]
    chunk_ids: list[str]
    topics: list[str]
    topic_taxonomy_version: str
    classification_prompt_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def manifest_path(qdrant_path: Path) -> Path:
    return qdrant_path / MANIFEST_FILENAME


def write_index_manifest(path: Path, manifest: MetadataIndexManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_index_manifest(path: Path) -> MetadataIndexManifest | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MetadataIndexManifest(
        collection=str(payload["collection"]),
        embedding_model=str(payload["embedding_model"]),
        chunk_strategy=str(payload.get("chunk_strategy", "continuous_sentence_aware")),
        chunk_target_size=int(payload["chunk_target_size"]),
        chunk_max_size=int(payload["chunk_max_size"]),
        chunk_min_size=int(payload["chunk_min_size"]),
        chunk_overlap=int(payload["chunk_overlap"]),
        source_files=[str(name) for name in payload.get("source_files", [])],
        chunk_ids=[str(chunk_id) for chunk_id in payload.get("chunk_ids", [])],
        topics=[str(topic) for topic in payload.get("topics", [])],
        topic_taxonomy_version=str(payload.get("topic_taxonomy_version", "")),
        classification_prompt_version=str(payload.get("classification_prompt_version", "")),
    )


def list_collection_chunk_ids(qdrant_path: Path, collection_name: str) -> list[str]:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(qdrant_path))
    try:
        if not client.collection_exists(collection_name):
            raise FileNotFoundError(
                f"Qdrant collection {collection_name!r} not found under {qdrant_path}."
            )

        chunk_ids: list[str] = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                chunk_ids.append(str(payload.get("chunk_id", record.id)))
            if offset is None:
                break
        return sorted(chunk_ids)
    finally:
        client.close()


def expected_manifest(
    cfg: MetadataSettings,
    chunks: list,
    topics: list[str],
) -> MetadataIndexManifest:
    source_files = [cfg.book_file.name] if cfg.book_file is not None else []
    return MetadataIndexManifest(
        collection=cfg.collection_name,
        embedding_model=cfg.embedding_model,
        chunk_strategy=cfg.chunk_strategy,
        chunk_target_size=cfg.chunk_target_size,
        chunk_max_size=cfg.chunk_max_size,
        chunk_min_size=cfg.chunk_min_size,
        chunk_overlap=cfg.chunk_overlap,
        source_files=sorted(source_files),
        chunk_ids=sorted(chunk.chunk_id for chunk in chunks),
        topics=topics,
        topic_taxonomy_version=TOPIC_TAXONOMY_VERSION,
        classification_prompt_version=CLASSIFICATION_PROMPT_VERSION,
    )


def verify_index_metadata(cfg: MetadataSettings = settings) -> MetadataIndexManifest:
    stored = read_index_manifest(manifest_path(cfg.qdrant_path))
    if stored is None:
        raise FileNotFoundError(
            f"Metadata index manifest not found at {manifest_path(cfg.qdrant_path)}. "
            "Run metadata_experiment/scripts/build_index.py first."
        )

    mismatches: list[str] = []
    if stored.topic_taxonomy_version != TOPIC_TAXONOMY_VERSION:
        mismatches.append(
            "topic_taxonomy_version: "
            f"index={stored.topic_taxonomy_version!r}, "
            f"config={TOPIC_TAXONOMY_VERSION!r}"
        )
    if stored.classification_prompt_version != CLASSIFICATION_PROMPT_VERSION:
        mismatches.append(
            "classification_prompt_version: "
            f"index={stored.classification_prompt_version!r}, "
            f"config={CLASSIFICATION_PROMPT_VERSION!r}"
        )
    expected_topics = sorted(topic_names())
    if sorted(stored.topics) != expected_topics:
        mismatches.append(
            "topics: index vocabulary differs from topics.py "
            f"(index={len(stored.topics)}, config={len(expected_topics)})"
        )
    if stored.embedding_model != cfg.embedding_model:
        mismatches.append(
            f"embedding_model: index={stored.embedding_model!r}, config={cfg.embedding_model!r}"
        )
    if stored.chunk_strategy != cfg.chunk_strategy:
        mismatches.append(
            f"chunk_strategy: index={stored.chunk_strategy!r}, config={cfg.chunk_strategy!r}"
        )
    if stored.chunk_target_size != cfg.chunk_target_size:
        mismatches.append(
            f"chunk_target_size: index={stored.chunk_target_size}, config={cfg.chunk_target_size}"
        )
    if stored.chunk_max_size != cfg.chunk_max_size:
        mismatches.append(
            f"chunk_max_size: index={stored.chunk_max_size}, config={cfg.chunk_max_size}"
        )
    if stored.chunk_min_size != cfg.chunk_min_size:
        mismatches.append(
            f"chunk_min_size: index={stored.chunk_min_size}, config={cfg.chunk_min_size}"
        )
    if stored.chunk_overlap != cfg.chunk_overlap:
        mismatches.append(
            f"chunk_overlap: index={stored.chunk_overlap}, config={cfg.chunk_overlap}"
        )
    if cfg.book_file is not None:
        expected_sources = sorted([cfg.book_file.name])
        if sorted(stored.source_files) != expected_sources:
            mismatches.append(
                f"source_files: index={stored.source_files!r}, config={expected_sources!r}"
            )

    live_chunk_ids = list_collection_chunk_ids(cfg.qdrant_path, cfg.collection_name)
    if stored.chunk_ids and stored.chunk_ids != live_chunk_ids:
        mismatches.append(
            f"chunk_ids: manifest lists {len(stored.chunk_ids)} ids, "
            f"collection has {len(live_chunk_ids)} ids"
        )

    if mismatches:
        raise ValueError(
            "Metadata index does not match current settings:\n- "
            + "\n- ".join(mismatches)
            + "\nRun metadata_experiment/scripts/build_index.py to rebuild the index."
        )

    return stored


def verify_chunk_parity_with_exp1(
    cfg: MetadataSettings = settings,
    meta_chunk_ids: list[str] | None = None,
) -> list[str]:
    exp1_metadata_path = cfg.exp1_qdrant_path / "index_metadata.json"
    if not exp1_metadata_path.exists():
        return [
            f"Experiment 1 index metadata not found at {exp1_metadata_path}; "
            "skipped chunk parity check."
        ]

    exp1_metadata = json.loads(exp1_metadata_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if exp1_metadata.get("embedding_model") != cfg.embedding_model:
        mismatches.append("embedding_model differs from experiment 1 index")
    if exp1_metadata.get("chunk_strategy") != cfg.chunk_strategy:
        mismatches.append("chunk_strategy differs from experiment 1 index")
    for field, cfg_value in (
        ("target_size", cfg.chunk_target_size),
        ("max_size", cfg.chunk_max_size),
        ("min_size", cfg.chunk_min_size),
        ("overlap", cfg.chunk_overlap),
    ):
        if exp1_metadata.get(field) != cfg_value:
            mismatches.append(f"{field} differs from experiment 1 index")

    if meta_chunk_ids is None:
        meta_chunk_ids = list_collection_chunk_ids(cfg.qdrant_path, cfg.collection_name)
    else:
        meta_chunk_ids = sorted(meta_chunk_ids)
    exp1_chunk_ids = list_collection_chunk_ids(cfg.exp1_qdrant_path, EXP1_COLLECTION)
    if meta_chunk_ids != exp1_chunk_ids:
        only_meta = sorted(set(meta_chunk_ids) - set(exp1_chunk_ids))
        only_exp1 = sorted(set(exp1_chunk_ids) - set(meta_chunk_ids))
        mismatches.append(
            "chunk_id sets differ from experiment 1 index "
            f"(meta-only={len(only_meta)}, exp1-only={len(only_exp1)})"
        )
    return mismatches
