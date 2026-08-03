from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from config import MetadataSettings, settings
from metadata_parsing import MAX_TOPICS
from prompts import build_metadata_prompt, load_allowed_topics, ontology_payload


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
    metadata_llm: str = ""
    ontology_version: str = ""
    ontology_hash: str = ""
    metadata_prompt_hash: str = ""
    max_topics_per_chunk: int = MAX_TOPICS
    topic_router_embedding_format: str = "label_colon_description"
    build_timestamp_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ontology_hash_from_file(allowed_topics_file: Path) -> tuple[str, str]:
    payload = ontology_payload(allowed_topics_file)
    version = str(payload.get("version", ""))
    catalog = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return version, sha256_text(catalog)


def metadata_prompt_hash(topics_file: Path, sample_chunk: str = "验收片段") -> str:
    topics = load_allowed_topics(topics_file)
    prompt = build_metadata_prompt(sample_chunk, topics, retry=False)
    return sha256_text(prompt)


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
            metadata_llm=str(payload.get("metadata_llm", "")),
            ontology_version=str(payload.get("ontology_version", "")),
            ontology_hash=str(payload.get("ontology_hash", "")),
            metadata_prompt_hash=str(payload.get("metadata_prompt_hash", "")),
            max_topics_per_chunk=int(payload.get("max_topics_per_chunk", MAX_TOPICS)),
            topic_router_embedding_format=str(
                payload.get("topic_router_embedding_format", "label_colon_description")
            ),
            build_timestamp_utc=str(payload.get("build_timestamp_utc", "")),
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
    ontology_version, ontology_hash = ontology_hash_from_file(cfg.allowed_topics_file)
    prompt_hash = metadata_prompt_hash(cfg.allowed_topics_file)
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
        metadata_llm=cfg.llm_model,
        ontology_version=ontology_version,
        ontology_hash=ontology_hash,
        metadata_prompt_hash=prompt_hash,
        max_topics_per_chunk=MAX_TOPICS,
        topic_router_embedding_format="label_colon_description",
        build_timestamp_utc=datetime.now(UTC).isoformat(),
    )


def verify_index_metadata(cfg: MetadataSettings = settings) -> IndexMetadata:
    stored = read_index_metadata(cfg.index_metadata_path)
    if stored is None:
        raise FileNotFoundError(
            f"Index metadata not found at {cfg.index_metadata_path}. "
            "Run metadata_experiment/scripts/build_metadata_index.py first."
        )

    expected = expected_metadata(cfg, stored.source_files)
    mismatches: list[str] = []
    checks = [
        ("embedding_model", stored.embedding_model, cfg.embedding_model),
        ("chunk_strategy", stored.chunk_strategy, cfg.chunk_strategy),
        ("target_size", stored.target_size, cfg.chunk_target_size),
        ("max_size", stored.max_size, cfg.chunk_max_size),
        ("min_size", stored.min_size, cfg.chunk_min_size),
        ("overlap", stored.overlap, cfg.chunk_overlap),
        ("collection_name", stored.collection_name, cfg.collection_name),
        ("ontology_version", stored.ontology_version, expected.ontology_version),
        ("ontology_hash", stored.ontology_hash, expected.ontology_hash),
        ("metadata_prompt_hash", stored.metadata_prompt_hash, expected.metadata_prompt_hash),
        ("max_topics_per_chunk", stored.max_topics_per_chunk, MAX_TOPICS),
    ]
    for name, actual, wanted in checks:
        if actual != wanted:
            mismatches.append(f"{name}: index={actual!r}, expected={wanted!r}")

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
