from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")


def _optional_path_from_env(env_key: str, default: Path) -> Path:
    value = os.getenv(env_key)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class MetadataSettings:
    root_dir: Path = ROOT_DIR
    experiment_dir: Path = EXPERIMENT_DIR
    book_dir: Path = ROOT_DIR / "data" / "book"
    book_file: Path = field(
        default_factory=lambda: _optional_path_from_env(
            "BOOK_FILE",
            ROOT_DIR / "data" / "book" / "活着.pdf",
        )
    )
    questions_file: Path = ROOT_DIR / "data" / "questions" / "questions.csv"
    qdrant_path: Path = ROOT_DIR / "qdrant_storage"
    results_dir: Path = field(
        default_factory=lambda: _optional_path_from_env(
            "METADATA_RESULTS_DIR",
            EXPERIMENT_DIR / "results",
        )
    )
    collection_name: str = os.getenv("METADATA_COLLECTION", "huozhe_meta")
    index_metadata_path: Path = field(
        default_factory=lambda: _optional_path_from_env(
            "METADATA_INDEX_METADATA_PATH",
            EXPERIMENT_DIR / "index_metadata.json",
        )
    )
    allowed_topics_file: Path = EXPERIMENT_DIR / "data" / "allowed_topics.json"
    topic_embeddings_file: Path = EXPERIMENT_DIR / "data" / "topic_embeddings.json"
    index_build_report_file: Path = EXPERIMENT_DIR / "index_build_report.json"
    topic_coverage_report_file: Path = EXPERIMENT_DIR / "topic_coverage_report.json"
    metadata_acceptance_samples_file: Path = EXPERIMENT_DIR / "data" / "metadata_acceptance_samples.json"
    metadata_holdout_samples_file: Path = EXPERIMENT_DIR / "data" / "metadata_holdout_samples.json"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "continuous_sentence_aware")
    chunk_target_size: int = int(os.getenv("CHUNK_TARGET_SIZE", "600"))
    chunk_max_size: int = int(os.getenv("CHUNK_MAX_SIZE", "800"))
    chunk_min_size: int = int(os.getenv("CHUNK_MIN_SIZE", "100"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = 4
    topic_routing_top_n: int = int(os.getenv("TOPIC_ROUTING_TOP_N", "2"))
    metadata_gen_max_retries: int = int(os.getenv("METADATA_GEN_MAX_RETRIES", "2"))
    metadata_max_new_tokens: int = int(os.getenv("METADATA_MAX_NEW_TOKENS", "384"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "200"))
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "16"))


settings = MetadataSettings()
