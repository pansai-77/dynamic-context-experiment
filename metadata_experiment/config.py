from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _book_file() -> Path | None:
    value = os.getenv("BOOK_FILE")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class MetadataSettings:
    root_dir: Path = ROOT_DIR
    experiment_dir: Path = EXPERIMENT_DIR
    book_dir: Path = ROOT_DIR / "data" / "book"
    book_file: Path | None = field(default_factory=_book_file)
    questions_file: Path = ROOT_DIR / "data" / "questions" / "questions.csv"
    qdrant_path: Path = ROOT_DIR / "qdrant_storage_metadata"
    exp1_qdrant_path: Path = ROOT_DIR / "qdrant_storage"
    results_dir: Path = EXPERIMENT_DIR / "results"
    collection_name: str = "huozhe_meta"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "continuous_sentence_aware")
    chunk_target_size: int = int(os.getenv("CHUNK_TARGET_SIZE", "600"))
    chunk_max_size: int = int(os.getenv("CHUNK_MAX_SIZE", "800"))
    chunk_min_size: int = int(os.getenv("CHUNK_MIN_SIZE", "100"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    router_top_n: int = int(os.getenv("METADATA_ROUTER_TOP_N", "2"))
    retrieval_top_k: int = 4
    classification_max_new_tokens: int = int(
        os.getenv("METADATA_CLASSIFICATION_MAX_NEW_TOKENS", "128")
    )
    classification_max_retries: int = int(os.getenv("METADATA_CLASSIFICATION_MAX_RETRIES", "3"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "200"))
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "16"))
    quality_sample_size: int = int(os.getenv("METADATA_QUALITY_SAMPLE_SIZE", "20"))
    quality_sample_file: Path = EXPERIMENT_DIR / "data" / "quality_sample_chunks.json"
    original_metadata_file: Path = EXPERIMENT_DIR / "data" / "original_metadata.csv"
    manual_topic_overrides_file: Path = EXPERIMENT_DIR / "data" / "manual_topic_overrides.csv"
    manual_topic_overrides_template_file: Path = (
        EXPERIMENT_DIR / "data" / "manual_topic_overrides.template.csv"
    )
    classification_failures_file: Path = EXPERIMENT_DIR / "data" / "classification_failures.csv"


settings = MetadataSettings()
