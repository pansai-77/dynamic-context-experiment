from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

def _optional_path_from_env(env_key: str) -> Path | None:
    value = os.getenv(env_key)
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path

@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    book_dir: Path = ROOT_DIR / "data" / "book"
    book_file: Path | None = field(default_factory=lambda: _optional_path_from_env("BOOK_FILE"))
    questions_file: Path = ROOT_DIR / "data" / "questions" / "questions.csv"
    qdrant_path: Path = ROOT_DIR / "qdrant_storage"
    results_dir: Path = ROOT_DIR / "results"
    collection_name: str = "huozhe"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "200"))
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

settings = Settings()
