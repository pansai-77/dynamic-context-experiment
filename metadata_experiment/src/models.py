from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models import Chunk, RetrievedChunk


@dataclass(frozen=True)
class TopicDefinition:
    id: str
    label: str
    description: str

    @property
    def embedding_text(self) -> str:
        return f"{self.label}：{self.description}"


@dataclass(frozen=True)
class ChunkMetadata:
    characters: list[str]
    topics: list[str]
    keywords: list[str]
    importance: int | None
    metadata_status: str


@dataclass(frozen=True)
class TopicPrediction:
    topic_id: str
    score: float


@dataclass(frozen=True)
class MetadataExperimentMethod:
    name: str
    use_metadata_filter: bool = False


@dataclass(frozen=True)
class RetrievalTiming:
    router_time_ms: float = 0.0
    filter_build_time_ms: float = 0.0
    vector_search_time_ms: float = 0.0
    retrieval_total_ms: float = 0.0


@dataclass
class MetadataExperimentRow:
    question_id: str
    question_type: str
    question: str
    method: str
    top_k: int
    used_retrieval: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    retrieval_time_ms: float
    llm_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    answer: str
    retrieved_chunk_ids: str
    retrieved_sources: str
    score_0_3: Any = None
