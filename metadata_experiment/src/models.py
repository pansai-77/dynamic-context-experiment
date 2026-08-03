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

    @property
    def search_only_ms(self) -> float:
        return self.router_time_ms + self.filter_build_time_ms + self.vector_search_time_ms


@dataclass
class MetadataExperimentRow:
    question_id: str
    question_type: str
    question: str
    method: str
    retrieved_chunks: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    embed_query_time_ms: float
    router_time_ms: float
    filter_time_ms: float
    vector_search_time_ms: float
    search_only_time_ms: float
    online_retrieval_time_ms: float
    generation_time_ms: float
    end_to_end_time_ms: float
    answer: str
    score_0_3: Any = None
