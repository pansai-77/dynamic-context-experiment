from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float

@dataclass(frozen=True)
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_time_ms: float
    tokens_per_second: float

@dataclass(frozen=True)
class ExperimentMethod:
    name: str
    top_k: int
    query_aware: bool = False

@dataclass
class ExperimentRow:
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
    retrieved_chunks: int
    retrieved_sources: str
    score_0_3: Any = None
    notes: str = ""
