from __future__ import annotations

import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.models import Chunk, RetrievedChunk

from .topics import router_topic_documents, routable_topic_names


class TopicRouter:
    def __init__(self, embedding_model: SentenceTransformer) -> None:
        self.embedding_model = embedding_model
        self.names = routable_topic_names()
        self.topic_vectors = self.embedding_model.encode(
            router_topic_documents(), normalize_embeddings=True, show_progress_bar=False
        )

    def route(self, query: str, top_n: int = 2) -> tuple[list[str], float]:
        started = time.perf_counter()
        query_vector = self.embedding_model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        scores = np.asarray(self.topic_vectors) @ np.asarray(query_vector)
        indices = np.argsort(-scores)[:top_n]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return [self.names[int(index)] for index in indices], elapsed_ms


class MetadataVectorStore:
    def __init__(self, storage_path: Path, collection_name: str, embedding_model_name: str) -> None:
        storage_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(storage_path))
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def close(self) -> None:
        self.client.close()

    def rebuild(
        self,
        chunks: list[Chunk],
        chunk_payloads: dict[str, dict],
        batch_size: int = 64,
    ) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        vector_size = self.embedding_model.get_sentence_embedding_dimension()
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedding_model.encode(
                [chunk.text for chunk in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            points = []
            for chunk, vector in zip(batch, vectors):
                if chunk.chunk_id not in chunk_payloads:
                    raise KeyError(f"Missing payload for chunk {chunk.chunk_id}")
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_index": chunk.chunk_index,
                    **chunk_payloads[chunk.chunk_id],
                }
                points.append(PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload=payload,
                ))
            self.client.upsert(self.collection_name, points=points, wait=True)

    def total_candidates(self) -> int:
        return int(self.client.count(self.collection_name, exact=True).count)

    @staticmethod
    def topic_filter(topics: list[str]) -> Filter:
        # OR semantics: keep chunks whose payload topics intersect routed Top-2.
        return Filter(must=[FieldCondition(key="topics", match=MatchAny(any=topics))])

    def candidate_count(self, topics: list[str]) -> int:
        return int(self.client.count(
            self.collection_name,
            count_filter=self.topic_filter(topics),
            exact=True,
        ).count)

    def search(
        self, query: str, top_k: int = 4, topics: list[str] | None = None
    ) -> tuple[list[RetrievedChunk], float]:
        started = time.perf_counter()
        query_vector = self.embedding_model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self.topic_filter(topics) if topics else None,
            limit=top_k,
            with_payload=True,
        ).points
        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieved: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            page_start = int(payload.get("page_start", payload.get("page_number", 0)))
            page_end = int(payload.get("page_end", page_start))
            retrieved.append(RetrievedChunk(
                chunk=Chunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    text=str(payload.get("text", "")),
                    source_file=str(payload.get("source_file", "")),
                    page_number=page_start,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_index=int(payload.get("chunk_index", 0)),
                ),
                score=float(point.score),
            ))
        return retrieved, elapsed_ms
