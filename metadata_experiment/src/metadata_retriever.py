from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.models import Chunk, RetrievedChunk

from models import ChunkMetadata, RetrievalTiming, TopicDefinition
from prompts import load_allowed_topics
from topic_router import TopicRouter


class MetadataVectorStore:
    def __init__(self, storage_path: Path, collection_name: str, embedding_model_name: str) -> None:
        storage_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(storage_path))
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def rebuild(
        self,
        chunks: list[Chunk],
        metadata_by_chunk_id: dict[str, ChunkMetadata],
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
                metadata = metadata_by_chunk_id[chunk.chunk_id]
                points.append(
                    PointStruct(
                        id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                        vector=vector.tolist(),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "source_file": chunk.source_file,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "characters": metadata.characters,
                            "topics": metadata.topics,
                            "keywords": metadata.keywords,
                            "importance": metadata.importance,
                            "metadata_status": metadata.metadata_status,
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def warm_up(self) -> None:
        self.embed_query("预热检索")
        self.search_by_vector(self.embed_query("预热检索"), top_k=1)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _points_to_retrieved(self, result_points) -> list[RetrievedChunk]:
        retrieved = []
        for point in result_points:
            payload = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=str(payload.get("chunk_id", point.id)),
                        text=str(payload.get("text", "")),
                        source_file=str(payload.get("source_file", "")),
                        page_number=int(payload.get("page_number", 0)),
                        chunk_index=int(payload.get("chunk_index", 0)),
                    ),
                    score=float(point.score),
                )
            )
        return retrieved

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        query_filter: Filter | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        if top_k <= 0:
            return [], 0.0
        started = time.perf_counter()
        result_points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points
        elapsed_ms = (time.perf_counter() - started) * 1000
        return self._points_to_retrieved(result_points), elapsed_ms

    def search_full(self, query_vector: np.ndarray, top_k: int) -> tuple[list[RetrievedChunk], RetrievalTiming]:
        retrieved, search_ms = self.search_by_vector(query_vector, top_k=top_k)
        timing = RetrievalTiming(
            router_time_ms=0.0,
            filter_build_time_ms=0.0,
            vector_search_time_ms=search_ms,
            retrieval_total_ms=search_ms,
        )
        return retrieved, timing

    def search_with_metadata(
        self,
        query_vector: np.ndarray,
        router: TopicRouter,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], RetrievalTiming, list[str]]:
        predictions, router_ms = router.route(query_vector)
        topic_ids = [prediction.topic_id for prediction in predictions]
        query_filter, filter_ms = router.build_or_filter(predictions)
        retrieved, search_ms = self.search_by_vector(query_vector, top_k=top_k, query_filter=query_filter)
        total_ms = router_ms + filter_ms + search_ms
        timing = RetrievalTiming(
            router_time_ms=router_ms,
            filter_build_time_ms=filter_ms,
            vector_search_time_ms=search_ms,
            retrieval_total_ms=total_ms,
        )
        return retrieved, timing, topic_ids


def build_topic_embeddings(
    store: MetadataVectorStore,
    topics: list[TopicDefinition],
    output_path: Path,
    embedding_model_name: str,
) -> dict[str, np.ndarray]:
    texts = [topic.embedding_text for topic in topics]
    vectors = store.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {topic.id: vectors[index] for index, topic in enumerate(topics)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model_name,
                "text_format": "{label}：{description}",
                "embeddings": {topic_id: vector.tolist() for topic_id, vector in embeddings.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return embeddings


def build_router_from_settings(
    store: MetadataVectorStore,
    allowed_topics_file: Path,
    topic_embeddings_file: Path,
    top_n: int,
    embedding_model_name: str,
) -> TopicRouter:
    topics = load_allowed_topics(allowed_topics_file)
    if not topic_embeddings_file.exists():
        build_topic_embeddings(store, topics, topic_embeddings_file, embedding_model_name)
    return TopicRouter.from_files(allowed_topics_file, topic_embeddings_file, top_n=top_n)
