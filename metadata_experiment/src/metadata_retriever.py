from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.models import Chunk, RetrievedChunk

from models import ChunkMetadata, RetrievalTiming, TopicDefinition, TopicPrediction
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
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="topics",
                field_schema="keyword",
            )
        except Exception:
            pass
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
                            "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
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
            page_start = int(payload.get("page_start", payload.get("page_number", 0)))
            page_end = int(payload.get("page_end", page_start))
            retrieved.append(
                RetrievedChunk(
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
                )
            )
        return retrieved

    def filtered_chunk_ids(self, topic_ids: list[str]) -> set[str]:
        if not topic_ids:
            return set()
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = Filter(
            should=[FieldCondition(key="topics", match=MatchAny(any=topic_ids))]
        )
        matched: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=200,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                matched.add(str(payload.get("chunk_id", point.id)))
            if offset is None:
                break
        return matched

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
        allow_topic_expansion: bool = False,
    ) -> tuple[list[RetrievedChunk], RetrievalTiming, list[str], list[TopicPrediction]]:
        if allow_topic_expansion:
            return self._search_with_metadata_expanded(query_vector, router, top_k)

        predictions, router_ms = router.route(query_vector)
        topic_ids = [prediction.topic_id for prediction in predictions]
        query_filter, filter_ms = router.build_or_filter(predictions)
        if query_filter is None:
            timing = RetrievalTiming(
                router_time_ms=router_ms,
                filter_build_time_ms=filter_ms,
                vector_search_time_ms=0.0,
                retrieval_total_ms=router_ms + filter_ms,
            )
            return [], timing, topic_ids, predictions

        retrieved, search_ms = self.search_by_vector(
            query_vector,
            top_k=top_k,
            query_filter=query_filter,
        )
        total_ms = router_ms + filter_ms + search_ms
        timing = RetrievalTiming(
            router_time_ms=router_ms,
            filter_build_time_ms=filter_ms,
            vector_search_time_ms=search_ms,
            retrieval_total_ms=total_ms,
        )
        return retrieved, timing, topic_ids, predictions

    def _search_with_metadata_expanded(
        self,
        query_vector: np.ndarray,
        router: TopicRouter,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], RetrievalTiming, list[str], list[TopicPrediction]]:
        all_predictions, router_ms = router.rank_all(query_vector)
        if not all_predictions:
            return [], RetrievalTiming(router_time_ms=router_ms), [], []

        max_topics = min(len(all_predictions), max(router.top_n + 2, router.top_n))
        attempt_limits = list(range(router.top_n, max_topics + 1))

        retrieved: list[RetrievedChunk] = []
        search_ms = 0.0
        filter_ms = 0.0
        topic_ids: list[str] = []
        predictions_used = all_predictions[: router.top_n]

        for limit in attempt_limits:
            predictions = all_predictions[:limit]
            topic_ids = [prediction.topic_id for prediction in predictions]
            query_filter, filter_ms = router.build_or_filter(predictions)
            if query_filter is None:
                continue
            retrieved, search_ms = self.search_by_vector(
                query_vector,
                top_k=top_k,
                query_filter=query_filter,
            )
            predictions_used = predictions
            if retrieved:
                break

        total_ms = router_ms + filter_ms + search_ms
        timing = RetrievalTiming(
            router_time_ms=router_ms,
            filter_build_time_ms=filter_ms,
            vector_search_time_ms=search_ms,
            retrieval_total_ms=total_ms,
        )
        return retrieved, timing, topic_ids, predictions_used


def topic_catalog_fingerprint(
    topics: list[TopicDefinition],
    allowed_topics_file: Path,
) -> dict[str, str]:
    payload = json.loads(allowed_topics_file.read_text(encoding="utf-8"))
    catalog = [{"id": topic.id, "embedding_text": topic.embedding_text} for topic in topics]
    catalog_json = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
    return {
        "topics_version": str(payload.get("version", "")),
        "topic_catalog_hash": hashlib.sha256(catalog_json.encode("utf-8")).hexdigest(),
    }


def build_topic_embeddings(
    store: MetadataVectorStore,
    topics: list[TopicDefinition],
    output_path: Path,
    embedding_model_name: str,
    allowed_topics_file: Path,
) -> dict[str, np.ndarray]:
    texts = [topic.embedding_text for topic in topics]
    vectors = store.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {topic.id: vectors[index] for index, topic in enumerate(topics)}
    fingerprint = topic_catalog_fingerprint(topics, allowed_topics_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model_name,
                "text_format": "{label}：{description}",
                **fingerprint,
                "embeddings": {topic_id: vector.tolist() for topic_id, vector in embeddings.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return embeddings


def topic_embeddings_cache_is_stale(
    topic_embeddings_file: Path,
    allowed_topics_file: Path,
    embedding_model_name: str,
    topics: list[TopicDefinition],
) -> bool:
    if not topic_embeddings_file.exists():
        return True
    payload = json.loads(topic_embeddings_file.read_text(encoding="utf-8"))
    if payload.get("embedding_model") != embedding_model_name:
        return True
    fingerprint = topic_catalog_fingerprint(topics, allowed_topics_file)
    for key, value in fingerprint.items():
        if payload.get(key) != value:
            return True
    return False


def build_router_from_settings(
    store: MetadataVectorStore,
    allowed_topics_file: Path,
    topic_embeddings_file: Path,
    top_n: int,
    embedding_model_name: str,
) -> TopicRouter:
    topics = load_allowed_topics(allowed_topics_file)
    if topic_embeddings_cache_is_stale(
        topic_embeddings_file,
        allowed_topics_file,
        embedding_model_name,
        topics,
    ):
        build_topic_embeddings(
            store,
            topics,
            topic_embeddings_file,
            embedding_model_name,
            allowed_topics_file,
        )
    return TopicRouter.from_files(allowed_topics_file, topic_embeddings_file, top_n=top_n)
