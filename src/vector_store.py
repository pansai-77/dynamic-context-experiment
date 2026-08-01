from __future__ import annotations
from pathlib import Path
import time
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from .models import Chunk, RetrievedChunk

class LocalVectorStore:
    def __init__(self, storage_path: Path, collection_name: str, embedding_model_name: str) -> None:
        storage_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(storage_path))
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def rebuild(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        vector_size = self.embedding_model.get_sentence_embedding_dimension()
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embedding_model.encode(
                [chunk.text for chunk in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            points = [
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    },
                )
                for chunk, vector in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def warm_up(self) -> None:
        self.search("预热检索", top_k=1)

    def search(self, query: str, top_k: int) -> tuple[list[RetrievedChunk], float]:
        if top_k <= 0:
            return [], 0.0
        started = time.perf_counter()
        query_vector = self.embedding_model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        result_points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieved = []
        for point in result_points:
            payload = point.payload or {}
            retrieved.append(RetrievedChunk(
                chunk=Chunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    text=str(payload.get("text", "")),
                    source_file=str(payload.get("source_file", "")),
                    page_number=int(payload.get("page_number", 0)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                ),
                score=float(point.score),
            ))
        return retrieved, elapsed_ms
