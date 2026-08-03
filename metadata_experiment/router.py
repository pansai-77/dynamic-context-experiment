from __future__ import annotations

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from .topics import routable_topic_names, topic_documents


class TopicRouter:
    def __init__(self, embedding_model: SentenceTransformer) -> None:
        self.embedding_model = embedding_model
        self.names = routable_topic_names()
        self.topic_vectors = self.embedding_model.encode(
            topic_documents(), normalize_embeddings=True, show_progress_bar=False
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
