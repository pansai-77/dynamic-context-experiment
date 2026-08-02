from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchAny

from models import TopicDefinition, TopicPrediction
from prompts import load_allowed_topics


class TopicRouter:
    def __init__(
        self,
        topics: list[TopicDefinition],
        topic_embeddings: dict[str, np.ndarray],
        top_n: int = 2,
    ) -> None:
        self.topics = topics
        self.topic_embeddings = topic_embeddings
        self.top_n = top_n

    @classmethod
    def from_files(
        cls,
        allowed_topics_file: Path,
        topic_embeddings_file: Path,
        top_n: int = 2,
    ) -> TopicRouter:
        topics = load_allowed_topics(allowed_topics_file)
        payload = json.loads(topic_embeddings_file.read_text(encoding="utf-8"))
        embeddings = {
            topic_id: np.array(vector, dtype=np.float32)
            for topic_id, vector in payload["embeddings"].items()
        }
        missing = {topic.id for topic in topics} - set(embeddings)
        if missing:
            raise ValueError(f"Missing topic embeddings for: {sorted(missing)}")
        return cls(topics=topics, topic_embeddings=embeddings, top_n=top_n)

    def route(self, query_vector: np.ndarray) -> tuple[list[TopicPrediction], float]:
        import time

        started = time.perf_counter()
        predictions: list[TopicPrediction] = []
        query = query_vector.astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return [], elapsed_ms

        for topic in self.topics:
            topic_vector = self.topic_embeddings[topic.id]
            topic_norm = np.linalg.norm(topic_vector)
            if topic_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(query, topic_vector) / (query_norm * topic_norm))
            predictions.append(TopicPrediction(topic_id=topic.id, score=score))

        predictions.sort(key=lambda item: item.score, reverse=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return predictions[: self.top_n], elapsed_ms

    def build_or_filter(self, predictions: list[TopicPrediction]) -> tuple[Filter | None, float]:
        import time

        started = time.perf_counter()
        topic_ids = [prediction.topic_id for prediction in predictions if prediction.topic_id]
        if not topic_ids:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return None, elapsed_ms

        query_filter = Filter(
            should=[
                FieldCondition(key="topics", match=MatchAny(any=topic_ids)),
            ]
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return query_filter, elapsed_ms
