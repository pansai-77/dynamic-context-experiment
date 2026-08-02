from __future__ import annotations

import numpy as np

from models import TopicDefinition
from topic_router import TopicRouter


def test_topic_router_returns_top_two_predictions():
    topics = [
        TopicDefinition("death_loss", "死亡与失去", "亲人离世"),
        TopicDefinition("disease_medical", "疾病与医疗", "献血与医疗"),
    ]
    embeddings = {
        "death_loss": np.array([1.0, 0.0], dtype=np.float32),
        "disease_medical": np.array([0.0, 1.0], dtype=np.float32),
    }
    router = TopicRouter(topics=topics, topic_embeddings=embeddings, top_n=2)
    predictions, elapsed = router.route(np.array([0.2, 0.9], dtype=np.float32))
    assert elapsed >= 0.0
    assert len(predictions) == 2
    assert predictions[0].topic_id == "disease_medical"
    assert predictions[1].topic_id == "death_loss"


def test_topic_router_builds_or_filter():
    topics = [TopicDefinition("death_loss", "死亡与失去", "亲人离世")]
    embeddings = {"death_loss": np.array([1.0], dtype=np.float32)}
    router = TopicRouter(topics=topics, topic_embeddings=embeddings, top_n=1)
    query_filter, elapsed = router.build_or_filter(
        router.route(np.array([1.0], dtype=np.float32))[0]
    )
    assert elapsed >= 0.0
    assert query_filter is not None


def test_topic_router_builds_no_filter_when_no_predictions():
    topics = [TopicDefinition("death_loss", "死亡与失去", "亲人离世")]
    embeddings = {"death_loss": np.array([1.0], dtype=np.float32)}
    router = TopicRouter(topics=topics, topic_embeddings=embeddings, top_n=1)
    query_filter, elapsed = router.build_or_filter([])
    assert elapsed >= 0.0
    assert query_filter is None
